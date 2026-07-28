"""Canonical trace loading and replay (DESIGN §10).

The canonical CSV schema is the union of the public cluster traces::

    job_id, user, tenant, class, submit_time, num_chips, chip_type,
    num_nodes, duration_s, walltime_limit_s, final_status

- ``submit_time``: **int microseconds** since the trace epoch (the
  fleetsim time convention: no ``_s`` suffix = microseconds).
- ``duration_s`` / ``walltime_limit_s``: float **seconds**
  (``walltime_limit_s`` may be empty -> no estimate).
- ``class``: ``pretrain | finetune | eval | infer_replica``
  (case-insensitive).
- ``chip_type``: may be empty -> unpinned gang (matches any type).
- ``final_status``: ``COMPLETED | FAILED | CANCELED | TIMEOUT |
  NODE_FAIL`` (Helios's enum, case-insensitive), replayed VERBATIM via
  ``terminal_status_override`` — the job runs for ``duration_s`` of work
  and then finishes with exactly that status (COMPLETED means no
  override).  ``num_nodes`` and ``user`` are carried by the schema but
  not used by replay in v1.

Lines that are blank or start with ``#`` are skipped, so trace files can
carry comments.  Extra columns are ignored; missing canonical columns
are an error.

Field mapping to :class:`~fleetsim.model.Job`: ``tenant`` -> ``tenant``;
``walltime_limit_s`` -> ``walltime_est_s`` (scheduler-visible; NOT
enforced as ``max_lifetime_s``, since the outcome is replayed verbatim);
``duration_s`` -> ``true_duration_s`` (modeled as WORK seconds — the
engine's checkpoint amortization applies on top); tier defaults to PROD
for pretrain/infer_replica and BATCH otherwise (the config-phase rule);
``min_runtime_s`` gets the DESIGN §5.1/§14 per-class preemption guard
(7200 s for pretrain, 0 otherwise).

CHIP QUANTIZATION (DESIGN §4.1): raw trace ``num_chips`` need not obey
the allocation grammar (sub-node OR whole-node multiple) — Philly's
widest-attempt GPU sums frequently do not.  When :class:`TraceSource` is
given the runtime fleet, any count ABOVE the node size is rounded UP to
the next whole-node multiple (sub-node counts replay verbatim; they are
legal at any size).  Without a fleet no quantization happens and a
non-conforming multi-node job is permanently unplaceable — always pass
the fleet for engine runs (``fleetsim.api`` does).

INVARIANTS: loading is a pure function of the file; jobs are returned
sorted by ``(submit_t, id)``; duplicate job ids raise ``ValueError``;
:class:`TraceSource` implements the same pull protocol as the synthetic
generator, so schedulers cannot tell replay from synthesis; quantization
never mutates caller-supplied jobs (changed jobs are replaced copies).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from ..model import GangSpec, Job, JobClass, JobStatus, Tier

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..fleet.tree import FleetTree

__all__ = [
    "CANONICAL_COLUMNS",
    "TraceJob",
    "load_trace",
    "write_trace",
    "TraceSource",
]

#: Canonical column order (DESIGN §10).
CANONICAL_COLUMNS: tuple[str, ...] = (
    "job_id",
    "user",
    "tenant",
    "class",
    "submit_time",
    "num_chips",
    "chip_type",
    "num_nodes",
    "duration_s",
    "walltime_limit_s",
    "final_status",
)

_JOB_CLASS = {
    "pretrain": JobClass.PRETRAIN,
    "finetune": JobClass.FINETUNE,
    "eval": JobClass.EVAL,
    "infer_replica": JobClass.INFER_REPLICA,
}

#: final_status -> terminal_status_override (None = natural COMPLETED).
_STATUS_OVERRIDE: dict[str, JobStatus | None] = {
    "COMPLETED": None,
    "FAILED": JobStatus.FAILED,
    "CANCELED": JobStatus.CANCELED,
    "TIMEOUT": JobStatus.TIMEOUT,
    "NODE_FAIL": JobStatus.NODE_FAIL,
}

#: Per-class preemption guard defaults (DESIGN §5.1 / §14, float seconds);
#: classes not listed default to 0.
_MIN_RUNTIME_S: dict[JobClass, float] = {JobClass.PRETRAIN: 7200.0}


@dataclass(slots=True)
class TraceJob(Job):
    """A replayed job; ``terminal_status_override`` (when set) is the
    trace's ``final_status``, applied by the engine in place of COMPLETED
    when the job finishes its replayed duration."""

    terminal_status_override: JobStatus | None = None


def _err(row_n: int, msg: str) -> ValueError:
    return ValueError(f"trace row {row_n}: {msg}")


def _parse_int(text: str, row_n: int, col: str) -> int:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return int(round(float(text)))
    except ValueError:
        raise _err(row_n, f"{col}: expected a number, got {text!r}") from None


def _parse_float(text: str, row_n: int, col: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise _err(row_n, f"{col}: expected a number, got {text!r}") from None


def load_trace(path: str | Path) -> list[Job]:
    """Load a canonical-schema CSV into :class:`TraceJob` objects, sorted
    by ``(submit_t, id)``.

    Raises ``ValueError`` on a missing header/columns or any malformed
    row (messages name the 1-based data-row number), and ``OSError`` if
    the file cannot be read.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"trace file {path} has no header row")
    reader = csv.DictReader(lines)
    missing = [c for c in CANONICAL_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(
            f"trace file {path} is missing column(s): {', '.join(missing)}"
        )

    jobs: list[Job] = []
    seen: set[str] = set()
    for row_n, row in enumerate(reader, start=1):
        jid = (row.get("job_id") or "").strip()
        if not jid:
            raise _err(row_n, "job_id is required")
        if jid in seen:
            raise _err(row_n, f"duplicate job_id {jid!r}")
        seen.add(jid)

        tenant = (row.get("tenant") or "").strip()
        if not tenant:
            raise _err(row_n, "tenant is required")

        cls_raw = (row.get("class") or "").strip().lower()
        job_class = _JOB_CLASS.get(cls_raw)
        if job_class is None:
            raise _err(
                row_n,
                f"unknown class {cls_raw!r} (known: {', '.join(_JOB_CLASS)})",
            )

        submit_t = _parse_int((row.get("submit_time") or "").strip(), row_n, "submit_time")
        if submit_t < 0:
            raise _err(row_n, f"submit_time must be >= 0, got {submit_t}")

        num_chips = _parse_int((row.get("num_chips") or "").strip(), row_n, "num_chips")
        if num_chips <= 0:
            raise _err(row_n, f"num_chips must be positive, got {num_chips}")

        duration_s = _parse_float(
            (row.get("duration_s") or "").strip(), row_n, "duration_s"
        )
        if duration_s < 0:
            raise _err(row_n, f"duration_s must be >= 0, got {duration_s}")

        wl_raw = (row.get("walltime_limit_s") or "").strip()
        walltime_est_s = (
            _parse_float(wl_raw, row_n, "walltime_limit_s") if wl_raw else None
        )

        status_raw = (row.get("final_status") or "").strip().upper()
        if status_raw not in _STATUS_OVERRIDE:
            raise _err(
                row_n,
                f"unknown final_status {status_raw!r}"
                f" (known: {', '.join(_STATUS_OVERRIDE)})",
            )

        chip_type = (row.get("chip_type") or "").strip() or None
        tier = (
            Tier.PROD
            if job_class in (JobClass.PRETRAIN, JobClass.INFER_REPLICA)
            else Tier.BATCH
        )
        jobs.append(
            TraceJob(
                id=jid,
                tenant=tenant,
                job_class=job_class,
                submit_t=submit_t,
                gangs=[GangSpec(chips=num_chips, chip_type=chip_type)],
                tier=tier,
                min_runtime_s=_MIN_RUNTIME_S.get(job_class, 0.0),
                walltime_est_s=walltime_est_s,
                true_duration_s=duration_s,
                terminal_status_override=_STATUS_OVERRIDE[status_raw],
            )
        )
    jobs.sort(key=lambda j: (j.submit_t, j.id))
    return jobs


def write_trace(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write canonical rows (e.g. from
    :func:`fleetsim.workload.philly.convert_philly`) as a canonical CSV.

    Rows are written in the given order with columns in
    :data:`CANONICAL_COLUMNS` order; missing/None values become empty
    fields; extra keys are ignored.
    """
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    "" if row.get(col) is None else row.get(col, "")
                    for col in CANONICAL_COLUMNS
                ]
            )


def _quantize_jobs(jobs: list[Job], fleet: "FleetTree") -> list[Job]:
    """DESIGN §4.1 quantization for trace replay: round each gang's chip
    count UP to a whole-node multiple when it exceeds the node size of
    its (pinned or any) chip type.  Jobs needing changes are replaced
    with shallow copies; unchanged jobs pass through untouched."""
    from .synthetic import node_sizes  # local: avoid import cycle surface

    sizes = node_sizes(fleet)
    default_node = max(sizes.values())
    out: list[Job] = []
    for job in jobs:
        new_gangs: list[GangSpec] | None = None
        for i, gang in enumerate(job.gangs):
            node = sizes.get(gang.chip_type or "", default_node)
            if gang.chips <= node:
                continue  # sub-node or exactly one node: verbatim
            quantized = node * -(-gang.chips // node)
            if quantized != gang.chips:
                if new_gangs is None:
                    new_gangs = list(job.gangs)
                new_gangs[i] = replace(gang, chips=quantized)
        out.append(job if new_gangs is None else replace(job, gangs=new_gangs))
    return out


class TraceSource:
    """Replay a canonical trace through the
    :class:`~fleetsim.workload.base.JobSource` pull protocol.

    Accepts a CSV path (loaded via :func:`load_trace`) or an iterable of
    already-built jobs (sorted here by ``(submit_t, id)``).  Each job is
    emitted exactly once at its ``submit_t``; ``next_arrival`` keeps
    returning ``None`` after exhaustion.

    ``fleet`` (recommended for engine runs; ``fleetsim.api`` passes it)
    enables DESIGN §4.1 chip quantization: gang chip counts above the
    node size (max leaf chips of the gang's pinned type; max over all
    types when unpinned) are rounded UP to the next whole-node multiple,
    so a 12-chip trace job places as 16 on 8-chip nodes instead of
    starving the queue forever.  Sub-node counts replay verbatim.
    Quantized jobs are shallow copies — caller-supplied jobs are never
    mutated.
    """

    __slots__ = ("_jobs", "_i")

    def __init__(
        self,
        source: str | Path | Iterable[Job],
        fleet: "FleetTree | None" = None,
    ):
        if isinstance(source, (str, Path)):
            self._jobs: list[Job] = load_trace(source)
        else:
            self._jobs = sorted(source, key=lambda j: (j.submit_t, j.id))
        if fleet is not None:
            self._jobs = _quantize_jobs(self._jobs, fleet)
        self._i = 0

    @property
    def jobs(self) -> tuple[Job, ...]:
        """All jobs in emission order (handy for post-run assertions)."""
        return tuple(self._jobs)

    def next_arrival(self) -> tuple[int, Job] | None:
        if self._i >= len(self._jobs):
            return None
        job = self._jobs[self._i]
        self._i += 1
        return (job.submit_t, job)
