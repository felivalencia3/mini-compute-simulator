"""Per-VC replay harness for the Helios validation (plan §4).

fleetsim schedules a **single global pool** and never routes jobs to a
cluster by tenant, but the Helios reference simulator schedules **each VC
independently** (one worker process per VC; jobs never cross VC
boundaries).  So the flagship V1/V2 replays are implemented HERE as a
harness that runs one :func:`fleetsim.api.run_scenario`-equivalent
simulation per VC and aggregates to cluster totals — not as an engine
change (the engine stays a global pool).

:func:`per_vc_replay` is the entry point: it fetches + extracts the real
Helios ``data.zip`` (:func:`fleetsim.validation.fetch.fetch_trace`),
converts one cluster's ``cluster_log.csv``
(:func:`fleetsim.validation.helios.convert_helios`), windows it to one
month (:func:`~fleetsim.validation.helios.month_window`), and for every
active VC on the Sept-1 snapshot runs an independent simulation on a
1-cluster fleet sized to that VC's node count (8 GPU/node), then applies
the metric adapters (:mod:`fleetsim.validation.adapters`) and aggregates.

REPLAY FIDELITY (plan §1, held fixed so service time == duration exactly):

- **failure_model OFF** — ``node_mtbf_days = 0`` and
  ``maintenance_rate_per_node_month = 0``, so no node ever fails or
  drains and no job is ever restarted.
- **checkpointing DISABLED for the replay jobs** —
  ``checkpoint_interval_s = checkpoint_save_s = restart_overhead_s = 0``,
  so a replayed job's wall time is exactly its trace ``duration_s`` (no
  checkpoint-save amortization on top).  The trace loader
  (:func:`fleetsim.workload.trace.load_trace`) uses the ``Job`` defaults
  (1 h checkpoint interval, 60 s save), so this harness builds the jobs
  DIRECTLY with those overheads zeroed rather than loading a CSV.
- **strict (blocking) scan** — both ``fifo`` and ``sjf`` run with
  ``strict=True`` (the head of the ordered queue blocks when it cannot be
  placed).  This is the LOAD-BEARING fidelity choice: the published Table 3
  FIFO numbers carry huge queuing (Saturn avg 50,202 s, 65,991 queued of
  105,698) that is head-of-line blocking — a large gang at the front of the
  FIFO queue stalls everything behind it.  A best-effort scan
  (``strict=False``) lets small jobs flow around the stall, collapses FIFO's
  queuing, and destroys the FIFO-vs-SJF ratio (measured aggregate JCT
  ratios ~1.4x, wrong cross-cluster rank).  The validation plan §2 V1(c)
  guessed the reference sim was non-blocking; the real trace says
  otherwise.  Both policies share the scan mode (one framework, two
  orderings), so SJF's advantage is purely its shortest-first ordering
  clearing short jobs before they pile up.
- **BATCH tier** — trace jobs map to class ``finetune`` -> tier BATCH, so
  they are INCLUDED in the papers' JCT/queue distributions (BEST_EFFORT
  would be excluded by ``metrics.summary``).

METRIC AGGREGATION (plan §4).  Per-VC ``jobs.parquet`` frames are
concatenated into one cluster frame; the cluster metrics are then the
adapters applied to that union — a job-weighted mean JCT over **all** VCs'
terminal jobs, a job-weighted mean queuing time, and a summed
``#Queuing``.  Concatenation makes the weighting automatic and exact.

DETERMINISM: a pure function of the trace bytes and the arguments — every
simulation runs at ``seed = 0`` with int-microsecond times; no wall clock
enters the sim path.  (The only I/O is the trace fetch/extract.)
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from ..config import load_scenario
from ..engine.sim import Simulator
from ..fleet.build import build_fleet
from ..metrics.collector import MetricsCollector
from ..metrics.summary import jobs_dataframe
from ..model import GangSpec, JobClass, JobStatus, Tier
from ..schedulers.base import get_scheduler
from ..workload.trace import TraceJob, TraceSource
from .adapters import (
    TERMINAL_STATUSES,
    jct_over_all_terminal,
    n_queuing_jobs,
)
from .fetch import fetch_trace, is_lfs_pointer
from .helios import (
    HELIOS_SEPT_LAST_DAY,
    HELIOS_SEPT_MONTH,
    POOL_SNAPSHOT_DATE,
    convert_helios,
    month_window,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Job

__all__ = [
    "per_vc_replay",
    "replay_canonical",
    "HELIOS_CLUSTERS",
    "DEFAULT_GPUS_PER_NODE",
]

#: The four Helios clusters (data.zip subdirectories), Table 1.
HELIOS_CLUSTERS: tuple[str, ...] = ("Venus", "Earth", "Saturn", "Uranus")

#: Uniform GPUs per node across all four clusters (Table 1).
DEFAULT_GPUS_PER_NODE: int = 8

#: One-day tail added past the last job's completion so every windowed job
#: reaches a terminal status before the horizon (a running job at the
#: horizon would be excluded from the JCT adapter and bias the mean).
_HORIZON_MARGIN_S: float = 86_400.0

#: canonical ``final_status`` -> replay terminal-status override (None =
#: natural COMPLETED).  Mirrors ``workload.trace._STATUS_OVERRIDE`` (kept
#: local to avoid importing a private name).
_STATUS_OVERRIDE: dict[str, JobStatus | None] = {
    "COMPLETED": None,
    "FAILED": JobStatus.FAILED,
    "CANCELED": JobStatus.CANCELED,
    "TIMEOUT": JobStatus.TIMEOUT,
    "NODE_FAIL": JobStatus.NODE_FAIL,
}

#: canonical ``class`` -> JobClass (Helios rows are always ``finetune``;
#: mapped defensively so a future caller passing other classes still
#: works).
_JOB_CLASS: dict[str, JobClass] = {
    "pretrain": JobClass.PRETRAIN,
    "finetune": JobClass.FINETUNE,
    "eval": JobClass.EVAL,
    "infer_replica": JobClass.INFER_REPLICA,
}


# ---------------------------------------------------------------------------
# Trace resolution (fetch + extract)
# ---------------------------------------------------------------------------


def _resolve_cluster_dir(
    cluster: str,
    cache_dir: str | Path | None,
    data_dir: str | Path | None,
) -> Path:
    """Locate a directory holding ``cluster_log.csv`` /
    ``cluster_gpu_number.csv`` for ``cluster``.

    ``data_dir`` (an already-extracted ``.../data`` root or a directory
    directly containing the cluster subdir) short-circuits the fetch.
    Otherwise the real ``data.zip`` is fetched into the cache and extracted
    once (idempotently) next to it, and the cluster subdirectory is
    returned.
    """
    if data_dir is not None:
        root = Path(data_dir).expanduser()
        for cand in (root / cluster, root / "data" / cluster, root):
            if (cand / "cluster_log.csv").is_file():
                return cand
        raise FileNotFoundError(
            f"per_vc_replay: no cluster_log.csv for {cluster!r} under {root}"
        )

    zip_path = fetch_trace("helios", cache_dir)
    if is_lfs_pointer(zip_path):  # pragma: no cover - helios is not LFS
        raise ValueError(f"{zip_path} is a Git-LFS pointer, not the real data.zip")
    extract_root = zip_path.parent / "extracted"
    cluster_dir = extract_root / "data" / cluster
    if not (cluster_dir / "cluster_log.csv").is_file():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
    if not (cluster_dir / "cluster_log.csv").is_file():
        # Fall back to a recursive search (archive layout drift).
        hits = list(extract_root.rglob(f"{cluster}/cluster_log.csv"))
        if not hits:
            raise FileNotFoundError(
                f"per_vc_replay: {cluster}/cluster_log.csv not found in {zip_path}"
            )
        return hits[0].parent
    return cluster_dir


# ---------------------------------------------------------------------------
# Job construction (checkpointing disabled -> service time == duration)
# ---------------------------------------------------------------------------


def _clean_chip_type(value: Any) -> str | None:
    """An empty / null ``chip_type`` cell -> ``None`` (unpinned gang)."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "None", "<NA>"):
        return None
    return text


def _build_jobs(rows: "pd.DataFrame") -> list["Job"]:
    """Build replay jobs from windowed canonical rows, with checkpointing
    disabled (service time == trace ``duration_s`` exactly) and the
    walltime estimate == duration (SJF-oracle).

    Mirrors :func:`fleetsim.workload.trace.load_trace`'s field mapping but
    zeroes the checkpoint/restart overheads and skips the CSV round-trip.
    Columns are read by name (``class`` is a Python keyword, so
    ``itertuples`` attribute access is not usable).
    """
    job_id = rows["job_id"].to_numpy()
    tenant = rows["tenant"].to_numpy()
    klass = rows["class"].to_numpy()
    submit = rows["submit_time"].to_numpy()
    num_chips = rows["num_chips"].to_numpy()
    chip_type = rows["chip_type"].to_numpy()
    duration = rows["duration_s"].to_numpy()
    walltime = rows["walltime_limit_s"].to_numpy()
    final_status = rows["final_status"].to_numpy()

    jobs: list["Job"] = []
    for i in range(len(rows)):
        job_class = _JOB_CLASS.get(str(klass[i]).strip().lower(), JobClass.FINETUNE)
        tier = (
            Tier.PROD
            if job_class in (JobClass.PRETRAIN, JobClass.INFER_REPLICA)
            else Tier.BATCH
        )
        wl = walltime[i]
        walltime_est_s = (
            None if wl is None or (isinstance(wl, float) and math.isnan(wl)) else float(wl)
        )
        jobs.append(
            TraceJob(
                id=str(job_id[i]),
                tenant=str(tenant[i]),
                job_class=job_class,
                submit_t=int(submit[i]),
                gangs=[
                    GangSpec(chips=int(num_chips[i]), chip_type=_clean_chip_type(chip_type[i]))
                ],
                tier=tier,
                min_runtime_s=0.0,
                walltime_est_s=walltime_est_s,
                true_duration_s=float(duration[i]),
                checkpoint_interval_s=0.0,
                checkpoint_save_s=0.0,
                restart_overhead_s=0.0,
                terminal_status_override=_STATUS_OVERRIDE.get(
                    str(final_status[i]).strip().upper()
                ),
            )
        )
    return jobs


# ---------------------------------------------------------------------------
# One-VC simulation
# ---------------------------------------------------------------------------


def _vc_scenario(
    name: str,
    vc_nodes: int,
    scheduler_name: str,
    horizon_s: int,
    gpus_per_node: int,
    round_s: float,
    strict: bool,
) -> dict[str, Any]:
    """A compact 1-cluster scenario dict for one VC: ``vc_nodes`` nodes of
    ``gpus_per_node`` v100 GPUs, failures off.  The ``workload.source`` is
    a placeholder (the harness feeds a :class:`TraceSource` directly,
    bypassing ``_make_source``)."""
    return {
        "sim": {"horizon": int(horizon_s), "round": f"{int(round_s)}s", "seed": 0},
        "fleet": {
            "metro": "helios",
            "clusters": [
                {
                    "name": name,
                    "chip": {"type": "v100", "per_node": gpus_per_node},
                    "topology": {"levels": ["node"], "counts": [int(vc_nodes)]},
                }
            ],
        },
        "failure_model": {
            "node_mtbf_days": 0,
            "maintenance_rate_per_node_month": 0,
        },
        # kind=trace requires a source string to validate; unused (we pass
        # a TraceSource object to the Simulator directly).
        "workload": {"kind": "trace", "source": "__inline__"},
        "scheduler": {"name": scheduler_name, "params": {"strict": bool(strict)}},
    }


def _estimate_horizon_s(jobs: list["Job"], pool_chips: int) -> int:
    """A completion-time estimate (seconds) generous enough that every job
    terminates in one pass for the common case.

    ``base`` is the last natural completion ignoring queue
    (``max(submit + duration)``); ``drain`` is the minimum busy time to
    push all work through the pool (``total_chip_seconds / pool_chips``) —
    a proxy for how far an over-subscribed VC's backlog spills past the
    window under a blocking scan.  Under strict FIFO with gang
    fragmentation the real completion can exceed ``base + drain``, which
    is why :func:`_run_one_vc` verifies and extends adaptively."""
    base_s = 0.0
    total_chip_s = 0.0
    for j in jobs:
        end = j.submit_t / 1e6 + j.true_duration_s
        base_s = max(base_s, end)
        total_chip_s += j.gangs[0].chips * j.true_duration_s
    drain_s = total_chip_s / pool_chips if pool_chips > 0 else 0.0
    return int(math.ceil(base_s + 1.5 * drain_s + _HORIZON_MARGIN_S))


def _run_one_vc(
    jobs: list["Job"],
    vc_nodes: int,
    scheduler_name: str,
    cluster: str,
    vc: str,
    gpus_per_node: int,
    round_s: float,
    strict: bool,
) -> "pd.DataFrame":
    """Run one VC's jobs through an independent simulation and return the
    per-job ``jobs.parquet`` frame (via
    :func:`fleetsim.metrics.summary.jobs_dataframe`).

    The scheduler runs **event-driven** (``wake_interval = None``): it is
    woken on every arrival and completion (which mark the engine dirty) but
    NOT on idle rounds, so a strict-FIFO pool blocked behind a large gang
    costs nothing until a completion frees capacity.  This is identical to
    the round-periodic schedule for a work-conserving list scheduler
    (``_mark_dirty`` quantizes each wake to the next round boundary) but
    far cheaper on the deeply-queued congested VCs.

    The horizon is estimated then VERIFIED: if any job is still
    non-terminal at the horizon (a backlog that spilled further than the
    estimate), the horizon is extended and the VC re-run, so the JCT /
    queuing means are never biased by a truncated long-waiting job (the
    very jobs that carry the FIFO-vs-SJF signal)."""
    pool_chips = int(vc_nodes) * int(gpus_per_node)
    horizon_s = max(_estimate_horizon_s(jobs, pool_chips), int(round_s) * 2)

    for _ in range(8):  # adaptive: converges (total work is finite)
        doc = _vc_scenario(
            f"{cluster}-{vc}",
            vc_nodes,
            scheduler_name,
            horizon_s,
            gpus_per_node,
            round_s,
            strict,
        )
        scenario = load_scenario(doc, strict=True)
        fleet = build_fleet(scenario)
        source = TraceSource(jobs, fleet=fleet)  # DESIGN §4.1 quantization
        scheduler = get_scheduler(scenario.scheduler.name, scenario.scheduler.params)
        scheduler.wake_interval = None  # event-driven: no idle-round wakes
        collector = MetricsCollector.from_scenario(scenario, fleet)
        sim = Simulator(scenario, fleet, source, scheduler, collector)
        sim.run()
        jobs_df = jobs_dataframe(collector)
        if _n_terminal(jobs_df) == len(jobs_df):
            return jobs_df
        horizon_s = int(horizon_s * 1.8)
    return jobs_df  # give up after 8 doublings (should never happen)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _avg_queuing(jobs_df: "pd.DataFrame") -> float:
    """Mean ``queue_wait_s`` over terminal jobs (the papers' "Avg
    Queuing"); ``nan`` when there are none.  Terminal jobs all started, so
    every wait is defined."""
    term = jobs_df[jobs_df["status"].isin(list(TERMINAL_STATUSES))]
    if len(term) == 0:
        return math.nan
    return float(term["queue_wait_s"].mean())


def _n_terminal(jobs_df: "pd.DataFrame") -> int:
    return int(jobs_df["status"].isin(list(TERMINAL_STATUSES)).sum())


def _month_max_pools(
    gpu_number_csv_path: str | Path,
    month: str,
    gpus_per_node: int,
) -> dict[str, int]:
    """Per-VC node counts from each VC's **maximum** GPU quota over ``month``
    (rows whose ``date`` starts with ``<month>``), rather than a single-day
    snapshot.

    The Sept-1 snapshot (``convert_helios``'s default) mis-sizes any VC
    whose quota drifts within the window — on Helios Uranus, ``vcUV3`` grows
    208->328 GPU and ``vc7hD`` spins up 0->416 GPU during September, so a
    Sept-1 pool over-congests the replay and inflates the FIFO/SJF ratio
    (measured Uranus JCT ratio 3.41 vs published 1.49).  Sizing each VC to
    its September-max quota recovers the published Uranus numbers (ratio
    1.69, FIFO JCT 20,833 vs published 19,758) — see the validation notes.
    """
    import pandas as _pd

    g = _pd.read_csv(gpu_number_csv_path)
    rows = g[g["date"].astype("string").str.startswith(month)]
    pools: dict[str, int] = {}
    for vc in g.columns:
        if vc in ("date", "total"):
            continue
        qmax = int(_pd.to_numeric(rows[vc], errors="coerce").max() or 0)
        if qmax > 0:
            pools[str(vc)] = -(-qmax // gpus_per_node)  # ceil division
    return pools


def per_vc_replay(
    cluster: str,
    month: str,
    scheduler_name: str,
    cache_dir: str | Path | None = None,
    *,
    last_day: int | None = None,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    round_s: float = 60.0,
    strict: bool = True,
    pool_snapshot: str = POOL_SNAPSHOT_DATE,
    data_dir: str | Path | None = None,
    progress_cb: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay one Helios ``cluster`` for one ``month`` under one scheduler,
    one independent simulation per active VC, and return cluster-aggregated
    metrics (plan §4).

    Parameters
    ----------
    cluster:
        One of :data:`HELIOS_CLUSTERS` (``"Venus"`` / ``"Earth"`` /
        ``"Saturn"`` / ``"Uranus"``).
    month:
        Calendar month ``"YYYY-MM"``.  For the Helios September validation
        window (plan §2 V1) pass ``"2020-09"``: ``last_day`` then defaults
        to 26 (the ``2020-09-01 .. 2020-09-26`` window).
    scheduler_name:
        ``"fifo"`` or ``"sjf"``.
    strict:
        Head-of-line-blocking scan (default ``True``).  **This is the
        load-bearing knob for reproducing Table 3** — the published huge
        FIFO queuing is head-of-line blocking (a large gang at the front
        stalls the queue), which only a *strict* scan reproduces; a
        best-effort scan lets small jobs flow around the stall and
        collapses the FIFO-vs-SJF ratio (see the module notes).  Both
        ``fifo`` and ``sjf`` use the same scan mode, matching the
        reference sim (one framework, two orderings).
    cache_dir:
        Trace cache root override (else ``$FLEETSIM_TRACE_CACHE`` /
        ``~/.cache/fleetsim/traces``).
    last_day:
        Window's inclusive last calendar day; defaults to 26 for the Sept
        window, else the month's last day.
    gpus_per_node, round_s:
        Fleet node size and scheduler round (both pinned to the reference:
        8 GPU/node, 60 s round).
    pool_snapshot:
        How to size each VC's pool.  A ``"YYYY-MM-DD"`` date (default the
        plan's ``2020-09-01`` snapshot) sizes from that day's
        ``cluster_gpu_number`` row; ``"max"`` sizes each VC to its peak
        quota over ``month`` (:func:`_month_max_pools`).  The Sept-1
        snapshot mis-sizes clusters whose per-VC quota drifts during the
        window (Uranus especially): a VC that spins up mid-month has zero
        Sept-1 quota, so its jobs are **dropped** (surfaced as
        ``n_dropped`` / ``dropped_vcs``) and the FIFO/SJF ratio is
        over-inflated.  ``"max"`` sizes every active VC (``n_dropped == 0``)
        and recovers the published cross-cluster ranks; it is the sizing
        the shipped V1 validation uses.  It is a deliberate capacity-model
        choice, not unambiguously "more faithful" than a snapshot — ``max``
        over-provisions VCs early in the month (a VC's peak quota is
        available from day 1) and so under-counts early-month queuing.  The
        two are alternatives with different biases; ``max`` is chosen
        because it loses no jobs and reproduces the published rank.
    data_dir:
        An already-extracted Helios ``data`` root, to skip the fetch.
    progress_cb:
        Optional ``(vc, info)`` callback invoked after each VC completes,
        with ``info`` = ``{done, total, n_jobs, n_terminal}``.

    Returns
    -------
    dict:
        ``avg_jct`` (job-weighted mean JCT over all terminal jobs, s),
        ``avg_queuing`` (job-weighted mean queue wait, s), ``n_queuing``
        (summed count of jobs waiting > one round), plus diagnostics:
        ``n_jobs`` (replayed), ``n_terminal``, ``n_windowed`` (all windowed
        jobs), ``n_dropped`` (windowed jobs whose VC was absent from the
        pool sizing and thus NOT replayed — ``0`` under ``"max"``),
        ``dropped_vcs`` (their VC names), ``per_vc`` (per-VC metric dict),
        ``pool_nodes`` (per-VC node counts), ``cluster``, ``scheduler``,
        ``window``.
    """
    cluster_dir = _resolve_cluster_dir(cluster, cache_dir, data_dir)
    gpu_csv = cluster_dir / "cluster_gpu_number.csv"
    snapshot_date = "2020-09-01" if pool_snapshot == "max" else pool_snapshot
    df, pools = convert_helios(
        cluster_dir / "cluster_log.csv",
        gpu_csv,
        gpus_per_node=gpus_per_node,
        pool_snapshot_date=snapshot_date,
    )
    ld = last_day
    if ld is None and month == HELIOS_SEPT_MONTH:
        ld = HELIOS_SEPT_LAST_DAY
    win = month_window(df, month, last_day=ld)
    if pool_snapshot == "max":
        # Size each VC to its peak quota over the window month rather than a
        # single-day snapshot (recovers clusters whose per-VC quota drifts;
        # see _month_max_pools).
        pools = _month_max_pools(gpu_csv, month, gpus_per_node)
    out = replay_canonical(
        win,
        pools,
        scheduler_name,
        cluster=cluster,
        gpus_per_node=gpus_per_node,
        round_s=round_s,
        strict=strict,
        progress_cb=progress_cb,
    )
    out["window"] = f"{month} (last_day={ld})"
    out["pool_snapshot"] = pool_snapshot
    return out


def replay_canonical(
    win: "pd.DataFrame",
    pools: dict[str, int],
    scheduler_name: str,
    *,
    cluster: str = "cluster",
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    round_s: float = 60.0,
    strict: bool = True,
    progress_cb: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay an already-windowed canonical frame per VC and aggregate to
    cluster totals (the core of :func:`per_vc_replay`, factored out so a
    vendored slice — read straight into a canonical DataFrame — can be
    replayed without a fetch).

    ``win`` is a canonical-schema frame (``submit_time`` int microseconds,
    re-based to the window start); ``pools`` maps VC -> node count.  Only
    VCs present in BOTH ``pools`` and ``win`` are replayed.  Returns the
    same dict shape as :func:`per_vc_replay` (minus ``window``).

    DROP DIAGNOSTICS.  A windowed job whose VC is **absent** from ``pools``
    (e.g. a VC with zero quota on the ``pool_snapshot`` day but jobs later
    in the month) carries no pool and is therefore NOT replayed.  Such jobs
    are counted, not silently discarded: the returned ``n_windowed`` (all
    windowed jobs), ``n_dropped`` (``n_windowed - n_replayed``), and
    ``dropped_vcs`` make the loss visible so a caller/test can assert it is
    zero.  A nonzero ``n_dropped`` flags a lossy sizing — the Sept-1
    snapshot drops VCs that spin up mid-month; ``pool_snapshot='max'``
    (what the shipped V1 validation uses) sizes every active VC, so
    ``n_dropped == 0``.  ``n_jobs`` is the REPLAYED count; the
    ``n_terminal == n_jobs`` guard therefore only proves no *replayed* job
    was truncated — pair it with ``n_dropped == 0`` to prove no windowed
    job was lost.
    """
    win_vcs = {str(v) for v in win["tenant"].unique()} if len(win) else set()
    active_vcs = [vc for vc in sorted(pools) if vc in win_vcs]
    dropped_vcs = sorted(win_vcs - set(pools))
    n_windowed = int(len(win))
    frames: list["pd.DataFrame"] = []
    per_vc: dict[str, dict[str, Any]] = {}
    total = len(active_vcs)
    for i, vc in enumerate(active_vcs, start=1):
        rows = win[win["tenant"] == vc]
        jobs = _build_jobs(rows)
        jobs_df = _run_one_vc(
            jobs, pools[vc], scheduler_name, cluster, vc, gpus_per_node, round_s, strict
        )
        frames.append(jobs_df)
        per_vc[vc] = {
            "nodes": int(pools[vc]),
            "n_jobs": int(len(jobs_df)),
            "n_terminal": _n_terminal(jobs_df),
            "avg_jct": jct_over_all_terminal(jobs_df),
            "avg_queuing": _avg_queuing(jobs_df),
            "n_queuing": n_queuing_jobs(jobs_df, round_s),
        }
        if progress_cb is not None:
            progress_cb(
                vc,
                {
                    "done": i,
                    "total": total,
                    "n_jobs": int(len(jobs_df)),
                    "n_terminal": per_vc[vc]["n_terminal"],
                },
            )

    if frames:
        alljobs = pd.concat(frames, ignore_index=True)
    else:  # pragma: no cover - a cluster always has active VCs
        alljobs = jobs_dataframe(MetricsCollector(window=(0, 1)))

    return {
        "cluster": cluster,
        "scheduler": scheduler_name,
        "avg_jct": jct_over_all_terminal(alljobs),
        "avg_queuing": _avg_queuing(alljobs),
        "n_queuing": n_queuing_jobs(alljobs, round_s),
        "n_jobs": int(len(alljobs)),
        "n_terminal": _n_terminal(alljobs),
        "n_windowed": n_windowed,
        "n_dropped": n_windowed - int(len(alljobs)),
        "dropped_vcs": dropped_vcs,
        "pool_nodes": {vc: int(pools[vc]) for vc in active_vcs},
        "per_vc": per_vc,
    }
