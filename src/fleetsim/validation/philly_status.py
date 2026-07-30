"""Philly job-status split helpers (validation plan §2 V3).

The single most directly replayable scheduler-trace fact: the fraction of
jobs in each terminal status, by **count** and by **GPU-time**.  These are
properties of the *converted* trace rows — no simulation is involved — so
they test the converter's fidelity to the Philly paper (Jeon et al., ATC
'19, Table 6), not the scheduler.

Both helpers consume the output of
:func:`fleetsim.workload.philly.convert_philly` (a list of canonical-row
dicts) or an equivalent ``pandas.DataFrame`` (e.g. a canonical CSV read
with pandas).  They read three canonical columns: ``final_status``,
``num_chips``, and ``duration_s``.

PHILLY PAPER LABELS.  ``convert_philly`` maps Philly's raw states to
canonical statuses; these helpers relabel to the paper's Table-6 buckets::

    COMPLETED -> "Passed"        (Philly "Pass")
    CANCELED  -> "Killed"        (Philly "Killed")
    FAILED    -> "Unsuccessful"  (Philly "Failed")

Published targets (Table 6, on the paper's 96,260-job window):

- **By count**: Passed 69.3% / Killed 13.5% / Unsuccessful 17.2%.
- **By GPU-time** (``num_chips x run_time``): Passed 44.53% / Killed 37.69%
  / Unsuccessful 17.76% — Killed+Unsuccessful are ~30.7% of *jobs* but
  ~55.45% of *GPU-time*.

RIGHT-CENSORING (plan §2 V3(e)).  A job whose attempts had unparseable
times contributes ``duration_s == 0`` from the converter.  Those rows must
NOT be coerced into the GPU-time split as zero-time work; they are
**excluded** from the by-GPU-time aggregate (they still count in the
by-count split, which is a pure headcount).

INVARIANTS: pure functions of the rows — no I/O, no randomness, no wall
clock.  Each returns ``{label: share}`` whose values sum to 1.0 (empty
input -> ``{}``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "PHILLY_STATUS_LABELS",
    "status_split_by_count",
    "status_split_by_gpu_time",
]

#: canonical ``final_status`` -> Philly Table-6 bucket label.
PHILLY_STATUS_LABELS: dict[str, str] = {
    "COMPLETED": "Passed",
    "CANCELED": "Killed",
    "FAILED": "Unsuccessful",
}


def _as_frame(rows: "pd.DataFrame | Iterable[Mapping[str, object]]") -> "pd.DataFrame":
    """Normalize ``convert_philly`` output (list of dicts) or a DataFrame
    to a DataFrame (never copies an already-DataFrame input's data)."""
    import pandas as pd

    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(list(rows))


def _labels(df: "pd.DataFrame") -> "pd.Series":
    """The paper-bucket label for each row (unmapped statuses pass
    through verbatim, so an unexpected status is visible, not silently
    dropped)."""
    status = df["final_status"].astype("string")
    return status.map(lambda s: PHILLY_STATUS_LABELS.get(str(s), str(s)))


def status_split_by_count(
    rows: "pd.DataFrame | Iterable[Mapping[str, object]]",
) -> dict[str, float]:
    """Share of jobs in each Philly status bucket, **by count**.

    Returns ``{label: fraction}`` over the rows present (Passed / Killed /
    Unsuccessful for a converted Philly trace); the fractions sum to 1.0.
    Empty input -> ``{}``.
    """
    df = _as_frame(rows)
    if len(df) == 0:
        return {}
    counts = _labels(df).value_counts()
    total = float(counts.sum())
    return {str(k): float(v) / total for k, v in counts.sort_index().items()}


def status_split_by_gpu_time(
    rows: "pd.DataFrame | Iterable[Mapping[str, object]]",
) -> dict[str, float]:
    """Share of GPU-time (``num_chips x duration_s``) in each Philly status
    bucket, **by GPU-time**.

    Rows with ``duration_s <= 0`` (right-censored / unparseable-time
    attempts) are EXCLUDED from the aggregate rather than counted as
    zero-time work (plan §2 V3(e)).  Returns ``{label: fraction}`` summing
    to 1.0; empty (or all-censored) input -> ``{}``.
    """
    df = _as_frame(rows)
    if len(df) == 0:
        return {}
    import pandas as pd

    duration = pd.to_numeric(df["duration_s"], errors="coerce").fillna(0.0)
    chips = pd.to_numeric(df["num_chips"], errors="coerce").fillna(0.0)
    keep = duration > 0.0
    if not bool(keep.any()):
        return {}
    gpu_time = (chips[keep] * duration[keep]).astype("float64")
    labels = _labels(df)[keep]
    grouped = gpu_time.groupby(labels).sum()
    total = float(grouped.sum())
    if total <= 0.0:
        return {}
    return {str(k): float(v) / total for k, v in grouped.sort_index().items()}
