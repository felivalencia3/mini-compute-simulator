"""Metric adapters: recompute the papers' summary definitions from a
per-job DataFrame (``jobs.parquet``), where they diverge from
fleetsim's own ``summary.json`` aggregation.

Why adapters exist (validation plan §1).  fleetsim's per-job records
already carry everything the Helios/Philly papers report, but the
*summary-level* aggregation differs on two axes, so a faithful replay
must recompute from the raw rows rather than read ``summary.json``:

- **Average JCT.**  The papers average job completion time over **all**
  simulated GPU jobs — a FAILED / CANCELED / TIMEOUT job's duration
  still counts.  ``summary.json['window']['jct_s']`` instead filters to
  COMPLETED jobs only.  :func:`jct_over_all_terminal` recomputes the
  mean of per-job ``jct_s`` (``= end_t - submit_t``, present for every
  terminal row) over **every** terminal job.
- **# Queuing jobs.**  The papers count jobs that waited; fleetsim does
  not surface this as a summary field.  :func:`n_queuing_jobs` counts
  rows whose ``queue_wait_s`` exceeds one scheduler round (default 60 s,
  so sub-round quantization is not miscounted as a wait).

Both read ``jobs.parquet`` columns directly.  All three functions are
pure functions of the DataFrame — no I/O, no global state, no wall
clock — so they are trivially unit-testable against a hand-built frame.

Columns read (a subset of the ``jobs.parquet`` schema — see
:func:`fleetsim.metrics.summary.jobs_dataframe`):

- ``status``           : str terminal/queue state (``COMPLETED`` /
                         ``FAILED`` / ``CANCELED`` / ``TIMEOUT`` /
                         ``NODE_FAIL`` are terminal).
- ``jct_s``            : float job completion time (``end_t - submit_t``);
                         defined for every terminal job.
- ``queue_wait_s``     : float ``first_start_t - submit_t``.
- ``chips``            : int gang chip count.
- ``running_elapsed_s``: float wall seconds the job held its allocation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "TERMINAL_STATUSES",
    "jct_over_all_terminal",
    "n_queuing_jobs",
    "gpu_time_by_status",
]

#: The terminal job statuses (a job that reached one of these has a
#: defined ``jct_s``).  Mirrors ``metrics.summary._TERMINAL_STATUSES``.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"COMPLETED", "FAILED", "CANCELED", "TIMEOUT", "NODE_FAIL"}
)


def _terminal_rows(jobs_df: "pd.DataFrame") -> "pd.DataFrame":
    """The subset of ``jobs_df`` whose ``status`` is terminal."""
    return jobs_df[jobs_df["status"].isin(list(TERMINAL_STATUSES))]


def jct_over_all_terminal(jobs_df: "pd.DataFrame") -> float:
    """Mean ``jct_s`` over **all** terminal jobs (COMPLETED *and*
    FAILED/CANCELED/TIMEOUT/NODE_FAIL) — the Helios/Philly "average JCT"
    definition (validation plan §1), which diverges from
    ``summary.json``'s COMPLETED-only ``jct_s`` mean.

    Returns ``nan`` when there are no terminal jobs (an undefined mean),
    so callers get a propagating sentinel rather than a raised error.
    """
    term = _terminal_rows(jobs_df)
    if len(term) == 0:
        return math.nan
    return float(term["jct_s"].mean())


def n_queuing_jobs(jobs_df: "pd.DataFrame", round_s: float = 60.0) -> int:
    """Count jobs whose ``queue_wait_s`` **strictly exceeds** ``round_s``
    — the papers' "# Queuing Jobs" (validation plan §1).

    Thresholding at one scheduler round (default 60 s) keeps sub-round
    scheduling quantization from being miscounted as real queuing.
    Counts over every row present (a job that never started but has a
    recorded wait still counts if it waited longer than a round).
    """
    return int((jobs_df["queue_wait_s"] > round_s).sum())


def gpu_time_by_status(jobs_df: "pd.DataFrame") -> dict[str, float]:
    """GPU-time (chip-seconds) consumed, grouped by job ``status``.

    Per-job GPU-time is ``chips * running_elapsed_s`` — the wall-clock
    chip-seconds the job held its allocation (a job that never ran
    contributes 0).  Returns a ``{status: chip_seconds}`` dict over every
    status present in the frame, ordered by status name for determinism.

    This is a **simulation-side** allocation-GPU-time helper: it measures
    the chip-seconds a job actually held in a *replay* (``running_elapsed_s``
    includes lost/restart work when checkpointing is on).  It is NOT what
    the Philly V3 status split uses — that path is
    :func:`fleetsim.validation.philly_status.status_split_by_gpu_time`,
    which computes ``num_chips * duration_s`` straight from the converted
    trace rows (no simulation).  The two coincide only when a job never
    restarts; do not substitute this helper for the V3 split.  To bucket
    this dict by the paper's labels, map ``COMPLETED`` -> "Passed",
    ``CANCELED`` -> "Killed", ``FAILED`` -> "Unsuccessful".  Empty frame ->
    ``{}``.
    """
    if len(jobs_df) == 0:
        return {}
    gpu_s = jobs_df["chips"].astype("float64") * jobs_df["running_elapsed_s"].astype(
        "float64"
    )
    grouped = gpu_s.groupby(jobs_df["status"]).sum()
    return {str(k): float(v) for k, v in sorted(grouped.items())}
