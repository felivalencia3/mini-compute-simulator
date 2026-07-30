"""Adapter for the Helios trace (S-Lab-System-Group/HeliosData, SC '21).

:func:`convert_helios` reads one cluster's ``cluster_log.csv`` and emits
rows in fleetsim's canonical trace schema
(:data:`fleetsim.workload.trace.CANONICAL_COLUMNS`) as a pandas
``DataFrame``, ready for :func:`fleetsim.workload.trace.write_trace` /
:class:`fleetsim.workload.trace.TraceSource`.  It also returns the
per-VC pool sizes (node counts) read from ``cluster_gpu_number.csv``, so
the per-VC replay harness (validation plan §4) can size one fleet per VC.

REAL HELIOS SCHEMA (verified against the released ``data.zip``)
--------------------------------------------------------------
``cluster_log.csv`` header (one row per job)::

    job_id,user,vc,gpu_num,cpu_num,node_num,state,submit_time,
    start_time,end_time,duration,queue

- ``submit_time`` / ``start_time`` / ``end_time``: ``"%Y-%m-%d %H:%M:%S"``
  wall-clock strings (NOT epoch, NO timezone).  ``convert_helios`` also
  accepts a numeric ``submit_time`` column as epoch **seconds** for
  robustness, but the released trace is string-typed.
- ``duration``: integer **seconds** the job ran (``end - start``); a few
  rows slightly exceed the 14-day Slurm max and are capped (see below).
- ``state`` enum (observed across all four clusters): ``COMPLETED``,
  ``CANCELLED`` (British spelling — two L's), ``FAILED``, ``TIMEOUT``,
  ``NODE_FAIL``, plus a negligible tail of non-terminal ``RUNNING`` /
  ``SUSPENDED`` (54 of 1.58M GPU rows) that are right-censored and dropped.

``cluster_gpu_number.csv`` is a date x VC pivot: a ``date`` column, one
column per VC holding that VC's GPU quota on that date, and a ``total``
column.  Pool sizes come from the **row** whose ``date`` is the snapshot
date (default ``2020-09-01``); each VC's GPU count / 8 is its node count.

HELIOS -> CANONICAL MAPPING (pinned)
------------------------------------
=================  ====================================================
canonical column   source
=================  ====================================================
job_id             ``job_id`` (stringified)
user               ``user``
tenant             ``vc`` (the virtual cluster)
class              ``"finetune"`` for every row (Helios jobs are DL
                   training jobs; ``finetune`` -> tier BATCH, so the job
                   is INCLUDED in the papers' JCT/queue distributions)
submit_time        ``submit_time`` parsed, converted to int MICROSECONDS
                   relative to the earliest kept row (the trace epoch)
num_chips          ``gpu_num``
chip_type          empty (unpinned gang -> matches the single-SKU fleet)
num_nodes          ``node_num``
duration_s         ``min(duration, 1209600)`` (capped at the 14-day max)
walltime_limit_s   the SAME capped duration -> ``walltime_est_s``.  Writing
                   the true duration into the scheduler-visible estimate
                   makes ``sjf`` an SJF-**oracle**, the Helios Table-3 SJF
                   reference (validation plan §2 V1).
final_status       ``COMPLETED`` -> COMPLETED, ``CANCELLED`` -> CANCELED,
                   ``FAILED`` -> FAILED, ``TIMEOUT`` -> TIMEOUT,
                   ``NODE_FAIL`` -> NODE_FAIL
=================  ====================================================

Rows are DROPPED (not errors) when: ``gpu_num <= 0`` (CPU-only jobs — the
paper's GPU-job analysis excludes them), ``submit_time`` is before
``2020-04-01`` (the released trace carries a March warm-up tail the paper
window excludes) or unparseable, or ``state`` is not one of the five
terminal states above (the ``RUNNING`` / ``SUSPENDED`` censored tail).

INVARIANTS: a pure function of its inputs — no randomness, no wall clock;
rows are returned sorted by ``(submit_time, job_id)``.  The trace epoch
(a ``pandas.Timestamp``) is stored on the returned frame's ``.attrs``
under ``"trace_epoch"`` so :func:`month_window` can reconstruct absolute
submit times and slice a calendar window.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..workload.trace import CANONICAL_COLUMNS

__all__ = [
    "convert_helios",
    "month_window",
    "DURATION_CAP_S",
    "MIN_SUBMIT_DATE",
    "DEFAULT_GPUS_PER_NODE",
    "POOL_SNAPSHOT_DATE",
    "HELIOS_SEPT_MONTH",
    "HELIOS_SEPT_LAST_DAY",
    "STATE_MAP",
]

#: 14-day Slurm walltime max; a handful of trace rows exceed it slightly
#: (``end - start`` with clock skew) and are capped here (plan §2 V2(e)).
DURATION_CAP_S: int = 1_209_600

#: The released trace begins ~2020-03-18 with a warm-up tail; the paper's
#: analysis window starts in April, so earlier submits are dropped.
MIN_SUBMIT_DATE: str = "2020-04-01"

#: Uniform GPUs per node across all four Helios clusters (Table 1).
DEFAULT_GPUS_PER_NODE: int = 8

#: The ``cluster_gpu_number.csv`` row used for per-VC pool sizing (the
#: Sept-1 snapshot whose per-cluster totals match Table 1:
#: Venus 1064 / Earth 1144 / Saturn 2096 / Uranus 2112 GPUs).
POOL_SNAPSHOT_DATE: str = "2020-09-01"

#: The Helios September validation window (plan §2 V1(d)):
#: 2020-09-01 00:00:00 .. 2020-09-26 23:59:59 inclusive.  Call
#: ``month_window(df, HELIOS_SEPT_MONTH, last_day=HELIOS_SEPT_LAST_DAY)``.
HELIOS_SEPT_MONTH: str = "2020-09"
HELIOS_SEPT_LAST_DAY: int = 26

#: Helios ``state`` -> canonical ``final_status``.  ``CANCELLED`` (Helios's
#: British spelling) maps to fleetsim's ``CANCELED``.  States absent here
#: (``RUNNING`` / ``SUSPENDED`` and anything unknown) are non-terminal and
#: their rows are dropped.
STATE_MAP: dict[str, str] = {
    "COMPLETED": "COMPLETED",
    "CANCELLED": "CANCELED",
    "FAILED": "FAILED",
    "TIMEOUT": "TIMEOUT",
    "NODE_FAIL": "NODE_FAIL",
}

#: cluster_log.csv columns convert_helios requires (extras are ignored).
_REQUIRED_LOG_COLUMNS: tuple[str, ...] = (
    "job_id",
    "user",
    "vc",
    "gpu_num",
    "node_num",
    "state",
    "submit_time",
    "duration",
)

_TRACE_EPOCH_ATTR = "trace_epoch"


def _parse_submit(col: "pd.Series") -> "pd.Series":
    """Parse the ``submit_time`` column to datetimes.

    The released Helios trace stores wall-clock strings
    (``%Y-%m-%d %H:%M:%S``); a numeric column is treated as epoch
    **seconds** for robustness.  Unparseable values become ``NaT`` and are
    dropped by the caller.
    """
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="s", errors="coerce")
    return pd.to_datetime(col, errors="coerce")


def convert_helios(
    cluster_log_csv_path: str | Path,
    gpu_number_csv_path: str | Path | None = None,
    *,
    min_submit_date: str = MIN_SUBMIT_DATE,
    duration_cap_s: int = DURATION_CAP_S,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    pool_snapshot_date: str = POOL_SNAPSHOT_DATE,
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Convert a Helios ``cluster_log.csv`` to a canonical-schema frame.

    Parameters
    ----------
    cluster_log_csv_path:
        One cluster's ``cluster_log.csv``.
    gpu_number_csv_path:
        That cluster's ``cluster_gpu_number.csv`` (optional).  When given,
        the second return value is ``{vc: node_count}`` for every VC with a
        nonzero GPU quota on ``pool_snapshot_date`` (GPUs / ``gpus_per_node``,
        rounded up).  When ``None``, the pool dict is empty.
    min_submit_date, duration_cap_s, gpus_per_node, pool_snapshot_date:
        Tunables for the drops/caps/snapshot documented on the module.

    Returns
    -------
    (df, vc_pool_sizes):
        ``df`` has exactly :data:`CANONICAL_COLUMNS`, one row per kept GPU
        job, sorted by ``(submit_time, job_id)``, with ``submit_time`` in
        int microseconds since the trace epoch (stored on
        ``df.attrs["trace_epoch"]``).  ``vc_pool_sizes`` maps VC -> node
        count.

    Raises
    ------
    ValueError:
        If required columns are missing, or (when a gpu-number path is
        given) the snapshot date is absent from that file.
    """
    df = pd.read_csv(cluster_log_csv_path)
    missing = [c for c in _REQUIRED_LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{cluster_log_csv_path}: cluster_log.csv is missing column(s): "
            f"{', '.join(missing)} (have: {', '.join(map(str, df.columns))})"
        )

    # 1) Drop CPU-only jobs (the paper's GPU analysis excludes gpu_num==0).
    df = df[pd.to_numeric(df["gpu_num"], errors="coerce").fillna(0) > 0]

    # 2) Parse submit_time; drop unparseable and pre-window submits.
    submit_dt = _parse_submit(df["submit_time"])
    keep = submit_dt.notna() & (submit_dt >= pd.Timestamp(min_submit_date))
    df = df[keep]
    submit_dt = submit_dt[keep]

    # 3) Map state -> canonical final_status; drop non-terminal/unknown.
    final_status = df["state"].map(STATE_MAP)
    keep = final_status.notna()
    df = df[keep]
    submit_dt = submit_dt[keep]
    final_status = final_status[keep]

    if len(df) == 0:
        out = _empty_canonical()
        out.attrs[_TRACE_EPOCH_ATTR] = None
        return out, _vc_pool_sizes(
            gpu_number_csv_path, gpus_per_node, pool_snapshot_date
        )

    # 4) Cap duration at the 14-day Slurm max; write it into BOTH the true
    #    duration and the scheduler-visible estimate (SJF-oracle).
    duration_s = (
        pd.to_numeric(df["duration"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=float(duration_cap_s))
        .astype("float64")
    )

    # 5) Trace epoch = earliest kept submit; submit_time -> int microseconds.
    epoch = submit_dt.min()
    submit_us = ((submit_dt - epoch) // pd.Timedelta(microseconds=1)).astype("int64")

    out = pd.DataFrame(
        {
            "job_id": df["job_id"].astype("string"),
            "user": df["user"].astype("string").fillna(""),
            "tenant": df["vc"].astype("string"),
            "class": "finetune",
            "submit_time": submit_us.to_numpy(),
            "num_chips": pd.to_numeric(df["gpu_num"]).astype("int64").to_numpy(),
            "chip_type": "",
            "num_nodes": pd.to_numeric(df["node_num"]).astype("int64").to_numpy(),
            "duration_s": duration_s.to_numpy(),
            "walltime_limit_s": duration_s.to_numpy(),
            "final_status": final_status.astype("string").to_numpy(),
        },
        columns=list(CANONICAL_COLUMNS),
    )
    out = out.sort_values(["submit_time", "job_id"], kind="stable").reset_index(
        drop=True
    )
    out.attrs[_TRACE_EPOCH_ATTR] = pd.Timestamp(epoch)

    pools = _vc_pool_sizes(gpu_number_csv_path, gpus_per_node, pool_snapshot_date)
    return out, pools


def _empty_canonical() -> "pd.DataFrame":
    """An empty canonical frame with the right columns/dtypes."""
    return pd.DataFrame(
        {c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS},
        columns=list(CANONICAL_COLUMNS),
    )


def _vc_pool_sizes(
    gpu_number_csv_path: str | Path | None,
    gpus_per_node: int,
    pool_snapshot_date: str,
) -> dict[str, int]:
    """``{vc: node_count}`` from the snapshot row of ``cluster_gpu_number.csv``.

    Node count = ceil(GPUs / ``gpus_per_node``) for every VC column with a
    positive GPU quota on ``pool_snapshot_date`` (the ``date`` /``total``
    columns are not VCs).  ``None`` path -> ``{}``.
    """
    if gpu_number_csv_path is None:
        return {}
    g = pd.read_csv(gpu_number_csv_path)
    if "date" not in g.columns:
        raise ValueError(
            f"{gpu_number_csv_path}: cluster_gpu_number.csv is missing a "
            f"'date' column (have: {', '.join(map(str, g.columns))})"
        )
    row = g[g["date"].astype("string") == pool_snapshot_date]
    if len(row) == 0:
        raise ValueError(
            f"{gpu_number_csv_path}: no row for snapshot date "
            f"{pool_snapshot_date!r} (available: "
            f"{g['date'].iloc[0]} .. {g['date'].iloc[-1]})"
        )
    r = row.iloc[0]
    pools: dict[str, int] = {}
    for vc in g.columns:
        if vc in ("date", "total"):
            continue
        gpus = int(pd.to_numeric(r[vc], errors="coerce") or 0)
        if gpus > 0:
            pools[str(vc)] = -(-gpus // gpus_per_node)  # ceil division
    return pools


def month_window(
    df: "pd.DataFrame",
    month: str,
    *,
    last_day: int | None = None,
    epoch: "pd.Timestamp | str | None" = None,
) -> "pd.DataFrame":
    """Slice a converted-Helios frame to one calendar window and re-base
    ``submit_time`` so the window start is ``t = 0``.

    The window is ``[<month>-01 00:00:00, <month>-<last_day> 23:59:59.999999]``
    inclusive.  ``last_day`` defaults to the calendar month's last day; for
    the Helios September validation window (plan §2 V1) pass
    ``last_day=26`` -> ``2020-09-01 .. 2020-09-26``.

    Absolute submit times are reconstructed from the trace epoch: either
    the ``epoch`` argument or ``df.attrs["trace_epoch"]`` (set by
    :func:`convert_helios`).  The returned frame's ``submit_time`` is
    re-based to the window start (so ``submit_time >= 0`` and the span fits
    a month-sized horizon), it is sorted by ``(submit_time, job_id)``, and
    its ``.attrs["trace_epoch"]`` is updated to the window start.

    Raises ``ValueError`` when no epoch is available to anchor absolute time.
    """
    ep = epoch if epoch is not None else df.attrs.get(_TRACE_EPOCH_ATTR)
    if ep is None:
        raise ValueError(
            "month_window: no trace epoch available (pass epoch=..., or use a "
            "frame from convert_helios which sets df.attrs['trace_epoch'])"
        )
    ep = pd.Timestamp(ep)

    window_start = pd.Timestamp(f"{month}-01 00:00:00")
    last = last_day if last_day is not None else window_start.days_in_month
    last_day_start = pd.Timestamp(f"{month}-{last:02d} 00:00:00")
    window_end = last_day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    absolute = ep + pd.to_timedelta(df["submit_time"], unit="us")
    mask = (absolute >= window_start) & (absolute <= window_end)

    out = df[mask].copy()
    rebased = (absolute[mask] - window_start) // pd.Timedelta(microseconds=1)
    out["submit_time"] = rebased.astype("int64").to_numpy()
    out = out.sort_values(["submit_time", "job_id"], kind="stable").reset_index(
        drop=True
    )
    out.attrs[_TRACE_EPOCH_ATTR] = window_start
    return out
