"""Unit tests for the Helios trace converter (v0.6, checklist item 2).

Fast and network-free: a hand-built ``cluster_log.csv`` /
``cluster_gpu_number.csv`` fixture is fed through
:func:`fleetsim.validation.helios.convert_helios` and
:func:`fleetsim.validation.helios.month_window`, pinning every documented
rule — CPU-job drop, pre-April drop, state mapping (incl. the British
``CANCELLED`` spelling and dropped non-terminal states), the 14-day
duration cap, the dual ``duration_s`` / ``walltime_limit_s`` write
(SJF-oracle), per-VC pool sizing, and the September window boundaries.

The exact real-trace schema these fixtures mirror was verified against the
released HeliosData ``data.zip`` (see the converter module docstring).
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from fleetsim.validation.helios import (
    HELIOS_SEPT_LAST_DAY,
    HELIOS_SEPT_MONTH,
    convert_helios,
    month_window,
)
from fleetsim.workload.trace import CANONICAL_COLUMNS, load_trace

#: Real Helios cluster_log.csv header.
LOG_COLUMNS = (
    "job_id,user,vc,gpu_num,cpu_num,node_num,state,"
    "submit_time,start_time,end_time,duration,queue"
).split(",")


def _log_row(job_id, vc, gpu, state, submit, duration, *, node=1, user="uX"):
    """One cluster_log.csv row (start/end are unused by the converter,
    which reads the ``duration`` column directly)."""
    return {
        "job_id": job_id,
        "user": user,
        "vc": vc,
        "gpu_num": gpu,
        "cpu_num": gpu * 4,
        "node_num": node,
        "state": state,
        "submit_time": submit,
        "start_time": submit,
        "end_time": submit,
        "duration": duration,
        "queue": 0,
    }


def _write_log(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_gpu_number(path, dates, vc_columns):
    """A date x VC pivot with a trailing ``total`` column."""
    vcs = list(vc_columns)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", *vcs, "total"])
        for d, values in dates.items():
            row = [values.get(vc, 0) for vc in vcs]
            w.writerow([d, *row, sum(row)])


# ---------------------------------------------------------------------------
# Mapping, drops, cap, dual-write
# ---------------------------------------------------------------------------


def _mapping_log(path):
    rows = [
        # kept: COMPLETED 8-GPU job (earliest kept -> trace epoch)
        _log_row("j_ok", "vcA", 8, "COMPLETED", "2020-05-01 00:00:00", 100, node=1),
        # dropped: CPU-only job (gpu_num == 0)
        _log_row("j_cpu", "vcA", 0, "COMPLETED", "2020-05-01 01:00:00", 500),
        # dropped: submitted before 2020-04-01
        _log_row("j_march", "vcA", 8, "COMPLETED", "2020-03-15 00:00:00", 100),
        # kept: CANCELLED -> CANCELED (British spelling maps to canonical)
        _log_row("j_cancel", "vcB", 16, "CANCELLED", "2020-05-01 02:00:00", 200, node=2),
        # kept: FAILED
        _log_row("j_fail", "vcB", 4, "FAILED", "2020-05-01 03:00:00", 300),
        # kept: TIMEOUT, duration over the 14-day cap -> capped
        _log_row("j_to", "vcA", 8, "TIMEOUT", "2020-05-01 04:00:00", 1_500_000),
        # kept: NODE_FAIL
        _log_row("j_nf", "vcB", 32, "NODE_FAIL", "2020-05-01 05:00:00", 50, node=4),
        # dropped: non-terminal RUNNING / SUSPENDED
        _log_row("j_run", "vcA", 8, "RUNNING", "2020-05-01 06:00:00", 10),
        _log_row("j_susp", "vcA", 8, "SUSPENDED", "2020-05-01 07:00:00", 10),
    ]
    _write_log(path, rows)


def test_convert_helios_mapping_drops_and_cap(tmp_path):
    log = tmp_path / "cluster_log.csv"
    _mapping_log(log)
    df, pools = convert_helios(log)

    # Exactly the terminal, GPU, post-April rows survive.
    assert set(df["job_id"]) == {"j_ok", "j_cancel", "j_fail", "j_to", "j_nf"}
    assert list(df.columns) == list(CANONICAL_COLUMNS)

    by_id = df.set_index("job_id")
    # State mapping (CANCELLED -> CANCELED is the key British-spelling case).
    assert by_id.loc["j_ok", "final_status"] == "COMPLETED"
    assert by_id.loc["j_cancel", "final_status"] == "CANCELED"
    assert by_id.loc["j_fail", "final_status"] == "FAILED"
    assert by_id.loc["j_to", "final_status"] == "TIMEOUT"
    assert by_id.loc["j_nf", "final_status"] == "NODE_FAIL"

    # Duration cap at the 14-day Slurm max, and the dual write.
    assert by_id.loc["j_to", "duration_s"] == 1_209_600.0
    assert (df["duration_s"] == df["walltime_limit_s"]).all()

    # Canonical field mapping.
    assert by_id.loc["j_cancel", "num_chips"] == 16
    assert by_id.loc["j_cancel", "num_nodes"] == 2
    assert by_id.loc["j_cancel", "tenant"] == "vcB"
    assert (df["class"] == "finetune").all()
    assert (df["chip_type"] == "").all()

    # submit_time is int microseconds since the earliest kept row (j_ok @ 05-01
    # 00:00 -> 0); j_cancel is 2h later.
    assert by_id.loc["j_ok", "submit_time"] == 0
    assert by_id.loc["j_cancel", "submit_time"] == 2 * 3600 * 1_000_000
    assert str(df.attrs["trace_epoch"]) == "2020-05-01 00:00:00"

    # No gpu-number file -> empty pool dict.
    assert pools == {}


def test_convert_helios_roundtrips_through_load_trace(tmp_path):
    """The emitted frame is real canonical schema: writing it and reading it
    back with load_trace yields the same terminal jobs."""
    from fleetsim.workload.trace import write_trace

    log = tmp_path / "cluster_log.csv"
    _mapping_log(log)
    df, _ = convert_helios(log)
    out = tmp_path / "canon.csv"
    write_trace(out, df.to_dict("records"))
    jobs = load_trace(out)
    assert {j.id for j in jobs} == {"j_ok", "j_cancel", "j_fail", "j_to", "j_nf"}
    # walltime_est_s carries the (capped) duration -> SJF-oracle input.
    to = next(j for j in jobs if j.id == "j_to")
    assert to.walltime_est_s == 1_209_600.0
    assert to.true_duration_s == 1_209_600.0


# ---------------------------------------------------------------------------
# Per-VC pool sizes
# ---------------------------------------------------------------------------


def test_vc_pool_sizes_from_snapshot_row(tmp_path):
    log = tmp_path / "cluster_log.csv"
    _mapping_log(log)
    gpu = tmp_path / "cluster_gpu_number.csv"
    _write_gpu_number(
        gpu,
        {
            # A non-snapshot date that must be ignored...
            "2020-08-01": {"vcA": 8, "vcB": 8, "vcC": 8, "vcD": 8},
            # ...and the Sept-1 snapshot the converter reads.
            "2020-09-01": {"vcA": 16, "vcB": 0, "vcC": 24, "vcD": 20},
        },
        ["vcA", "vcB", "vcC", "vcD"],
    )
    _, pools = convert_helios(log, gpu)
    # Nodes = ceil(GPUs / 8): vcA 16->2, vcC 24->3, vcD 20->3 (ceil). vcB=0
    # is inactive and excluded.
    assert pools == {"vcA": 2, "vcC": 3, "vcD": 3}


def test_vc_pool_sizes_missing_snapshot_date_raises(tmp_path):
    log = tmp_path / "cluster_log.csv"
    _mapping_log(log)
    gpu = tmp_path / "cluster_gpu_number.csv"
    _write_gpu_number(gpu, {"2020-08-01": {"vcA": 8}}, ["vcA"])
    with pytest.raises(ValueError, match="snapshot date"):
        convert_helios(log, gpu)


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------


def test_convert_helios_missing_column_raises(tmp_path):
    log = tmp_path / "bad.csv"
    with log.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "vc", "gpu_num"])  # missing state/submit/duration/...
        w.writerow(["j1", "vcA", 8])
    with pytest.raises(ValueError, match="missing column"):
        convert_helios(log)


# ---------------------------------------------------------------------------
# month_window boundaries + re-basing
# ---------------------------------------------------------------------------


def _boundary_log(path):
    rows = [
        # April anchor so the trace epoch is well before September.
        _log_row("j_apr", "vcA", 8, "COMPLETED", "2020-04-10 12:00:00", 100),
        # Just BEFORE the September window -> excluded.
        _log_row("j_aug31", "vcA", 8, "COMPLETED", "2020-08-31 23:59:59", 100),
        # First instant of the window -> included, re-based to submit_time 0.
        _log_row("j_sep01", "vcA", 8, "COMPLETED", "2020-09-01 00:00:00", 100),
        # A mid-window job.
        _log_row("j_sep10", "vcB", 8, "FAILED", "2020-09-10 06:00:00", 100),
        # Last instant of the Helios Sept window (26th) -> included.
        _log_row("j_sep26", "vcB", 8, "CANCELLED", "2020-09-26 23:59:59", 100),
        # Just AFTER the window (27th) -> excluded.
        _log_row("j_sep27", "vcA", 8, "COMPLETED", "2020-09-27 00:00:00", 100),
    ]
    _write_log(path, rows)


def test_month_window_september_boundaries(tmp_path):
    log = tmp_path / "cluster_log.csv"
    _boundary_log(log)
    df, _ = convert_helios(log)
    sep = month_window(df, HELIOS_SEPT_MONTH, last_day=HELIOS_SEPT_LAST_DAY)

    # 08-31, 09-27, and the April anchor fall outside; the three in-window
    # jobs remain.
    assert set(sep["job_id"]) == {"j_sep01", "j_sep10", "j_sep26"}

    by_id = sep.set_index("job_id")
    # Re-based so 2020-09-01 00:00:00 is submit_time 0.
    assert by_id.loc["j_sep01", "submit_time"] == 0
    # 09-26 23:59:59 -> 25 days, 23:59:59 in microseconds.
    expected = ((25 * 86400) + 23 * 3600 + 59 * 60 + 59) * 1_000_000
    assert by_id.loc["j_sep26", "submit_time"] == expected
    assert str(sep.attrs["trace_epoch"]) == "2020-09-01 00:00:00"

    # Rows stay sorted by (submit_time, job_id) and non-negative.
    assert list(sep["submit_time"]) == sorted(sep["submit_time"])
    assert (sep["submit_time"] >= 0).all()


def test_month_window_defaults_to_calendar_month(tmp_path):
    """Without last_day the window is the whole calendar month; the 27th is
    then INSIDE September."""
    log = tmp_path / "cluster_log.csv"
    _boundary_log(log)
    df, _ = convert_helios(log)
    sep_full = month_window(df, "2020-09")  # calendar month (30 days)
    assert "j_sep27" in set(sep_full["job_id"])
    assert "j_sep01" in set(sep_full["job_id"])


def test_month_window_requires_epoch(tmp_path):
    """A frame with no trace epoch (and no explicit epoch=) cannot anchor
    absolute time."""
    df = pd.DataFrame({c: [] for c in CANONICAL_COLUMNS})
    df.attrs.pop("trace_epoch", None)
    with pytest.raises(ValueError, match="trace epoch"):
        month_window(df, "2020-09")
