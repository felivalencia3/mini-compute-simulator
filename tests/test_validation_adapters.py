"""Unit tests for the validation metric adapters (v0.6, checklist item 3).

Built against a hand-crafted ``jobs.parquet``-shaped DataFrame with known
values, so each adapter's definition (and its deliberate divergence from
``summary.json``) is pinned exactly.
"""

from __future__ import annotations

import math

import pandas as pd

from fleetsim.validation.adapters import (
    TERMINAL_STATUSES,
    gpu_time_by_status,
    jct_over_all_terminal,
    n_queuing_jobs,
)


def make_jobs_df() -> pd.DataFrame:
    """A small frame covering every terminal status plus a still-RUNNING
    job (non-terminal), with hand-chosen jct/queue/chip/elapsed values."""
    rows = [
        # job_id, status,      jct_s, queue_wait_s, chips, running_elapsed_s
        ("c0", "COMPLETED", 100.0, 30.0, 8, 70.0),
        ("c1", "COMPLETED", 300.0, 120.0, 8, 180.0),
        ("f0", "FAILED", 200.0, 90.0, 16, 110.0),
        ("k0", "CANCELED", 400.0, 400.0, 4, 0.0),  # killed before ever running
        ("t0", "TIMEOUT", 600.0, 61.0, 8, 539.0),
        ("n0", "NODE_FAIL", 50.0, 10.0, 32, 40.0),
        ("r0", "RUNNING", math.nan, 5.0, 8, 25.0),  # non-terminal, excluded
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "job_id",
            "status",
            "jct_s",
            "queue_wait_s",
            "chips",
            "running_elapsed_s",
        ],
    )
    df["status"] = df["status"].astype("string")
    df["jct_s"] = df["jct_s"].astype("float64")
    df["queue_wait_s"] = df["queue_wait_s"].astype("float64")
    df["chips"] = df["chips"].astype("int64")
    df["running_elapsed_s"] = df["running_elapsed_s"].astype("float64")
    return df


def test_jct_over_all_terminal_includes_non_completed():
    df = make_jobs_df()
    # Mean over ALL six terminal jobs (RUNNING excluded).
    expected = (100.0 + 300.0 + 200.0 + 400.0 + 600.0 + 50.0) / 6
    assert jct_over_all_terminal(df) == expected


def test_jct_diverges_from_completed_only_mean():
    """The whole point of the adapter (plan §1): it is NOT the
    COMPLETED-only mean that ``summary.json`` reports."""
    df = make_jobs_df()
    completed_only = (100.0 + 300.0) / 2  # 200.0
    all_terminal = jct_over_all_terminal(df)
    assert all_terminal != completed_only
    # And the non-completed terminals pull the mean up here.
    assert all_terminal > completed_only


def test_jct_over_all_terminal_empty_is_nan():
    df = make_jobs_df().iloc[0:0]
    assert math.isnan(jct_over_all_terminal(df))
    # Also nan when only non-terminal rows are present.
    running_only = make_jobs_df().query("status == 'RUNNING'")
    assert math.isnan(jct_over_all_terminal(running_only))


def test_n_queuing_jobs_strictly_exceeds_round():
    df = make_jobs_df()
    # queue_wait_s > 60: c1(120), f0(90), k0(400), t0(61). c0(30),
    # n0(10), r0(5) are <= 60 or below; 30/10/5 excluded, 70? none.
    assert n_queuing_jobs(df, round_s=60.0) == 4


def test_n_queuing_jobs_default_round_is_60():
    df = make_jobs_df()
    assert n_queuing_jobs(df) == n_queuing_jobs(df, round_s=60.0)


def test_n_queuing_jobs_custom_round():
    df = make_jobs_df()
    # Threshold at 100 s: only c1(120) and k0(400) exceed it.
    assert n_queuing_jobs(df, round_s=100.0) == 2


def test_n_queuing_jobs_boundary_excludes_equal():
    df = pd.DataFrame(
        {
            "status": pd.array(["COMPLETED", "COMPLETED"], dtype="string"),
            "jct_s": [10.0, 10.0],
            "queue_wait_s": [60.0, 60.001],  # == round excluded; just over counts
            "chips": [1, 1],
            "running_elapsed_s": [1.0, 1.0],
        }
    )
    assert n_queuing_jobs(df, round_s=60.0) == 1


def test_gpu_time_by_status():
    df = make_jobs_df()
    got = gpu_time_by_status(df)
    # chips * running_elapsed_s, grouped by status.
    assert got == {
        "CANCELED": 4 * 0.0,
        "COMPLETED": 8 * 70.0 + 8 * 180.0,  # 2000.0
        "FAILED": 16 * 110.0,  # 1760.0
        "NODE_FAIL": 32 * 40.0,  # 1280.0
        "RUNNING": 8 * 25.0,  # 200.0 (helper reports every status present)
        "TIMEOUT": 8 * 539.0,  # 4312.0
    }
    # Deterministic key order (sorted status names).
    assert list(got) == sorted(got)


def test_gpu_time_by_status_empty():
    assert gpu_time_by_status(make_jobs_df().iloc[0:0]) == {}


def test_terminal_statuses_constant():
    assert TERMINAL_STATUSES == frozenset(
        {"COMPLETED", "FAILED", "CANCELED", "TIMEOUT", "NODE_FAIL"}
    )
