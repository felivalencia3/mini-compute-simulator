"""Determinism contract (DESIGN §6.2): a run is a pure function of
(scenario, seed).

- Same scenario + same seed, run twice through the full ``run_scenario``
  pipeline -> identical ``jobs.parquet`` / ``timeseries.parquet``
  contents and identical ``summary.json`` bytes.
- Different seed -> a different arrival sequence (and different job
  outcomes), proving the seed actually feeds every stream.

Failures, maintenance, and preemption are all ON so the failure/repair/
maintenance RNG streams participate in the check.
"""

import json

import pandas as pd
import pandas.testing as pdt

from fleetsim import run_scenario

SCENARIO = {
    "sim": {"horizon": "6h", "round": "60s", "seed": 42},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "h100", "per_node": 8},
                "topology": {"levels": ["rack", "node"], "counts": [2, 4]},
            }
        ],
    },
    "failure_model": {
        "node_mtbf_days": 1.0,  # hot, so failure streams definitely draw
        "repair_auto_min": [10, 20],
        "repair_manual_frac": 0.1,
        "repair_manual_days": [0.05, 0.1],
        "maintenance_rate_per_node_month": 5.0,
        "drain_grace": "5m",
    },
    "workload": {
        "kind": "synthetic",
        "classes": {
            "eval": {
                "rate_per_hour": 40,
                "chips": "pow2[1, 8]",
                "duration": "lognormal[median=2m, p90=30m]",
                "tier": "batch",
                "abort_prob": 0.2,
                "diurnal": True,
            },
            "pretrain": {
                "rate_per_day": 8,
                "chips": "pow2[16, 32]",
                "duration": "lognormal[median=30m, p90=2h]",
                "tier": "prod",
                "checkpoint_interval": "15m",
                "within": "rack",
            },
        },
    },
    "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
}


def test_same_seed_twice_is_byte_identical(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_scenario(SCENARIO, out_dir=out_a)
    run_scenario(SCENARIO, out_dir=out_b)

    jobs_a = pd.read_parquet(out_a / "jobs.parquet")
    jobs_b = pd.read_parquet(out_b / "jobs.parquet")
    pdt.assert_frame_equal(jobs_a, jobs_b)
    assert len(jobs_a) > 100, "scenario produced too few jobs to be meaningful"

    ts_a = pd.read_parquet(out_a / "timeseries.parquet")
    ts_b = pd.read_parquet(out_b / "timeseries.parquet")
    pdt.assert_frame_equal(ts_a, ts_b)

    assert (out_a / "summary.json").read_bytes() == (
        out_b / "summary.json"
    ).read_bytes()


def test_different_seed_changes_arrivals(tmp_path):
    out_a = tmp_path / "a"
    out_c = tmp_path / "c"
    run_scenario(SCENARIO, out_dir=out_a)
    run_scenario(SCENARIO, out_dir=out_c, seed_override=43)

    jobs_a = pd.read_parquet(out_a / "jobs.parquet")
    jobs_c = pd.read_parquet(out_c / "jobs.parquet")
    arrivals_a = list(zip(jobs_a["job_id"], jobs_a["submit_t_us"]))
    arrivals_c = list(zip(jobs_c["job_id"], jobs_c["submit_t_us"]))
    assert arrivals_a != arrivals_c, "changing the seed did not change arrivals"

    # seed_override is equivalent to editing the document.
    summary_c = json.loads((out_c / "summary.json").read_text())
    out_d = tmp_path / "d"
    summary_d = run_scenario(
        SCENARIO, out_dir=out_d, overrides={"sim.seed": "43"}
    )
    assert summary_c == summary_d
    assert (out_c / "summary.json").read_bytes() == (
        out_d / "summary.json"
    ).read_bytes()
