"""Allocation-stint recording tests (``outputs: {stints: <level>}``,
the v0.3 visualizer's who-ran-where-when input).

Conventions: the hand-built fleet is one cluster ``c`` with two pods
``c/p0``/``c/p1`` of two 8-chip h100 nodes each; the collector horizon is
1000 s.  Synthetic sink-call sequences mirror the engine's chronological
emission order exactly (job_preempted / node_failed BEFORE the matching
job_requeued; job_finished terminal).  Every expected row is
hand-computed.
"""

import pandas as pd
import pytest

from fleetsim.api import run_scenario
from fleetsim.config import ScenarioError, load_scenario
from fleetsim.fleet.tree import FleetTree
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.metrics.summary import stints_dataframe
from fleetsim.model import (
    Allocation,
    Domain,
    GangAlloc,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    Tier,
)

S = 1_000_000  # one second in microseconds
H = 1000 * S  # hand-test horizon: 1000 s

END_REASONS = {
    "completed",
    "preempted",
    "failed",
    "drained",
    "canceled",
    "timeout",
    "running_at_horizon",
}


def pod_fleet():
    doms = [
        Domain(id="c", level="cluster", parent=None,
               children=["c/p0", "c/p1"], chip_type="h100"),
    ]
    for p in ("p0", "p1"):
        pid = f"c/{p}"
        doms.append(
            Domain(id=pid, level="pod", parent="c",
                   children=[f"{pid}/n0", f"{pid}/n1"], chip_type="h100")
        )
        for n in ("n0", "n1"):
            doms.append(
                Domain(id=f"{pid}/{n}", level="node", parent=pid,
                       children=[], chip_type="h100", chips=8)
            )
    return FleetTree(doms)


def mk_collector(stints="pod"):
    return MetricsCollector(H, fleet=pod_fleet(), stints=stints)


def mk_job(jid, chips=8, submit_s=0.0, klass=JobClass.EVAL,
           tier=Tier.BATCH, source_class="eval"):
    return Job(
        id=jid,
        tenant="t0",
        job_class=klass,
        submit_t=int(round(submit_s * S)),
        gangs=[GangSpec(chips=chips, chip_type="h100")],
        tier=tier,
        true_duration_s=100.0,
        source_class=source_class,
    )


# -- protocol-shaped helpers (chronological emission) -----------------------


def submit(c, job, t):
    c.job_submitted(job, t)
    c.job_admitted(job, t)


def start(c, job, t, nodes):
    """nodes: list of leaf ids (whole-node) or {leaf: chips} (sub-node)."""
    alloc = Allocation(job.id, [GangAlloc(nodes=nodes, anchor="c")])
    c.job_started(job, alloc, t)
    c.chips_allocated(job.gangs[0].chips, "h100", t)
    return alloc


def finish(c, job, t, status=JobStatus.COMPLETED):
    c.chips_freed(job.gangs[0].chips, "h100", t)
    c.job_finished(job, t, status, 0.0, 0.0)


def requeue(c, job, t):
    c.chips_freed(job.gangs[0].chips, "h100", t)
    c.job_requeued(job, t)


# ---------------------------------------------------------------------------
# Row correctness: sub-node, whole-node, segmented multi-pod
# ---------------------------------------------------------------------------


def test_sub_node_job_one_row_with_its_chip_share():
    c = mk_collector()
    j = mk_job("j1", chips=4)
    submit(c, j, 0)
    start(c, j, 10 * S, {"c/p0/n0": 4})
    finish(c, j, 50 * S)
    assert c.stint_rows() == [
        {
            "job_id": "j1",
            "class_name": "eval",
            "job_class": "EVAL",
            "tier": "BATCH",
            "domain": "c/p0",
            "chips": 4,
            "t0_us": 10 * S,
            "t1_us": 50 * S,
            "end_reason": "completed",
        }
    ]


def test_class_name_falls_back_to_job_class_for_untagged_jobs():
    c = mk_collector()
    j = mk_job("j1", chips=4, klass=JobClass.PRETRAIN, tier=Tier.PROD,
               source_class=None)
    submit(c, j, 0)
    start(c, j, 10 * S, {"c/p0/n0": 4})
    finish(c, j, 50 * S)
    (row,) = c.stint_rows()
    assert row["class_name"] == "PRETRAIN"
    assert row["job_class"] == "PRETRAIN"
    assert row["tier"] == "PROD"


def test_whole_node_job_one_row_full_leaf_chips():
    c = mk_collector()
    j = mk_job("j1", chips=16)
    submit(c, j, 0)
    start(c, j, 10 * S, ["c/p0/n0", "c/p0/n1"])
    finish(c, j, 90 * S)
    (row,) = c.stint_rows()
    assert (row["domain"], row["chips"]) == ("c/p0", 16)
    assert (row["t0_us"], row["t1_us"], row["end_reason"]) == (
        10 * S, 90 * S, "completed",
    )


def test_multi_pod_job_one_row_per_domain_shares_sum_to_chips():
    c = mk_collector()
    j = mk_job("big", chips=24)
    submit(c, j, 0)
    # Segmented across pods: 2 nodes in p0, 1 node in p1.
    start(c, j, 20 * S, ["c/p0/n0", "c/p0/n1", "c/p1/n0"])
    finish(c, j, 120 * S)
    rows = c.stint_rows()
    assert [(r["domain"], r["chips"]) for r in rows] == [
        ("c/p0", 16), ("c/p1", 8),
    ]
    assert sum(r["chips"] for r in rows) == 24
    assert {(r["t0_us"], r["t1_us"], r["end_reason"]) for r in rows} == {
        (20 * S, 120 * S, "completed")
    }


# ---------------------------------------------------------------------------
# Settlement reasons
# ---------------------------------------------------------------------------


def test_preempted_then_resumed_two_stints():
    c = mk_collector()
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 100 * S, ["c/p0/n0"])
    c.job_preempted(j, 200 * S, "scheduler")
    requeue(c, j, 260 * S)  # grace expiry releases the allocation
    start(c, j, 300 * S, ["c/p1/n1"])  # resume lands elsewhere
    finish(c, j, 400 * S)
    rows = c.stint_rows()
    assert [
        (r["domain"], r["t0_us"], r["t1_us"], r["end_reason"]) for r in rows
    ] == [
        ("c/p0", 100 * S, 260 * S, "preempted"),
        ("c/p1", 300 * S, 400 * S, "completed"),
    ]


def test_failure_kill_settles_failed():
    c = mk_collector()
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 100 * S, ["c/p0/n0"])
    c.node_failed("c/p0/n0", 150 * S, ["j1"], cause="gpu_hbm")
    requeue(c, j, 150 * S)
    (row,) = c.stint_rows()
    assert (row["t1_us"], row["end_reason"]) == (150 * S, "failed")


def test_failure_during_grace_overrides_pending_preemption():
    # Scheduler preempts; the node dies during the grace window: the
    # engine reports node_failed then requeues the victim -> "failed".
    c = mk_collector()
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 100 * S, ["c/p0/n0"])
    c.job_preempted(j, 200 * S, "scheduler")
    c.node_failed("c/p0/n0", 210 * S, ["j1"], cause="network")
    requeue(c, j, 210 * S)
    (row,) = c.stint_rows()
    assert row["end_reason"] == "failed"


def test_drain_kill_settles_drained():
    c = mk_collector()
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 100 * S, ["c/p0/n0"])
    c.job_preempted(j, 200 * S, "maintenance")
    requeue(c, j, 260 * S)
    (row,) = c.stint_rows()
    assert row["end_reason"] == "drained"


def test_cancel_preemption_settles_canceled():
    c = mk_collector()
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 100 * S, ["c/p0/n0"])
    c.job_preempted(j, 200 * S, "scheduler")
    finish(c, j, 200 * S, JobStatus.CANCELED)  # CANCEL mode: no requeue
    (row,) = c.stint_rows()
    assert (row["t1_us"], row["end_reason"]) == (200 * S, "canceled")


def test_timeout_and_node_fail_statuses_map():
    c = mk_collector()
    j1 = mk_job("j1", chips=8)
    submit(c, j1, 0)
    start(c, j1, 10 * S, ["c/p0/n0"])
    finish(c, j1, 500 * S, JobStatus.TIMEOUT)
    j2 = mk_job("j2", chips=8)
    submit(c, j2, 0)
    start(c, j2, 20 * S, ["c/p1/n0"])
    finish(c, j2, 400 * S, JobStatus.NODE_FAIL)  # trace-replayed terminal
    reasons = {r["job_id"]: r["end_reason"] for r in c.stint_rows()}
    assert reasons == {"j1": "timeout", "j2": "failed"}


# ---------------------------------------------------------------------------
# Horizon truncation (read-side, non-mutating)
# ---------------------------------------------------------------------------


def test_open_stint_truncates_at_horizon_without_mutation():
    c = mk_collector()
    j = mk_job("j1", chips=16)
    submit(c, j, 0)
    start(c, j, 500 * S, ["c/p0/n0", "c/p1/n1"])  # never settles
    rows = c.stint_rows()
    assert [(r["domain"], r["chips"], r["t1_us"], r["end_reason"]) for r in rows] == [
        ("c/p0", 8, H, "running_at_horizon"),
        ("c/p1", 8, H, "running_at_horizon"),
    ]
    assert c.stint_rows() == rows  # reads are idempotent
    # A real settlement after the read still wins over the horizon copy.
    finish(c, j, 900 * S)
    (r0, r1) = c.stint_rows()
    assert {r0["end_reason"], r1["end_reason"]} == {"completed"}
    assert {r0["t1_us"], r1["t1_us"]} == {900 * S}


def test_rows_sorted_by_t0_job_domain():
    c = mk_collector()
    jb = mk_job("b", chips=8)
    ja = mk_job("a", chips=8)
    for j in (jb, ja):
        submit(c, j, 0)
    start(c, jb, 10 * S, ["c/p0/n0"])
    start(c, ja, 10 * S, ["c/p1/n0"])
    finish(c, jb, 50 * S)
    finish(c, ja, 60 * S)
    assert [r["job_id"] for r in c.stint_rows()] == ["a", "b"]


# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------


def test_stints_true_means_level_below_cluster_root():
    c = mk_collector(stints=True)
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 10 * S, ["c/p0/n0"])
    finish(c, j, 50 * S)
    (row,) = c.stint_rows()
    assert row["domain"] == "c/p0"  # pods sit directly below root "c"


def test_disabled_by_default_records_nothing():
    c = MetricsCollector(H, fleet=pod_fleet())
    j = mk_job("j1", chips=8)
    submit(c, j, 0)
    start(c, j, 10 * S, ["c/p0/n0"])
    finish(c, j, 50 * S)
    assert c.stint_level is None
    assert c.stint_rows() == []
    df = stints_dataframe(c)
    assert len(df) == 0
    assert list(df.columns) == [
        "job_id", "class_name", "job_class", "tier", "domain",
        "chips", "t0_us", "t1_us", "end_reason",
    ]


def test_stints_require_fleet_at_construction():
    with pytest.raises(ValueError, match="requires the fleet"):
        MetricsCollector(H, stints="pod")


def test_unknown_level_raises_at_construction():
    with pytest.raises(ValueError, match="no ancestor at level 'rack'"):
        mk_collector(stints="rack")


def _doc(stints=None, levels=("pod", "node"), counts=(2, 2)):
    doc = {
        "sim": {"horizon": "10m", "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": list(levels), "counts": list(counts)},
                }
            ],
        },
        "failure_model": {
            "node_mtbf_days": 0.0,
            "maintenance_rate_per_node_month": 0.0,
        },
        "workload": {
            "kind": "synthetic",
            "classes": {
                "eval": {
                    "rate_per_hour": 60,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=2m, p90=5m]",
                    "abort_prob": 0,
                }
            },
        },
    }
    if stints is not None:
        doc["outputs"] = {"stints": stints}
    return doc


def test_config_parses_level_true_and_false():
    assert load_scenario(_doc()).outputs.stints is None
    assert load_scenario(_doc("pod")).outputs.stints == "pod"
    assert load_scenario(_doc(True)).outputs.stints is True
    assert load_scenario(_doc(False)).outputs.stints is None
    # The typed key never leaks into the extra passthrough dict.
    assert "stints" not in load_scenario(_doc("pod")).outputs.extra


def test_config_rejects_bad_type_and_unknown_level():
    with pytest.raises(ScenarioError, match="outputs.stints"):
        load_scenario(_doc(3))
    with pytest.raises(ScenarioError, match="outputs.stints"):
        load_scenario(_doc("rack"))  # cluster declares cluster/pod/node only


# ---------------------------------------------------------------------------
# End to end against the real engine + absent-key byte-compat
# ---------------------------------------------------------------------------


def test_end_to_end_engine_stints(tmp_path):
    run_scenario(_doc("pod"), out_dir=tmp_path)
    stints = pd.read_parquet(tmp_path / "stints.parquet")
    jobs = pd.read_parquet(tmp_path / "jobs.parquet")
    assert len(stints) > 0
    assert str(stints["chips"].dtype) == "int32"
    assert str(stints["t0_us"].dtype) == "int64"
    assert str(stints["t1_us"].dtype) == "int64"
    assert set(stints["end_reason"]) <= END_REASONS
    assert (stints["t0_us"] <= stints["t1_us"]).all()
    # Every stint's chip shares sum to the job's chips; the number of
    # stints per job equals its start count.
    jrows = jobs.set_index("job_id")
    per_stint = stints.groupby(["job_id", "t0_us", "t1_us"])["chips"].sum()
    for (jid, _, _), chips in per_stint.items():
        assert chips == jrows.loc[jid, "chips"]
    n_stints = per_stint.groupby("job_id").size()
    for jid, n in n_stints.items():
        assert n == jrows.loc[jid, "n_starts"]
    started = jobs[jobs["n_starts"] > 0]["job_id"]
    assert set(started) == set(stints["job_id"])
    # Traceability: stint domains are real pod ids of this fleet.
    assert set(stints["domain"]) <= {"m/c/pod0", "m/c/pod1"}


def test_absent_key_is_byte_compatible_and_writes_no_file(tmp_path):
    run_scenario(_doc(), out_dir=tmp_path / "off")
    run_scenario(_doc("pod"), out_dir=tmp_path / "on")
    assert not (tmp_path / "off" / "stints.parquet").exists()
    assert (tmp_path / "on" / "stints.parquet").exists()
    # Stint recording is observation-only: the v0.2 outputs are
    # byte-identical with and without the key.
    for name in ("jobs.parquet", "timeseries.parquet", "summary.json"):
        assert (tmp_path / "off" / name).read_bytes() == \
            (tmp_path / "on" / name).read_bytes(), name


def test_example_scenarios_declare_pod_stints():
    from pathlib import Path

    examples = Path(__file__).resolve().parent.parent / "examples"
    for name in ("01_minimal", "04_frontier"):
        scn = load_scenario(examples / name / "scenario.yaml")
        assert scn.outputs.stints == "pod", name
