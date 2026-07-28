"""Metrics pipeline tests: collector integrals, both weightings, window
filtering, summary numbers, parquet round-trip, plots, and an end-to-end
run against the real engine.

All expected numbers are hand-computed.  Conventions: the hand-built
fleet is 2 nodes x 8 h100 chips under cluster root "c"; the collector
horizon is 1000 s with the default 0.1/0.1 steady-state fractions, so the
window is [100 s, 900 s].  Synthetic sink-call sequences are emitted in
chronological order, exactly as the engine would.
"""

import json

import pandas as pd
import pytest

from fleetsim.fleet.tree import FleetTree
from fleetsim.metrics.base import MetricsSink
from fleetsim.metrics.collector import MetricsCollector, TimeWeighted
from fleetsim.metrics.plots import _require_matplotlib, render_plots
from fleetsim.metrics.summary import (
    build_summary,
    format_summary_table,
    jobs_dataframe,
    timeseries_dataframe,
    write_outputs,
)
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
W0, W1 = 100 * S, 900 * S  # default 0.1/0.1 window


def tiny_fleet():
    return FleetTree(
        [
            Domain(id="c", level="cluster", parent=None,
                   children=["c/n0", "c/n1"], chip_type="h100"),
            Domain(id="c/n0", level="node", parent="c", children=[],
                   chip_type="h100", chips=8),
            Domain(id="c/n1", level="node", parent="c", children=[],
                   chip_type="h100", chips=8),
        ]
    )


def mk_job(jid, klass=JobClass.EVAL, tenant="t0", chips=8, submit_s=0.0,
           tier=Tier.BATCH, dur_s=100.0):
    return Job(
        id=jid,
        tenant=tenant,
        job_class=klass,
        submit_t=int(round(submit_s * S)),
        gangs=[GangSpec(chips=chips, chip_type="h100")],
        tier=tier,
        true_duration_s=dur_s,
    )


# -- protocol-shaped helpers (chronological emission) -----------------------


def submit(c, job, t):
    c.job_submitted(job, t)
    c.job_admitted(job, t)


def start(c, job, t):
    c.job_started(job, Allocation(job.id, []), t)
    c.chips_allocated(job.gangs[0].chips, "h100", t)


def finish(c, job, t, status=JobStatus.COMPLETED, productive=0.0, lost=0.0):
    # The engine reports settled work via job_progress (point credit at
    # the terminal time keeps these hand-computed scenarios simple), then
    # the terminal totals via job_finished.
    if productive or lost:
        c.job_progress(job, t, t, productive, lost)
    c.chips_freed(job.gangs[0].chips, "h100", t)
    c.job_finished(job, t, status, productive, lost)


def requeue(c, job, t):
    c.chips_freed(job.gangs[0].chips, "h100", t)
    c.job_requeued(job, t)


def row_of(collector, jid):
    return next(r for r in collector.job_rows() if r["job_id"] == jid)


# ---------------------------------------------------------------------------
# TimeWeighted
# ---------------------------------------------------------------------------


def test_time_weighted_full_and_window_integrals():
    tw = TimeWeighted((W0, W1))
    tw.add(0, 8)
    tw.add(400 * S, -8)
    assert tw.integral(H) == 8 * 400 * S          # chip-us, exact ints
    assert tw.window_integral(H) == 8 * 300 * S   # clipped to [100, 400]
    # Level entirely outside the window contributes nothing to it.
    tw2 = TimeWeighted((W0, W1))
    tw2.add(950 * S, 4)
    assert tw2.integral(H) == 4 * 50 * S
    assert tw2.window_integral(H) == 0


def test_time_weighted_retro_add_back_corrects_since_epoch():
    tw = TimeWeighted((W0, W1))
    tw.add(50 * S, -8)     # delta arrives before the base level is known
    tw.advance(60 * S)
    tw.retro_add(16)       # learn the t=0 level was 16
    assert tw.value == 8
    assert tw.integral(60 * S) == 16 * 50 * S + 8 * 10 * S


# ---------------------------------------------------------------------------
# Collector: integrals, occupancy, window
# ---------------------------------------------------------------------------


def _summary_collector():
    """3 jobs against a 16-chip fleet; every number hand-computed.

    a: EVAL/t0, 1 chip,  starts 150 s, COMPLETED 450 s, productive 300.
    b: EVAL/t1, 8 chips, starts 200 s, COMPLETED 800 s, productive 4800.
    x: FINETUNE/t0, 2 chips, starts 10 s, CANCELED 50 s, productive 0,
       lost 80 -> entirely outside the [100, 900] window.
    """
    fleet = tiny_fleet()
    c = MetricsCollector(H, fleet=fleet)
    a = mk_job("a", chips=1, tenant="t0")
    b = mk_job("b", chips=8, tenant="t1")
    x = mk_job("x", klass=JobClass.FINETUNE, chips=2, tenant="t0")
    for j in (a, b, x):
        submit(c, j, 0)
    start(c, x, 10 * S)
    finish(c, x, 50 * S, JobStatus.CANCELED, productive=0.0, lost=80.0)
    start(c, a, 150 * S)
    start(c, b, 200 * S)
    finish(c, a, 450 * S, productive=300.0)
    finish(c, b, 800 * S, productive=4800.0)
    return c, fleet


def test_collector_is_a_metrics_sink():
    c, _ = _summary_collector()
    assert isinstance(c, MetricsSink)


def test_allocated_and_healthy_integrals():
    c, _ = _summary_collector()
    ints = c.integral_report()
    # full: 1x300 + 8x600 + 2x40 = 5180 chip-s; healthy 16 x 1000 s.
    assert ints["full"]["allocated_chip_s"] == pytest.approx(5180.0)
    assert ints["full"]["healthy_chip_s"] == pytest.approx(16000.0)
    assert ints["full"]["total_chip_s"] == pytest.approx(16000.0)
    assert ints["full"]["allocated_chip_s_by_type"] == {
        "h100": pytest.approx(5180.0)
    }
    assert ints["full"]["allocated_chip_s_by_class"] == {
        "EVAL": pytest.approx(5100.0),
        "FINETUNE": pytest.approx(80.0),
    }
    assert ints["full"]["allocated_chip_s_by_tenant"] == {
        "t0": pytest.approx(380.0),
        "t1": pytest.approx(4800.0),
    }
    # pending: a queued 0-150 (150 job-s), b 0-200, x 0-10.
    assert ints["full"]["pending_job_s_by_class"] == {
        "EVAL": pytest.approx(350.0),
        "FINETUNE": pytest.approx(10.0),
    }
    # window: x entirely outside; a/b stints fully inside.
    assert ints["window"]["allocated_chip_s"] == pytest.approx(5100.0)
    assert ints["window"]["healthy_chip_s"] == pytest.approx(12800.0)
    assert ints["window"]["pending_job_s_by_class"] == {
        "EVAL": pytest.approx(150.0),  # a: [100,150] = 50, b: [100,200] = 100
        "FINETUNE": pytest.approx(0.0),
    }


def test_summary_ratios_full_and_window():
    c, _ = _summary_collector()
    summary = build_summary(c)
    full, win = summary["full"], summary["window"]
    assert full["occupancy"] == pytest.approx(5180 / 16000)
    assert full["allocation_rate"] == pytest.approx(5180 / 16000)
    assert full["goodput"] == pytest.approx(5100 / 5180)
    assert win["occupancy"] == pytest.approx(5100 / 12800)
    assert win["goodput"] == pytest.approx(1.0)  # a+b end in-window
    assert full["duration_s"] == pytest.approx(1000.0)
    assert win["duration_s"] == pytest.approx(800.0)
    assert summary["steady_state_window"] == {
        "warmup_frac": 0.1, "drain_frac": 0.1, "start_us": W0, "end_us": W1,
    }


def test_queue_wait_both_weightings_hand_computed():
    c, _ = _summary_collector()
    full = build_summary(c)["full"]
    ev = full["queue_wait_s"]["EVAL"]
    # job-weighted over [150, 200]: p50 = 150 (inverted CDF), mean 175.
    assert ev["job_weighted"]["n"] == 2
    assert ev["job_weighted"]["p50"] == pytest.approx(150.0)
    assert ev["job_weighted"]["p99"] == pytest.approx(200.0)
    assert ev["job_weighted"]["mean"] == pytest.approx(175.0)
    # chip-hour weights: a = 1x300/3600, b = 8x600/3600 -> p50 lands on b.
    assert ev["chip_hour_weighted"]["p50"] == pytest.approx(200.0)
    assert ev["chip_hour_weighted"]["mean"] == pytest.approx(3350 / 17)
    # FINETUNE started at 10 s: present in full, absent from the window.
    assert full["queue_wait_s"]["FINETUNE"]["job_weighted"]["p50"] == \
        pytest.approx(10.0)
    win = build_summary(c)["window"]
    assert "FINETUNE" not in win["queue_wait_s"]


def test_jct_completed_only_both_weightings():
    c, _ = _summary_collector()
    full = build_summary(c)["full"]
    # x is CANCELED -> no FINETUNE key in JCT at all.
    assert set(full["jct_s"]) == {"EVAL"}
    ev = full["jct_s"]["EVAL"]
    assert ev["job_weighted"]["p50"] == pytest.approx(450.0)
    assert ev["chip_hour_weighted"]["p50"] == pytest.approx(800.0)


def test_ettr_and_status_counts():
    c, _ = _summary_collector()
    full = build_summary(c)["full"]
    # ettr: a = 300/300 = 1, b = 4800/8/600 = 1, x = 0/2/40 = 0.
    jw = full["ettr"]["job_weighted"]
    assert jw["n"] == 3
    assert jw["mean"] == pytest.approx(2 / 3)
    assert jw["p50"] == pytest.approx(1.0)
    assert full["counts"]["jobs_finished"] == 3
    assert full["counts"]["jobs_by_status"] == {"CANCELED": 1, "COMPLETED": 2}
    win = build_summary(c)["window"]
    assert win["counts"]["jobs_by_status"] == {"COMPLETED": 2}


def test_per_tenant_and_pending_depth():
    c, _ = _summary_collector()
    summary = build_summary(c)
    full_t = summary["full"]["per_tenant"]
    assert full_t["t0"]["chip_hours"] == pytest.approx(380 / 3600)
    assert full_t["t0"]["median_queue_wait_s"] == pytest.approx(10.0)
    assert full_t["t0"]["n_jobs_submitted"] == 2
    assert full_t["t1"]["chip_hours"] == pytest.approx(4800 / 3600)
    assert full_t["t1"]["median_queue_wait_s"] == pytest.approx(200.0)
    win_t = summary["window"]["per_tenant"]
    assert win_t["t0"]["chip_hours"] == pytest.approx(300 / 3600)
    assert win_t["t0"]["median_queue_wait_s"] == pytest.approx(150.0)
    assert win_t["t0"]["n_jobs_submitted"] == 0  # submitted at t=0
    assert summary["full"]["mean_pending_by_class"]["EVAL"] == \
        pytest.approx(0.35)
    assert summary["window"]["mean_pending_by_class"]["EVAL"] == \
        pytest.approx(150 / 800)
    assert summary["full"]["replica_availability"] is None


# ---------------------------------------------------------------------------
# Preemption counters, restarts, ETTR across stints
# ---------------------------------------------------------------------------


def test_preemption_counters_restarts_and_ettr():
    fleet = tiny_fleet()
    c = MetricsCollector(H, fleet=fleet)
    p = mk_job("p", klass=JobClass.FINETUNE, chips=4, tenant="tz")
    submit(c, p, 0)
    start(c, p, 0)
    c.job_preempted(p, 100 * S, "scheduler")
    requeue(c, p, 110 * S)  # 10 s grace held the chips
    start(c, p, 200 * S)
    c.job_preempted(p, 300 * S, "maintenance")
    requeue(c, p, 310 * S)
    start(c, p, 400 * S)
    finish(c, p, 500 * S, productive=1200.0, lost=80.0)

    row = row_of(c, "p")
    assert row["n_starts"] == 3
    assert row["n_restarts"] == 2
    assert row["n_preemptions"] == 2
    assert row["n_preempt_scheduler"] == 1
    assert row["n_preempt_maintenance"] == 1
    assert row["running_elapsed_s"] == pytest.approx(320.0)  # 110+110+100
    assert row["ettr"] == pytest.approx((1200 / 4) / 320)
    assert row["queue_wait_s"] == pytest.approx(0.0)
    assert row["jct_s"] == pytest.approx(500.0)
    assert c.preempt_triggers() == ("maintenance", "scheduler")

    ints = c.integral_report()["full"]
    assert ints["allocated_chip_s"] == pytest.approx(1280.0)  # 4 x 320
    assert ints["allocated_chip_s_by_class"]["FINETUNE"] == \
        pytest.approx(1280.0)
    assert ints["pending_job_s_by_class"]["FINETUNE"] == \
        pytest.approx(180.0)  # requeued 110-200 and 310-400

    summary = build_summary(c)
    assert summary["full"]["preemptions_per_min"] == {
        "maintenance": pytest.approx(1 / (1000 / 60)),
        "scheduler": pytest.approx(1 / (1000 / 60)),
        "node_failure": 0.0,
        "total": pytest.approx(2 / (1000 / 60)),
    }
    assert summary["window"]["preemptions_per_min"]["total"] == \
        pytest.approx(2 / (800 / 60))


# ---------------------------------------------------------------------------
# Node failures, repairs, healthy integral
# ---------------------------------------------------------------------------


def test_node_failure_healthy_integral_and_kill_counters():
    fleet = tiny_fleet()
    c = MetricsCollector(H, fleet=fleet)
    a = mk_job("a", chips=8)
    submit(c, a, 0)
    start(c, a, 0)
    c.node_failed("c/n0", 200 * S, ("a",))
    c.healthy_delta(-8, "h100", 200 * S)
    requeue(c, a, 200 * S)
    c.node_repaired("c/n0", 500 * S)
    c.healthy_delta(8, "h100", 500 * S)
    start(c, a, 520 * S)
    finish(c, a, 800 * S, productive=6400.0, lost=1600.0)

    assert row_of(c, "a")["n_node_failures"] == 1
    counts = c.event_counts()
    assert counts["full"]["node_failures"] == 1
    assert counts["full"]["node_repairs"] == 1
    assert counts["full"]["failure_kills"] == 1
    assert counts["window"]["failure_kills"] == 1  # t=200 in window
    ints = c.integral_report()
    # healthy: 16 chips minus 8 over [200, 500].
    assert ints["full"]["healthy_chip_s"] == pytest.approx(16000.0 - 2400.0)
    assert ints["window"]["healthy_chip_s"] == pytest.approx(12800.0 - 2400.0)
    summary = build_summary(c)
    assert summary["window"]["preemptions_per_min"]["node_failure"] == \
        pytest.approx(1 / (800 / 60))
    assert summary["full"]["counts"]["node_failures"] == 1


def test_fleet_statics_learned_at_first_flush_back_corrects_healthy():
    fleet = tiny_fleet()
    c = MetricsCollector(H)  # no fleet at construction
    c.healthy_delta(-8, "h100", 50 * S)
    fleet.fail_node("c/n0")  # make the live fleet agree at flush time
    c.flush(60 * S, fleet, 0, 0)
    ints = c.integral_report()["full"]
    # 16 chips over [0, 50], then 8 to the horizon.
    assert ints["healthy_chip_s"] == pytest.approx(16 * 50 + 8 * 950)
    assert ints["total_chip_s"] == pytest.approx(16000.0)


# ---------------------------------------------------------------------------
# Replica availability
# ---------------------------------------------------------------------------


def test_replica_availability():
    fleet = tiny_fleet()
    c = MetricsCollector(H, fleet=fleet)
    r = mk_job("r", klass=JobClass.INFER_REPLICA, chips=8, tenant="svc",
               tier=Tier.PROD)
    submit(c, r, 0)
    start(c, r, 100 * S)
    finish(c, r, 300 * S, productive=1600.0)
    ints = c.integral_report()
    assert ints["full"]["replica_desired_s"] == pytest.approx(300.0)
    assert ints["full"]["replica_running_s"] == pytest.approx(200.0)
    summary = build_summary(c)
    assert summary["full"]["replica_availability"] == pytest.approx(200 / 300)
    assert summary["window"]["replica_availability"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Flush sampling: timeseries + fragmentation
# ---------------------------------------------------------------------------


def test_flush_samples_fragmentation_and_ratios():
    fleet = tiny_fleet()
    c = MetricsCollector(H, fleet=fleet)
    a = mk_job("a", chips=4)
    submit(c, a, 0)
    start(c, a, 0)
    # Mirror the allocation on the real tree so frag queries see it.
    fleet.apply(Allocation("a", [GangAlloc(nodes={"c/n0": 4}, anchor="c")]))
    c.flush(60 * S, fleet, n_pending=2, n_running=1)

    (row,) = c.timeseries_rows()
    assert row["t_us"] == 60 * S
    assert row["allocated_chips"] == 4
    assert row["healthy_chips"] == 16
    assert row["pending_jobs"] == 2
    assert row["running_jobs"] == 1
    assert row["occupancy_to_date"] == pytest.approx(0.25)
    assert row["allocation_rate_to_date"] == pytest.approx(0.25)
    assert row["goodput_to_date"] == 0.0  # nothing finished yet
    # node level: free 4 (n0) + 8 (n1), largest 8 -> frag 1/3.  Stranded
    # uses the SMALLEST GANG quantum (DESIGN 9): the 4-chip gang could use
    # n0's 4 free chips, so nothing is stranded.
    assert row["largest_placeable_node"] == 8
    assert row["frag_index_node"] == pytest.approx(1 / 3)
    assert row["largest_placeable_cluster"] == 12
    assert row["frag_index_cluster"] == pytest.approx(0.0)
    assert row["stranded_chips"] == 0

    frag = c.frag_stats()
    assert frag["full"]["node"] == {
        "mean": pytest.approx(1 / 3), "max": pytest.approx(1 / 3),
        "n_samples": 1,
    }
    assert frag["window"] == {}  # t=60 s is before the window opens


# ---------------------------------------------------------------------------
# DataFrames, parquet round-trip, console table
# ---------------------------------------------------------------------------


def test_write_outputs_parquet_roundtrip(tmp_path):
    c, fleet = _summary_collector()
    c.flush(H, fleet, 0, 0)
    summary = write_outputs(c, tmp_path)

    jobs = pd.read_parquet(tmp_path / "jobs.parquet")
    assert list(jobs["job_id"]) == ["a", "b", "x"]  # (submit_t, id) order
    assert jobs.loc[0, "queue_wait_s"] == pytest.approx(150.0)
    assert jobs.loc[2, "status"] == "CANCELED"
    assert jobs.loc[2, "lost_chip_s"] == pytest.approx(80.0)
    assert int(jobs.loc[1, "end_t_us"]) == 800 * S
    assert jobs["chips"].dtype == "int64"

    ts = pd.read_parquet(tmp_path / "timeseries.parquet")
    assert len(ts) == 1
    assert int(ts.loc[0, "t_us"]) == H

    loaded = json.loads((tmp_path / "summary.json").read_text())
    assert loaded == summary

    table = format_summary_table(summary)
    assert "occupancy" in table and "EVAL" in table and "t1" in table


def test_empty_collector_dataframes_and_summary():
    c = MetricsCollector(H)
    jobs = jobs_dataframe(c)
    ts = timeseries_dataframe(c)
    assert len(jobs) == 0 and "job_id" in jobs.columns
    assert len(ts) == 0 and "t_us" in ts.columns
    summary = build_summary(c)
    assert summary["full"]["occupancy"] is None       # no healthy observed
    assert summary["full"]["allocation_rate"] is None  # fleet never seen
    assert summary["full"]["goodput"] is None
    assert summary["full"]["queue_wait_s"] == {}
    assert isinstance(format_summary_table(summary), str)


def test_unfinished_job_row_is_clamped_at_horizon():
    fleet = tiny_fleet()
    c = MetricsCollector(H, fleet=fleet)
    a = mk_job("a", chips=8)
    submit(c, a, 0)
    start(c, a, 400 * S)
    row = row_of(c, "a")
    assert row["status"] == "RUNNING"
    assert row["end_t_us"] is None and row["jct_s"] is None
    assert row["ettr"] is None
    assert row["running_elapsed_s"] == pytest.approx(600.0)  # to horizon


def test_constructor_validation():
    with pytest.raises(ValueError, match="horizon"):
        MetricsCollector(0)
    with pytest.raises(ValueError, match="fractions"):
        MetricsCollector(H, warmup_frac=0.6, drain_frac=0.5)
    with pytest.raises(ValueError, match="fractions"):
        MetricsCollector(H, warmup_frac=-0.1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def test_render_plots_writes_four_pngs(tmp_path):
    c, fleet = _summary_collector()
    c.flush(500 * S, fleet, 1, 1)
    c.flush(H, fleet, 0, 0)
    write_outputs(c, tmp_path)
    paths = render_plots(tmp_path)
    assert [p.name for p in paths] == [
        "jct_cdf.png", "queue_wait_cdf.png",
        "occupancy_timeline.png", "goodput_timeline.png",
    ]
    for p in paths:
        assert p.parent == tmp_path / "plots"
        assert p.is_file() and p.stat().st_size > 0


def test_plots_missing_matplotlib_raises_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="matplotlib is required"):
        _require_matplotlib()


# ---------------------------------------------------------------------------
# End-to-end against the real engine
# ---------------------------------------------------------------------------


def _engine_run():
    from fleetsim.config import load_scenario
    from fleetsim.engine.sim import Simulator
    from fleetsim.fleet.build import build_fleet
    from fleetsim.schedulers.fifo import FIFOScheduler
    from fleetsim.workload.base import ListSource

    scn = load_scenario(
        {
            "sim": {"horizon": "10m", "round": "60s", "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["node"], "counts": [2]},
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
                        "rate_per_hour": 1,
                        "chips": "pow2[1, 8]",
                        "duration": "lognormal[median=2m, p90=30m]",
                    }
                },
            },
        }
    )
    fleet = build_fleet(scn)
    collector = MetricsCollector.from_scenario(scn, fleet)
    jobs = [
        mk_job("a", chips=8, submit_s=0, dur_s=120.0),
        mk_job("b", chips=8, submit_s=0, dur_s=60.0),
    ]
    for j in jobs:  # disable checkpointing: work-seconds == wall-seconds
        j.checkpoint_interval_s = 0.0
    sim = Simulator(scn, fleet, ListSource(jobs), FIFOScheduler(), collector)
    sim.run()
    return collector


def test_end_to_end_engine_run(tmp_path):
    collector = _engine_run()
    summary = write_outputs(collector, tmp_path)
    # Both jobs run [0,120] and [0,60] on 16 healthy chips over 600 s.
    assert summary["full"]["occupancy"] == pytest.approx(1440 / 9600)
    assert summary["full"]["goodput"] == pytest.approx(1.0)
    assert summary["full"]["counts"]["jobs_by_status"] == {"COMPLETED": 2}
    jobs = pd.read_parquet(tmp_path / "jobs.parquet")
    assert set(jobs["status"]) == {"COMPLETED"}
    assert jobs["queue_wait_s"].max() == pytest.approx(0.0)
    ts = pd.read_parquet(tmp_path / "timeseries.parquet")
    # Flushes at 60..540 s (9 rounds) plus exactly one at the horizon.
    assert list(ts["t_us"]) == [60 * k * S for k in range(1, 11)]
    assert ts.loc[9, "occupancy_to_date"] == pytest.approx(1440 / 9600)
    paths = render_plots(tmp_path)
    assert len(paths) == 4


def test_end_to_end_deterministic_summary(tmp_path):
    s1 = write_outputs(_engine_run(), tmp_path / "r1")
    s2 = write_outputs(_engine_run(), tmp_path / "r2")
    assert s1 == s2
    assert (tmp_path / "r1" / "jobs.parquet").read_bytes() == \
        (tmp_path / "r2" / "jobs.parquet").read_bytes()


def test_from_scenario_reads_window_fracs_from_outputs_extra():
    from fleetsim.config import load_scenario

    scn = load_scenario(
        {
            "sim": {"horizon": "1000s", "round": "60s"},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["node"], "counts": [1]},
                    }
                ],
            },
            "workload": {
                "kind": "synthetic",
                "classes": {
                    "eval": {
                        "rate_per_hour": 1,
                        "chips": "pow2[1, 8]",
                        "duration": "lognormal[median=2m, p90=30m]",
                    }
                },
            },
            "outputs": {"warmup_frac": 0.2, "drain_frac": 0.05},
        }
    )
    c = MetricsCollector.from_scenario(scn)
    assert c.warmup_frac == 0.2
    assert c.drain_frac == 0.05
    assert c.window == (200 * S, 950 * S)
