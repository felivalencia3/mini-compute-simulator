"""Regression tests for the adversarial-review fixes.

Each test pins one reviewer finding: trace chip quantization, Place-vs-
GangSpec validation, the anonymous-Preempt PROD guardrail, attained
service vs goodput, sim.round wake wiring, event-triggered queue wakes,
drain-grace checkpoint banking, failure-cause sampling, second-order
preemption tagging, live-job goodput crediting, chip-bucketed queue
waits, mixed-leaf-size whole-node covers, lemon nodes, services wiring,
and the scheduler-registry error contract.
"""

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.events import EventType
from fleetsim.engine.sim import FAILURE_CAUSE_MIX, Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.fleet.tree import FleetTree, Placement
from fleetsim.metrics.base import NullSink
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.metrics.summary import build_summary
from fleetsim.model import (
    Constraint,
    Domain,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    PreemptMode,
    Tier,
)
from fleetsim.schedulers.base import Place, Preempt, Scheduler, get_scheduler
from fleetsim.schedulers.fifo import FIFOScheduler
from fleetsim.workload.base import ListSource
from fleetsim.workload.trace import TraceJob, TraceSource

S = 1_000_000

WORKLOAD = {
    "kind": "synthetic",
    "classes": {
        "eval": {
            "rate_per_hour": 1,
            "chips": "pow2[1, 8]",
            "duration": "lognormal[median=2m, p90=30m]",
        }
    },
}


def make_scenario(n_nodes=2, per_node=8, horizon="1h", round_="60s", seed=0,
                  fm=None):
    failure_model = {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0}
    failure_model.update(fm or {})
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": round_, "seed": seed},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": per_node},
                        "topology": {"levels": ["node"], "counts": [n_nodes]},
                    }
                ],
            },
            "failure_model": failure_model,
            "workload": WORKLOAD,
        }
    )


def mk_job(jid, chips=8, dur_s=100.0, submit_s=0.0, tier=Tier.BATCH,
           interval=0.0, save=0.0, restart=0.0, within=None,
           klass=JobClass.EVAL, valid_until_s=None):
    return Job(
        id=jid,
        tenant="t0",
        job_class=klass,
        submit_t=int(round(submit_s * 1e6)),
        gangs=[
            GangSpec(
                chips=chips,
                chip_type="h100",
                within=Constraint(level=within) if within else None,
            )
        ],
        tier=tier,
        true_duration_s=dur_s,
        checkpoint_interval_s=interval,
        checkpoint_save_s=save,
        restart_overhead_s=restart,
        valid_until=(
            int(round(valid_until_s * 1e6)) if valid_until_s is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Trace chip quantization (critical)
# ---------------------------------------------------------------------------


def _trace_job(jid, chips, dur=60.0, submit=0):
    return TraceJob(
        id=jid, tenant="t", job_class=JobClass.FINETUNE, submit_t=submit,
        gangs=[GangSpec(chips=chips, chip_type="h100")], tier=Tier.BATCH,
        true_duration_s=dur, checkpoint_interval_s=0.0,
    )


def test_trace_source_quantizes_multi_node_chips():
    scn = make_scenario(n_nodes=4)
    fleet = build_fleet(scn)
    jobs = [_trace_job("j12", 12), _trace_job("j8", 8), _trace_job("j3", 3)]
    src = TraceSource(jobs, fleet=fleet)
    by_id = {j.id: j.gangs[0].chips for j in src.jobs}
    assert by_id == {"j12": 16, "j8": 8, "j3": 3}  # sub-node verbatim
    # Caller-supplied jobs are never mutated.
    assert jobs[0].gangs[0].chips == 12


def test_trace_non_multiple_job_no_longer_starves_strict_fifo():
    scn = make_scenario(n_nodes=4, horizon="2h")
    fleet = build_fleet(scn)
    jobs = [
        _trace_job("big", 12, dur=300.0),
        _trace_job("e1", 1, submit=1),
    ]
    src = TraceSource(jobs, fleet=fleet)
    sim = Simulator(scn, fleet, src, FIFOScheduler(strict=True), NullSink())
    sim.run()
    assert all(j.status is JobStatus.COMPLETED for j in src.jobs)


def test_trace_without_fleet_is_verbatim():
    src = TraceSource([_trace_job("j12", 12)])
    assert src.jobs[0].gangs[0].chips == 12


def test_trace_pretrain_gets_min_runtime_guard(tmp_path):
    from fleetsim.workload.trace import CANONICAL_COLUMNS, load_trace

    p = tmp_path / "t.csv"
    p.write_text(
        ",".join(CANONICAL_COLUMNS) + "\n"
        + "p0,u,t0,pretrain,0,32,h100,4,7200,,COMPLETED\n"
        + "e0,u,t0,eval,0,1,h100,1,60,,COMPLETED\n"
    )
    by_id = {j.id: j for j in load_trace(p)}
    assert by_id["p0"].min_runtime_s == 7200.0  # DESIGN 14
    assert by_id["e0"].min_runtime_s == 0.0


# ---------------------------------------------------------------------------
# Place validation against the GangSpec (major)
# ---------------------------------------------------------------------------


class OnePlace(Scheduler):
    """Emits a single hand-built Place for the first pending job."""

    def __init__(self, placement):
        self.placement = placement
        self.fired = False

    def schedule(self, view):
        if self.fired:
            return []
        for jv in view.pending():
            self.fired = True
            return [Place(jv.id, self.placement)]
        return []


def test_place_with_wrong_chip_count_raises_strict():
    scn = make_scenario(n_nodes=2)
    fleet = build_fleet(scn)
    bad = Placement(leaves=(("m/c/node0", 1),), anchor="m/c",
                    chip_type="h100", whole_node=False)
    sim = Simulator(scn, fleet, ListSource([mk_job("a", chips=8)]),
                    OnePlace(bad), NullSink())
    with pytest.raises(ValueError, match="chips"):
        sim.run()


def test_place_violating_within_raises_strict():
    scn = make_scenario(n_nodes=2)
    fleet = build_fleet(scn)
    span = Placement(
        leaves=(("m/c/node0", 8), ("m/c/node1", 8)), anchor="m/c",
        chip_type="h100", whole_node=True,
    )
    job = mk_job("w", chips=16, within="node")
    sim = Simulator(scn, fleet, ListSource([job]), OnePlace(span), NullSink())
    with pytest.raises(ValueError, match="within"):
        sim.run()


def test_place_with_wrong_chip_type_raises_strict():
    scn = make_scenario(n_nodes=2)
    fleet = build_fleet(scn)
    bad = Placement(leaves=(("m/c/node0", 8),), anchor="m/c",
                    chip_type="b200", whole_node=True)
    sim = Simulator(scn, fleet, ListSource([mk_job("a", chips=8)]),
                    OnePlace(bad), NullSink())
    with pytest.raises(ValueError, match="chip type"):
        sim.run()


def test_bad_place_skipped_in_lenient_mode():
    scn = make_scenario(n_nodes=2)
    fleet = build_fleet(scn)
    bad = Placement(leaves=(("m/c/node0", 1),), anchor="m/c",
                    chip_type="h100", whole_node=False)
    sim = Simulator(scn, fleet, ListSource([mk_job("a", chips=8)]),
                    OnePlace(bad), NullSink(), strict=False)
    sim.run()  # no raise; the job simply never starts via the bad action
    fleet.check_invariants()


# ---------------------------------------------------------------------------
# Anonymous Preempt guardrail + second-order tagging (major)
# ---------------------------------------------------------------------------


class PreemptOnce(FIFOScheduler):
    def __init__(self, victim, at_us, preemptor=None, strict=True):
        super().__init__(strict=strict)
        self.victim, self.at_us, self.preemptor = victim, at_us, preemptor
        self.fired = False

    def schedule(self, view):
        if not self.fired and view.now >= self.at_us:
            self.fired = True
            return [Preempt(self.victim, PreemptMode.REQUEUE, self.preemptor)]
        return super().schedule(view)


def test_anonymous_preempt_cannot_target_prod():
    scn = make_scenario(n_nodes=1)
    fleet = build_fleet(scn)
    jobs = [mk_job("p", dur_s=600.0, tier=Tier.PROD)]
    sim = Simulator(scn, fleet, ListSource(jobs),
                    PreemptOnce("p", 60 * S), NullSink())
    with pytest.raises(ValueError, match="anonymous"):
        sim.run()


def test_anonymous_preempt_of_batch_still_allowed():
    scn = make_scenario(n_nodes=1)
    fleet = build_fleet(scn)

    class Rec(NullSink):
        def __init__(self):
            self.triggers = []

        def job_preempted(self, job, t, trigger):
            self.triggers.append((job.id, trigger))

    rec = Rec()
    jobs = [mk_job("b", dur_s=600.0, tier=Tier.BATCH)]
    Simulator(scn, fleet, ListSource(jobs), PreemptOnce("b", 60 * S), rec).run()
    assert rec.triggers == [("b", "scheduler")]


def test_second_order_preemption_tagged():
    # A failure-requeued PROD job evicting BATCH work is tagged
    # "failure_second_order", not "scheduler" (DESIGN 8/9).
    scn = make_scenario(n_nodes=2, horizon="3h")
    fleet = build_fleet(scn)

    class Rec(NullSink):
        def __init__(self):
            self.triggers = []

        def job_preempted(self, job, t, trigger):
            self.triggers.append((job.id, trigger))

    # p runs on both nodes; node0 fails at 100 s -> p requeues (n_failures=1).
    # Node repair takes exactly 60 min.  Meanwhile b grabs node1 (best-effort
    # FIFO lets it flow around the stuck p).  After the repair p still cannot
    # fit -> the scheduler preempts b naming p as preemptor.
    scn = make_scenario(n_nodes=2, horizon="3h",
                        fm={"repair_auto_min": [60, 60]})
    fleet = build_fleet(scn)
    p = mk_job("p", chips=16, dur_s=4000.0, tier=Tier.PROD, interval=100.0)
    b = mk_job("b", chips=8, dur_s=9000.0, submit_s=150.0, tier=Tier.BATCH)
    rec = Rec()
    sched = PreemptOnce("b", 4000 * S, preemptor="p", strict=False)
    sim = Simulator(scn, fleet, ListSource([p, b]), sched, rec)
    sim.queue.push(100 * S, EventType.NODE_FAILURE, "m/c/node0")
    sim.run()
    assert ("b", "failure_second_order") in rec.triggers


# ---------------------------------------------------------------------------
# Attained service vs goodput (major)
# ---------------------------------------------------------------------------


def test_attained_service_counts_lost_work():
    scn = make_scenario(n_nodes=2, horizon="4h")
    fleet = build_fleet(scn)
    j = mk_job("a", chips=16, dur_s=7200.0, interval=3600.0)
    sim = Simulator(scn, fleet, ListSource([j]), FIFOScheduler(), NullSink())
    # Fail at 5400 s: cum=5400, kept floors to 3600, lost=1800.
    sim.queue.push(5400 * S, EventType.NODE_FAILURE, "m/c/node0")
    sim.run()
    assert j.attained_service_chip_s - j.goodput_chip_s == pytest.approx(
        1800.0 * 16
    )
    assert j.goodput_chip_s >= 3600.0 * 16  # surviving work only


def test_view_attained_service_includes_lost_work():
    scn = make_scenario(n_nodes=2, horizon="4h")
    fleet = build_fleet(scn)

    seen = {}

    class Probe(FIFOScheduler):
        def schedule(self, view):
            for jv in view.pending():
                seen[view.now] = jv.attained_service_chip_s
            return super().schedule(view)

    j = mk_job("a", chips=16, dur_s=7200.0, interval=3600.0, restart=10_000.0)
    sim = Simulator(scn, fleet, ListSource([j]), Probe(), NullSink())
    sim.queue.push(5400 * S, EventType.NODE_FAIL if False else EventType.NODE_FAILURE,
                   "m/c/node0")
    sim.run()
    # While requeued (node down), the view reports kept+lost = 5400 work-s.
    assert any(v == pytest.approx(5400.0 * 16) for v in seen.values())


# ---------------------------------------------------------------------------
# sim.round wake wiring + event-triggered queue wakes (minor)
# ---------------------------------------------------------------------------


def test_sim_round_drives_default_scheduler_cadence():
    class Count(FIFOScheduler):
        def __init__(self):
            super().__init__()
            self.wakes = []

        def schedule(self, view):
            self.wakes.append(view.now)
            return super().schedule(view)

    scn = make_scenario(horizon="1h", round_="10m")
    fleet = build_fleet(scn)
    c = Count()
    Simulator(scn, fleet, ListSource([]), c, NullSink()).run()
    gaps = {b - a for a, b in zip(c.wakes, c.wakes[1:])}
    assert gaps == {600 * S}
    assert c.wake_interval == 600 * S  # rewired from the base default


def test_explicit_wake_interval_is_respected():
    class Fast(FIFOScheduler):
        wake_interval = 30 * S

    scn = make_scenario(horizon="10m", round_="60s")
    fleet = build_fleet(scn)
    s = Fast()
    Simulator(scn, fleet, ListSource([]), s, NullSink()).run()
    assert s.wake_interval == 30 * S  # not overwritten


def test_queue_expiry_wakes_event_triggered_scheduler():
    class EventFIFO(FIFOScheduler):
        wake_interval = None  # DESIGN 7: event-triggered only

    scn = make_scenario(n_nodes=1, horizon="1h")
    fleet = build_fleet(scn)
    head = mk_job("head", chips=16, dur_s=60.0, valid_until_s=1800.0)  # never fits
    tail = mk_job("tail", chips=1, dur_s=60.0, submit_s=1.0)
    sim = Simulator(scn, fleet, ListSource([head, tail]), EventFIFO(), NullSink())
    sim.run()
    assert head.status is JobStatus.FAILED  # valid_until expiry
    assert tail.status is JobStatus.COMPLETED  # expiry woke the scheduler


# ---------------------------------------------------------------------------
# Drain-grace checkpoint banking (minor)
# ---------------------------------------------------------------------------


def test_drain_grace_banks_full_progress():
    scn = make_scenario(n_nodes=2, horizon="4h", fm={"drain_grace": "10m"})
    fleet = build_fleet(scn)
    j = mk_job("a", chips=8, dur_s=7200.0, interval=3600.0, save=60.0)
    sim = Simulator(scn, fleet, ListSource([j]), FIFOScheduler(), NullSink())
    # Drain at 1860 s; grace ends 2460 s (mid-interval) -> banked, not floored.
    sim.queue.push(1860 * S, EventType.MAINTENANCE_DRAIN, "m/c/node0")
    sim.run()
    rt = sim._jobs["a"]
    assert rt.lost_work_s == 0.0  # the grace WAS the checkpoint window
    assert j.status is JobStatus.COMPLETED


def test_failure_during_drain_save_window_voids_the_bank():
    scn = make_scenario(n_nodes=1, horizon="4h", fm={"drain_grace": "10m"})
    fleet = build_fleet(scn)
    j = mk_job("a", chips=8, dur_s=7200.0, interval=3600.0, save=60.0)
    sim = Simulator(scn, fleet, ListSource([j]), FIFOScheduler(), NullSink())
    sim.queue.push(1860 * S, EventType.MAINTENANCE_DRAIN, "m/c/node0")
    # Node crashes 30 s into the 60 s save window: floor semantics apply.
    sim.queue.push(2490 * S, EventType.NODE_FAILURE, "m/c/node0")
    sim.run()
    rt = sim._jobs["a"]
    assert rt.kept_work_s == 0.0  # cum < one interval, nothing banked
    assert rt.lost_work_s > 0.0


# ---------------------------------------------------------------------------
# Failure causes (minor)
# ---------------------------------------------------------------------------


def test_failure_cause_sampled_and_counted():
    scn = make_scenario(n_nodes=2, horizon="1h")
    fleet = build_fleet(scn)
    collector = MetricsCollector(scn.sim.horizon_us, fleet=fleet)
    sim = Simulator(scn, fleet, ListSource([]), FIFOScheduler(), collector)
    for t in (100, 200):
        sim.queue.push(t * S, EventType.NODE_FAILURE, f"m/c/node{t // 100 - 1}")
    sim.run()
    causes = collector.event_counts()["full"]["node_failures_by_cause"]
    valid = {name for name, _ in FAILURE_CAUSE_MIX}
    assert sum(causes.values()) == 2
    assert set(causes) <= valid
    summary = build_summary(collector)
    assert summary["full"]["counts"]["node_failures_by_cause"] == causes


# ---------------------------------------------------------------------------
# Goodput credits live jobs; window goodput <= 1 (critical)
# ---------------------------------------------------------------------------


def test_goodput_credits_still_running_checkpointed_work():
    scn = make_scenario(n_nodes=2, horizon="4h")
    fleet = build_fleet(scn)
    collector = MetricsCollector(scn.sim.horizon_us, fleet=fleet)
    # 10-day job: never finishes in 4 h; banked = floor(14400/3300)*3300.
    j = mk_job("live", chips=16, dur_s=10 * 86400.0, interval=3300.0)
    Simulator(scn, fleet, ListSource([j]), FIFOScheduler(), collector).run()
    summary = build_summary(collector)
    assert summary["full"]["goodput"] == pytest.approx(13200 / 14400)
    for scope in ("full", "window"):
        g = summary[scope]["goodput"]
        assert g is None or 0.0 <= g <= 1.0 + 1e-12


def test_window_goodput_spreads_straddling_stints():
    # A stint straddling the window edge contributes only its overlap
    # share to the window numerator (the old end-attribution pushed
    # windowed goodput above 1).
    fleet = FleetTree(
        [
            Domain(id="c", level="cluster", parent=None, children=["c/n0"],
                   chip_type="h100"),
            Domain(id="c/n0", level="node", parent="c", children=[],
                   chip_type="h100", chips=8),
        ]
    )
    horizon = 1000 * S  # window [100 s, 900 s]
    c = MetricsCollector(horizon, fleet=fleet)
    j = mk_job("a", chips=8, dur_s=200.0)
    c.job_submitted(j, 0)
    c.job_admitted(j, 0)
    from fleetsim.model import Allocation

    c.job_started(j, Allocation("a", []), 0)
    c.chips_allocated(8, "h100", 0)
    # Stint [0, 200 s], surviving work 200 x 8; window overlap = 100/200.
    c.job_progress(j, 0, 200 * S, 1600.0, 0.0)
    c.chips_freed(8, "h100", 200 * S)
    c.job_finished(j, 200 * S, JobStatus.COMPLETED, 1600.0, 0.0)
    ints = c.integral_report()
    assert ints["full"]["productive_chip_s"] == pytest.approx(1600.0)
    assert ints["window"]["productive_chip_s"] == pytest.approx(800.0)
    win = build_summary(c)["window"]
    assert win["goodput"] == pytest.approx(1.0)  # 800 / (8 x 100 s)


# ---------------------------------------------------------------------------
# Chip-bucketed queue waits (major)
# ---------------------------------------------------------------------------


def test_queue_wait_bucketed_by_chip_count():
    fleet = FleetTree(
        [
            Domain(id="c", level="cluster", parent=None, children=["c/n0"],
                   chip_type="h100"),
            Domain(id="c/n0", level="node", parent="c", children=[],
                   chip_type="h100", chips=1024),
        ]
    )
    c = MetricsCollector(1000 * S, fleet=fleet)
    from fleetsim.model import Allocation

    for jid, chips, start_s in (("s", 4, 10.0), ("m", 32, 50.0), ("l", 256, 90.0)):
        j = mk_job(jid, chips=chips)
        c.job_submitted(j, 0)
        c.job_admitted(j, 0)
        c.job_started(j, Allocation(jid, []), int(start_s * 1e6))
        c.chips_allocated(chips, "h100", int(start_s * 1e6))
    block = build_summary(c)["full"]["queue_wait_s_by_chips"]
    assert set(block) == {"1-8", "9-64", "65-512"}
    assert block["1-8"]["job_weighted"]["p50"] == pytest.approx(10.0)
    assert block["9-64"]["job_weighted"]["p50"] == pytest.approx(50.0)
    assert block["65-512"]["job_weighted"]["p50"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Mixed leaf sizes: whole-node exact cover (minor, tree)
# ---------------------------------------------------------------------------


def test_whole_node_cover_with_mixed_leaf_sizes():
    # Free leaves [8, 16, 16]: a 32-chip gang must find 16+16 (the greedy
    # id-order scan used to grab 8+16 and give up).
    fleet = FleetTree(
        [
            Domain(id="c", level="cluster", parent=None,
                   children=["c/n0", "c/n1", "c/n2"], chip_type="h100"),
            Domain(id="c/n0", level="node", parent="c", children=[],
                   chip_type="h100", chips=8),
            Domain(id="c/n1", level="node", parent="c", children=[],
                   chip_type="h100", chips=16),
            Domain(id="c/n2", level="node", parent="c", children=[],
                   chip_type="h100", chips=16),
        ]
    )
    p = fleet.search_first_fit(GangSpec(chips=32, chip_type="h100"))
    assert p is not None
    assert p.leaves == (("c/n1", 16), ("c/n2", 16))
    assert p.whole_node


# ---------------------------------------------------------------------------
# Lemon nodes (minor, config + build)
# ---------------------------------------------------------------------------


def test_lemon_nodes_set_deterministically():
    doc = {
        "sim": {"horizon": "1d"},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [200]},
                }
            ],
        },
        "failure_model": {
            "node_mtbf_days": 42,
            "lemon_frac": 0.1,
            "lemon_multiplier": 10,
        },
        "workload": WORKLOAD,
    }
    scn = load_scenario(doc)
    f1, f2 = build_fleet(scn), build_fleet(scn)
    lemons1 = [l for l in f1.leaves() if f1.domain(l).lemon_factor == 10.0]
    lemons2 = [l for l in f2.leaves() if f2.domain(l).lemon_factor == 10.0]
    assert lemons1 == lemons2  # deterministic
    assert 5 <= len(lemons1) <= 40  # ~10% of 200, hash-selected


# ---------------------------------------------------------------------------
# Services reachable from YAML (minor, api)
# ---------------------------------------------------------------------------


def test_services_section_runs_end_to_end():
    from fleetsim import api

    doc = {
        "sim": {"horizon": "30m", "round": "60s", "seed": 1},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [4]},
                }
            ],
        },
        "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
        "workload": {
            "kind": "synthetic",
            "classes": {
                "eval": {
                    "rate_per_hour": 10,
                    "chips": "pow2[1, 4]",
                    "duration": "lognormal[median=2m, p90=10m]",
                }
            },
        },
        "services": [{"id": "chat", "tenant": "svc", "replicas": 2}],
    }
    summary = api.run_scenario(doc)
    # Replica availability is now derivable from a YAML run (was null).
    assert summary["full"]["replica_availability"] is not None
    assert summary["full"]["replica_availability"] > 0.9


# ---------------------------------------------------------------------------
# Scheduler registry error contract (critical, CLI)
# ---------------------------------------------------------------------------


def test_get_scheduler_bad_params_is_value_error():
    with pytest.raises(ValueError, match="rejected params"):
        get_scheduler("fifo", {"preempt": "requeue"})


def test_run_scenario_scheduler_param_mismatch_is_clean(tmp_path):
    from fleetsim import api
    from fleetsim.cli import main
    import yaml

    doc = {
        "sim": {"horizon": "10m", "round": "60s", "seed": 0},
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
        "workload": WORKLOAD,
        "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
    }
    # API contract: ValueError, not TypeError.
    with pytest.raises(ValueError, match="rejected params"):
        api.run_scenario(doc, overrides={"scheduler.name": "fifo"})
    # CLI contract: exit code 2, no traceback.
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    rc = main(["run", str(p), "-o", str(tmp_path / "o"),
               "--override", "scheduler.name=fifo"])
    assert rc == 2


# ---------------------------------------------------------------------------
# Compare-table fixes (minor, CLI)
# ---------------------------------------------------------------------------


def test_compare_disambiguates_duplicate_basenames(tmp_path, capsys):
    import json

    from fleetsim import api
    from fleetsim.cli import main

    doc = make_scenario(horizon="10m")
    del doc  # make_scenario returns a Scenario; build a raw doc instead
    raw = {
        "sim": {"horizon": "10m", "round": "60s", "seed": 0},
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
        "workload": WORKLOAD,
    }
    api.run_scenario(raw, out_dir=tmp_path / "exp_a" / "out")
    api.run_scenario(raw, out_dir=tmp_path / "exp_b" / "out")
    rc = main(["compare", str(tmp_path / "exp_a" / "out"),
               str(tmp_path / "exp_b" / "out")])
    assert rc == 0
    head = capsys.readouterr().out.splitlines()[0]
    assert "exp_a" in head and "exp_b" in head  # not two bare "out" columns
