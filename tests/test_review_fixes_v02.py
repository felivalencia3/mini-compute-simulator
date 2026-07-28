"""Regression tests for the v0.2 review fixes (semantics-scale lens).

Each test reproduces a reviewed failure scenario and asserts the fixed
behavior:

1. Reclaim shape-blindness: chip-count plans whose freed chips cannot
   form the needed whole nodes must emit NO evictions (no thrash loop).
2. Claims memory: a grace window longer than the scheduler round must
   not trigger a second, redundant victim set.
3. Leaf health: victims on DRAINING nodes free nothing and must not be
   evicted toward a claim they cannot serve.
4. Event-triggered scheduler (wake_interval=None) + closed-loop backlog
   must still generate work (a wake is seeded for refill sources).
5. failure_second_order tagging keys on the MOST RECENT requeue cause,
   not the lifetime failure count.
6. jobs.parquet carries source_class; per-class wait/JCT stats exclude
   BEST_EFFORT-tier jobs; per-job productive_chip_s accumulates for
   still-running jobs.

Conventions match test_tiered_priority: tiny fleets ("m/c/node0"...),
60 s rounds, failures/maintenance OFF unless the test injects them,
strict engine mode, hand-computed times (int µs; *_s float seconds).
"""

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.events import EventType
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.metrics.summary import build_summary, jobs_dataframe
from fleetsim.model import (
    Constraint,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    Tier,
)
from fleetsim.schedulers.base import Scheduler
from fleetsim.schedulers.tiered_priority import TieredPriorityScheduler
from fleetsim.workload.base import ListSource
from fleetsim.workload.synthetic import SyntheticSource

S = 1_000_000  # one second in microseconds
CLUSTER = "m/c"


def make_scenario(
    n_nodes=2, per_node=8, horizon="30m", round_="60s", levels=None,
    counts=None, fm=None,
):
    if levels is None:
        levels, counts = ["node"], [n_nodes]
    failure_model = {
        "node_mtbf_days": 0.0,
        "maintenance_rate_per_node_month": 0.0,
    }
    failure_model.update(fm or {})
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": round_, "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": per_node},
                        "topology": {"levels": levels, "counts": counts},
                    }
                ],
            },
            "failure_model": failure_model,
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


def mk_job(
    jid, submit_s=0.0, chips=8, dur_s=100.0, tier=Tier.BATCH, interval=0.0,
    save=0.0, restart=0.0, preemptible=True, min_runtime=0.0, within=None,
):
    return Job(
        id=jid,
        tenant="t0",
        job_class=JobClass.EVAL,
        submit_t=int(round(submit_s * 1e6)),
        gangs=[
            GangSpec(
                chips=chips,
                chip_type="h100",
                within=Constraint(level=within) if within else None,
            )
        ],
        tier=tier,
        preemptible=preemptible,
        min_runtime_s=min_runtime,
        true_duration_s=dur_s,
        checkpoint_interval_s=interval,
        checkpoint_save_s=save,
        restart_overhead_s=restart,
    )


class RecSink:
    def __init__(self):
        self.calls = []

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]

    def job_submitted(self, job, t):
        self.calls.append(("job_submitted", job.id, t))

    def job_admitted(self, job, t):
        self.calls.append(("job_admitted", job.id, t))

    def job_started(self, job, alloc, t):
        self.calls.append(("job_started", job.id, t))

    def job_preempted(self, job, t, trigger):
        self.calls.append(("job_preempted", job.id, t, trigger))

    def job_requeued(self, job, t):
        self.calls.append(("job_requeued", job.id, t))

    def job_progress(self, job, start_us, end_us, productive, lost):
        self.calls.append(("job_progress", job.id, start_us, end_us))

    def job_finished(self, job, t, status, productive, lost):
        self.calls.append(("job_finished", job.id, t, status.name))

    def node_failed(self, node_id, t, killed_alloc_ids, cause="unknown"):
        self.calls.append(("node_failed", node_id, t, tuple(killed_alloc_ids)))

    def node_repaired(self, node_id, t):
        self.calls.append(("node_repaired", node_id, t))

    def node_drain_started(self, node_id, t):
        self.calls.append(("node_drain_started", node_id, t))

    def chips_allocated(self, n, chip_type, t):
        pass

    def chips_freed(self, n, chip_type, t):
        pass

    def healthy_delta(self, n_chips, chip_type, t):
        pass

    def flush(self, t, fleet, n_pending, n_running):
        pass


def run_sim(scenario, jobs, scheduler=None, sink=None, pre_events=()):
    fleet = build_fleet(scenario)
    sink = sink if sink is not None else RecSink()
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        scheduler if scheduler is not None else TieredPriorityScheduler(),
        sink,
        strict=True,
    )
    for t, etype, payload in pre_events:
        sim.queue.push(t, etype, payload)
    sim.run()
    fleet.check_invariants()
    return sim, fleet, sink


def preempted(sink):
    return [(c[1], c[2], c[3]) for c in sink.of("job_preempted")]


# ---------------------------------------------------------------------------
# 1. Reclaim shape-blindness: no eviction that provably cannot help
# ---------------------------------------------------------------------------


def test_no_thrash_when_freed_chips_cannot_form_whole_nodes():
    # 3 nodes x 8.  node0: 1-chip non-preemptible pin + 7-chip BE mouse;
    # node1: same; node2 free.  A PROD 16-chip whole-node gang can NEVER
    # place (only node2 can ever be fully free), yet free(8)+victims(14)
    # >= 16 under chip counts.  The fixed planner dry-runs the plan,
    # sees the freed chips are shape-useless, and emits NOTHING —
    # previously this preempted the mice every other wake forever.
    scn = make_scenario(n_nodes=3, horizon="20m")
    jobs = [
        mk_job("pin0", submit_s=0, chips=1, dur_s=10_000.0, preemptible=False),
        mk_job("m0", submit_s=0, chips=7, dur_s=10_000.0,
               tier=Tier.BEST_EFFORT),
        mk_job("pin1", submit_s=61, chips=1, dur_s=10_000.0,
               preemptible=False),
        mk_job("m1", submit_s=61, chips=7, dur_s=10_000.0,
               tier=Tier.BEST_EFFORT),
        mk_job("p", submit_s=130, chips=16, dur_s=100.0, tier=Tier.PROD),
    ]
    sim, fleet, sink = run_sim(scn, jobs)
    assert preempted(sink) == []  # zero wasted preemptions, no storm
    assert sim._jobs["p"].job.status is JobStatus.ADMITTED  # honestly stuck
    # The mice were never disturbed.
    assert sim._jobs["m0"].job.status is JobStatus.RUNNING
    assert sim._jobs["m1"].job.status is JobStatus.RUNNING


def test_refine_prunes_useless_sub_node_victims():
    # node0: 1-chip pin (not evictable) + 7-chip mouse; node1: 8-chip
    # whole-node mouse.  A PROD 8-chip whole-node job: the chip-count
    # greedy would pick the 7-chip mouse first (least attained after the
    # pin blocks nothing) — but evicting it frees no whole node.  The
    # refined plan must evict ONLY the whole-node victim.
    scn = make_scenario(n_nodes=2, horizon="20m")
    jobs = [
        mk_job("pin", submit_s=0, chips=1, dur_s=10_000.0, preemptible=False),
        mk_job("m_sub", submit_s=0, chips=7, dur_s=10_000.0,
               tier=Tier.BEST_EFFORT),
        mk_job("m_whole", submit_s=61, chips=8, dur_s=10_000.0,
               tier=Tier.BEST_EFFORT),
        mk_job("p", submit_s=130, chips=8, dur_s=100.0, tier=Tier.PROD),
    ]
    sim, fleet, sink = run_sim(scn, jobs)
    pre = preempted(sink)
    assert [p[0] for p in pre] == ["m_whole"]  # the sub-node mouse spared
    assert sim._jobs["p"].job.status is JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# 2. Claims memory: grace window > round must not double-evict
# ---------------------------------------------------------------------------


def test_no_redundant_second_victim_set_while_grace_pending():
    # 4 nodes x 8, four BE mice with save=120 s (grace 120 > round 60).
    # PROD p needs 16 chips (2 nodes).  Wake 120 evicts exactly 2 mice;
    # wake 180 must NOT evict the other two (previously it did: the
    # graced victims were invisible and the planner re-planned).  p
    # places at 240 when the grace ends.
    scn = make_scenario(n_nodes=4, horizon="30m")
    mice = [
        mk_job(f"m{i}", submit_s=0, chips=8, dur_s=10_000.0, save=120.0,
               tier=Tier.BEST_EFFORT)
        for i in range(4)
    ]
    p = mk_job("p", submit_s=70, chips=16, dur_s=100.0, tier=Tier.PROD)
    sim, fleet, sink = run_sim(scn, mice + [p])
    pre = preempted(sink)
    assert len(pre) == 2  # minimum victim set, exactly once
    assert all(t == 120 * S for _, t, _ in pre)
    assert ("job_started", "p", 240 * S) in sink.of("job_started")
    # The spared mice never left their nodes.
    spared = {"m0", "m1", "m2", "m3"} - {jid for jid, _, _ in pre}
    for jid in spared:
        assert sim._jobs[jid].job.status is JobStatus.RUNNING


# ---------------------------------------------------------------------------
# 3. Leaf health: DRAINING-node victims are not credited
# ---------------------------------------------------------------------------


def test_draining_node_victims_not_evicted_for_unreachable_capacity():
    # 2 nodes x 8, both holding BE mice; node0 drained at t=70 (default
    # 1 h grace keeps its resident running).  A PROD 16-chip job arrives:
    # only 8 healthy chips can ever be offered, so evicting BOTH mice
    # (chip-count 16) buys nothing — the fixed planner emits no
    # scheduler evictions at all inside the horizon.
    scn = make_scenario(n_nodes=2, horizon="30m")
    mice = [
        mk_job(f"m{i}", submit_s=0, chips=8, dur_s=10_000.0,
               tier=Tier.BEST_EFFORT)
        for i in range(2)
    ]
    p = mk_job("p", submit_s=90, chips=16, dur_s=100.0, tier=Tier.PROD)
    sim, fleet, sink = run_sim(
        scn,
        mice + [p],
        pre_events=[(70 * S, EventType.MAINTENANCE_DRAIN, "m/c/node0")],
    )
    triggers = {c[3] for c in sink.of("job_preempted")}
    assert "scheduler" not in triggers  # no churn for unreachable chips
    assert sim._jobs["p"].job.status is JobStatus.ADMITTED


# ---------------------------------------------------------------------------
# 4. Event-triggered scheduler + closed-loop backlog
# ---------------------------------------------------------------------------


def test_event_triggered_scheduler_with_pure_backlog_generates_work():
    # wake_interval=None + a workload with ONLY a closed-loop class:
    # previously run() seeded no wake, refill never ran, and the run
    # completed with zero submissions.  The engine now seeds one wake at
    # t=0 whenever the source implements refill.
    class EventTriggered(TieredPriorityScheduler):
        wake_interval = None

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
                    "bemice": {
                        "class": "eval",
                        "arrival": "backlog[target_pending=2]",
                        "chips": 8,
                        "duration": "2m",
                        "checkpoint_interval": 0,
                        "abort_prob": 0,
                    }
                },
            },
        }
    )
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    sink = RecSink()
    sim = Simulator(scn, fleet, source, EventTriggered(), sink, rng=rng)
    sim.run()
    assert len(sink.of("job_submitted")) > 0  # was 0 before the fix
    assert len(sink.of("job_finished")) > 0


# ---------------------------------------------------------------------------
# 5. failure_second_order keys on the last requeue cause
# ---------------------------------------------------------------------------


def test_failure_tag_not_sticky_after_restart():
    # p (PROD 16) fails once, restarts by evicting b — that eviction IS
    # failure_second_order.  Later p is drained off its nodes (an
    # ordinary maintenance requeue) and evicts b again to get back in:
    # that eviction must be tagged "scheduler" (previously the lifetime
    # n_failures counter kept tagging it failure_second_order forever).
    scn = make_scenario(
        n_nodes=2, horizon="4h",
        fm={"repair_auto_min": [60, 60], "drain_grace": "10m"},
    )
    p = mk_job("p", submit_s=0, chips=16, dur_s=20_000.0, tier=Tier.PROD,
               interval=100.0)
    b = mk_job("b", submit_s=150, chips=8, dur_s=30_000.0, tier=Tier.BATCH)
    sim, fleet, sink = run_sim(
        scn,
        [p, b],
        pre_events=[
            (100 * S, EventType.NODE_FAILURE, "m/c/node0"),
            (5000 * S, EventType.MAINTENANCE_DRAIN, "m/c/node1"),
        ],
    )
    by_victim: dict[str, list[str]] = {}
    for _, jid, _, trigger in sink.of("job_preempted"):
        by_victim.setdefault(jid, []).append(trigger)
    # b was evicted twice by p: once as failure fallout, once as an
    # ordinary priority eviction after p's non-failure (drain) requeue.
    assert by_victim["b"][0] == "failure_second_order"
    assert by_victim["b"][-1] == "scheduler"
    # p itself was drained off with the maintenance trigger in between.
    assert "maintenance" in by_victim.get("p", [])


# ---------------------------------------------------------------------------
# 6. Metrics honesty: source_class, BE exclusion, live progress
# ---------------------------------------------------------------------------


def _collector_run():
    scn = make_scenario(n_nodes=2, horizon="10m")
    fleet = build_fleet(scn)
    collector = MetricsCollector(scn.sim.horizon_us, fleet=fleet)
    open_job = mk_job("ft", chips=8, dur_s=60.0, tier=Tier.BATCH)
    open_job.job_class = JobClass.FINETUNE
    open_job.source_class = "finetune"
    be_job = mk_job("be", chips=8, dur_s=60.0, tier=Tier.BEST_EFFORT)
    be_job.job_class = JobClass.FINETUNE
    be_job.source_class = "best_effort"
    live = mk_job("live", submit_s=61.0, chips=8, dur_s=10_000.0,
                  tier=Tier.BATCH, interval=100.0)
    sim = Simulator(
        scn, fleet, ListSource([open_job, be_job, live]),
        TieredPriorityScheduler(), collector,
    )
    sim.run()
    return collector


def test_source_class_in_jobs_parquet_and_be_excluded_from_waits():
    collector = _collector_run()
    df = jobs_dataframe(collector)
    assert "source_class" in df.columns
    by_id = df.set_index("job_id")
    assert by_id.loc["ft", "source_class"] == "finetune"
    assert by_id.loc["be", "source_class"] == "best_effort"
    summary = build_summary(collector)
    full = summary["full"]
    # Per-JobClass stats exclude the BEST_EFFORT-tier job: only the open
    # finetune and the live BATCH eval are counted.
    assert full["queue_wait_s"]["FINETUNE"]["job_weighted"]["n"] == 1
    # Source-class breakdown exists and carries no best-effort key.
    assert "finetune" in full["queue_wait_s_by_source_class"]
    assert "best_effort" not in full["queue_wait_s_by_source_class"]
    assert "best_effort" not in full.get("jct_s_by_source_class", {})


def test_still_running_job_carries_banked_progress_in_jobs_parquet():
    collector = _collector_run()
    df = jobs_dataframe(collector).set_index("job_id")
    # "live" runs from 120 s to the 600 s horizon with interval=100:
    # floor(480/100)*100 = 400 work-seconds banked x 8 chips.
    assert df.loc["live", "status"] == "RUNNING"
    assert df.loc["live", "productive_chip_s"] == pytest.approx(400.0 * 8)
