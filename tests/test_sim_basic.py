"""Engine tests: FIFO ordering, completion/checkpoint math, preemption,
failures, drains, timeouts, and the determinism contract.

All expected times below are hand-computed.  Conventions: fleets are tiny
(1-2 nodes x 8 chips, cluster root "m/c", leaves "m/c/node0"...), the
scheduler round is 60 s, failures/maintenance are OFF unless a test turns
them on, and forced failures/drains are injected by pushing events into
``sim.queue`` before ``run()``.
"""

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.events import EventType
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import PassThrough, Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.model import (
    Constraint,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    NodeState,
    PreemptMode,
    Tier,
)
from fleetsim.schedulers.base import Preempt, get_scheduler, registered_schedulers
from fleetsim.schedulers.fifo import FIFOScheduler
from fleetsim.workload.base import ListSource

S = 1_000_000  # one second in microseconds

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

N0, N1 = "m/c/node0", "m/c/node1"
CLUSTER = "m/c"


def make_scenario(
    n_nodes=2,
    per_node=8,
    horizon="1h",
    round_="60s",
    mtbf_days=0.0,
    maint_rate=0.0,
    drain_grace="10m",
    repair_min=(60, 60),
    manual_frac=0.0,
    seed=0,
):
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
            "failure_model": {
                "node_mtbf_days": mtbf_days,
                "repair_auto_min": list(repair_min),
                "repair_manual_frac": manual_frac,
                "repair_manual_days": [1, 1],
                "maintenance_rate_per_node_month": maint_rate,
                "drain_grace": drain_grace,
            },
            "workload": WORKLOAD,
        }
    )


def mk_job(
    jid,
    submit_s=0.0,
    chips=8,
    dur_s=100.0,
    interval=0.0,
    save=0.0,
    restart=0.0,
    tier=Tier.BATCH,
    preemptible=True,
    min_runtime=0.0,
    max_lifetime=None,
    valid_until_s=None,
    within=None,
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
        max_lifetime_s=max_lifetime,
        true_duration_s=dur_s,
        checkpoint_interval_s=interval,
        checkpoint_save_s=save,
        restart_overhead_s=restart,
        valid_until=int(round(valid_until_s * 1e6)) if valid_until_s is not None else None,
    )


class ListSink:
    """Records every sink call as a comparable tuple."""

    def __init__(self):
        self.calls = []

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]

    def events(self):
        """All calls except periodic flush noise and progress accounting."""
        return [c for c in self.calls if c[0] not in ("flush", "job_progress")]

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

    def job_progress(self, job, start_us, end_us, productive_chip_s, lost_chip_s):
        self.calls.append(
            ("job_progress", job.id, start_us, end_us,
             round(productive_chip_s, 6), round(lost_chip_s, 6))
        )

    def job_finished(self, job, t, status, productive_chip_s, lost_chip_s):
        self.calls.append(
            ("job_finished", job.id, t, status.name,
             round(productive_chip_s, 6), round(lost_chip_s, 6))
        )

    def node_failed(self, node_id, t, killed_alloc_ids, cause="unknown"):
        self.calls.append(("node_failed", node_id, t, tuple(killed_alloc_ids)))

    def node_repaired(self, node_id, t):
        self.calls.append(("node_repaired", node_id, t))

    def node_drain_started(self, node_id, t):
        self.calls.append(("node_drain_started", node_id, t))

    def chips_allocated(self, n, chip_type, t):
        self.calls.append(("chips_allocated", n, chip_type, t))

    def chips_freed(self, n, chip_type, t):
        self.calls.append(("chips_freed", n, chip_type, t))

    def healthy_delta(self, n_chips, chip_type, t):
        self.calls.append(("healthy_delta", n_chips, chip_type, t))

    def flush(self, t, fleet, n_pending, n_running):
        self.calls.append(("flush", t, n_pending, n_running))


def run_sim(scenario, jobs, scheduler=None, pre_events=(), rng=None, strict=True):
    fleet = build_fleet(scenario)
    sink = ListSink()
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        scheduler if scheduler is not None else FIFOScheduler(),
        sink,
        rng=rng,
        strict=strict,
    )
    for t, etype, payload in pre_events:
        sim.queue.push(t, etype, payload)
    sim.run()
    fleet.check_invariants()
    return sim, fleet, sink


class PreemptOnce(FIFOScheduler):
    """FIFO, but emits one Preempt action at the first wake >= at_us."""

    def __init__(self, victim, at_us, mode=PreemptMode.REQUEUE, preemptor=None):
        super().__init__()
        self.victim, self.at_us = victim, at_us
        self.mode, self.preemptor = mode, preemptor
        self.fired = False

    def schedule(self, view):
        if not self.fired and view.now >= self.at_us:
            self.fired = True
            return [Preempt(self.victim, self.mode, self.preemptor)]
        return super().schedule(view)


# ---------------------------------------------------------------------------
# Completion math
# ---------------------------------------------------------------------------


def test_single_job_completes_no_checkpoint():
    # No checkpointing (interval=0 -> eff=1): 100 s of work = 100 s wall.
    scn = make_scenario(n_nodes=1)
    sim, fleet, sink = run_sim(scn, [mk_job("a", dur_s=100.0)])
    assert sink.events() == [
        ("job_submitted", "a", 0),
        ("job_admitted", "a", 0),
        ("job_started", "a", 0),
        ("chips_allocated", 8, "h100", 0),
        ("chips_freed", 8, "h100", 100 * S),
        ("job_finished", "a", 100 * S, "COMPLETED", 800.0, 0.0),
    ]
    assert sim._jobs["a"].job.status is JobStatus.COMPLETED
    assert fleet.free_chips(CLUSTER) == 8


def test_completion_math_with_checkpoint_overhead():
    # eff = 100/(100+25) = 0.8: 180 s of work runs in 225 s of wall clock.
    # Submitted at t=30 s -> placed at the next round boundary (60 s).
    scn = make_scenario(n_nodes=1)
    _, _, sink = run_sim(
        scn, [mk_job("a", submit_s=30, dur_s=180.0, interval=100.0, save=25.0)]
    )
    assert sink.of("job_started") == [("job_started", "a", 60 * S)]
    assert sink.of("job_finished") == [
        ("job_finished", "a", 285 * S, "COMPLETED", 1440.0, 0.0)
    ]


def test_short_job_pays_no_checkpoint_amortization():
    # Remaining work (80 s) <= interval (100 s): the job never writes a
    # checkpoint, so eff = 1 — no save tax (finding: amortized eff taxed
    # 2-minute evals under a 1 h interval).
    scn = make_scenario(n_nodes=1)
    _, _, sink = run_sim(
        scn, [mk_job("a", submit_s=30, dur_s=80.0, interval=100.0, save=25.0)]
    )
    assert sink.of("job_finished") == [
        ("job_finished", "a", 140 * S, "COMPLETED", 640.0, 0.0)
    ]


def test_flush_cadence_and_counts():
    scn = make_scenario(n_nodes=1)  # horizon 1 h, round 60 s
    _, _, sink = run_sim(scn, [mk_job("a", dur_s=100.0)])
    flushes = sink.of("flush")
    # First chained flush at one round, final flush exactly at the horizon.
    assert flushes[0] == ("flush", 60 * S, 0, 1)  # "a" still running at 60 s
    assert flushes[-1] == ("flush", 3600 * S, 0, 0)
    assert [f[1] for f in flushes] == [60 * k * S for k in range(1, 61)]


# ---------------------------------------------------------------------------
# FIFO: strict vs best-effort
# ---------------------------------------------------------------------------


def _fifo_jobs():
    return [
        mk_job("a", submit_s=0, chips=8, dur_s=120.0),   # takes node0
        mk_job("b", submit_s=1, chips=16, dur_s=50.0),   # needs both nodes
        mk_job("c", submit_s=2, chips=1, dur_s=50.0),    # would fit on node1
    ]


def test_fifo_strict_head_of_line_blocks():
    scn = make_scenario(n_nodes=2)
    _, _, sink = run_sim(scn, _fifo_jobs(), scheduler=FIFOScheduler(strict=True))
    # b (head of line at wake 60) can't fit while a runs, so c is blocked
    # behind it even though a chip is free.
    assert sink.of("job_started") == [
        ("job_started", "a", 0),
        ("job_started", "b", 120 * S),
        ("job_started", "c", 180 * S),
    ]


def test_fifo_best_effort_skips_blocked_head():
    scn = make_scenario(n_nodes=2)
    _, _, sink = run_sim(scn, _fifo_jobs(), scheduler=FIFOScheduler(strict=False))
    # c flows around the stuck 16-chip job b.
    assert sink.of("job_started") == [
        ("job_started", "a", 0),
        ("job_started", "c", 60 * S),
        ("job_started", "b", 120 * S),
    ]


def test_find_placement_reserves_within_wake():
    # Two 8-chip jobs on a 2-node fleet, same wake: sequential
    # find_placement calls must not double-book node0.
    scn = make_scenario(n_nodes=2)
    _, fleet, sink = run_sim(
        scn, [mk_job("a", chips=8, dur_s=50.0), mk_job("b", chips=8, dur_s=50.0)]
    )
    assert sink.of("job_started") == [
        ("job_started", "a", 0),
        ("job_started", "b", 0),
    ]
    assert fleet.free_chips(CLUSTER) == 16  # both completed and released


# ---------------------------------------------------------------------------
# Preemption
# ---------------------------------------------------------------------------


def test_preempt_requeue_floors_to_checkpoint_and_keeps_submit_order():
    # a: 1000 s work, ckpt every 100 s (save=0 -> eff=1, zero grace),
    # restart overhead 100 s.  Preempted at t=240: kept=200, lost=40.
    # c arrived later (t=20) but a resumes FIRST (original submit_t).
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("a", submit_s=0, dur_s=1000.0, interval=100.0, restart=100.0),
        mk_job("c", submit_s=20, dur_s=50.0),
    ]
    sim, _, sink = run_sim(scn, jobs, scheduler=PreemptOnce("a", 240 * S))
    assert sink.of("job_preempted") == [("job_preempted", "a", 240 * S, "scheduler")]
    assert sink.of("job_requeued") == [("job_requeued", "a", 240 * S)]
    # Resume at next boundary (300): 100 s restart + 800 s remaining -> 1200.
    assert sink.of("job_started") == [
        ("job_started", "a", 0),
        ("job_started", "a", 300 * S),
        ("job_started", "c", 1200 * S),
    ]
    assert sink.of("job_finished")[0] == (
        "job_finished", "a", 1200 * S, "COMPLETED", 8000.0, 320.0,
    )
    assert sim._jobs["a"].job.status is JobStatus.COMPLETED


def test_preempt_cancel_frees_immediately():
    scn = make_scenario(n_nodes=1)
    jobs = [mk_job("a", dur_s=1000.0, interval=100.0)]
    sim, fleet, sink = run_sim(
        scn, jobs, scheduler=PreemptOnce("a", 240 * S, mode=PreemptMode.CANCEL)
    )
    assert sink.of("job_finished") == [
        ("job_finished", "a", 240 * S, "CANCELED", 1600.0, 320.0)
    ]
    assert sink.of("chips_freed") == [("chips_freed", 8, "h100", 240 * S)]
    assert sim._jobs["a"].job.status is JobStatus.CANCELED
    assert fleet.free_chips(CLUSTER) == 8


def test_preempt_validation_strict_raises():
    scn = make_scenario(n_nodes=1)
    # Non-preemptible victim.
    with pytest.raises(ValueError, match="not preemptible"):
        run_sim(
            scn,
            [mk_job("a", dur_s=500.0, preemptible=False)],
            scheduler=PreemptOnce("a", 60 * S),
        )
    # MONITORING is always protected.
    with pytest.raises(ValueError, match="MONITORING"):
        run_sim(
            scn,
            [mk_job("a", dur_s=500.0, tier=Tier.MONITORING)],
            scheduler=PreemptOnce("a", 60 * S),
        )
    # min_runtime guard (this stint).
    with pytest.raises(ValueError, match="min_runtime"):
        run_sim(
            scn,
            [mk_job("a", dur_s=500.0, min_runtime=300.0)],
            scheduler=PreemptOnce("a", 60 * S),
        )
    # No preemption within PROD (preemptor context present).
    with pytest.raises(ValueError, match="PROD"):
        run_sim(
            scn,
            [
                mk_job("a", submit_s=0, dur_s=500.0, tier=Tier.PROD),
                mk_job("p", submit_s=1, dur_s=50.0, tier=Tier.PROD),
            ],
            scheduler=PreemptOnce("a", 60 * S, preemptor="p"),
        )
    # Equal band (BATCH vs BATCH) is refused with a preemptor context.
    with pytest.raises(ValueError, match="not below"):
        run_sim(
            scn,
            [
                mk_job("a", submit_s=0, dur_s=500.0, tier=Tier.BATCH),
                mk_job("p", submit_s=1, dur_s=50.0, tier=Tier.BATCH),
            ],
            scheduler=PreemptOnce("a", 60 * S, preemptor="p"),
        )


def test_preempt_validation_lenient_skips():
    scn = make_scenario(n_nodes=1)
    _, _, sink = run_sim(
        scn,
        [mk_job("a", dur_s=500.0, preemptible=False)],
        scheduler=PreemptOnce("a", 60 * S),
        strict=False,
    )
    # Illegal intent ignored: a runs to completion untouched.
    assert sink.of("job_preempted") == []
    assert sink.of("job_finished") == [
        ("job_finished", "a", 500 * S, "COMPLETED", 4000.0, 0.0)
    ]


# ---------------------------------------------------------------------------
# Node failure
# ---------------------------------------------------------------------------


def test_node_failure_kills_gang_job_retries_and_completes():
    # a spans both nodes (16 chips); node0 forced down at t=130.
    # kept=100 (ckpt 100 s), lost=30; repair exactly 60 min later (uniform
    # [60,60], manual_frac=0); resume at wake 3780 with 50 s restart.
    scn = make_scenario(n_nodes=2, horizon="2h")
    jobs = [mk_job("a", chips=16, dur_s=500.0, interval=100.0, restart=50.0)]
    sim, fleet, sink = run_sim(
        scn, jobs, pre_events=[(130 * S, EventType.NODE_FAILURE, N0)]
    )
    assert sink.of("node_failed") == [("node_failed", N0, 130 * S, ("a",))]
    assert sink.of("healthy_delta") == [
        ("healthy_delta", -8, "h100", 130 * S),
        ("healthy_delta", 8, "h100", 3730 * S),
    ]
    assert sink.of("job_requeued") == [("job_requeued", "a", 130 * S)]
    assert sink.of("node_repaired") == [("node_repaired", N0, 3730 * S)]
    # 3780 (first boundary after repair) + 50 restart + 400 remaining = 4230.
    assert sink.of("job_started") == [
        ("job_started", "a", 0),
        ("job_started", "a", 3780 * S),
    ]
    assert sink.of("job_finished") == [
        ("job_finished", "a", 4230 * S, "COMPLETED", 8000.0, 480.0)
    ]
    assert sim._jobs["a"].n_failures == 1
    assert fleet.free_chips(CLUSTER) == 16
    assert fleet.domain(N0).state is NodeState.HEALTHY


def test_failure_never_terminal_counters_consistent():
    # Sub-node job sharing a leaf with another job: failure kills both
    # residents, both requeue (never NODE_FAIL), both finish eventually.
    scn = make_scenario(n_nodes=1, horizon="2h")
    jobs = [
        mk_job("a", chips=4, dur_s=300.0, interval=0.0),
        mk_job("b", chips=4, dur_s=300.0, interval=0.0),
    ]
    sim, fleet, sink = run_sim(
        scn, jobs, pre_events=[(100 * S, EventType.NODE_FAILURE, N0)]
    )
    assert sink.of("node_failed") == [("node_failed", N0, 100 * S, ("a", "b"))]
    # No checkpointing: ALL progress lost (kept floors at 0).
    finished = {c[1]: c for c in sink.of("job_finished")}
    # Repair at 100+3600=3700, resume at 3720, complete at 4020.
    assert finished["a"] == ("job_finished", "a", 4020 * S, "COMPLETED", 1200.0, 400.0)
    assert finished["b"] == ("job_finished", "b", 4020 * S, "COMPLETED", 1200.0, 400.0)
    for jid in ("a", "b"):
        assert sim._jobs[jid].n_failures == 1
        assert sim._jobs[jid].job.status is JobStatus.COMPLETED
    assert fleet.free_chips(CLUSTER) == 8


# ---------------------------------------------------------------------------
# Maintenance drain lifecycle
# ---------------------------------------------------------------------------


def test_drain_lifecycle():
    # node0 drained at t=100 (grace 10 min): resident a keeps running until
    # the grace expires at 700, is REQUEUE-preempted (trigger maintenance),
    # node0 goes MAINTENANCE for 60 min, HEALTHY again at 4300.
    scn = make_scenario(n_nodes=2, horizon="2h")
    jobs = [
        mk_job("a", submit_s=0, chips=8, dur_s=2000.0, interval=100.0),
        mk_job("b", submit_s=5, chips=8, dur_s=50.0),
    ]
    sim, fleet, sink = run_sim(
        scn, jobs, pre_events=[(100 * S, EventType.MAINTENANCE_DRAIN, N0)]
    )
    assert sink.of("node_drain_started") == [("node_drain_started", N0, 100 * S)]
    assert sink.of("job_preempted") == [
        ("job_preempted", "a", 700 * S, "maintenance")
    ]
    assert sink.of("job_requeued") == [("job_requeued", "a", 700 * S)]
    # a resumes on node1 at the next boundary: kept 700 of 2000, no restart
    # overhead configured -> completes 720 + 1300 = 2020.
    assert sink.of("job_started") == [
        ("job_started", "a", 0),
        ("job_started", "b", 60 * S),
        ("job_started", "a", 720 * S),
    ]
    assert sink.of("job_finished")[-1] == (
        "job_finished", "a", 2020 * S, "COMPLETED", 16000.0, 0.0,
    )
    assert sink.of("node_repaired") == [("node_repaired", N0, 4300 * S)]
    assert fleet.domain(N0).state is NodeState.HEALTHY
    assert fleet.free_chips(CLUSTER) == 16
    assert sim._jobs["a"].n_failures == 0  # a drain is not a failure


def test_draining_node_blocks_new_placements():
    scn = make_scenario(n_nodes=1, horizon="30m")
    # Drain the only node before the job arrives: it can never place.
    jobs = [mk_job("a", submit_s=120, chips=8, dur_s=50.0)]
    sim, fleet, sink = run_sim(
        scn, jobs, pre_events=[(60 * S, EventType.MAINTENANCE_DRAIN, N0)]
    )
    assert sink.of("job_started") == []
    # Empty drained node transitions to MAINTENANCE at grace expiry
    # (60+600=660) and returns HEALTHY 60 min later — after the horizon.
    assert fleet.domain(N0).state is NodeState.MAINTENANCE
    assert sim._jobs["a"].job.status is JobStatus.ADMITTED  # still queued


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


def test_valid_until_expires_queued_job():
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("a", submit_s=0, chips=8, dur_s=300.0),  # blocks the node
        mk_job("v", submit_s=0, chips=8, dur_s=50.0, valid_until_s=100.0),
    ]
    sim, _, sink = run_sim(scn, jobs)
    assert ("job_finished", "v", 100 * S, "FAILED", 0.0, 0.0) in sink.calls
    assert sim._jobs["v"].job.status is JobStatus.FAILED
    assert sink.of("job_started") == [("job_started", "a", 0)]
    assert len(sim._pending) == 0


def test_max_lifetime_timeout_running_job():
    scn = make_scenario(n_nodes=1)
    jobs = [mk_job("a", dur_s=1000.0, interval=100.0, max_lifetime=250.0)]
    sim, fleet, sink = run_sim(scn, jobs)
    assert sink.of("job_finished") == [
        ("job_finished", "a", 250 * S, "TIMEOUT", 1600.0, 400.0)
    ]
    assert sim._jobs["a"].job.status is JobStatus.TIMEOUT
    assert fleet.free_chips(CLUSTER) == 8


# ---------------------------------------------------------------------------
# Views, admission, sources, registry
# ---------------------------------------------------------------------------


def test_view_attained_service_and_checkpoint_age():
    class Probe(FIFOScheduler):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def schedule(self, view):
            self.seen[view.now] = [
                (j.id, j.attained_service_chip_s, j.checkpoint_age_s)
                for j in view.running()
            ]
            return super().schedule(view)

    scn = make_scenario(n_nodes=1)
    probe = Probe()
    run_sim(scn, [mk_job("a", dur_s=1000.0, interval=100.0)], scheduler=probe)
    # At wake 120: 120 s of work done x 8 chips; 20 s since the 100 s ckpt.
    assert probe.seen[120 * S] == [("a", 960.0, 20.0)]


def test_terminal_status_override_honored():
    # The workload phase marks aborting jobs by subclassing Job with a
    # terminal_status_override field (Job itself is slots=True); the engine
    # finishes such jobs with that status instead of COMPLETED.
    from dataclasses import dataclass

    @dataclass(slots=True)
    class AbortJob(Job):
        terminal_status_override: JobStatus | None = None

    base = mk_job("a", dur_s=100.0)
    job = AbortJob(
        **{f: getattr(base, f) for f in (
            "id", "tenant", "job_class", "submit_t", "gangs", "tier",
            "true_duration_s",
        )},
        checkpoint_interval_s=0.0,  # disable ckpt: 100 s work = 100 s wall
        terminal_status_override=JobStatus.FAILED,
    )
    scn = make_scenario(n_nodes=1)
    sim, _, sink = run_sim(scn, [job])
    assert sink.of("job_finished") == [
        ("job_finished", "a", 100 * S, "FAILED", 800.0, 0.0)
    ]
    assert sim._jobs["a"].job.status is JobStatus.FAILED


def test_admission_reject_terminal_failed():
    class RejectAll:
        def admit(self, job, t):
            return False

    scn = make_scenario(n_nodes=1)
    fleet = build_fleet(scn)
    sink = ListSink()
    sim = Simulator(
        scn, fleet, ListSource([mk_job("a")]), FIFOScheduler(), sink,
        admission=RejectAll(),
    )
    sim.run()
    assert sink.events() == [
        ("job_submitted", "a", 0),
        ("job_finished", "a", 0, "FAILED", 0.0, 0.0),
    ]
    assert isinstance(PassThrough().admit(mk_job("x"), 0), bool)


def test_list_source_sorts_by_submit_then_id():
    a, b = mk_job("a", submit_s=10), mk_job("b", submit_s=5)
    src = ListSource([a, b])
    assert src.next_arrival() == (5 * S, b)
    assert src.next_arrival() == (10 * S, a)
    assert src.next_arrival() is None
    assert src.next_arrival() is None  # stays exhausted


def test_scheduler_registry():
    assert "fifo" in registered_schedulers()
    s = get_scheduler("fifo", {"strict": False})
    assert isinstance(s, FIFOScheduler)
    assert s.strict is False
    with pytest.raises(ValueError, match="unknown scheduler"):
        get_scheduler("definitely-not-registered")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _det_jobs():
    return [
        mk_job("j0", submit_s=0, chips=8, dur_s=600, interval=100, save=10, restart=60),
        mk_job("j1", submit_s=120, chips=16, dur_s=1800, interval=100, save=10,
               restart=60),
        mk_job("j2", submit_s=600, chips=8, dur_s=900, interval=100, save=10,
               restart=60),
        mk_job("j3", submit_s=1200, chips=4, dur_s=300),
        mk_job("j4", submit_s=2400, chips=16, dur_s=3600, interval=200, save=20,
               restart=120),
    ]


def _det_scn():
    # Failures ON: mtbf 0.2 node-days over 2 nodes -> ~10 failures/day.
    return make_scenario(n_nodes=2, horizon="1d", mtbf_days=0.2)


def test_identical_runs_produce_identical_sink_sequences():
    _, _, sink1 = run_sim(_det_scn(), _det_jobs())
    _, _, sink2 = run_sim(_det_scn(), _det_jobs())
    assert sink1.calls == sink2.calls
    assert len(sink1.of("node_failed")) > 0  # failures actually exercised


def test_failures_seed_changes_failures_not_arrivals():
    _, _, base = run_sim(_det_scn(), _det_jobs())
    _, _, tweaked = run_sim(
        _det_scn(), _det_jobs(), rng=RngStreams(0, overrides={"failures": 4242})
    )
    # Arrival side is a paired sample: byte-identical.
    assert base.of("job_submitted") == tweaked.of("job_submitted")
    assert base.of("job_admitted") == tweaked.of("job_admitted")
    # Failure realization changed.
    assert len(base.of("node_failed")) > 0
    assert len(tweaked.of("node_failed")) > 0
    assert base.of("node_failed") != tweaked.of("node_failed")
