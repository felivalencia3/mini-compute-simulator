"""Tiered-priority scheduler tests, run against the real engine.

All expected times are hand-computed.  Conventions match test_sim_basic:
tiny fleets (cluster root "m/c", leaves "m/c/node0"... or
"m/c/pod0/node0"...), 60 s scheduler round, failures/maintenance OFF, and
strict engine mode — so any doomed Preempt the policy emitted would raise
instead of being silently skipped.

Checkpoint conventions used below: ``interval=0`` disables checkpointing
(victims lose ALL progress, eff=1 keeps wall math trivial) while ``save``
still sets the REQUEUE grace window (engine grace = checkpoint_save_s).
"""

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.model import (
    Constraint,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    PreemptMode,
    Tier,
)
from fleetsim.schedulers.base import get_scheduler, registered_schedulers
from fleetsim.schedulers.placement import FirstFit
from fleetsim.schedulers.tiered_priority import TieredPriorityScheduler
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

CLUSTER = "m/c"


def make_scenario(
    n_nodes=2, per_node=8, horizon="30m", round_="60s", levels=None, counts=None
):
    if levels is None:
        levels, counts = ["node"], [n_nodes]
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
            "failure_model": {
                "node_mtbf_days": 0.0,
                "maintenance_rate_per_node_month": 0.0,
            },
            "workload": WORKLOAD,
        }
    )


def mk_job(
    jid,
    submit_s=0.0,
    chips=8,
    dur_s=100.0,
    tier=Tier.BATCH,
    interval=0.0,
    save=0.0,
    restart=0.0,
    preemptible=True,
    min_runtime=0.0,
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
        true_duration_s=dur_s,
        checkpoint_interval_s=interval,
        checkpoint_save_s=save,
        restart_overhead_s=restart,
    )


class RecSink:
    """Records every sink call as a comparable tuple (job_started keeps the
    gang anchor so placements can be asserted)."""

    def __init__(self):
        self.calls = []

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]

    def job_submitted(self, job, t):
        self.calls.append(("job_submitted", job.id, t))

    def job_admitted(self, job, t):
        self.calls.append(("job_admitted", job.id, t))

    def job_started(self, job, alloc, t):
        self.calls.append(("job_started", job.id, t, alloc.gangs[0].anchor))

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


def run_sim(scenario, jobs, scheduler=None, strict=True):
    fleet = build_fleet(scenario)
    sink = RecSink()
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        scheduler if scheduler is not None else TieredPriorityScheduler(),
        sink,
        strict=strict,
    )
    sim.run()
    fleet.check_invariants()
    return sim, fleet, sink


def started(sink):
    return [(c[1], c[2]) for c in sink.of("job_started")]


def preempted(sink):
    return [(c[1], c[2], c[3]) for c in sink.of("job_preempted")]


# ---------------------------------------------------------------------------
# Core preemption behavior
# ---------------------------------------------------------------------------


def test_prod_preempts_batch_and_eventually_starts():
    # 1 node x 8.  BATCH b (600 s, no ckpt, 30 s grace) holds the node.
    # PROD p (8 chips) arrives at 90 -> wake 120: preempt b (REQUEUE).
    # Grace ends 150 (b loses all 150 s of work), wake 180: p places.
    # p done 280 -> wake 300: b restarts from zero, done 900.
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("b", submit_s=0, dur_s=600.0, save=30.0, tier=Tier.BATCH),
        mk_job("p", submit_s=90, dur_s=100.0, tier=Tier.PROD),
    ]
    sim, fleet, sink = run_sim(scn, jobs)
    assert preempted(sink) == [("b", 120 * S, "scheduler")]
    assert sink.of("job_requeued") == [("job_requeued", "b", 150 * S)]
    assert started(sink) == [("b", 0), ("p", 180 * S), ("b", 300 * S)]
    assert sink.of("job_finished") == [
        ("job_finished", "p", 280 * S, "COMPLETED", 800.0, 0.0),
        ("job_finished", "b", 900 * S, "COMPLETED", 4800.0, 1200.0),
    ]
    assert sim._jobs["b"].job.status is JobStatus.COMPLETED
    assert fleet.free_chips(CLUSTER) == 8


def test_freed_chip_accounting_includes_sub_node_victims():
    # 2 nodes x 8.  node0: b1 (4 chips) + b2 (2 chips) sub-node, 2 free.
    # node1: b3 (8 chips, whole).  PROD p needs 8 whole-node chips.
    # Greedy at wake 120: free=2; victim order by attained chip-seconds
    # (b2=240 < b1=480 < b3=960): 2+2=4 < 8, +4=8 -> {b2, b1}; b3 spared.
    scn = make_scenario(n_nodes=2)
    jobs = [
        mk_job("b1", submit_s=0, chips=4, dur_s=1000.0, save=30.0),
        mk_job("b2", submit_s=0, chips=2, dur_s=1000.0, save=30.0),
        mk_job("b3", submit_s=0, chips=8, dur_s=1000.0, save=30.0),
        mk_job("p", submit_s=90, chips=8, dur_s=100.0, tier=Tier.PROD),
    ]
    sim, fleet, sink = run_sim(scn, jobs)
    assert preempted(sink) == [
        ("b2", 120 * S, "scheduler"),
        ("b1", 120 * S, "scheduler"),
    ]
    # Sub-node victims free exactly their own chip counts at grace end.
    assert sink.of("chips_freed")[:2] == [
        ("chips_freed", 2, "h100", 150 * S),
        ("chips_freed", 4, "h100", 150 * S),
    ]
    # p takes the fully-freed node0 at the next wake; b3 runs undisturbed.
    assert ("p", 180 * S) in started(sink)
    finished = {c[1]: c for c in sink.of("job_finished")}
    assert finished["b3"] == (
        "job_finished", "b3", 1000 * S, "COMPLETED", 8000.0, 0.0,
    )
    # b1/b2 restart from zero at wake 300 (p done at 280) and complete.
    assert finished["b1"][3] == "COMPLETED" and finished["b2"][3] == "COMPLETED"
    assert fleet.free_chips(CLUSTER) == 16


def test_prod_never_preempts_prod():
    # 1 node x 8.  PROD p1 runs 400 s; PROD p2 (same band) must wait for
    # natural completion — no Preempt is ever emitted (strict mode would
    # raise if one were).
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("p1", submit_s=0, dur_s=400.0, save=30.0, tier=Tier.PROD),
        mk_job("p2", submit_s=90, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(scn, jobs)
    assert preempted(sink) == []
    assert started(sink) == [("p1", 0), ("p2", 420 * S)]


def test_no_preemption_when_domain_cannot_yield_enough():
    # node0: PROD p1 (6 chips, not a candidate) + BATCH b (2 chips).
    # PROD p2 needs 8: free(0) + preemptable(2) < 8 -> no preemption at
    # all (partial eviction of b would be pure waste).
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("p1", submit_s=0, chips=6, dur_s=300.0, tier=Tier.PROD),
        mk_job("b", submit_s=0, chips=2, dur_s=300.0, tier=Tier.BATCH),
        mk_job("p2", submit_s=90, chips=8, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(scn, jobs)
    assert preempted(sink) == []
    assert started(sink) == [("p1", 0), ("b", 0), ("p2", 300 * S)]


def test_monitoring_outranks_prod():
    # Bands, not "PROD is special": MONITORING sits above PROD and may
    # evict a preemptible PROD victim.
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("p1", submit_s=0, dur_s=600.0, save=30.0, tier=Tier.PROD),
        mk_job("mon", submit_s=90, dur_s=100.0, tier=Tier.MONITORING),
    ]
    _, _, sink = run_sim(scn, jobs)
    assert preempted(sink) == [("p1", 120 * S, "scheduler")]
    assert ("mon", 180 * S) in started(sink)


def test_min_runtime_shields_young_victims():
    # b has min_runtime 300 s: wakes at 120/180/240 must NOT touch it
    # (the engine would raise on a doomed Preempt); the wake at exactly
    # 300 s may.  p then starts after b's 30 s grace, at wake 360.
    scn = make_scenario(n_nodes=1, horizon="1h")
    jobs = [
        mk_job("b", submit_s=0, dur_s=2000.0, save=30.0, min_runtime=300.0),
        mk_job("p", submit_s=90, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(scn, jobs)
    assert preempted(sink) == [("b", 300 * S, "scheduler")]
    assert started(sink) == [("b", 0), ("p", 360 * S), ("b", 480 * S)]


def test_non_preemptible_victims_are_skipped():
    # preemptible=False shields b regardless of band; p waits it out.
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("b", submit_s=0, dur_s=300.0, preemptible=False),
        mk_job("p", submit_s=90, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(scn, jobs)
    assert preempted(sink) == []
    assert started(sink) == [("b", 0), ("p", 300 * S)]


# ---------------------------------------------------------------------------
# Determinism of victim selection
# ---------------------------------------------------------------------------


def test_victim_ties_break_by_id_and_runs_are_identical():
    # Two identical BATCH jobs (same tier, same attained chip-seconds,
    # same submit) -> the tie falls through to id order: ba before bb.
    scn = make_scenario(n_nodes=2, horizon="1h")

    def jobs():
        return [
            mk_job("ba", submit_s=0, chips=8, dur_s=1000.0, save=30.0),
            mk_job("bb", submit_s=0, chips=8, dur_s=1000.0, save=30.0),
            mk_job("p", submit_s=90, chips=16, dur_s=100.0, tier=Tier.PROD),
        ]

    _, _, sink1 = run_sim(scn, jobs())
    assert preempted(sink1) == [
        ("ba", 120 * S, "scheduler"),
        ("bb", 120 * S, "scheduler"),
    ]
    assert ("p", 180 * S) in started(sink1)
    # Byte-identical event sequence on a re-run (fresh scheduler instance).
    _, _, sink2 = run_sim(make_scenario(n_nodes=2, horizon="1h"), jobs())
    assert sink1.calls == sink2.calls


# ---------------------------------------------------------------------------
# Priority-FIFO reduction (no preemption needed)
# ---------------------------------------------------------------------------


def test_reduces_to_priority_fifo_without_preemption():
    # All three jobs arrive at t=0; ids are chosen so plain FIFO order
    # (submit, id) would start a_free first.  Band order wins instead:
    # c_prod and b_batch take the two nodes, a_free waits for c_prod's
    # completion (its FREE band outranks nothing, so no preemption).
    scn = make_scenario(n_nodes=2)
    jobs = [
        mk_job("a_free", submit_s=0, dur_s=100.0, tier=Tier.FREE),
        mk_job("b_batch", submit_s=0, dur_s=200.0, tier=Tier.BATCH),
        mk_job("c_prod", submit_s=0, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(scn, jobs)
    assert preempted(sink) == []
    assert started(sink) == [
        ("c_prod", 0),
        ("b_batch", 0),
        ("a_free", 120 * S),
    ]


# ---------------------------------------------------------------------------
# Preemption mode and storm cap params
# ---------------------------------------------------------------------------


def test_cancel_mode_kills_victim_and_frees_immediately():
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("b", submit_s=0, dur_s=600.0, save=30.0),
        mk_job("p", submit_s=90, dur_s=100.0, tier=Tier.PROD),
    ]
    sim, fleet, sink = run_sim(
        scn, jobs, scheduler=TieredPriorityScheduler(preempt="cancel")
    )
    assert preempted(sink) == [("b", 120 * S, "scheduler")]
    assert sink.of("job_requeued") == []
    # No checkpointing: everything (120 s x 8 chips) is lost, terminal now.
    assert sink.of("job_finished")[0] == (
        "job_finished", "b", 120 * S, "CANCELED", 0.0, 960.0,
    )
    assert sim._jobs["b"].job.status is JobStatus.CANCELED
    assert ("p", 180 * S) in started(sink)
    assert fleet.free_chips(CLUSTER) == 8


def test_preemption_cap_spreads_across_wakes():
    # p needs 16 (both nodes); the full victim set is {ba, bb} but the cap
    # is 1/wake: ba at wake 120, bb at wake 180, p places at wake 240.
    # The stop-scan rule keeps the requeued ba from stealing node0's freed
    # chips at wake 180.
    scn = make_scenario(n_nodes=2, horizon="2h")
    jobs = [
        mk_job("ba", submit_s=0, chips=8, dur_s=2000.0, save=30.0),
        mk_job("bb", submit_s=0, chips=8, dur_s=2000.0, save=30.0),
        mk_job("p", submit_s=90, chips=16, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(
        scn, jobs, scheduler=TieredPriorityScheduler(max_preemptions_per_wake=1)
    )
    assert preempted(sink) == [
        ("ba", 120 * S, "scheduler"),
        ("bb", 180 * S, "scheduler"),
    ]
    assert ("p", 240 * S) in started(sink)


def test_zero_cap_disables_preemption():
    scn = make_scenario(n_nodes=1)
    jobs = [
        mk_job("b", submit_s=0, dur_s=300.0, save=30.0),
        mk_job("p", submit_s=90, dur_s=100.0, tier=Tier.PROD),
    ]
    _, _, sink = run_sim(
        scn, jobs, scheduler=TieredPriorityScheduler(max_preemptions_per_wake=0)
    )
    assert preempted(sink) == []
    assert started(sink) == [("b", 0), ("p", 300 * S)]


# ---------------------------------------------------------------------------
# `within` constraints: victims come from ONE satisfying domain
# ---------------------------------------------------------------------------


def test_within_constraint_selects_victims_in_one_domain():
    # 2 pods x 2 nodes x 8.  pod0 holds b0a+b0b (16 chips), pod1 holds b1
    # (8 chips, 8 free).  p wants 16 `within: pod`: pod1 can never yield
    # 16, pod0 can — both pod0 victims go, b1 is untouched, and p lands
    # anchored at pod0.
    scn = make_scenario(levels=["pod", "node"], counts=[2, 2], horizon="1h")
    jobs = [
        mk_job("b0a", submit_s=0, chips=8, dur_s=2000.0, save=30.0),
        mk_job("b0b", submit_s=0, chips=8, dur_s=2000.0, save=30.0),
        mk_job("b1", submit_s=0, chips=8, dur_s=2000.0, save=30.0),
        mk_job("p", submit_s=90, chips=16, dur_s=100.0, tier=Tier.PROD,
               within="pod"),
    ]
    sim, fleet, sink = run_sim(scn, jobs)
    assert preempted(sink) == [
        ("b0a", 120 * S, "scheduler"),
        ("b0b", 120 * S, "scheduler"),
    ]
    starts = [(c[1], c[2], c[3]) for c in sink.of("job_started")]
    assert ("p", 180 * S, "m/c/pod0") in starts
    # b0a immediately reuses pod1's free node at the same wake; b0b waits
    # for p's completion and restarts in pod0.
    assert ("b0a", 180 * S, "m/c") in starts
    assert ("b0b", 300 * S, "m/c") in starts
    assert sim._jobs["b1"].n_failures == 0
    assert {c[1] for c in sink.of("job_finished")} == {"b0a", "b0b", "b1", "p"}
    assert fleet.free_chips(CLUSTER) == 32


# ---------------------------------------------------------------------------
# Registry and params
# ---------------------------------------------------------------------------


def test_registry_and_params():
    assert "tiered_priority" in registered_schedulers()
    s = get_scheduler("tiered_priority")
    assert isinstance(s, TieredPriorityScheduler)
    assert s.preempt_mode is PreemptMode.REQUEUE
    assert s.max_preemptions_per_wake == 512  # v0.2 default (multi-pod reclaim)
    assert isinstance(s.placement, FirstFit)
    s2 = get_scheduler(
        "tiered_priority", {"preempt": "cancel", "max_preemptions_per_wake": 3}
    )
    assert s2.preempt_mode is PreemptMode.CANCEL
    assert s2.max_preemptions_per_wake == 3
    assert TieredPriorityScheduler(preempt=PreemptMode.CANCEL).preempt_mode is (
        PreemptMode.CANCEL
    )
    with pytest.raises(ValueError, match="preempt"):
        TieredPriorityScheduler(preempt="suspend")
    with pytest.raises(ValueError, match="max_preemptions_per_wake"):
        TieredPriorityScheduler(max_preemptions_per_wake=-1)
