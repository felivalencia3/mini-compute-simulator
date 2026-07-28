"""Validation rung 2 (DESIGN §12): invariant/property checks on a
mid-size run with failures, maintenance drains, and tiered-priority
preemption all active.

A probe sink (wrapping the real MetricsCollector) checks invariants at
every sink callback and at every metrics flush:

- gang atomicity: every started job's allocation covers exactly its
  gang's chip count — never partial;
- accounting: chips_allocated/chips_freed deltas equal the fleet tree's
  own used-chip counters; healthy_delta events reproduce the fleet's
  healthy capacity; no chip is double-booked (``fleet.check_invariants``);
- capacity: allocated chips on HEALTHY leaves never exceed healthy
  capacity, and total allocated never exceeds total capacity.  (The
  stricter "allocated <= healthy" does NOT hold by design: jobs keep
  their allocation on a DRAINING node through the drain grace window and
  on a FAILED node until the engine releases each victim.)

Post-run properties: chip-hour conservation (the allocated integral
equals the per-job chips x running-elapsed sum), every preempted job is
eventually requeued or terminal, and summary sanity (occupancy <= 1,
goodput <= 1, allocation_rate <= occupancy).
"""

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.fleet.tree import FleetTree
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.metrics.summary import build_summary
from fleetsim.model import NodeState
from fleetsim.schedulers.base import get_scheduler
from fleetsim.workload.synthetic import SyntheticSource

HORIZON_US = 2 * 24 * 3600 * 1_000_000

SCENARIO = {
    "sim": {"horizon": "2d", "round": "60s", "seed": 7},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "h100", "per_node": 8},
                # 2 racks x 8 nodes x 8 chips = 128 chips
                "topology": {"levels": ["rack", "node"], "counts": [2, 8]},
            }
        ],
    },
    "failure_model": {
        "node_mtbf_days": 2.0,  # deliberately hot: ~8 failures/day expected
        "repair_auto_min": [30, 60],
        "repair_manual_frac": 0.1,
        "repair_manual_days": [0.1, 0.2],
        "maintenance_rate_per_node_month": 3.0,
        "drain_grace": "10m",
    },
    "workload": {
        "kind": "synthetic",
        "classes": {
            "eval": {
                "rate_per_hour": 60,
                "chips": "pow2[1, 8]",
                "duration": "lognormal[median=2m, p90=30m]",
                "tier": "batch",
                "abort_prob": 0.2,
            },
            "finetune": {
                "rate_per_hour": 2,
                "chips": "pow2[8, 32]",
                "duration": "lognormal[median=30m, p90=2h]",
                "tier": "batch",
            },
            "pretrain": {
                "rate_per_day": 3,
                "chips": "pow2[32, 64]",
                "duration": "lognormal[median=1h, p90=6h]",
                "tier": "prod",
                "min_runtime": "10m",
                "checkpoint_interval": "30m",
                "within": "rack",
            },
        },
    },
    "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
}


class ProbeSink:
    """MetricsSink that asserts invariants on every callback, then
    forwards to the wrapped collector."""

    def __init__(self, inner: MetricsCollector, fleet: FleetTree, horizon_us: int):
        self.inner = inner
        self.fleet = fleet
        self.horizon_us = horizon_us
        self.allocated = 0  # tracked from chips_allocated/chips_freed
        self.healthy = sum(fleet.domain(l).chips for l in fleet.leaves())
        self.open_preemptions: dict[str, tuple[int, float]] = {}  # id -> (t, grace_s)
        self.preempt_triggers: dict[str, int] = {}
        self.requeues = 0
        self.n_flush_checks = 0

    # -- helpers ---------------------------------------------------------

    def _fleet_used(self) -> tuple[int, int, int]:
        """(allocated total, allocated on HEALTHY leaves, healthy chips)."""
        total = on_healthy = healthy = 0
        for lid in self.fleet.leaves():
            leaf = self.fleet.domain(lid)
            used = self.fleet.used_chips(lid)
            total += used
            if leaf.state is NodeState.HEALTHY:
                on_healthy += used
                healthy += leaf.chips
        return total, on_healthy, healthy

    def _gang_chips(self, alloc) -> int:
        chips = 0
        for gang in alloc.gangs:
            if isinstance(gang.nodes, dict):
                chips += sum(gang.nodes.values())
            else:  # whole-node list form: the leaf's full capacity each
                chips += sum(self.fleet.domain(l).chips for l in gang.nodes)
        return chips

    # -- MetricsSink protocol -------------------------------------------

    def job_submitted(self, job, t):
        self.inner.job_submitted(job, t)

    def job_admitted(self, job, t):
        self.inner.job_admitted(job, t)

    def job_started(self, job, alloc, t):
        spec_chips = sum(g.chips for g in job.gangs)
        assert self._gang_chips(alloc) == spec_chips, (
            f"gang atomicity violated for {job.id}: allocation covers"
            f" {self._gang_chips(alloc)} chips, spec wants {spec_chips}"
        )
        self.inner.job_started(job, alloc, t)

    def job_preempted(self, job, t, trigger):
        self.preempt_triggers[trigger] = self.preempt_triggers.get(trigger, 0) + 1
        self.open_preemptions[job.id] = (t, job.checkpoint_save_s)
        self.inner.job_preempted(job, t, trigger)

    def job_requeued(self, job, t):
        self.requeues += 1
        self.open_preemptions.pop(job.id, None)
        self.inner.job_requeued(job, t)

    def job_progress(self, job, start_us, end_us, productive_chip_s, lost_chip_s):
        assert start_us <= end_us
        assert productive_chip_s >= 0 and lost_chip_s >= 0
        # Surviving work in a stint can never exceed the stint's chip-time
        # (slack: completion times are rounded to int microseconds).
        chips = sum(g.chips for g in job.gangs)
        assert productive_chip_s <= (end_us - start_us) / 1e6 * chips + chips * 1e-6
        self.inner.job_progress(job, start_us, end_us, productive_chip_s, lost_chip_s)

    def job_finished(self, job, t, status, productive_chip_s, lost_chip_s):
        assert productive_chip_s >= 0 and lost_chip_s >= 0
        self.open_preemptions.pop(job.id, None)
        self.inner.job_finished(job, t, status, productive_chip_s, lost_chip_s)

    def node_failed(self, node_id, t, killed_alloc_ids, cause="unknown"):
        assert list(killed_alloc_ids) == sorted(killed_alloc_ids)
        assert cause in {"gpu_hbm", "network", "software", "other", "unknown"}
        self.inner.node_failed(node_id, t, killed_alloc_ids, cause=cause)

    def node_repaired(self, node_id, t):
        self.inner.node_repaired(node_id, t)

    def node_drain_started(self, node_id, t):
        self.inner.node_drain_started(node_id, t)

    def chips_allocated(self, n, chip_type, t):
        assert n > 0
        self.allocated += n
        self.inner.chips_allocated(n, chip_type, t)

    def chips_freed(self, n, chip_type, t):
        assert n > 0
        self.allocated -= n
        assert self.allocated >= 0, "freed more chips than were allocated"
        self.inner.chips_freed(n, chip_type, t)

    def healthy_delta(self, n_chips, chip_type, t):
        self.healthy += n_chips
        assert self.healthy >= 0
        self.inner.healthy_delta(n_chips, chip_type, t)

    def flush(self, t, fleet, n_pending, n_running):
        fleet.check_invariants()  # no double-booking, counters consistent
        total_used, on_healthy, healthy = self._fleet_used()
        assert total_used == self.allocated, (
            f"t={t}: sink-tracked allocated {self.allocated} !="
            f" fleet used {total_used}"
        )
        assert healthy == self.healthy, (
            f"t={t}: healthy_delta-tracked {self.healthy} !="
            f" fleet healthy {healthy}"
        )
        assert on_healthy <= healthy, (
            f"t={t}: allocated on healthy leaves {on_healthy} >"
            f" healthy capacity {healthy}"
        )
        assert total_used <= 128, f"t={t}: allocated {total_used} > total 128"
        assert n_pending >= 0 and n_running >= 0
        self.n_flush_checks += 1
        self.inner.flush(t, fleet, n_pending, n_running)


def run_probed() -> tuple[ProbeSink, MetricsCollector]:
    scn = load_scenario(SCENARIO)
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    collector = MetricsCollector.from_scenario(scn, fleet)
    probe = ProbeSink(collector, fleet, scn.sim.horizon_us)
    Simulator(scn, fleet, source, scheduler, probe, rng=rng).run()
    return probe, collector


def test_invariants_hold_under_failures_and_preemption():
    probe, collector = run_probed()

    # The scenario actually exercised the machinery.
    counts = collector.event_counts()["full"]
    assert probe.n_flush_checks > 1000
    assert counts["node_failures"] > 0, "no failures fired"
    assert counts["drains_started"] > 0, "no maintenance drains fired"
    assert sum(probe.preempt_triggers.values()) > 0, "no preemptions fired"
    assert probe.requeues > 0

    # Every preemption resolved (requeued or terminal), unless its grace
    # window was still open at the horizon.
    for jid, (t, grace_s) in probe.open_preemptions.items():
        assert t + int(grace_s * 1e6) > probe.horizon_us, (
            f"job {jid} preempted at t={t} never requeued/finished"
        )

    # Chip-hour conservation: allocated integral == sum of per-job
    # running chip-time (both derived from the same event stream).
    rep = collector.integral_report()["full"]
    per_job = sum(
        r["chips"] * r["running_elapsed_s"] for r in collector.job_rows()
    )
    assert abs(rep["allocated_chip_s"] - per_job) <= max(
        1e-6 * per_job, 1e-3
    ), f"allocated integral {rep['allocated_chip_s']} != per-job sum {per_job}"

    # Summary sanity.
    summary = build_summary(collector)
    for scope in ("full", "window"):
        s = summary[scope]
        assert s["occupancy"] is not None and 0.0 < s["occupancy"] <= 1.0
        assert s["allocation_rate"] <= s["occupancy"] + 1e-12
        assert s["goodput"] is not None and 0.0 <= s["goodput"] <= 1.0
        assert s["counts"]["jobs_finished"] > 0
    # Failure kills are surfaced separately from scheduler/maintenance
    # preemptions and both paths ran.
    assert summary["full"]["counts"]["failure_kills"] > 0
    assert probe.preempt_triggers.get("scheduler", 0) > 0, (
        "tiered_priority never emitted a scheduler preemption"
    )
