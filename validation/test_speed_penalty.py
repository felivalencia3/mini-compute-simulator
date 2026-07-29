"""Validation rung (v0.4): speed-penalty completion identity.

The ``penalties.xover`` cost model is analytic: a placement that spans
more than one domain at a configured level runs at that level's
multiplier, several configured levels multiply, and — with checkpointing
off (``eff = 1``) and no restarts — the engine schedules completion at
exactly ``start + true_duration / speed`` (int µs, round-half-even).
These tests pin the identity on a deterministic mini-fleet (2 pods x 2
nodes x 8 chips = 32 chips), including the relax-after-timeout start
time and the segmented (cross-pod) case that v0.3 deferred to this cost
model.  No RNG stream is consulted anywhere (failures off, ListSource).
"""

from fleetsim.config import load_scenario
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.model import Constraint, GangSpec, Job, JobClass, Tier
from fleetsim.schedulers.base import get_scheduler
from fleetsim.workload.base import ListSource

S = 1_000_000  # one second in microseconds


def make_scenario(penalties, horizon="2h"):
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": "60s", "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["pod", "node"], "counts": [2, 2]},
                    }
                ],
            },
            "failure_model": {
                "node_mtbf_days": 0,
                "maintenance_rate_per_node_month": 0,
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
            "penalties": penalties,
        }
    )


def mk_job(jid, chips, dur_s, within=None, required=True, relax_after=0.0,
           segments=None):
    return Job(
        id=jid,
        tenant="t0",
        job_class=JobClass.PRETRAIN,
        submit_t=0,
        gangs=[
            GangSpec(
                chips=chips,
                chip_type="h100",
                within=(
                    Constraint(
                        level=within, required=required, relax_after_s=relax_after
                    )
                    if within
                    else None
                ),
                segments=segments,
            )
        ],
        tier=Tier.PROD,
        true_duration_s=dur_s,
        checkpoint_interval_s=0.0,  # eff = 1: the identity is exact
        checkpoint_save_s=0.0,
        restart_overhead_s=0.0,
    )


class ProbeSink:
    """Records starts (with the alloc's relaxed flag) and finishes."""

    def __init__(self):
        self.starts = {}  # job id -> (t, relaxed)
        self.finishes = {}  # job id -> (t, status name)

    def job_started(self, job, alloc, t):
        self.starts[job.id] = (t, any(g.relaxed for g in alloc.gangs))

    def job_finished(self, job, t, status, productive_chip_s, lost_chip_s):
        self.finishes[job.id] = (t, status.name)

    def __getattr__(self, name):  # every other sink callback: no-op
        return lambda *a, **k: None


def run(scenario, jobs):
    fleet = build_fleet(scenario)
    sink = ProbeSink()
    sim = Simulator(
        scenario, fleet, ListSource(jobs), get_scheduler("fifo", {}), sink
    )
    sim.run()
    return sink


def test_within_one_pod_runs_at_speed_1():
    scenario = make_scenario({"xover": {"pod": 0.5}})
    sink = run(scenario, [mk_job("a", 16, 1000.0, within="pod")])
    assert sink.starts["a"] == (0, False)
    assert sink.finishes["a"] == (1000 * S, "COMPLETED")  # dur / 1.0, exact


def test_relaxed_cross_pod_placement_pays_the_penalty_exactly():
    # 24 chips can never fit one 16-chip pod; relax_after=0 relaxes at
    # the first wake; the placement spans both pods -> speed 0.5.
    scenario = make_scenario({"xover": {"pod": 0.5}})
    sink = run(
        scenario,
        [mk_job("a", 24, 1000.0, within="pod", required=False, relax_after=0.0)],
    )
    assert sink.starts["a"] == (0, True)  # relaxed placement, marked
    assert sink.finishes["a"] == (2000 * S, "COMPLETED")  # dur / 0.5, exact


def test_relax_after_timeout_gates_the_start():
    # relax_after=300s: constrained searches fail at wakes 0..240s; the
    # 300s wake is the first where the relaxed retry is allowed.
    scenario = make_scenario({"xover": {"pod": 0.5}})
    sink = run(
        scenario,
        [mk_job("a", 24, 1000.0, within="pod", required=False, relax_after=300.0)],
    )
    assert sink.starts["a"] == (300 * S, True)
    assert sink.finishes["a"] == ((300 + 2000) * S, "COMPLETED")


def test_multiplicative_levels():
    # Spans nodes (4 > 1 node) AND pods -> 0.5 * 0.8 = 0.4 -> 2500 s.
    scenario = make_scenario({"xover": {"pod": 0.5, "node": 0.8}})
    sink = run(
        scenario,
        [mk_job("a", 32, 1000.0, within="pod", required=False, relax_after=0.0)],
    )
    assert sink.starts["a"] == (0, True)
    assert sink.finishes["a"] == (2500 * S, "COMPLETED")


def test_single_leaf_gang_never_pays_a_node_crossing():
    scenario = make_scenario({"xover": {"pod": 0.5, "node": 0.8}})
    sink = run(scenario, [mk_job("a", 8, 1000.0)])  # one whole node
    assert sink.finishes["a"] == (1000 * S, "COMPLETED")


def test_segmented_cross_pod_gang_pays_the_penalty():
    # The v0.3-deferred cross-segment penalty: 32 chips as 2 x 2-node
    # pod-segments necessarily spans both pods -> speed 0.5.
    scenario = make_scenario({"xover": {"pod": 0.5}})
    sink = run(
        scenario, [mk_job("a", 32, 1000.0, segments=(2, "pod"))]
    )
    assert sink.starts["a"] == (0, False)  # segmented, not relaxed
    assert sink.finishes["a"] == (2000 * S, "COMPLETED")


def test_no_penalties_section_means_speed_is_exactly_1():
    scenario = make_scenario(None)
    sink = run(
        scenario, [mk_job("a", 32, 1000.0, segments=(2, "pod"))]
    )
    assert sink.finishes["a"] == (1000 * S, "COMPLETED")
