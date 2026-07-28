"""Validation rung 1 (DESIGN §12): an M/M/c scenario matches Erlang-C.

Setup: c = 4 single-chip nodes, 1-chip jobs, Poisson arrivals at rate
lambda, exponential service times with rate mu, strict FIFO — a textbook
M/M/c queue.  The measured steady-state mean queue wait must be within
10% of the analytic Erlang-C value

    W_q = C(c, a) / (c*mu - lambda),   a = lambda/mu.

Modeling notes: checkpointing is disabled (interval 0) so service time ==
sampled duration; failures/maintenance are off; the 60 s scheduler round
adds ~30 s of quantization to each wait (~1.6% of W_q here), well inside
the tolerance.  The run is fully deterministic (fixed seed).
"""

import math

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.schedulers.base import get_scheduler
from fleetsim.workload.synthetic import SyntheticSource

C_NODES = 4
LAMBDA_PER_S = 3.0 / 3600.0  # 3 jobs/hour
MU_PER_S = 1.0 / 3600.0  # mean service 3600 s

SCENARIO = {
    "sim": {"horizon": "42d", "round": "60s", "seed": 1},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "x", "per_node": 1},
                "topology": {"levels": ["node"], "counts": [C_NODES]},
            }
        ],
    },
    # No failures, no maintenance: pure queueing.
    "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
    "workload": {
        "kind": "synthetic",
        "classes": {
            "eval": {
                "rate_per_hour": 3600.0 * LAMBDA_PER_S,
                "chips": "fixed[1]",
                "duration": "exponential[mean=3600s]",
                "checkpoint_interval": "0s",  # service == sampled duration
                "abort_prob": 0,  # opt out of the 30% abort default
            }
        },
    },
    "scheduler": {"name": "fifo", "params": {"strict": True}},
}


def erlang_c(c: int, a: float) -> float:
    """Probability an arrival waits (Erlang-C), offered load a = lambda/mu < c."""
    assert 0 < a < c
    s = sum(a**k / math.factorial(k) for k in range(c))
    top = (a**c / math.factorial(c)) * (c / (c - a))
    return top / (s + top)


def analytic_wq_s() -> float:
    a = LAMBDA_PER_S / MU_PER_S
    return erlang_c(C_NODES, a) / (C_NODES * MU_PER_S - LAMBDA_PER_S)


def test_mmc_mean_queue_wait_matches_erlang_c():
    scn = load_scenario(SCENARIO)
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    collector = MetricsCollector.from_scenario(scn, fleet)
    Simulator(scn, fleet, source, scheduler, collector, rng=rng).run()

    w0, w1 = collector.window
    waits = [
        r["queue_wait_s"]
        for r in collector.job_rows()
        if r["first_start_t_us"] is not None and w0 <= r["first_start_t_us"] <= w1
    ]
    assert len(waits) > 1000, "steady-state window too thin for a stable mean"
    measured = sum(waits) / len(waits)
    expected = analytic_wq_s()
    rel_err = abs(measured - expected) / expected
    assert rel_err < 0.10, (
        f"M/M/{C_NODES} mean wait {measured:.1f}s vs Erlang-C {expected:.1f}s"
        f" (rel err {rel_err:.1%}, n={len(waits)})"
    )


def test_mmc_utilization_matches_offered_load():
    """Secondary sanity: measured occupancy ~= rho = lambda/(c*mu)."""
    scn = load_scenario(SCENARIO)
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    collector = MetricsCollector.from_scenario(scn, fleet)
    Simulator(scn, fleet, source, scheduler, collector, rng=rng).run()

    rep = collector.integral_report()["window"]
    occupancy = rep["allocated_chip_s"] / rep["healthy_chip_s"]
    rho = LAMBDA_PER_S / (C_NODES * MU_PER_S)
    assert abs(occupancy - rho) / rho < 0.10, (
        f"occupancy {occupancy:.3f} vs rho {rho:.3f}"
    )
