"""Validation rung (traffic-math.md §5.1): Pollaczek-Khinchine M/G/1.

Setup: ONE single-chip node, Poisson arrivals, LOGNORMAL service times
with sigma = 0.8 (c_s^2 = e^(sigma^2) - 1 ~= 0.90 — genuinely
non-exponential, "heavy-tailed-ish", but CI-safe: the doc forbids
CI-testing mean waits past sigma >= 2), strict FIFO.  The measured
steady-state mean queue wait is checked against the exact P-K formula

    E[W] = lambda * E[S^2] / (2 * (1 - rho)),    rho = lambda * E[S]

with E[S] = e^(mu + sigma^2/2) and E[S^2] = e^(2mu + 2sigma^2) for the
lognormal.  This rung catches duration-sampler bias (the quantile
parameterization must deliver the implied second moment) and
event-engine wait-accounting bias under service-time variance — the
Erlang-C rung (test_mmc.py) only exercises exponential service.

ROUND QUANTIZATION (why the sharp assert models it).  fleetsim
schedules in rounds (DESIGN §6.1): when a job completes mid-round, the
next start waits for the round boundary, so inside a busy period every
service is effectively followed by an alignment gap U ~ Uniform(0,
round).  That is a service-time inflation E[S] -> E[S] + round/2, and
near rho = 0.7 the P-K denominator amplifies the +30 s (+3.6%) into
~+17% on E[W] — a real, documented property of round-driven scheduling,
not an accounting bug.  The sharp assert therefore evaluates P-K at the
window's REALIZED arrival rate and service moments with the alignment
folded in (S' = S + U, S independent of U):

    E[S']  = E[S] + r/2,      E[S'^2] = E[S^2] + r*E[S] + r^2/3

and must match within 5% (the pinned seed measures ~2%; the residual is
second-order alignment/queue-state correlation).  A bracket assert pins
the frictionless analytic formula from both sides: measured E[W] must
sit ABOVE it (quantization only adds wait) and below 1.35x (the
premium plus sampling scatter — E[S^2]'s sample mean has relative std
sqrt((e^(4 sigma^2) - 1)/n) ~= 2.5% here, amplified ~2.3x by the
denominator).

Modeling notes: checkpointing off (service == sampled duration), aborts
off, failures/maintenance off, fully deterministic (fixed seed; asserts
guard regressions, not sampling luck).

UNITS: rates are per-second floats; times are float seconds unless
suffixed ``_us``.
"""

import math

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.schedulers.base import get_scheduler
from fleetsim.workload.distributions import Z90  # the parser's pinned z_90
from fleetsim.workload.synthetic import SyntheticSource

MEDIAN_S = 600.0  # lognormal median => mu = ln(600)
SIGMA = 0.8
P90_S = MEDIAN_S * math.exp(Z90 * SIGMA)  # ~1672.8 s
RHO = 0.7
ROUND_S = 60.0

MU = math.log(MEDIAN_S)
E_S = math.exp(MU + SIGMA**2 / 2.0)  # ~826 s
E_S2 = math.exp(2.0 * MU + 2.0 * SIGMA**2)
LAMBDA_PER_S = RHO / E_S  # ~3.05 jobs/hour

SCENARIO = {
    # 273 days => ~2e4 arrivals at ~3/h; steady-state window is the
    # default middle 80%.
    "sim": {"horizon": "273d", "round": "60s", "seed": 11},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "x", "per_node": 1},
                "topology": {"levels": ["node"], "counts": [1]},
            }
        ],
    },
    "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
    "workload": {
        "kind": "synthetic",
        "classes": {
            "eval": {
                "rate_per_hour": 3600.0 * LAMBDA_PER_S,
                "chips": "fixed[1]",
                "duration": f"lognormal[median={MEDIAN_S}s, p90={P90_S:.6f}s]",
                "checkpoint_interval": "0s",  # service == sampled duration
                "abort_prob": 0,
            }
        },
    },
    "scheduler": {"name": "fifo", "params": {"strict": True}},
}


def pk_wait_s(lam: float, e_s: float, e_s2: float) -> float:
    """The exact P-K mean queue wait (seconds); requires rho < 1."""
    rho = lam * e_s
    assert rho < 1.0
    return lam * e_s2 / (2.0 * (1.0 - rho))


def _run() -> MetricsCollector:
    scn = load_scenario(SCENARIO)
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    collector = MetricsCollector.from_scenario(scn, fleet)
    Simulator(scn, fleet, source, scheduler, collector, rng=rng).run()
    return collector


def _window_rows(collector: MetricsCollector) -> list[dict]:
    w0, w1 = collector.window
    return [
        r
        for r in collector.job_rows()
        if r["first_start_t_us"] is not None
        and w0 <= r["first_start_t_us"] <= w1
    ]


def test_pk_mg1_mean_wait_matches_formula():
    collector = _run()
    rows = _window_rows(collector)
    assert len(rows) > 15000, "steady-state window too thin for a stable mean"
    measured = sum(r["queue_wait_s"] for r in rows) / len(rows)

    # Sharp assert: P-K at the realized moments with the round-alignment
    # service inflation folded in (see module docstring).
    services = [
        r["jct_s"] - r["queue_wait_s"] for r in rows if r["jct_s"] is not None
    ]
    assert len(services) == len(rows), "window job still open at horizon"
    e_s = sum(services) / len(services)
    e_s2 = sum(s * s for s in services) / len(services)
    w0, w1 = collector.window
    lam = len(rows) / ((w1 - w0) / 1e6)
    r = ROUND_S
    expected = pk_wait_s(lam, e_s + r / 2.0, e_s2 + r * e_s + r * r / 3.0)
    rel_err = abs(measured - expected) / expected
    assert rel_err < 0.05, (
        f"M/G/1 mean wait {measured:.1f}s vs alignment-corrected P-K"
        f" {expected:.1f}s (rel err {rel_err:.1%}, n={len(rows)})"
    )

    # Bracket assert against the frictionless analytic formula: the
    # quantization premium is strictly upward, and bounded (~17% at this
    # rho plus <= ~6% amplified sampling scatter).
    frictionless = pk_wait_s(LAMBDA_PER_S, E_S, E_S2)
    assert frictionless < measured < 1.35 * frictionless, (
        f"M/G/1 mean wait {measured:.1f}s outside"
        f" [1.0, 1.35] x analytic P-K {frictionless:.1f}s"
    )


def test_pk_mg1_utilization_matches_offered_load():
    """Secondary sanity: measured occupancy ~= rho (pins E[S] too)."""
    collector = _run()
    rep = collector.integral_report()["window"]
    occupancy = rep["allocated_chip_s"] / rep["healthy_chip_s"]
    assert abs(occupancy - RHO) / RHO < 0.05, (
        f"occupancy {occupancy:.3f} vs rho {RHO:.3f}"
    )
