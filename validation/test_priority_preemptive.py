"""Validation rung (traffic-math.md §5.2): preemptive-resume priority
M/M/1 per-class sojourn times, plus best-effort backlog shielding.

TWO-CLASS FORMULA (Adan & Resing ch. 11).  On one 1-chip node with
Poisson classes at lambda_1 (PROD) and lambda_2 (BATCH), both Exp(mu)
service, preemptive-resume priority gives

    E[T_1] = (1/mu) / (1 - rho_1)
    E[T_2] = (1/mu) / [(1 - rho_1) * (1 - rho)],    rho = rho_1 + rho_2

fleetsim setup: ``tiered_priority`` + REQUEUE, hand-built jobs (via
``ListSource``) with ``restart_overhead_s = 0``, ``checkpoint_save_s =
0`` (zero-length preemption grace) and ``checkpoint_interval_s = 1``
(continuous checkpointing => at most 1 s of work lost per preemption —
the engine's resume semantics; ``interval = 0`` would DISABLE
checkpointing and lose the whole stint).  PROD preempts BATCH; requeued
BATCH keeps its original submit time and its checkpointed progress, so
the discipline is preemptive-resume with FCFS inside each band.

TOLERANCES (two-stage, same pattern as test_pk_mg1).  The class-2
denominator (1-sigma_1)(1-sigma_2) amplifies service-moment sampling
scatter ~3.3x: at this n the realized sample runs rho_hat ~= 0.717 vs
the nominal 0.70 and class-2 sojourns land ~10% above the nominal
formula — moment noise, not engine bias.  So:

1. SHARP assert (5%): the general M/G/1 preemptive-resume formula
   (E[W_i] = sum_{j<=i} rho_j E[R_j] / [(1-sigma_{i-1})(1-sigma_i)],
   E[R_j] = E[B_j^2]/(2 E[B_j]); Adan & Resing ch. 11) evaluated at the
   REALIZED window rates and service moments — moment noise cancels;
   the pinned seed measures ~1.5% (class 1) / ~2.5% (class 2).  The
   residual is the round-alignment premium (each stint start waits for
   a round boundary; ~+0.4% effective service at round = 15 s,
   amplified by the denominators) plus queue-dynamics noise.
2. Analytic M/M/1 asserts: class 1 at 6% (mild amplification), class 2
   at 15% (the ~3.3x moment-noise amplification, documented above).
   The doc's ideal ~2% would need ~10x the completions; this rung
   trades that for CI runtime while still pinning both formulas and the
   shielding property.

SHIELDING (traffic-math.md §2.1, closed-loop backlog): a saturating
best-effort backlog (band 0) must leave PROD sojourns at their
SOLO M/M/1 value E[T] = (1/mu)/(1 - rho_prod) while occupancy -> ~1.
This runs the real ``SyntheticSource`` closed loop (``process:
closed_loop``), so it also validates the v0.2 refill machinery under
preemption.  Best-effort mean wait is asserted on NOTHING — it is
undefined under saturation (the doc's contract); BE goodput merely has
to be nonzero (the backlog does run when PROD is idle).

UNITS: engine times int microseconds; ``*_s`` float seconds.
Deterministic: numpy ``default_rng`` with fixed seeds, no wall clock.
"""

import numpy as np

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.model import GangSpec, Job, JobClass, Tier
from fleetsim.schedulers.base import get_scheduler
from fleetsim.schedulers.tiered_priority import TieredPriorityScheduler
from fleetsim.workload.base import ListSource
from fleetsim.workload.synthetic import SyntheticSource

US = 1_000_000

# -- two-class preemptive-resume M/M/1 --------------------------------------

E_S = 1800.0  # Exp(mu) mean service, both classes (seconds)
RHO_1 = 0.25  # PROD offered load
RHO_2 = 0.45  # BATCH offered load  (rho = 0.70)
HORIZON_DAYS = 146
ROUND = "15s"

LAM_1 = RHO_1 / E_S  # 0.5/h
LAM_2 = RHO_2 / E_S  # 0.9/h


def _mk_scenario(horizon_days: int) -> object:
    return load_scenario(
        {
            "sim": {"horizon": f"{horizon_days}d", "round": ROUND, "seed": 0},
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
            "failure_model": {
                "node_mtbf_days": 0,
                "maintenance_rate_per_node_month": 0,
            },
            # Placeholder class: the run feeds a ListSource instead.
            "workload": {
                "kind": "synthetic",
                "classes": {
                    "eval": {
                        "rate_per_hour": 1,
                        "chips": "fixed[1]",
                        "duration": "exponential[mean=600s]",
                    }
                },
            },
            "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
        }
    )


def _poisson_class(
    rng: np.random.Generator,
    prefix: str,
    lam_per_s: float,
    mean_service_s: float,
    tier: Tier,
    horizon_s: float,
) -> list[Job]:
    """Hand-built Poisson stream with Exp service and resume semantics
    (continuous 1 s checkpoints, zero save/restart overhead)."""
    jobs: list[Job] = []
    t = 0.0
    i = 0
    while True:
        t += rng.exponential(1.0 / lam_per_s)
        if t >= horizon_s:
            return jobs
        jobs.append(
            Job(
                id=f"{prefix}-{i:06d}",
                tenant="t0",
                job_class=JobClass.EVAL,
                submit_t=int(round(t * US)),
                gangs=[GangSpec(chips=1, chip_type="x")],
                tier=tier,
                preemptible=True,
                min_runtime_s=0.0,
                true_duration_s=float(rng.exponential(mean_service_s)),
                checkpoint_interval_s=1.0,  # continuous ckpt => resume
                checkpoint_save_s=0.0,  # zero-length preemption grace
                restart_overhead_s=0.0,
            )
        )
        i += 1


def _sojourns(collector: MetricsCollector, prefix: str) -> list[float]:
    """COMPLETED jobs' sojourn times (s) for arrivals inside the
    steady-state window."""
    w0, w1 = collector.window
    return [
        r["jct_s"]
        for r in collector.job_rows()
        if r["job_id"].startswith(prefix)
        and r["status"] == "COMPLETED"
        and w0 <= r["submit_t_us"] <= w1
    ]


def _empirical_expectations(
    collector: MetricsCollector, prod: list[Job], batch: list[Job]
) -> tuple[float, float]:
    """(E[T_1], E[T_2]) from the general M/G/1 preemptive-resume formula
    evaluated at the REALIZED window arrival rates and service moments."""
    w0, w1 = collector.window
    window_s = (w1 - w0) / 1e6

    def moments(jobs: list[Job]) -> tuple[float, float, float]:
        b = np.array(
            [j.true_duration_s for j in jobs if w0 <= j.submit_t <= w1]
        )
        return len(b) / window_s, float(b.mean()), float((b * b).mean())

    lam1, b1, b1sq = moments(prod)
    lam2, b2, b2sq = moments(batch)
    rho1, rho2 = lam1 * b1, lam2 * b2
    r1, r2 = b1sq / (2.0 * b1), b2sq / (2.0 * b2)
    sigma1, sigma2 = rho1, rho1 + rho2
    t1 = (rho1 * r1) / (1.0 - sigma1) + b1
    t2 = (rho1 * r1 + rho2 * r2) / ((1.0 - sigma1) * (1.0 - sigma2)) + b2 / (
        1.0 - sigma1
    )
    return t1, t2


def test_two_class_preemptive_resume_waits_match_formula():
    scn = _mk_scenario(HORIZON_DAYS)
    horizon_s = HORIZON_DAYS * 86400.0
    rng = np.random.default_rng(7)
    prod = _poisson_class(rng, "p", LAM_1, E_S, Tier.PROD, horizon_s)
    batch = _poisson_class(rng, "b", LAM_2, E_S, Tier.BATCH, horizon_s)
    fleet = build_fleet(scn)
    collector = MetricsCollector.from_scenario(scn, fleet)
    Simulator(
        scn,
        fleet,
        ListSource(prod + batch),
        TieredPriorityScheduler(),
        collector,
    ).run()
    fleet.check_invariants()

    t1 = _sojourns(collector, "p-")
    t2 = _sojourns(collector, "b-")
    assert len(t1) > 1200 and len(t2) > 2200, "window too thin"
    m1 = sum(t1) / len(t1)
    m2 = sum(t2) / len(t2)

    # 1. Sharp assert: the general formula at the realized moments.
    emp_t1, emp_t2 = _empirical_expectations(collector, prod, batch)
    err1 = abs(m1 - emp_t1) / emp_t1
    err2 = abs(m2 - emp_t2) / emp_t2
    assert err1 < 0.05, (
        f"class-1 E[T] {m1:.0f}s vs realized-moment formula {emp_t1:.0f}s"
        f" (rel err {err1:.1%}, n={len(t1)})"
    )
    assert err2 < 0.05, (
        f"class-2 E[T] {m2:.0f}s vs realized-moment formula {emp_t2:.0f}s"
        f" (rel err {err2:.1%}, n={len(t2)})"
    )

    # 2. Analytic M/M/1 asserts (moment-noise-amplified tolerances, see
    #    module docstring).
    exp_t1 = E_S / (1.0 - RHO_1)
    exp_t2 = E_S / ((1.0 - RHO_1) * (1.0 - RHO_1 - RHO_2))
    aerr1 = abs(m1 - exp_t1) / exp_t1
    aerr2 = abs(m2 - exp_t2) / exp_t2
    assert aerr1 < 0.06, (
        f"class-1 E[T] {m1:.0f}s vs analytic {exp_t1:.0f}s"
        f" (rel err {aerr1:.1%})"
    )
    assert aerr2 < 0.15, (
        f"class-2 E[T] {m2:.0f}s vs analytic {exp_t2:.0f}s"
        f" (rel err {aerr2:.1%})"
    )

    # The preemptive path must actually have been exercised.
    n_preempted = sum(
        1
        for r in collector.job_rows()
        if r["job_id"].startswith("b-") and r["n_preemptions"] > 0
    )
    assert n_preempted > 200, f"only {n_preempted} batch jobs were preempted"


# -- shielding: PROD unaffected by a saturating best-effort backlog ---------

SHIELD_E_S = 600.0
SHIELD_RHO_PROD = 0.5
SHIELD_SCENARIO = {
    "sim": {"horizon": "42d", "round": ROUND, "seed": 3},
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
            # PROD M/M/1 at rho = 0.5 (never preempted: nothing outranks it).
            "prod": {
                "class": "finetune",
                "rate_per_hour": 3600.0 * SHIELD_RHO_PROD / SHIELD_E_S,
                "chips": "fixed[1]",
                "duration": f"exponential[mean={SHIELD_E_S}s]",
                "tier": "prod",
                "checkpoint_interval": "0s",
                "abort_prob": 0,
            },
            # Closed-loop best-effort backlog: always 2 jobs pending.
            "filler": {
                "class": "finetune",
                "arrival": {
                    "process": "closed_loop",
                    "closed_loop": {"target_pending": 2},
                },
                "chips": "fixed[1]",
                "duration": f"exponential[mean={SHIELD_E_S}s]",
                "checkpoint_interval": "0s",
                "abort_prob": 0,
            },
        },
    },
    "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
}


def test_backlog_shields_prod_and_saturates_occupancy():
    scn = load_scenario(SHIELD_SCENARIO)
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    collector = MetricsCollector.from_scenario(scn, fleet)
    Simulator(scn, fleet, source, scheduler, collector, rng=rng).run()
    fleet.check_invariants()

    # PROD sojourns match the SOLO M/M/1 value: the backlog is invisible
    # to it (preemptive shielding).
    t_prod = _sojourns(collector, "prod-")
    assert len(t_prod) > 2400, "window too thin"
    expected = SHIELD_E_S / (1.0 - SHIELD_RHO_PROD)
    measured = sum(t_prod) / len(t_prod)
    rel_err = abs(measured - expected) / expected
    assert rel_err < 0.06, (
        f"shielded PROD E[T] {measured:.0f}s vs solo M/M/1 {expected:.0f}s"
        f" (rel err {rel_err:.1%}, n={len(t_prod)})"
    )

    # ... while the backlog drives occupancy far above rho_prod.
    rep = collector.integral_report()["window"]
    occupancy = rep["allocated_chip_s"] / rep["healthy_chip_s"]
    assert occupancy > 0.93, f"occupancy {occupancy:.3f} not saturated"

    # BE goodput exists (the filler does run) but BE wait is asserted on
    # nothing — undefined under saturation by contract.
    be_productive = sum(
        r["productive_chip_s"]
        for r in collector.job_rows()
        if r["job_id"].startswith("filler-")
    )
    assert be_productive > 0.0
