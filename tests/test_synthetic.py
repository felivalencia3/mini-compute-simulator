"""SyntheticSource tests: arrival rates, the pinned diurnal curve,
quantization, tenant skew, outcomes, determinism, service expansion, and
an end-to-end engine smoke run."""

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.base import NullSink
from fleetsim.model import GangSpec, JobClass, JobStatus, Service, Tier
from fleetsim.schedulers.fifo import FIFOScheduler
from fleetsim.workload.services import expand_services
from fleetsim.workload.synthetic import (
    DIURNAL_MEAN,
    SyntheticJob,
    SyntheticSource,
    diurnal_multiplier,
    node_sizes,
    quantize_chips,
)

S = 1_000_000
HOUR = 3600 * S
DAY = 24 * HOUR


def make_scenario(classes, horizon="7d", n_nodes=2, per_node=8, seed=0):
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": "60s", "seed": seed},
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
            "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
            "workload": {"kind": "synthetic", "classes": classes},
        }
    )


EVAL = {
    "rate_per_hour": 40,
    "chips": "pow2[1, 8]",
    "duration": "lognormal[median=2m, p90=30m]",
}


def make_source(classes, horizon="7d", seed=0, overrides=None, **kw):
    scn = make_scenario(classes, horizon=horizon, seed=seed, **{k: v for k, v in kw.items() if k in ("n_nodes", "per_node")})
    fleet = build_fleet(scn)
    rng = RngStreams(seed, overrides)
    return SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us), scn, fleet


def drain(src):
    out = []
    while (nxt := src.next_arrival()) is not None:
        out.append(nxt)
    return out


# ---------------------------------------------------------------------------
# Arrival process
# ---------------------------------------------------------------------------


def test_poisson_rate_over_long_horizon():
    src, scn, _ = make_source({"eval": dict(EVAL)}, horizon="7d")
    arrivals = drain(src)
    expected = 40 * 24 * 7  # 6720
    assert abs(len(arrivals) - expected) / expected < 0.05
    # Times non-decreasing, strictly below the horizon, submit_t == time.
    times = [t for t, _ in arrivals]
    assert times == sorted(times)
    assert times[-1] < scn.sim.horizon_us
    assert all(j.submit_t == t for t, j in arrivals)
    # Exhaustion is sticky.
    assert src.next_arrival() is None
    assert src.next_arrival() is None


def test_diurnal_multiplier_curve_pinned():
    assert diurnal_multiplier(0) == 0.5
    assert diurnal_multiplier(6 * HOUR - 1) == 0.5
    assert diurnal_multiplier(6 * HOUR) == 1.0
    assert diurnal_multiplier(12 * HOUR) == 0.8
    assert diurnal_multiplier(13 * HOUR) == 1.0
    assert diurnal_multiplier(18 * HOUR + 30 * 60 * S) == 0.8
    assert diurnal_multiplier(19 * HOUR) == 1.0
    assert diurnal_multiplier(23 * HOUR) == 1.0
    assert diurnal_multiplier(DAY + 3 * HOUR) == 0.5  # wraps daily
    assert DIURNAL_MEAN == pytest.approx((6 * 0.5 + 2 * 0.8 + 16 * 1.0) / 24)


def test_diurnal_thinning_shapes_hourly_histogram():
    src, scn, _ = make_source(
        {"eval": dict(EVAL, rate_per_hour=60, diurnal=True)}, horizon="14d"
    )
    arrivals = drain(src)
    days = 14
    # Configured rate is the MEAN rate over a day.
    expected_total = 60 * 24 * days
    assert abs(len(arrivals) - expected_total) / expected_total < 0.05
    by_hour = [0] * 24
    for t, _ in arrivals:
        by_hour[(t % DAY) // HOUR] += 1
    plain_hours = [h for h in range(24) if not (h < 6 or h in (12, 18))]
    night = sum(by_hour[h] for h in range(6)) / 6
    dip = (by_hour[12] + by_hour[18]) / 2
    plain = sum(by_hour[h] for h in plain_hours) / len(plain_hours)
    assert abs(night / plain - 0.5) < 0.06
    assert abs(dip / plain - 0.8) < 0.10


def test_classes_merge_time_ordered():
    src, _, _ = make_source(
        {
            "eval": dict(EVAL),
            "finetune": {
                "rate_per_hour": 10,
                "chips": "pow2[8, 32]",
                "duration": "lognormal[median=4h, p90=24h]",
            },
        },
        horizon="2d",
    )
    arrivals = drain(src)
    times = [t for t, _ in arrivals]
    assert times == sorted(times)
    names = {j.id.rsplit("-", 1)[0] for _, j in arrivals}
    assert names == {"eval", "finetune"}
    # Per-class ids are sequential in emission order.
    eval_ids = [j.id for _, j in arrivals if j.id.startswith("eval-")]
    assert eval_ids == [f"eval-{i}" for i in range(len(eval_ids))]


# ---------------------------------------------------------------------------
# Quantization (DESIGN 4.1)
# ---------------------------------------------------------------------------


def test_quantize_rules():
    # Sub-node: round UP to the next power of two.
    assert quantize_chips(1, 8) == 1
    assert quantize_chips(2.0, 8) == 2
    assert quantize_chips(3, 8) == 4
    assert quantize_chips(2.5, 8) == 4  # ceil(2.5)=3 -> 4
    # Reaching the node size becomes one whole node.
    assert quantize_chips(5, 8) == 8
    assert quantize_chips(7, 8) == 8
    # At/above node size: round UP to whole-node multiples.
    assert quantize_chips(8, 8) == 8
    assert quantize_chips(9, 8) == 16
    assert quantize_chips(17, 8) == 24
    assert quantize_chips(64, 8) == 64
    # Tiny/zero samples floor at one chip.
    assert quantize_chips(0.01, 8) == 1
    # Non-power-of-two node sizes still work (e.g. NVL72-ish).
    assert quantize_chips(5, 6) == 6  # next pow2 (8) >= node -> one node
    assert quantize_chips(4, 6) == 4
    assert quantize_chips(7, 6) == 12
    with pytest.raises(ValueError):
        quantize_chips(1, 0)
    with pytest.raises(ValueError):
        quantize_chips(float("nan"), 8)


def test_emitted_sizes_are_quantized():
    src, _, _ = make_source(
        {"mix": dict(EVAL, **{"class": "finetune", "chips": "uniform[1, 40]"})},
        horizon="2d",
        n_nodes=8,
    )
    sizes = {j.gangs[0].chips for _, j in drain(src)}
    assert sizes  # non-empty
    for c in sizes:
        if c < 8:
            assert c & (c - 1) == 0, f"sub-node size {c} not a power of two"
        else:
            assert c % 8 == 0, f"size {c} not a whole-node multiple"
    assert any(c < 8 for c in sizes) and any(c > 8 for c in sizes)


def test_node_sizes_and_chip_type_pinning():
    _, _, fleet = make_source({"eval": dict(EVAL)})
    assert node_sizes(fleet) == {"h100": 8}
    src, _, _ = make_source({"eval": dict(EVAL)}, horizon="6h")
    for _, j in drain(src):
        assert j.gangs[0].chip_type == "h100"  # sole fleet type is pinned


# ---------------------------------------------------------------------------
# Job fields, tenants, outcomes
# ---------------------------------------------------------------------------


def test_job_fields_from_class_config():
    src, _, _ = make_source(
        {
            "pretrain": {
                "rate_per_hour": 2,
                "chips": "fixed[16]",
                "duration": "fixed[10h]",
                "tier": "prod",
                "min_runtime": "2h",
                "checkpoint_interval": "1h",
                "within": "node",
                "abort_prob": 0,  # keep true_duration_s untruncated
            }
        },
        horizon="2d",
    )
    arrivals = drain(src)
    assert arrivals
    seen_within = []
    for _, j in arrivals:
        assert isinstance(j, SyntheticJob)
        assert j.job_class is JobClass.PRETRAIN
        assert j.tier is Tier.PROD
        assert j.min_runtime_s == 7200.0
        assert j.checkpoint_interval_s == 3600.0
        assert j.gangs[0].chips == 16
        assert j.true_duration_s == 36000.0
        assert j.walltime_est_s > 0
        assert j.gangs[0].within.level == "node"
        seen_within.append(j.gangs[0].within)
    # Constraints are fresh objects per job, never shared.
    assert seen_within[0] is not seen_within[1]


def test_walltime_overestimate_factor():
    src, _, _ = make_source(
        {"eval": dict(EVAL, duration="fixed[600s]")}, horizon="14d"
    )
    factors = sorted(j.walltime_est_s / 600.0 for _, j in drain(src))
    med = factors[len(factors) // 2]
    assert abs(med - 1.5) / 1.5 < 0.10  # lognormal median 1.5
    p90 = factors[int(len(factors) * 0.9)]
    assert abs(p90 - 3.0) / 3.0 < 0.20  # p90 ~ 3


def test_tenant_zipf_skew():
    src, _, _ = make_source({"eval": dict(EVAL)}, horizon="14d")
    counts = {}
    for _, j in drain(src):
        counts[j.tenant] = counts.get(j.tenant, 0) + 1
    assert set(counts) <= {f"t{i}" for i in range(8)}
    total = sum(counts.values())
    # t0 dominant, ranks roughly monotone.
    assert counts["t0"] == max(counts.values())
    assert counts["t0"] / total > 0.35
    assert counts["t0"] > counts["t1"] > counts["t3"]


def test_abort_outcomes():
    src, _, _ = make_source(
        {"eval": dict(EVAL, duration="fixed[600s]", abort_prob=1.0)}, horizon="7d"
    )
    jobs = [j for _, j in drain(src)]
    assert jobs
    statuses = [j.terminal_status_override for j in jobs]
    assert all(s in (JobStatus.FAILED, JobStatus.CANCELED) for s in statuses)
    failed_frac = statuses.count(JobStatus.FAILED) / len(statuses)
    assert 0.75 < failed_frac < 0.85  # 80/20 split
    # Truncated early: U^2 fraction of the sampled duration.
    fracs = [j.true_duration_s / 600.0 for j in jobs]
    assert all(0.0 <= f <= 1.0 for f in fracs)
    mean_frac = sum(fracs) / len(fracs)
    assert abs(mean_frac - 1 / 3) < 0.03  # E[U^2] = 1/3
    assert sum(1 for f in fracs if f < 0.25) > len(fracs) * 0.4  # skewed early


def test_no_aborts_when_prob_zero():
    # abort_prob defaults to 0.3 (DESIGN 5.1); 0 is the explicit opt-out.
    src, _, _ = make_source({"eval": dict(EVAL, abort_prob=0)}, horizon="1d")
    assert all(j.terminal_status_override is None for _, j in drain(src))


def test_abort_prob_defaults_on():
    # With no abort_prob key, roughly 30% of jobs carry an abort override.
    src, _, _ = make_source({"eval": dict(EVAL)}, horizon="7d")
    jobs = [j for _, j in drain(src)]
    frac = sum(1 for j in jobs if j.terminal_status_override is not None) / len(jobs)
    assert 0.25 < frac < 0.35


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def fingerprint(arrivals):
    return [
        (
            t,
            j.id,
            j.tenant,
            j.gangs[0].chips,
            j.true_duration_s,
            j.walltime_est_s,
            j.terminal_status_override,
        )
        for t, j in arrivals
    ]


def test_same_seed_identical_stream():
    classes = {"eval": dict(EVAL, abort_prob=0.3)}
    a = fingerprint(drain(make_source(classes, horizon="2d", seed=42)[0]))
    b = fingerprint(drain(make_source(classes, horizon="2d", seed=42)[0]))
    assert a == b
    c = fingerprint(drain(make_source(classes, horizon="2d", seed=43)[0]))
    assert a != c


def test_stream_override_is_a_paired_experiment():
    # Re-seeding only outcome/<class> changes outcomes but NOT arrival
    # times, sizes, durations, or tenants (independent named streams).
    classes = {"eval": dict(EVAL, abort_prob=0.5)}
    a = drain(make_source(classes, horizon="2d", seed=42)[0])
    b = drain(
        make_source(classes, horizon="2d", seed=42, overrides={"outcome/eval": 999})[0]
    )
    assert [t for t, _ in a] == [t for t, _ in b]
    assert [j.gangs[0].chips for _, j in a] == [j.gangs[0].chips for _, j in b]
    assert [j.tenant for _, j in a] == [j.tenant for _, j in b]
    assert [j.walltime_est_s for _, j in a] == [j.walltime_est_s for _, j in b]
    assert [j.terminal_status_override for _, j in a] != [
        j.terminal_status_override for _, j in b
    ]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_rejects_trace_kind():
    scn = make_scenario({"eval": dict(EVAL)})
    fleet = build_fleet(scn)
    scn.workload.kind = "trace"
    with pytest.raises(ValueError, match="synthetic"):
        SyntheticSource(scn.workload, fleet, RngStreams(0), scn.sim.horizon_us)


def test_rejects_bad_horizon_and_multi_gang():
    scn = make_scenario({"eval": dict(EVAL)})
    fleet = build_fleet(scn)
    with pytest.raises(ValueError, match="horizon"):
        SyntheticSource(scn.workload, fleet, RngStreams(0), 0)
    scn.workload.classes[0].n_gangs = 2
    with pytest.raises(ValueError, match="not implemented in v0.1"):
        SyntheticSource(scn.workload, fleet, RngStreams(0), scn.sim.horizon_us)


def test_empty_classes_is_immediately_exhausted():
    scn = make_scenario({"eval": dict(EVAL)})
    fleet = build_fleet(scn)
    scn.workload.classes = []
    src = SyntheticSource(scn.workload, fleet, RngStreams(0), scn.sim.horizon_us)
    assert src.next_arrival() is None


# ---------------------------------------------------------------------------
# Services expansion
# ---------------------------------------------------------------------------


def make_service(sid="svc", replicas=3, chip_type="h100", **kw):
    return Service(
        id=sid,
        tenant="t0",
        replica_spec=GangSpec(chips=8, chip_type=chip_type),
        min_replicas=replicas,
        max_replicas=kw.pop("max_replicas", replicas),
    )


def test_expand_services_frozen_replicas():
    scn = make_scenario({"eval": dict(EVAL)}, horizon="1d")
    fleet = build_fleet(scn)
    jobs = expand_services([make_service(replicas=3)], fleet, scn.sim.horizon_us)
    assert [j.id for j in jobs] == ["svc-r0", "svc-r1", "svc-r2"]
    for j in jobs:
        assert j.job_class is JobClass.INFER_REPLICA
        assert j.tier is Tier.PROD
        assert j.submit_t == 0
        assert j.gangs[0].chips == 8  # one whole node
        assert j.gangs[0].chip_type == "h100"
        assert j.true_duration_s == 86400.0  # open-ended = horizon
        assert j.checkpoint_interval_s == 0.0
        assert j.service_id == "svc"


def test_expand_services_rejects_autoscaling():
    scn = make_scenario({"eval": dict(EVAL)}, horizon="1d")
    fleet = build_fleet(scn)
    svc = make_service(replicas=1, max_replicas=4)
    with pytest.raises(ValueError, match="not implemented in v0.1"):
        expand_services([svc], fleet, scn.sim.horizon_us)


def test_expand_services_rejects_unknown_chip_type():
    scn = make_scenario({"eval": dict(EVAL)}, horizon="1d")
    fleet = build_fleet(scn)
    with pytest.raises(ValueError, match="tpu_v5p"):
        expand_services([make_service(chip_type="tpu_v5p")], fleet, scn.sim.horizon_us)


def test_initial_jobs_merge_ahead_of_arrivals():
    scn = make_scenario({"eval": dict(EVAL)}, horizon="1d")
    fleet = build_fleet(scn)
    replicas = expand_services([make_service(replicas=2)], fleet, scn.sim.horizon_us)
    src = SyntheticSource(
        scn.workload, fleet, RngStreams(0), scn.sim.horizon_us, initial_jobs=replicas
    )
    arrivals = drain(src)
    assert [j.id for _, j in arrivals[:2]] == ["svc-r0", "svc-r1"]
    assert arrivals[0][0] == 0 and arrivals[1][0] == 0
    times = [t for t, _ in arrivals]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# End-to-end engine smoke
# ---------------------------------------------------------------------------


def test_engine_smoke_run():
    scn = make_scenario(
        {"eval": dict(EVAL, abort_prob=0.3)}, horizon="6h", n_nodes=4
    )
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    src = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    sim = Simulator(scn, fleet, src, FIFOScheduler(strict=False), NullSink(), rng=rng)
    sim.run()
    fleet.check_invariants()
    statuses = [rt.job.status for rt in sim._jobs.values()]
    assert statuses, "no jobs arrived"
    done = [s for s in statuses if s in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED)]
    assert len(done) > len(statuses) * 0.5  # most short evals finish in 6h
    # Aborting jobs really surface their override statuses.
    assert any(s in (JobStatus.FAILED, JobStatus.CANCELED) for s in statuses)
