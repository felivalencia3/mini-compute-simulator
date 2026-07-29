"""v0.2 math-informed traffic generator tests (docs/traffic-math.md).

Covers: the new duration/size samplers (lognormal p99, pareto, weibull,
weighted pmf, lognormal-body/Pareto-tail splice), the pluggable arrival
processes (poisson, nhpp harmonic, mmpp2, hawkes) with statistical
recovery of their pinned properties, the bounded-Zipf tenant model, the
arrival config surface, per-stream determinism, the v0.1
backward-compatibility regression on examples/01_minimal, and the
google_fleet preset expansion.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fleetsim.config import (
    HELIOS_V01_DAILY,
    PMF_PRESETS,
    TENANT_ZIPF_S_DEFAULT,
    load_scenario,
    validate,
)
from fleetsim.engine.rng import RngStreams
from fleetsim.fleet.build import build_fleet
from fleetsim.model import JobClass, Tier
from fleetsim.workload.distributions import (
    LogNormal,
    Pareto,
    Pmf,
    SpliceLogNormalPareto,
    Weibull,
    Z99,
    from_spec,
    norm_cdf,
    norm_ppf,
)
from fleetsim.workload.synthetic import (
    HarmonicCurve,
    HawkesArrivals,
    MMPP2Arrivals,
    NHPPArrivals,
    PoissonArrivals,
    StepCurve,
    SyntheticSource,
    bounded_zipf_cdf,
)

S = 1_000_000
HOUR = 3600 * S
DAY = 24 * HOUR

EXAMPLE_01 = Path(__file__).parent.parent / "examples" / "01_minimal" / "scenario.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scenario(classes, horizon="7d", n_nodes=2, per_node=8, seed=0, workload_extra=None):
    workload = {"kind": "synthetic", "classes": classes}
    if workload_extra:
        workload.update(workload_extra)
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
            "workload": workload,
        }
    )


def make_source(classes, horizon="7d", seed=0, overrides=None, **kw):
    scn = make_scenario(classes, horizon=horizon, seed=seed, **kw)
    fleet = build_fleet(scn)
    rng = RngStreams(seed, overrides)
    return SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us), scn, fleet


def drain(src):
    out = []
    while (nxt := src.next_arrival()) is not None:
        out.append(nxt)
    return out


def errors_for(doc):
    return validate(load_scenario(doc, strict=False))


def base_doc(classes):
    return {
        "sim": {"horizon": "1d", "round": "60s", "seed": 0},
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
        "workload": {"kind": "synthetic", "classes": classes},
    }


EVAL = {
    "rate_per_hour": 40,
    "chips": "pow2[1, 8]",
    "duration": "lognormal[median=2m, p90=30m]",
}


def drain_process(proc, rng, horizon_us):
    """Drain a bare arrival process into a list of int-µs times."""
    out, prev = [], 0
    while (t := proc.next_time(rng, prev, horizon_us)) is not None:
        out.append(t)
        prev = t
    return out


def hourly_fano(times_us, horizon_us):
    counts, _ = np.histogram(
        np.asarray(times_us), bins=np.arange(0, horizon_us + 1, HOUR)
    )
    return counts.var() / counts.mean(), counts


# ---------------------------------------------------------------------------
# New samplers
# ---------------------------------------------------------------------------


class TestNewSamplers:
    def test_lognormal_p99_quantile_recovery(self):
        s = LogNormal(120.0, p99=3600.0)
        g = np.random.default_rng(1)
        x = np.array([s.sample(g) for _ in range(40_000)])
        assert abs(np.median(x) - 120.0) / 120.0 < 0.05
        assert abs(np.percentile(x, 99) - 3600.0) / 3600.0 < 0.10
        # sigma formula uses Z99
        assert s.sigma == pytest.approx((math.log(3600) - math.log(120)) / Z99)

    def test_lognormal_exactly_one_quantile(self):
        with pytest.raises(ValueError, match="exactly one"):
            LogNormal(120.0, 1800.0, p99=3600.0)
        with pytest.raises(ValueError, match="exactly one"):
            LogNormal(120.0)

    def test_lognormal_p99_below_median_rejected(self):
        with pytest.raises(ValueError, match="p99 >= median"):
            LogNormal(120.0, p99=60.0)

    def test_pareto_bounds_and_tail_exponent(self):
        s = Pareto(1.5, 60.0)
        g = np.random.default_rng(2)
        x = np.array([s.sample(g) for _ in range(50_000)])
        assert x.min() >= 60.0
        # Hill estimator at the exact threshold xm recovers alpha.
        alpha_hat = 1.0 / np.mean(np.log(x / 60.0))
        assert abs(alpha_hat - 1.5) / 1.5 < 0.05

    def test_pareto_validation(self):
        with pytest.raises(ValueError):
            Pareto(0.0, 1.0)
        with pytest.raises(ValueError):
            Pareto(1.5, 0.0)

    def test_pareto_from_spec_duration_scale(self):
        from fleetsim.config import parse_dist

        s = from_spec(parse_dist("pareto[alpha=1.5, xm=1h]"), scale=1e-6)
        assert isinstance(s, Pareto)
        assert s.xm == pytest.approx(3600.0)

    def test_weibull_mean_recovery(self):
        s = Weibull(1.5, 100.0)
        g = np.random.default_rng(3)
        x = np.array([s.sample(g) for _ in range(50_000)])
        expected = 100.0 * math.gamma(1.0 + 1.0 / 1.5)
        assert abs(x.mean() - expected) / expected < 0.03
        assert (x >= 0).all()

    def test_weibull_validation(self):
        with pytest.raises(ValueError):
            Weibull(0.0, 1.0)
        with pytest.raises(ValueError):
            Weibull(1.0, -1.0)

    def test_pmf_frequency_recovery(self):
        s = Pmf.from_weights({1: 0.55, 2: 0.15, 4: 0.15, 8: 0.15})
        g = np.random.default_rng(4)
        x = np.array([s.sample(g) for _ in range(40_000)])
        values, counts = np.unique(x, return_counts=True)
        assert list(values) == [1.0, 2.0, 4.0, 8.0]
        freqs = dict(zip(values, counts / len(x)))
        for v, w in {1: 0.55, 2: 0.15, 4: 0.15, 8: 0.15}.items():
            assert abs(freqs[v] - w) < 0.01

    def test_pmf_normalizes_weights(self):
        s = Pmf.from_weights({1: 2.0, 2: 2.0})
        assert s.cum == (0.5, 1.0)

    def test_pmf_validation(self):
        with pytest.raises(ValueError):
            Pmf.from_weights({})
        with pytest.raises(ValueError):
            Pmf.from_weights({1: 0.0})

    def test_norm_ppf_round_trip_and_pinned_quantiles(self):
        for x in (-3.0, -1.2816, -0.5, 0.0, 0.5, 1.2816, 2.3263, 3.5):
            assert abs(norm_ppf(norm_cdf(x)) - x) < 1e-6
        assert norm_ppf(0.9) == pytest.approx(1.281552, abs=1e-4)
        # Extreme inputs never blow up (raw rng.random() can be 0.0).
        assert math.isfinite(norm_ppf(0.0))
        assert math.isfinite(norm_ppf(1.0))


class TestSplice:
    # The doc-pinned pretrain example: median 12 d, p90 30 d, alpha 1.5,
    # splice at p90, cap 54 d (traffic-math §2.3 table).
    D = 86_400.0  # seconds per day

    def pretrain(self):
        return SpliceLogNormalPareto.from_quantiles(
            12 * self.D, 1.5, 54 * self.D, p90=30 * self.D, splice_q=0.90
        )

    def test_doc_pinned_derived_constants(self):
        s = self.pretrain()
        assert s.sigma == pytest.approx(0.715, abs=1e-3)
        assert s.w == pytest.approx(0.846, abs=1e-3)
        assert s.theta == pytest.approx(30 * self.D, rel=1e-3)

    def test_body_weight_is_w(self):
        # P(X <= theta) == w exactly (the branch probability).
        s = self.pretrain()
        g = np.random.default_rng(5)
        x = np.array([s.sample(g) for _ in range(100_000)])
        assert abs((x <= s.theta).mean() - s.w) < 0.005

    def test_bounded_by_cap_and_positive(self):
        s = self.pretrain()
        g = np.random.default_rng(6)
        x = np.array([s.sample(g) for _ in range(50_000)])
        assert x.max() <= 54 * self.D
        assert x.min() > 0

    def test_continuity_at_the_joint(self):
        # With a distant cap the truncation factor 1/q -> 1 and the
        # density is continuous at theta: adjacent narrow bins on each
        # side hold approximately equal mass.
        s = SpliceLogNormalPareto.from_quantiles(
            14_400.0, 2.0, 100 * 86_400.0, p90=86_400.0, splice_q=0.90
        )
        g = np.random.default_rng(7)
        x = np.array([s.sample(g) for _ in range(400_000)])
        th = s.theta
        left = ((x >= 0.96 * th) & (x < th)).sum()
        right = ((x >= th) & (x < 1.04 * th)).sum()
        assert left > 500 and right > 500
        assert 0.75 < right / left < 1.25

    def test_tail_exponent_recovery(self):
        # Hill estimator at the exact threshold theta over the Pareto
        # tail (cap far away so truncation bias is negligible).
        s = SpliceLogNormalPareto.from_quantiles(
            14_400.0, 2.0, 1000 * 86_400.0, p90=86_400.0, splice_q=0.90
        )
        g = np.random.default_rng(8)
        x = np.array([s.sample(g) for _ in range(300_000)])
        tail = x[x > s.theta]
        assert len(tail) > 5_000
        alpha_hat = 1.0 / np.mean(np.log(tail / s.theta))
        assert abs(alpha_hat - 2.0) / 2.0 < 0.05

    def test_exactly_two_draws_per_sample(self):
        s = self.pretrain()
        g1 = np.random.default_rng(9)
        s.sample(g1)
        g2 = np.random.default_rng(9)
        g2.random()
        g2.random()
        # After 1 sample vs 2 raw uniforms, the streams are in the same
        # state — the fixed-draw-count contract (traffic-math §3.4).
        assert g1.random() == g2.random()

    def test_validation(self):
        with pytest.raises(ValueError, match="alpha"):
            SpliceLogNormalPareto.from_quantiles(
                100.0, 1.0, 1e6, p90=200.0, splice_q=0.9
            )
        with pytest.raises(ValueError, match="theta < cap"):
            SpliceLogNormalPareto.from_quantiles(
                100.0, 1.5, 150.0, p90=200.0, splice_q=0.9
            )
        with pytest.raises(ValueError, match="exactly one"):
            SpliceLogNormalPareto.from_quantiles(100.0, 1.5, 1e6, p90=200.0)


# ---------------------------------------------------------------------------
# Arrival processes — statistical recovery
# ---------------------------------------------------------------------------


class TestArrivalProcesses:
    def test_poisson_mean_rate(self):
        H = 14 * DAY
        ts = drain_process(PoissonArrivals(100.0), np.random.default_rng(1), H)
        assert abs(len(ts) / (14 * 24) - 100.0) / 100.0 < 0.03
        fano, _ = hourly_fano(ts, H)
        assert fano < 1.3  # Poisson reference: Fano ~ 1

    def test_nhpp_harmonic_mean_rate_and_diurnal_shape(self):
        H = 28 * DAY
        curve = HarmonicCurve(HELIOS_V01_DAILY)
        ts = drain_process(NHPPArrivals(60.0, curve), np.random.default_rng(2), H)
        assert abs(len(ts) / (28 * 24) - 60.0) / 60.0 < 0.05
        by_hour = np.zeros(24)
        for t in ts:
            by_hour[(t % DAY) // HOUR] += 1
        # helios_v01 (mean-normalized): trough ~0.59x around 03:00, single
        # peak ~1.27x around 10:15 (the K=2 fit smooths the step curve's
        # 12:00/18:00 dips away).
        trough = by_hour[3] / by_hour.mean()
        peak = by_hour.max() / by_hour.mean()
        assert trough < 0.7
        assert peak > 1.0

    def test_harmonic_theta0_normalizes_mean_to_one(self):
        curve = HarmonicCurve(HELIOS_V01_DAILY, ((0.1, -0.05),))
        grid = np.arange(0, 168 * 60) * (60 * S)  # µs, 1-min grid over a week
        mult = np.array([curve.multiplier(int(t)) for t in grid])
        assert abs(mult.mean() - 1.0) < 1e-6
        # The closed-form bound dominates the curve everywhere.
        assert mult.max() <= curve.vmax + 1e-12

    def test_nhpp_weekly_harmonics_shape(self):
        # A pure weekly harmonic must modulate the day-of-week totals.
        H = 56 * DAY
        curve = HarmonicCurve((), ((0.5, 0.0),))
        ts = drain_process(NHPPArrivals(30.0, curve), np.random.default_rng(3), H)
        assert abs(len(ts) / (56 * 24) - 30.0) / 30.0 < 0.05
        by_dow = np.zeros(7)
        for t in ts:
            by_dow[(t // DAY) % 7] += 1
        assert by_dow.max() / by_dow.min() > 1.5

    def test_mmpp2_mean_rate_and_overdispersion(self):
        H = 28 * DAY
        proc = MMPP2Arrivals(60.0, 4.0, 0.25, 7200.0)
        ts = drain_process(proc, np.random.default_rng(4), H)
        assert abs(len(ts) / (28 * 24) - 60.0) / 60.0 < 0.10
        fano, _ = hourly_fano(ts, H)
        assert fano > 5.0  # strongly overdispersed vs Poisson's ~1

    def test_mmpp2_burst_regimes_visible(self):
        H = 28 * DAY
        proc = MMPP2Arrivals(50.0, 20.0, 0.2, 6 * 3600.0)
        ts = drain_process(proc, np.random.default_rng(5), H)
        _, counts = hourly_fano(ts, H)
        counts = np.sort(counts)
        n = len(counts)
        quiet = counts[: n // 10].mean()  # bottom decile ~ quiet regime
        burst = counts[-n // 10 :].mean()  # top decile ~ burst regime
        assert burst / max(quiet, 0.5) > 5.0

    def test_mmpp2_closed_form_rates(self):
        proc = MMPP2Arrivals(60.0, 4.0, 0.25, 7200.0)
        lam_bar = 60.0 / 3600.0
        lam_q = lam_bar / (1 - 0.25 + 0.25 * 4)
        assert proc.lam_q_s == pytest.approx(lam_q)
        assert proc.lam_b_s == pytest.approx(4 * lam_q)
        # pi_b = sig_qb/(sig_qb+sig_bq) = burst_frac; 1/(sum) = switch_tau
        assert proc.sig_qb / (proc.sig_qb + proc.sig_bq) == pytest.approx(0.25)
        assert 1.0 / (proc.sig_qb + proc.sig_bq) == pytest.approx(7200.0)

    def test_hawkes_mean_rate_and_clustering(self):
        H = 28 * DAY
        ts = drain_process(
            HawkesArrivals(60.0, 0.4, 900.0), np.random.default_rng(6), H
        )
        assert abs(len(ts) / (28 * 24) - 60.0) / 60.0 < 0.05
        fano, _ = hourly_fano(ts, H)
        assert fano > 1.5  # variance-to-mean >> 1: self-excitation clusters

    def test_hawkes_zero_branching_degenerates_to_poisson(self):
        H = 14 * DAY
        ts = drain_process(
            HawkesArrivals(60.0, 0.0, 900.0), np.random.default_rng(7), H
        )
        assert abs(len(ts) / (14 * 24) - 60.0) / 60.0 < 0.05
        fano, _ = hourly_fano(ts, H)
        assert fano < 1.3

    def test_hawkes_stability_enforced(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            HawkesArrivals(60.0, 1.0, 900.0)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            HawkesArrivals(60.0, -0.1, 900.0)

    def test_processes_stay_inside_horizon(self):
        H = 6 * HOUR
        for proc in (
            PoissonArrivals(10.0),
            NHPPArrivals(10.0, StepCurve()),
            MMPP2Arrivals(10.0, 4.0, 0.25, 3600.0),
            HawkesArrivals(10.0, 0.3, 600.0),
        ):
            rng = np.random.default_rng(8)
            ts = drain_process(proc, rng, H)
            assert all(0 <= t < H for t in ts)
            assert ts == sorted(ts)
            # Past-horizon prev always yields None.
            assert proc.next_time(rng, H, H) is None

    def test_stateful_processes_exhaust_sticky(self):
        # MMPP2/Hawkes carry internal clocks: once past the horizon they
        # keep returning None regardless of prev (the source's sticky-
        # exhaustion contract; stateless renewal processes rely on the
        # source not calling again after None).
        H = 6 * HOUR
        for proc in (
            MMPP2Arrivals(10.0, 4.0, 0.25, 3600.0),
            HawkesArrivals(10.0, 0.3, 600.0),
        ):
            rng = np.random.default_rng(8)
            drain_process(proc, rng, H)
            assert proc.next_time(rng, 0, H) is None
            assert proc.next_time(rng, 0, H) is None


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestArrivalConfig:
    def test_mapping_form_poisson_rate_inside_block(self):
        scn = load_scenario(
            base_doc(
                {
                    "eval": {
                        "arrival": {"process": "poisson", "rate_per_day": 240},
                        "chips": "pow2[1, 8]",
                        "duration": "fixed[60s]",
                    }
                }
            )
        )
        c = scn.workload.classes[0]
        assert c.arrival_process.process == "poisson"
        assert c.rate_per_hour == pytest.approx(10.0)
        assert c.arrival is None

    def test_top_level_rate_with_arrival_block_rejected(self):
        doc = base_doc(
            {
                "eval": {
                    "rate_per_hour": 40,
                    "arrival": {"process": "poisson", "rate_per_hour": 40},
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("INSIDE the block" in e for e in errs)

    def test_unknown_process_rejected(self):
        doc = base_doc(
            {
                "eval": {
                    "arrival": {"process": "cox", "rate_per_hour": 1},
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("unknown arrival process" in e and "cox" in e for e in errs)

    def test_rate_required_inside_block(self):
        doc = base_doc(
            {
                "eval": {
                    "arrival": {"process": "poisson"},
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("exactly one of" in e and "arrival block" in e for e in errs)

    def test_closed_loop_maps_to_backlog_with_defaults(self):
        scn = load_scenario(
            base_doc(
                {
                    "filler": {
                        "class": "finetune",
                        "arrival": {
                            "process": "closed_loop",
                            "closed_loop": {"concurrency": 6},
                        },
                        "chips": "pow2[1, 8]",
                        "duration": "fixed[60s]",
                    }
                }
            )
        )
        c = scn.workload.classes[0]
        assert c.arrival is not None and c.arrival.kind == "backlog"
        assert c.arrival.params == {"target_pending": 6}
        assert c.arrival_process is None
        assert c.tier is Tier.BEST_EFFORT  # backlog-class default
        assert c.min_runtime_s == 0.0

    def test_closed_loop_rejects_rate_keys(self):
        doc = base_doc(
            {
                "filler": {
                    "class": "finetune",
                    "arrival": {"process": "closed_loop", "rate_per_hour": 5},
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("forbidden" in e and "closed_loop" in e for e in errs)

    def test_seasonality_pair_limits(self):
        for key, pairs, ok in (
            ("daily", [[0.1, 0.1]] * 3, True),
            ("daily", [[0.1, 0.1]] * 4, False),
            ("weekly", [[0.1, 0.1]] * 2, True),
            ("weekly", [[0.1, 0.1]] * 3, False),
        ):
            doc = base_doc(
                {
                    "eval": {
                        "arrival": {
                            "process": "nhpp",
                            "rate_per_hour": 10,
                            "seasonality": {key: pairs},
                        },
                        "chips": "pow2[1, 8]",
                        "duration": "fixed[60s]",
                    }
                }
            )
            errs = errors_for(doc)
            if ok:
                assert errs == [], errs
            else:
                assert any("harmonic pairs" in e for e in errs)

    def test_nhpp_requires_seasonality(self):
        doc = base_doc(
            {
                "eval": {
                    "arrival": {"process": "nhpp", "rate_per_hour": 10},
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("requires a seasonality" in e for e in errs)

    def test_seasonality_rejected_for_poisson(self):
        doc = base_doc(
            {
                "eval": {
                    "arrival": {
                        "process": "poisson",
                        "rate_per_hour": 10,
                        "seasonality": "helios_v01",
                    },
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("not accepted with process" in e for e in errs)

    def test_hawkes_branching_stability_at_validate(self):
        for bad in (1.0, 1.5, -0.2):
            doc = base_doc(
                {
                    "eval": {
                        "arrival": {
                            "process": "hawkes",
                            "rate_per_hour": 10,
                            "hawkes": {"branching": bad, "kernel_tau": "15m"},
                        },
                        "chips": "pow2[1, 8]",
                        "duration": "fixed[60s]",
                    }
                }
            )
            errs = errors_for(doc)
            assert any("branching" in e for e in errs), bad

    def test_mmpp2_param_ranges(self):
        for block in (
            {"rate_ratio": 1.0},
            {"burst_frac": 0.0},
            {"burst_frac": 1.0},
            {"switch_tau": "0s"},
        ):
            doc = base_doc(
                {
                    "ft": {
                        "class": "finetune",
                        "arrival": {
                            "process": "mmpp2",
                            "rate_per_hour": 10,
                            "mmpp2": block,
                        },
                        "chips": "pow2[1, 8]",
                        "duration": "fixed[60s]",
                    }
                }
            )
            errs = errors_for(doc)
            assert any("mmpp2" in e for e in errs), block

    def test_process_block_mismatch_rejected(self):
        doc = base_doc(
            {
                "eval": {
                    "arrival": {
                        "process": "poisson",
                        "rate_per_hour": 10,
                        "hawkes": {"branching": 0.4},
                    },
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("only accepted with process" in e for e in errs)

    def test_diurnal_with_arrival_block_rejected(self):
        doc = base_doc(
            {
                "eval": {
                    "diurnal": True,
                    "arrival": {
                        "process": "nhpp",
                        "rate_per_hour": 10,
                        "seasonality": "helios_v01",
                    },
                    "chips": "pow2[1, 8]",
                    "duration": "fixed[60s]",
                }
            }
        )
        errs = errors_for(doc)
        assert any("diurnal" in e and "arrival" in e for e in errs)

    def test_v01_sugar_normalizes_to_arrival_process(self):
        scn = load_scenario(base_doc({"eval": dict(EVAL)}))
        c = scn.workload.classes[0]
        assert c.arrival_process.process == "poisson"
        scn = load_scenario(base_doc({"eval": dict(EVAL, diurnal=True)}))
        c = scn.workload.classes[0]
        assert c.arrival_process.process == "nhpp"
        assert c.arrival_process.seasonality.preset == "v01_steps"


class TestDistConfig:
    def test_lognormal_p99_accepted(self):
        scn = load_scenario(
            base_doc(
                {"eval": dict(EVAL, duration="lognormal[median=2m, p99=1h]")}
            )
        )
        d = scn.workload.classes[0].duration
        assert d.params == {"median": 2 * 60 * S, "p99": HOUR}

    def test_lognormal_both_quantiles_rejected(self):
        doc = base_doc(
            {"eval": dict(EVAL, duration="lognormal[median=2m, p90=30m, p99=1h]")}
        )
        errs = errors_for(doc)
        assert any("p90 OR p99" in e for e in errs)

    def test_pmf_mapping_and_preset(self):
        doc = base_doc(
            {
                "eval": dict(EVAL, chips={"pmf": {1: 0.5, 2: 0.5}}),
                "tpu": dict(
                    EVAL, **{"class": "pretrain", "chips": {"pmf": "tpu_isca23"}}
                ),
            }
        )
        scn = load_scenario(doc)
        c0, c1 = scn.workload.classes
        assert c0.chips.kind == "pmf"
        assert c0.chips.params == {"1": 0.5, "2": 0.5}
        assert c1.chips.params == {
            str(k): pytest.approx(v) for k, v in PMF_PRESETS["tpu_isca23"].items()
        }

    def test_pmf_rejects_non_pow2_keys(self):
        doc = base_doc({"eval": dict(EVAL, chips={"pmf": {3: 1.0}})})
        errs = errors_for(doc)
        assert any("powers of two" in e for e in errs)

    def test_pmf_rejects_unknown_preset(self):
        doc = base_doc({"eval": dict(EVAL, chips={"pmf": "bogus"})})
        errs = errors_for(doc)
        assert any("unknown pmf preset" in e for e in errs)

    def test_pmf_as_duration_rejected(self):
        doc = base_doc({"eval": dict(EVAL, duration={"pmf": {1: 1.0}})})
        errs = errors_for(doc)
        assert any("chip-count distribution" in e for e in errs)

    def test_splice_parses_and_sets_max_lifetime(self):
        doc = base_doc(
            {
                "pre": {
                    "class": "pretrain",
                    "rate_per_week": 1,
                    "chips": "fixed[8]",
                    "duration": {
                        "body": "lognormal[median=12d, p90=30d]",
                        "tail": {"alpha": 1.5, "splice": "p90", "cap": "54d"},
                    },
                }
            }
        )
        scn = load_scenario(doc)
        c = scn.workload.classes[0]
        assert c.duration.kind == "splice"
        assert c.duration.params["splice_q"] == 0.90
        assert c.duration.params["cap"] == 54 * DAY
        # max_lifetime inherits the tail cap when omitted.
        assert c.max_lifetime_s == pytest.approx(54 * 86400.0)

    def test_splice_cap_above_max_lifetime_rejected(self):
        doc = base_doc(
            {
                "pre": {
                    "class": "pretrain",
                    "rate_per_week": 1,
                    "max_lifetime": "30d",
                    "chips": "fixed[8]",
                    "duration": {
                        "body": "lognormal[median=12d, p90=30d]",
                        "tail": {"alpha": 1.5, "splice": "p90", "cap": "54d"},
                    },
                }
            }
        )
        errs = errors_for(doc)
        assert any("exceeds max_lifetime" in e for e in errs)

    def test_splice_alpha_must_exceed_one(self):
        doc = base_doc(
            {
                "pre": {
                    "class": "pretrain",
                    "rate_per_week": 1,
                    "chips": "fixed[8]",
                    "duration": {
                        "body": "lognormal[median=12d, p90=30d]",
                        "tail": {"alpha": 1.0, "splice": "p90", "cap": "54d"},
                    },
                }
            }
        )
        errs = errors_for(doc)
        assert any("alpha" in e and "> 1" in e for e in errs)

    def test_splice_cap_required(self):
        doc = base_doc(
            {
                "pre": {
                    "class": "pretrain",
                    "rate_per_week": 1,
                    "chips": "fixed[8]",
                    "duration": {
                        "body": "lognormal[median=12d, p90=30d]",
                        "tail": {"alpha": 1.5},
                    },
                }
            }
        )
        errs = errors_for(doc)
        assert any("cap" in e and "required" in e for e in errs)

    def test_splice_as_chips_rejected(self):
        doc = base_doc(
            {
                "eval": dict(
                    EVAL,
                    chips={
                        "body": "lognormal[median=2m, p90=4m]",
                        "tail": {"alpha": 1.5, "cap": "1h"},
                    },
                )
            }
        )
        errs = errors_for(doc)
        assert any("duration distribution" in e for e in errs)

    def test_weibull_and_pareto_duration_strings(self):
        scn = load_scenario(
            base_doc(
                {
                    "a": dict(
                        EVAL,
                        **{"class": "eval", "duration": "weibull[shape=1.5, scale=10m]"},
                    ),
                    "b": dict(
                        EVAL,
                        **{"class": "eval", "duration": "pareto[alpha=2.5, xm=5m]"},
                    ),
                }
            )
        )
        a, b = scn.workload.classes
        assert a.duration.kind == "weibull"
        assert b.duration.kind == "pareto"
        # And they sample (seconds via the µs scale rule).
        sa = from_spec(a.duration, scale=1e-6)
        sb = from_spec(b.duration, scale=1e-6)
        g = np.random.default_rng(0)
        assert sa.sample(g) > 0
        assert sb.sample(g) >= 300.0


# ---------------------------------------------------------------------------
# Tenants — bounded Zipf
# ---------------------------------------------------------------------------


class TestTenantZipf:
    def test_cdf_shape(self):
        cdf = bounded_zipf_cdf(8, 1.2)
        assert len(cdf) == 8
        assert cdf[-1] == 1.0
        assert all(b > a for a, b in zip(cdf, cdf[1:]))
        # s = 0 degenerates to uniform.
        uni = bounded_zipf_cdf(4, 0.0)
        assert uni == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_doc_calibration_top5_share(self):
        # traffic-math §2.4: s = 1.2 at U = 1300 puts the top-5% tenant
        # share of submissions in the [70, 85]% band (PAI: 77%).
        cdf = bounded_zipf_cdf(1300, TENANT_ZIPF_S_DEFAULT)
        top5 = cdf[int(1300 * 0.05) - 1]
        assert 0.70 < top5 < 0.85

    def test_default_exponent_is_1_2(self):
        scn = load_scenario(base_doc({"eval": dict(EVAL)}))
        assert scn.workload.tenant_zipf_s == 1.2
        assert scn.workload.classes[0].tenant_zipf_s == 1.2

    def test_workload_and_class_overrides(self):
        doc = base_doc(
            {
                "eval": dict(EVAL),
                "pre": dict(EVAL, **{"class": "pretrain", "tenant_zipf_s": 2.0}),
            }
        )
        doc["workload"]["tenant_zipf_s"] = 0.9
        scn = load_scenario(doc)
        by_name = {c.name: c for c in scn.workload.classes}
        assert by_name["eval"].tenant_zipf_s == 0.9
        assert by_name["pre"].tenant_zipf_s == 2.0

    def test_negative_exponent_rejected(self):
        doc = base_doc({"eval": dict(EVAL, tenant_zipf_s=-1)})
        errs = errors_for(doc)
        assert any("tenant_zipf_s" in e for e in errs)

    def test_sampled_skew_follows_exponent(self):
        # Higher s concentrates more mass on t0.
        flat, _, _ = make_source(
            {"eval": dict(EVAL, tenant_zipf_s=0.0)}, horizon="4d"
        )
        skew, _, _ = make_source(
            {"eval": dict(EVAL, tenant_zipf_s=2.0)}, horizon="4d"
        )

        def t0_share(src):
            jobs = [j for _, j in drain(src)]
            return sum(1 for j in jobs if j.tenant == "t0") / len(jobs)

        assert t0_share(flat) < 0.2  # ~1/8
        assert t0_share(skew) > 0.55


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


BURSTY = {
    "eval": {
        "arrival": {
            "process": "hawkes",
            "rate_per_hour": 30,
            "seasonality": "helios_v01",
            "hawkes": {"branching": 0.4, "kernel_tau": "15m"},
        },
        "chips": {"pmf": {1: 0.7, 2: 0.3}},
        "duration": "lognormal[median=2m, p99=1h]",
    },
    "finetune": {
        "arrival": {
            "process": "mmpp2",
            "rate_per_day": 40,
            "seasonality": {"daily": [[-0.2, -0.2]], "weekly": [[0.1, 0.0]]},
            "mmpp2": {"rate_ratio": 4, "burst_frac": 0.25, "switch_tau": "12h"},
        },
        "chips": "pow2[8, 32]",
        "duration": {
            "body": "lognormal[median=4h, p90=24h]",
            "tail": {"alpha": 1.5, "splice": "p90", "cap": "7d"},
        },
    },
}


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


class TestDeterminism:
    def test_same_seed_identical_stream_all_processes(self):
        a = fingerprint(drain(make_source(BURSTY, horizon="4d", seed=42)[0]))
        b = fingerprint(drain(make_source(BURSTY, horizon="4d", seed=42)[0]))
        assert a == b
        c = fingerprint(drain(make_source(BURSTY, horizon="4d", seed=43)[0]))
        assert a != c

    def test_stream_independence_paired_experiment(self):
        # Re-seeding size/eval leaves every arrival time and the other
        # classes' draws untouched (named-stream contract).
        a = drain(make_source(BURSTY, horizon="4d", seed=42)[0])
        b = drain(
            make_source(
                BURSTY, horizon="4d", seed=42, overrides={"size/eval": 999}
            )[0]
        )
        assert [t for t, _ in a] == [t for t, _ in b]
        assert [j.true_duration_s for _, j in a] == [
            j.true_duration_s for _, j in b
        ]
        eval_a = [j.gangs[0].chips for _, j in a if j.source_class == "eval"]
        eval_b = [j.gangs[0].chips for _, j in b if j.source_class == "eval"]
        assert eval_a != eval_b
        ft_a = [j.gangs[0].chips for _, j in a if j.source_class == "finetune"]
        ft_b = [j.gangs[0].chips for _, j in b if j.source_class == "finetune"]
        assert ft_a == ft_b


# ---------------------------------------------------------------------------
# Backward compatibility (v0.1)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Guards the v0.1 arrival stream of examples/01_minimal.

    ``tests/data/v01_arrivals_golden.json`` freezes the stream as produced
    at v0.2 (verified equivalent to the v0.1 generator for v0.1-feature
    scenarios at capture time; the one pinned change is the tenant default
    tenant_zipf_s 1.5 -> 1.2, traffic-math §2.4).  Determinism is exact on
    a given platform, but the lognormal/exponential samplers route through
    the platform libm, whose exp/log may differ by 1 ULP across OS/arch —
    so continuous values are compared with tolerances and knife-edge
    discrete draws get a small mismatch budget, instead of a golden hash.
    """

    GOLDEN = Path(__file__).parent / "data" / "v01_arrivals_golden.json"

    @staticmethod
    def _drain_example_01():
        scn = load_scenario(EXAMPLE_01)
        fleet = build_fleet(scn)
        rng = RngStreams(scn.sim.seed)
        src = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
        rows = []
        while (nxt := src.next_arrival()) is not None:
            t, j = nxt
            rows.append(
                {
                    "t_us": t,
                    "id": j.id,
                    "chips": j.gangs[0].chips,
                    "duration_s": j.true_duration_s,
                    "walltime_s": j.walltime_est_s,
                    "outcome": str(j.terminal_status_override),
                    "tenant": j.tenant,
                }
            )
        return rows

    def test_examples_01_minimal_deterministic_in_process(self):
        # Exact equality is the contract on a fixed platform.
        assert self._drain_example_01() == self._drain_example_01()

    def test_examples_01_minimal_regression(self):
        rows = self._drain_example_01()
        golden = json.loads(self.GOLDEN.read_text())

        # Aggregate shape: ULP flips can move an arrival across the horizon
        # edge, so allow +/-2 on counts; wholesale logic changes move these
        # by orders of magnitude.
        assert abs(len(rows) - golden["total"]) <= 2
        per_class: dict[str, int] = {}
        for r in rows:
            cls = r["id"].rsplit("-", 1)[0]
            per_class[cls] = per_class.get(cls, 0) + 1
        for cls, n in golden["per_class"].items():
            assert abs(per_class.get(cls, 0) - n) <= 2, (cls, per_class.get(cls), n)

        def row_matches(got: dict, exp: dict) -> bool:
            if got["id"] != exp["id"] or got["chips"] != exp["chips"]:
                return False
            if abs(got["t_us"] - exp["t_us"]) > 2:
                return False
            for key in ("duration_s", "walltime_s"):
                g, e = got[key], exp[key]
                if (g is None) != (e is None):
                    return False
                # Golden values are stored at 9 significant digits (rel
                # error <= 5e-9); 1e-6 clears that plus libm ULP noise.
                if g is not None and abs(g - e) > 1e-6 * max(abs(g), abs(e), 1.0):
                    return False
            return got["outcome"] == exp["outcome"] and got["tenant"] == exp["tenant"]

        checked = golden["rows"]
        mismatches = sum(
            0 if row_matches(got, exp) else 1
            for got, exp in zip(rows[: len(checked)], checked)
        )
        # Knife-edge uniform-vs-threshold draws (outcome, tenant) may flip on
        # a 1-ULP pmf difference; a real regression flips nearly every row.
        assert mismatches <= 5, f"{mismatches} of {len(checked)} golden rows diverged"

    def test_diurnal_sugar_equals_explicit_nhpp_v01_steps(self):
        legacy = {"eval": dict(EVAL, diurnal=True)}
        explicit = {
            "eval": {
                "arrival": {
                    "process": "nhpp",
                    "rate_per_hour": 40,
                    "seasonality": "v01_steps",
                },
                "chips": "pow2[1, 8]",
                "duration": "lognormal[median=2m, p90=30m]",
            }
        }
        a = fingerprint(drain(make_source(legacy, horizon="3d", seed=7)[0]))
        b = fingerprint(drain(make_source(explicit, horizon="3d", seed=7)[0]))
        assert a == b

    def test_plain_rate_sugar_equals_explicit_poisson(self):
        legacy = {"eval": dict(EVAL)}
        explicit = {
            "eval": {
                "arrival": {"process": "poisson", "rate_per_hour": 40},
                "chips": "pow2[1, 8]",
                "duration": "lognormal[median=2m, p90=30m]",
            }
        }
        a = fingerprint(drain(make_source(legacy, horizon="3d", seed=7)[0]))
        b = fingerprint(drain(make_source(explicit, horizon="3d", seed=7)[0]))
        assert a == b


# ---------------------------------------------------------------------------
# google_fleet preset
# ---------------------------------------------------------------------------


def preset_doc(workload=None, levels=("pod", "rack", "node"), counts=(2, 16, 8)):
    return {
        "sim": {"horizon": "1d", "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": list(levels), "counts": list(counts)},
                }
            ],
        },
        "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
        "workload": workload or {"preset": "google_fleet"},
    }


class TestGoogleFleetPreset:
    def test_expansion_classes_processes_tiers(self):
        scn = load_scenario(preset_doc())
        by_name = {c.name: c for c in scn.workload.classes}
        assert list(by_name) == ["pretrain", "finetune", "eval", "best_effort"]
        assert scn.workload.preset == "google_fleet"
        pre, ft, ev, be = (
            by_name["pretrain"],
            by_name["finetune"],
            by_name["eval"],
            by_name["best_effort"],
        )
        assert pre.job_class is JobClass.PRETRAIN and pre.tier is Tier.PROD
        assert pre.arrival_process.process == "poisson"
        assert pre.duration.kind == "splice"
        assert ft.arrival_process.process == "mmpp2"
        assert ft.arrival_process.seasonality.preset == "helios_v01"
        assert ft.tier is Tier.BATCH
        assert ev.arrival_process.process == "hawkes"
        assert ev.arrival_process.hawkes_branching == pytest.approx(0.4)
        assert ev.tier is Tier.BATCH
        assert be.tier is Tier.BEST_EFFORT
        assert be.arrival.kind == "backlog"
        assert be.checkpoint_interval_s == 0.0  # cheap-kill filler

    def test_scale_defaults_to_fleet_chips_and_pins_rates(self):
        # 2 pods x 16 racks x 8 nodes x 8 chips = 2,048 chips -> f = 2.
        scn = load_scenario(preset_doc())
        by_name = {c.name: c for c in scn.workload.classes}
        assert by_name["eval"].rate_per_hour == pytest.approx(40.0)
        assert by_name["finetune"].rate_per_hour == pytest.approx(90.0 / 24)
        assert by_name["pretrain"].rate_per_hour == pytest.approx(2.0 / 168)
        assert by_name["best_effort"].arrival.params["target_pending"] == 4

    def test_explicit_scale_scales_rates(self):
        scn = load_scenario(
            preset_doc({"preset": "google_fleet", "scale": 4096})
        )
        by_name = {c.name: c for c in scn.workload.classes}
        assert by_name["eval"].rate_per_hour == pytest.approx(80.0)
        assert by_name["best_effort"].arrival.params["target_pending"] == 8

    def test_pretrain_pmf_truncated_to_scale(self):
        scn = load_scenario(preset_doc())  # 2,048 chips -> keep <= 512
        pre = next(c for c in scn.workload.classes if c.name == "pretrain")
        keys = sorted(int(k) for k in pre.chips.params)
        assert keys == [256, 512]
        assert sum(pre.chips.params.values()) == pytest.approx(1.0)

    def test_segments_derived_from_fleet_shape(self):
        scn = load_scenario(preset_doc())
        pre = next(c for c in scn.workload.classes if c.name == "pretrain")
        assert pre.segment_level == "rack"
        assert pre.segment_nodes == 8
        assert pre.within is not None and pre.within.level == "pod"

    def test_flat_fleet_gets_no_segments(self):
        scn = load_scenario(
            preset_doc(levels=("node",), counts=(64,))
        )
        pre = next(c for c in scn.workload.classes if c.name == "pretrain")
        assert pre.segment_nodes is None and pre.segment_level is None
        assert pre.within is None

    def test_override_merging_removal_and_addition(self):
        scn = load_scenario(
            preset_doc(
                {
                    "preset": "google_fleet",
                    "classes": {
                        "eval": {"n_tenants": 100},
                        "best_effort": None,
                        "monitor": {
                            "class": "eval",
                            "rate_per_hour": 1,
                            "chips": "fixed[1]",
                            "duration": "fixed[60s]",
                            "tier": "monitoring",
                        },
                    },
                }
            )
        )
        by_name = {c.name: c for c in scn.workload.classes}
        assert list(by_name) == ["pretrain", "finetune", "eval", "monitor"]
        assert by_name["eval"].n_tenants == 100
        # non-overridden keys keep preset values
        assert by_name["eval"].arrival_process.process == "hawkes"
        assert by_name["monitor"].tier is Tier.MONITORING

    def test_unknown_preset_and_scale_without_preset(self):
        errs = errors_for(preset_doc({"preset": "azure_fleet"}))
        assert any("unknown preset" in e for e in errs)
        errs = errors_for(
            preset_doc({"kind": "synthetic", "scale": 2048, "classes": {"eval": dict(EVAL)}})
        )
        assert any("only valid together with 'preset'" in e for e in errs)

    def test_preset_source_generates_all_classes(self):
        doc = preset_doc()
        doc["sim"]["horizon"] = "12h"
        scn = load_scenario(doc)
        fleet = build_fleet(scn)
        rng = RngStreams(scn.sim.seed)
        src = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
        jobs = [j for _, j in drain(src)]
        names = {j.source_class for j in jobs}
        # Open-loop classes emit through next_arrival; best_effort only
        # through refill.
        assert {"finetune", "eval"} <= names
        # And the backlog class tops up through refill.
        refilled = src.refill(0, {})
        assert len(refilled) == 4
        assert all(j.source_class == "best_effort" for j in refilled)
        # Segmented pretrains carry their segment spec when they appear.
        pre_doc = preset_doc()
        pre_scn = load_scenario(pre_doc)
        pre_cfg = next(
            c for c in pre_scn.workload.classes if c.name == "pretrain"
        )
        assert pre_cfg.segment_nodes == 8

    def test_preset_end_to_end_engine_run(self):
        from fleetsim.api import run_scenario

        doc = preset_doc()
        doc["sim"]["horizon"] = "6h"
        doc["scheduler"] = {
            "name": "tiered_priority",
            "params": {"preempt": "requeue"},
        }
        summary = run_scenario(doc)
        # The standing best-effort backlog drives occupancy up.
        assert summary["full"]["occupancy"] > 0.5
