"""Sampler tests: parameter recovery, bounds, validation, and the
DistSpec -> sampler factory (including the microseconds -> seconds scale
rule for duration specs)."""

import math

import numpy as np
import pytest

from fleetsim.config import DistSpec, parse_dist
from fleetsim.workload.distributions import (
    Exponential,
    Fixed,
    LogNormal,
    Pow2,
    Sampler,
    Uniform,
    Z90,
    from_spec,
)

N = 20_000


def rng():
    return np.random.default_rng(1234)


def draws(sampler, n=N, seed_gen=None):
    g = seed_gen if seed_gen is not None else rng()
    return np.array([sampler.sample(g) for _ in range(n)])


# ---------------------------------------------------------------------------
# Individual samplers
# ---------------------------------------------------------------------------


def test_fixed_is_degenerate():
    s = Fixed(42)
    assert [s.sample(rng()) for _ in range(5)] == [42.0] * 5


def test_uniform_bounds_and_mean():
    x = draws(Uniform(10.0, 20.0))
    assert x.min() >= 10.0 and x.max() <= 20.0
    assert abs(x.mean() - 15.0) < 0.1


def test_uniform_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Uniform(5.0, 1.0)


def test_exponential_mean_recovery():
    x = draws(Exponential(30.0))
    assert (x >= 0).all()
    assert abs(x.mean() - 30.0) / 30.0 < 0.05


def test_exponential_rejects_nonpositive_mean():
    with pytest.raises(ValueError):
        Exponential(0.0)
    with pytest.raises(ValueError):
        Exponential(-1.0)


def test_pow2_values_and_uniform_exponents():
    x = draws(Pow2(1, 8))
    values, counts = np.unique(x, return_counts=True)
    assert list(values) == [1.0, 2.0, 4.0, 8.0]
    # Uniform over EXPONENTS: each of the 4 exponents ~ 25%.
    freqs = counts / len(x)
    assert (np.abs(freqs - 0.25) < 0.02).all()


def test_pow2_degenerate_and_validation():
    assert draws(Pow2(16, 16), n=50).tolist() == [16.0] * 50
    with pytest.raises(ValueError):
        Pow2(3, 8)  # not a power of two
    with pytest.raises(ValueError):
        Pow2(8, 4)  # inverted
    with pytest.raises(ValueError):
        Pow2(0, 8)


def test_lognormal_median_p90_recovery():
    # The headline property test: (median, p90) parameterization recovers
    # its own quantiles.  median 2 min, p90 30 min (in seconds).
    x = draws(LogNormal(120.0, 1800.0))
    assert abs(np.median(x) - 120.0) / 120.0 < 0.05
    assert abs(np.percentile(x, 90) - 1800.0) / 1800.0 < 0.10


def test_lognormal_sigma_formula():
    # sigma = (ln p90 - ln median) / Z90 pins the shape exactly.
    s = LogNormal(120.0, 1800.0)
    sigma = (math.log(1800.0) - math.log(120.0)) / Z90
    x = np.log(draws(s))
    assert abs(x.std() - sigma) / sigma < 0.03


def test_lognormal_degenerate_point_mass():
    x = draws(LogNormal(300.0, 300.0), n=100)
    assert np.allclose(x, 300.0)


def test_lognormal_cap_is_hard_max():
    x = draws(LogNormal(120.0, 1800.0, cap=600.0))
    assert x.max() <= 600.0
    assert (x == 600.0).sum() > 0  # the tail actually hits the cap


def test_lognormal_validation():
    with pytest.raises(ValueError):
        LogNormal(0.0, 10.0)
    with pytest.raises(ValueError):
        LogNormal(10.0, 5.0)  # p90 < median
    with pytest.raises(ValueError):
        LogNormal(10.0, 20.0, cap=0.0)


# ---------------------------------------------------------------------------
# from_spec factory
# ---------------------------------------------------------------------------


def test_from_spec_kinds():
    assert isinstance(from_spec(parse_dist("fixed[8]")), Fixed)
    assert isinstance(from_spec(parse_dist("uniform[1, 9]")), Uniform)
    assert isinstance(from_spec(parse_dist("pow2[1, 8]")), Pow2)
    assert isinstance(from_spec(parse_dist("exponential[mean=30s]"), scale=1e-6), Exponential)
    assert isinstance(from_spec(parse_dist("lognormal[median=2m, p90=30m]"), scale=1e-6), LogNormal)


def test_from_spec_duration_scale_us_to_seconds():
    # Duration DistSpec params are int MICROSECONDS; scale=1e-6 yields a
    # sampler in seconds.
    spec = parse_dist("lognormal[median=2m, p90=30m]")
    assert spec.params == {"median": 120_000_000, "p90": 1_800_000_000}
    x = draws(from_spec(spec, scale=1e-6))
    assert abs(np.median(x) - 120.0) / 120.0 < 0.05
    assert abs(np.percentile(x, 90) - 1800.0) / 1800.0 < 0.10


def test_from_spec_bare_duration_string_becomes_fixed_seconds():
    s = from_spec(parse_dist("600s"), scale=1e-6)
    assert s.sample(rng()) == 600.0


def test_from_spec_cap_applies_to_any_kind():
    s = from_spec(parse_dist("exponential[mean=100]"), cap=50.0)
    x = draws(s, n=2000)
    assert x.max() <= 50.0


def test_from_spec_pow2_rejects_scale():
    with pytest.raises(ValueError, match="scaled"):
        from_spec(parse_dist("pow2[1, 8]"), scale=1e-6)


def test_from_spec_invalid_sentinel_raises():
    with pytest.raises(ValueError, match="invalid"):
        from_spec(DistSpec("invalid", {}))


def test_from_spec_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown"):
        from_spec(DistSpec("weibull", {"k": 1.0}))


def test_from_spec_missing_param_raises():
    with pytest.raises(ValueError, match="missing parameter"):
        from_spec(DistSpec("lognormal", {"median": 10.0}))


def test_samplers_satisfy_protocol():
    for s in (Fixed(1), Uniform(0, 1), Pow2(1, 2), Exponential(1.0), LogNormal(1.0, 2.0)):
        assert isinstance(s, Sampler)


def test_sampling_is_deterministic_per_generator_state():
    s = LogNormal(120.0, 1800.0)
    a = draws(s, n=100, seed_gen=np.random.default_rng(7))
    b = draws(s, n=100, seed_gen=np.random.default_rng(7))
    assert np.array_equal(a, b)
