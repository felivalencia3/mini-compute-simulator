"""Samplers for the declarative distribution expressions of ``fleetsim.config``.

A :class:`~fleetsim.config.DistSpec` is a parsed, declarative record; this
module turns it into a callable sampler via :func:`from_spec`.  Samplers
are immutable value objects — all randomness comes from the
``numpy.random.Generator`` passed to :meth:`Sampler.sample`, so the same
generator state always yields the same draw (the determinism contract).

UNITS
-----
A sampler returns **floats in the unit of its constructor parameters**.
DistSpec params that were written as duration strings are stored as **int
microseconds** (see ``fleetsim.config``); duration samplers are therefore
built with ``from_spec(spec, scale=1e-6)`` so they sample **seconds** —
callers convert to int microseconds themselves.  Chip-count specs use
``scale=1.0`` (the default).

INVARIANTS
----------
- Construction validates parameters and raises ``ValueError`` on bad ones
  (non-positive lognormal median, non-power-of-two ``Pow2`` bounds, ...).
- ``sample`` draws a bounded, fixed number of variates per call (exactly
  one for every kind here), so per-stream draw counts are predictable —
  part of the reproducibility contract.
- The DistSpec sentinel kind ``"invalid"`` (a recorded parse failure) is
  rejected by :func:`from_spec` with ``ValueError``.

Kinds and parameters (canonical names, per ``fleetsim.config``):

- ``fixed(value)`` — degenerate.
- ``uniform(lo, hi)`` — continuous uniform on ``[lo, hi]``.
- ``exponential(mean)`` — mean > 0.
- ``pow2(lo, hi)`` — uniform over the **exponents** of the powers of two
  in ``[lo, hi]`` (DESIGN 5.1: every trace clusters at powers of two, so
  uniform-over-exponents, not uniform-over-values).
- ``lognormal(median, p90 | p99)`` — parameterized by its median and one
  upper quantile: ``mu = ln(median)``, ``sigma = (ln(p_q) - mu) / z_q``
  with ``z_90 = Z90 = 1.2816`` or ``z_99 = Z99 = 2.3263``.  Optionally
  capped at ``cap`` (a hard max applied per draw).
- ``pareto(alpha, xm)`` — Pareto tail exponent ``alpha > 0``, minimum
  ``xm > 0``; one-uniform inverse CDF ``xm * (1-u)^(-1/alpha)``.
- ``weibull(shape, scale)`` — Weibull with ``shape`` k > 0 and
  ``scale`` > 0.
- ``pmf({value: weight, ...})`` — discrete weighted pmf (v0.2 sizes,
  traffic-math §2.2); one uniform against a precomputed CDF.
- ``splice`` — lognormal body + Pareto tail (v0.2 pretrain durations,
  traffic-math §2.3/§3.4): see :class:`SpliceLogNormalPareto`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from ..config import DistSpec

__all__ = [
    "Z90",
    "Z99",
    "Sampler",
    "Fixed",
    "Uniform",
    "Pow2",
    "Exponential",
    "LogNormal",
    "Pareto",
    "Weibull",
    "Pmf",
    "SpliceLogNormalPareto",
    "norm_cdf",
    "norm_ppf",
    "from_spec",
]

#: Standard-normal 90th percentile used by the LogNormal (median, p90)
#: parameterization.  Pinned to 4 decimals — part of the determinism
#: contract (changing it changes every sampled duration).
Z90 = 1.2816

#: Standard-normal 99th percentile for the (median, p99) alternative
#: (traffic-math §2.3).  Pinned to 4 decimals, same contract as Z90.
Z99 = 2.3263


def norm_cdf(x: float) -> float:
    """Standard normal CDF Phi(x) via ``math.erf`` (double precision)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation of the standard normal inverse CDF
# (|relative err| < 1.2e-9) — pinned coefficients, pure stdlib math.
_ACKLAM_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_PLOW = 0.02425


def norm_ppf(p: float) -> float:
    """Standard normal inverse CDF Phi^-1(p) (Acklam's approximation).

    ``p`` is clamped to ``[1e-300, 1 - 1e-16]`` so callers feeding raw
    ``rng.random()`` output (which can be exactly 0.0) never see ``inf``.
    Deterministic, |err| < 1.2e-9 — part of the splice sampler's pinned
    behavior (traffic-math §3.4).
    """
    p = min(max(p, 1e-300), 1.0 - 1e-16)
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    if p < _ACKLAM_PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - _ACKLAM_PLOW:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


@runtime_checkable
class Sampler(Protocol):
    """Anything that can draw one float from a numpy Generator."""

    def sample(self, rng: np.random.Generator) -> float: ...


def _require_finite_number(value: object, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{ctx}: expected a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{ctx}: must be finite, got {value!r}")
    return v


@dataclass(frozen=True, slots=True)
class Fixed:
    """Degenerate distribution: every draw is ``value``."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_finite_number(self.value, "Fixed.value"))

    def sample(self, rng: np.random.Generator) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class Uniform:
    """Continuous uniform on ``[lo, hi]``."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        lo = _require_finite_number(self.lo, "Uniform.lo")
        hi = _require_finite_number(self.hi, "Uniform.hi")
        if lo > hi:
            raise ValueError(f"Uniform requires lo <= hi, got lo={lo}, hi={hi}")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


def _is_pow2(n: object) -> bool:
    return (
        isinstance(n, int)
        and not isinstance(n, bool)
        and n > 0
        and n & (n - 1) == 0
    )


@dataclass(frozen=True, slots=True)
class Pow2:
    """Uniform over the exponents of the powers of two in ``[lo, hi]``.

    ``lo`` and ``hi`` must be powers of two with ``lo <= hi``; a draw is
    ``2 ** e`` with ``e`` integer-uniform on ``[log2(lo), log2(hi)]``
    (inclusive), returned as a float.
    """

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not (_is_pow2(self.lo) and _is_pow2(self.hi)):
            raise ValueError(
                f"Pow2 bounds must be powers of two, got lo={self.lo!r},"
                f" hi={self.hi!r}"
            )
        if self.lo > self.hi:
            raise ValueError(f"Pow2 requires lo <= hi, got lo={self.lo}, hi={self.hi}")

    def sample(self, rng: np.random.Generator) -> float:
        lo_e = self.lo.bit_length() - 1
        hi_e = self.hi.bit_length() - 1
        e = int(rng.integers(lo_e, hi_e + 1))
        return float(1 << e)


@dataclass(frozen=True, slots=True)
class Exponential:
    """Exponential with the given mean (> 0)."""

    mean: float

    def __post_init__(self) -> None:
        mean = _require_finite_number(self.mean, "Exponential.mean")
        if mean <= 0:
            raise ValueError(f"Exponential mean must be positive, got {mean}")
        object.__setattr__(self, "mean", mean)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.mean))


@dataclass(frozen=True, slots=True)
class LogNormal:
    """Lognormal parameterized by (median, p90 | p99), optionally capped.

    ``mu = ln(median)``; ``sigma = (ln(p_q) - mu) / z_q`` with
    ``z_90 = Z90`` or ``z_99 = Z99`` — exactly one of ``p90`` / ``p99``
    must be given (traffic-math §2.3).  ``p_q == median`` degenerates to
    a point mass at the median.  When ``cap`` is set, each draw is
    ``min(draw, cap)`` (a hard max — DESIGN 5.1's "lognormal-body, tail
    to N days" durations).
    """

    median: float
    p90: float | None = None
    cap: float | None = None
    p99: float | None = None

    def __post_init__(self) -> None:
        median = _require_finite_number(self.median, "LogNormal.median")
        if (self.p90 is None) == (self.p99 is None):
            raise ValueError(
                "LogNormal requires exactly one of p90 | p99,"
                f" got p90={self.p90!r}, p99={self.p99!r}"
            )
        qname = "p90" if self.p90 is not None else "p99"
        q = _require_finite_number(
            self.p90 if self.p90 is not None else self.p99, f"LogNormal.{qname}"
        )
        if median <= 0 or q <= 0:
            raise ValueError(
                f"LogNormal median and {qname} must be positive,"
                f" got median={median}, {qname}={q}"
            )
        if q < median:
            raise ValueError(
                f"LogNormal requires {qname} >= median,"
                f" got median={median}, {qname}={q}"
            )
        if self.cap is not None:
            cap = _require_finite_number(self.cap, "LogNormal.cap")
            if cap <= 0:
                raise ValueError(f"LogNormal cap must be positive, got {cap}")
            object.__setattr__(self, "cap", cap)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, qname, q)

    @property
    def sigma(self) -> float:
        """The derived lognormal shape parameter (float, dimensionless)."""
        mu = math.log(self.median)
        if self.p90 is not None:
            return (math.log(self.p90) - mu) / Z90
        return (math.log(self.p99) - mu) / Z99

    def sample(self, rng: np.random.Generator) -> float:
        mu = math.log(self.median)
        draw = float(rng.lognormal(mu, self.sigma))
        if self.cap is not None:
            draw = min(draw, self.cap)
        return draw


@dataclass(frozen=True, slots=True)
class Pareto:
    """Pareto with tail exponent ``alpha`` and minimum ``xm`` (both > 0).

    One-uniform inverse CDF: ``xm * (1 - u)^(-1/alpha)`` — draws are in
    ``[xm, inf)``.  Cap it via ``from_spec(cap=...)`` (or the class
    ``max_lifetime``): untruncated ``alpha <= 1.2`` tails make sample
    means non-convergent (traffic-math §2.3).
    """

    alpha: float
    xm: float

    def __post_init__(self) -> None:
        alpha = _require_finite_number(self.alpha, "Pareto.alpha")
        xm = _require_finite_number(self.xm, "Pareto.xm")
        if alpha <= 0 or xm <= 0:
            raise ValueError(
                f"Pareto alpha and xm must be positive, got alpha={alpha}, xm={xm}"
            )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "xm", xm)

    def sample(self, rng: np.random.Generator) -> float:
        u = float(rng.random())
        return self.xm * (1.0 - u) ** (-1.0 / self.alpha)


@dataclass(frozen=True, slots=True)
class Weibull:
    """Weibull with ``shape`` k and ``scale`` lambda (both > 0).

    Mean = ``scale * Gamma(1 + 1/shape)``; one variate per draw.
    """

    shape: float
    scale: float

    def __post_init__(self) -> None:
        shape = _require_finite_number(self.shape, "Weibull.shape")
        scale = _require_finite_number(self.scale, "Weibull.scale")
        if shape <= 0 or scale <= 0:
            raise ValueError(
                f"Weibull shape and scale must be positive,"
                f" got shape={shape}, scale={scale}"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "scale", scale)

    def sample(self, rng: np.random.Generator) -> float:
        return self.scale * float(rng.weibull(self.shape))


@dataclass(frozen=True, slots=True)
class Pmf:
    """Discrete weighted pmf sampled with ONE uniform against a
    precomputed CDF (traffic-math §2.2 — never ``rng.choice`` per draw).

    ``values`` ascend; ``cum`` is the matching cumulative distribution
    with ``cum[-1] == 1.0``.  Build from a ``{value: weight}`` mapping
    with :meth:`from_weights` (weights normalized there).
    """

    values: tuple[float, ...]
    cum: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values or len(self.values) != len(self.cum):
            raise ValueError(
                "Pmf requires equal-length non-empty values and cum"
            )
        if list(self.values) != sorted(self.values):
            raise ValueError(f"Pmf values must ascend, got {self.values}")
        if abs(self.cum[-1] - 1.0) > 1e-9:
            raise ValueError(f"Pmf cum must end at 1.0, got {self.cum[-1]}")

    @classmethod
    def from_weights(cls, weights: Mapping[int | float, float]) -> "Pmf":
        """Build from ``{value: weight}``; weights must be positive and
        are normalized here.  Values are sorted ascending (deterministic
        regardless of mapping order)."""
        if not weights:
            raise ValueError("Pmf requires at least one entry")
        items = sorted((float(v), float(w)) for v, w in weights.items())
        total = sum(w for _, w in items)
        if total <= 0 or any(w <= 0 for _, w in items):
            raise ValueError(f"Pmf weights must be positive, got {dict(weights)}")
        cum: list[float] = []
        acc = 0.0
        for _, w in items:
            acc += w / total
            cum.append(acc)
        cum[-1] = 1.0  # guard float accumulation
        return cls(tuple(v for v, _ in items), tuple(cum))

    def sample(self, rng: np.random.Generator) -> float:
        u = float(rng.random())
        i = min(bisect_right(self.cum, u), len(self.values) - 1)
        return self.values[i]


@dataclass(frozen=True, slots=True)
class SpliceLogNormalPareto:
    """Lognormal body + Pareto tail splice (traffic-math §2.3/§3.4).

    Density ``w * f_LN(x)/F_LN(theta)`` for ``x <= theta`` and
    ``(1-w) * alpha*theta^alpha/x^(alpha+1)`` above, truncated at ``cap``
    by renormalized inversion (doc §3.4 pseudocode — normative).  The
    continuity weight ``w = (alpha/theta) / (f_LN(theta)/F_LN(theta) +
    alpha/theta)`` is COMPUTED, never configured; it makes the density
    continuous at ``theta`` for the untruncated tail (exact continuity
    when ``cap >> theta``; a nearby cap raises the tail side by the
    truncation factor ``1/q``).

    Fixed TWO uniforms per draw (branch + inversion, Acklam Phi^-1) —
    no rejection loops, preserving per-stream draw accounting.
    Invariants: draws lie in ``(0, cap]``; ``P(X <= theta) == w``.
    Build via :meth:`from_quantiles`.
    """

    mu: float
    sigma: float
    theta: float
    alpha: float
    cap: float
    w: float
    f_theta: float  # F_LN(theta), the body CDF at the splice point

    def __post_init__(self) -> None:
        if self.alpha <= 1.0:
            raise ValueError(
                f"splice tail alpha must be > 1, got {self.alpha}"
            )
        if not (0.0 < self.theta < self.cap):
            raise ValueError(
                f"splice requires 0 < theta < cap,"
                f" got theta={self.theta}, cap={self.cap}"
            )
        if self.sigma < 0:
            raise ValueError(f"splice sigma must be >= 0, got {self.sigma}")

    @classmethod
    def from_quantiles(
        cls,
        median: float,
        alpha: float,
        cap: float,
        *,
        p90: float | None = None,
        p99: float | None = None,
        splice_q: float | None = None,
        splice_at: float | None = None,
    ) -> "SpliceLogNormalPareto":
        """Build from body quantiles plus the splice point.

        ``median`` and exactly one of ``p90`` / ``p99`` pin the body;
        the splice point ``theta`` is the body quantile ``splice_q``
        (e.g. 0.9) or the explicit value ``splice_at`` (exactly one).
        All value-typed args share one unit (seconds for durations).
        """
        body = LogNormal(median, p90, p99=p99)  # validates the quantile pair
        mu = math.log(median)
        sigma = body.sigma
        if (splice_q is None) == (splice_at is None):
            raise ValueError(
                "splice requires exactly one of splice_q | splice_at"
            )
        if splice_q is not None:
            if not 0.0 < splice_q < 1.0:
                raise ValueError(f"splice_q must be in (0, 1), got {splice_q}")
            theta = math.exp(mu + sigma * norm_ppf(splice_q))
        else:
            theta = float(splice_at)
        if sigma <= 0:
            raise ValueError("splice requires a non-degenerate body (sigma > 0)")
        z_theta = (math.log(theta) - mu) / sigma
        f_theta = norm_cdf(z_theta)
        if f_theta <= 0.0:
            raise ValueError(
                f"splice point theta={theta} is below the body's support"
            )
        # Lognormal pdf at theta over CDF at theta (the body hazard term).
        pdf_theta = (
            math.exp(-0.5 * z_theta * z_theta)
            / (theta * sigma * math.sqrt(2.0 * math.pi))
        )
        w = (alpha / theta) / (pdf_theta / f_theta + alpha / theta)
        return cls(
            mu=mu, sigma=sigma, theta=theta, alpha=float(alpha),
            cap=float(cap), w=w, f_theta=f_theta,
        )

    def sample(self, rng: np.random.Generator) -> float:
        u1 = float(rng.random())
        u2 = float(rng.random())
        if u1 < self.w:  # body: truncated lognormal on (0, theta]
            return math.exp(
                self.mu + self.sigma * norm_ppf(u2 * self.f_theta)
            )
        # tail: Pareto truncated at cap, exact inversion
        q = 1.0 - (self.theta / self.cap) ** self.alpha
        return self.theta * (1.0 - u2 * q) ** (-1.0 / self.alpha)


@dataclass(frozen=True, slots=True)
class _Capped:
    """Wrap any sampler with a hard per-draw max (used by :func:`from_spec`
    when ``cap`` is given for a non-lognormal kind)."""

    inner: Sampler
    cap: float

    def sample(self, rng: np.random.Generator) -> float:
        return min(self.inner.sample(rng), self.cap)


def from_spec(
    spec: DistSpec, *, scale: float = 1.0, cap: float | None = None
) -> Sampler:
    """Build a sampler from a parsed :class:`~fleetsim.config.DistSpec`.

    ``scale`` multiplies every parameter at construction — pass ``1e-6``
    for duration specs (whose params are int microseconds) to obtain a
    sampler in **seconds**.  ``cap`` is a hard per-draw maximum in the
    *output* unit (i.e. after scaling).  ``pow2`` and ``pmf`` cannot be
    scaled (their values are chip counts) and reject ``scale != 1.0``.

    Raises ``ValueError`` for the ``"invalid"`` sentinel kind, unknown
    kinds, missing parameters, or parameter values the sampler rejects.
    """
    if spec.kind == "invalid":
        raise ValueError(
            "cannot build a sampler from an invalid DistSpec"
            " (the parse error was recorded at load time)"
        )
    p = spec.params
    try:
        if spec.kind == "fixed":
            sampler: Sampler = Fixed(p["value"] * scale)
        elif spec.kind == "uniform":
            sampler = Uniform(p["lo"] * scale, p["hi"] * scale)
        elif spec.kind == "exponential":
            sampler = Exponential(p["mean"] * scale)
        elif spec.kind == "pow2":
            if scale != 1.0:
                raise ValueError(
                    f"pow2 bounds are chip counts and cannot be scaled"
                    f" (got scale={scale!r})"
                )
            sampler = Pow2(p["lo"], p["hi"])
        elif spec.kind == "pmf":
            if scale != 1.0:
                raise ValueError(
                    f"pmf values are chip counts and cannot be scaled"
                    f" (got scale={scale!r})"
                )
            sampler = Pmf.from_weights(
                {int(k): float(v) for k, v in p.items()}
            )
        elif spec.kind == "pareto":
            sampler = Pareto(p["alpha"], p["xm"] * scale)
        elif spec.kind == "weibull":
            sampler = Weibull(p["shape"], p["scale"] * scale)
        elif spec.kind == "lognormal":
            p90 = p.get("p90")
            p99 = p.get("p99")
            if p90 is None and p99 is None:
                raise ValueError(
                    "distribution 'lognormal' is missing parameter 'p90'"
                    " (or 'p99')"
                )
            return LogNormal(
                p["median"] * scale,
                p90 * scale if p90 is not None else None,
                cap=cap,
                p99=p99 * scale if p99 is not None else None,
            )
        elif spec.kind == "splice":
            sampler = SpliceLogNormalPareto.from_quantiles(
                p["median"] * scale,
                p["alpha"],
                p["cap"] * scale,
                p90=p["p90"] * scale if "p90" in p else None,
                p99=p["p99"] * scale if "p99" in p else None,
                splice_q=p.get("splice_q"),
                splice_at=(
                    p["splice_at"] * scale if "splice_at" in p else None
                ),
            )
        else:
            raise ValueError(f"unknown distribution kind {spec.kind!r}")
    except KeyError as exc:
        raise ValueError(
            f"distribution {spec.kind!r} is missing parameter {exc.args[0]!r}"
        ) from None
    if cap is not None:
        return _Capped(sampler, _require_finite_number(cap, "cap"))
    return sampler
