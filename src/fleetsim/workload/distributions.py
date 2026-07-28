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
- ``lognormal(median, p90)`` — parameterized by its median and 90th
  percentile: ``mu = ln(median)``, ``sigma = (ln(p90) - mu) / Z90`` with
  ``Z90 = 1.2816`` (the standard-normal 90th percentile).  Optionally
  capped at ``cap`` (a hard max applied per draw).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import DistSpec

__all__ = [
    "Z90",
    "Sampler",
    "Fixed",
    "Uniform",
    "Pow2",
    "Exponential",
    "LogNormal",
    "from_spec",
]

#: Standard-normal 90th percentile used by the LogNormal (median, p90)
#: parameterization.  Pinned to 4 decimals — part of the determinism
#: contract (changing it changes every sampled duration).
Z90 = 1.2816


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
    """Lognormal parameterized by (median, p90), optionally capped.

    ``mu = ln(median)``; ``sigma = (ln(p90) - mu) / Z90``.  ``p90 ==
    median`` degenerates to a point mass at the median.  When ``cap`` is
    set, each draw is ``min(draw, cap)`` (a hard max — DESIGN 5.1's
    "lognormal-body, tail to N days" durations).
    """

    median: float
    p90: float
    cap: float | None = None

    def __post_init__(self) -> None:
        median = _require_finite_number(self.median, "LogNormal.median")
        p90 = _require_finite_number(self.p90, "LogNormal.p90")
        if median <= 0 or p90 <= 0:
            raise ValueError(
                f"LogNormal median and p90 must be positive,"
                f" got median={median}, p90={p90}"
            )
        if p90 < median:
            raise ValueError(
                f"LogNormal requires p90 >= median, got median={median}, p90={p90}"
            )
        if self.cap is not None:
            cap = _require_finite_number(self.cap, "LogNormal.cap")
            if cap <= 0:
                raise ValueError(f"LogNormal cap must be positive, got {cap}")
            object.__setattr__(self, "cap", cap)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "p90", p90)

    def sample(self, rng: np.random.Generator) -> float:
        mu = math.log(self.median)
        sigma = (math.log(self.p90) - mu) / Z90
        draw = float(rng.lognormal(mu, sigma))
        if self.cap is not None:
            draw = min(draw, self.cap)
        return draw


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
    *output* unit (i.e. after scaling).  ``pow2`` cannot be scaled (its
    bounds are chip counts) and rejects ``scale != 1.0``.

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
        elif spec.kind == "lognormal":
            return LogNormal(p["median"] * scale, p["p90"] * scale, cap=cap)
        else:
            raise ValueError(f"unknown distribution kind {spec.kind!r}")
    except KeyError as exc:
        raise ValueError(
            f"distribution {spec.kind!r} is missing parameter {exc.args[0]!r}"
        ) from None
    if cap is not None:
        return _Capped(sampler, _require_finite_number(cap, "cap"))
    return sampler
