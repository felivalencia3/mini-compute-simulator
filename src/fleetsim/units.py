"""Integer-microsecond time units, and duration parsing/formatting.

UNITS
-----
The simulator's canonical time unit is the **int microsecond** (months of
simulated time fit comfortably in int64).  Every constant in this module is
an int microsecond count.  Throughout fleetsim, a time-valued field is int
microseconds unless its name ends in ``_s`` (float seconds).

INVARIANTS
----------
- ``parse_duration`` returns a non-negative ``int`` (microseconds) or raises
  ``ValueError``/``TypeError``.  It never returns a float and never uses
  wall-clock state; it is a pure function.
- ``format_duration`` always emits an *exact* single-unit representation
  (largest unit that divides the value), so
  ``parse_duration(format_duration(x)) == x`` for every non-negative int
  ``x``.
"""

from __future__ import annotations

import math

__all__ = [
    "US",
    "MS",
    "S",
    "MIN",
    "HOUR",
    "DAY",
    "WEEK",
    "parse_duration",
    "format_duration",
]

US: int = 1
MS: int = 1_000 * US
S: int = 1_000 * MS
MIN: int = 60 * S
HOUR: int = 60 * MIN
DAY: int = 24 * HOUR
WEEK: int = 7 * DAY

# Suffix table for parsing.  Two-character suffixes MUST come before the
# single-character ones ("500us" ends with both "us" and "s").
_PARSE_SUFFIXES: tuple[tuple[str, int], ...] = (
    ("us", US),
    ("ms", MS),
    ("s", S),
    ("m", MIN),
    ("h", HOUR),
    ("d", DAY),
    ("w", WEEK),
)

# Format table: largest unit first; US divides everything, so the scan
# always terminates with an exact representation.
_FORMAT_UNITS: tuple[tuple[str, int], ...] = (
    ("w", WEEK),
    ("d", DAY),
    ("h", HOUR),
    ("m", MIN),
    ("s", S),
    ("ms", MS),
    ("us", US),
)


def parse_duration(value: int | float | str) -> int:
    """Parse a duration into int microseconds.

    Accepted forms:

    - ``int`` / ``float``: interpreted as **seconds** (``60`` -> 60 s).
    - ``str`` with a unit suffix ``us|ms|s|m|h|d|w``: ``"60s"``, ``"2m"``,
      ``"1.5h"``, ``"14d"``, ``"250us"``.  ``m`` is minutes.
    - bare numeric ``str``: interpreted as seconds (``"90"`` -> 90 s).

    Raises ``ValueError`` for negative, non-finite, or unparseable input,
    and ``TypeError`` for unsupported types (including ``bool``, which YAML
    would otherwise smuggle in as an int).
    """
    if isinstance(value, bool):
        raise TypeError(f"cannot parse duration from bool: {value!r}")
    if isinstance(value, (int, float)):
        return _seconds_to_us(value, original=value)
    if isinstance(value, str):
        text = value.strip()
        for suffix, mult in _PARSE_SUFFIXES:
            if text.endswith(suffix):
                num_text = text[: -len(suffix)].strip()
                if not num_text:
                    raise ValueError(f"cannot parse duration: {value!r}")
                try:
                    num = float(num_text)
                except ValueError:
                    raise ValueError(f"cannot parse duration: {value!r}") from None
                return _scaled_to_us(num, mult, original=value)
        # No unit suffix: bare number string, interpreted as seconds.
        try:
            num = float(text)
        except ValueError:
            raise ValueError(f"cannot parse duration: {value!r}") from None
        return _seconds_to_us(num, original=value)
    raise TypeError(f"cannot parse duration from {type(value).__name__}: {value!r}")


def _seconds_to_us(seconds: int | float, *, original: object) -> int:
    return _scaled_to_us(seconds, S, original=original)


def _scaled_to_us(num: float, mult: int, *, original: object) -> int:
    if not math.isfinite(num):
        raise ValueError(f"duration must be finite: {original!r}")
    if num < 0:
        raise ValueError(f"duration must be non-negative: {original!r}")
    if isinstance(num, int):
        return num * mult
    return round(num * mult)


def format_duration(us: int) -> str:
    """Format int microseconds as a human string, e.g. ``"90m"``, ``"14d"``.

    Uses the largest unit that divides the value exactly, so the output
    always round-trips: ``parse_duration(format_duration(x)) == x``.
    Note ``format_duration(14 * DAY) == "2w"`` — same value, larger unit.
    """
    if isinstance(us, bool) or not isinstance(us, int):
        raise TypeError(f"format_duration expects int microseconds, got {us!r}")
    if us < 0:
        raise ValueError(f"duration must be non-negative: {us!r}")
    if us == 0:
        return "0s"
    for suffix, mult in _FORMAT_UNITS:
        if us % mult == 0:
            return f"{us // mult}{suffix}"
    raise AssertionError("unreachable: US divides every int")
