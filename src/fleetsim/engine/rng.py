"""Named independent RNG streams (DESIGN 6.2, determinism contract).

Every run of the simulator is a pure function of ``(scenario, seed)``.
:class:`RngStreams` derives one independent NumPy ``Generator`` per named
stream from a single root seed via ``SeedSequence`` spawn keys, so enabling
one stochastic subsystem can never perturb another: draws from
``stream("failures")`` never change the sequence ``stream("arrivals")``
yields, no matter how many are taken or in what order the streams are
created.

DETERMINISM CONTRACT
--------------------
**Stream names are part of the public determinism contract.**  A stream's
sequence is a pure function of ``(seed_for_that_stream, name)`` — renaming
a stream, like changing the seed, is a behavior change and must be treated
as such.  The engine's reserved names in v0.1: ``"failures"``, ``"repair"``,
``"maintenance"``; the workload generator adds ``"arrivals"``,
``"job_size"``, ``"job_duration"`` (DESIGN 6.2), plus per-entity streams
keyed by stable strings.  Name derivation uses SHA-256 (never Python's
salted ``hash``), so sequences are identical across platforms and runs.

``overrides`` re-seeds individual streams without touching the others —
this is how A/B experiments vary, say, only the failure realization while
keeping the arrival sequence as a paired sample.

INVARIANTS: no global mutable state, no wall clock; ``stream(name)``
returns the SAME ``Generator`` object for repeated calls with one name
(streams are stateful cursors — callers share position by design).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

__all__ = ["RngStreams"]


def _name_spawn_key(name: str) -> tuple[int, ...]:
    """Stable 128-bit spawn key for a stream name (four uint32 words from
    SHA-256; platform- and process-independent)."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()[:16]
    return tuple(
        int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)
    )


class RngStreams:
    """Factory and cache of named, independent ``numpy.random.Generator``\\ s.

    ``seed`` is the root seed for every stream; ``overrides`` maps stream
    names to replacement seeds (only those streams change — the paired-
    experiment knob).  Two ``RngStreams`` with equal ``(seed, overrides)``
    produce byte-identical draw sequences per name.
    """

    __slots__ = ("_seed", "_overrides", "_streams")

    def __init__(self, seed: int, overrides: Mapping[str, int] | None = None):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {seed!r}")
        self._seed = seed
        self._overrides: dict[str, int] = dict(overrides or {})
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        """The root seed (overrides excluded)."""
        return self._seed

    def stream(self, name: str) -> np.random.Generator:
        """The named stream, created on first use and cached by name.

        The generator is seeded by ``SeedSequence(seed_for_name,
        spawn_key=sha256(name))`` where ``seed_for_name`` is
        ``overrides.get(name, seed)`` — independent across names, stable
        across platforms.
        """
        gen = self._streams.get(name)
        if gen is None:
            entropy = self._overrides.get(name, self._seed)
            ss = np.random.SeedSequence(entropy, spawn_key=_name_spawn_key(name))
            gen = np.random.default_rng(ss)
            self._streams[name] = gen
        return gen
