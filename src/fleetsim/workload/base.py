"""Job sources: the pull interface the engine drives arrivals through.

A :class:`JobSource` is an iterator-style supplier of jobs.  The engine
pulls **lazily**: it asks for one arrival, schedules a single
``JOB_ARRIVAL`` event for it, and asks for the next only after that event
has been consumed.  Synthetic generators and trace replayers implement the
same protocol (DESIGN §10), so schedulers cannot tell replay from
synthesis.

UNITS: arrival times are int microseconds since sim epoch.

INVARIANTS
----------
- ``next_arrival`` returns ``(time_us, job)`` pairs with **non-decreasing**
  ``time_us`` (the engine raises if time goes backwards), or ``None`` when
  the source is exhausted (it must keep returning ``None`` afterwards).
- The returned ``time_us`` is the job's arrival time; sources emit jobs
  whose ``submit_t`` equals it.
- Sources own any randomness they use (seeded streams from
  :class:`~fleetsim.engine.rng.RngStreams`); the protocol itself is pull-
  only and side-effect free from the engine's point of view.

REFILL (closed-loop backlog hook, v0.2)
---------------------------------------
A source MAY additionally implement::

    def refill(self, now_us: int, pending_by_class: dict[str, int]) -> list[Job]

The engine calls it at EVERY ``SCHED_WAKE``, after all same-timestamp
state changes have settled and **before** the scheduler is invoked.
``pending_by_class`` maps a class label to the number of currently
pending (queued, admission-passed) jobs with that label, where a job's
label is ``job.source_class`` when set, else ``job.job_class.name``;
labels with zero pending jobs are absent.  Every returned job is
submitted at ``now_us`` through the normal admission path (all metrics
sink calls included) and is visible to the scheduler in the same wake —
this is how standing-backlog classes keep N jobs pending at all times.
Returned jobs must have ``submit_t == now_us``.  Determinism: ``refill``
must draw only from the source's own named RNG streams.  Sources without
the method are plain open-loop sources (the engine probes with
``getattr``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from ..model import Job

__all__ = ["JobSource", "ListSource", "MergedSource"]


@runtime_checkable
class JobSource(Protocol):
    """Anything that can hand the engine the next arriving job."""

    def next_arrival(self) -> tuple[int, Job] | None:
        """The next ``(time_us, job)`` arrival, or ``None`` when exhausted."""
        ...


class ListSource:
    """A finite, in-memory job source for tests and tiny scenarios.

    Jobs are emitted in ``(submit_t, id)`` order (sorted here, so callers
    may pass them unordered); each job is emitted exactly once with arrival
    time ``job.submit_t``.
    """

    __slots__ = ("_jobs", "_i")

    def __init__(self, jobs: Iterable[Job]):
        self._jobs: list[Job] = sorted(jobs, key=lambda j: (j.submit_t, j.id))
        self._i = 0

    def next_arrival(self) -> tuple[int, Job] | None:
        if self._i >= len(self._jobs):
            return None
        job = self._jobs[self._i]
        self._i += 1
        return (job.submit_t, job)


class MergedSource:
    """Combine several :class:`JobSource`\\ s into one (v0.2).

    ``next_arrival`` merges the children's arrival streams lazily in time
    order (one-item lookahead per child; ties break by child position, so
    the merge is deterministic given deterministic children).  ``refill``
    concatenates the children's refills in child order — children without
    the method are skipped — so one scenario can mix open-loop (Poisson /
    trace) and closed-loop (standing backlog) sources behind a single
    source object.

    INVARIANTS: each child's own non-decreasing-time contract is
    preserved; the merged stream is globally non-decreasing iff every
    child's is.  Job ids must be globally unique across children (the
    engine rejects duplicates).
    """

    __slots__ = ("_sources", "_peek")

    def __init__(self, sources: Sequence[JobSource]):
        self._sources: tuple[JobSource, ...] = tuple(sources)
        #: Per-child lookahead: (time_us, job) or None once exhausted.
        self._peek: list[tuple[int, Job] | None] = [
            src.next_arrival() for src in self._sources
        ]

    def next_arrival(self) -> tuple[int, Job] | None:
        best_i = -1
        best_t = 0
        for i, peek in enumerate(self._peek):
            if peek is None:
                continue
            if best_i < 0 or peek[0] < best_t:
                best_i, best_t = i, peek[0]
        if best_i < 0:
            return None
        out = self._peek[best_i]
        self._peek[best_i] = self._sources[best_i].next_arrival()
        return out

    def refill(
        self, now_us: int, pending_by_class: dict[str, int]
    ) -> list[Job]:
        """Concatenated child refills, child order (see class docstring)."""
        jobs: list[Job] = []
        for src in self._sources:
            refill = getattr(src, "refill", None)
            if refill is not None:
                jobs.extend(refill(now_us, pending_by_class))
        return jobs
