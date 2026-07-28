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
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..model import Job

__all__ = ["JobSource", "ListSource"]


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
