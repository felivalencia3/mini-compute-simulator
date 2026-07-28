"""Event types, the event record, and the tombstoning event queue.

DESIGN 6.1: time is **int64 microseconds** since sim epoch; the queue is a
``heapq`` keyed ``(time, type, seq)``.  The :class:`EventType` value IS the
same-timestamp ordering rank — all completions/failures at time *t* land
before ``SCHED_WAKE`` at *t*, so the scheduler always sees settled state;
``seq`` makes the ordering total (and deterministic across platforms).

UNITS: ``Event.time`` is int microseconds since sim epoch.

INVARIANTS
----------
- ``push`` returns a strictly increasing ``seq``; two events never compare
  equal, so heap order is total and deterministic.
- ``cancel(seq)`` tombstones lazily: the event stays in the heap and is
  skipped (and its tombstone discarded) when it reaches the front.  This is
  DESIGN's "lazy completions" mechanic — preemption/failure cancels the
  stale ``JOB_COMPLETION`` in O(1).
- Cancelling a seq that was already popped is a silent no-op (the tombstone
  entry lingers; callers clear their stored seqs on fire to avoid this).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import IntEnum

__all__ = ["EventType", "Event", "EventQueue"]


class EventType(IntEnum):
    """Event kinds; the value is the same-timestamp ordering rank
    (DESIGN 6.1, exact)."""

    NODE_REPAIR = 0
    JOB_COMPLETION = 1
    NODE_FAILURE = 2
    PREEMPTION_DONE = 3
    JOB_ARRIVAL = 4
    MAINTENANCE_DRAIN = 5
    JOB_TIMEOUT = 6
    SCHED_WAKE = 7
    METRICS_FLUSH = 8


@dataclass(frozen=True, slots=True)
class Event:
    """One scheduled event.  ``time`` int microseconds; ``seq`` is the
    queue-assigned total-order tiebreaker; ``payload`` is event-specific
    (see ``fleetsim.engine.sim`` for the payload conventions)."""

    time: int
    type: EventType
    seq: int
    payload: object = None


class EventQueue:
    """Min-heap of events keyed ``(time, type, seq)`` with O(1) tombstone
    cancellation.

    INVARIANTS: ``pop``/``peek_time``/``empty`` all agree — a tombstoned
    event is invisible to every query.  ``seq`` values are assigned in push
    order starting at 0 and never reused.
    """

    __slots__ = ("_heap", "_next_seq", "_cancelled")

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, Event]] = []
        self._next_seq: int = 0
        self._cancelled: set[int] = set()

    def push(self, time: int, type: EventType, payload: object = None) -> int:
        """Schedule an event; returns its ``seq`` (for later ``cancel``)."""
        if time < 0:
            raise ValueError(f"event time must be non-negative, got {time}")
        seq = self._next_seq
        self._next_seq = seq + 1
        ev = Event(time=time, type=type, seq=seq, payload=payload)
        heapq.heappush(self._heap, (time, int(type), seq, ev))
        return seq

    def cancel(self, seq: int) -> None:
        """Tombstone the event with this ``seq``; it will be skipped."""
        self._cancelled.add(seq)

    def _purge(self) -> None:
        """Drop tombstoned events sitting at the front of the heap."""
        while self._heap and self._heap[0][2] in self._cancelled:
            self._cancelled.discard(self._heap[0][2])
            heapq.heappop(self._heap)

    def pop(self) -> Event:
        """Remove and return the earliest live event (IndexError if empty)."""
        self._purge()
        if not self._heap:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._heap)[3]

    def peek_time(self) -> int | None:
        """Time of the earliest live event, or ``None`` when empty."""
        self._purge()
        return self._heap[0][0] if self._heap else None

    def empty(self) -> bool:
        """True when no live (non-tombstoned) events remain."""
        self._purge()
        return not self._heap

    def __len__(self) -> int:
        """Number of live events (purges front tombstones only, so this is
        an upper bound if tombstones are buried; exact after full drain)."""
        self._purge()
        return len(self._heap) - len(
            self._cancelled.intersection(entry[2] for entry in self._heap)
        )
