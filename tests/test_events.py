"""Tests for fleetsim.engine.events: (time, type, seq) ordering, tombstone
cancellation, and queue queries."""

import pytest

from fleetsim.engine.events import Event, EventQueue, EventType


def test_event_type_ranks_exact():
    # DESIGN 6.1: value IS the same-timestamp ordering rank.
    assert EventType.NODE_REPAIR == 0
    assert EventType.JOB_COMPLETION == 1
    assert EventType.NODE_FAILURE == 2
    assert EventType.PREEMPTION_DONE == 3
    assert EventType.JOB_ARRIVAL == 4
    assert EventType.MAINTENANCE_DRAIN == 5
    assert EventType.JOB_TIMEOUT == 6
    assert EventType.SCHED_WAKE == 7
    assert EventType.METRICS_FLUSH == 8
    assert len(EventType) == 9


def test_event_is_frozen():
    ev = Event(time=1, type=EventType.JOB_ARRIVAL, seq=0, payload="x")
    with pytest.raises(AttributeError):
        ev.time = 2


def test_pop_orders_by_time():
    q = EventQueue()
    q.push(300, EventType.JOB_ARRIVAL, "c")
    q.push(100, EventType.JOB_ARRIVAL, "a")
    q.push(200, EventType.JOB_ARRIVAL, "b")
    assert [q.pop().payload for _ in range(3)] == ["a", "b", "c"]


def test_same_time_orders_by_type_rank():
    q = EventQueue()
    q.push(50, EventType.SCHED_WAKE, "wake")
    q.push(50, EventType.JOB_COMPLETION, "done")
    q.push(50, EventType.NODE_FAILURE, "boom")
    q.push(50, EventType.NODE_REPAIR, "fixed")
    q.push(50, EventType.METRICS_FLUSH, "flush")
    order = [q.pop().payload for _ in range(5)]
    # Completions/failures land before the wake; flush after (DESIGN 6.1).
    assert order == ["fixed", "done", "boom", "wake", "flush"]


def test_same_time_and_type_orders_by_seq():
    q = EventQueue()
    s1 = q.push(10, EventType.JOB_ARRIVAL, "first")
    s2 = q.push(10, EventType.JOB_ARRIVAL, "second")
    assert s2 > s1  # push returns strictly increasing seqs
    assert [q.pop().payload, q.pop().payload] == ["first", "second"]


def test_pop_returns_full_event():
    q = EventQueue()
    seq = q.push(7, EventType.JOB_TIMEOUT, ("lifetime", "j1"))
    ev = q.pop()
    assert ev == Event(time=7, type=EventType.JOB_TIMEOUT, seq=seq,
                       payload=("lifetime", "j1"))


def test_cancel_tombstones_event():
    q = EventQueue()
    q.push(1, EventType.JOB_ARRIVAL, "keep1")
    victim = q.push(2, EventType.JOB_ARRIVAL, "cancelled")
    q.push(3, EventType.JOB_ARRIVAL, "keep2")
    q.cancel(victim)
    assert [q.pop().payload, q.pop().payload] == ["keep1", "keep2"]
    assert q.empty()


def test_peek_time_skips_tombstones():
    q = EventQueue()
    victim = q.push(5, EventType.JOB_COMPLETION, "x")
    q.push(9, EventType.JOB_ARRIVAL, "y")
    assert q.peek_time() == 5
    q.cancel(victim)
    assert q.peek_time() == 9


def test_empty_accounts_for_tombstones():
    q = EventQueue()
    assert q.empty()
    assert q.peek_time() is None
    victim = q.push(1, EventType.JOB_ARRIVAL, None)
    assert not q.empty()
    q.cancel(victim)
    assert q.empty()
    with pytest.raises(IndexError):
        q.pop()


def test_cancel_all_then_reuse():
    q = EventQueue()
    seqs = [q.push(t, EventType.JOB_ARRIVAL, t) for t in (1, 2, 3)]
    for s in seqs:
        q.cancel(s)
    assert q.empty()
    q.push(10, EventType.SCHED_WAKE, "later")
    assert q.pop().payload == "later"


def test_negative_time_rejected():
    q = EventQueue()
    with pytest.raises(ValueError):
        q.push(-1, EventType.JOB_ARRIVAL, None)
