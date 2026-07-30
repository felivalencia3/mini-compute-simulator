"""Unit tests for the SJF scheduler (v0.6, checklist item 1).

These test the ordering key and placement/scan semantics in isolation
against a hand-built :class:`FakeView` (no engine): SJF is a pure
function of the view, so the ordered ``Place`` list is directly
assertable.  The end-to-end SPT-optimality property (SJF mean JCT <=
FIFO) is proven analytically against the real engine in
``validation/test_sjf_ordering.py``.
"""

from __future__ import annotations

from fleetsim.model import JobClass, Tier
from fleetsim.schedulers.base import (
    JobView,
    Place,
    get_scheduler,
    registered_schedulers,
)
from fleetsim.schedulers.sjf import SJFScheduler


def mk_view(job_id: str, submit_time: int, est: float | None) -> JobView:
    """A minimal :class:`JobView` — only ``id``/``submit_time``/
    ``walltime_est_s`` matter to SJF ordering; the rest are inert."""
    return JobView(
        id=job_id,
        submit_time=submit_time,
        chips=8,
        chip_type=None,
        tier=Tier.BATCH,
        job_class=JobClass.FINETUNE,
        preemptible=True,
        min_runtime_s=0.0,
        attained_service_chip_s=0.0,
        checkpoint_age_s=0.0,
        walltime_est_s=est,
        within=None,
        tenant="t0",
    )


class _Sentinel:
    """A stand-in Placement (SJF only checks ``is not None``)."""


class FakeView:
    """Implements the two methods SJF touches: ``pending()`` and
    ``find_placement``.  ``unplaceable`` names jobs whose placement
    returns ``None``; every other job gets a fresh sentinel placement."""

    def __init__(self, jobs: list[JobView], unplaceable: set[str] | None = None):
        self._jobs = jobs
        self._unplaceable = unplaceable or set()
        self.placement_calls: list[str] = []

    @property
    def now(self) -> int:  # pragma: no cover - unused by SJF
        return 0

    def pending(self) -> list[JobView]:
        # Engine contract: pending() is pre-sorted (submit_time, id).
        return sorted(self._jobs, key=lambda j: (j.submit_time, j.id))

    def find_placement(self, job: JobView, policy):
        self.placement_calls.append(job.id)
        if job.id in self._unplaceable:
            return None
        return _Sentinel()


def placed_order(actions: list) -> list[str]:
    return [a.job_id for a in actions if isinstance(a, Place)]


def test_registered_and_default_best_effort():
    assert "sjf" in registered_schedulers()
    sched = get_scheduler("sjf", {})
    assert isinstance(sched, SJFScheduler)
    # Default is best-effort (skip-and-continue), NOT strict.
    assert sched.strict is False


def test_orders_shortest_estimate_first():
    jobs = [
        mk_view("j0", 0, 400.0),
        mk_view("j1", 0, 100.0),
        mk_view("j2", 0, 300.0),
        mk_view("j3", 0, 200.0),
    ]
    actions = SJFScheduler().schedule(FakeView(jobs))
    # Ascending walltime estimate regardless of submit/id arrival order.
    assert placed_order(actions) == ["j1", "j3", "j2", "j0"]


def test_ties_break_by_submit_then_id():
    jobs = [
        mk_view("b", 5, 100.0),
        mk_view("a", 5, 100.0),  # same est+submit -> id breaks: a before b
        mk_view("c", 1, 100.0),  # same est, earlier submit -> first
    ]
    actions = SJFScheduler().schedule(FakeView(jobs))
    assert placed_order(actions) == ["c", "a", "b"]


def test_none_estimate_sorts_last():
    jobs = [
        mk_view("known", 0, 500.0),
        mk_view("unknown", 0, None),  # +inf -> after any finite estimate
        mk_view("short", 0, 50.0),
    ]
    actions = SJFScheduler().schedule(FakeView(jobs))
    assert placed_order(actions) == ["short", "known", "unknown"]


def test_best_effort_skips_unplaceable_and_continues():
    jobs = [
        mk_view("short", 0, 100.0),  # shortest, but cannot place
        mk_view("mid", 0, 200.0),
        mk_view("long", 0, 300.0),
    ]
    view = FakeView(jobs, unplaceable={"short"})
    actions = SJFScheduler(strict=False).schedule(view)
    # 'short' is skipped; the scan continues to the placeable jobs.
    assert placed_order(actions) == ["mid", "long"]
    # Every job was still considered, in SJF order.
    assert view.placement_calls == ["short", "mid", "long"]


def test_strict_blocks_on_shortest_head_of_line():
    jobs = [
        mk_view("short", 0, 100.0),  # shortest head, cannot place -> blocks
        mk_view("mid", 0, 200.0),
        mk_view("long", 0, 300.0),
    ]
    view = FakeView(jobs, unplaceable={"short"})
    actions = SJFScheduler(strict=True).schedule(view)
    assert placed_order(actions) == []
    # Strict stops at the first unplaceable head — no further probing.
    assert view.placement_calls == ["short"]


def test_get_scheduler_rejects_unknown_param():
    import pytest

    with pytest.raises(ValueError):
        get_scheduler("sjf", {"nonsense": 1})
