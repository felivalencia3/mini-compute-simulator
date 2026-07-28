"""FIFO: the canonical baseline scheduler (DESIGN §7, complete).

Two flavors behind one flag (Kueue's naming):

- **strict** (StrictFIFO): the head of the line blocks — if the oldest
  pending job cannot be placed, nothing behind it runs this round.  This
  is the mode that exposes gang-scheduling fragmentation (a stuck 512-chip
  gang starves everything).
- **best-effort** (BestEffortFIFO): unplaceable jobs are skipped and the
  scan continues — small jobs flow around a stuck giant.

INVARIANTS: emits only ``Place`` actions; deterministic — ordering is
``(submit_time, id)`` (requeued jobs keep their original submit time) and
placement is the deterministic policy passed in (FirstFit by default).
"""

from __future__ import annotations

from .base import Action, ClusterView, Place, PlacementPolicy, Scheduler, register
from .placement import FirstFit

__all__ = ["FIFOScheduler"]


@register("fifo")
class FIFOScheduler(Scheduler):
    """First-in-first-out over ``(submit_time, id)`` with pluggable
    placement.  ``strict=True`` blocks on the head of line."""

    def __init__(self, placement: PlacementPolicy | None = None, strict: bool = True):
        self.placement: PlacementPolicy = placement if placement is not None else FirstFit()
        self.strict = bool(strict)

    def schedule(self, view: ClusterView) -> list[Action]:
        actions: list[Action] = []
        for job in sorted(view.pending(), key=lambda j: (j.submit_time, j.id)):
            p = view.find_placement(job, self.placement)
            if p is not None:
                actions.append(Place(job.id, p))
            elif self.strict:
                break  # StrictFIFO (Kueue): head-of-line blocks
            # else BestEffortFIFO: skip and continue
        return actions
