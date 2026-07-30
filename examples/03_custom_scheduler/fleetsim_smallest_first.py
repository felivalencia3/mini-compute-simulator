"""Smallest-job-first: an example out-of-tree fleetsim scheduler plugin.

Ordering policy: pending jobs ascending by chip count (FIFO tie-break on
``(submit_time, id)``), so mice never wait behind hogs.  Best-effort by
default: unplaceable jobs are skipped, not head-of-line blocking (pass
``strict: true`` in ``scheduler.params`` to block instead).

PLACEMENT: this plugin follows the built-in convention of taking a
``placement`` keyword defaulting to ``None`` (= ``FirstFit``), which is all
it takes for ``scheduler: {name: smallest_first, params: {placement:
best_fit}}`` to work from YAML — ``get_scheduler`` resolves the name to a
policy instance before construction.  An out-of-tree scheduler that omits
the keyword simply does not offer the *where* axis, and passing the param
raises a clean config error.

This module registers itself twice, demonstrating both mechanisms:

- the ``@register("smallest_first")`` decorator (works as soon as the
  module is imported), and
- the ``fleetsim.schedulers`` entry point declared in ``pyproject.toml``
  (lets ``scheduler: {name: smallest_first}`` in a scenario find the
  class without any import — fleetsim loads it on demand).
"""

from __future__ import annotations

from fleetsim import Action, Place, PlacementPolicy, Scheduler, register
from fleetsim.schedulers.base import ClusterView
from fleetsim.schedulers.placement import FirstFit

__all__ = ["SmallestFirstScheduler"]


@register("smallest_first")
class SmallestFirstScheduler(Scheduler):
    """Place the smallest pending job first (best-effort)."""

    def __init__(
        self, placement: PlacementPolicy | None = None, strict: bool = False
    ):
        self.placement: PlacementPolicy = (
            placement if placement is not None else FirstFit()
        )
        self.strict = bool(strict)

    def schedule(self, view: ClusterView) -> list[Action]:
        actions: list[Action] = []
        for job in sorted(
            view.pending(), key=lambda j: (j.chips, j.submit_time, j.id)
        ):
            p = view.find_placement(job, self.placement)
            if p is not None:
                actions.append(Place(job.id, p))
            elif self.strict:
                break
        return actions
