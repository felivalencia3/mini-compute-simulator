"""Placement policies: the *where* axis of scheduling.

v0.1 shipped :class:`FirstFit` — the DESIGN §7 default that delegates to
:meth:`fleetsim.fleet.tree.FleetTree.search_first_fit` via the view's raw
search primitive.  v0.4 adds RELAXABLE-constraint handling to FirstFit
(the DESIGN §4.2 relax/penalty matched pair): a job whose ``within`` is
``required: false`` first searches under the constraint; once
``relax_after_s`` has elapsed since submission and the constrained search
still fails, the search retries WITHOUT the constraint and the returned
placement is marked ``relaxed=True`` (the engine gates the timeout again
and the cost model applies the configured crossing penalty).

INVARIANTS: policies are pure functions of ``(job, view)`` — no internal
state, no randomness — so placement is deterministic given the view.
"""

from __future__ import annotations

from dataclasses import replace

from ..fleet.tree import Placement
from ..model import Constraint, GangSpec
from .base import ClusterView, JobView

__all__ = ["FirstFit"]


class FirstFit:
    """First fit in deterministic tree order (ascending domain/leaf ids).

    Rebuilds the gang's :class:`~fleetsim.model.GangSpec` from the view
    fields (v1 jobs have exactly one gang; ``shape`` is v0.3 and ignored
    by the search) and runs the raw first-fit search, passing the job's
    tenant so calendar-reservation holds are honored (v0.4).  A job with
    ``segments`` set runs the segmented search instead (v0.2 Slurm-block
    semantics).  A relaxable ``within`` (``within_required == False``)
    falls back to an unconstrained search once ``relax_after_s`` has
    elapsed since submission — the placement is then marked ``relaxed``.
    The engine's ``find_placement`` wrapper handles reservation.
    """

    def place(self, job: JobView, view: ClusterView) -> Placement | None:
        within = (
            Constraint(
                level=job.within,
                required=job.within_required,
                relax_after_s=job.relax_after_s,
            )
            if job.within is not None
            else None
        )
        spec = GangSpec(
            chips=job.chips,
            chip_type=job.chip_type,
            within=within,
            segments=job.segments,
        )
        if spec.segments is not None:
            return view.search_segmented(spec, job.tenant)
        placement = view.search_first_fit(spec, job.tenant)
        if placement is not None:
            return placement
        if (
            within is not None
            and not within.required
            and (view.now - job.submit_time) / 1e6 >= within.relax_after_s
        ):
            relaxed_spec = GangSpec(chips=job.chips, chip_type=job.chip_type)
            placement = view.search_first_fit(relaxed_spec, job.tenant)
            if placement is not None:
                return replace(placement, relaxed=True)
        return None
