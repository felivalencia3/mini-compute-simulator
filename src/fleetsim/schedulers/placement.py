"""Placement policies: the *where* axis of scheduling.

v0.1 ships :class:`FirstFit` only — the DESIGN §7 default that delegates to
:meth:`fleetsim.fleet.tree.FleetTree.search_first_fit` via the view's raw
search primitive.  Topology-aware scoring (LCA depth, edge-packing) arrives
in v0.3 as more policies in this module; ordering code never changes.

INVARIANTS: policies are pure functions of ``(job, view)`` — no internal
state, no randomness — so placement is deterministic given the view.
"""

from __future__ import annotations

from ..fleet.tree import Placement
from ..model import Constraint, GangSpec
from .base import ClusterView, JobView

__all__ = ["FirstFit"]


class FirstFit:
    """First fit in deterministic tree order (ascending domain/leaf ids).

    Rebuilds the gang's :class:`~fleetsim.model.GangSpec` from the view
    fields (v1 jobs have exactly one gang; ``shape`` is v0.3 and ignored
    by the search) and runs the raw first-fit search.  A job with
    ``segments`` set runs the segmented search instead (v0.2 Slurm-block
    semantics; ``search_first_fit`` would delegate anyway — the explicit
    dispatch keeps the two search axes visible to policy authors).  The
    engine's ``find_placement`` wrapper handles reservation.
    """

    def place(self, job: JobView, view: ClusterView) -> Placement | None:
        within = Constraint(level=job.within) if job.within is not None else None
        spec = GangSpec(
            chips=job.chips,
            chip_type=job.chip_type,
            within=within,
            segments=job.segments,
        )
        if spec.segments is not None:
            return view.search_segmented(spec)
        return view.search_first_fit(spec)
