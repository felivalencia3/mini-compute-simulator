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

v0.7 adds three OPT-IN siblings — :class:`BestFit`, :class:`Consolidate`
and :class:`Spread` — selected by name through ``scheduler.params``:

.. code-block:: yaml

    scheduler: {name: fifo, params: {placement: best_fit}}

**The default remains ``first_fit``.**  A scenario that does not name a
placement policy behaves exactly as it did before v0.7, byte for byte.

WHY THEY EXIST (the v0.7 Helios finding).  The whole-node allocation rule
is that a leaf with ANY owner is ineligible for a whole-node request.
FirstFit places sub-node gangs by ascending leaf id with no preference for
already-partially-used nodes, so it opens a *fully free* node for a 1-chip
job whenever the lower-id nodes are momentarily full — and re-dirties any
node a whole-node gang just released.  Free chips then accumulate as
1..n-1-chip remainders no whole-node gang can use, and under a strict
(blocking) scan the queue head is very often such a gang.  On the Helios
Saturn replay (70.8 % single-GPU jobs, 94.4 % offered load in the dominant
VC) that stranding — *not* gang consolidation across domains — accounted
for the entire FIFO over-inflation: switching to sub-node best fit moves
Saturn's FIFO average JCT from 75,329 s to 55,978 s against a published
55,984 s.  See docs/validation.md §4.2 for the trace evidence,
docs/placement.md for the policy semantics and a selection guide, and
examples/07_placement_study/ for a ~10-second reproducible study.

HONEST SCOPE.  ``consolidate`` is **not** uniformly better than
``first_fit`` per virtual cluster.  On the Helios replay, of the five Saturn
VCs that carry 97 % of the FIFO-SJF gap, three get WORSE (``vcQ4H``
+18.5 %, ``vcBLw`` +5.6 %, ``vcOIr`` +4.2 % on FIFO mean JCT) and the
cluster-level win is carried by ``vczIT`` (-35.6 %, 41 % of Saturn's jobs).
Nor does it improve every published quantity: 6 of the 8 in
docs/validation.md §5 move toward published and 2 move away.  The claim
these policies support is "reproduces the reference placer's aggregate
behavior", never "dominates first-fit".

INVARIANTS: policies are pure functions of ``(job, view)`` — no internal
state, no randomness — so placement is deterministic given the view.  Each
carries a ``search_mode`` naming the tree primitive it uses, so a
preempting scheduler can plan reclaims (``search_after_release``) under
the same semantics it places with.
"""

from __future__ import annotations

from dataclasses import replace

from ..fleet.tree import Placement
from ..model import Constraint, GangSpec
from .base import ClusterView, JobView, PlacementPolicy

__all__ = [
    "FirstFit",
    "BestFit",
    "Consolidate",
    "Spread",
    "PLACEMENT_POLICIES",
    "placement_names",
    "get_placement",
    "resolve_placement",
    "takes_placement_policy",
]


class _SearchPolicy:
    """Shared body of the built-in policies: rebuild the gang spec from the
    view fields, run this policy's search primitive, and handle a RELAXABLE
    ``within`` exactly as v0.4 pinned it.

    Subclasses set :attr:`search_mode` (a
    :data:`fleetsim.fleet.tree.FleetTree.PACK_MODES` name) and
    :attr:`_view_method` (the :class:`~fleetsim.schedulers.base.ClusterView`
    pass-through to call).  Everything else — spec construction, the
    segmented delegation, the relax retry and its ``relaxed=True`` marking
    — is identical across policies, so the four differ *only* in which free
    leaves they pick.
    """

    #: Tree search mode this policy places with (see
    #: :meth:`fleetsim.fleet.tree.FleetTree.search`).
    search_mode: str = "first_fit"
    #: Name of the ClusterView raw-search method to call.
    _view_method: str = "search_first_fit"

    def _search(
        self, view: ClusterView, spec: GangSpec, tenant: str | None
    ) -> Placement | None:
        return getattr(view, self._view_method)(spec, tenant)

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
        placement = self._search(view, spec, job.tenant)
        if placement is not None:
            return placement
        if (
            within is not None
            and not within.required
            and (view.now - job.submit_time) / 1e6 >= within.relax_after_s
        ):
            relaxed_spec = GangSpec(chips=job.chips, chip_type=job.chip_type)
            placement = self._search(view, relaxed_spec, job.tenant)
            if placement is not None:
                return replace(placement, relaxed=True)
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def __hash__(self) -> int:
        return hash(type(self))


class FirstFit(_SearchPolicy):
    """First fit in deterministic tree order (ascending domain/leaf ids) —
    the DEFAULT, unchanged since v0.1.

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

    search_mode = "first_fit"
    _view_method = "search_first_fit"


class BestFit(_SearchPolicy):
    """Tightest fit: pack sub-node gangs into the fullest node that still
    has room, and whole-node gangs into the tightest domain that fits
    (opt-in; ``placement: best_fit``).

    Concretely (see
    :meth:`fleetsim.fleet.tree.FleetTree.search_best_fit` for the exact
    rules):

    - **sub-node** request (``chips < node size``) → the eligible HEALTHY
      leaf whose FREE chips are smallest-but-sufficient, ties by ascending
      leaf id, early exit on an exact fit.  This is Slurm ``cons_tres``
      best-fit / ``CR_Pack_Nodes``.  It never opens a fully-free node
      while a partially-used one has room, so **whole free nodes stay
      whole** for the gangs that need them.
    - **whole-node** request → the tightest of the leaves' parent domains
      whose free whole-node capacity covers the request (ties ascending
      id); leaves inside it are taken largest-first exactly as
      ``first_fit`` does.  Search domains are likewise tried
      tightest-first rather than by id.

    On a single-level fleet (``levels: ["node"]``) there is one parent
    domain and one cluster root, so the whole-node path is *identical* to
    ``first_fit`` and the only change is the sub-node packing — which is
    exactly the axis the Helios finding identified (module docstring).

    DETERMINISM: the total order ``(free_chips, leaf_id)`` is a pure
    function of tree state.  Note leaf ids sort LEXICOGRAPHICALLY
    (``node0, node1, node10, ..., node2``), so "ascending leaf id" is not
    numeric order — the convention ``first_fit`` has always used.
    """

    search_mode = "best_fit"
    _view_method = "search_best_fit"


class Consolidate(_SearchPolicy):
    """Fewest PARENT domains touched, tightest fit within (opt-in;
    ``placement: consolidate``).

    Identical to :class:`BestFit` except when a whole-node gang fits in no
    single parent domain: the parents are then consumed by DESCENDING free
    capacity so the gang spans the minimum number of them (the rule
    :meth:`fleetsim.fleet.tree.FleetTree._pack_segments` already applies to
    segmented gangs), where :class:`BestFit` would fill the tight holes
    first.

    WHAT "FEWEST DOMAINS" DOES NOT MEAN (measured; see
    :meth:`fleetsim.fleet.tree.FleetTree._packed_whole_order`).  The
    grouping is flat and at the PARENT level only, so on a 3-level fleet
    this policy can span MORE pods than ``first_fit`` while touching the
    same number of racks — a 40-chip gang measured at 2 pods / 2 racks
    against ``first_fit``'s 1 pod / 2 racks.  It is therefore NOT the
    policy to pick for minimizing ``penalties.xover`` above the parent
    level; use it for the sub-node packing it shares with
    :class:`BestFit`, and measure the span you actually care about.

    HONEST DEGENERACY (worth stating, because it was the v0.6
    misdiagnosis): on a fleet whose leaves are all direct children of the
    searched domain — every ``levels: ["node"]`` config, including the
    Helios validation replay — there is exactly ONE parent domain and this
    policy is *bit-for-bit* :class:`BestFit`.  "Consolidate large gangs
    across domains" therefore could not have explained the Helios Saturn
    gap: that fleet has no domains to consolidate within, and no
    ``penalties.xover`` to price a span.  What actually changes there is
    the sub-node packing both policies share.
    """

    search_mode = "consolidate"
    _view_method = "search_consolidate"


class Spread(_SearchPolicy):
    """Maximum spread — the deliberate ANTI-policy (opt-in;
    ``placement: spread``).

    Sub-node gangs take the eligible leaf with the MOST free chips (worst
    fit), whole-node gangs round-robin across the leaves' parent domains,
    and search domains are tried emptiest-first.  It manufactures exactly
    the remainders :class:`BestFit` suppresses.

    Ships as a first-class policy for two reasons: it is the control arm
    that proves a placement study's effect is real (on the Helios replay
    it is measurably worse than ``first_fit``), and "spread for blast-radius
    / thermal reasons" is a policy real operators actually run.
    """

    search_mode = "spread"
    _view_method = "search_spread"


# ---------------------------------------------------------------------------
# Name registry (the ``scheduler.params.placement`` surface)
# ---------------------------------------------------------------------------

#: Configurable name -> policy class.  ``first_fit`` is the default for
#: every scheduler; the rest are opt-in.  Insertion order is the order
#: error messages list them in.
PLACEMENT_POLICIES: dict[str, type[_SearchPolicy]] = {
    "first_fit": FirstFit,
    "best_fit": BestFit,
    "consolidate": Consolidate,
    "spread": Spread,
}


def placement_names() -> tuple[str, ...]:
    """Every configurable placement-policy name, sorted."""
    return tuple(sorted(PLACEMENT_POLICIES))


def get_placement(name: str) -> PlacementPolicy:
    """Instantiate the placement policy registered under ``name``.

    Raises ``ValueError`` naming every available policy for an unknown
    name, so ``scheduler: {params: {placement: bestfit}}`` surfaces as a
    clean config error rather than a traceback.
    """
    cls = PLACEMENT_POLICIES.get(str(name))
    if cls is None:
        raise ValueError(
            f"unknown placement policy {name!r}"
            f" (available: {', '.join(PLACEMENT_POLICIES)})"
        )
    return cls()


def takes_placement_policy(cls: object) -> bool:
    """Does ``cls.__init__`` declare its ``placement`` parameter to be a
    :class:`~fleetsim.schedulers.base.PlacementPolicy`?

    This is the gate on the v0.7 name registry.  ``placement`` is a
    *convention*, not a reserved word: an out-of-tree scheduler is free to
    take a ``placement`` param with its own vocabulary
    (``placement: lowest_rack``), and v0.6 passed such a string through
    untouched.  So the name resolution in
    :func:`fleetsim.schedulers.base.get_scheduler` and the closed-set check
    in :func:`fleetsim.config.validate` apply ONLY when the target
    scheduler opts in by annotating the parameter — which all four
    built-ins do (``placement: PlacementPolicy | None = None``).

    The annotation is matched TEXTUALLY (``"PlacementPolicy"`` appearing in
    it) so ``from __future__ import annotations`` modules, quoted
    annotations and unions all work without importing the plugin's own
    module namespace.  A scheduler with no ``placement`` parameter, or one
    with a bare/absent annotation, reads as "not opted in".
    """
    import inspect

    init = getattr(cls, "__init__", None)
    if init is None:
        return False
    try:
        params = inspect.signature(init).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    param = params.get("placement")
    if param is None:
        return False
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        return False
    return "PlacementPolicy" in (ann if isinstance(ann, str) else str(ann))


def resolve_placement(value: object) -> object:
    """Map a placement-policy NAME to an instance; pass anything else
    through untouched.

    This is the single conversion point between the config surface
    (``scheduler.params.placement: best_fit`` — a string) and the
    programmatic surface (``FIFOScheduler(placement=BestFit())`` — an
    object implementing :class:`~fleetsim.schedulers.base.PlacementPolicy`).
    ``None`` passes through so each scheduler keeps its own default.
    """
    if isinstance(value, str):
        return get_placement(value)
    return value
