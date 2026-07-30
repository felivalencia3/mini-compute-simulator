"""Fleet domain tree: capacity counters, gang allocation, placement search.

The :class:`FleetTree` is the runtime view over
:class:`~fleetsim.model.Domain` objects built by
:func:`fleetsim.fleet.build.build_fleet`.  Topology is immutable after
construction; the tree mutates only leaf health states and chip ownership,
keeping per-domain counters incrementally correct.

ALLOCATION MODEL (DESIGN 4.1)
-----------------------------
- **Sub-node**: a request smaller than a leaf's chip count shares that
  single leaf; the leaf tracks owners as ``{alloc_id: chips}`` and may host
  many sub-node gangs up to capacity.
- **Whole-node**: a request of one node or larger (validated upstream to be
  a whole-node multiple) takes exclusive, fully-free leaves; a leaf with
  ANY owner (sub-node or otherwise) is not eligible.
- ``alloc_id`` is the owning job id (``Allocation.job_id``); multi-gang
  allocations of one job merge into a single owner entry per leaf.

SEARCH PRIMITIVES
-----------------
Every search is a pure, non-mutating query returning a
:class:`Placement` or ``None``.  On uniform leaf sizes they all answer the
same feasibility question and differ only in WHICH fitting leaves they
choose; on mixed leaf sizes (template-form fleets only) the exact
whole-node cover is a subset-sum problem, and the packed modes' answer is
a SUPERSET of first-fit's — never a subset (:meth:`_scan_leaves_packed`):

- :meth:`FleetTree.search_first_fit` — the v0.1 default: ascending
  domain/leaf id.  **Frozen**: nothing in v0.7 changes its behavior.
- :meth:`FleetTree.search_segmented` — v0.2 Slurm-block segment packing.
- :meth:`FleetTree.search_best_fit` / :meth:`FleetTree.search_consolidate`
  / :meth:`FleetTree.search_spread` — v0.7, OPT-IN (see
  :mod:`fleetsim.schedulers.placement`).  ``search_best_fit`` packs
  sub-node gangs tightest-fit-first so partially-used nodes get filled
  instead of fresh ones being opened; that is what keeps whole nodes
  available for gangs, since a leaf with ANY owner is invisible to every
  whole-node request.  :meth:`FleetTree.search` dispatches all four by
  mode name.

UNITS: every quantity in this module is an integer chip count; nothing
here is a time.

INVARIANTS
----------
- For every domain ``D`` (leaves included):
  ``D.total_chips == sum(leaf.chips)`` over leaves under ``D`` (any state),
  ``D.free_chips  == sum(leaf.chips - used(leaf))`` over HEALTHY leaves,
  ``healthy_chips(D) == sum(leaf.chips)`` over HEALTHY leaves.
- ``leaf.free_chips == leaf.chips - used(leaf)`` when HEALTHY, else ``0``
  (DRAINING / FAILED / MAINTENANCE leaves contribute zero free chips).
- Free-NODE counters (v0.2): for every domain and ``(chip_type, size)``
  key, the count equals the number of HEALTHY zero-owner leaves of that
  type/size under the domain — the segmented-placement primitive.
- Counter updates are O(depth) per touched leaf; searches and iterators
  use sorted-id or insertion order — never raw set iteration order.
- :meth:`FleetTree.apply` / :meth:`FleetTree.release` are atomic (validate
  everything, then mutate) and idempotence-guarded: double apply or double
  release raises ``ValueError`` and changes nothing.
- :meth:`FleetTree.check_invariants` re-derives all counters from scratch
  and raises ``AssertionError`` on any mismatch (O(N·depth); for tests and
  debugging only).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from ..model import Allocation, Domain, GangAlloc, GangSpec, NodeState

__all__ = ["Placement", "FleetTree"]


@dataclass(frozen=True, slots=True)
class Placement:
    """A concrete first-fit placement for one gang.

    ``leaves`` is a tuple of ``(leaf_id, chips)`` pairs in ascending
    leaf-id order: exactly one entry with ``chips < leaf.chips`` for a
    sub-node gang, or one entry per fully-taken leaf (``chips ==
    leaf.chips``) for a whole-node gang.  ``anchor`` is the domain that
    satisfied the ``within`` constraint (the searched domain), or the
    cluster root when the spec carried no constraint — except for
    SEGMENTED placements, where ``anchor`` is the LCA of all segment
    domains (DESIGN §4.2: the placement-quality score).

    ``segment_domains`` (v0.2, segmented gangs only) lists the
    segment-level domain id hosting each segment, one entry per segment
    in deterministic assignment order; ids REPEAT when several segments
    share one domain.  Empty tuple for non-segmented placements.

    ``relaxed`` (v0.4) marks a placement found AFTER dropping a
    relaxable (``required: false``) ``within`` constraint whose
    ``relax_after`` timeout elapsed — the engine gates it and the
    cost model penalizes it (the DESIGN §4.2 matched pair).

    INVARIANT: ``sum(chips for _, chips in leaves) == spec.chips`` of the
    searched :class:`~fleetsim.model.GangSpec`.
    """

    leaves: tuple[tuple[str, int], ...]
    anchor: str
    chip_type: str
    whole_node: bool
    segment_domains: tuple[str, ...] = ()
    relaxed: bool = False

    @property
    def chips(self) -> int:
        """Total chips this placement covers."""
        return sum(chips for _, chips in self.leaves)

    @property
    def n_domains_spanned(self) -> int:
        """Distinct segment-level domains used (1 for non-segmented —
        a plain gang always sits under its single anchor)."""
        if not self.segment_domains:
            return 1
        return len(set(self.segment_domains))

    def to_gang_alloc(self) -> GangAlloc:
        """Render as a :class:`~fleetsim.model.GangAlloc`: a list of leaf
        ids for a whole-node gang, a ``{leaf: chips}`` dict for sub-node.
        Segmented placements record ``segment_domains`` and
        ``n_domains_spanned`` in ``GangAlloc.attrs``."""
        attrs: dict = {}
        if self.segment_domains:
            attrs = {
                "segment_domains": list(self.segment_domains),
                "n_domains_spanned": self.n_domains_spanned,
            }
        if self.whole_node:
            return GangAlloc(
                nodes=[leaf for leaf, _ in self.leaves],
                anchor=self.anchor,
                relaxed=self.relaxed,
                attrs=attrs,
            )
        return GangAlloc(
            nodes={leaf: chips for leaf, chips in self.leaves},
            anchor=self.anchor,
            relaxed=self.relaxed,
            attrs=attrs,
        )


class FleetTree:
    """Mutable runtime capacity tree over immutable topology.

    Construct with the full set of :class:`~fleetsim.model.Domain` objects
    (parents before or after children — order does not matter, ids must be
    unique and parent/children links consistent).  ``cluster_roots`` are
    the domains searched when a :class:`~fleetsim.model.GangSpec` has no
    ``within`` constraint; when omitted they default to all domains at
    level ``"cluster"``, falling back to the parentless roots.

    All counters (``total_chips``, ``free_chips`` on the Domain objects;
    healthy and per-chip-type free counters in side tables) are recomputed
    once here and maintained incrementally afterwards — see the module
    docstring for the exact invariants.
    """

    def __init__(
        self,
        domains: Iterable[Domain],
        cluster_roots: Sequence[str] | None = None,
    ) -> None:
        self._domains: dict[str, Domain] = {}
        for d in domains:
            if d.id in self._domains:
                raise ValueError(f"duplicate domain id {d.id!r}")
            self._domains[d.id] = d
        if not self._domains:
            raise ValueError("FleetTree requires at least one domain")

        # --- structural validation -----------------------------------
        for d in self._domains.values():
            if d.parent is not None:
                parent = self._domains.get(d.parent)
                if parent is None:
                    raise ValueError(f"domain {d.id!r} has unknown parent {d.parent!r}")
                if d.id not in parent.children:
                    raise ValueError(
                        f"domain {d.id!r} is not listed in parent {d.parent!r}.children"
                    )
            for cid in d.children:
                child = self._domains.get(cid)
                if child is None:
                    raise ValueError(f"domain {d.id!r} has unknown child {cid!r}")
                if child.parent != d.id:
                    raise ValueError(
                        f"child {cid!r} of {d.id!r} has parent {child.parent!r}"
                    )
            if not d.children:
                if d.chips <= 0:
                    raise ValueError(f"leaf {d.id!r} must have chips > 0, got {d.chips}")
                if d.chip_type is None:
                    raise ValueError(f"leaf {d.id!r} must have a chip_type")
            elif d.chips:
                raise ValueError(
                    f"interior domain {d.id!r} must not carry chips (got {d.chips})"
                )

        # --- depth (with cycle guard) --------------------------------
        self._depth: dict[str, int] = {}
        limit = len(self._domains)
        for did in self._domains:
            chain: list[str] = []
            cur: str | None = did
            while cur is not None and cur not in self._depth:
                chain.append(cur)
                if len(chain) > limit:
                    raise ValueError(f"parent cycle involving domain {did!r}")
                cur = self._domains[cur].parent
            base = -1 if cur is None else self._depth[cur]
            for i, cid in enumerate(reversed(chain)):
                self._depth[cid] = base + 1 + i

        # --- static indexes (sorted-id order everywhere) -------------
        self._leaf_ids: tuple[str, ...] = tuple(
            sorted(d.id for d in self._domains.values() if not d.children)
        )
        by_level: dict[str, list[str]] = {}
        for did in sorted(self._domains):
            by_level.setdefault(self._domains[did].level, []).append(did)
        self._levels: dict[str, tuple[str, ...]] = {
            lvl: tuple(ids) for lvl, ids in by_level.items()
        }
        under: dict[str, list[str]] = {did: [] for did in self._domains}
        for lid in self._leaf_ids:  # sorted, so each list ends up sorted
            cur = lid
            while cur is not None:
                under[cur].append(lid)
                cur = self._domains[cur].parent
        self._leaves_under: dict[str, tuple[str, ...]] = {
            did: tuple(ids) for did, ids in under.items()
        }
        self._types_under: dict[str, tuple[str, ...]] = {
            did: tuple(sorted({self._domains[l].chip_type for l in ids}))
            for did, ids in self._leaves_under.items()
        }

        if cluster_roots is None:
            roots = list(self._levels.get("cluster", ()))
            if not roots:
                roots = sorted(d.id for d in self._domains.values() if d.parent is None)
        else:
            for rid in cluster_roots:
                if rid not in self._domains:
                    raise ValueError(f"unknown cluster root {rid!r}")
            roots = sorted(cluster_roots)
        self._cluster_roots: tuple[str, ...] = tuple(roots)

        # --- dynamic state -------------------------------------------
        self._owners: dict[str, dict[str, int]] = {lid: {} for lid in self._leaf_ids}
        self._allocs: dict[str, Allocation] = {}
        # Reservation holds (v0.4 calendar blocks): leaf id -> owner
        # tenant.  A reserved leaf is invisible to placement searches of
        # every OTHER tenant; empty dict = zero overhead on every path.
        self._reserved: dict[str, str] = {}
        self._healthy: dict[str, int] = {did: 0 for did in self._domains}
        self._free_ct: dict[str, dict[str, int]] = {did: {} for did in self._domains}
        # Free-NODE counters (v0.2, segmented placement): per domain, the
        # number of HEALTHY, zero-owner leaves keyed by (chip_type, leaf
        # chips).  Maintained incrementally like the chip counters so
        # segment packing operates on domain counters, never fleet-wide
        # leaf scans.
        self._free_nodes: dict[str, dict[tuple[str, int], int]] = {
            did: {} for did in self._domains
        }
        # Lazy static cache: (level, ancestor_id) -> domains at `level`
        # under `ancestor_id`, ascending id order (topology is immutable,
        # so entries never invalidate).
        self._at_level_under: dict[tuple[str, str], tuple[str, ...]] = {}

        for dom in self._domains.values():
            dom.total_chips = 0
            dom.free_chips = 0
        for lid in self._leaf_ids:
            leaf = self._domains[lid]
            cur: str | None = lid
            while cur is not None:
                self._domains[cur].total_chips += leaf.chips
                cur = self._domains[cur].parent
            if leaf.state is NodeState.HEALTHY:
                self._add_free(lid, leaf.chips)
                self._add_healthy(lid, leaf.chips)
                self._add_free_nodes(lid, 1)

    # ------------------------------------------------------------------
    # Lookups and helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._domains)

    def __contains__(self, domain_id: object) -> bool:
        return domain_id in self._domains

    def __iter__(self) -> Iterator[str]:
        """Iterate all domain ids in insertion (build/document) order."""
        return iter(self._domains)

    def domain(self, domain_id: str) -> Domain:
        """The :class:`Domain` object for ``domain_id`` (KeyError if unknown)."""
        return self._domains[domain_id]

    def is_leaf(self, domain_id: str) -> bool:
        return not self._domains[domain_id].children

    def leaves(self) -> tuple[str, ...]:
        """All leaf ids, ascending id order."""
        return self._leaf_ids

    def leaves_under(self, domain_id: str) -> tuple[str, ...]:
        """Leaf ids under ``domain_id`` (inclusive of itself if a leaf),
        ascending id order."""
        return self._leaves_under[domain_id]

    def domains_at(self, level: str) -> tuple[str, ...]:
        """All domain ids at ``level``, ascending id order (empty tuple for
        an unknown level)."""
        return self._levels.get(level, ())

    def levels(self) -> tuple[str, ...]:
        """All level names present in the tree, sorted."""
        return tuple(sorted(self._levels))

    @property
    def cluster_roots(self) -> tuple[str, ...]:
        """Cluster root ids (searched when a spec has no ``within``),
        ascending id order."""
        return self._cluster_roots

    def parent(self, domain_id: str) -> str | None:
        return self._domains[domain_id].parent

    def depth(self, domain_id: str) -> int:
        """0 for roots, +1 per level down."""
        return self._depth[domain_id]

    def ancestors(self, domain_id: str, *, include_self: bool = False) -> Iterator[str]:
        """Yield ancestor ids walking up to the root (nearest first)."""
        cur = domain_id if include_self else self._domains[domain_id].parent
        while cur is not None:
            yield cur
            cur = self._domains[cur].parent

    def lca(self, a: str, b: str) -> str | None:
        """Lowest common ancestor of two domains, or ``None`` when they
        live under different roots (e.g. different metros)."""
        da, db = self._depth[a], self._depth[b]
        while da > db:
            a = self._domains[a].parent  # type: ignore[assignment]
            da -= 1
        while db > da:
            b = self._domains[b].parent  # type: ignore[assignment]
            db -= 1
        while a != b:
            pa, pb = self._domains[a].parent, self._domains[b].parent
            if pa is None or pb is None:
                return None
            a, b = pa, pb
        return a

    # ------------------------------------------------------------------
    # Counter queries
    # ------------------------------------------------------------------

    def total_chips(self, domain_id: str) -> int:
        """Chips under ``domain_id`` regardless of leaf state."""
        return self._domains[domain_id].total_chips

    def free_chips(self, domain_id: str, chip_type: str | None = None) -> int:
        """Free chips under ``domain_id``: unallocated chips on HEALTHY
        leaves only, optionally restricted to one ``chip_type``."""
        if chip_type is None:
            return self._domains[domain_id].free_chips
        return self._free_ct[domain_id].get(chip_type, 0)

    def healthy_chips(self, domain_id: str) -> int:
        """Capacity on HEALTHY leaves under ``domain_id`` (allocated or not)."""
        return self._healthy[domain_id]

    def free_full_nodes(
        self, domain_id: str, chip_type: str, leaf_chips: int
    ) -> int:
        """Count of HEALTHY, zero-owner leaves of ``chip_type`` with
        exactly ``leaf_chips`` chips under ``domain_id`` (an O(1) counter
        read — the segmented-placement primitive)."""
        return self._free_nodes[domain_id].get((chip_type, leaf_chips), 0)

    def domains_at_under(self, level: str, ancestor_id: str) -> tuple[str, ...]:
        """Domain ids at ``level`` under ``ancestor_id`` (inclusive of
        itself when it sits at ``level``), ascending id order.  Cached —
        topology is immutable."""
        key = (level, ancestor_id)
        cached = self._at_level_under.get(key)
        if cached is not None:
            return cached
        if ancestor_id not in self._domains:
            raise KeyError(ancestor_id)
        out = tuple(
            did
            for did in self._levels.get(level, ())
            if did == ancestor_id
            or ancestor_id in self.ancestors(did)
        )
        self._at_level_under[key] = out
        return out

    def owners(self, leaf_id: str) -> dict[str, int]:
        """Copy of the ``{alloc_id: chips}`` owner map of a leaf."""
        return dict(self._owners[leaf_id])

    def used_chips(self, leaf_id: str) -> int:
        """Chips on ``leaf_id`` currently owned by allocations."""
        return sum(self._owners[leaf_id].values())

    def has_allocation(self, job_id: str) -> bool:
        return job_id in self._allocs

    # ------------------------------------------------------------------
    # Reservation holds (v0.4 calendar blocks)
    # ------------------------------------------------------------------

    def reserve_leaves(self, leaf_ids: Iterable[str], tenant: str) -> None:
        """Mark leaves as held for ``tenant``: placement searches of any
        other tenant skip them from now on.  Raises for non-leaf ids or
        leaves already reserved (holds never overlap)."""
        for lid in leaf_ids:
            self._require_leaf(lid)
            if lid in self._reserved:
                raise ValueError(
                    f"leaf {lid!r} is already reserved for"
                    f" {self._reserved[lid]!r}"
                )
            self._reserved[lid] = tenant

    def release_reservation(self, leaf_ids: Iterable[str]) -> None:
        """Lift the hold on ``leaf_ids`` (missing entries are ignored)."""
        for lid in leaf_ids:
            self._reserved.pop(lid, None)

    def reserved_owner(self, leaf_id: str) -> str | None:
        """The tenant holding ``leaf_id``, or None when unreserved."""
        return self._reserved.get(leaf_id)

    @property
    def has_reservations(self) -> bool:
        return bool(self._reserved)

    def _leaf_eligible(self, leaf_id: str, tenant: str | None) -> bool:
        """False iff the leaf is reserved for a DIFFERENT tenant."""
        owner = self._reserved.get(leaf_id)
        return owner is None or owner == tenant

    def reserved_free_chips(self, domain_id: str, tenant: str | None) -> int:
        """Free chips under ``domain_id`` on HEALTHY leaves held by a
        reservation for a DIFFERENT tenant — free capacity ``tenant``'s
        placements can never use while the hold lasts (v0.4).  0 with no
        active hold.  O(|reserved| x depth); chip-count-honest consumers
        (e.g. EASY's shadow accounting) subtract this from
        :meth:`free_chips`."""
        if not self._reserved:
            return 0
        n = 0
        for lid, owner in self._reserved.items():
            if owner == tenant:
                continue
            leaf = self._domains[lid]
            if leaf.state is NodeState.HEALTHY and (
                lid == domain_id
                or domain_id in self.ancestors(lid)
            ):
                n += leaf.free_chips
        return n

    def _ineligible_free_nodes(
        self, domain_id: str, chip_type: str, leaf_chips: int, tenant: str | None
    ) -> int:
        """How many of ``free_full_nodes(domain_id, chip_type,
        leaf_chips)`` are reserved for a different tenant (and therefore
        unusable by this search).  O(|reserved| x depth) — reservations
        are rare and small."""
        if not self._reserved:
            return 0
        n = 0
        for lid, owner in self._reserved.items():
            if owner == tenant:
                continue
            leaf = self._domains[lid]
            if (
                leaf.chip_type == chip_type
                and leaf.chips == leaf_chips
                and leaf.state is NodeState.HEALTHY
                and not self._owners[lid]
                and domain_id in self.ancestors(lid, include_self=True)
            ):
                n += 1
        return n

    def allocation(self, job_id: str) -> Allocation:
        """The applied allocation for ``job_id`` (KeyError if none)."""
        return self._allocs[job_id]

    def applied_jobs(self) -> tuple[str, ...]:
        """Job ids with live allocations, in apply (insertion) order."""
        return tuple(self._allocs)

    # ------------------------------------------------------------------
    # Allocation apply / release
    # ------------------------------------------------------------------

    def apply(self, allocation: Allocation) -> None:
        """Apply an allocation: record owners and decrement free counters.

        Atomic — every leaf is validated before any mutation.  Raises
        ``ValueError`` on double-apply (same ``job_id``), unknown or
        non-leaf ids, non-HEALTHY leaves, over-capacity sub-node gangs, or
        whole-node gangs targeting a leaf with any existing owner.
        """
        jid = allocation.job_id
        if jid in self._allocs:
            raise ValueError(f"allocation for job {jid!r} already applied")
        demand = self._demand(allocation)
        for lid, (chips, whole) in demand.items():
            leaf = self._require_leaf(lid)
            if leaf.state is not NodeState.HEALTHY:
                raise ValueError(
                    f"cannot allocate on {lid!r}: state is {leaf.state.name}"
                )
            if whole and self._owners[lid]:
                raise ValueError(
                    f"whole-node gang needs exclusive leaf {lid!r}, but it has"
                    f" owners {sorted(self._owners[lid])}"
                )
            if chips > leaf.free_chips:
                raise ValueError(
                    f"over-capacity on {lid!r}: need {chips},"
                    f" free {leaf.free_chips}"
                )
        for lid, (chips, _) in demand.items():
            if not self._owners[lid]:  # free node -> owned node
                self._add_free_nodes(lid, -1)
            self._owners[lid][jid] = chips
            self._add_free(lid, -chips)
        self._allocs[jid] = allocation

    def release(self, allocation: Allocation | str) -> None:
        """Release a previously-applied allocation (by object or job id).

        Uses the stored allocation keyed by ``job_id``; raises
        ``ValueError`` if none is applied (double-release guard).  Chips on
        non-HEALTHY leaves do not return to the free counters — those
        leaves already contribute zero free.
        """
        jid = allocation if isinstance(allocation, str) else allocation.job_id
        stored = self._allocs.pop(jid, None)
        if stored is None:
            raise ValueError(f"no applied allocation for job {jid!r}")
        leaf_ids: dict[str, None] = {}
        for gang in stored.gangs:
            nodes = gang.nodes
            for lid in nodes if isinstance(nodes, list) else nodes.keys():
                leaf_ids.setdefault(lid)
        for lid in leaf_ids:
            chips = self._owners[lid].pop(jid)
            if self._domains[lid].state is NodeState.HEALTHY:
                self._add_free(lid, chips)
                if not self._owners[lid]:  # owned node -> free node
                    self._add_free_nodes(lid, 1)

    def _demand(self, allocation: Allocation) -> dict[str, tuple[int, bool]]:
        """Merge an allocation's gangs into ``{leaf_id: (chips, whole)}``.

        List-form gangs claim each listed leaf in full (whole-node); a
        dict-form gang must occupy exactly one leaf (sub-node).
        """
        demand: dict[str, list] = {}
        for gang in allocation.gangs:
            nodes = gang.nodes
            if isinstance(nodes, dict):
                if len(nodes) != 1:
                    raise ValueError(
                        "sub-node gang must occupy exactly one leaf,"
                        f" got {sorted(nodes)}"
                    )
                for lid, chips in nodes.items():
                    if not isinstance(chips, int) or chips <= 0:
                        raise ValueError(
                            f"sub-node gang on {lid!r} needs a positive chip"
                            f" count, got {chips!r}"
                        )
                    entry = demand.setdefault(lid, [0, False])
                    entry[0] += chips
            else:
                if not nodes:
                    raise ValueError("whole-node gang has an empty leaf list")
                for lid in nodes:
                    leaf = self._require_leaf(lid)
                    entry = demand.setdefault(lid, [0, False])
                    entry[0] += leaf.chips
                    entry[1] = True
        return {lid: (entry[0], entry[1]) for lid, entry in demand.items()}

    def _require_leaf(self, leaf_id: str) -> Domain:
        dom = self._domains.get(leaf_id)
        if dom is None:
            raise ValueError(f"unknown domain {leaf_id!r}")
        if dom.children:
            raise ValueError(f"domain {leaf_id!r} is not a leaf")
        return dom

    # ------------------------------------------------------------------
    # Placement search
    # ------------------------------------------------------------------

    def search_first_fit(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """First-fit search for one gang, or ``None`` if nothing fits now.

        Search domains: all domains at ``spec.within.level`` when the spec
        has a ``within`` constraint (treated as hard here — RELAXATION is
        the placement policy's retry-without-constraint, not a search
        concern), else the cluster roots.  Domains are tried independently
        in ascending id order; within a domain, leaves are scanned in
        ascending id order.  ``spec.chip_type`` pins the chip type; when
        unpinned, each chip type present under the domain is tried in
        sorted order (v1 configs pin, per DESIGN §11).  A spec with
        ``segments`` delegates to :meth:`search_segmented` (first-fit over
        the same outer domains, segment bin-packing inside).
        ``shape``/``twisted`` are v0.3 semantics and ignored.

        ``tenant`` (v0.4) is the requesting job's tenant: leaves held by a
        calendar reservation for a DIFFERENT tenant are skipped; ``None``
        skips every reserved leaf (conservative).

        Per-leaf mode: a request smaller than the leaf's chip count is
        sub-node (needs ``free >= chips``); otherwise whole-node leaves are
        accumulated (each must be HEALTHY and fully free) until the request
        is exactly covered.
        """
        if spec.chips <= 0:
            raise ValueError(f"gang spec needs a positive chip count, got {spec.chips}")
        if spec.segments is not None:
            return self.search_segmented(spec, tenant)
        if spec.within is not None:
            search_domains = self.domains_at(spec.within.level)
        else:
            search_domains = self._cluster_roots
        for did in search_domains:
            placement = self._search_domain(did, spec, tenant)
            if placement is not None:
                return placement
        return None

    def search_segmented(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """Segmented (Slurm-block) search for one gang, or ``None``.

        SEMANTICS (v0.2, pinned): the gang is WHOLE-NODE only and splits
        into ``spec.chips / (nodes_per_segment * leaf_size)`` equal
        segments of exactly ``nodes_per_segment`` fully-free HEALTHY
        leaves each; every segment is contained in ONE domain at the
        segment level, and multiple segments MAY share a domain that has
        room.  All segments place atomically or none.  Outer search
        domains are ``spec.within.level`` domains (the OUTER constraint)
        or the cluster roots, tried first-fit in ascending id order.
        Inside an outer domain, segment-hosting domains are packed by
        DESCENDING free-node capacity (ties ascending id) — bin-packing
        that concentrates the job into the fewest SEGMENT-LEVEL domains
        (the only level grouped; nothing coarser is minimized) and
        preserves empty ones for future large jobs.  ``anchor`` is the LCA of
        all segment domains.  Feasibility is unchanged by penalties;
        when ``penalties.xover`` is configured (v0.4) the engine prices
        the multi-domain span at stint start (see ``Simulator.speed`` /
        DESIGN §17.1) — the search itself never rejects a spanning
        placement on cost grounds.

        The candidate filter and packing operate on the incrementally
        maintained free-node counters (O(domains at segment level)); only
        the leaves of CHOSEN segment domains are scanned, so cost scales
        with the allocation, never the fleet.

        Raises ``ValueError`` for a spec without ``segments``, a
        non-positive ``nodes_per_segment``, or a non-positive chip count.
        Chip counts that do not decompose into whole segments of any
        available leaf size simply find no fit.
        """
        if spec.segments is None:
            raise ValueError("search_segmented requires spec.segments")
        nodes_per_seg, seg_level = spec.segments
        if nodes_per_seg <= 0:
            raise ValueError(
                f"segments nodes_per_segment must be positive, got {nodes_per_seg}"
            )
        if spec.chips <= 0:
            raise ValueError(f"gang spec needs a positive chip count, got {spec.chips}")
        if spec.within is not None:
            search_domains = self.domains_at(spec.within.level)
        else:
            search_domains = self._cluster_roots
        for did in search_domains:
            placement = self._search_segmented_domain(
                did, spec, nodes_per_seg, seg_level, tenant
            )
            if placement is not None:
                return placement
        return None

    def _search_segmented_domain(
        self,
        did: str,
        spec: GangSpec,
        nodes_per_seg: int,
        seg_level: str,
        tenant: str | None = None,
    ) -> Placement | None:
        """Segment-pack ``spec`` under one outer domain, or ``None``."""
        types = (
            (spec.chip_type,) if spec.chip_type is not None else self._types_under[did]
        )
        for ct in types:
            # Candidate leaf sizes for this chip type, from the domain's
            # free-node counter keys (a size with zero free nodes cannot
            # host anything; deterministic ascending-size order).
            sizes = sorted(
                size
                for (t, size), n in self._free_nodes[did].items()
                if t == ct and n > 0
            )
            for leaf_size in sizes:
                seg_chips = nodes_per_seg * leaf_size
                if spec.chips % seg_chips:
                    continue  # does not decompose into whole segments
                n_segments = spec.chips // seg_chips
                total_nodes = n_segments * nodes_per_seg
                avail = self._free_nodes[did].get(
                    (ct, leaf_size), 0
                ) - self._ineligible_free_nodes(did, ct, leaf_size, tenant)
                if avail < total_nodes:
                    continue  # counter prune: not enough free nodes at all
                chosen = self._pack_segments(
                    did, ct, leaf_size, seg_level, nodes_per_seg, n_segments,
                    tenant,
                )
                if chosen is None:
                    continue
                return self._realize_segments(
                    chosen, ct, leaf_size, nodes_per_seg, tenant
                )
        return None

    def _pack_segments(
        self,
        did: str,
        ct: str,
        leaf_size: int,
        seg_level: str,
        nodes_per_seg: int,
        n_segments: int,
        tenant: str | None = None,
    ) -> list[tuple[str, int]] | None:
        """Assign ``n_segments`` segments to segment-level domains under
        ``did`` by descending free-node capacity, or ``None`` if they do
        not all fit.  Returns ``[(segment_domain_id, k_segments), ...]``
        in assignment order."""
        cands: list[tuple[int, str]] = []
        for sd in self.domains_at_under(seg_level, did):
            free_n = self._free_nodes[sd].get(
                (ct, leaf_size), 0
            ) - self._ineligible_free_nodes(sd, ct, leaf_size, tenant)
            if free_n >= nodes_per_seg:
                cands.append((free_n, sd))
        cands.sort(key=lambda pair: (-pair[0], pair[1]))
        remaining = n_segments
        chosen: list[tuple[str, int]] = []
        for free_n, sd in cands:
            if remaining <= 0:
                break
            k = min(remaining, free_n // nodes_per_seg)
            if k > 0:
                chosen.append((sd, k))
                remaining -= k
        return chosen if remaining == 0 else None

    def _realize_segments(
        self,
        chosen: list[tuple[str, int]],
        ct: str,
        leaf_size: int,
        nodes_per_seg: int,
        tenant: str | None = None,
    ) -> Placement:
        """Pick concrete leaves for a committed segment assignment and
        build the Placement (leaf scan limited to chosen domains)."""
        leaves: list[tuple[str, int]] = []
        seg_domains: list[str] = []
        for sd, k in chosen:
            need = k * nodes_per_seg
            for lid in self._leaves_under[sd]:  # ascending id order
                leaf = self._domains[lid]
                if (
                    leaf.chip_type == ct
                    and leaf.chips == leaf_size
                    and leaf.state is NodeState.HEALTHY
                    and not self._owners[lid]
                    and self._leaf_eligible(lid, tenant)
                ):
                    leaves.append((lid, leaf.chips))
                    need -= 1
                    if need == 0:
                        break
            if need:  # pragma: no cover - counters guarantee availability
                raise AssertionError(
                    f"free-node counter mismatch under {sd!r}:"
                    f" {need} nodes short"
                )
            seg_domains.extend([sd] * k)
        anchor = chosen[0][0]
        for sd, _ in chosen[1:]:
            lca = self.lca(anchor, sd)
            if lca is None:  # pragma: no cover - one outer domain root
                raise AssertionError("segment domains have no common root")
            anchor = lca
        leaves.sort(key=lambda pair: pair[0])  # Placement invariant
        return Placement(
            leaves=tuple(leaves),
            anchor=anchor,
            chip_type=ct,
            whole_node=True,
            segment_domains=tuple(seg_domains),
        )

    def _search_domain(
        self, did: str, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        types = (
            (spec.chip_type,) if spec.chip_type is not None else self._types_under[did]
        )
        for ct in types:
            if self._free_ct[did].get(ct, 0) < spec.chips:
                continue  # cheap upper-bound prune
            found = self._scan_leaves(did, ct, spec.chips, tenant)
            if found is not None:
                leaves, whole = found
                return Placement(
                    leaves=leaves, anchor=did, chip_type=ct, whole_node=whole
                )
        return None

    def _scan_leaves(
        self, did: str, chip_type: str, chips: int, tenant: str | None = None
    ) -> tuple[tuple[tuple[str, int], ...], bool] | None:
        """One-domain scan: first-fit sub-node placement (ascending leaf-id
        order), else an exact whole-node cover accumulated LARGEST leaves
        first (ties by ascending id).  Largest-first is exact whenever leaf
        sizes under the domain are uniform (every COMPACT-form fleet — the
        template form can mix ``chips`` across a cluster's ``children``
        templates) or form a divisor chain (8/16/32...); for arbitrary
        mixed sizes an exact cover is a subset-sum problem and this greedy
        may miss one — a documented v0.1 limitation, not silent (returns
        None = no fit).
        Leaves reserved for a different tenant are skipped (v0.4).
        """
        candidates: list[tuple[str, int]] = []
        reserved = bool(self._reserved)
        for lid in self._leaves_under[did]:
            leaf = self._domains[lid]
            if leaf.chip_type != chip_type or leaf.state is not NodeState.HEALTHY:
                continue
            if reserved and not self._leaf_eligible(lid, tenant):
                continue
            if chips < leaf.chips:
                if leaf.free_chips >= chips:
                    return ((lid, chips),), False  # sub-node first fit
            elif leaf.free_chips == leaf.chips:
                candidates.append((lid, leaf.chips))
        # Whole-node exact cover, largest leaves first (stable in id).
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        whole: list[tuple[str, int]] = []
        total = 0
        for lid, size in candidates:
            if total + size <= chips:
                whole.append((lid, size))
                total += size
                if total == chips:
                    whole.sort(key=lambda pair: pair[0])  # Placement invariant
                    return tuple(whole), True
        return None

    # ------------------------------------------------------------------
    # Packed placement search (v0.7): best-fit / consolidate / spread
    #
    # These are ADDITIVE raw primitives, exactly like search_first_fit and
    # search_segmented: they never mutate the tree and they leave
    # search_first_fit / _scan_leaves untouched (the v0.1 default path is
    # byte-identical).  On UNIFORM leaf sizes a gang's FEASIBILITY is the
    # same question for all of them — they differ only in WHICH of the
    # fitting leaves is chosen.  On mixed leaf sizes (reachable only via
    # the template form) the packed modes' whole-node feasibility is a
    # SUPERSET of first-fit's, never a subset: see _scan_leaves_packed.
    # ------------------------------------------------------------------

    #: The packing modes :meth:`search` understands (v0.7).  ``first_fit``
    #: is the v0.1 default; the other three are the opt-in policies.
    PACK_MODES: tuple[str, ...] = ("first_fit", "best_fit", "consolidate", "spread")

    def search(
        self, spec: GangSpec, tenant: str | None = None, *, mode: str = "first_fit"
    ) -> Placement | None:
        """Dispatch a raw placement search by packing ``mode`` (one of
        :data:`PACK_MODES`; unknown modes raise ``ValueError``).

        ``mode="first_fit"`` is exactly :meth:`search_first_fit` — the
        v0.1 default path, unchanged.
        """
        if mode == "first_fit":
            return self.search_first_fit(spec, tenant)
        if mode == "best_fit":
            return self.search_best_fit(spec, tenant)
        if mode == "consolidate":
            return self.search_consolidate(spec, tenant)
        if mode == "spread":
            return self.search_spread(spec, tenant)
        raise ValueError(
            f"unknown placement search mode {mode!r}"
            f" (available: {', '.join(self.PACK_MODES)})"
        )

    def search_best_fit(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """TIGHTEST-FIT search for one gang, or ``None`` if nothing fits.

        Same feasibility as :meth:`search_first_fit` on uniform leaf sizes
        (a superset of it on mixed ones — see :meth:`_scan_leaves_packed`),
        different choice:

        - **search domains** (``spec.within.level`` domains, else the
          cluster roots) are tried in ascending ``(free_chips, id)`` order
          — the tightest-fitting domain first — instead of ascending id.
          A single-root fleet has exactly one candidate, so this is a
          no-op there.
        - **sub-node** requests (``chips < leaf.chips``) take the eligible
          leaf whose FREE chips are smallest-but-sufficient, ties by
          ascending leaf id, with an early exit on an exact fit.  This is
          Slurm ``cons_tres`` best-fit / ``CR_Pack_Nodes``: small gangs
          fill existing remainders instead of manufacturing new ones, so
          fully-free leaves stay whole for gangs that need whole nodes.
        - **whole-node** requests take the tightest of the leaves' PARENT
          domains whose free whole-node capacity covers the request; when
          no single parent covers it, parents are consumed in ASCENDING
          capacity order (fill the tight holes first, keep big empty
          domains intact).  Leaves inside a chosen parent are taken
          largest-first, ties ascending id — unchanged from
          :meth:`_scan_leaves`.

        A spec with ``segments`` delegates unchanged to
        :meth:`search_segmented` (segment packing already bin-packs by
        free-node capacity).

        WHY THIS MATTERS (v0.7 Helios finding): a leaf with ANY owner is
        invisible to every whole-node request, so first-fit-by-id
        placement of sub-node gangs strands free chips as 1..n-1-chip
        remainders that no whole-node gang can use.  On the Helios Saturn
        replay (70.8% single-GPU jobs) that stranding, not gang
        consolidation, was the whole FIFO-inflation mechanism — see
        docs/validation.md §4.2.

        DETERMINISM: the total order ``(free_chips, leaf_id)`` is a pure
        function of tree state.  NOTE that leaf ids sort
        LEXICOGRAPHICALLY (``node0, node1, node10, ..., node2``), so
        "ascending leaf id" is not numeric order — the same convention
        :meth:`_scan_leaves` has always used.
        """
        return self._search_packed(spec, tenant, "best_fit")

    def search_consolidate(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """FEWEST-DOMAINS-TOUCHED search for one gang, or ``None``.

        Identical to :meth:`search_best_fit` except for a whole-node
        request that no single parent domain can cover alone: parents are
        then consumed in DESCENDING free-node capacity (ties ascending
        id), which minimizes the NUMBER of distinct parent domains the
        gang touches — the same bin-packing rule
        :meth:`_pack_segments` uses for segmented gangs.

        The minimized quantity is the count of PARENT domains (the deepest
        grouping level) and nothing coarser: on a 3-level fleet this policy
        can span MORE pods than ``first_fit`` while touching the same
        number of racks (worked example and measured numbers in
        :meth:`_packed_whole_order`).  "Fewest domains" therefore means
        "fewest parents", not "fewest crossings at every level".

        HONEST DEGENERACY: on a fleet whose leaves are all direct children
        of the searched domain — one level, i.e. every ``levels: ["node"]``
        config, including the Helios validation replay — there is exactly
        ONE parent domain, so the whole-node path here is *identical* to
        :meth:`search_first_fit`'s and the only behavior change is the
        sub-node tightest-fit.  That is precisely why the v0.6
        "consolidate large gangs across domains" hypothesis could not
        have explained the Helios Saturn gap: there were no domains to
        consolidate within.
        """
        return self._search_packed(spec, tenant, "consolidate")

    def search_spread(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """MAXIMUM-SPREAD search for one gang, or ``None`` — the
        deliberate ANTI-policy (a control arm for placement studies, and
        the proof that the policies really differ).

        - search domains in DESCENDING ``free_chips`` (ties ascending id);
        - sub-node requests take the eligible leaf with the MOST free
          chips (worst fit) — it opens fresh leaves and manufactures the
          remainders best-fit avoids;
        - whole-node requests round-robin one leaf at a time across the
          leaves' parent domains (parents by descending capacity, ties
          ascending id; leaves inside a parent largest-first) so a gang
          lands on as many distinct domains as it can.

        Use it to bracket a placement study: on the Helios replay it is
        measurably WORSE than first-fit, which is the point.
        """
        return self._search_packed(spec, tenant, "spread")

    def _search_packed(
        self, spec: GangSpec, tenant: str | None, mode: str
    ) -> Placement | None:
        """Shared driver for the three packed modes (see their docstrings)."""
        if spec.chips <= 0:
            raise ValueError(f"gang spec needs a positive chip count, got {spec.chips}")
        if spec.segments is not None:
            return self.search_segmented(spec, tenant)
        if spec.within is not None:
            search_domains: Sequence[str] = self.domains_at(spec.within.level)
        else:
            search_domains = self._cluster_roots
        if len(search_domains) > 1:
            sign = -1 if mode == "spread" else 1
            search_domains = sorted(
                search_domains,
                key=lambda did: (sign * self._domains[did].free_chips, did),
            )
        for did in search_domains:
            placement = self._search_domain_packed(did, spec, tenant, mode)
            if placement is not None:
                return placement
        return None

    def _search_domain_packed(
        self, did: str, spec: GangSpec, tenant: str | None, mode: str
    ) -> Placement | None:
        """One-domain packed search — the :meth:`_search_domain` shape
        (same chip-type order, same cheap free-chip prune)."""
        types = (
            (spec.chip_type,) if spec.chip_type is not None else self._types_under[did]
        )
        for ct in types:
            if self._free_ct[did].get(ct, 0) < spec.chips:
                continue  # cheap upper-bound prune
            found = self._scan_leaves_packed(did, ct, spec.chips, tenant, mode)
            if found is not None:
                leaves, whole = found
                return Placement(
                    leaves=leaves, anchor=did, chip_type=ct, whole_node=whole
                )
        return None

    def _scan_leaves_packed(
        self, did: str, chip_type: str, chips: int, tenant: str | None, mode: str
    ) -> tuple[tuple[tuple[str, int], ...], bool] | None:
        """One-domain scan under a packed ``mode``.

        Mirrors :meth:`_scan_leaves`'s structure and rules exactly — same
        eligibility filter (chip type, HEALTHY, v0.4 tenant reservation
        holds), same sub-node-beats-whole-node precedence, same
        largest-leaf-first greedy exact cover inside a chosen group — and
        changes only WHICH candidate wins.

        FEASIBILITY CAVEAT (identical in kind to :meth:`_scan_leaves`'s):
        with MIXED leaf sizes under one domain, an exact whole-node cover
        is a subset-sum problem and any greedy may miss one.  Grouping the
        leaves before the greedy runs can hide a cover the UNGROUPED
        largest-first order finds, so when the candidates are not all the
        same size the scan retries once in exactly
        :meth:`_scan_leaves`'s order.  That makes a packed mode's
        whole-node feasibility a SUPERSET of ``first_fit``'s — never a
        subset — so opting into a policy can never strand a gang the
        default would have placed.  (Mixed leaf sizes are reachable in a
        v1 config: the COMPACT form is uniform by construction, but the
        template form lets one cluster hold ``children`` templates with
        different ``chips``.)  Uniform candidates take exactly one pass,
        so every compact-form fleet is bit-identical to the un-retried
        version.
        """
        sub: list[tuple[int, str]] = []  # (sort key, leaf id)
        whole_cands: list[tuple[str, int]] = []  # (leaf id, leaf chips)
        reserved = bool(self._reserved)
        for lid in self._leaves_under[did]:
            leaf = self._domains[lid]
            if leaf.chip_type != chip_type or leaf.state is not NodeState.HEALTHY:
                continue
            if reserved and not self._leaf_eligible(lid, tenant):
                continue
            if chips < leaf.chips:
                free = leaf.free_chips
                if free < chips:
                    continue
                if mode == "spread":
                    sub.append((-free, lid))  # worst fit: most free first
                else:
                    if free == chips:
                        return ((lid, chips),), False  # exact fit is minimal
                    sub.append((free, lid))
            elif leaf.free_chips == leaf.chips:
                whole_cands.append((lid, leaf.chips))
        if sub:
            sub.sort()  # (key, id) — deterministic total order
            return ((sub[0][1], chips),), False
        if not whole_cands:
            return None
        orders = [self._packed_whole_order(whole_cands, chips, mode)]
        if len({size for _, size in whole_cands}) > 1:
            # MIXED leaf sizes only: grouping can hide a cover the ungrouped
            # largest-first greedy finds, so retry in _scan_leaves' exact
            # order rather than lose feasibility to the policy choice.
            orders.append(sorted(whole_cands, key=lambda pair: (-pair[1], pair[0])))
        for order in orders:
            whole: list[tuple[str, int]] = []
            total = 0
            for lid, size in order:
                if total + size <= chips:
                    whole.append((lid, size))
                    total += size
                    if total == chips:
                        whole.sort(key=lambda pair: pair[0])  # Placement invariant
                        return tuple(whole), True
        return None

    def _packed_whole_order(
        self, cands: list[tuple[str, int]], chips: int, mode: str
    ) -> list[tuple[str, int]]:
        """Order whole-node candidates for the greedy exact cover, by mode.

        Candidates are grouped by their PARENT domain — the deepest
        grouping level, which is the searched domain itself when the
        leaves are its direct children (or when the searched domain IS a
        leaf).  Within every group leaves stay largest-first, ties
        ascending id: exactly :meth:`_scan_leaves`'s order.  Groups are
        then visited:

        - ``best_fit`` / ``consolidate``: groups whose capacity covers
          ``chips`` ALONE come first, tightest (smallest sufficient)
          capacity first — so the gang lands inside one domain and the
          domain it lands in is the one with the least room to spare.
          Groups that cannot cover it alone follow, ASCENDING by capacity
          for ``best_fit`` (fill the tight holes, keep big empty domains
          intact) and DESCENDING for ``consolidate`` (fewest PARENT
          domains touched — the :meth:`_pack_segments` rule).
        - ``spread``: round-robin one leaf per group, groups by descending
          capacity, ties ascending id.

        SCOPE OF "FEWEST DOMAINS" (measured, not argued).  The grouping is
        FLAT and happens at the parent level only, so ``consolidate``
        minimizes the number of PARENT domains — never the number of
        domains at any coarser level.  On a 3-level fleet
        (``levels: [pod, rack, node]``, ``counts: [2, 2, 4]``) with
        pod0/rack0 3 free nodes, pod0/rack1 2 free, pod1/rack0 4 free and
        pod1/rack1 none, a 40-chip gang goes: ``first_fit`` and
        ``best_fit`` 1 pod / 2 racks, ``consolidate`` **2 pods** / 2 racks
        — more pods and no fewer racks.  So ``consolidate`` is NOT the
        policy to reach for to minimize ``penalties.xover`` crossings above
        the parent level; ``tests/test_placement.py`` pins this case so the
        limitation stays visible.

        With ONE group — every single-level fleet, e.g. the Helios replay's
        ``levels: ["node"]`` — all three modes return precisely
        :meth:`_scan_leaves`'s order, so the whole-node path is unchanged
        there.  Ties break on ascending group id throughout.
        """
        groups: dict[str, list[tuple[str, int]]] = {}
        for lid, size in cands:
            gid = self._domains[lid].parent or lid
            groups.setdefault(gid, []).append((lid, size))
        for members in groups.values():
            members.sort(key=lambda pair: (-pair[1], pair[0]))
        if len(groups) == 1:
            return next(iter(groups.values()))
        caps = {gid: sum(size for _, size in m) for gid, m in groups.items()}
        if mode == "spread":
            order = sorted(groups, key=lambda gid: (-caps[gid], gid))
            out: list[tuple[str, int]] = []
            i = 0
            while any(i < len(groups[gid]) for gid in order):
                for gid in order:
                    members = groups[gid]
                    if i < len(members):
                        out.append(members[i])
                i += 1
            return out
        fits = sorted(
            (gid for gid in groups if caps[gid] >= chips),
            key=lambda gid: (caps[gid], gid),
        )
        short = [gid for gid in groups if caps[gid] < chips]
        if mode == "consolidate":
            # Fewest groups: biggest first, ties ASCENDING id.
            rest = sorted(short, key=lambda gid: (-caps[gid], gid))
        else:
            # best_fit: tightest holes first, ties ascending id.
            rest = sorted(short, key=lambda gid: (caps[gid], gid))
        out = []
        for gid in fits + rest:
            out.extend(groups[gid])
        return out

    def search_after_release(
        self,
        spec: GangSpec,
        job_ids: Iterable[str],
        tenant: str | None = None,
        *,
        mode: str = "first_fit",
    ) -> Placement | None:
        """Dry-run search: the placement ``spec`` would find if the
        allocations of ``job_ids`` were released — WITHOUT changing tree
        state (v0.2, preemption planning).

        Releases each applied allocation in ``job_ids`` (ids without an
        applied allocation are skipped), runs :meth:`search` under
        ``mode`` (which delegates segmented specs), then restores every
        released allocation exactly — including on non-HEALTHY leaves,
        whose chips correctly contribute nothing to the free counters in
        either direction.  This makes reclaim planning exact under both
        leaf health and node-shape effects: a victim on a DRAINING node
        frees nothing, and sub-node co-residents keep their leaves out of
        the whole-node pool.

        ``mode`` (v0.7) must match the PLACEMENT POLICY the calling
        scheduler actually places with, or reclaim planning silently
        predicts a placement the policy would not make.  It defaults to
        ``"first_fit"`` — the v0.2 behavior, unchanged — because that is
        the default policy; a scheduler running an opt-in packed policy
        passes its own mode (see
        :attr:`fleetsim.schedulers.placement.FirstFit.search_mode`).

        INVARIANT: tree state (counters, owners, allocations) is
        byte-identical on exit; only the insertion order of restored
        allocations/owner entries may move to the end of their tables
        (all consumers sort).
        """
        saved: list[Allocation] = []
        try:
            for jid in job_ids:
                alloc = self._allocs.get(jid)
                if alloc is None:
                    continue
                self.release(jid)
                saved.append(alloc)
            return self.search(spec, tenant, mode=mode)
        finally:
            for alloc in saved:
                self._reapply(alloc)

    def _reapply(self, allocation: Allocation) -> None:
        """Exact inverse of :meth:`release` for a just-released allocation:
        re-record owners and re-decrement free counters on HEALTHY leaves
        only (non-HEALTHY leaves already contribute zero free).  No
        validation — callers guarantee the allocation was applied moments
        ago and nothing else touched its leaves in between."""
        jid = allocation.job_id
        for lid, (chips, _) in self._demand(allocation).items():
            leaf = self._domains[lid]
            healthy = leaf.state is NodeState.HEALTHY
            if healthy and not self._owners[lid]:  # free node -> owned node
                self._add_free_nodes(lid, -1)
            self._owners[lid][jid] = chips
            if healthy:
                self._add_free(lid, -chips)
        self._allocs[jid] = allocation

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------

    def fail_node(self, leaf_id: str) -> list[str]:
        """Mark a leaf FAILED.  Allowed from HEALTHY or DRAINING.

        Returns the resident ``alloc_id``\\ s (job ids) in sorted order —
        the engine kills those gangs and then :meth:`release`\\ s them.
        Owners are NOT removed here; their chips simply stop counting as
        free (they already did for a DRAINING leaf).
        """
        leaf = self._require_leaf(leaf_id)
        if leaf.state not in (NodeState.HEALTHY, NodeState.DRAINING):
            raise ValueError(
                f"cannot fail {leaf_id!r} from state {leaf.state.name}"
            )
        if leaf.state is NodeState.HEALTHY:
            self._make_unhealthy(leaf_id)
        leaf.state = NodeState.FAILED
        return sorted(self._owners[leaf_id])

    def repair_node(self, leaf_id: str) -> None:
        """Return a non-HEALTHY leaf to HEALTHY (post-repair, end of
        maintenance, or cancelled drain).  Raises if already HEALTHY.
        Free counters resume at ``chips - used`` — any owners still present
        (e.g. drain residents) keep their chips."""
        leaf = self._require_leaf(leaf_id)
        if leaf.state is NodeState.HEALTHY:
            raise ValueError(f"{leaf_id!r} is already HEALTHY")
        leaf.state = NodeState.HEALTHY
        self._add_healthy(leaf_id, leaf.chips)
        avail = leaf.chips - sum(self._owners[leaf_id].values())
        if avail:
            self._add_free(leaf_id, avail)
        if not self._owners[leaf_id]:  # returns as a free node
            self._add_free_nodes(leaf_id, 1)

    def drain_node(self, leaf_id: str) -> None:
        """HEALTHY -> DRAINING: blocks new placements (free contribution
        drops to zero) while residents keep running (owners untouched)."""
        leaf = self._require_leaf(leaf_id)
        if leaf.state is not NodeState.HEALTHY:
            raise ValueError(
                f"cannot drain {leaf_id!r} from state {leaf.state.name}"
            )
        self._make_unhealthy(leaf_id)
        leaf.state = NodeState.DRAINING

    def to_maintenance(self, leaf_id: str) -> None:
        """DRAINING -> MAINTENANCE, after the drain grace: residents must
        already be gone (raises if any owner remains)."""
        leaf = self._require_leaf(leaf_id)
        if leaf.state is not NodeState.DRAINING:
            raise ValueError(
                f"cannot enter maintenance on {leaf_id!r} from state"
                f" {leaf.state.name}"
            )
        if self._owners[leaf_id]:
            raise ValueError(
                f"cannot enter maintenance on {leaf_id!r}: residents remain"
                f" {sorted(self._owners[leaf_id])}"
            )
        leaf.state = NodeState.MAINTENANCE

    def _make_unhealthy(self, leaf_id: str) -> None:
        """Counter side of a HEALTHY -> non-HEALTHY transition."""
        leaf = self._domains[leaf_id]
        free = leaf.free_chips
        if free:
            self._add_free(leaf_id, -free)
        self._add_healthy(leaf_id, -leaf.chips)
        if not self._owners[leaf_id]:  # free node -> unhealthy node
            self._add_free_nodes(leaf_id, -1)

    # ------------------------------------------------------------------
    # Fragmentation queries (DESIGN §9; v1 tree semantics, no geometry)
    #
    # NOTE (v0.4): these are tenant-agnostic — free chips on leaves held
    # by a calendar reservation still count, so during an active hold
    # ``largest_placeable`` can exceed what any NON-owner tenant could
    # place (subtract :meth:`reserved_free_chips` for a tenant-honest
    # view).  Documented blind spot; holds are rare and short.
    # ------------------------------------------------------------------

    def largest_placeable(self, level: str, chip_type: str | None = None) -> int:
        """Max over domains at ``level`` of free healthy chips — the
        biggest single-domain gang placeable at that level right now."""
        best = 0
        for did in self.domains_at(level):
            free = self.free_chips(did, chip_type)
            if free > best:
                best = free
        return best

    def fragmentation_index(self, level: str, chip_type: str | None = None) -> float:
        """``1 - largest_placeable / total_free`` over domains at
        ``level`` (0.0 when nothing is free)."""
        total = 0
        best = 0
        for did in self.domains_at(level):
            free = self.free_chips(did, chip_type)
            total += free
            if free > best:
                best = free
        if total == 0:
            return 0.0
        return 1.0 - best / total

    def stranded_chips(self, quantum: int, chip_type: str | None = None) -> int:
        """Free chips sitting in leaves whose free count is below
        ``quantum`` (the smallest gang quantum) — capacity no gang of that
        size can use."""
        if quantum <= 0:
            raise ValueError(f"quantum must be positive, got {quantum}")
        total = 0
        for lid in self._leaf_ids:
            leaf = self._domains[lid]
            if chip_type is not None and leaf.chip_type != chip_type:
                continue
            free = leaf.free_chips
            if free < quantum:
                total += free
        return total

    def stranded_whole_nodes(self, chip_type: str | None = None) -> int:
        """Count of HEALTHY leaves that are PARTIALLY occupied
        (``0 < free_chips < leaf.chips``) — v0.7.

        This is the node-granularity fragmentation the allocation model
        creates: a leaf with ANY owner is ineligible for every whole-node
        request (see the module docstring), so a partially-used node's
        free chips can be claimed only by another SUB-NODE gang.  A rising
        count means small gangs are manufacturing remainders that starve
        whole-node gangs — the exact mechanism
        :meth:`search_best_fit` exists to suppress.

        O(leaves); the metrics collector samples it at flush.
        """
        n = 0
        for lid in self._leaf_ids:
            leaf = self._domains[lid]
            if chip_type is not None and leaf.chip_type != chip_type:
                continue
            if leaf.state is not NodeState.HEALTHY:
                continue
            if 0 < leaf.free_chips < leaf.chips:
                n += 1
        return n

    def stranded_whole_node_chips(self, chip_type: str | None = None) -> int:
        """Free chips sitting on partially-occupied HEALTHY leaves (the
        chip-weighted companion of :meth:`stranded_whole_nodes`) — free
        capacity that no whole-node gang can ever claim, v0.7."""
        total = 0
        for lid in self._leaf_ids:
            leaf = self._domains[lid]
            if chip_type is not None and leaf.chip_type != chip_type:
                continue
            if leaf.state is not NodeState.HEALTHY:
                continue
            free = leaf.free_chips
            if 0 < free < leaf.chips:
                total += free
        return total

    # ------------------------------------------------------------------
    # Counter plumbing (O(depth) per call)
    # ------------------------------------------------------------------

    def _add_free(self, leaf_id: str, delta: int) -> None:
        if delta == 0:
            return
        chip_type = self._domains[leaf_id].chip_type
        cur: str | None = leaf_id
        while cur is not None:
            dom = self._domains[cur]
            dom.free_chips += delta
            by_type = self._free_ct[cur]
            by_type[chip_type] = by_type.get(chip_type, 0) + delta
            cur = dom.parent

    def _add_healthy(self, leaf_id: str, delta: int) -> None:
        cur: str | None = leaf_id
        while cur is not None:
            self._healthy[cur] += delta
            cur = self._domains[cur].parent

    def _add_free_nodes(self, leaf_id: str, delta: int) -> None:
        """Adjust the free-NODE counter for ``leaf_id`` on every ancestor
        (called on HEALTHY/zero-owner boundary transitions only)."""
        leaf = self._domains[leaf_id]
        key = (leaf.chip_type, leaf.chips)
        cur: str | None = leaf_id
        while cur is not None:
            table = self._free_nodes[cur]
            table[key] = table.get(key, 0) + delta
            cur = self._domains[cur].parent

    # ------------------------------------------------------------------
    # Debug / test support
    # ------------------------------------------------------------------

    def check_invariants(self) -> None:
        """Recompute every counter from scratch and compare with the
        incrementally-maintained values; verify owner/allocation
        consistency.  Raises ``AssertionError`` on the first mismatch.
        O(N·depth) — tests and debugging only, never the hot path.
        """
        for did, dom in self._domains.items():
            leaf_ids = self._leaves_under[did]
            total = healthy = free = 0
            free_ct: dict[str, int] = {}
            free_nodes: dict[tuple[str, int], int] = {}
            for lid in leaf_ids:
                leaf = self._domains[lid]
                total += leaf.chips
                if leaf.state is NodeState.HEALTHY:
                    healthy += leaf.chips
                    avail = leaf.chips - sum(self._owners[lid].values())
                    free += avail
                    free_ct[leaf.chip_type] = free_ct.get(leaf.chip_type, 0) + avail
                    if not self._owners[lid]:
                        key = (leaf.chip_type, leaf.chips)
                        free_nodes[key] = free_nodes.get(key, 0) + 1
            if dom.total_chips != total:
                raise AssertionError(
                    f"{did!r}: total_chips {dom.total_chips} != recomputed {total}"
                )
            if self._healthy[did] != healthy:
                raise AssertionError(
                    f"{did!r}: healthy {self._healthy[did]} != recomputed {healthy}"
                )
            if dom.free_chips != free:
                raise AssertionError(
                    f"{did!r}: free_chips {dom.free_chips} != recomputed {free}"
                )
            for ct in sorted(set(free_ct) | set(self._free_ct[did])):
                if self._free_ct[did].get(ct, 0) != free_ct.get(ct, 0):
                    raise AssertionError(
                        f"{did!r}: free[{ct!r}] {self._free_ct[did].get(ct, 0)}"
                        f" != recomputed {free_ct.get(ct, 0)}"
                    )
            for key in sorted(set(free_nodes) | set(self._free_nodes[did])):
                if self._free_nodes[did].get(key, 0) != free_nodes.get(key, 0):
                    raise AssertionError(
                        f"{did!r}: free_nodes[{key!r}]"
                        f" {self._free_nodes[did].get(key, 0)}"
                        f" != recomputed {free_nodes.get(key, 0)}"
                    )
        for lid in self._leaf_ids:
            leaf = self._domains[lid]
            used = sum(self._owners[lid].values())
            if used > leaf.chips:
                raise AssertionError(f"{lid!r}: used {used} > chips {leaf.chips}")
            if leaf.state is NodeState.MAINTENANCE and self._owners[lid]:
                raise AssertionError(
                    f"{lid!r}: MAINTENANCE leaf has owners"
                    f" {sorted(self._owners[lid])}"
                )
            expected_free = leaf.chips - used if leaf.state is NodeState.HEALTHY else 0
            if leaf.free_chips != expected_free:
                raise AssertionError(
                    f"{lid!r}: leaf free_chips {leaf.free_chips}"
                    f" != expected {expected_free}"
                )
            for jid in self._owners[lid]:
                if jid not in self._allocs:
                    raise AssertionError(
                        f"{lid!r}: owner {jid!r} has no applied allocation"
                    )
        for jid, alloc in self._allocs.items():
            for lid, (chips, _) in self._demand(alloc).items():
                if self._owners[lid].get(jid) != chips:
                    raise AssertionError(
                        f"allocation {jid!r} expects {chips} chips on {lid!r},"
                        f" owners say {self._owners[lid].get(jid)}"
                    )
