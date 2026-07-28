"""Fleet domain tree: capacity counters, gang allocation, first-fit search.

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
    cluster root when the spec carried no constraint.

    INVARIANT: ``sum(chips for _, chips in leaves) == spec.chips`` of the
    searched :class:`~fleetsim.model.GangSpec`.
    """

    leaves: tuple[tuple[str, int], ...]
    anchor: str
    chip_type: str
    whole_node: bool

    @property
    def chips(self) -> int:
        """Total chips this placement covers."""
        return sum(chips for _, chips in self.leaves)

    def to_gang_alloc(self) -> GangAlloc:
        """Render as a :class:`~fleetsim.model.GangAlloc`: a list of leaf
        ids for a whole-node gang, a ``{leaf: chips}`` dict for sub-node."""
        if self.whole_node:
            return GangAlloc(nodes=[leaf for leaf, _ in self.leaves], anchor=self.anchor)
        return GangAlloc(
            nodes={leaf: chips for leaf, chips in self.leaves}, anchor=self.anchor
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
        self._healthy: dict[str, int] = {did: 0 for did in self._domains}
        self._free_ct: dict[str, dict[str, int]] = {did: {} for did in self._domains}

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

    def owners(self, leaf_id: str) -> dict[str, int]:
        """Copy of the ``{alloc_id: chips}`` owner map of a leaf."""
        return dict(self._owners[leaf_id])

    def used_chips(self, leaf_id: str) -> int:
        """Chips on ``leaf_id`` currently owned by allocations."""
        return sum(self._owners[leaf_id].values())

    def has_allocation(self, job_id: str) -> bool:
        return job_id in self._allocs

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

    def search_first_fit(self, spec: GangSpec) -> Placement | None:
        """First-fit search for one gang, or ``None`` if nothing fits now.

        Search domains: all domains at ``spec.within.level`` when the spec
        has a ``within`` constraint (treated as hard — v0.1 rejects
        relaxable constraints upstream), else the cluster roots.  Domains
        are tried independently in ascending id order; within a domain,
        leaves are scanned in ascending id order.  ``spec.chip_type`` pins
        the chip type; when unpinned, each chip type present under the
        domain is tried in sorted order (v1 configs pin, per DESIGN §11).
        ``segments``/``shape``/``twisted`` are v0.3 semantics and ignored.

        Per-leaf mode: a request smaller than the leaf's chip count is
        sub-node (needs ``free >= chips``); otherwise whole-node leaves are
        accumulated (each must be HEALTHY and fully free) until the request
        is exactly covered.
        """
        if spec.chips <= 0:
            raise ValueError(f"gang spec needs a positive chip count, got {spec.chips}")
        if spec.within is not None:
            search_domains = self.domains_at(spec.within.level)
        else:
            search_domains = self._cluster_roots
        for did in search_domains:
            placement = self._search_domain(did, spec)
            if placement is not None:
                return placement
        return None

    def _search_domain(self, did: str, spec: GangSpec) -> Placement | None:
        types = (
            (spec.chip_type,) if spec.chip_type is not None else self._types_under[did]
        )
        for ct in types:
            if self._free_ct[did].get(ct, 0) < spec.chips:
                continue  # cheap upper-bound prune
            found = self._scan_leaves(did, ct, spec.chips)
            if found is not None:
                leaves, whole = found
                return Placement(
                    leaves=leaves, anchor=did, chip_type=ct, whole_node=whole
                )
        return None

    def _scan_leaves(
        self, did: str, chip_type: str, chips: int
    ) -> tuple[tuple[tuple[str, int], ...], bool] | None:
        """One-domain scan: first-fit sub-node placement (ascending leaf-id
        order), else an exact whole-node cover accumulated LARGEST leaves
        first (ties by ascending id).  Largest-first is exact whenever leaf
        sizes under the domain are uniform (every v1-config fleet) or form
        a divisor chain (8/16/32...); for arbitrary mixed sizes an exact
        cover is a subset-sum problem and this greedy may miss one — a
        documented v0.1 limitation, not silent (returns None = no fit).
        """
        candidates: list[tuple[str, int]] = []
        for lid in self._leaves_under[did]:
            leaf = self._domains[lid]
            if leaf.chip_type != chip_type or leaf.state is not NodeState.HEALTHY:
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

    # ------------------------------------------------------------------
    # Fragmentation queries (DESIGN §9; v1 tree semantics, no geometry)
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
            for lid in leaf_ids:
                leaf = self._domains[lid]
                total += leaf.chips
                if leaf.state is NodeState.HEALTHY:
                    healthy += leaf.chips
                    avail = leaf.chips - sum(self._owners[lid].values())
                    free += avail
                    free_ct[leaf.chip_type] = free_ct.get(leaf.chip_type, 0) + avail
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
