"""Instantiate the runtime :class:`~fleetsim.fleet.tree.FleetTree` from a
parsed scenario's declarative fleet config.

:func:`build_fleet` expands each cluster's :class:`~fleetsim.config.NodeGroup`
count-tree (both YAML forms are already normalized to it by
``fleetsim.config``) into concrete :class:`~fleetsim.model.Domain` objects
with deterministic path ids::

    <metro>/<cluster>/<level><i>/.../<level><j>

for example ``us-central/h100-main/pod0/rack3/node7``.  The index counts
instances of that level under one parent, in document order (no zero
padding — id order is lexicographic, which is the tree's deterministic
search order).

Datacenters exist in the config for grouping only and do NOT appear as
domains or id segments in v0.1 — cluster domains hang directly off their
metro (so cluster ids must be unique within a metro; duplicates raise).

``chip_type`` is set on every interior domain whose subtree is
homogeneous (DESIGN 3.2), including cluster roots and metros.

LEMON NODES (DESIGN §8): when a cluster's failure model sets
``lemon_frac > 0``, that fraction of its leaves get ``lemon_factor =
lemon_multiplier``.  Selection is a pure function of the leaf id (SHA-256
hash mapped to [0, 1), lemon iff below ``lemon_frac``) — deterministic,
seed-independent, and stable under fleet growth.

UNITS: chip counts only — nothing here is a time.

INVARIANTS: a pure function of the config — no randomness beyond the
id-hash lemon selection (which is itself deterministic), no wall clock,
deterministic ids and iteration order.  The returned tree starts fully
HEALTHY and unallocated.
"""

from __future__ import annotations

import hashlib

from ..config import FleetConfig, NodeGroup, Scenario
from ..model import Domain
from .tree import FleetTree

__all__ = ["build_fleet"]


def _lemon_u(leaf_id: str) -> float:
    """Stable uniform-[0,1) value for a leaf id (SHA-256, platform- and
    process-independent)."""
    digest = hashlib.sha256(leaf_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") / 2.0**64


def build_fleet(scenario: Scenario | FleetConfig) -> FleetTree:
    """Build the runtime fleet tree from a :class:`~fleetsim.config.Scenario`
    (or a bare :class:`~fleetsim.config.FleetConfig`).

    Raises ``ValueError`` on id collisions (e.g. the same cluster id twice
    in one metro) or structurally invalid groups — configs that passed
    ``fleetsim.config.validate`` never trip these.
    """
    fleet = scenario.fleet if isinstance(scenario, Scenario) else scenario
    domains: list[Domain] = []
    cluster_roots: list[str] = []
    for metro in fleet.metros:
        metro_dom = Domain(
            id=metro.name, level="metro", parent=None, children=[], chip_type=None
        )
        domains.append(metro_dom)
        metro_types: set[str] = set()
        for dc in metro.datacenters:
            for cluster in dc.clusters:
                cluster_id = f"{metro.name}/{cluster.id}"
                cluster_level = cluster.levels[0] if cluster.levels else "cluster"
                types = set()
                for group in cluster.children:
                    types |= _subtree_chip_types(group)
                metro_types |= types
                root = Domain(
                    id=cluster_id,
                    level=cluster_level,
                    parent=metro_dom.id,
                    children=[],
                    chip_type=_sole(types),
                    attrs=dict(cluster.attrs),
                )
                metro_dom.children.append(cluster_id)
                domains.append(root)
                cluster_roots.append(cluster_id)
                counters: dict[tuple[str, str], int] = {}
                start = len(domains)
                for group in cluster.children:
                    _instantiate(group, root, counters, domains)
                fm = cluster.failure_model
                if fm.lemon_frac > 0.0 and fm.lemon_multiplier != 1.0:
                    for dom in domains[start:]:
                        if not dom.children and _lemon_u(dom.id) < fm.lemon_frac:
                            dom.lemon_factor = fm.lemon_multiplier
        metro_dom.chip_type = _sole(metro_types)
    return FleetTree(domains, cluster_roots)


def _instantiate(
    group: NodeGroup,
    parent: Domain,
    counters: dict[tuple[str, str], int],
    domains: list[Domain],
) -> None:
    """Append ``group.count`` concrete copies of ``group``'s subtree under
    ``parent``, numbering each level per parent in document order."""
    is_leaf = not group.children
    chip_type = group.chip_type if is_leaf else _sole(_subtree_chip_types(group))
    for _ in range(group.count):
        key = (parent.id, group.level)
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        dom = Domain(
            id=f"{parent.id}/{group.level}{idx}",
            level=group.level,
            parent=parent.id,
            children=[],
            chip_type=chip_type,
            chips=group.chips if is_leaf else 0,
            attrs=dict(group.attrs),
        )
        parent.children.append(dom.id)
        domains.append(dom)
        for child in group.children:
            _instantiate(child, dom, counters, domains)


def _subtree_chip_types(group: NodeGroup) -> set[str]:
    """Chip types of all leaves in a group subtree (declarative, so the
    answer is identical for every instantiated copy)."""
    if not group.children:
        return set() if group.chip_type is None else {group.chip_type}
    types: set[str] = set()
    for child in group.children:
        types |= _subtree_chip_types(child)
    return types


def _sole(types: set[str]) -> str | None:
    """The single element of a homogeneous type set, else ``None``."""
    if len(types) == 1:
        return next(iter(types))
    return None
