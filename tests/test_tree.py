"""Tests for fleetsim.fleet: building the domain tree from both YAML forms,
allocation apply/release invariants, sub-node vs whole-node rules,
within-constraint search, node lifecycle, and fragmentation queries."""

import pytest

from fleetsim.config import load_scenario
from fleetsim.fleet.build import build_fleet
from fleetsim.fleet.tree import FleetTree, Placement
from fleetsim.model import Allocation, Constraint, Domain, GangAlloc, GangSpec, NodeState

# ---------------------------------------------------------------------------
# Scenario builders (dict input avoids the YAML flow-mapping bracket gotcha)
# ---------------------------------------------------------------------------

WORKLOAD = {
    "kind": "synthetic",
    "classes": {
        "eval": {
            "rate_per_hour": 1,
            "chips": "pow2[1, 8]",
            "duration": "lognormal[median=2m, p90=30m]",
        }
    },
}


def compact_scenario(counts=(2, 2, 2), per_node=8):
    return load_scenario(
        {
            "sim": {"horizon": "1d"},
            "fleet": {
                "metro": "us-central",
                "clusters": [
                    {
                        "name": "h100-main",
                        "chip": {"type": "h100", "per_node": per_node},
                        "topology": {
                            "levels": ["pod", "rack", "node"],
                            "counts": list(counts),
                        },
                    }
                ],
            },
            "workload": WORKLOAD,
        }
    )


def template_scenario():
    """Template-form fleet describing the same tree as compact_scenario()."""
    return load_scenario(
        {
            "sim": {"horizon": "1d"},
            "chip_types": {
                "h100": {"vendor": "nvidia", "hbm_gib": 80, "peak_tflops_bf16": 989}
            },
            "templates": {
                "node_t": {"level": "node", "chips": 8, "chip_type": "h100"},
                "rack_t": {"level": "rack", "children": {"template": "node_t", "count": 2}},
                "pod_t": {"level": "pod", "children": {"template": "rack_t", "count": 2}},
            },
            "fleet": [
                {
                    "metro": "us-central",
                    "datacenters": [
                        {
                            "id": "dc1",
                            "clusters": [
                                {
                                    "id": "h100-main",
                                    "levels": ["cluster", "pod", "rack", "node"],
                                    "children": [{"template": "pod_t", "count": 2}],
                                }
                            ],
                        }
                    ],
                }
            ],
            "workload": WORKLOAD,
        }
    )


def two_type_scenario():
    """Two clusters with different chip types under one metro.  The
    workload class pins chip_type (heterogeneous fleets require it)."""
    workload = {
        "kind": "synthetic",
        "classes": {
            "eval": {
                "rate_per_hour": 1,
                "chips": "pow2[1, 8]",
                "chip_type": "h100",
                "duration": "lognormal[median=2m, p90=30m]",
            }
        },
    }
    return load_scenario(
        {
            "sim": {"horizon": "1d"},
            "fleet": {
                "metro": "us-central",
                "clusters": [
                    {
                        "name": "a-h100",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["rack", "node"], "counts": [1, 2]},
                    },
                    {
                        "name": "b-tpu",
                        "chip": {"type": "tpu_v5p", "per_node": 4},
                        "topology": {"levels": ["rack", "node"], "counts": [1, 2]},
                    },
                ],
            },
            "workload": workload,
        }
    )


C = "us-central/h100-main"  # cluster root id used by most tests


def sub_alloc(job_id, leaf, chips, anchor=C):
    return Allocation(job_id, [GangAlloc(nodes={leaf: chips}, anchor=anchor)])


def whole_alloc(job_id, leaves, anchor=C):
    return Allocation(job_id, [GangAlloc(nodes=list(leaves), anchor=anchor)])


# ---------------------------------------------------------------------------
# Building the tree
# ---------------------------------------------------------------------------


class TestBuild:
    def test_compact_form_ids_and_counters(self):
        tree = build_fleet(compact_scenario())
        assert "us-central" in tree
        assert C in tree
        assert f"{C}/pod0/rack0/node0" in tree
        assert f"{C}/pod1/rack1/node1" in tree
        assert len(tree.leaves()) == 8
        assert len(tree) == 1 + 1 + 2 + 4 + 8  # metro, cluster, pods, racks, nodes
        assert tree.total_chips("us-central") == 64
        assert tree.free_chips("us-central") == 64
        assert tree.healthy_chips("us-central") == 64
        assert tree.total_chips(f"{C}/pod0") == 32
        assert tree.total_chips(f"{C}/pod0/rack1") == 16
        leaf = tree.domain(f"{C}/pod0/rack0/node0")
        assert leaf.chips == 8
        assert leaf.chip_type == "h100"
        assert leaf.state is NodeState.HEALTHY
        assert tree.cluster_roots == (C,)
        tree.check_invariants()

    def test_homogeneous_chip_type_propagates_to_interior(self):
        tree = build_fleet(compact_scenario())
        assert tree.domain(C).chip_type == "h100"
        assert tree.domain(f"{C}/pod0").chip_type == "h100"
        assert tree.domain("us-central").chip_type == "h100"

    def test_mixed_chip_types_leave_interior_untyped(self):
        tree = build_fleet(two_type_scenario())
        assert tree.domain("us-central").chip_type is None
        assert tree.domain("us-central/a-h100").chip_type == "h100"
        assert tree.domain("us-central/b-tpu").chip_type == "tpu_v5p"

    def test_template_form_builds_identical_tree(self):
        compact = build_fleet(compact_scenario())
        template = build_fleet(template_scenario())
        assert sorted(compact) == sorted(template)
        for did in compact:
            cd, td = compact.domain(did), template.domain(did)
            assert (cd.level, cd.parent, cd.chips, cd.chip_type) == (
                td.level,
                td.parent,
                td.chips,
                td.chip_type,
            )
            assert cd.total_chips == td.total_chips
            assert cd.free_chips == td.free_chips
        assert compact.cluster_roots == template.cluster_roots
        template.check_invariants()

    def test_build_accepts_bare_fleet_config(self):
        scenario = compact_scenario()
        tree = build_fleet(scenario.fleet)
        assert tree.total_chips("us-central") == 64

    def test_levels_and_domains_at_sorted(self):
        tree = build_fleet(compact_scenario())
        assert tree.levels() == ("cluster", "metro", "node", "pod", "rack")
        assert tree.domains_at("metro") == ("us-central",)
        assert tree.domains_at("cluster") == (C,)
        assert tree.domains_at("pod") == (f"{C}/pod0", f"{C}/pod1")
        racks = tree.domains_at("rack")
        assert racks == tuple(sorted(racks))
        assert len(racks) == 4
        assert tree.domains_at("nope") == ()

    def test_parent_ancestors_depth_lca(self):
        tree = build_fleet(compact_scenario())
        leaf = f"{C}/pod0/rack0/node0"
        assert tree.parent(leaf) == f"{C}/pod0/rack0"
        assert tree.parent("us-central") is None
        assert list(tree.ancestors(leaf)) == [
            f"{C}/pod0/rack0",
            f"{C}/pod0",
            C,
            "us-central",
        ]
        assert next(tree.ancestors(leaf, include_self=True)) == leaf
        assert tree.depth("us-central") == 0
        assert tree.depth(leaf) == 4
        assert tree.lca(leaf, f"{C}/pod0/rack1/node1") == f"{C}/pod0"
        assert tree.lca(leaf, f"{C}/pod1/rack0/node0") == C
        assert tree.lca(leaf, leaf) == leaf
        assert tree.lca(leaf, f"{C}/pod0") == f"{C}/pod0"

    def test_lca_none_across_metros(self):
        tree = build_fleet(
            load_scenario(
                {
                    "sim": {"horizon": "1d"},
                    "fleet": [
                        {
                            "metro": "m1",
                            "datacenters": [
                                {
                                    "id": "dc0",
                                    "clusters": [
                                        {
                                            "name": "c",
                                            "chip": {"type": "h100", "per_node": 8},
                                            "topology": {
                                                "levels": ["node"],
                                                "counts": [1],
                                            },
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "metro": "m2",
                            "datacenters": [
                                {
                                    "id": "dc0",
                                    "clusters": [
                                        {
                                            "name": "c",
                                            "chip": {"type": "h100", "per_node": 8},
                                            "topology": {
                                                "levels": ["node"],
                                                "counts": [1],
                                            },
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                    "workload": WORKLOAD,
                }
            )
        )
        assert tree.lca("m1/c/node0", "m2/c/node0") is None
        assert tree.cluster_roots == ("m1/c", "m2/c")

    def test_duplicate_cluster_id_within_metro_raises(self):
        scenario = compact_scenario()
        fleet = scenario.fleet
        dc = fleet.metros[0].datacenters[0]
        dc.clusters.append(dc.clusters[0])  # same cluster id twice
        with pytest.raises(ValueError, match="duplicate domain id"):
            build_fleet(fleet)


# ---------------------------------------------------------------------------
# Apply / release
# ---------------------------------------------------------------------------


def assert_conserved(tree):
    """free + used == chips on every healthy leaf; full recheck of counters."""
    tree.check_invariants()
    for lid in tree.leaves():
        leaf = tree.domain(lid)
        if leaf.state is NodeState.HEALTHY:
            assert leaf.free_chips + tree.used_chips(lid) == leaf.chips
        else:
            assert leaf.free_chips == 0


class TestApplyRelease:
    def setup_method(self):
        self.tree = build_fleet(compact_scenario())
        self.n0 = f"{C}/pod0/rack0/node0"
        self.n1 = f"{C}/pod0/rack0/node1"

    def test_subnode_sharing_up_to_capacity(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        assert_conserved(t)
        assert t.owners(self.n0) == {"A": 4}
        assert t.free_chips(self.n0) == 4
        assert t.free_chips(C) == 60
        t.apply(sub_alloc("B", self.n0, 4))
        assert_conserved(t)
        assert t.owners(self.n0) == {"A": 4, "B": 4}
        assert t.free_chips(self.n0) == 0
        with pytest.raises(ValueError, match="over-capacity"):
            t.apply(sub_alloc("Z", self.n0, 1))
        assert_conserved(t)

    def test_release_restores_counters_at_every_level(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 5))
        for did in (self.n0, f"{C}/pod0/rack0", f"{C}/pod0", C, "us-central"):
            assert t.free_chips(did) == t.total_chips(did) - 5
        t.release("A")
        assert_conserved(t)
        assert t.owners(self.n0) == {}
        assert t.free_chips("us-central") == 64
        assert not t.has_allocation("A")

    def test_whole_node_takes_full_leaves(self):
        t = self.tree
        t.apply(whole_alloc("A", [self.n0, self.n1]))
        assert_conserved(t)
        assert t.owners(self.n0) == {"A": 8}
        assert t.owners(self.n1) == {"A": 8}
        assert t.free_chips(f"{C}/pod0/rack0") == 0
        t.release(t.allocation("A"))
        assert t.free_chips(f"{C}/pod0/rack0") == 16
        assert_conserved(t)

    def test_whole_node_rejects_leaf_with_subnode_owner(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 1))
        with pytest.raises(ValueError, match="exclusive"):
            t.apply(whole_alloc("B", [self.n0]))
        assert_conserved(t)

    def test_subnode_rejected_on_fully_owned_leaf(self):
        t = self.tree
        t.apply(whole_alloc("A", [self.n0]))
        with pytest.raises(ValueError, match="over-capacity"):
            t.apply(sub_alloc("B", self.n0, 1))
        assert_conserved(t)

    def test_double_apply_raises(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 2))
        with pytest.raises(ValueError, match="already applied"):
            t.apply(sub_alloc("A", self.n1, 2))
        assert_conserved(t)

    def test_double_release_and_unknown_release_raise(self):
        t = self.tree
        alloc = sub_alloc("A", self.n0, 2)
        t.apply(alloc)
        t.release(alloc)
        with pytest.raises(ValueError, match="no applied allocation"):
            t.release(alloc)
        with pytest.raises(ValueError, match="no applied allocation"):
            t.release("ghost")

    def test_apply_is_atomic_on_partial_failure(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n1, 1))  # taints n1 for whole-node use
        with pytest.raises(ValueError, match="exclusive"):
            t.apply(whole_alloc("B", [self.n0, self.n1]))
        # nothing from B landed, including on the valid leaf n0
        assert t.owners(self.n0) == {}
        assert t.free_chips(self.n0) == 8
        assert not t.has_allocation("B")
        assert_conserved(t)

    def test_apply_validates_leaf_ids(self):
        t = self.tree
        with pytest.raises(ValueError, match="unknown domain"):
            t.apply(whole_alloc("A", ["nope"]))
        with pytest.raises(ValueError, match="not a leaf"):
            t.apply(whole_alloc("A", [f"{C}/pod0"]))
        with pytest.raises(ValueError, match="exactly one leaf"):
            t.apply(
                Allocation("A", [GangAlloc(nodes={self.n0: 2, self.n1: 2}, anchor=C)])
            )
        assert_conserved(t)

    def test_applied_jobs_in_apply_order(self):
        t = self.tree
        t.apply(sub_alloc("B", self.n0, 1))
        t.apply(sub_alloc("A", self.n0, 1))
        assert t.applied_jobs() == ("B", "A")


# ---------------------------------------------------------------------------
# First-fit search
# ---------------------------------------------------------------------------


class TestSearch:
    def setup_method(self):
        self.tree = build_fleet(compact_scenario())

    def leaf(self, pod, rack, node):
        return f"{C}/pod{pod}/rack{rack}/node{node}"

    def test_subnode_first_fit_packs_first_leaf(self):
        t = self.tree
        p = t.search_first_fit(GangSpec(chips=4, chip_type="h100"))
        assert p == Placement(
            leaves=((self.leaf(0, 0, 0), 4),),
            anchor=C,
            chip_type="h100",
            whole_node=False,
        )
        t.apply(Allocation("A", [p.to_gang_alloc()]))
        # still 4 free on node0 -> first fit lands there again
        p2 = t.search_first_fit(GangSpec(chips=4, chip_type="h100"))
        assert p2.leaves == ((self.leaf(0, 0, 0), 4),)
        t.apply(Allocation("B", [p2.to_gang_alloc()]))
        p3 = t.search_first_fit(GangSpec(chips=4, chip_type="h100"))
        assert p3.leaves == ((self.leaf(0, 0, 1), 4),)
        assert_conserved(t)

    def test_whole_node_multi_leaf(self):
        t = self.tree
        p = t.search_first_fit(GangSpec(chips=16, chip_type="h100"))
        assert p.whole_node is True
        assert p.chips == 16
        assert p.anchor == C
        assert p.leaves == ((self.leaf(0, 0, 0), 8), (self.leaf(0, 0, 1), 8))
        gang = p.to_gang_alloc()
        assert gang.nodes == [self.leaf(0, 0, 0), self.leaf(0, 0, 1)]
        assert gang.anchor == C

    def test_whole_node_skips_leaf_with_subnode_owner(self):
        t = self.tree
        t.apply(sub_alloc("A", self.leaf(0, 0, 0), 1))
        p = t.search_first_fit(GangSpec(chips=8, chip_type="h100"))
        assert p.whole_node is True
        assert p.leaves == ((self.leaf(0, 0, 1), 8),)

    def test_subnode_gang_can_share_leaf_with_other_subnode(self):
        t = self.tree
        t.apply(sub_alloc("A", self.leaf(0, 0, 0), 7))
        p = t.search_first_fit(GangSpec(chips=1, chip_type="h100"))
        assert p.leaves == ((self.leaf(0, 0, 0), 1),)
        assert p.whole_node is False

    def test_within_constraint_searches_each_domain_independently(self):
        t = self.tree
        # occupy node0 of every rack: each rack keeps 8 free, none has 16
        for i, rack in enumerate(t.domains_at("rack")):
            t.apply(whole_alloc(f"J{i}", [f"{rack}/node0"], anchor=rack))
        within_rack = GangSpec(
            chips=16, chip_type="h100", within=Constraint(level="rack")
        )
        assert t.search_first_fit(within_rack) is None  # fragmentation!
        # without the constraint the same 16 chips fit across racks
        p = t.search_first_fit(GangSpec(chips=16, chip_type="h100"))
        assert p is not None
        assert p.anchor == C
        assert p.leaves == ((self.leaf(0, 0, 1), 8), (self.leaf(0, 1, 1), 8))

    def test_within_anchor_is_the_satisfying_domain(self):
        t = self.tree
        # fill rack0 of pod0 completely so the first fitting rack is pod0/rack1
        t.apply(whole_alloc("A", [self.leaf(0, 0, 0), self.leaf(0, 0, 1)]))
        p = t.search_first_fit(
            GangSpec(chips=16, chip_type="h100", within=Constraint(level="rack"))
        )
        assert p.anchor == f"{C}/pod0/rack1"
        assert p.leaves == ((self.leaf(0, 1, 0), 8), (self.leaf(0, 1, 1), 8))

    def test_within_unknown_level_finds_nothing(self):
        assert (
            self.tree.search_first_fit(
                GangSpec(chips=8, chip_type="h100", within=Constraint(level="su"))
            )
            is None
        )

    def test_no_fit_returns_none(self):
        t = self.tree
        assert t.search_first_fit(GangSpec(chips=128, chip_type="h100")) is None
        assert t.search_first_fit(GangSpec(chips=8, chip_type="tpu_v5p")) is None

    def test_search_skips_unhealthy_leaves(self):
        t = self.tree
        t.fail_node(self.leaf(0, 0, 0))
        t.drain_node(self.leaf(0, 0, 1))
        p = t.search_first_fit(GangSpec(chips=4, chip_type="h100"))
        assert p.leaves == ((self.leaf(0, 1, 0), 4),)

    def test_nonpositive_chips_raises(self):
        with pytest.raises(ValueError, match="positive chip count"):
            self.tree.search_first_fit(GangSpec(chips=0, chip_type="h100"))


class TestSearchChipTypes:
    def setup_method(self):
        self.tree = build_fleet(two_type_scenario())

    def test_pinned_chip_type_routes_to_matching_cluster(self):
        t = self.tree
        p = t.search_first_fit(GangSpec(chips=4, chip_type="tpu_v5p"))
        # 4 chips == tpu node size -> whole-node on the first tpu leaf
        assert p.whole_node is True
        assert p.leaves == (("us-central/b-tpu/rack0/node0", 4),)
        assert p.anchor == "us-central/b-tpu"
        assert p.chip_type == "tpu_v5p"

    def test_pinned_subnode_on_tpu(self):
        p = self.tree.search_first_fit(GangSpec(chips=2, chip_type="tpu_v5p"))
        assert p.whole_node is False
        assert p.leaves == (("us-central/b-tpu/rack0/node0", 2),)

    def test_unpinned_takes_first_cluster_in_sorted_order(self):
        p = self.tree.search_first_fit(GangSpec(chips=4, chip_type=None))
        assert p.chip_type == "h100"
        assert p.leaves == (("us-central/a-h100/rack0/node0", 4),)

    def test_gang_never_mixes_chip_types(self):
        t = self.tree
        # 16 chips: a-h100 has 16 (2x8 nodes), b-tpu has 8 (2x4): only h100 fits
        p = t.search_first_fit(GangSpec(chips=16, chip_type=None))
        assert p.chip_type == "h100"
        assert [c for _, c in p.leaves] == [8, 8]
        assert t.search_first_fit(GangSpec(chips=16, chip_type="tpu_v5p")) is None


# ---------------------------------------------------------------------------
# Node lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def setup_method(self):
        # 1 pod x 1 rack x 2 nodes of 8 = 16 chips
        self.tree = build_fleet(compact_scenario(counts=(1, 1, 2)))
        self.n0 = f"{C}/pod0/rack0/node0"
        self.n1 = f"{C}/pod0/rack0/node1"

    def test_fail_with_subnode_residents(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        t.apply(sub_alloc("B", self.n0, 2))
        assert t.free_chips(C) == 10
        victims = t.fail_node(self.n0)
        assert victims == ["A", "B"]
        assert t.domain(self.n0).state is NodeState.FAILED
        assert t.free_chips(C) == 8  # only node1 counts
        assert t.healthy_chips(C) == 8
        assert t.total_chips(C) == 16  # total includes failed capacity
        assert_conserved(t)
        # engine kills the gangs and releases them; free stays put (leaf down)
        t.release("A")
        t.release("B")
        assert t.free_chips(C) == 8
        assert t.owners(self.n0) == {}
        assert_conserved(t)
        t.repair_node(self.n0)
        assert t.domain(self.n0).state is NodeState.HEALTHY
        assert t.free_chips(C) == 16
        assert t.healthy_chips(C) == 16
        assert_conserved(t)

    def test_fail_whole_node_gang_spanning_leaves(self):
        t = self.tree
        t.apply(whole_alloc("A", [self.n0, self.n1]))
        assert t.fail_node(self.n0) == ["A"]
        assert t.domain(self.n1).state is NodeState.HEALTHY  # other member unhurt
        assert t.free_chips(C) == 0  # n0 down, n1 still fully owned by A
        t.release("A")
        assert t.free_chips(C) == 8  # n1 back; n0 still FAILED
        t.repair_node(self.n0)
        assert t.free_chips(C) == 16
        assert_conserved(t)

    def test_repair_before_release_keeps_counters_consistent(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        t.fail_node(self.n0)
        t.repair_node(self.n0)  # engine misordering: repair before release
        assert t.free_chips(self.n0) == 4  # A still holds its 4 chips
        assert_conserved(t)
        t.release("A")
        assert t.free_chips(self.n0) == 8
        assert_conserved(t)

    def test_drain_keeps_residents_blocks_placement(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        t.drain_node(self.n0)
        assert t.domain(self.n0).state is NodeState.DRAINING
        assert t.owners(self.n0) == {"A": 4}  # residents keep running
        assert t.free_chips(self.n0) == 0
        assert t.free_chips(C) == 8
        assert t.healthy_chips(C) == 8
        p = t.search_first_fit(GangSpec(chips=4, chip_type="h100"))
        assert p.leaves == ((self.n1, 4),)  # drained leaf not placeable
        assert_conserved(t)

    def test_drain_to_maintenance_requires_empty(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        t.drain_node(self.n0)
        with pytest.raises(ValueError, match="residents remain"):
            t.to_maintenance(self.n0)
        t.release("A")  # grace expired, engine preempted the resident
        t.to_maintenance(self.n0)
        assert t.domain(self.n0).state is NodeState.MAINTENANCE
        assert t.free_chips(C) == 8
        assert_conserved(t)
        t.repair_node(self.n0)
        assert t.free_chips(C) == 16
        assert_conserved(t)

    def test_fail_a_draining_node(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        t.drain_node(self.n0)
        free_before = t.free_chips(C)
        assert t.fail_node(self.n0) == ["A"]
        assert t.free_chips(C) == free_before  # already excluded while draining
        assert t.domain(self.n0).state is NodeState.FAILED
        assert_conserved(t)

    def test_cancel_drain_via_repair(self):
        t = self.tree
        t.apply(sub_alloc("A", self.n0, 4))
        t.drain_node(self.n0)
        t.repair_node(self.n0)
        assert t.domain(self.n0).state is NodeState.HEALTHY
        assert t.free_chips(self.n0) == 4  # A's chips still owned
        assert_conserved(t)

    def test_invalid_transitions_raise(self):
        t = self.tree
        with pytest.raises(ValueError, match="already HEALTHY"):
            t.repair_node(self.n0)
        with pytest.raises(ValueError, match="cannot enter maintenance"):
            t.to_maintenance(self.n0)  # not draining
        t.fail_node(self.n0)
        with pytest.raises(ValueError, match="cannot fail"):
            t.fail_node(self.n0)  # double fail
        with pytest.raises(ValueError, match="cannot drain"):
            t.drain_node(self.n0)  # drain a failed node
        with pytest.raises(ValueError, match="cannot enter maintenance"):
            t.to_maintenance(self.n0)  # failed, not draining
        t.repair_node(self.n0)
        t.drain_node(self.n0)
        with pytest.raises(ValueError, match="cannot drain"):
            t.drain_node(self.n0)  # double drain
        t.to_maintenance(self.n0)
        with pytest.raises(ValueError, match="cannot fail"):
            t.fail_node(self.n0)  # fail during maintenance
        assert_conserved(t)

    def test_lifecycle_on_interior_domain_raises(self):
        with pytest.raises(ValueError, match="not a leaf"):
            self.tree.fail_node(f"{C}/pod0")

    def test_allocation_blocked_on_unhealthy_leaf(self):
        t = self.tree
        t.fail_node(self.n0)
        with pytest.raises(ValueError, match="state is FAILED"):
            t.apply(sub_alloc("A", self.n0, 1))
        t.repair_node(self.n0)
        t.drain_node(self.n0)
        with pytest.raises(ValueError, match="state is DRAINING"):
            t.apply(whole_alloc("A", [self.n0]))
        assert_conserved(t)


# ---------------------------------------------------------------------------
# Fragmentation queries
# ---------------------------------------------------------------------------


class TestFragmentation:
    def setup_method(self):
        # 1 pod x 2 racks x 2 nodes of 8 = 32 chips
        self.tree = build_fleet(compact_scenario(counts=(1, 2, 2)))
        t = self.tree
        # rack0: node0 fully taken (whole-node), node1 has 5 of 8 taken
        t.apply(whole_alloc("A", [f"{C}/pod0/rack0/node0"]))
        t.apply(sub_alloc("B", f"{C}/pod0/rack0/node1", 5))
        # free now: node0=0, node1=3, rack0=3, rack1=16, cluster=19

    def test_largest_placeable(self):
        t = self.tree
        assert t.largest_placeable("rack") == 16
        assert t.largest_placeable("node") == 8
        assert t.largest_placeable("cluster") == 19
        assert t.largest_placeable("nope") == 0
        assert t.largest_placeable("rack", "h100") == 16
        assert t.largest_placeable("rack", "tpu_v5p") == 0

    def test_fragmentation_index(self):
        t = self.tree
        assert t.fragmentation_index("rack") == pytest.approx(1 - 16 / 19)
        assert t.fragmentation_index("node") == pytest.approx(1 - 8 / 19)
        assert t.fragmentation_index("cluster") == 0.0  # one domain holds all free

    def test_fragmentation_index_zero_when_no_free(self):
        t = self.tree
        t.apply(sub_alloc("C", f"{C}/pod0/rack0/node1", 3))
        t.apply(whole_alloc("D", [f"{C}/pod0/rack1/node0", f"{C}/pod0/rack1/node1"]))
        assert t.free_chips(C) == 0
        assert t.fragmentation_index("rack") == 0.0
        assert t.fragmentation_index("node") == 0.0

    def test_stranded_chips(self):
        t = self.tree
        # leaves free: 0, 3, 8, 8 -> below quantum 8: 0 + 3
        assert t.stranded_chips(8) == 3
        assert t.stranded_chips(4) == 3
        assert t.stranded_chips(3) == 0  # node1's 3 free exactly fits a 3-gang
        assert t.stranded_chips(16) == 19  # every leaf is below a 16 quantum
        assert t.stranded_chips(8, "tpu_v5p") == 0
        with pytest.raises(ValueError, match="quantum"):
            t.stranded_chips(0)

    def test_failure_shifts_fragmentation(self):
        t = self.tree
        t.fail_node(f"{C}/pod0/rack1/node0")
        # free now: rack0=3, rack1=8, total=11
        assert t.largest_placeable("rack") == 8
        assert t.fragmentation_index("rack") == pytest.approx(1 - 8 / 11)
        assert t.stranded_chips(8) == 3  # failed leaf contributes 0 free
        assert_conserved(t)

    def test_release_restores_fragmentation(self):
        t = self.tree
        t.release("A")
        t.release("B")
        assert t.largest_placeable("rack") == 16
        assert t.fragmentation_index("rack") == pytest.approx(0.5)  # 16 vs 32
        assert t.stranded_chips(8) == 0


# ---------------------------------------------------------------------------
# Hand-built trees (FleetTree without build_fleet) and validation
# ---------------------------------------------------------------------------


class TestHandBuiltTree:
    def make_domains(self):
        return [
            Domain(id="c", level="cluster", parent=None, children=["c/n0", "c/n1"],
                   chip_type="h100"),
            Domain(id="c/n0", level="node", parent="c", children=[], chip_type="h100",
                   chips=8),
            Domain(id="c/n1", level="node", parent="c", children=[], chip_type="h100",
                   chips=8),
        ]

    def test_counters_recomputed_and_roots_derived(self):
        tree = FleetTree(self.make_domains())
        assert tree.cluster_roots == ("c",)
        assert tree.total_chips("c") == 16
        assert tree.free_chips("c") == 16
        tree.check_invariants()

    def test_initially_unhealthy_leaf_excluded(self):
        domains = self.make_domains()
        domains[2].state = NodeState.FAILED
        tree = FleetTree(domains)
        assert tree.total_chips("c") == 16
        assert tree.free_chips("c") == 8
        assert tree.healthy_chips("c") == 8
        tree.check_invariants()

    def test_duplicate_id_raises(self):
        domains = self.make_domains() + [
            Domain(id="c/n0", level="node", parent="c", children=[], chip_type="h100",
                   chips=8)
        ]
        with pytest.raises(ValueError, match="duplicate domain id"):
            FleetTree(domains)

    def test_inconsistent_links_raise(self):
        domains = self.make_domains()
        domains[1].parent = "ghost"
        with pytest.raises(ValueError, match="unknown parent|has parent"):
            FleetTree(domains)
        domains = self.make_domains()
        domains[0].children.append("ghost")
        with pytest.raises(ValueError, match="unknown child"):
            FleetTree(domains)
        domains = self.make_domains()
        domains[0].children.remove("c/n1")
        with pytest.raises(ValueError, match="not listed in parent"):
            FleetTree(domains)

    def test_bad_leaves_raise(self):
        domains = self.make_domains()
        domains[1].chips = 0
        with pytest.raises(ValueError, match="chips > 0"):
            FleetTree(domains)
        domains = self.make_domains()
        domains[1].chip_type = None
        with pytest.raises(ValueError, match="chip_type"):
            FleetTree(domains)
        domains = self.make_domains()
        domains[0].chips = 4
        with pytest.raises(ValueError, match="must not carry chips"):
            FleetTree(domains)

    def test_explicit_cluster_roots_validated(self):
        with pytest.raises(ValueError, match="unknown cluster root"):
            FleetTree(self.make_domains(), cluster_roots=["ghost"])

    def test_empty_tree_raises(self):
        with pytest.raises(ValueError, match="at least one domain"):
            FleetTree([])


# ---------------------------------------------------------------------------
# End-to-end conservation through a scripted mixed sequence
# ---------------------------------------------------------------------------


def test_scripted_sequence_conserves_chips():
    tree = build_fleet(compact_scenario(counts=(2, 2, 2)))
    n = lambda p, r, k: f"{C}/pod{p}/rack{r}/node{k}"

    tree.apply(sub_alloc("e1", n(0, 0, 0), 1))
    assert_conserved(tree)
    tree.apply(sub_alloc("e2", n(0, 0, 0), 2))
    assert_conserved(tree)
    tree.apply(whole_alloc("t1", [n(0, 0, 1), n(0, 1, 0)]))
    assert_conserved(tree)
    victims = tree.fail_node(n(0, 0, 0))
    assert victims == ["e1", "e2"]
    assert_conserved(tree)
    tree.release("e1")
    tree.release("e2")
    assert_conserved(tree)
    tree.drain_node(n(0, 1, 1))
    assert_conserved(tree)
    p = tree.search_first_fit(GangSpec(chips=16, chip_type="h100"))
    assert p.leaves == ((n(1, 0, 0), 8), (n(1, 0, 1), 8))
    tree.apply(Allocation("t2", [p.to_gang_alloc()]))
    assert_conserved(tree)
    tree.repair_node(n(0, 0, 0))
    assert_conserved(tree)
    tree.to_maintenance(n(0, 1, 1))
    assert_conserved(tree)
    tree.release("t1")
    tree.release("t2")
    assert_conserved(tree)
    # everything back except the two down nodes
    assert tree.free_chips("us-central") == 64 - 8  # maintenance node excluded
    assert tree.healthy_chips("us-central") == 56
    assert tree.total_chips("us-central") == 64
    tree.repair_node(n(0, 1, 1))
    assert tree.free_chips("us-central") == 64
    assert_conserved(tree)
