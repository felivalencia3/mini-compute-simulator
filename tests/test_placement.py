"""v0.7 placement policies: BestFit / Consolidate / Spread (opt-in).

WHAT THIS FILE PINS

1. **Exact choices.** Each policy's chosen leaf ids on hand-built fleets —
   tightest-fit vs first-fit vs worst-fit really differ, and they differ in
   the direction each docstring claims.
2. **The H1 mechanism** (the reason the family exists): sub-node best-fit
   packing PRESERVES whole free nodes, so a whole-node gang places where
   first-fit has stranded the pool.  This is the Helios Saturn finding
   reduced to a five-node unit test.
3. **What must NOT change.** ``first_fit`` is the default everywhere,
   whole-node placement on a single-level fleet is identical under all
   three packed modes, segmented gangs are untouched, and a scenario that
   names no policy keeps the pre-v0.7 output schema.
4. **Config surface.** Every policy round-trips through
   ``scheduler.params.placement`` on all four built-in schedulers; unknown
   names are a validation error listing the available ones.
5. **Determinism and reservation/health honesty.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleetsim.api import run_scenario
from fleetsim.config import (
    ScenarioError,
    load_scenario,
    scheduler_placement_name,
    validate,
)
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.metrics.summary import (
    build_summary,
    jobs_dataframe,
    timeseries_dataframe,
)
from fleetsim.model import Allocation, GangAlloc, GangSpec, JobClass, Tier
from fleetsim.schedulers.base import get_scheduler
from fleetsim.schedulers.placement import (
    PLACEMENT_POLICIES,
    BestFit,
    Consolidate,
    FirstFit,
    Spread,
    get_placement,
    placement_names,
    resolve_placement,
)
from fleetsim.workload.trace import TraceJob, TraceSource

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_PACKED = ("best_fit", "consolidate", "spread")
_ALL_MODES = ("first_fit",) + _PACKED
_SCHEDULERS = ("fifo", "sjf", "tiered_priority", "easy_backfill")


# ---------------------------------------------------------------------------
# Fleet helpers
# ---------------------------------------------------------------------------


def _fleet(levels, counts, per_node=8, chip="h100"):
    """A one-cluster fleet with the given topology (ids ``m/c/...``)."""
    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": chip, "per_node": per_node},
                    "topology": {"levels": list(levels), "counts": list(counts)},
                }
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
    }
    return build_fleet(load_scenario(doc, strict=True))


def _occupy(fleet, leaf_id, chips, job_id):
    """Give ``job_id`` a sub-node slice of ``leaf_id``."""
    fleet.apply(
        Allocation(job_id, [GangAlloc(nodes={leaf_id: chips}, anchor="m/c")])
    )


def _take_whole(fleet, leaf_ids, job_id):
    fleet.apply(
        Allocation(job_id, [GangAlloc(nodes=list(leaf_ids), anchor="m/c")])
    )


def _ids(placement):
    return tuple(lid for lid, _ in placement.leaves)


# ---------------------------------------------------------------------------
# 1. Exact choices — the three policies really differ
# ---------------------------------------------------------------------------


class TestSubNodeChoice:
    """A 4-node/8-chip pool with two partially-used nodes.  free chips:
    node0=4, node1=2, node2=8, node3=8.  A 2-chip request:

    - first_fit  -> node0 (lowest id with room)
    - best_fit   -> node1 (smallest sufficient free)
    - consolidate-> node1 (same rule for sub-node)
    - spread     -> node2 (most free, ties by ascending id)
    """

    @staticmethod
    def _pool():
        f = _fleet(["node"], [4])
        _occupy(f, "m/c/node0", 4, "a")
        _occupy(f, "m/c/node1", 6, "b")
        return f

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("first_fit", "m/c/node0"),
            ("best_fit", "m/c/node1"),
            ("consolidate", "m/c/node1"),
            ("spread", "m/c/node2"),
        ],
    )
    def test_exact_leaf_chosen(self, mode, expected):
        f = self._pool()
        p = f.search(GangSpec(chips=2, chip_type="h100"), mode=mode)
        assert _ids(p) == (expected,), (mode, _ids(p))
        assert p.whole_node is False
        assert p.chips == 2
        assert p.anchor == "m/c"

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("first_fit", "m/c/node0"),
            ("best_fit", "m/c/node1"),
            ("consolidate", "m/c/node1"),
            ("spread", "m/c/node2"),
        ],
    )
    def test_policy_object_makes_the_same_choice(self, mode, expected):
        """The same table, driven through the POLICY OBJECT rather than the
        tree dispatcher — i.e. through ``_SearchPolicy._view_method`` and
        ``ClusterView``, the wiring every scenario actually uses.

        This exists because the tree-level tests do not pin it: silently
        repointing ``BestFit``/``Consolidate`` at ``search_first_fit`` left
        the whole suite green (verified by sabotage), so the policy -> tree
        binding needs its own exact-choice assertion for EVERY mode.
        """
        f = self._pool()
        p = get_placement(mode).place(_job_view(chips=2), _FleetView(f))
        assert _ids(p) == (expected,), (mode, _ids(p))
        assert p.whole_node is False
        assert p.chips == 2

    def test_best_fit_prefers_exact_fit_and_exits_early(self):
        """An EXACT fit is the minimum of ``(free, id)``, so it wins over a
        lower-id leaf with more room — and the scan may stop there."""
        f = _fleet(["node"], [4])
        _occupy(f, "m/c/node0", 4, "a")  # free 4
        _occupy(f, "m/c/node2", 5, "b")  # free 3
        p = f.search_best_fit(GangSpec(chips=3, chip_type="h100"))
        assert _ids(p) == ("m/c/node2",)

    def test_spread_is_worst_fit(self):
        """Spread deliberately opens a fresh node rather than filling a
        remainder — it MANUFACTURES the stranding best-fit suppresses."""
        f = _fleet(["node"], [3])
        _occupy(f, "m/c/node0", 7, "a")  # free 1, enough for a 1-chip job
        p = f.search_spread(GangSpec(chips=1, chip_type="h100"))
        assert _ids(p) == ("m/c/node1",)  # a fully free node
        assert _ids(f.search_best_fit(GangSpec(chips=1, chip_type="h100"))) == (
            "m/c/node0",
        )


class TestWholeNodeChoice:
    """Two racks of 3 nodes; rack0 has one node taken, so rack0 offers 16
    free chips and rack1 offers 24.  A 16-chip (2-node) gang:

    - first_fit   -> rack0's two free nodes (lowest ids)
    - best_fit    -> rack0 (tightest rack that fits alone)
    - consolidate -> rack0 (same)
    - spread      -> one node from each rack (maximum spread)
    """

    @staticmethod
    def _pool():
        f = _fleet(["rack", "node"], [2, 3])
        _take_whole(f, ["m/c/rack0/node0"], "x")
        return f

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("first_fit", ("m/c/rack0/node1", "m/c/rack0/node2")),
            ("best_fit", ("m/c/rack0/node1", "m/c/rack0/node2")),
            ("consolidate", ("m/c/rack0/node1", "m/c/rack0/node2")),
            ("spread", ("m/c/rack0/node1", "m/c/rack1/node0")),
        ],
    )
    def test_exact_leaves_chosen(self, mode, expected):
        f = self._pool()
        p = f.search(GangSpec(chips=16, chip_type="h100"), mode=mode)
        assert _ids(p) == expected, (mode, _ids(p))
        assert p.whole_node is True

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("first_fit", ("m/c/rack0/node1", "m/c/rack0/node2")),
            ("best_fit", ("m/c/rack0/node1", "m/c/rack0/node2")),
            ("consolidate", ("m/c/rack0/node1", "m/c/rack0/node2")),
            ("spread", ("m/c/rack0/node1", "m/c/rack1/node0")),
        ],
    )
    def test_policy_object_makes_the_same_choice(self, mode, expected):
        """Whole-node choice through the POLICY OBJECT for every mode — the
        binding the tree-level table above does not exercise (see
        ``TestSubNodeChoice.test_policy_object_makes_the_same_choice``)."""
        f = self._pool()
        p = get_placement(mode).place(_job_view(chips=16), _FleetView(f))
        assert _ids(p) == expected, (mode, _ids(p))
        assert p.whole_node is True

    def test_best_fit_and_consolidate_differ_when_no_rack_fits_alone(self):
        """3 racks x 2 nodes; rack1 loses one node (8 free whole-node
        chips), rack0 and rack2 are whole (16 each).  A 24-chip gang fits
        in no single rack, so the tie-break between the two policies shows:

        - best_fit consumes the TIGHT hole first -> rack1's remaining node
          plus rack0;
        - consolidate consumes the BIGGEST racks first -> rack0 + rack2,
          leaving rack1's odd node untouched.

        Both span two racks here (three nodes cannot come from fewer), but
        they pick different ones, and only best_fit uses up the small rack.
        """
        def pool():
            f = _fleet(["rack", "node"], [3, 2])
            _take_whole(f, ["m/c/rack1/node0"], "x")
            return f

        spec = GangSpec(chips=24, chip_type="h100")
        bf = _ids(pool().search_best_fit(spec))
        co = _ids(pool().search_consolidate(spec))
        assert bf == ("m/c/rack0/node0", "m/c/rack0/node1", "m/c/rack1/node1")
        assert co == ("m/c/rack0/node0", "m/c/rack0/node1", "m/c/rack2/node0")
        assert bf != co
        # best_fit left rack2 pristine (a future 16-chip gang still fits in
        # one rack); consolidate left rack1's odd node pristine instead.
        assert not any(lid.startswith("m/c/rack2/") for lid in bf)
        assert not any(lid.startswith("m/c/rack1/") for lid in co)


class TestSearchDomainOrder:
    """With several cluster roots the packed modes order the roots by free
    capacity (ascending; descending for spread) instead of by id."""

    @staticmethod
    def _two_clusters():
        doc = {
            "sim": {"horizon": 3600, "round": "60s", "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "big",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["node"], "counts": [4]},
                    },
                    {
                        "name": "small",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["node"], "counts": [2]},
                    },
                ],
            },
            "workload": {"kind": "trace", "source": "__inline__"},
        }
        return build_fleet(load_scenario(doc, strict=True))

    def test_tightest_root_first(self):
        f = self._two_clusters()
        spec = GangSpec(chips=8, chip_type="h100")
        # first_fit: ascending root id -> "m/big" sorts before "m/small".
        assert f.search_first_fit(spec).anchor == "m/big"
        # best_fit / consolidate: the tighter root (16 free vs 32) first.
        assert f.search_best_fit(spec).anchor == "m/small"
        assert f.search_consolidate(spec).anchor == "m/small"
        # spread: the emptiest root first.
        assert f.search_spread(spec).anchor == "m/big"


# ---------------------------------------------------------------------------
# 2. The H1 mechanism — sub-node packing preserves whole free nodes
# ---------------------------------------------------------------------------


class TestStrandingMechanism:
    """The v0.7 Helios finding as a unit test.

    A leaf with ANY owner is invisible to every whole-node request, so
    placing sub-node gangs by ascending id strands free chips as
    1..n-1-chip remainders.  Drive four 1-chip jobs into a 4-node pool
    whose low-id nodes are momentarily full, then ask for a whole node.
    """

    @staticmethod
    def _drive(mode, n_small=4):
        f = _fleet(["node"], [4])
        for i in range(n_small):
            p = f.search(GangSpec(chips=1, chip_type="h100"), mode=mode)
            assert p is not None
            f.apply(Allocation(f"s{i}", [p.to_gang_alloc()]))
        return f

    def test_best_fit_keeps_whole_nodes_whole(self):
        """first_fit dirties 1 node per arrival only until it fills; the
        harmful case is the one measured on Helios — see the next test.
        Here the invariant: best-fit NEVER opens a second node while the
        first has room, so exactly one node is partial."""
        ff = self._drive("first_fit")
        bf = self._drive("best_fit")
        sp = self._drive("spread")
        assert ff.stranded_whole_nodes() == 1  # id-order fills node0 first
        assert bf.stranded_whole_nodes() == 1
        # spread opens a FRESH node for every arrival: 4 partial nodes and
        # zero whole free nodes left, from 4 chips of demand on 32.
        assert sp.stranded_whole_nodes() == 4
        assert sp.free_full_nodes("m/c", "h100", 8) == 0
        assert bf.free_full_nodes("m/c", "h100", 8) == 3
        for f in (ff, bf, sp):
            f.check_invariants()

    def test_spread_strands_a_whole_node_request_best_fit_does_not(self):
        """The end-to-end consequence: with 28 of 32 chips free, a 3-node
        (24-chip) gang PLACES under best-fit and FAILS under spread —
        purely because of where the four 1-chip jobs went."""
        gang = GangSpec(chips=24, chip_type="h100")
        assert self._drive("best_fit").search_best_fit(gang) is not None
        assert self._drive("spread").search_spread(gang) is None
        # ... and the chips are all still there: it is shape, not capacity.
        assert self._drive("spread").free_chips("m/c") == 28

    def test_best_fit_refills_a_released_node_instead_of_dirtying_a_fresh_one(self):
        """The other half of the mechanism: a node a whole-node gang just
        released must not be immediately re-dirtied by the next 1-chip
        arrival while a partial node still has room."""
        f = _fleet(["node"], [3])
        _take_whole(f, ["m/c/node0"], "gang")  # node0 exclusively held
        _occupy(f, "m/c/node1", 4, "resident")  # node1 partial, 4 free
        f.release("gang")  # node0 is now pristine and lowest-id
        assert _ids(f.search_first_fit(GangSpec(1, "h100"))) == ("m/c/node0",)
        assert _ids(f.search_best_fit(GangSpec(1, "h100"))) == ("m/c/node1",)


# ---------------------------------------------------------------------------
# 3. What must NOT change
# ---------------------------------------------------------------------------


def test_first_fit_search_is_untouched():
    """The v0.1 default path: FirstFit must return exactly what the raw
    ``search_first_fit`` primitive returns, for sub-node, whole-node and
    no-fit cases alike."""
    f = _fleet(["rack", "node"], [2, 2])
    _occupy(f, "m/c/rack0/node0", 5, "a")
    for chips in (1, 3, 8, 16, 24, 40):
        spec = GangSpec(chips=chips, chip_type="h100")
        assert f.search(spec, mode="first_fit") == f.search_first_fit(spec)


@pytest.mark.parametrize("mode", _PACKED)
def test_whole_node_placement_identical_on_single_level_fleet(mode):
    """On a single-level fleet every leaf is a direct child of the cluster
    root, so there is exactly ONE parent domain and the packed modes'
    whole-node path degenerates to first-fit's.  This is the honest
    degeneracy that makes "consolidate large gangs across domains" a
    non-explanation for the Helios Saturn gap."""
    f = _fleet(["node"], [8])
    _occupy(f, "m/c/node1", 3, "a")  # one partial node, ignored by whole-node
    for chips in (8, 16, 32, 40):
        spec = GangSpec(chips=chips, chip_type="h100")
        assert f.search(spec, mode=mode) == f.search_first_fit(spec), chips


def test_consolidate_equals_best_fit_on_single_level_fleet():
    """Stated in Consolidate's docstring; asserted here so it stays true."""
    f = _fleet(["node"], [6])
    _occupy(f, "m/c/node0", 5, "a")
    _occupy(f, "m/c/node3", 2, "b")
    for chips in (1, 2, 3, 6, 8, 16, 24):
        spec = GangSpec(chips=chips, chip_type="h100")
        assert f.search_consolidate(spec) == f.search_best_fit(spec), chips


@pytest.mark.parametrize("mode", _PACKED)
def test_segmented_specs_delegate_unchanged(mode):
    """A spec with ``segments`` is Slurm-block semantics; the packed modes
    must not touch it."""
    f = _fleet(["rack", "node"], [4, 4])
    spec = GangSpec(chips=64, chip_type="h100", segments=(2, "rack"))
    assert f.search(spec, mode=mode) == f.search_segmented(spec)


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_feasibility_is_mode_independent_on_uniform_fleets(mode):
    """The modes choose differently but must agree on WHETHER a gang fits:
    on a uniform-leaf fleet a fit exists for all modes or for none.  Driven
    over a deterministic sweep of tree states."""
    for taken in range(0, 4):
        for partial in range(0, 4):
            f = _fleet(["rack", "node"], [2, 3])
            leaves = f.leaves()
            if taken:
                _take_whole(f, leaves[:taken], "whole")
            for i in range(min(partial, len(leaves) - taken)):
                _occupy(f, leaves[taken + i], 1 + i, f"p{i}")
            for chips in (1, 4, 8, 16, 24, 48):
                spec = GangSpec(chips=chips, chip_type="h100")
                got = f.search(spec, mode=mode)
                ref = f.search_first_fit(spec)
                assert (got is None) == (ref is None), (
                    mode, taken, partial, chips
                )
                if got is not None:
                    assert got.chips == chips
                    assert got.whole_node == ref.whole_node


# ---------------------------------------------------------------------------
# 4. Determinism, reservations, health
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_search_is_deterministic(mode):
    f = _fleet(["rack", "node"], [3, 3])
    _occupy(f, "m/c/rack1/node1", 3, "a")
    _occupy(f, "m/c/rack2/node0", 6, "b")
    _take_whole(f, ["m/c/rack0/node0"], "c")
    for chips in (2, 5, 8, 24):
        spec = GangSpec(chips=chips, chip_type="h100")
        first = f.search(spec, mode=mode)
        for _ in range(3):
            assert f.search(spec, mode=mode) == first


@pytest.mark.parametrize("mode", _PACKED)
def test_reservation_holds_are_honored(mode):
    """v0.4 calendar holds: a leaf reserved for another tenant is invisible
    to every packed scan too — the owning tenant places on it, nobody else
    can, and a tenant-less search skips it (conservative)."""
    f = _fleet(["node"], [3])
    _take_whole(f, ["m/c/node1", "m/c/node2"], "occupant")  # only node0 free
    f.reserve_leaves(["m/c/node0"], "tenant-x")
    spec = GangSpec(chips=2, chip_type="h100")
    assert _ids(f.search(spec, "tenant-x", mode=mode)) == ("m/c/node0",)
    assert f.search(spec, "tenant-y", mode=mode) is None
    assert f.search(spec, None, mode=mode) is None


@pytest.mark.parametrize("mode", _PACKED)
def test_non_healthy_leaves_are_skipped(mode):
    f = _fleet(["node"], [3])
    f.drain_node("m/c/node0")
    f.fail_node("m/c/node1")
    spec = GangSpec(chips=2, chip_type="h100")
    assert _ids(f.search(spec, mode=mode)) == ("m/c/node2",)
    assert f.search(GangSpec(chips=24, chip_type="h100"), mode=mode) is None


@pytest.mark.parametrize("mode", _PACKED)
def test_chip_type_is_respected(mode):
    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "gpu",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [2]},
                },
                {
                    "name": "tpu",
                    "chip": {"type": "tpu_v5p", "per_node": 4},
                    "topology": {"levels": ["node"], "counts": [2]},
                },
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
    }
    f = build_fleet(load_scenario(doc, strict=True))
    p = f.search(GangSpec(chips=2, chip_type="tpu_v5p"), mode=mode)
    assert p.chip_type == "tpu_v5p"
    assert all(lid.startswith("m/tpu/") for lid, _ in p.leaves)


def test_randomized_sweep_agrees_on_feasibility_and_produces_valid_placements():
    """Seeded randomized sweep over 1-, 2- and 3-level uniform-leaf fleets
    with random whole-node and sub-node occupancy.

    Two properties, both load-bearing:

    1. all four modes AGREE on whether a gang fits (they choose
       differently, never *feasibly* differently, on uniform leaves);
    2. every returned placement is one the tree actually accepts —
       ``apply`` succeeds, ``check_invariants`` passes, ``release`` restores
       state.  A placement policy that returned a subtly invalid gang
       (double-booked leaf, wrong total, whole-node claim on a dirty leaf)
       would raise in the engine's strict mode, so this is the guard.
    """
    import random

    rng = random.Random(7)  # fixed: the sweep is a deterministic test
    topologies = (
        (["node"], [6]),
        (["rack", "node"], [3, 3]),
        (["pod", "rack", "node"], [2, 2, 3]),
    )
    for trial in range(60):
        levels, counts = topologies[trial % len(topologies)]
        f = _fleet(levels, counts)
        for i, lid in enumerate(f.leaves()):
            r = rng.random()
            if r < 0.3:
                _take_whole(f, [lid], f"w{i}")
            elif r < 0.6:
                _occupy(f, lid, rng.randint(1, 7), f"s{i}")
        for chips in (1, 2, 5, 8, 16, 24, 48):
            spec = GangSpec(chips=chips, chip_type="h100")
            found = {m: f.search(spec, mode=m) for m in _ALL_MODES}
            feasible = {m: p is not None for m, p in found.items()}
            assert len(set(feasible.values())) == 1, (trial, chips, feasible)
            for mode, p in found.items():
                if p is None:
                    continue
                assert p.chips == chips, (mode, p)
                f.apply(Allocation("probe", [p.to_gang_alloc()]))
                f.check_invariants()
                f.release("probe")
                f.check_invariants()


def _mixed_fleet(big=16, small=12, per_rack=2):
    """A TEMPLATE-form cluster with two racks of different node sizes.

    Mixed leaf sizes are reachable in a v1 config — the compact form is
    uniform by construction, but ``templates`` + ``children`` are not — and
    ``validate()`` accepts this document.  That is why the packed modes must
    not lose feasibility on it.
    """
    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "templates": {
            "big_node": {"level": "node", "chips": big, "chip_type": "h100"},
            "small_node": {"level": "node", "chips": small, "chip_type": "h100"},
            "rack_big": {
                "level": "rack",
                "children": [{"template": "big_node", "count": per_rack}],
            },
            "rack_small": {
                "level": "rack",
                "children": [{"template": "small_node", "count": per_rack}],
            },
        },
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "id": "c",
                    "levels": ["rack", "node"],
                    "children": [
                        {"template": "rack_big", "count": 1},
                        {"template": "rack_small", "count": 1},
                    ],
                }
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
    }
    scn = load_scenario(doc, strict=True)
    assert validate(scn) == []  # a legal v1 config, not a contrived one
    return build_fleet(scn)


def test_mixed_leaf_sizes_do_not_lose_a_whole_node_fit():
    """REGRESSION: with mixed leaf sizes the group-ordered greedy used to
    return ``None`` where ``first_fit`` placed.

    rack0 = 2 x 16 chips, rack1 = 2 x 12 chips, everything free.  A 16-chip
    whole-node request: ``best_fit``/``consolidate`` order rack1 first (24 is
    the tightest sufficient capacity), commit to a 12-chip leaf and then
    cannot reach 16 — while a perfectly-sized fully-free node sits idle in
    rack0.  Under a strict scan that is an indefinite stall caused purely by
    opting into a placement policy, so the packed scan retries once in
    ``_scan_leaves``' ungrouped order.
    """
    spec = GangSpec(chips=16, chip_type="h100")
    for mode in _ALL_MODES:
        p = _mixed_fleet().search(spec, mode=mode)
        assert p is not None, mode
        assert p.whole_node is True and p.chips == 16, mode
        assert _ids(p) == ("m/c/rack0/node0",), (mode, _ids(p))


def test_mixed_leaf_sizes_feasibility_is_never_worse_than_first_fit():
    """The general contract on mixed leaves: a packed mode places whenever
    ``first_fit`` does (a SUPERSET, not an equality — a packed order can
    also find a cover largest-first misses).  Swept over random occupancy of
    several mixed-size fleets; a single counterexample is a real stall."""
    import random

    rng = random.Random(11)
    worse = []
    better = 0
    for trial in range(150):
        big, small = rng.choice([(16, 12), (8, 4), (16, 8), (12, 4)])
        f = _mixed_fleet(big=big, small=small, per_rack=rng.choice([2, 3]))
        for i, lid in enumerate(f.leaves()):
            r = rng.random()
            size = f._domains[lid].chips
            if r < 0.25:
                _take_whole(f, [lid], f"w{i}")
            elif r < 0.5:
                _occupy(f, lid, rng.randint(1, size - 1), f"s{i}")
        for chips in (1, 4, 8, 12, 16, 20, 24, 28, 32):
            spec = GangSpec(chips=chips, chip_type="h100")
            ref = f.search_first_fit(spec)
            for mode in _PACKED:
                got = f.search(spec, mode=mode)
                if ref is not None and got is None:
                    worse.append((trial, chips, mode))
                elif ref is None and got is not None:
                    better += 1
    assert worse == [], worse[:10]
    assert better >= 0  # a packed order finding MORE is allowed, not required


def test_consolidate_minimizes_parent_domains_not_pods():
    """DOCUMENTED LIMITATION, pinned so the claim cannot silently widen.

    ``consolidate`` groups whole-node candidates by their PARENT domain
    only, so "fewest domains touched" means fewest RACKS here, never fewest
    pods.  On this 3-level fleet a 40-chip (5-node) gang measures:
    ``first_fit`` and ``best_fit`` 1 pod / 2 racks, ``consolidate`` 2 pods /
    2 racks — more pods, no fewer racks.  So ``consolidate`` is not the
    policy that minimizes ``penalties.xover`` crossings above the parent
    level, and docs/placement.md must not say it is.
    """
    def pool():
        f = _fleet(["pod", "rack", "node"], [2, 2, 4])
        _take_whole(
            f,
            ["m/c/pod0/rack0/node3", "m/c/pod0/rack1/node2", "m/c/pod0/rack1/node3"]
            + [f"m/c/pod1/rack1/node{i}" for i in range(4)],
            "x",
        )
        return f

    spec = GangSpec(chips=40, chip_type="h100")
    spans = {}
    for mode in _ALL_MODES:
        ids = _ids(pool().search(spec, mode=mode))
        assert len(ids) == 5, (mode, ids)
        spans[mode] = (
            len({lid.split("/")[2] for lid in ids}),
            len({"/".join(lid.split("/")[:4]) for lid in ids}),
        )
    assert spans["first_fit"] == (1, 2), spans
    assert spans["best_fit"] == (1, 2), spans
    assert spans["consolidate"] == (2, 2), spans  # MORE pods than first_fit
    assert spans["consolidate"][1] == spans["first_fit"][1]  # and no fewer racks


def test_bad_mode_and_bad_chip_count_raise():
    f = _fleet(["node"], [2])
    with pytest.raises(ValueError, match="unknown placement search mode"):
        f.search(GangSpec(chips=1, chip_type="h100"), mode="nope")
    for mode in _PACKED:
        with pytest.raises(ValueError, match="positive chip count"):
            f.search(GangSpec(chips=0, chip_type="h100"), mode=mode)


def test_search_after_release_threads_the_mode():
    """Reclaim planning must search the way the scheduler PLACES.  With a
    partial node and a whole node held by a victim, the dry-run's chosen
    leaf differs by mode — and the tree is restored exactly either way."""
    f = _fleet(["node"], [3])
    _occupy(f, "m/c/node0", 6, "resident")  # free 2
    _take_whole(f, ["m/c/node1", "m/c/node2"], "victim")
    spec = GangSpec(chips=2, chip_type="h100")
    ff = f.search_after_release(spec, ["victim"])
    bf = f.search_after_release(spec, ["victim"], mode="best_fit")
    assert _ids(ff) == ("m/c/node0",)  # lowest id with room
    assert _ids(bf) == ("m/c/node0",)  # also the tightest here
    sp = f.search_after_release(spec, ["victim"], mode="spread")
    assert _ids(sp) == ("m/c/node1",)  # freshly freed, most room
    f.check_invariants()
    assert f.has_allocation("victim")


# ---------------------------------------------------------------------------
# 5. Policy objects and the config surface
# ---------------------------------------------------------------------------


def test_registry_contents_and_lookup():
    assert placement_names() == ("best_fit", "consolidate", "first_fit", "spread")
    assert set(PLACEMENT_POLICIES) == set(_ALL_MODES)
    assert isinstance(get_placement("first_fit"), FirstFit)
    assert isinstance(get_placement("best_fit"), BestFit)
    assert isinstance(get_placement("consolidate"), Consolidate)
    assert isinstance(get_placement("spread"), Spread)
    for name, cls in PLACEMENT_POLICIES.items():
        assert cls().search_mode == name, name


def test_unknown_placement_name_lists_the_available_ones():
    with pytest.raises(ValueError) as exc:
        get_placement("bestfit")
    msg = str(exc.value)
    assert "unknown placement policy 'bestfit'" in msg
    for name in _ALL_MODES:
        assert name in msg


def test_resolve_placement_passes_objects_and_none_through():
    obj = BestFit()
    assert resolve_placement(obj) is obj
    assert resolve_placement(None) is None
    assert isinstance(resolve_placement("spread"), Spread)


@pytest.mark.parametrize("sched", _SCHEDULERS)
@pytest.mark.parametrize("mode", _ALL_MODES)
def test_every_policy_round_trips_through_config_on_every_scheduler(sched, mode):
    """``scheduler: {name: X, params: {placement: Y}}`` must work for all
    four built-ins and all four policies — including through the real
    ``load_scenario`` -> ``validate`` -> ``get_scheduler`` path."""
    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [2]},
                }
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
        "scheduler": {"name": sched, "params": {"placement": mode}},
    }
    scn = load_scenario(doc, strict=True)
    assert validate(scn) == []
    assert scheduler_placement_name(scn) == mode
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    assert isinstance(scheduler.placement, PLACEMENT_POLICIES[mode])
    assert scheduler.placement.search_mode == mode


def test_unknown_placement_name_is_a_validation_error():
    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [2]},
                }
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
        "scheduler": {"name": "fifo", "params": {"placement": "consolidated"}},
    }
    with pytest.raises(ScenarioError) as exc:
        load_scenario(doc, strict=True)
    msg = "\n".join(exc.value.errors)
    assert "scheduler.params.placement" in msg
    assert "consolidated" in msg
    for name in _ALL_MODES:
        assert name in msg


def test_reserved_level_names_match_the_metrics_layer():
    """``config`` spells the reserved level names literally to avoid
    importing metrics; the two lists must never drift."""
    from fleetsim.config import _RESERVED_LEVEL_NAMES
    from fleetsim.metrics.collector import FRAG_NON_LEVEL_KEYS

    assert _RESERVED_LEVEL_NAMES == FRAG_NON_LEVEL_KEYS


def test_reserved_level_name_is_rejected():
    """The summary's ``fragmentation`` map is keyed by level name and also
    carries ``stranded_whole_nodes``; a cluster may not shadow it."""
    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {
                        "levels": ["stranded_whole_nodes", "node"],
                        "counts": [2, 2],
                    },
                }
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
    }
    with pytest.raises(ScenarioError, match="reserved by the metrics layer"):
        load_scenario(doc, strict=True)


# ---------------------------------------------------------------------------
# 6. Policies inside the engine (relaxable constraints, end-to-end runs)
# ---------------------------------------------------------------------------


def _replay_doc(mode: str | None, nodes: int = 3):
    doc = {
        "sim": {"horizon": 7200, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [nodes]},
                }
            ],
        },
        "failure_model": {
            "node_mtbf_days": 0,
            "maintenance_rate_per_node_month": 0,
        },
        "workload": {"kind": "trace", "source": "__inline__"},
        "scheduler": {"name": "fifo", "params": {"strict": True}},
    }
    if mode is not None:
        doc["scheduler"]["params"]["placement"] = mode
    return doc


def _replay_jobs():
    """Six 1-chip jobs then one 2-node gang — the stranding shape.

    Six chips of demand on a 3-node / 24-chip pool.  Best-fit puts all six
    on one node and leaves two nodes whole, so the 16-chip gang starts at
    once; spread puts two on each node, leaving zero whole nodes, so the
    gang waits for the small jobs to finish.
    """
    jobs = []
    for i in range(6):
        jobs.append(
            TraceJob(
                id=f"small{i}",
                tenant="t",
                job_class=JobClass.FINETUNE,
                submit_t=i * 60 * 1_000_000,
                gangs=[GangSpec(chips=1, chip_type="h100")],
                tier=Tier.BATCH,
                min_runtime_s=0.0,
                walltime_est_s=1800.0,
                true_duration_s=1800.0,
                checkpoint_interval_s=0.0,
                checkpoint_save_s=0.0,
                restart_overhead_s=0.0,
            )
        )
    jobs.append(
        TraceJob(
            id="gang",
            tenant="t",
            job_class=JobClass.FINETUNE,
            submit_t=7 * 60 * 1_000_000,
            gangs=[GangSpec(chips=16, chip_type="h100")],
            tier=Tier.BATCH,
            min_runtime_s=0.0,
            walltime_est_s=600.0,
            true_duration_s=600.0,
            checkpoint_interval_s=0.0,
            checkpoint_save_s=0.0,
            restart_overhead_s=0.0,
        )
    )
    return jobs


def _redirty_jobs():
    """The Helios stranding shape, minimal: a released node that first-fit
    RE-DIRTIES and best-fit leaves whole.

    3 nodes x 8 chips.  ``whole`` takes node0 exclusively and finishes;
    ``resident`` (3 chips, long) sits on node1.  Then ``mouse`` (2 chips,
    long) arrives while node0 is pristine and node1 has 5 free:

    - ``first_fit`` takes node0 (lowest id) -> node0 AND node1 partial, so
      only node2 is whole and the 2-node ``gang`` behind it must wait for
      ``resident`` to finish;
    - ``best_fit``/``consolidate`` take node1 (tightest sufficient) -> node0
      and node2 stay whole and ``gang`` starts on arrival.

    This is the scenario the pre-existing engine test lacked: on a pool
    where the low-id nodes fill in id order, ``first_fit`` and ``best_fit``
    coincide exactly, so only ``spread`` separated from them.
    """
    def job(jid, chips, submit_s, dur_s):
        return TraceJob(
            id=jid,
            tenant="t",
            job_class=JobClass.FINETUNE,
            submit_t=int(submit_s * 1_000_000),
            gangs=[GangSpec(chips=chips, chip_type="h100")],
            tier=Tier.BATCH,
            min_runtime_s=0.0,
            walltime_est_s=float(dur_s),
            true_duration_s=float(dur_s),
            checkpoint_interval_s=0.0,
            checkpoint_save_s=0.0,
            restart_overhead_s=0.0,
        )

    return [
        job("a_whole", 8, 0, 600),
        job("b_resident", 3, 60, 3600),
        job("c_mouse", 2, 1200, 3600),
        job("d_gang", 16, 1800, 600),
    ]


def _run(doc, jobs=None):
    scn = load_scenario(doc, strict=True)
    fleet = build_fleet(scn)
    source = TraceSource(_replay_jobs() if jobs is None else jobs, fleet=fleet)
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    collector = MetricsCollector.from_scenario(scn, fleet)
    Simulator(scn, fleet, source, scheduler, collector).run()
    return collector


def test_placement_changes_the_simulated_outcome():
    """End to end through the real engine: with 6 single-chip jobs holding
    the pool, a strict-FIFO 2-node gang waits behind the stranding under
    ``spread`` and starts immediately under ``best_fit``."""
    starts = {}
    for mode in ("best_fit", "spread"):
        rows = {r["job_id"]: r for r in _run(_replay_doc(mode)).job_rows()}
        starts[mode] = rows["gang"]["queue_wait_s"]
    assert starts["best_fit"] < starts["spread"], starts
    assert starts["best_fit"] == 0.0


def _redirty_doc(mode):
    doc = _replay_doc(mode)
    doc["sim"]["horizon"] = 21_600
    return doc


@pytest.mark.parametrize("packed", ("best_fit", "consolidate"))
def test_first_fit_and_packed_diverge_end_to_end(packed):
    """The engine-level separation between ``first_fit`` and the packed
    policies (not merely between ``best_fit`` and ``spread``).

    Load-bearing: ``best_fit`` and ``consolidate`` carry the whole v0.7
    Helios result, and on the previous 6-mice scenario they were
    bit-identical to ``first_fit``, so nothing in CI would have noticed them
    being rewired to first-fit.  Here first-fit re-dirties a just-released
    node and the 2-node gang pays 1,860 s of queue wait for it.
    """
    waits = {}
    for mode in ("first_fit", packed):
        rows = {
            r["job_id"]: r
            for r in _run(_redirty_doc(mode), _redirty_jobs()).job_rows()
        }
        assert rows["d_gang"]["status"] == "COMPLETED", (mode, rows["d_gang"])
        waits[mode] = rows["d_gang"]["queue_wait_s"]
    assert waits[packed] == 0.0, waits
    assert waits["first_fit"] == pytest.approx(1860.0), waits
    assert waits["first_fit"] > waits[packed], waits


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_end_to_end_runs_are_deterministic(mode):
    a = _run(_replay_doc(mode))
    b = _run(_replay_doc(mode))
    assert build_summary(a) == build_summary(b)


class _FleetView:
    """Minimal ClusterView exposing only the raw search primitives — enough
    to drive a placement policy directly."""

    def __init__(self, fleet, now: int = 10_000_000):
        self._f = fleet
        self.now = now

    def search_first_fit(self, spec, tenant=None):
        return self._f.search_first_fit(spec, tenant)

    def search_segmented(self, spec, tenant=None):
        return self._f.search_segmented(spec, tenant)

    def search_best_fit(self, spec, tenant=None):
        return self._f.search_best_fit(spec, tenant)

    def search_consolidate(self, spec, tenant=None):
        return self._f.search_consolidate(spec, tenant)

    def search_spread(self, spec, tenant=None):
        return self._f.search_spread(spec, tenant)


def _job_view(**kw):
    from fleetsim.schedulers.base import JobView

    base = dict(
        id="j",
        submit_time=0,
        chips=16,
        chip_type="h100",
        tier=Tier.BATCH,
        job_class=JobClass.FINETUNE,
        preemptible=True,
        min_runtime_s=0.0,
        attained_service_chip_s=0.0,
        checkpoint_age_s=0.0,
        walltime_est_s=60.0,
        within=None,
        tenant="t",
    )
    base.update(kw)
    return JobView(**base)


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_relaxable_within_still_relaxes(mode):
    """v0.4's relax/penalty pair is policy-independent: a relaxable
    ``within`` that cannot be satisfied must retry unconstrained once
    ``relax_after_s`` has elapsed, and come back marked ``relaxed``.

    Setup: 3 racks x 2 nodes with EVERY node carrying a 1-chip resident, so
    no rack has a whole free node.  A 16-chip (2-node) gang constrained to
    one rack cannot place at all.  Freeing one node in rack0 and one in
    rack1 leaves each rack with a single whole free node — still no
    single-rack fit, but a cross-rack (relaxed) fit exists.
    """
    f = _fleet(["rack", "node"], [3, 2])
    for i, lid in enumerate(f.leaves()):
        _occupy(f, lid, 1, f"r{i}")
    policy = get_placement(mode)
    job = _job_view(within="rack", within_required=False, relax_after_s=0.0)

    assert policy.place(job, _FleetView(f)) is None  # nothing fits anywhere
    f.release("r0")  # rack0/node0 pristine
    f.release("r2")  # rack1/node0 pristine
    placed = policy.place(job, _FleetView(f))
    assert placed is not None and placed.relaxed is True
    assert len(placed.leaves) == 2
    # Two racks, one node each: the constrained search really did fail.
    assert len({lid.rsplit("/", 1)[0] for lid, _ in placed.leaves}) == 2

    # A REQUIRED constraint must never relax, under any policy.
    hard = _job_view(within="rack", within_required=True)
    assert policy.place(hard, _FleetView(f)) is None


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_hard_within_constraint_is_respected(mode):
    """A required ``within`` restricts the search to that level's domains
    for every policy (and the anchor is the satisfying domain)."""
    f = _fleet(["rack", "node"], [2, 2])
    _occupy(f, "m/c/rack0/node0", 1, "resident")
    policy = get_placement(mode)
    job = _job_view(chips=16, within="rack", within_required=True)
    placed = policy.place(job, _FleetView(f))
    assert placed is not None and placed.relaxed is False
    assert placed.anchor == "m/c/rack1"
    assert _ids(placed) == ("m/c/rack1/node0", "m/c/rack1/node1")


# ---------------------------------------------------------------------------
# 7. Metrics: stranded-whole-nodes + counts.placement_policy
# ---------------------------------------------------------------------------


def test_tree_stranded_whole_node_metrics():
    f = _fleet(["node"], [4])
    assert f.stranded_whole_nodes() == 0
    assert f.stranded_whole_node_chips() == 0
    _occupy(f, "m/c/node0", 3, "a")  # partial: 5 free chips stranded
    _take_whole(f, ["m/c/node1"], "b")  # full: not stranded
    assert f.stranded_whole_nodes() == 1
    assert f.stranded_whole_node_chips() == 5
    _occupy(f, "m/c/node2", 7, "c")
    assert f.stranded_whole_nodes() == 2
    assert f.stranded_whole_node_chips() == 6
    # A non-HEALTHY leaf contributes nothing (its free chips are already 0).
    f.drain_node("m/c/node3")
    assert f.stranded_whole_nodes() == 2
    assert f.stranded_whole_nodes("tpu_v5p") == 0


def test_placement_diagnostics_are_opt_in():
    """Naming a policy switches the diagnostics on; naming none keeps the
    exact pre-v0.7 timeseries and summary schema."""
    off = _run(_replay_doc(None))
    on = _run(_replay_doc("best_fit"))

    assert off.placement_policy is None
    assert on.placement_policy == "best_fit"

    off_cols = set(timeseries_dataframe(off).columns)
    on_cols = set(timeseries_dataframe(on).columns)
    assert "stranded_whole_nodes" not in off_cols
    assert on_cols - off_cols == {
        "stranded_whole_nodes",
        "stranded_whole_node_chips",
    }

    for scope in ("full", "window"):
        assert "placement_policy" not in build_summary(off)[scope]["counts"]
        assert build_summary(on)[scope]["counts"]["placement_policy"] == "best_fit"
        assert "stranded_whole_nodes" not in off.frag_stats()[scope]
    swn = on.frag_stats()["full"]["stranded_whole_nodes"]
    assert set(swn) == {"mean", "max", "n_samples"}
    assert swn["n_samples"] > 0
    assert swn["max"] >= swn["mean"] >= 0.0


def test_stranded_whole_nodes_metric_separates_the_policies():
    """The metric exists so a study can SEE the mechanism: spread strands
    more nodes than best-fit on the same workload."""
    means = {}
    for mode in ("best_fit", "spread"):
        stats = _run(_replay_doc(mode)).frag_stats()["full"]
        means[mode] = stats["stranded_whole_nodes"]["mean"]
    assert means["spread"] > means["best_fit"], means


# ---------------------------------------------------------------------------
# 8. Backward compatibility of the shipped examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ("01_minimal", "04_frontier"))
def test_examples_keep_the_default_first_fit_placement(name):
    """BACKWARD-COMPAT CONTRACT: the shipped examples name NO placement
    policy, so they run FirstFit and their outputs keep the pre-v0.7
    schema.  Asserted at the config/collector level for both examples;
    example 01's real output bytes are checked below (04 takes ~80 s to
    run, so it is pinned here rather than executed)."""
    scn = load_scenario(_EXAMPLES / name / "scenario.yaml")
    assert scheduler_placement_name(scn) is None
    assert "placement" not in scn.scheduler.params
    scheduler = get_scheduler(scn.scheduler.name, scn.scheduler.params)
    assert type(scheduler.placement) is FirstFit
    assert scheduler.placement.search_mode == "first_fit"
    fleet = build_fleet(scn)
    assert MetricsCollector.from_scenario(scn, fleet).placement_policy is None


def test_example_01_outputs_carry_no_v07_keys(tmp_path):
    """Run examples/01_minimal for real and assert the output SCHEMA is
    unchanged: no v0.7 timeseries columns, no ``counts.placement_policy``,
    no ``fragmentation['stranded_whole_nodes']``.  Together with the
    default-policy assertions above this is the byte-compat guard — the
    v0.7 additions are all gated on a named policy, so a scenario that
    names none cannot produce different bytes."""
    out = tmp_path / "ex01"
    summary = run_scenario(_EXAMPLES / "01_minimal" / "scenario.yaml", out_dir=out)
    for scope in ("full", "window"):
        assert "placement_policy" not in summary[scope]["counts"]
        assert "stranded_whole_nodes" not in summary[scope]["fragmentation"]
    written = json.loads((out / "summary.json").read_text())
    assert written == summary
    import pandas as pd

    cols = set(pd.read_parquet(out / "timeseries.parquet").columns)
    assert not {c for c in cols if c.startswith("stranded_whole_node_")}
    assert "stranded_chips" in cols  # the pre-v0.7 stranding column stays


def test_default_scheduler_placement_is_first_fit_for_every_builtin():
    for name in _SCHEDULERS:
        sched = get_scheduler(name, {})
        assert type(sched.placement) is FirstFit, name


def test_tiered_priority_reclaim_uses_its_own_placement_mode():
    """The latent v0.2 inconsistency closed: a preempting scheduler running
    a packed policy must plan reclaims under the SAME semantics.  With the
    default policy the dry-run call keeps its v0.2 two-argument form."""
    from fleetsim.schedulers.tiered_priority import TieredPriorityScheduler

    seen: list[dict] = []

    class _View:
        def reclaim_feasible(self, job, victim_ids, **kw):
            seen.append(dict(kw))
            return True

    job = object()
    victim = type("V", (), {"id": "v"})()
    TieredPriorityScheduler()._verify_and_refine(job, _View(), [victim], [])
    assert seen == [{}]  # default policy -> unchanged call shape

    seen.clear()
    TieredPriorityScheduler(placement=Consolidate())._verify_and_refine(
        job, _View(), [victim], []
    )
    assert seen == [{"mode": "consolidate"}]


def test_placement_policies_compare_and_hash_by_type():
    assert FirstFit() == FirstFit()
    assert BestFit() != FirstFit()
    assert len({FirstFit(), FirstFit(), BestFit()}) == 2


def test_jobs_dataframe_schema_is_unaffected():
    """Placement selection must not add or drop per-job columns."""
    off = set(jobs_dataframe(_run(_replay_doc(None))).columns)
    on = set(jobs_dataframe(_run(_replay_doc("consolidate"))).columns)
    assert off == on


def test_fragmentation_max_is_always_a_float():
    """SCHEMA: every ``max`` in ``fragmentation`` is a float, including
    ``stranded_whole_nodes`` — whose accumulator takes ``max()`` of a float
    and an INT sample, so it used to emit an int whenever the run maximum
    fell on the first flush.  A summary-schema validator or a golden-file
    diff would see the type flip between runs."""
    stats = _run(_replay_doc("spread")).frag_stats()
    for scope, table in stats.items():
        for key, acc in table.items():
            assert type(acc["max"]) is float, (scope, key, acc["max"])
            assert type(acc["mean"]) is float, (scope, key)
            assert type(acc["n_samples"]) is int, (scope, key)
    assert "stranded_whole_nodes" in stats["full"]


# ---------------------------------------------------------------------------
# 9. Protocol contracts and the plugin surface
# ---------------------------------------------------------------------------


def test_placement_policy_protocol_requires_search_mode():
    """``search_mode`` is a PROTOCOL MEMBER, not a comment-documented
    convention: a policy that omits it would place with one tree primitive
    while ``tiered_priority`` planned its evictions with another (§19.3's
    inconsistency).  Now the Protocol catches it."""
    from fleetsim.schedulers.base import PlacementPolicy

    for name in _ALL_MODES:
        assert isinstance(get_placement(name), PlacementPolicy), name

    class _NoMode:
        def place(self, job, view):  # pragma: no cover - never called
            return None

    class _WithMode(_NoMode):
        search_mode = "best_fit"

    assert not isinstance(_NoMode(), PlacementPolicy)
    assert isinstance(_WithMode(), PlacementPolicy)


def test_cluster_view_protocol_declares_reclaim_feasible_with_mode():
    """The engine view's ``reclaim_feasible`` accepts the keyword-only
    ``mode``, and the Protocol declares it — a custom view implementing only
    the pre-v0.7 two-argument form raises ``TypeError`` under a packed
    policy, which the docs must state rather than promise the opposite."""
    import inspect

    from fleetsim.engine.sim import _EngineView
    from fleetsim.schedulers.base import ClusterView

    for cls in (ClusterView, _EngineView):
        params = inspect.signature(cls.reclaim_feasible).parameters
        assert "mode" in params, cls
        assert params["mode"].kind is inspect.Parameter.KEYWORD_ONLY, cls
        assert params["mode"].default == "first_fit", cls


def _temp_scheduler(name, cls):
    """Register ``cls`` under ``name`` for one test, then remove it."""
    import contextlib

    from fleetsim.schedulers.base import _REGISTRY

    @contextlib.contextmanager
    def ctx():
        _REGISTRY[name] = cls
        try:
            yield
        finally:
            _REGISTRY.pop(name, None)

    return ctx()


def test_out_of_tree_scheduler_keeps_its_own_placement_vocabulary():
    """BACKWARD COMPAT of the plugin surface: ``placement`` is a convention,
    not a reserved word.

    A plugin whose ``placement`` param is its OWN vocabulary (no
    ``PlacementPolicy`` annotation) must still be constructible and still
    pass ``validate()`` — v0.6 passed such a string through untouched, and
    the built-ins documented ``placement`` as programmatic-only, so a YAML
    ``placement:`` string could only ever have belonged to a plugin.
    """
    from fleetsim.schedulers.base import Scheduler

    class RackAwareDemo(Scheduler):
        def __init__(self, placement: str = "lowest_rack"):
            self.placement = placement

        def schedule(self, view):  # pragma: no cover - never run
            return []

    doc = {
        "sim": {"horizon": 3600, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [2]},
                }
            ],
        },
        "workload": {"kind": "trace", "source": "__inline__"},
        "scheduler": {
            "name": "rack_aware_demo",
            "params": {"placement": "lowest_rack"},
        },
    }
    with _temp_scheduler("rack_aware_demo", RackAwareDemo):
        scn = load_scenario(doc, strict=True)
        assert validate(scn) == []  # not a closed-set violation
        sched = get_scheduler("rack_aware_demo", scn.scheduler.params)
        assert sched.placement == "lowest_rack"  # passed through, not resolved

    # ...while a scheduler that DOES opt in (annotates PlacementPolicy) still
    # gets the name resolved and an unknown name still rejected.
    from fleetsim.schedulers.placement import takes_placement_policy

    for name in _SCHEDULERS:
        cls = get_scheduler(name, {}).__class__
        assert takes_placement_policy(cls), name
    assert not takes_placement_policy(RackAwareDemo)
    with pytest.raises(ValueError, match="unknown placement policy"):
        get_scheduler("fifo", {"placement": "lowest_rack"})
