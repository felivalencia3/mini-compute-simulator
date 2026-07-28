"""v0.2 core-semantics tests: BEST_EFFORT tier rename, closed-loop
backlog refill, segmented multi-pod gangs, and reclaim at scale.

Conventions match test_sim_basic / test_tiered_priority: tiny fleets with
hierarchical ids (``m/c``, ``m/c/pod0/node0``...), 60 s rounds, failures
and maintenance OFF, strict engine mode, hand-computed expected times.
"""

import pytest

from fleetsim.config import (
    ClusterConfig,
    DatacenterConfig,
    FleetConfig,
    MetroConfig,
    NodeGroup,
    ScenarioError,
    load_scenario,
    parse_dist,
    validate,
)
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.model import (
    Allocation,
    ChipType,
    Constraint,
    GangAlloc,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    Tier,
)
from fleetsim.schedulers.base import Place, Scheduler
from fleetsim.schedulers.placement import FirstFit
from fleetsim.schedulers.tiered_priority import TieredPriorityScheduler
from fleetsim.workload.base import ListSource, MergedSource
from fleetsim.workload.synthetic import SyntheticSource, quantize_chips

S = 1_000_000  # one second in microseconds

CLUSTER = "m/c"


# ---------------------------------------------------------------------------
# Shared helpers (same shape as test_tiered_priority)
# ---------------------------------------------------------------------------


def make_scenario(
    levels,
    counts,
    per_node=8,
    horizon="30m",
    round_="60s",
    workload=None,
    seed=0,
):
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": round_, "seed": seed},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": per_node},
                        "topology": {"levels": levels, "counts": counts},
                    }
                ],
            },
            "failure_model": {
                "node_mtbf_days": 0.0,
                "maintenance_rate_per_node_month": 0.0,
            },
            "workload": workload
            or {
                "kind": "synthetic",
                "classes": {
                    "eval": {
                        "rate_per_hour": 1,
                        "chips": "pow2[1, 8]",
                        "duration": "lognormal[median=2m, p90=30m]",
                    }
                },
            },
        }
    )


def mk_job(
    jid,
    submit_s=0.0,
    chips=8,
    dur_s=100.0,
    tier=Tier.BATCH,
    interval=0.0,
    save=0.0,
    restart=0.0,
    within=None,
    segments=None,
):
    return Job(
        id=jid,
        tenant="t0",
        job_class=JobClass.EVAL,
        submit_t=int(round(submit_s * 1e6)),
        gangs=[
            GangSpec(
                chips=chips,
                chip_type="h100",
                within=Constraint(level=within) if within else None,
                segments=segments,
            )
        ],
        tier=tier,
        min_runtime_s=0.0,
        true_duration_s=dur_s,
        checkpoint_interval_s=interval,
        checkpoint_save_s=save,
        restart_overhead_s=restart,
    )


class RecSink:
    """Minimal recording sink (subset of test_tiered_priority's)."""

    def __init__(self):
        self.calls = []

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]

    def job_submitted(self, job, t):
        self.calls.append(("job_submitted", job.id, t))

    def job_admitted(self, job, t):
        self.calls.append(("job_admitted", job.id, t))

    def job_started(self, job, alloc, t):
        self.calls.append(
            (
                "job_started",
                job.id,
                t,
                alloc.gangs[0].anchor,
                alloc.gangs[0].attrs.get("n_domains_spanned", 1),
            )
        )

    def job_preempted(self, job, t, trigger):
        self.calls.append(("job_preempted", job.id, t, trigger))

    def job_requeued(self, job, t):
        self.calls.append(("job_requeued", job.id, t))

    def job_progress(self, job, start_us, end_us, productive_chip_s, lost_chip_s):
        self.calls.append(
            ("job_progress", job.id, start_us, end_us,
             round(productive_chip_s, 6), round(lost_chip_s, 6))
        )

    def job_finished(self, job, t, status, productive_chip_s, lost_chip_s):
        self.calls.append(
            ("job_finished", job.id, t, status.name,
             round(productive_chip_s, 6), round(lost_chip_s, 6))
        )

    def node_failed(self, node_id, t, killed_alloc_ids, cause="unknown"):
        self.calls.append(("node_failed", node_id, t, tuple(killed_alloc_ids)))

    def node_repaired(self, node_id, t):
        self.calls.append(("node_repaired", node_id, t))

    def node_drain_started(self, node_id, t):
        self.calls.append(("node_drain_started", node_id, t))

    def chips_allocated(self, n, chip_type, t):
        self.calls.append(("chips_allocated", n, chip_type, t))

    def chips_freed(self, n, chip_type, t):
        self.calls.append(("chips_freed", n, chip_type, t))

    def healthy_delta(self, n_chips, chip_type, t):
        self.calls.append(("healthy_delta", n_chips, chip_type, t))

    def flush(self, t, fleet, n_pending, n_running):
        self.calls.append(("flush", t, n_pending, n_running))


class RecordingFIFO(Scheduler):
    """Best-effort FIFO that records the pending count seen at each wake
    (AFTER the engine's refill hook ran)."""

    def __init__(self):
        self.placement = FirstFit()
        self.pending_seen: list[tuple[int, int]] = []

    def schedule(self, view):
        self.pending_seen.append((view.now, len(view.pending())))
        actions = []
        for job in view.pending():
            p = view.find_placement(job, self.placement)
            if p is not None:
                actions.append(Place(job.id, p))
        return actions


def run_sim(scenario, jobs, scheduler=None, sink=None, pre_events=()):
    fleet = build_fleet(scenario)
    sink = sink if sink is not None else RecSink()
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        scheduler if scheduler is not None else TieredPriorityScheduler(),
        sink,
        strict=True,
    )
    for t, etype, payload in pre_events:
        sim.queue.push(t, etype, payload)
    sim.run()
    fleet.check_invariants()
    return sim, fleet, sink


# ---------------------------------------------------------------------------
# 1. Tier rename: BEST_EFFORT canonical, FREE alias
# ---------------------------------------------------------------------------


class TestTierRename:
    def test_alias_identity(self):
        assert Tier.BEST_EFFORT == 0
        assert Tier.FREE is Tier.BEST_EFFORT
        assert Tier(0).name == "BEST_EFFORT"
        assert Tier["FREE"] is Tier.BEST_EFFORT
        assert Tier.BEST_EFFORT < Tier.BATCH < Tier.PROD < Tier.MONITORING
        # FREE is an alias, not a member: exactly four canonical bands.
        assert [m.name for m in Tier] == [
            "BEST_EFFORT",
            "BATCH",
            "PROD",
            "MONITORING",
        ]

    @pytest.mark.parametrize("spelling", ["best_effort", "free"])
    def test_config_accepts_both_spellings(self, spelling):
        scn = make_scenario(
            ["node"],
            [2],
            workload={
                "kind": "synthetic",
                "classes": {
                    "filler": {
                        "class": "eval",
                        "rate_per_hour": 1,
                        "chips": 1,
                        "duration": "2m",
                        "tier": spelling,
                    }
                },
            },
        )
        assert scn.workload.classes[0].tier is Tier.BEST_EFFORT


# ---------------------------------------------------------------------------
# 2. Closed-loop backlog: config surface
# ---------------------------------------------------------------------------


BACKLOG_CLASS = {
    "class": "eval",
    "arrival": "backlog[target_pending=3]",
    "chips": 8,
    "duration": "2m",
    "checkpoint_interval": 0,
    "abort_prob": 0,
}


def backlog_scenario(horizon="330s", seed=0, target=3, **extra):
    cls = dict(BACKLOG_CLASS)
    cls["arrival"] = f"backlog[target_pending={target}]"
    cls.update(extra)
    return make_scenario(
        ["node"],
        [2],
        horizon=horizon,
        seed=seed,
        workload={"kind": "synthetic", "classes": {"bemice": cls}},
    )


class TestBacklogConfig:
    def test_parse_dist_backlog_positional(self):
        assert parse_dist("backlog[4]").params == {"target_pending": 4}
        assert parse_dist("backlog[target_pending=4]").kind == "backlog"

    def test_backlog_class_defaults(self):
        scn = backlog_scenario()
        c = scn.workload.classes[0]
        assert c.arrival is not None and c.arrival.kind == "backlog"
        assert c.tier is Tier.BEST_EFFORT
        assert c.min_runtime_s == 0.0
        assert c.rate_per_hour == 0.0

    def test_backlog_rejects_rate_keys(self):
        with pytest.raises(ScenarioError, match="cannot be combined"):
            backlog_scenario(rate_per_hour=5)

    def test_backlog_rejects_diurnal(self):
        with pytest.raises(ScenarioError, match="diurnal"):
            backlog_scenario(diurnal=True)

    def test_backlog_requires_positive_target(self):
        with pytest.raises(ScenarioError, match="target_pending"):
            backlog_scenario(target=0)

    def test_backlog_invalid_as_chips_distribution(self):
        with pytest.raises(ScenarioError, match="arrival"):
            make_scenario(
                ["node"],
                [2],
                workload={
                    "kind": "synthetic",
                    "classes": {
                        "bad": {
                            "rate_per_hour": 1,
                            "chips": "backlog[target_pending=2]",
                            "duration": "2m",
                        }
                    },
                },
            )

    def test_rate_still_required_without_arrival(self):
        with pytest.raises(ScenarioError, match="exactly one of"):
            make_scenario(
                ["node"],
                [2],
                workload={
                    "kind": "synthetic",
                    "classes": {"bad": {"chips": 8, "duration": "2m"}},
                },
            )


# ---------------------------------------------------------------------------
# 2b. Closed-loop backlog: engine refill behavior
# ---------------------------------------------------------------------------


def run_backlog(seed=0, horizon="330s", target=3):
    scn = backlog_scenario(horizon=horizon, seed=seed, target=target)
    fleet = build_fleet(scn)
    rng = RngStreams(scn.sim.seed)
    source = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
    sched = RecordingFIFO()
    sink = RecSink()
    sim = Simulator(scn, fleet, source, sched, sink, rng=rng)
    sim.run()
    fleet.check_invariants()
    return sim, sched, sink


class TestBacklogRefill:
    def test_pending_held_at_target_under_churn(self):
        # 2 nodes x 8; backlog target 3 of 8-chip 120 s jobs.  Every wake
        # sees EXACTLY 3 pending after refill: placements drain the queue
        # and completions churn capacity, but the hook tops it back up
        # before the scheduler runs.
        sim, sched, sink = run_backlog()
        assert sched.pending_seen  # wakes at 0, 60, ..., 300
        assert [t for t, _ in sched.pending_seen] == [
            i * 60 * S for i in range(6)
        ]
        assert all(n == 3 for _, n in sched.pending_seen)
        # Churn actually happened: more jobs were submitted than the
        # target, and some ran to completion.
        assert len(sink.of("job_submitted")) > 3
        assert len(sink.of("job_finished")) >= 2
        # Refilled jobs went through the normal admission path.
        assert len(sink.of("job_admitted")) == len(sink.of("job_submitted"))
        # Jobs carry the class label for pending_by_class keying.
        assert sim._jobs["bemice-0"].job.source_class == "bemice"
        assert sim._jobs["bemice-0"].job.tier is Tier.BEST_EFFORT

    def test_refill_is_deterministic(self):
        _, sched_a, sink_a = run_backlog(seed=7)
        _, sched_b, sink_b = run_backlog(seed=7)
        assert sched_a.pending_seen == sched_b.pending_seen
        assert sink_a.calls == sink_b.calls

    def test_refill_submit_times_equal_wake_times(self):
        _, _, sink = run_backlog()
        for _, jid, t in sink.of("job_submitted"):
            assert t % (60 * S) == 0  # submitted at wake boundaries only


class TestMergedSource:
    def test_merges_streams_in_time_order(self):
        a = [mk_job("a0", submit_s=0), mk_job("a1", submit_s=100)]
        b = [mk_job("b0", submit_s=50)]
        merged = MergedSource([ListSource(a), ListSource(b)])
        order = []
        while (nxt := merged.next_arrival()) is not None:
            order.append(nxt[1].id)
        assert order == ["a0", "b0", "a1"]
        assert merged.next_arrival() is None

    def test_tie_breaks_by_child_position(self):
        a = [mk_job("x", submit_s=10)]
        b = [mk_job("w", submit_s=10)]
        merged = MergedSource([ListSource(a), ListSource(b)])
        assert merged.next_arrival()[1].id == "x"  # first child wins ties
        assert merged.next_arrival()[1].id == "w"

    def test_refill_concatenates_children_in_order(self):
        class FakeBacklog:
            def __init__(self, jid):
                self.jid = jid

            def next_arrival(self):
                return None

            def refill(self, now_us, pending_by_class):
                job = mk_job(self.jid, submit_s=now_us / 1e6)
                return [job]

        merged = MergedSource(
            [ListSource([]), FakeBacklog("r1"), FakeBacklog("r2")]
        )
        jobs = merged.refill(60 * S, {})
        assert [j.id for j in jobs] == ["r1", "r2"]


# ---------------------------------------------------------------------------
# 3. Segmented gangs: config surface
# ---------------------------------------------------------------------------


def segmented_workload(**extra):
    cls = {
        "class": "pretrain",
        "rate_per_hour": 1,
        "chips": 64,
        "duration": "10m",
        "tier": "prod",
        "within": "cluster",
        "segment_nodes": 2,
        "segment_level": "pod",
        "min_runtime": 0,
        "abort_prob": 0,
        "checkpoint_interval": 0,
    }
    cls.update(extra)
    return {"kind": "synthetic", "classes": {"seg": cls}}


class TestSegmentConfig:
    def test_valid_segmented_class_loads(self):
        scn = make_scenario(["pod", "node"], [2, 4], workload=segmented_workload())
        c = scn.workload.classes[0]
        assert c.segment_nodes == 2
        assert c.segment_level == "pod"

    def test_segment_keys_are_both_or_neither(self):
        doc = segmented_workload()
        del doc["classes"]["seg"]["segment_level"]
        with pytest.raises(ScenarioError, match="together"):
            make_scenario(["pod", "node"], [2, 4], workload=doc)

    def test_unknown_segment_level_rejected(self):
        with pytest.raises(ScenarioError, match="segment_level"):
            make_scenario(
                ["pod", "node"], [2, 4],
                workload=segmented_workload(segment_level="rack"),
            )

    def test_within_must_be_above_segment_level(self):
        with pytest.raises(ScenarioError, match="strictly above"):
            make_scenario(
                ["pod", "node"], [2, 4],
                workload=segmented_workload(within="node", segment_level="pod"),
            )

    def test_segment_nodes_must_be_positive(self):
        with pytest.raises(ScenarioError, match="segment_nodes"):
            make_scenario(
                ["pod", "node"], [2, 4],
                workload=segmented_workload(segment_nodes=0),
            )


class TestSegmentQuantization:
    def test_quantize_rounds_up_to_segment_quantum(self):
        assert quantize_chips(50, 8, segment_nodes=2) == 64
        assert quantize_chips(16, 8, segment_nodes=2) == 16
        assert quantize_chips(1, 8, segment_nodes=4) == 32
        assert quantize_chips(3, 8) == 4  # segment_nodes=1 keeps v0.1 rules

    def test_synthetic_source_emits_segmented_specs(self):
        scn = make_scenario(
            ["pod", "node"], [2, 4],
            workload=segmented_workload(
                chips="pow2[8, 64]", rate_per_hour=120
            ),
        )
        fleet = build_fleet(scn)
        rng = RngStreams(3)
        src = SyntheticSource(scn.workload, fleet, rng, scn.sim.horizon_us)
        seen = 0
        while (nxt := src.next_arrival()) is not None and seen < 5:
            spec = nxt[1].gangs[0]
            assert spec.segments == (2, "pod")
            assert spec.chips % 16 == 0  # node_size 8 x segment_nodes 2
            seen += 1
        assert seen > 0


# ---------------------------------------------------------------------------
# 3b. Segmented placement: FleetTree.search_segmented
# ---------------------------------------------------------------------------


def two_pod_tree(pods=2, nodes=4, per_node=8):
    scn = make_scenario(["pod", "node"], [pods, nodes], per_node=per_node)
    return build_fleet(scn)


def seg_spec(chips, nodes_per_seg=2, level="pod", within="cluster"):
    return GangSpec(
        chips=chips,
        chip_type="h100",
        within=Constraint(level=within) if within else None,
        segments=(nodes_per_seg, level),
    )


class TestSearchSegmented:
    def test_spans_pods_and_anchors_at_lca(self):
        # 2 pods x 4 nodes x 8.  64 chips = 8 nodes = 4 segments of 2:
        # pod0 hosts 2 segments (4 free nodes), pod1 the other 2.
        tree = two_pod_tree()
        p = tree.search_segmented(seg_spec(64))
        assert p is not None
        assert p.chips == 64 and p.whole_node
        assert p.segment_domains == (
            "m/c/pod0", "m/c/pod0", "m/c/pod1", "m/c/pod1",
        )
        assert p.n_domains_spanned == 2
        assert p.anchor == CLUSTER  # LCA of the two pods
        # search_first_fit delegates for segmented specs.
        assert tree.search_first_fit(seg_spec(64)) == p

    def test_multiple_segments_share_one_domain(self):
        # 32 chips = 2 segments; both fit in pod0 -> pod1 stays empty.
        tree = two_pod_tree()
        p = tree.search_segmented(seg_spec(32))
        assert p.segment_domains == ("m/c/pod0", "m/c/pod0")
        assert p.n_domains_spanned == 1
        assert p.anchor == "m/c/pod0"  # LCA of one pod is itself

    def test_bin_pack_descending_free_capacity(self):
        # Take one node in pod0 -> pod0 free 3, pod1 free 4.  A 2-segment
        # job packs into pod1 alone (most-free domain first), preserving
        # pod0's remaining nodes for smaller work.
        tree = two_pod_tree()
        tree.apply(
            Allocation(
                "filler", [GangAlloc(nodes=["m/c/pod0/node0"], anchor="m/c/pod0")]
            )
        )
        p = tree.search_segmented(seg_spec(32))
        assert p.segment_domains == ("m/c/pod1", "m/c/pod1")
        tree.check_invariants()

    def test_atomic_all_or_nothing(self):
        # One node taken in EACH pod: 6 free nodes, 48 free chips — but a
        # single 4-node segment fits in neither pod (3 free each) -> None,
        # and nothing was mutated.
        tree = two_pod_tree()
        for i, pod in enumerate(("m/c/pod0", "m/c/pod1")):
            tree.apply(
                Allocation(
                    f"filler{i}",
                    [GangAlloc(nodes=[f"{pod}/node0"], anchor=pod)],
                )
            )
        free_before = tree.free_chips(CLUSTER)
        assert free_before == 48
        assert tree.free_full_nodes(CLUSTER, "h100", 8) == 6
        assert tree.search_segmented(seg_spec(32, nodes_per_seg=4)) is None
        assert tree.free_chips(CLUSTER) == free_before
        tree.check_invariants()

    def test_sub_node_owner_blocks_free_node(self):
        # A 1-chip sub-node job on a leaf removes it from the free-node
        # pool even though 7 chips stay free.
        tree = two_pod_tree()
        sub = tree.search_first_fit(GangSpec(1, "h100"))
        tree.apply(Allocation("tiny", [sub.to_gang_alloc()]))
        assert tree.free_full_nodes(CLUSTER, "h100", 8) == 7
        p = tree.search_segmented(seg_spec(64))  # needs all 8 nodes
        assert p is None
        tree.check_invariants()

    def test_apply_release_roundtrip_and_counters(self):
        tree = two_pod_tree()
        p = tree.search_segmented(seg_spec(64))
        tree.apply(Allocation("big", [p.to_gang_alloc()]))
        assert tree.free_chips(CLUSTER) == 0
        assert tree.free_full_nodes(CLUSTER, "h100", 8) == 0
        tree.check_invariants()
        tree.release("big")
        assert tree.free_chips(CLUSTER) == 64
        assert tree.free_full_nodes(CLUSTER, "h100", 8) == 8
        tree.check_invariants()

    def test_requires_segments(self):
        tree = two_pod_tree()
        with pytest.raises(ValueError, match="segments"):
            tree.search_segmented(GangSpec(8, "h100"))


# ---------------------------------------------------------------------------
# 3c. Segmented gangs through the engine
# ---------------------------------------------------------------------------


class TestSegmentedEngine:
    def test_end_to_end_placement_and_n_domains_spanned(self):
        # 2 pods x 2 nodes x 8 = 32 chips.  A 32-chip job in segments of
        # 2 nodes at pod level spans both pods.
        scn = make_scenario(["pod", "node"], [2, 2])
        fleet = build_fleet(scn)
        collector = MetricsCollector(scn.sim.horizon_us, fleet=fleet)
        job = mk_job(
            "seg", chips=32, dur_s=100.0, tier=Tier.PROD,
            within="cluster", segments=(2, "pod"),
        )
        sim = Simulator(
            scn, fleet, ListSource([job]), TieredPriorityScheduler(), collector
        )
        sim.run()
        fleet.check_invariants()
        rows = {r["job_id"]: r for r in collector.job_rows()}
        assert rows["seg"]["status"] == "COMPLETED"
        assert rows["seg"]["n_domains_spanned"] == 2
        # The jobs.parquet schema carries the column (nullable Int64).
        from fleetsim.metrics.summary import jobs_dataframe

        df = jobs_dataframe(collector)
        assert "n_domains_spanned" in df.columns
        assert str(df["n_domains_spanned"].dtype) == "Int64"
        assert int(df.set_index("job_id").loc["seg", "n_domains_spanned"]) == 2

    def test_malformed_segmented_spec_fails_job_not_run(self):
        # A spec violation is a PER-JOB terminal FAILED, never an uncaught
        # ValueError out of Simulator.run() (review fix): 12 chips is not
        # a whole-node multiple (node size 8).
        scn = make_scenario(["pod", "node"], [2, 2])
        job = mk_job("bad", chips=12, segments=(2, "pod"), within="cluster")
        sim, fleet, sink = run_sim(scn, [job])
        assert sim._jobs["bad"].job.status is JobStatus.FAILED
        finished = sink.of("job_finished")
        assert finished and finished[0][1] == "bad" and finished[0][3] == "FAILED"
        assert fleet.free_chips(CLUSTER) == 32  # nothing was allocated

    def test_indivisible_node_count_fails_job_not_run(self):
        # 24 chips = 3 nodes, not divisible into 2-node segments -> the
        # job fails terminally at submission; the run continues.
        scn = make_scenario(["pod", "node"], [2, 2])
        bad = mk_job("bad", chips=24, segments=(2, "pod"), within="cluster")
        ok = mk_job("ok", submit_s=1.0, chips=8, dur_s=60.0)
        sim, fleet, sink = run_sim(scn, [bad, ok])
        assert sim._jobs["bad"].job.status is JobStatus.FAILED
        assert sim._jobs["ok"].job.status is JobStatus.COMPLETED

    def test_mixed_node_sizes_accept_any_decomposable_size(self):
        # Two clusters of the same chip type with per_node 8 and 16: a
        # 16-chip job in 2-node pod segments decomposes on the 8-chip
        # leaves (2 nodes) even though the MAX leaf size (16) does not
        # divide it into 2-node segments — it must submit, place, and
        # complete instead of crashing the run (review fix).
        scn = load_scenario(
            {
                "sim": {"horizon": "10m", "round": "60s", "seed": 0},
                "fleet": {
                    "metro": "m",
                    "clusters": [
                        {
                            "name": "small",
                            "chip": {"type": "h100", "per_node": 8},
                            "topology": {"levels": ["pod", "node"], "counts": [2, 2]},
                        },
                        {
                            "name": "big",
                            "chip": {"type": "h100", "per_node": 16},
                            "topology": {"levels": ["pod", "node"], "counts": [2, 2]},
                        },
                    ],
                },
                "failure_model": {
                    "node_mtbf_days": 0.0,
                    "maintenance_rate_per_node_month": 0.0,
                },
                "workload": {
                    "kind": "synthetic",
                    "classes": {
                        "eval": {
                            "rate_per_hour": 1,
                            "chips": "pow2[1, 8]",
                            "duration": "lognormal[median=2m, p90=30m]",
                        }
                    },
                },
            }
        )
        job = mk_job(
            "seg16", chips=16, dur_s=60.0, tier=Tier.PROD,
            segments=(2, "pod"),
        )
        sim, fleet, sink = run_sim(scn, [job])
        assert sim._jobs["seg16"].n_segments == 1  # 2 x 8-chip nodes
        assert sim._jobs["seg16"].job.status is JobStatus.COMPLETED

    def test_one_node_death_kills_whole_segmented_job(self):
        # Failure semantics unchanged (DESIGN §4.2): a segmented job
        # spans both pods; ONE node failing kills the whole job and
        # releases every leaf across all segments.
        from fleetsim.engine.events import EventType

        scn = make_scenario(["pod", "node"], [2, 2], horizon="10m")
        job = mk_job(
            "seg", chips=32, dur_s=500.0, tier=Tier.PROD,
            within="cluster", segments=(2, "pod"),
        )
        sim, fleet, sink = run_sim(
            scn,
            [job],
            pre_events=[(100 * S, EventType.NODE_FAILURE, "m/c/pod1/node0")],
        )
        assert sink.of("node_failed") == [
            ("node_failed", "m/c/pod1/node0", 100 * S, ("seg",))
        ]
        assert sink.of("job_requeued") == [("job_requeued", "seg", 100 * S)]
        assert sink.of("chips_freed") == [("chips_freed", 32, "h100", 100 * S)]
        # Healthy nodes returned to the free pool; the dead one did not.
        assert fleet.free_full_nodes(CLUSTER, "h100", 8) == 3


# ---------------------------------------------------------------------------
# 4. Reclaim at scale: segmented preemption empties multiple pods
# ---------------------------------------------------------------------------


class TestSegmentedPreemption:
    def test_empties_two_pods_of_best_effort_mice(self):
        # 2 pods x 2 nodes x 8.  Four BEST_EFFORT whole-node mice (no
        # checkpointing, save=0 -> grace 0, cheap kills) fill the fleet
        # at wake 0.  PROD "big" (32 chips, 2-node pod segments) arrives
        # at 90; wake 120 preempts ALL FOUR mice across both pods in one
        # wake; grace 0 frees the chips at 120; wake 180 places big
        # spanning both pods.
        scn = make_scenario(["pod", "node"], [2, 2])
        mice = [
            mk_job(f"m{i}", submit_s=0, chips=8, dur_s=10_000.0,
                   tier=Tier.BEST_EFFORT)
            for i in range(4)
        ]
        big = mk_job(
            "big", submit_s=90, chips=32, dur_s=100.0, tier=Tier.PROD,
            within="cluster", segments=(2, "pod"),
        )
        sim, fleet, sink = run_sim(scn, mice + [big])
        assert [(c[1], c[2]) for c in sink.of("job_preempted")] == [
            ("m0", 120 * S),
            ("m1", 120 * S),
            ("m2", 120 * S),
            ("m3", 120 * S),
        ]
        # Cheap kill: grace 0 -> requeue at the SAME timestamp, kept
        # work 0 (no checkpoints), all 120 s x 8 chips lost per mouse.
        assert sink.of("job_requeued") == [
            ("job_requeued", f"m{i}", 120 * S) for i in range(4)
        ]
        for c in sink.of("job_progress")[:4]:
            assert c[4] == 0.0 and c[5] == 960.0
        starts = [(c[1], c[2], c[3], c[4]) for c in sink.of("job_started")]
        assert ("big", 180 * S, CLUSTER, 2) in starts  # spans 2 pods
        assert sim._jobs["big"].job.status is JobStatus.COMPLETED
        # Mice restart after big completes (280 -> wake 300); their gangs
        # are unconstrained, so anchors are the cluster root.
        assert [s for s in starts if s[1] == 300 * S] == [
            (f"m{i}", 300 * S, CLUSTER, 1) for i in range(4)
        ]

    def test_prefers_domains_needing_fewest_preemptions(self):
        # pod0 holds ONE mouse (1 node busy, 1 free); pod1 holds TWO.
        # A 2-segment job (1 node per segment) needs pod0's free node
        # (0 preemptions) plus ONE eviction in pod0?  No: segment 1 goes
        # to pod0 free (cost 0), segment 2 cheapest is one eviction —
        # pod0 (1 candidate) ties pod1 (1 needed of 2) -> pod0 wins by id.
        scn = make_scenario(["pod", "node"], [2, 2])
        mice = [
            mk_job("a", submit_s=0, chips=8, dur_s=10_000.0,
                   tier=Tier.BEST_EFFORT),  # -> pod0/node0
            mk_job("b", submit_s=0, chips=8, dur_s=10_000.0,
                   tier=Tier.BEST_EFFORT),  # -> pod0/node1
            mk_job("c", submit_s=0, chips=8, dur_s=10_000.0,
                   tier=Tier.BEST_EFFORT),  # -> pod1/node0
        ]
        big = mk_job(
            "big", submit_s=90, chips=16, dur_s=100.0, tier=Tier.PROD,
            within="cluster", segments=(1, "pod"),
        )
        sim, fleet, sink = run_sim(scn, mice + [big])
        # pod1 has the free node (cost 0 segment); the second segment
        # needs one eviction; pod0's youngest/least-served candidate is
        # chosen deterministically.
        pre = [(c[1], c[2]) for c in sink.of("job_preempted")]
        assert len(pre) == 1 and pre[0][1] == 120 * S
        assert sim._jobs["big"].job.status is JobStatus.COMPLETED

    def test_no_preemption_when_free_space_suffices(self):
        # Both pods empty: a segmented job just places; no Preempts.
        scn = make_scenario(["pod", "node"], [2, 2])
        big = mk_job(
            "big", submit_s=0, chips=32, dur_s=100.0, tier=Tier.PROD,
            within="cluster", segments=(2, "pod"),
        )
        sim, fleet, sink = run_sim(scn, [big])
        assert sink.of("job_preempted") == []
        assert sim._jobs["big"].job.status is JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# 5. Perf smoke: 500K-chip fleet, 128K-chip segmented job across pods
# ---------------------------------------------------------------------------


def build_500k_fleet():
    """4 clusters x 32 pods x 64 racks x 8 nodes x 8 chips = 524,288."""
    clusters = [
        ClusterConfig(
            id=f"c{i}",
            levels=["cluster", "pod", "rack", "node"],
            children=[
                NodeGroup(
                    level="pod",
                    count=32,
                    children=[
                        NodeGroup(
                            level="rack",
                            count=64,
                            children=[
                                NodeGroup(
                                    level="node",
                                    count=8,
                                    chips=8,
                                    chip_type="h100",
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        for i in range(4)
    ]
    cfg = FleetConfig(
        chip_types={"h100": ChipType("h100", "nvidia", 80.0, 989.0)},
        metros=[
            MetroConfig(
                name="m",
                datacenters=[DatacenterConfig(id="dc0", clusters=clusters)],
            )
        ],
    )
    return build_fleet(cfg)


class TestScaleSmoke:
    def test_128k_chip_job_across_pods_on_500k_fleet(self):
        # Correctness at scale (NOT perf-gated in CI): the segmented
        # search must run on domain counters and complete; the placement
        # must be exactly one cluster's 32 pods, one 512-node segment
        # each.
        tree = build_500k_fleet()
        assert tree.total_chips("m") == 524_288
        spec = GangSpec(
            chips=131_072,  # 16,384 nodes = 32 segments of 512
            chip_type="h100",
            within=Constraint("cluster"),
            segments=(512, "pod"),
        )
        p = tree.search_segmented(spec)
        assert p is not None
        assert p.chips == 131_072
        assert len(p.leaves) == 16_384
        assert p.n_domains_spanned == 32
        assert p.anchor == "m/c0"
        assert all(sd.startswith("m/c0/pod") for sd in p.segment_domains)
        tree.apply(Allocation("big", [p.to_gang_alloc()]))
        assert tree.free_chips("m/c0") == 0
        assert tree.free_chips("m") == 524_288 - 131_072
        # A second identical job lands on the next cluster, atomically.
        p2 = tree.search_segmented(spec)
        assert p2 is not None and p2.anchor == "m/c1"
        # An impossible job (more segments than any cluster's pods under
        # the outer constraint) fails atomically without mutation.
        free_before = tree.free_chips("m")
        too_big = GangSpec(
            chips=163_840,  # 40 segments of 512 nodes > 32 pods
            chip_type="h100",
            within=Constraint("cluster"),
            segments=(512, "pod"),
        )
        assert tree.search_segmented(too_big) is None
        assert tree.free_chips("m") == free_before
