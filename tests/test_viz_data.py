"""Tests for the v0.3 visualizer data pipeline (fleetsim.viz.data).

Coverage per the phase contract: exact schema keys, frontier-size
budget, no-stints degradation, compare mode, determinism, NaN-freedom —
plus the reconstruction rules (fleet map chips, frames downsampling,
frontier events, gantt selection, palette consistency).

Fixtures: one tiny 10-minute scenario run per variant (with stints,
without stints, different seed) — real engine outputs, seconds to
produce.  The frontier-scale budget check on examples/04_frontier takes
~80 s of simulation, so it only runs when FLEETSIM_FRONTIER_BUDGET=1 is
set (CI smoke stays fast); the measured size at the time of writing is
~4.7 MB against the 15 MB budget.
"""

import json
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fleetsim.api import run_scenario
from fleetsim.fleet.tree import FleetTree
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.metrics.summary import write_outputs
from fleetsim.model import (
    Allocation,
    Domain,
    GangAlloc,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    Tier,
)
from fleetsim.viz import build_viz_model, to_json

S = 1_000_000

# ---------------------------------------------------------------------------
# Fixtures: tiny real-engine runs
# ---------------------------------------------------------------------------


def _doc(stints="pod", seed=0):
    return {
        "sim": {"horizon": "10m", "round": "60s", "seed": seed},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["pod", "node"], "counts": [2, 2]},
                }
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
                    "rate_per_hour": 60,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=2m, p90=5m]",
                    "abort_prob": 0,
                }
            },
        },
        **({"outputs": {"stints": stints}} if stints is not None else {}),
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("viz_runs")
    run_scenario(_doc("pod"), out_dir=base / "a")
    run_scenario(_doc(None), out_dir=base / "nostints")
    run_scenario(_doc("pod", seed=1), out_dir=base / "b")
    return base


@pytest.fixture(scope="module")
def model(runs):
    return build_viz_model(runs / "a")


# ---------------------------------------------------------------------------
# Schema: exact keys at every pinned layer
# ---------------------------------------------------------------------------


def test_top_level_schema_keys_exact(model):
    assert list(model) == [
        "meta",
        "capabilities",
        "palette",
        "fleet",
        "frames",
        "stints",
        "gantt",
        "cdfs",
        "events",
        "summary_cards",
        "compare",
    ]
    assert list(model["meta"]) == [
        "title",
        "out_dir",
        "horizon_us",
        "round_us",
        "seed",
        "scenario_name",
        "fleetsim_version",
        "generated_unix_ms",
        "notes",
    ]
    assert list(model["capabilities"]) == ["map", "compare"]
    assert list(model["fleet"]) == ["map_level", "clusters"]
    assert list(model["frames"]) == [
        "t_us",
        "occupancy",
        "allocation",
        "goodput_to_date",
        "pending_by_class",
        "preemptions_delta",
        "failures_delta",
    ]
    assert list(model["stints"]) == [
        "job_id",
        "class_name",
        "tier",
        "domain_idx",
        "chips",
        "t0_us",
        "t1_us",
        "end_reason",
    ]
    assert list(model["cdfs"]) == ["queue_wait_s", "jct_s"]
    for entry in model["gantt"]:
        assert list(entry) == [
            "id",
            "class_name",
            "chips",
            "submit_us",
            "start_us",
            "end_us",
            "status",
            "n_preemptions",
            "n_restarts",
            "domains_spanned",
        ]
    for ev in model["events"]:
        assert list(ev) == ["t_us", "kind", "label", "magnitude"]
        assert ev["kind"] in {
            "preemption_wave",
            "failure",
            "frontier_start",
            "frontier_submit",
        }
    for card in model["summary_cards"]:
        assert list(card) == ["label", "value", "sub"]
    assert model["compare"] is None
    assert model["meta"]["generated_unix_ms"] is None
    for cluster in model["fleet"]["clusters"]:
        assert list(cluster) == ["id", "chips", "domains"]
        for dom in cluster["domains"]:
            assert list(dom) == ["id", "short", "chips"]


def test_meta_values_traceable(runs, model):
    summary = json.loads((runs / "a" / "summary.json").read_text())
    assert model["meta"]["horizon_us"] == summary["horizon_us"] == 600 * S
    # No scenario copy in the out dir: round from the modal sample gap,
    # seed/scenario_name unknown.
    assert model["meta"]["round_us"] == 60 * S
    assert model["meta"]["seed"] is None
    assert model["meta"]["scenario_name"] is None
    assert model["meta"]["title"] == "fleetsim replay — a"
    assert any("round_us" in n for n in model["meta"]["notes"])


# ---------------------------------------------------------------------------
# Fleet map reconstruction
# ---------------------------------------------------------------------------


def test_fleet_map_from_stints_observed_chips(runs, model):
    assert model["capabilities"]["map"] is True
    assert model["fleet"]["map_level"] == "pod"
    (cluster,) = model["fleet"]["clusters"]
    assert cluster["id"] == "m/c"
    ids = [d["id"] for d in cluster["domains"]]
    assert ids == ["m/c/pod0", "m/c/pod1"]
    assert [d["short"] for d in cluster["domains"]] == ["pod0", "pod1"]
    # Observed chips are a lower bound on the real 16-chip pods.
    for d in cluster["domains"]:
        assert 0 < d["chips"] <= 16
    assert cluster["chips"] == sum(d["chips"] for d in cluster["domains"])
    assert any("max concurrent" in n for n in model["meta"]["notes"])
    # domain_idx indexes the flattened domain list.
    n_domains = len(ids)
    assert all(0 <= i < n_domains for i in model["stints"]["domain_idx"])


def test_scenario_copy_gives_exact_chips_and_meta(runs, tmp_path):
    out = tmp_path / "with_cfg"
    shutil.copytree(runs / "a", out)
    (out / "scenario.yaml").write_text(yaml.safe_dump(_doc("pod")))
    m = build_viz_model(out)
    (cluster,) = m["fleet"]["clusters"]
    assert [d["chips"] for d in cluster["domains"]] == [16, 16]
    assert cluster["chips"] == 32
    assert any("exact" in n for n in m["meta"]["notes"])
    assert m["meta"]["seed"] == 0
    assert m["meta"]["round_us"] == 60 * S
    # The generic copy names (scenario.yaml / config.yaml) carry no
    # information: a "scenario scenario" metaline is suppressed.
    assert m["meta"]["scenario_name"] is None


def test_map_level_hint_passthrough(runs):
    m = build_viz_model(runs / "a", map_level_hint="pod")
    assert m["fleet"]["map_level"] == "pod"
    assert not any("does not match" in n for n in m["meta"]["notes"])


def test_map_level_hint_mismatch_is_noted(runs):
    # A real level name that is NOT what the stints were recorded at
    # must not silently mislabel the map (review fix): the hint still
    # wins, but a note states the disagreement.
    m = build_viz_model(runs / "a", map_level_hint="node")
    assert m["fleet"]["map_level"] == "node"
    note = [n for n in m["meta"]["notes"] if "does not match" in n]
    assert note and "'pod'" in note[0]


def test_stints_columnar_traceable_to_parquet(runs, model):
    st = model["stints"]
    n = len(st["job_id"])
    assert n > 0
    assert all(len(v) == n for v in st.values())
    df = pd.read_parquet(runs / "a" / "stints.parquet")
    assert n == len(df)
    assert st["t0_us"] == sorted(st["t0_us"])
    assert set(st["end_reason"]) <= {
        "completed",
        "preempted",
        "failed",
        "drained",
        "canceled",
        "timeout",
        "running_at_horizon",
    }
    assert set(st["class_name"]) == {"eval"}
    assert set(st["tier"]) == {"batch"}
    assert sum(st["chips"]) == int(df["chips"].sum())


# ---------------------------------------------------------------------------
# Frames downsampling
# ---------------------------------------------------------------------------


def test_frames_full_resolution_when_under_budget(runs, model):
    ts = pd.read_parquet(runs / "a" / "timeseries.parquet")
    frames = model["frames"]
    assert len(frames["t_us"]) == len(ts) == 10  # 10m at 60s rounds
    assert frames["t_us"] == ts["t_us"].tolist()
    for key in ("occupancy", "allocation", "goodput_to_date"):
        assert len(frames[key]) == len(ts)
    for arr in frames["pending_by_class"].values():
        assert len(arr) == len(ts)
    # Full resolution: goodput frames ARE the timeseries column.
    assert frames["goodput_to_date"] == [
        pytest.approx(v) for v in ts["goodput_to_date"].tolist()
    ]


def test_frames_downsampled_preserve_totals_and_ends(runs):
    m = build_viz_model(runs / "a", max_frames=4)
    ts = pd.read_parquet(runs / "a" / "timeseries.parquet")
    frames = m["frames"]
    assert len(frames["t_us"]) == 4
    assert frames["t_us"] == sorted(frames["t_us"])
    assert frames["t_us"][-1] == int(ts["t_us"].iloc[-1])
    # Delta series are bucket-SUMS: totals survive downsampling exactly.
    assert sum(frames["preemptions_delta"]) == int(ts["cum_preemptions"].iloc[-1])
    assert sum(frames["failures_delta"]) == int(ts["cum_node_failures"].iloc[-1])
    for key in ("occupancy", "allocation", "goodput_to_date"):
        assert len(frames[key]) == 4
    for arr in frames["pending_by_class"].values():
        assert len(arr) == 4
        assert all(isinstance(x, int) for x in arr)


def test_frames_values_in_range(model):
    for key in ("occupancy", "allocation"):
        for v in model["frames"][key]:
            assert v is None or 0.0 <= v <= 1.0, (key, v)


# ---------------------------------------------------------------------------
# Gantt, CDFs, palette
# ---------------------------------------------------------------------------


def test_gantt_selection_and_order(runs, model):
    jobs = pd.read_parquet(runs / "a" / "jobs.parquet")
    assert 0 < len(model["gantt"]) <= 300
    keys = [(e["submit_us"], e["id"]) for e in model["gantt"]]
    assert keys == sorted(keys)
    by_id = jobs.set_index("job_id")
    for e in model["gantt"]:
        row = by_id.loc[e["id"]]
        assert e["chips"] == int(row["chips"])
        assert e["submit_us"] == int(row["submit_t_us"])
        assert e["status"] == str(row["status"]).lower()
    m2 = build_viz_model(runs / "a", max_gantt_jobs=5)
    assert len(m2["gantt"]) == 5  # no >=4096-chip jobs in this tiny run


def test_cdfs_shape_and_bounds(model):
    for key in ("queue_wait_s", "jct_s"):
        assert "eval" in model["cdfs"][key]
        for pts in model["cdfs"][key].values():
            assert 0 < len(pts) <= 200
            xs = [p[0] for p in pts]
            ps = [p[1] for p in pts]
            assert xs == sorted(xs)
            assert all(0.0 < p <= 1.0 for p in ps)
            assert ps[-1] == pytest.approx(1.0)


def test_palette_pinned_colors_and_consistency(model):
    pal = model["palette"]
    assert pal["pretrain"] == "#4c6ef5"
    assert pal["finetune"] == "#12b886"
    assert pal["eval"] == "#fab005"
    assert pal["best_effort"] == "#64748b"
    assert pal["inference"] == "#9775fa"
    assert pal["failed"] == "#e03131"
    assert pal["draining"] == "#f76707"
    assert pal["maintenance"] == "#845ef7"
    assert pal["idle"] == "rgba(255,255,255,.05)"
    # Every class label anywhere in the model resolves in the palette.
    labels = (
        set(model["stints"]["class_name"])
        | {e["class_name"] for e in model["gantt"]}
        | set(model["frames"]["pending_by_class"])
        | set(model["cdfs"]["queue_wait_s"])
        | set(model["cdfs"]["jct_s"])
    )
    assert labels <= set(pal)


def test_palette_shared_bucket_labels_get_distinct_shades():
    # Review fix: 'frontier' next to 'pretrain' must not render in the
    # identical indigo — extra labels in a claimed bucket take pinned,
    # separation-validated shade variants of the bucket color.
    from fleetsim.viz.data import _BUCKET_VARIANTS, _build_palette

    notes: list = []
    pal = _build_palette(
        {"eval": "eval", "frontier": "pretrain", "pretrain": "pretrain"}, notes
    )
    assert pal["pretrain"] == "#4c6ef5"
    assert pal["frontier"] == _BUCKET_VARIANTS["pretrain"][0]
    assert pal["frontier"] != pal["pretrain"]
    assert notes == []
    # A custom label ALONE in its bucket keeps the exact bucket color.
    pal2 = _build_palette({"frontier": "pretrain"}, [])
    assert pal2["frontier"] == "#4c6ef5"
    # Deterministic assignment order (sorted labels), variants distinct.
    pal3 = _build_palette(
        {"pretrain": "pretrain", "a": "pretrain", "b": "pretrain"}, []
    )
    assert pal3["a"] == _BUCKET_VARIANTS["pretrain"][0]
    assert pal3["b"] == _BUCKET_VARIANTS["pretrain"][1]
    assert len({pal3["pretrain"], pal3["a"], pal3["b"]}) == 3
    # Exhausting the pinned variants is stated in the notes.
    notes4: list = []
    labs = {"pretrain": "pretrain", "a": "pretrain", "b": "pretrain",
            "c": "pretrain"}
    pal4 = _build_palette(labs, notes4)
    assert pal4["c"] == _BUCKET_VARIANTS["pretrain"][0]
    assert any("exhausted" in n for n in notes4)


def test_summary_cards_from_summary_json_only(runs, model):
    summary = json.loads((runs / "a" / "summary.json").read_text())
    labels = [c["label"] for c in model["summary_cards"]]
    assert labels[:4] == [
        "occupancy",
        "goodput",
        "jobs finished",
        "preemptions/min",
    ]
    finished = next(c for c in model["summary_cards"] if c["label"] == "jobs finished")
    assert finished["value"] == str(summary["full"]["counts"]["jobs_finished"])
    occ = next(c for c in model["summary_cards"] if c["label"] == "occupancy")
    want = summary["window"]["occupancy"]
    assert occ["value"] == (f"{want * 100:.1f}%" if want is not None else "n/a")


# ---------------------------------------------------------------------------
# No-stints degradation
# ---------------------------------------------------------------------------


def test_no_stints_degrades_to_fleet_level_replay(runs):
    m = build_viz_model(runs / "nostints")
    assert m["capabilities"]["map"] is False
    assert m["fleet"] == {"map_level": None, "clusters": []}
    assert all(v == [] for v in m["stints"].values())
    assert any("stints" in n for n in m["meta"]["notes"])
    # Fleet-level replay still fully populated.
    assert len(m["frames"]["t_us"]) == 10
    assert len(m["gantt"]) > 0
    assert m["cdfs"]["queue_wait_s"]
    to_json(m)  # serializable


def test_missing_required_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a fleetsim output"):
        build_viz_model(tmp_path)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def test_compare_mode(runs):
    m = build_viz_model(runs / "a", compare_dir=runs / "b")
    assert m["capabilities"]["compare"] is True
    cmp = m["compare"]
    assert list(cmp) == ["label_a", "label_b", "frames_b", "summary_cards_b"]
    assert cmp["label_a"] == "a"
    assert cmp["label_b"] == "b"
    assert list(cmp["frames_b"]) == list(m["frames"])
    assert len(cmp["frames_b"]["t_us"]) == 10
    assert [c["label"] for c in cmp["summary_cards_b"][:4]] == [
        "occupancy",
        "goodput",
        "jobs finished",
        "preemptions/min",
    ]
    assert "vs" in m["meta"]["title"]
    # Map/stints/gantt still describe run A only.
    assert m["capabilities"]["map"] is True


def test_compare_colliding_basenames_use_full_paths(runs, tmp_path):
    for sub in ("x", "y"):
        shutil.copytree(runs / "a", tmp_path / sub / "out")
    m = build_viz_model(tmp_path / "x" / "out", compare_dir=tmp_path / "y" / "out")
    assert m["compare"]["label_a"] != m["compare"]["label_b"]
    assert m["compare"]["label_a"].endswith(os.path.join("x", "out"))


# ---------------------------------------------------------------------------
# Determinism and NaN-freedom
# ---------------------------------------------------------------------------


def test_deterministic_serialization(runs):
    a = to_json(build_viz_model(runs / "a", compare_dir=runs / "b"))
    b = to_json(build_viz_model(runs / "a", compare_dir=runs / "b"))
    assert a == b


def _walk_no_bad_floats(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            assert isinstance(k, str), f"non-string key at {path}: {k!r}"
            _walk_no_bad_floats(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk_no_bad_floats(v, f"{path}[{i}]")
    elif isinstance(node, float):
        assert math.isfinite(node), f"non-finite float at {path}"
    else:
        assert node is None or isinstance(node, (str, int, bool)), (
            f"non-JSON type at {path}: {type(node).__name__}"
        )


def test_nan_free_and_json_safe(runs):
    m = build_viz_model(runs / "a", compare_dir=runs / "b")
    _walk_no_bad_floats(m)
    # to_json's allow_nan=False is the enforced guarantee.
    json.loads(to_json(m))


# ---------------------------------------------------------------------------
# Frontier semantics (hand-driven collector: fast, exact)
# ---------------------------------------------------------------------------

H = 1000 * S


def _metro_fleet():
    """m/c with two pods of two 8-chip nodes (engine id grammar)."""
    doms = [
        Domain(id="m", level="metro", parent=None, children=["m/c"],
               chip_type="h100"),
        Domain(id="m/c", level="cluster", parent="m",
               children=["m/c/pod0", "m/c/pod1"], chip_type="h100"),
    ]
    for p in ("pod0", "pod1"):
        pid = f"m/c/{p}"
        doms.append(
            Domain(id=pid, level="pod", parent="m/c",
                   children=[f"{pid}/n0", f"{pid}/n1"], chip_type="h100")
        )
        for nid in ("n0", "n1"):
            doms.append(
                Domain(id=f"{pid}/{nid}", level="node", parent=pid,
                       children=[], chip_type="h100", chips=8)
            )
    return FleetTree(doms, cluster_roots=["m/c"])


def _mk_job(jid, chips, submit_s, klass=JobClass.PRETRAIN, tier=Tier.PROD,
            source_class="frontier"):
    return Job(
        id=jid,
        tenant="t0",
        job_class=klass,
        submit_t=int(submit_s * S),
        gangs=[GangSpec(chips=chips, chip_type="h100")],
        tier=tier,
        true_duration_s=100.0,
        source_class=source_class,
    )


@pytest.fixture()
def frontier_out(tmp_path):
    """Outputs of a hand-driven collector holding one 32,768-chip job."""
    c = MetricsCollector(H, fleet=_metro_fleet(), stints="pod")
    big = _mk_job("frontier-0", 32768, submit_s=100.0)
    c.job_submitted(big, 100 * S)
    c.job_admitted(big, 100 * S)
    alloc = Allocation(
        big.id,
        [GangAlloc(nodes={"m/c/pod0/n0": 16384, "m/c/pod1/n0": 16384},
                   anchor="m/c")],
    )
    c.job_started(big, alloc, 150 * S)
    c.chips_allocated(32768, "h100", 150 * S)
    c.chips_freed(32768, "h100", 900 * S)
    c.job_finished(big, 900 * S, JobStatus.COMPLETED, 32768 * 700.0, 0.0)
    small = _mk_job("eval-0", 8, submit_s=0.0, klass=JobClass.EVAL,
                    tier=Tier.BATCH, source_class="eval")
    c.job_submitted(small, 0)
    c.job_admitted(small, 0)
    out = tmp_path / "frontier"
    write_outputs(c, out)
    return out


def test_frontier_events_emitted(frontier_out):
    m = build_viz_model(frontier_out)
    frontier = [e for e in m["events"] if e["kind"].startswith("frontier")]
    assert [(e["kind"], e["t_us"], e["magnitude"]) for e in frontier] == [
        ("frontier_submit", 100 * S, 32768),
        ("frontier_start", 150 * S, 32768),
    ]
    assert "frontier-0" in frontier[0]["label"]


def test_gantt_always_includes_big_jobs(frontier_out):
    # max_gantt_jobs=0: only the >=4096-chip rule can admit a job.
    m = build_viz_model(frontier_out, max_gantt_jobs=0)
    assert [e["id"] for e in m["gantt"]] == ["frontier-0"]
    (entry,) = m["gantt"]
    assert entry["class_name"] == "frontier"
    assert entry["start_us"] == 150 * S
    assert entry["end_us"] == 900 * S
    assert entry["status"] == "completed"


def test_frontier_stint_shares_and_empty_timeseries(frontier_out):
    m = build_viz_model(frontier_out)
    # One stint x two pods, 16,384 chips each — shares sum to the job.
    st = m["stints"]
    fr = [i for i, j in enumerate(st["job_id"]) if j == "frontier-0"]
    assert sorted(st["domain_idx"][i] for i in fr) == [0, 1]
    assert sum(st["chips"][i] for i in fr) == 32768
    # A hand-driven collector never flushed: frames degrade to empty
    # arrays (and the model still serializes).
    assert m["frames"]["t_us"] == []
    assert m["meta"]["round_us"] is None
    to_json(m)


# ---------------------------------------------------------------------------
# Frontier-scale size budget (opt-in: ~80 s of simulation)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FLEETSIM_FRONTIER_BUDGET"),
    reason="set FLEETSIM_FRONTIER_BUDGET=1 to run the ~80 s"
    " examples/04_frontier size-budget check (last measured: ~4.7 MB)",
)
def test_frontier_example_model_under_15mb(tmp_path):
    example = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "04_frontier"
        / "scenario.yaml"
    )
    out = tmp_path / "out04"
    run_scenario(example, out_dir=out)
    m = build_viz_model(out)
    size = len(to_json(m).encode("utf-8"))
    assert size <= 15 * 1024 * 1024, f"model is {size / 1e6:.1f} MB"
    assert m["capabilities"]["map"] is True
    assert len(m["frames"]["t_us"]) <= 1200
    assert any(e["kind"] == "frontier_start" for e in m["events"])
