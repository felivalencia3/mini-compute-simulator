"""v0.8 analysis affordances: the derived model series and the kernels
the analysis tab attributes with.

Two halves, both fast:

1. THE MODEL (pure Python).  ``fleetsim.viz.data`` gained an additive
   block of ``frames`` series — allocated/healthy chips, the collector's
   own pending/running totals, failure kills and the per-level
   fragmentation index.  These tests pin that every one of them is a
   straight read of ``timeseries.parquet`` under the documented bucket
   rule (so a number on the analysis screen is traceable to a recorded
   column), that a missing column degrades to null rather than zero, and
   that building the model is a PURE READ — no recorded byte moves,
   which is what keeps the shipped examples byte-identical.

2. THE KERNELS (JavaScript, executed under node).
   ``serve/static/insight.js`` exports its analysis kernels as pure
   functions precisely so the arithmetic on that screen can be tested
   rather than eyeballed.  A synthetic model with a PLANTED preemption
   wave and a PLANTED occupancy dip is fed through
   ``buildStintIndex`` / ``drillDown`` / ``pearsonFit`` / ``detectDips``
   / ``attributeDip``, and the results are checked against values
   computed by hand here.  node ships on every GitHub runner; the
   JS half skips (loudly) if the interpreter is absent locally.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from fleetsim.api import run_scenario
from fleetsim.viz import build_viz_model, to_json

S = 1_000_000
STATIC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fleetsim"
    / "serve"
    / "static"
)

#: The analysis series added in v0.8, in schema order.
ANALYSIS_KEYS = (
    "allocated_chips",
    "healthy_chips",
    "pending_jobs",
    "running_jobs",
    "failure_kills_delta",
    "frag_index",
)


# ===========================================================================
# 1. the model extension
# ===========================================================================


def _doc(seed: int = 3):
    return {
        "sim": {"horizon": "12m", "round": "60s", "seed": seed},
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
                    "rate_per_hour": 90,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=2m, p90=5m]",
                    "abort_prob": 0,
                }
            },
        },
        "outputs": {"stints": "pod"},
    }


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("insight") / "a"
    run_scenario(_doc(), out_dir=out)
    return out


@pytest.fixture(scope="module")
def model(run_dir):
    return build_viz_model(run_dir)


def test_analysis_series_are_appended_without_disturbing_the_v03_keys(model):
    keys = list(model["frames"])
    assert keys[:7] == [
        "t_us",
        "occupancy",
        "allocation",
        "goodput_to_date",
        "pending_by_class",
        "preemptions_delta",
        "failures_delta",
    ], "the pinned v0.3 frames prefix must not move"
    assert keys[7:] == list(ANALYSIS_KEYS)


def test_analysis_series_are_verbatim_columns_at_full_resolution(run_dir, model):
    ts = pd.read_parquet(run_dir / "timeseries.parquet")
    frames = model["frames"]
    n = len(ts)
    assert len(frames["t_us"]) == n  # 12 rounds: under the frame budget

    for key in ("allocated_chips", "healthy_chips"):
        assert frames[key] == [pytest.approx(v) for v in ts[key].tolist()]
    for key in ("pending_jobs", "running_jobs"):
        assert frames[key] == [int(v) for v in ts[key].tolist()]
        assert all(isinstance(v, int) for v in frames[key])

    kills = ts["cum_failure_kills"].tolist()
    assert sum(frames["failure_kills_delta"]) == int(kills[-1])
    assert frames["failure_kills_delta"][0] == int(kills[0])

    # one frag series per fleet level, keys sorted, values verbatim
    levels = sorted(
        c[len("frag_index_") :] for c in ts.columns if c.startswith("frag_index_")
    )
    assert levels and list(frames["frag_index"]) == levels
    for level in levels:
        assert frames["frag_index"][level] == [
            pytest.approx(v) for v in ts[f"frag_index_{level}"].tolist()
        ]


def test_analysis_series_follow_the_documented_bucket_rule(run_dir):
    """Downsampled: value series are bucket MEANS, deltas bucket SUMS."""
    ts = pd.read_parquet(run_dir / "timeseries.parquet")
    m = build_viz_model(run_dir, max_frames=3)
    frames = m["frames"]
    n = len(ts)
    edges = [(n * i // 3, n * (i + 1) // 3) for i in range(3)]

    for i, (a, b) in enumerate(edges):
        assert frames["allocated_chips"][i] == pytest.approx(
            ts["allocated_chips"][a:b].mean()
        )
        assert frames["healthy_chips"][i] == pytest.approx(
            ts["healthy_chips"][a:b].mean()
        )
        assert frames["pending_jobs"][i] == round(ts["pending_jobs"][a:b].mean())
        for level, series in frames["frag_index"].items():
            assert series[i] == pytest.approx(
                ts[f"frag_index_{level}"][a:b].mean()
            )
    # totals survive downsampling exactly (a sum, not a mean)
    assert sum(frames["failure_kills_delta"]) == int(
        ts["cum_failure_kills"].iloc[-1]
    )


def test_analysis_series_are_null_not_zero_when_a_column_is_absent(
    run_dir, tmp_path
):
    """An older run without ``pending_jobs`` must read "unknown", not 0."""
    stripped = tmp_path / "stripped"
    stripped.mkdir()
    for name in ("summary.json", "jobs.parquet"):
        shutil.copy(run_dir / name, stripped / name)
    ts = pd.read_parquet(run_dir / "timeseries.parquet")
    ts.drop(columns=["pending_jobs", "cum_failure_kills"]).to_parquet(
        stripped / "timeseries.parquet"
    )

    m = build_viz_model(stripped)
    frames = m["frames"]
    assert frames["pending_jobs"] == [None] * len(frames["t_us"])
    # no cum_failure_kills column: UNKNOWN, not "no kills happened".  A
    # flat run of real zeros drew a confident "nothing was ever killed by
    # a node failure" while the note on the same page said otherwise.
    assert frames["failure_kills_delta"] == [None] * len(frames["t_us"])
    for column in ("pending_jobs", "cum_failure_kills"):
        assert any(
            "null, not zero" in note and column in note
            for note in m["meta"]["notes"]
        ), (column, m["meta"]["notes"])
    to_json(m)  # still JSON-safe


def test_analysis_series_survive_an_empty_timeseries(model, tmp_path, run_dir):
    """The degenerate model (a collector that never flushed) stays whole."""
    empty = tmp_path / "empty"
    empty.mkdir()
    for name in ("summary.json", "jobs.parquet"):
        shutil.copy(run_dir / name, empty / name)
    ts = pd.read_parquet(run_dir / "timeseries.parquet")
    ts.iloc[0:0].to_parquet(empty / "timeseries.parquet")

    frames = build_viz_model(empty)["frames"]
    for key in ANALYSIS_KEYS:
        assert frames[key] == ([] if key != "frag_index" else {})


def test_building_the_model_is_a_pure_read(run_dir, tmp_path):
    """BYTE-COMPAT: the analysis additions are read-side only.

    The shipped examples' byte-identity is asserted end to end in
    tests/test_serve_live.py::test_shipped_examples_are_byte_identical_under_v08;
    what belongs HERE is the property that makes it hold — building the
    (now larger) model touches no recorded byte, and is deterministic.
    """
    before = {p.name: p.read_bytes() for p in sorted(run_dir.iterdir()) if p.is_file()}
    first = to_json(build_viz_model(run_dir))
    second = to_json(build_viz_model(run_dir))
    after = {p.name: p.read_bytes() for p in sorted(run_dir.iterdir()) if p.is_file()}
    assert after == before, "build_viz_model wrote to the run directory"
    assert first == second, "the model is not a pure function of its inputs"


def test_model_cache_markers_track_the_current_schema(model):
    """A stale ``viz_model.json`` must be rebuilt, not served.

    Runs are immutable so the cache never invalidates on content — but
    the schema grows, and an old cache would silently starve the analysis
    tab.  The server's markers have to be keys the current model really
    carries.
    """
    from fleetsim.serve.runs import MODEL_CACHE_MARKERS, RunManager

    payload = to_json(model)
    assert RunManager._model_cache_is_current(payload)
    for marker in MODEL_CACHE_MARKERS:
        assert marker in payload, marker
    # a v0.7-shaped cache (frames without the analysis block) is stale
    old = dict(model)
    old["frames"] = {
        k: v for k, v in model["frames"].items() if k not in ANALYSIS_KEYS
    }
    assert not RunManager._model_cache_is_current(to_json(old))


# ===========================================================================
# 2. the insight.js kernels, under node
# ===========================================================================

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(
    NODE is None,
    reason="node is required to execute the insight.js kernels (CI runners ship it)",
)

#: The harness imports insight.js as an ES module, runs the kernels over a
#: model handed in on argv, and prints one JSON document.  insight.js is
#: copied to a .mjs sibling first: the repo ships no package.json, so a
#: bare .js file would be parsed as CommonJS and its `export` rejected.
_HARNESS = """\
import {
  buildStintIndex, drillDown, pearsonFit, detectDips, attributeDip,
  TIER_RANK, DISRUPTIVE_END_REASONS,
} from "./insight.mjs";

const req = JSON.parse(process.argv[2]);
const model = req.model;
const index = buildStintIndex(model);
const out = { tiers: TIER_RANK, disruptive: [...DISRUPTIVE_END_REASONS].sort() };

out.stint_index = { n: index.n, domains: index.domains };
out.drill = drillDown(model, index, req.event_t_us, req.lookahead || 0);
out.drill_ahead = drillDown(model, index, req.event_t_us, 1);
out.fits = {};
for (const [name, pair] of Object.entries(req.pairs || {})) {
  const fit = pearsonFit(pair[0], pair[1]);
  out.fits[name] = { n: fit.n, r: fit.r, slope: fit.slope,
                     intercept: fit.intercept, reason: fit.reason || null };
}
const found = detectDips(model.frames, req.k, req.dip_window);
out.dips = {
  sigma: found.sigma, reason: found.reason || null,
  items: found.dips.map((d) => attributeDip(model, index, d)),
};
process.stdout.write(JSON.stringify(out));
"""


def run_kernels(request: dict, tmp_path: Path) -> dict:
    """Execute the insight.js kernels under node and return their output."""
    work = tmp_path / "js"
    work.mkdir(exist_ok=True)
    shutil.copy(STATIC / "insight.js", work / "insight.mjs")
    (work / "harness.mjs").write_text(_HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(work / "harness.mjs"), json.dumps(request)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# the synthetic model: a planted wave and a planted dip
# ---------------------------------------------------------------------------

ROUND = 60 * S
#: 20 rounds.  Frame i is the flush at t = (i + 1) * ROUND, covering the
#: half-open round (i * ROUND, (i + 1) * ROUND] — so the disruption
#: planted in frames DIP_AT / DIP_AT+1 lands in the window (W0, DIP_T].
N_ROUNDS = 20
DIP_AT = 10
DIP_T = (DIP_AT + 1) * ROUND
W0 = DIP_AT * ROUND


def _stint(job, cls, tier, dom, chips, t0, t1, reason):
    return {
        "job_id": job,
        "class_name": cls,
        "tier": tier,
        "domain_idx": dom,
        "chips": chips,
        "t0_us": t0,
        "t1_us": t1,
        "end_reason": reason,
    }


def synthetic_model() -> dict:
    """A hand-built model with known answers.

    Fleet: two pods, 200 chips.  Occupancy sits at 0.50 (100 allocated)
    every round except frames DIP_AT and DIP_AT+1, where a preemption
    wave and a node failure take chips away (100 -> 60 -> 70).

    Planted into the window (W0, DIP_T] — which is both the drill-down's
    round AND the dip's attribution window:
      * ``vic-a`` (batch, 16 chips, pod0) preempted
      * ``vic-b`` (batch, 8 chips, pod1)  preempted
      * ``vic-c`` (batch, 12 chips, pod0) FAILED
      * ``vic-d`` (batch, 4 chips, pod0)  completed  <- freed, not a victim
      * ``bystander`` is preempted in the NEXT round <- outside the window
      * ``winner`` (prod, 20 chips, pod0) starts in the window -> the
        inferred displacer of vic-a and vic-c (same domain, higher tier)
      * ``peer`` (batch, 6 chips, pod1) starts in the window -> NOT a
        displacer of vic-b: same tier, so there is no ordering evidence
      * ``early`` lives entirely before the window
    """
    t = [(i + 1) * ROUND for i in range(N_ROUNDS)]
    occ, alloc, healthy, pend, frag, fails, preempts = [], [], [], [], [], [], []
    for i in range(N_ROUNDS):
        if i == DIP_AT:
            a, h = 60.0, 200.0
        elif i == DIP_AT + 1:
            a, h = 70.0, 200.0
        else:
            a, h = 100.0, 200.0
        alloc.append(a)
        healthy.append(h)
        occ.append(a / h)
        # pending rises exactly with fragmentation: a known, exact r = 1
        frag.append(0.10 + 0.01 * i)
        pend.append(2 + i)
        fails.append(1 if i == DIP_AT else 0)
        preempts.append(2 if i == DIP_AT else 0)

    stints = [
        _stint("early", "eval", "batch", 0, 5, ROUND, 2 * ROUND, "completed"),
        _stint("vic-a", "finetune", "batch", 0, 16, 3 * ROUND, DIP_T, "preempted"),
        _stint("vic-b", "finetune", "batch", 1, 8, 4 * ROUND, DIP_T, "preempted"),
        _stint("vic-c", "eval", "batch", 0, 12, 5 * ROUND, DIP_T - 10, "failed"),
        _stint("vic-d", "eval", "batch", 0, 4, 5 * ROUND, DIP_T - 5, "completed"),
        _stint("bystander", "eval", "batch", 1, 9, 6 * ROUND, 12 * ROUND, "preempted"),
        _stint("peer", "eval", "batch", 1, 6, DIP_T - 30, 15 * ROUND, "completed"),
        _stint("winner", "pretrain", "prod", 0, 20, DIP_T - 20, 18 * ROUND, "completed"),
    ]
    # the schema pins stints sorted by (t0_us, job_id, domain, t1_us)
    stints.sort(key=lambda s: (s["t0_us"], s["job_id"], s["domain_idx"], s["t1_us"]))
    columnar = {
        key: [s[key] for s in stints]
        for key in (
            "job_id", "class_name", "tier", "domain_idx", "chips",
            "t0_us", "t1_us", "end_reason",
        )
    }
    return {
        "meta": {"round_us": ROUND, "horizon_us": N_ROUNDS * ROUND},
        "capabilities": {"map": True, "compare": False},
        "fleet": {
            "map_level": "pod",
            "clusters": [
                {
                    "id": "m/c",
                    "chips": 200,
                    "domains": [
                        {"id": "m/c/pod0", "short": "pod0", "chips": 100},
                        {"id": "m/c/pod1", "short": "pod1", "chips": 100},
                    ],
                }
            ],
        },
        "frames": {
            "t_us": t,
            "occupancy": occ,
            "allocated_chips": alloc,
            "healthy_chips": healthy,
            "pending_jobs": pend,
            "failures_delta": fails,
            "preemptions_delta": preempts,
            "frag_index": {"pod": frag},
        },
        "stints": columnar,
        "events": [
            {
                "t_us": DIP_T,
                "kind": "preemption_wave",
                "label": "preemption wave: 2 jobs",
                "magnitude": 2,
            }
        ],
    }


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    if NODE is None:
        pytest.skip("node not available")
    model = synthetic_model()
    request = {
        "model": model,
        "event_t_us": DIP_T,
        "k": 2.0,
        "dip_window": 7,
        "pairs": {
            "frag_pending": [
                model["frames"]["frag_index"]["pod"],
                model["frames"]["pending_jobs"],
            ],
            "fail_occ": [
                model["frames"]["failures_delta"],
                model["frames"]["occupancy"],
            ],
            "flat": [[1, 1, 1, 1], [2, 3, 4, 5]],
            "short": [[1, 2], [3, 4]],
            "holey": [[1, None, 3, 4, 5], [2, 9, 6, 8, 10]],
        },
    }
    return run_kernels(request, tmp_path_factory.mktemp("kernels"))


@needs_node
def test_kernel_constants_match_the_simulator(kernels):
    from fleetsim.model import Tier

    assert kernels["tiers"] == {
        name.lower(): int(member)
        for name, member in Tier.__members__.items()
        if name != "FREE"  # legacy alias for BEST_EFFORT
    }
    assert kernels["disruptive"] == ["drained", "failed", "preempted"]
    assert kernels["stint_index"] == {
        "n": 8,
        "domains": ["m/c/pod0", "m/c/pod1"],
    }


@needs_node
def test_drill_down_finds_exactly_the_planted_jobs(kernels):
    d = kernels["drill"]
    assert d["ok"] is True
    assert d["window"] == [W0, DIP_T]
    assert d["round_us"] == ROUND

    # exactly the three disruptively-ended stints in this round: the
    # completion (vic-d) and the next round's preemption (bystander) are
    # NOT victims, and neither is anything outside the window
    assert [j["job_id"] for j in d["jobs"]] == ["vic-a", "vic-c", "vic-b"]
    assert [j["chips"] for j in d["jobs"]] == [16, 12, 8]
    assert [j["end_reasons"] for j in d["jobs"]] == [
        ["preempted"],
        ["failed"],
        ["preempted"],
    ]
    assert d["n_jobs"] == 3
    assert d["n_in_round"] == 3  # nothing here needed the lookahead
    assert d["chips_freed"] == 16 + 12 + 8

    # chips claimed = every stint that STARTED in the round (winner+peer)
    assert d["chips_claimed"] == 20 + 6
    # only `winner` qualifies as a displacer: higher tier, same domain
    assert d["chips_to_displacers"] == 20
    assert d["n_with_displacer"] == 2


@needs_node
def test_the_settlement_lookahead_widens_the_window_and_says_so(kernels):
    """A preempted job keeps its chips through its grace window.

    Real runs settle a preemption wave one round AFTER the round that
    counted it, so the drill-down offers an explicit lookahead — and
    reports which victims needed it, instead of quietly widening.
    """
    tight = kernels["drill"]
    wide = kernels["drill_ahead"]
    assert tight["window"] == [W0, DIP_T]
    assert wide["window"] == [W0, DIP_T + ROUND]
    assert wide["round_end_us"] == DIP_T  # the event's own round still ends here

    # `bystander` is preempted one round later: invisible at 0, found at 1
    assert "bystander" not in [j["job_id"] for j in tight["jobs"]]
    late = [j for j in wide["jobs"] if j["job_id"] == "bystander"]
    assert len(late) == 1 and late[0]["in_round"] is False
    assert wide["n_jobs"] == 4 and wide["n_in_round"] == 3


@needs_node
def test_displacement_is_inferred_only_where_the_evidence_exists(kernels):
    by_id = {j["job_id"]: j for j in kernels["drill"]["jobs"]}
    for victim in ("vic-a", "vic-c"):  # pod0, batch — `winner` is prod on pod0
        assert [d["job_id"] for d in by_id[victim]["displacers"]] == ["winner"]
        assert by_id[victim]["displacers"][0]["chips"] == 20
        assert by_id[victim]["displacers"][0]["tier"] == "prod"
    # vic-b is on pod1, where the only starter (`peer`) is the SAME tier:
    # no ordering evidence, so no inference is offered
    assert by_id["vic-b"]["displacers"] == []
    assert by_id["vic-b"]["rank_known"] is True


@needs_node
def test_correlation_matches_a_hand_computation(kernels):
    fits = kernels["fits"]

    # fragmentation and pending were both planted as exact linear ramps
    # (0.10 + 0.01 i and 2 + i), so r is exactly 1 and the slope is 100
    frag = fits["frag_pending"]
    assert frag["n"] == N_ROUNDS
    assert frag["r"] == pytest.approx(1.0)
    assert frag["slope"] == pytest.approx(100.0)
    assert frag["intercept"] == pytest.approx(-8.0)

    # failures vs occupancy, computed here from the same arrays
    model = synthetic_model()
    xs = model["frames"]["failures_delta"]
    ys = model["frames"]["occupancy"]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    assert fits["fail_occ"]["n"] == n
    assert fits["fail_occ"]["r"] == pytest.approx(sxy / math.sqrt(sxx * syy))
    assert fits["fail_occ"]["slope"] == pytest.approx(sxy / sxx)
    assert fits["fail_occ"]["r"] < 0  # the one failure round is the dip

    # honest refusals rather than a fabricated number
    assert fits["flat"]["r"] is None and fits["flat"]["reason"] == "one axis never varies"
    assert fits["short"]["r"] is None and "fewer than 3" in fits["short"]["reason"]
    # a null on either axis drops the PAIR, it does not shift the series
    assert fits["holey"]["n"] == 4
    assert fits["holey"]["r"] == pytest.approx(1.0)
    assert fits["holey"]["slope"] == pytest.approx(2.0)


@needs_node
def test_dip_detection_finds_the_planted_dip(kernels):
    dips = kernels["dips"]
    assert dips["reason"] is None
    assert dips["sigma"] > 0
    assert len(dips["items"]) == 1, dips["items"]
    dip = dips["items"][0]
    # rounds 10 and 11 (0-based) are the dip; round 10 is the trough
    assert dip["i0"] == DIP_AT
    assert dip["i1"] == DIP_AT + 1
    assert dip["trough"] == DIP_AT
    assert dip["pre"] == DIP_AT - 1
    assert dip["t_start_us"] == W0  # the flush time of the frame BEFORE
    assert dip["t_end_us"] == DIP_T
    assert dip["occ_before"] == pytest.approx(0.5)
    assert dip["occ_trough"] == pytest.approx(0.3)
    assert dip["alloc_drop"] == pytest.approx(40.0)
    assert dip["healthy_delta"] == pytest.approx(0.0)
    assert dip["baseline_known"] is True


@needs_node
def test_dip_decomposition_sums_to_the_observed_drop(kernels):
    dip = kernels["dips"]["items"][0]
    assert dip["attributable"] is True

    # the window (W0, DIP_T] holds exactly the planted endings
    assert dip["freed"] == {
        "failed": 12,     # vic-c
        "preempted": 24,  # vic-a + vic-b
        "drained": 0,
        "other": 4,       # vic-d completed
    }
    assert dip["freed_total"] == 40
    assert dip["claimed"] == 26  # winner + peer started in the same window

    # THE identity the panel promises: freed − re-claimed + residual is
    # EXACTLY the observed drop in allocated chips, with the residual
    # carrying everything the recorded stints do not explain.
    assert dip["freed_total"] - dip["claimed"] + dip["residual"] == pytest.approx(
        dip["alloc_drop"]
    )
    assert dip["residual"] == pytest.approx(26.0)


@needs_node
def test_a_run_without_stints_is_reported_as_undecomposable(tmp_path_factory):
    """No stint data must read as "cannot attribute", never as zero."""
    model = synthetic_model()
    model["stints"] = {k: [] for k in model["stints"]}
    model["capabilities"]["map"] = False
    out = run_kernels(
        {"model": model, "event_t_us": DIP_T, "k": 2.0, "dip_window": 7},
        tmp_path_factory.mktemp("nostints"),
    )
    assert out["drill"] == {
        "ok": False, "reason": "no-stints", "t_us": DIP_T, "window": [W0, DIP_T],
    }
    (dip,) = out["dips"]["items"]
    assert dip["attributable"] is False
    assert dip["residual"] is None  # never 0: nothing was measured
    assert dip["freed_total"] == 0 and dip["claimed"] == 0
    # the dip itself is still SIZED — only its causes are unknown
    assert dip["alloc_drop"] == pytest.approx(40.0)


@needs_node
def test_a_run_without_a_round_length_refuses_to_attribute(tmp_path_factory):
    model = synthetic_model()
    model["meta"]["round_us"] = None
    out = run_kernels(
        {"model": model, "event_t_us": DIP_T, "k": 2.0, "dip_window": 7},
        tmp_path_factory.mktemp("noround"),
    )
    assert out["drill"]["ok"] is False
    assert out["drill"]["reason"] == "no-round"


@needs_node
def test_k_widens_and_narrows_the_net(tmp_path_factory):
    model = synthetic_model()
    base = {"model": model, "event_t_us": DIP_T, "dip_window": 7}
    work = tmp_path_factory.mktemp("ksweep")
    loose = run_kernels({**base, "k": 0.5}, work)
    tight = run_kernels({**base, "k": 6.0}, work)
    assert len(loose["dips"]["items"]) >= len(tight["dips"]["items"])
    assert not tight["dips"]["items"], "k=6 must not flag a 2-sigma dip"


# ---------------------------------------------------------------------------
# the module's shape: what app.js and the CSP depend on
# ---------------------------------------------------------------------------


def test_insight_module_keeps_the_markup_hygiene_rules():
    js = (STATIC / "insight.js").read_text(encoding="utf-8")
    for banned in (".innerHTML", "document.write", "eval("):
        assert banned not in js, banned
    assert "createElementNS" in js
    # Self-contained AT LOAD TIME: no STATIC import, so node can load the
    # file bare (run_kernels copies only insight.js into a temp dir, where
    # any sibling specifier would be unresolvable).  The one dynamic
    # import — export.js, for the charts' PNG buttons — sits inside a
    # function body, so it is resolved only when a chart is built, which
    # the kernel harness never does.
    assert "\nimport " not in js
    assert js.count("import(") == 1, "only export.js may be imported, lazily"
    assert 'import("./export.js")' in js


def test_app_js_lazy_loads_insight_and_routes_to_it():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'import("./insight.js")' in js
    assert "report|fleet3d|insight" in js
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for needle in ('id="tabInsight"', 'id="insight"', 'id="insightMount"'):
        assert needle in html, needle


def test_dip_palette_is_the_apps_state_palette_and_stays_distinct():
    """The dip causes borrow the app's SEMANTIC state colors.

    failures/drains/normal-endings are the design system's --failed,
    --maintenance and --done; only the preemption slot is a free choice,
    and it must not collide with any of them.  (CVD separation was
    validated with the dataviz palette validator — see the comment on
    DIP_PARTS.)
    """
    import re

    from fleetsim.viz.data import _STATE_COLORS

    js = (STATIC / "insight.js").read_text(encoding="utf-8")
    block = re.search(r"const DIP_PARTS = \[(.*?)\];", js, re.S)
    assert block, "insight.js must define DIP_PARTS"
    colors = [h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}", block.group(1))]
    assert len(colors) == 4 and len(set(colors)) == 4, colors
    assert colors[0] == _STATE_COLORS["failed"]
    assert colors[2] == _STATE_COLORS["maintenance"]
    keys = re.findall(r'key: "([a-z_]+)"', block.group(1))
    assert keys == ["failed", "preempted", "drained", "other"]
