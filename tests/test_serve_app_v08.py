"""v0.8 phase 3: live replay, node drill-down, deep links + export, the
validation tab, and the editor's fleet-shape preview.

These are structural/contract tests, deliberately CI-fast — no simulation
runs here.  What they pin:

- ``scenario_fleet_shape`` is arithmetic (it must never materialize a
  fleet: it runs on every debounced keystroke, and a scenario may legally
  declare 262,144 nodes), exact about the counts, and bounded;
- ``POST /api/validate`` carries that shape for a valid scenario and
  omits it for an invalid one, without disturbing the old envelope;
- ``GET /api/validation`` answers with the payload the validation tab
  renders, and every rendering rule the tab relies on holds in the data;
- the new static modules keep the shell's hygiene rules (no markup
  injection, no external URL, same-origin ES modules only);
- ``list_runs`` reports CONTIGUOUS 1-based queue positions even when the
  dispatcher admits a run mid-listing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from fleetsim.serve.runs import (
    MAX_PREVIEW_CLUSTERS,
    MAX_PREVIEW_DOMAINS,
    RunManager,
    scenario_fleet_shape,
)
from fleetsim.serve.server import FleetsimHTTPServer

STATIC = Path(__file__).resolve().parents[1] / "src" / "fleetsim" / "serve" / "static"

#: Modules added by this phase.  They live under /static and are loaded as
#: same-origin ES modules, so the CSP rules that cover app.js cover them.
NEW_MODULES = ("live.js", "fleetmap.js", "validation.js", "export.js")


def tiny_doc(counts=(2, 4), levels=("rack", "node"), per_node=8):
    return {
        "sim": {"horizon": "10m", "round": "60s", "seed": 1},
        "fleet": {
            "metro": "demo",
            "clusters": [
                {
                    "name": "c1",
                    "chip": {"type": "h100", "per_node": per_node},
                    "topology": {"levels": list(levels), "counts": list(counts)},
                }
            ],
        },
        "workload": {
            "kind": "synthetic",
            "classes": {
                "eval": {
                    "rate_per_hour": 10,
                    "chips": "pow2[1, 4]",
                    "duration": "lognormal[median=1m, p90=3m]",
                    "tier": "batch",
                }
            },
        },
        "scheduler": {"name": "tiered_priority"},
        "outputs": {"stints": True},
    }


# ---------------------------------------------------------------------------
# scenario_fleet_shape
# ---------------------------------------------------------------------------


def test_fleet_shape_counts_match_the_declared_topology():
    shape = scenario_fleet_shape(yaml.safe_dump(tiny_doc()))
    assert shape["n_clusters"] == 1
    assert shape["total_nodes"] == 2 * 4
    assert shape["total_chips"] == 2 * 4 * 8
    assert shape["chip_types"] == ["h100"]
    (cluster,) = shape["clusters"]
    assert cluster["id"] == "demo/c1"
    assert cluster["levels"] == ["cluster", "rack", "node"]
    assert cluster["map_level"] == "rack"
    assert cluster["chips_per_node"] == 8
    assert cluster["n_domains"] == 2
    # ids are numbered exactly the way fleet.build numbers them, so a
    # preview block and a later stint row name the same domain
    assert [d["short"] for d in cluster["domains"]] == ["rack0", "rack1"]
    assert all(d["chips"] == 32 and d["nodes"] == 4 for d in cluster["domains"])


def test_fleet_shape_agrees_with_the_materialized_fleet():
    """The preview is arithmetic; the real tree is instantiation.  They
    must agree on every number the preview shows, or the block diagram is
    lying about what will run."""
    from fleetsim.config import load_scenario
    from fleetsim.fleet.build import build_fleet

    doc = tiny_doc(counts=(3, 2, 4), levels=("pod", "rack", "node"))
    shape = scenario_fleet_shape(yaml.safe_dump(doc))
    tree = build_fleet(load_scenario(doc))

    (cluster,) = shape["clusters"]
    assert cluster["map_level"] == "pod"
    assert cluster["chips"] == tree.total_chips("demo/c1")
    assert cluster["nodes"] == len(tree.leaves_under("demo/c1"))
    ids = [f"demo/c1/{d['short']}" for d in cluster["domains"]]
    assert ids == list(tree.domains_at_under("pod", "demo/c1"))
    for dom, dom_id in zip(cluster["domains"], ids):
        assert dom["chips"] == tree.total_chips(dom_id)
        assert dom["nodes"] == len(tree.leaves_under(dom_id))


def test_fleet_shape_is_arithmetic_not_materialized():
    """A fleet at the validation ceiling must preview instantly.  The
    guard is behavioural: building this tree would allocate 262,144
    Domain objects, which is exactly what the preview must not do."""
    doc = tiny_doc(counts=(64, 64, 64), levels=("pod", "rack", "node"))
    shape = scenario_fleet_shape(yaml.safe_dump(doc))
    assert shape["total_nodes"] == 64 * 64 * 64
    assert shape["total_chips"] == 64 * 64 * 64 * 8
    (cluster,) = shape["clusters"]
    assert cluster["n_domains"] == 64
    assert len(cluster["domains"]) == 64


def test_fleet_shape_truncates_and_says_so():
    doc = tiny_doc(counts=(MAX_PREVIEW_DOMAINS + 20, 1), levels=("rack", "node"))
    (cluster,) = scenario_fleet_shape(yaml.safe_dump(doc))["clusters"]
    assert cluster["n_domains"] == MAX_PREVIEW_DOMAINS + 20
    assert len(cluster["domains"]) == MAX_PREVIEW_DOMAINS
    assert cluster["domains_truncated"] is True
    assert MAX_PREVIEW_CLUSTERS >= 1


@pytest.mark.parametrize(
    "text",
    ["a: [unclosed", "just a string", "", "- 1\n- 2\n", "fleet: {}\n"],
)
def test_fleet_shape_never_raises_on_bad_input(text):
    shape = scenario_fleet_shape(text)
    assert shape is None or isinstance(shape, dict)


def test_fleet_shape_is_json_serializable():
    shape = scenario_fleet_shape(yaml.safe_dump(tiny_doc()))
    assert json.loads(json.dumps(shape)) == shape


# ---------------------------------------------------------------------------
# HTTP: /api/validate carries the shape; /api/validation feeds the tab
# ---------------------------------------------------------------------------


@pytest.fixture()
def served(tmp_path):
    manager = RunManager(tmp_path / "ws", start_worker=False)
    httpd = FleetsimHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], manager
    finally:
        httpd.shutdown()
        httpd.server_close()
        manager.shutdown(timeout=10.0)


def post(port, path, body, expect=200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == expect
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        assert exc.code == expect, exc.code
        return json.loads(exc.read())


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def test_validate_carries_the_fleet_shape_only_when_valid(served):
    port, _ = served
    out = post(port, "/api/validate", {"yaml": yaml.safe_dump(tiny_doc())})
    assert out["ok"] is True and out["errors"] == []
    assert out["fleet"]["total_chips"] == 64

    bad = post(port, "/api/validate", {"yaml": "a: [unclosed"})
    assert bad["ok"] is False and bad["errors"]
    # an invalid scenario has no shape to draw — the key is ABSENT, never
    # an empty fleet (which the preview would render as a real fleet)
    assert "fleet" not in bad


def test_validation_endpoint_feeds_the_tab(served):
    port, _ = served
    status, doc = get(port, "/api/validation")
    assert status == 200
    from fleetsim.validation.results import payload

    assert doc == payload()
    # every key the tab reads
    for key in (
        "version", "headline", "results", "groups", "counts", "ladder",
        "anti_goals", "placer_sweep", "citations", "doc",
    ):
        assert key in doc, key
    # the tab renders one panel per group and looks rows up by id
    by_id = {r["id"]: r for r in doc["results"]}
    for group in doc["groups"]:
        assert group["ids"], group["group"]
        for rid in group["ids"]:
            assert rid in by_id, rid
    # THE HONESTY RULES the renderer depends on
    for row in doc["results"]:
        if row["fleetsim"] is None:
            # unmeasured is never scored: it must not read as agreement
            assert row["in_band"] is None
            assert row["rel_error"] is None
        if row["band"] is None and row["tolerance"] is None:
            assert row["in_band"] is None, f"{row['id']} is reported, not asserted"
    assert doc["counts"]["measured"] <= doc["counts"]["total"]
    assert doc["counts"]["in_band"] <= doc["counts"]["asserted"]
    assert doc["anti_goals"], "the tab's honest note has nothing to show"


# ---------------------------------------------------------------------------
# queue positions stay contiguous
# ---------------------------------------------------------------------------


def test_queue_positions_are_contiguous_even_when_a_run_is_admitted(tmp_path):
    """``list_runs`` snapshots queue positions BEFORE walking the run
    directories, so a run admitted mid-walk used to leave a gap ([2, 3]
    with no 1).  Positions are renumbered against the rows actually
    observed as queued."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        text = yaml.safe_dump(tiny_doc())
        ids = [manager.submit(text, f"run {i}") for i in range(4)]
        rows = {r["id"]: r for r in manager.list_runs()}
        assert [rows[i]["queue_position"] for i in ids] == [1, 2, 3, 4]

        # simulate the race: the dispatcher takes the head of the queue and
        # flips its status while a listing is in flight
        with manager._lock:
            manager._queued.remove(ids[0])
        manager._set_status(ids[0], "running")
        rows = {r["id"]: r for r in manager.list_runs()}
        assert rows[ids[0]]["queue_position"] is None
        positions = sorted(
            r["queue_position"] for r in rows.values() if r["status"] == "queued"
        )
        assert positions == [1, 2, 3]
    finally:
        manager.shutdown(timeout=10.0)


# ---------------------------------------------------------------------------
# byte-compat across the version bump
# ---------------------------------------------------------------------------


def test_the_package_version_never_reaches_a_recorded_output(tmp_path):
    """v0.8 bumps ``fleetsim.__version__``.  That must not move a single
    recorded byte, or every release would silently invalidate every
    stored run.

    The property that makes it true: the version is stamped ONLY into the
    derived viz model (``meta.fleetsim_version``), never into
    ``summary.json`` / ``jobs.parquet`` / ``timeseries.parquet`` /
    ``stints.parquet``.  Asserted directly here — the same scenario run
    twice is also compared byte for byte, which pins the determinism the
    claim rests on.  (``tests/test_serve_live.py`` pins the other axis:
    the live machinery on vs off, over the shipped examples.)
    """
    import fleetsim
    from fleetsim import api
    from fleetsim.viz import build_viz_model

    doc = tiny_doc()
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        api.run_scenario(doc, out_dir=out)
        outs.append(out)

    produced = sorted(p.name for p in outs[0].iterdir() if p.is_file())
    assert "summary.json" in produced and "stints.parquet" in produced
    version = fleetsim.__version__.encode()
    for name in produced:
        first = (outs[0] / name).read_bytes()
        assert (outs[1] / name).read_bytes() == first, name
        if name == "scenario.yaml":
            continue  # the submitted text, copied verbatim
        assert version not in first, (
            f"{name} embeds the package version — a release bump would"
            f" rewrite every recorded run"
        )
    # ...and the version IS carried where it belongs: the derived model
    assert build_viz_model(outs[0])["meta"]["fleetsim_version"] == fleetsim.__version__


# ---------------------------------------------------------------------------
# the new static modules
# ---------------------------------------------------------------------------


def test_new_modules_keep_the_markup_hygiene_rules():
    for name in NEW_MODULES:
        js = (STATIC / name).read_text(encoding="utf-8")
        for banned in (".innerHTML", "document.write", "eval("):
            assert banned not in js, f"{name}: {banned}"


def test_new_modules_are_lazily_imported_by_the_shell():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for spec in ('import("./validation.js")', 'import("./live.js")',
                 'import("./fleetmap.js")', 'import("./export.js")'):
        assert spec in js, spec
    # every import in the shell is a relative, same-origin module path
    for match in re.finditer(r'import\(\s*"([^"]+)"', js):
        assert match.group(1).startswith("./"), match.group(1)


def test_live_module_mirrors_the_server_row_cap():
    """The client's live poll must honour the server's own cap constant —
    a client that assumed a bigger page would silently drop rows."""
    from fleetsim.serve.runs import LIVE_ROW_LIMIT

    js = (STATIC / "live.js").read_text(encoding="utf-8")
    assert "LIVE_MAX_ROWS" in js
    # the cursor contract's three load-bearing clauses are implemented
    assert "cursor=" in js
    assert "doc.more" in js
    assert "setOpen" in js and "addSettled" in js
    # an open stint becomes the model's own token, so the replay's sweep
    # needs no live-only branch
    assert '"running_at_horizon"' in js
    assert LIVE_ROW_LIMIT == 5000  # docs quote this number


def test_live_palette_constants_match_the_model():
    """A live pod and the same pod in the finished report must be the same
    color: the live stream carries class NAMES only, so the hexes are
    restated in JS and have to stay in step with viz.data."""
    from fleetsim.viz.data import _BUCKET_VARIANTS, _CLASS_COLORS, _STATE_COLORS

    js = (STATIC / "live.js").read_text(encoding="utf-8")
    block = js[js.index("export const CLASS_COLORS"):js.index("export const STATE_COLORS")]
    found = dict(re.findall(r"(\w+):\s*\"(#[0-9a-f]{6})\"", block))
    assert found == _CLASS_COLORS
    sblock = js[
        js.index("export const STATE_COLORS"):js.index("export const BUCKET_VARIANTS")
    ]
    sfound = dict(re.findall(r"(\w+):\s*\"(#[0-9a-f]{6})\"", sblock))
    assert sfound == {
        k: v for k, v in _STATE_COLORS.items() if v.startswith("#")
    }
    # the variants are per BUCKET on both sides, in the same pinned order
    vblock = js[
        js.index("export const BUCKET_VARIANTS"):js.index("export const BUCKET_ORDER")
    ]
    vfound = {
        name: re.findall(r"#[0-9a-f]{6}", body)
        for name, body in re.findall(r"(\w+):\s*\[([^\]]*)\]", vblock)
    }
    assert vfound == {k: list(v) for k, v in _BUCKET_VARIANTS.items()}


#: Harness: run live.js's palette assignment under node over a set of
#: (label, tier, job_class) rows and print the resulting palette.
_PALETTE_HARNESS = """\
import { LiveStints, livePalette } from "./live.mjs";

const req = JSON.parse(process.argv[2]);
const st = new LiveStints();
st.setFleet({ map_level: "node", clusters: [], domains: ["d0"] });
let i = 0;
for (const row of req.rows) {
  st.addSettled([{
    job_id: "j" + (i++), class_name: row[0], tier: row[1],
    job_class: row[2], domain: "d0", chips: 1, t0_us: 0, t1_us: 1,
    end_reason: "completed",
  }]);
}
process.stdout.write(JSON.stringify({
  buckets: st.buckets(),
  palette: livePalette(st.buckets()),
  classes: st.classes,
}));
"""


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is required to execute the live.js palette (CI runners ship it)",
)
@pytest.mark.parametrize(
    "rows",
    [
        # the review's exact case: two CUSTOM labels whose buckets are
        # unclaimed.  First-seen assignment from a flat variant list made
        # `gangs` periwinkle (a PRETRAIN shade) live and finetune green on
        # completion — the same work in two palette families.
        [("gangs", "BATCH", "FINETUNE"), ("mice", "BATCH", "EVAL")],
        # arrival order must not matter (the model assigns in sorted order)
        [("mice", "BATCH", "EVAL"), ("gangs", "BATCH", "FINETUNE")],
        # a canonical label CLAIMS its bucket: the custom one takes a shade
        [
            ("finetune", "BATCH", "FINETUNE"),
            ("gangs", "BATCH", "FINETUNE"),
            ("whales", "BATCH", "FINETUNE"),
        ],
        # tier BEST_EFFORT overrides the job class, on both sides
        [("scavenger", "BEST_EFFORT", "PRETRAIN"), ("big", "BATCH", "PRETRAIN")],
        # a label seen in two buckets takes the majority
        [
            ("mixed", "BATCH", "PRETRAIN"),
            ("mixed", "BATCH", "PRETRAIN"),
            ("mixed", "BATCH", "EVAL"),
        ],
        # a trace's uppercase enum labels: the model lowercases them
        [("EVAL", "BATCH", "EVAL"), ("PRETRAIN", "BATCH", "PRETRAIN")],
    ],
)
def test_live_palette_assignment_matches_the_model(rows, tmp_path):
    """THE PROPERTY, end to end: the color a class gets while the run is
    live is the color it gets when the run finishes.

    Pinning the two constant dicts was not enough — the ASSIGNMENT RULE
    is where they diverged.  This runs live.js's real assignment under
    node and compares it against ``viz.data._build_palette`` fed the
    equivalent jobs table.
    """
    import pandas as pd

    from fleetsim.viz.data import _build_palette, _label_buckets

    work = tmp_path / "js"
    work.mkdir()
    shutil.copy(STATIC / "live.js", work / "live.mjs")
    (work / "harness.mjs").write_text(_PALETTE_HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [shutil.which("node"), str(work / "harness.mjs"),
         json.dumps({"rows": rows})],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    live = json.loads(proc.stdout)

    jobs = pd.DataFrame(
        {
            "source_class": [r[0] for r in rows],
            "job_class": [r[2] for r in rows],
            "tier": [r[1] for r in rows],
        }
    )
    buckets = _label_buckets(jobs)
    model_palette = _build_palette(buckets, [])

    assert live["buckets"] == buckets
    for label in buckets:
        assert live["palette"][label] == model_palette[label], label


def test_fleet3d_exposes_live_drilldown_and_deep_links():
    js = (STATIC / "fleet3d.js").read_text(encoding="utf-8")
    # live replay rides live.js, not a second cursor implementation
    assert 'from "./live.js"' in js
    assert "applyLive" in js and "setFollowing" in js
    # the drill-down is a SECOND instanced mesh, never per-node objects
    assert js.count("new THREE.InstancedMesh") == 3
    assert "MAX_NODE_SLABS" in js
    # PNG export needs the drawing buffer kept
    assert "preserveDrawingBuffer: true" in js
    # deep links round-trip through one pair of functions
    assert "linkState()" in js and "applyDeepLink(" in js


def test_index_has_the_v08_phase3_anatomy():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for needle in (
        'id="view-validation"', 'id="validationBody"', 'id="valRung"',
        'id="validationMeta"', 'id="validationLink"',
        'id="liveMap"', 'id="liveMapMount"', 'id="liveMapSub"',
        'id="fleetPreview"', 'id="previewMount"', 'id="previewSub"',
    ):
        assert needle in html, needle


def test_app_js_routes_validation_and_decodes_deep_links():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '"validation"' in js
    assert "parseDeep" in js
    # every deep-link field the 3D view restores
    for field in ('"t"', '"cam"', '"pin"', '"x"', '"hide"'):
        assert field in js, field
