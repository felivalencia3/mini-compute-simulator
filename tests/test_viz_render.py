"""Tests for the v0.3 single-file HTML renderer (fleetsim.viz.render).

Phase contract: the packaged template carries the injection token;
``render_html`` produces valid standalone HTML with the model inlined
as ``const DATA = {...};``; the page makes zero external requests (no
``http(s)://`` and no ``url(...)`` references anywhere); the injected
DATA parses back as JSON equal to the model.  The model here is a small
hand-built synthetic one — no simulation runs in this phase.

The ``_client_core`` tests execute the template's own replay-core JS
(carved verbatim from the rendered page) under node, pinning the
client-side review fixes; they skip when node is not on PATH.
"""

import json
import re
import shutil
import subprocess

import pytest

from fleetsim.viz import render_html
from fleetsim.viz.render import _DATA_TOKEN, _TEMPLATE_PATH, _TITLE_TOKEN

S = 1_000_000

_CLASS_COLORS = {
    "pretrain": "#4c6ef5",
    "finetune": "#12b886",
    "eval": "#fab005",
    "best_effort": "#64748b",
    "inference": "#9775fa",
}
_STATE_COLORS = {
    "failed": "#e03131",
    "draining": "#f76707",
    "maintenance": "#845ef7",
    "idle": "rgba(255,255,255,.05)",
}


def _frames(n=4):
    return {
        "t_us": [(i + 1) * 60 * S for i in range(n)],
        "occupancy": [0.5, 0.75, None, 1.0][:n],
        "allocation": [0.5, 0.7, 0.8, 0.9][:n],
        "goodput_to_date": [1.0, 0.9, 0.85, 0.8][:n],
        "pending_by_class": {"finetune": [1, 0, 2, 1][:n], "pretrain": [0, 1, 1, 0][:n]},
        "preemptions_delta": [0, 2, 0, 1][:n],
        "failures_delta": [0, 0, 1, 0][:n],
    }


def _cards(occ="97.0%"):
    return [
        {"label": "occupancy", "value": occ, "sub": "steady-state window"},
        {"label": "goodput", "value": "91.2%", "sub": "steady-state window"},
        {"label": "jobs finished", "value": "12", "sub": "full run"},
        {"label": "preemptions/min", "value": "0.10", "sub": "window, all triggers"},
        {"label": "finetune wait p50/p99", "value": "2.0m / 8.5m", "sub": "n=9, window"},
    ]


def _model(**over):
    """A tiny schema-complete synthetic model (built by hand)."""
    model = {
        "meta": {
            "title": "fleetsim replay — synthetic <A&B>",
            "out_dir": "/runs/out-a",
            "horizon_us": 600 * S,
            "round_us": 60 * S,
            "seed": 42,
            "scenario_name": "tiny",
            "fleetsim_version": "0.1.0",
            "generated_unix_ms": None,
            "notes": ["synthetic model for the render tests"],
        },
        "capabilities": {"map": True, "compare": False},
        "palette": {**_CLASS_COLORS, **_STATE_COLORS},
        "fleet": {
            "map_level": "pod",
            "clusters": [
                {
                    "id": "m/c",
                    "chips": 32,
                    "domains": [
                        {"id": "m/c/pod0", "short": "pod0", "chips": 16},
                        {"id": "m/c/pod1", "short": "pod1", "chips": 16},
                    ],
                }
            ],
        },
        "frames": _frames(),
        "stints": {
            "job_id": ["j1", "j1", "j2", "j3"],
            "class_name": ["pretrain", "pretrain", "finetune", "finetune"],
            "tier": ["prod", "prod", "batch", "batch"],
            "domain_idx": [0, 1, 1, 0],
            "chips": [16, 8, 8, 8],
            "t0_us": [60 * S, 60 * S, 120 * S, 300 * S],
            "t1_us": [240 * S, 240 * S, 180 * S, 600 * S],
            "end_reason": ["preempted", "preempted", "failed", "running_at_horizon"],
        },
        "gantt": [
            {
                "id": "j1",
                "class_name": "pretrain",
                "chips": 24,
                "submit_us": 0,
                "start_us": 60 * S,
                "end_us": 480 * S,
                "status": "completed",
                "n_preemptions": 1,
                "n_restarts": 1,
                "domains_spanned": 2,
            },
            {
                "id": "j2",
                "class_name": "finetune",
                "chips": 8,
                "submit_us": 30 * S,
                "start_us": 120 * S,
                "end_us": 180 * S,
                "status": "failed",
                "n_preemptions": 0,
                "n_restarts": 0,
                "domains_spanned": 1,
            },
            {
                "id": "j3",
                "class_name": "finetune",
                "chips": 8,
                "submit_us": 250 * S,
                "start_us": 300 * S,
                "end_us": None,
                "status": "running",
                "n_preemptions": 0,
                "n_restarts": 0,
                "domains_spanned": 1,
            },
        ],
        "cdfs": {
            "queue_wait_s": {
                "finetune": [[30.0, 0.5], [90.0, 1.0]],
                "pretrain": [[60.0, 1.0]],
            },
            "jct_s": {"pretrain": [[480.0, 1.0]]},
        },
        "events": [
            {"t_us": 120 * S, "kind": "failure", "label": "node failures: 1", "magnitude": 1},
            {
                "t_us": 240 * S,
                "kind": "preemption_wave",
                "label": "preemption wave: 12 jobs",
                "magnitude": 12,
            },
        ],
        "summary_cards": _cards(),
        "compare": None,
    }
    model.update(over)
    return model


def _extract_data(doc: str) -> dict:
    """The injected DATA payload, parsed back from the rendered page."""
    m = re.search(r"^const DATA = (.*);$", doc, re.M)
    assert m, "rendered page has no `const DATA = ...;` line"
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Template invariants
# ---------------------------------------------------------------------------


def test_template_carries_injection_tokens():
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    assert text.count(_DATA_TOKEN) == 1
    assert _TITLE_TOKEN in text
    assert f"<title>{_TITLE_TOKEN}</title>" in text


def test_template_and_rendered_page_reference_no_external_urls():
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = render_html(_model())
    for text in (template, rendered):
        low = text.lower()
        assert "http://" not in low
        assert "https://" not in low
        assert "url(" not in low  # no CSS url refs at all, remote or otherwise
        assert "@import" not in low
        assert "<link" not in low  # no stylesheets/fonts/preloads
        assert "fetch(" not in low and "xmlhttprequest" not in low


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


def test_render_produces_standalone_html(tmp_path):
    model = _model()
    doc = render_html(model)
    assert doc.lstrip().startswith("<!DOCTYPE html")
    assert doc.rstrip().endswith("</html>")
    assert _DATA_TOKEN not in doc
    assert _TITLE_TOKEN not in doc
    # the <title> is the model title, HTML-escaped
    assert "<title>fleetsim replay — synthetic &lt;A&amp;B&gt;</title>" in doc
    # footer inputs are present verbatim in the payload
    assert "/runs/out-a" in doc
    # tiny harness page: written once, loadable as a plain file
    page = tmp_path / "report.html"
    page.write_text(doc, encoding="utf-8")
    assert page.stat().st_size > 10_000


def test_injected_data_parses_and_round_trips():
    model = _model()
    doc = render_html(model)
    assert _extract_data(doc) == model


def test_render_is_deterministic():
    assert render_html(_model()) == render_html(_model())


def test_hostile_labels_cannot_break_the_script_block():
    model = _model()
    model["gantt"][0]["id"] = 'j</script><script>alert("x")</script>'
    model["meta"]["notes"] = ["<!-- sneaky --> note"]
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    doc = render_html(model)
    # the injection adds no script-terminators and no comment-openers
    assert doc.count("</script>") == template.count("</script>")
    assert doc.count("<script") == template.count("<script")
    assert "<!--" not in doc
    # and the payload still decodes to the exact model
    assert _extract_data(doc) == model


def test_degraded_no_map_model_renders():
    model = _model(
        capabilities={"map": False, "compare": False},
        fleet={"map_level": None, "clusters": []},
        stints={
            "job_id": [],
            "class_name": [],
            "tier": [],
            "domain_idx": [],
            "chips": [],
            "t0_us": [],
            "t1_us": [],
            "end_reason": [],
        },
    )
    doc = render_html(model)
    assert _extract_data(doc) == model


def test_compare_model_renders():
    model = _model(
        capabilities={"map": True, "compare": True},
        compare={
            "label_a": "out-a",
            "label_b": "out-b",
            "frames_b": _frames(3),
            "summary_cards_b": _cards(occ="88.8%"),
        },
    )
    doc = render_html(model)
    assert _extract_data(doc) == model


def test_render_rejects_non_finite_floats():
    model = _model()
    model["frames"]["allocation"][0] = float("inf")
    with pytest.raises(ValueError):
        render_html(model)


def test_missing_title_falls_back():
    model = _model()
    model["meta"]["title"] = None
    doc = render_html(model)
    assert "<title>fleetsim replay</title>" in doc


# ---------------------------------------------------------------------------
# Client replay core, executed verbatim under node (review-fix pins)
# ---------------------------------------------------------------------------

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not on PATH"
)


def _client_core(model, epilogue, tmp_path):
    """Run the rendered page's replay core (constants through seek())
    verbatim in node, followed by ``epilogue``; returns parsed JSON
    printed by the epilogue's ``console.log``."""
    doc = render_html(model)
    start = doc.index("const US = 1e6;")
    end = doc.index("function nearestFrame")
    seg = doc[start:end]
    assert "function seek(T)" in seg, "replay core carve markers moved"
    harness = (
        "'use strict';\n"
        "const window = { devicePixelRatio: 1 };\n"
        "const document = { querySelector: () => null, addEventListener: () => {} };\n"
        f"const M = {json.dumps(model)};\n"
        f"{seg}\n{epilogue}\n"
    )
    js = tmp_path / "client_core.js"
    js.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(js)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@needs_node
def test_client_seek_keeps_running_at_horizon_stints_at_horizon(tmp_path):
    """Map/aggregate agreement at T = horizon (review fix): stints
    truncated with end_reason=running_at_horizon (t1_us == horizon) must
    still be shown as allocated when the playhead rests at the horizon —
    releasing them there contradicts the final timeseries flush."""
    model = _model()
    out = _client_core(
        model,
        "seek(M.meta.horizon_us);\n"
        "let total = 0;\n"
        "for (let i = 0; i < cur.occ.length; i++) total += cur.occ[i];\n"
        "console.log(JSON.stringify({ total, active: cur.active.size }));",
        tmp_path,
    )
    # j3 (8 chips, running_at_horizon) is the only stint open at T=HOR.
    assert out == {"total": 8, "active": 1}


@needs_node
def test_client_labels_include_compare_b_only_classes(tmp_path):
    """Review fix: a class present only in run B's pending_by_class must
    appear in the client label set (it feeds the pending chart's dashed
    B overlay and stack totals)."""
    frames_b = _frames(3)
    frames_b["pending_by_class"] = {"mystery": [2, 1, 0]}
    model = _model(
        capabilities={"map": True, "compare": True},
        compare={
            "label_a": "out-a",
            "label_b": "out-b",
            "frames_b": frames_b,
            "summary_cards_b": _cards(occ="88.8%"),
        },
    )
    out = _client_core(
        model, "console.log(JSON.stringify(LABELS));", tmp_path
    )
    assert "mystery" in out and "pretrain" in out and "finetune" in out
