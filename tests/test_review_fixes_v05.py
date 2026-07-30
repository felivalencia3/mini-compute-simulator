"""Regression guards for the v0.5 (web app) review fixes.

Each test pins one reviewed defect on the server side:

- ONE SERVER PER WORKSPACE: a second live ``RunManager`` on the same
  workspace must refuse to start (``WorkspaceLockError``) instead of
  "repairing" the first server's queued/running meta to ``failed``
  while those runs are still executing.  Stale locks (dead pid or
  garbage content) are reclaimed; shutdown releases the lock.
- VALIDATION MESSAGE HYGIENE: a relative ``workload.source`` fails web
  validation (web runs execute from a fresh run directory), but the
  error must not leak the internal ``_pending-run`` anchor path and
  must state the absolute-path remedy.
- REPORT PATH PRIVACY: the downloadable ``report.html`` / viz model
  JSON must show the run id as its display path, never the operator's
  absolute workspace path (usernames, machine layout travel with a
  shared file otherwise).

The client-side fixes from the same review (URIError-safe hash routing,
live reduced-motion, rail timestamp refresh, tab semantics, rail empty
note, shell intensity) are DOM/WebGL behavior verified in the browser;
the static-file invariants they rely on live in ``test_serve_static``.
"""

from __future__ import annotations

import json
import os

import pytest
import yaml

from fleetsim import api
from fleetsim.serve.runs import RunManager, WorkspaceLockError
from fleetsim.serve.server import serve

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def tiny_doc():
    """A seconds-fast scenario with stints on (same as test_serve)."""
    return {
        "sim": {"horizon": "10m", "round": "30s", "seed": 5},
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
        "failure_model": {
            "node_mtbf_days": 0,
            "maintenance_rate_per_node_month": 0,
        },
        "workload": {
            "kind": "synthetic",
            "classes": {
                "eval": {
                    "rate_per_hour": 60,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=1m, p90=5m]",
                }
            },
        },
        "scheduler": {"name": "fifo"},
        "outputs": {"stints": True},
    }


def tiny_yaml() -> str:
    return yaml.safe_dump(tiny_doc())


# ---------------------------------------------------------------------------
# one live server per workspace (.serve.lock)
# ---------------------------------------------------------------------------


def test_second_manager_on_same_workspace_refuses(tmp_path):
    ws = tmp_path / "ws"
    first = RunManager(ws, start_worker=False)
    try:
        with pytest.raises(WorkspaceLockError) as exc:
            RunManager(ws, start_worker=False)
        # the error tells the operator whose lock it is and what to do
        assert str(os.getpid()) in str(exc.value)
        assert ".serve.lock" in str(exc.value)
    finally:
        first.shutdown(timeout=5.0)
    # shutdown released the lock: a fresh server can now own the workspace
    assert not (ws / ".serve.lock").exists()
    third = RunManager(ws, start_worker=False)
    third.shutdown(timeout=5.0)


def test_lock_prevents_boot_repair_of_live_runs(tmp_path):
    """The exact corruption from the review: server B's boot repair must
    NOT flip server A's still-executing run to failed."""
    ws = tmp_path / "ws"
    first = RunManager(ws, start_worker=False)
    try:
        slug = first.submit(tiny_yaml())
        meta_path = ws / slug / "meta.json"
        # simulate the run being actively executed by server A
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = "running"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(WorkspaceLockError):
            RunManager(ws, start_worker=False)

        # the live run's meta is untouched — no false "failed"
        meta_after = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta_after["status"] == "running"
        assert "error" not in meta_after
    finally:
        first.shutdown(timeout=5.0)


def test_stale_lock_with_garbage_content_is_reclaimed(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".serve.lock").write_text("not-a-pid\n", encoding="utf-8")
    manager = RunManager(ws, start_worker=False)  # must not raise
    try:
        assert (
            ws / ".serve.lock"
        ).read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        manager.shutdown(timeout=5.0)


def _noop() -> None:  # multiprocessing target: must be module-level
    return None


@pytest.mark.skipif(
    os.name != "posix", reason="pid liveness probe is POSIX-only"
)
def test_stale_lock_with_dead_pid_is_reclaimed(tmp_path):
    import multiprocessing

    proc = multiprocessing.Process(target=_noop)
    proc.start()
    proc.join()
    dead_pid = proc.pid  # reaped by join(): os.kill(pid, 0) now fails
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".serve.lock").write_text(f"{dead_pid}\n", encoding="utf-8")
    manager = RunManager(ws, start_worker=False)  # reclaims, must not raise
    manager.shutdown(timeout=5.0)


def test_serve_exits_2_on_locked_workspace(tmp_path, capsys):
    ws = tmp_path / "ws"
    owner = RunManager(ws, start_worker=False)
    try:
        code = serve(port=0, workspace=ws)
        assert code == 2
        out = capsys.readouterr().out
        assert "error:" in out
        assert "already owned" in out
    finally:
        owner.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# web validation message hygiene (relative trace source)
# ---------------------------------------------------------------------------


def test_relative_trace_error_hides_pending_run_anchor(tmp_path):
    doc = tiny_doc()
    doc["workload"] = {"kind": "trace", "source": "sample_trace.csv"}
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        errors = manager.validate_text(yaml.safe_dump(doc))
    finally:
        manager.shutdown(timeout=5.0)
    assert errors, "a relative trace source must fail web validation"
    joined = "\n".join(errors)
    # the internal anchor directory never reaches the user...
    assert "_pending-run" not in joined
    assert str(tmp_path) not in joined
    # ...the scenario's own path and the actual remedy do
    assert "sample_trace.csv" in joined
    assert "absolute path" in joined


# ---------------------------------------------------------------------------
# report / model display path privacy
# ---------------------------------------------------------------------------


def test_model_and_report_do_not_embed_workspace_path(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    api.run_scenario(tiny_doc(), out_dir=ws / "cli-drop")
    manager = RunManager(ws, start_worker=False)
    try:
        code, payload = manager.model_json("cli-drop")
        assert code == 200
        model = json.loads(payload)
        assert model["meta"]["out_dir"] == "cli-drop"
        assert str(ws) not in payload

        code, html = manager.report_html("cli-drop")
        assert code == 200
        assert str(ws) not in html
        assert "cli-drop" in html
        # the disk caches carry the same sanitized display path
        cached = json.loads(
            (ws / "cli-drop" / "viz_model.json").read_text(encoding="utf-8")
        )
        assert cached["meta"]["out_dir"] == "cli-drop"
    finally:
        manager.shutdown(timeout=5.0)
