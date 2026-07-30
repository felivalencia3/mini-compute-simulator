"""Tests for `fleetsim serve` (v0.5 server core): the engine progress
hook, the RunManager lifecycle, and the HTTP route contract.

The server is started on port 0 in a thread; requests go through
urllib (well-formed paths) and raw http.client (hostile paths that a
URL library would normalize away, e.g. ``/api/runs/../../etc``).
Scenarios are tiny (2 nodes, minutes of sim time) so a full
validate -> submit -> progress -> model -> report lifecycle runs in
seconds.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import yaml

from fleetsim import api
from fleetsim.cli import build_parser
from fleetsim.serve.runs import RunManager
from fleetsim.serve.server import CSP_APP, CSP_REPORT, FleetsimHTTPServer

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def tiny_doc(**sim_over):
    """A seconds-fast scenario with stints on (exercises the viz map)."""
    sim = {"horizon": "10m", "round": "30s", "seed": 5}
    sim.update(sim_over)
    return {
        "sim": sim,
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


def tiny_yaml(**sim_over) -> str:
    return yaml.safe_dump(tiny_doc(**sim_over))


def get_json(port: int, path: str, expect: int = 200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == expect
            return json.loads(resp.read().decode("utf-8")), dict(resp.headers)
    except urllib.error.HTTPError as err:
        assert err.code == expect, f"{path}: got {err.code}, want {expect}"
        return json.loads(err.read().decode("utf-8")), dict(err.headers)


def request_json(port: int, method: str, path: str, body=None, expect: int = 200):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == expect
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        assert err.code == expect, f"{method} {path}: got {err.code}, want {expect}"
        return json.loads(err.read().decode("utf-8"))


def raw_get(port: int, raw_path: str) -> tuple[int, bytes]:
    """GET with the path sent VERBATIM (no client-side normalization)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.putrequest("GET", raw_path, skip_host=True)
        conn.putheader("Host", f"127.0.0.1:{port}")
        conn.putheader("Connection", "close")
        conn.endheaders()
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def wait_for_status(port: int, run_id: str, want: str, timeout: float = 60.0):
    """Poll the progress endpoint until the run reaches ``want``."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last, _ = get_json(port, f"/api/runs/{run_id}/progress")
        if last["status"] == want:
            return last
        assert last["status"] != "failed" or want == "failed", (
            f"run failed while waiting for {want}: "
            f"{get_json(port, f'/api/runs/{run_id}')[0]}"
        )
        time.sleep(0.05)
    raise AssertionError(f"run never reached {want!r}; last: {last}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def served(tmp_path):
    """(port, manager, workspace) for a live server with a worker."""
    manager = RunManager(tmp_path / "ws")
    httpd = FleetsimHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], manager, manager.workspace
    finally:
        httpd.shutdown()
        httpd.server_close()
        manager.shutdown(timeout=10.0)


@pytest.fixture()
def served_no_worker(tmp_path):
    """Same, but runs stay queued (worker never starts)."""
    manager = RunManager(tmp_path / "ws2", start_worker=False)
    httpd = FleetsimHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], manager, manager.workspace
    finally:
        httpd.shutdown()
        httpd.server_close()
        manager.shutdown(timeout=10.0)


# ---------------------------------------------------------------------------
# engine progress hook
# ---------------------------------------------------------------------------

_SNAPSHOT_KEYS = {
    "t_us",
    "horizon_us",
    "jobs_finished",
    "jobs_running",
    "pending",
    "occupancy_to_date",
    "allocated_chips",
    "healthy_chips",
}


def test_progress_cb_snapshots_shape_and_order():
    snaps: list[dict] = []
    api.run_scenario(tiny_doc(), progress_cb=snaps.append)
    assert snaps, "no progress snapshots emitted"
    horizon = 10 * 60 * 1_000_000
    for s in snaps:
        assert set(s) == _SNAPSHOT_KEYS
        assert s["horizon_us"] == horizon
        assert isinstance(s["jobs_finished"], int)
        assert isinstance(s["jobs_running"], int)
        assert isinstance(s["pending"], int)
        # sink is a MetricsCollector -> the sampled fields are filled
        assert s["allocated_chips"] is not None
        assert s["healthy_chips"] == 16
        assert 0.0 <= s["occupancy_to_date"] <= 1.0
    times = [s["t_us"] for s in snaps]
    assert times == sorted(times)
    assert times[-1] == horizon  # final flush at exactly the horizon
    # one snapshot per METRICS_FLUSH: round=30s over 10m -> 19 interior
    # flushes + the final one
    assert len(snaps) == 20


def test_progress_cb_none_is_byte_compatible():
    a = api.run_scenario(tiny_doc())
    b = api.run_scenario(tiny_doc(), progress_cb=lambda s: None)
    assert a == b


def test_progress_cb_exception_aborts_run():
    class Boom(Exception):
        pass

    def cb(s):
        raise Boom()

    with pytest.raises(Boom):
        api.run_scenario(tiny_doc(), progress_cb=cb)


# ---------------------------------------------------------------------------
# static shell + CSP
# ---------------------------------------------------------------------------


def test_root_serves_index_with_pinned_csp(served):
    port, _, _ = served
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        assert resp.headers["Content-Security-Policy"] == CSP_APP
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        body = resp.read().decode("utf-8")
    assert "fleetsim" in body


def test_static_file_served_and_traversal_blocked(served):
    port, _, _ = served
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/static/index.html", timeout=30
    ) as resp:
        assert resp.status == 200
    # Raw traversal attempts, plain and percent-encoded: never anything
    # but 404, and never a file from outside the static root.
    for path in (
        "/static/../server.py",
        "/static/../../api.py",
        "/static/..%2F..%2Fserver.py",
        "/static/%2e%2e/server.py",
        "/static/..",
    ):
        status, body = raw_get(port, path)
        assert status == 404, path
        assert b"http.server" not in body  # JSON error, not a stdlib page


# ---------------------------------------------------------------------------
# validate endpoint
# ---------------------------------------------------------------------------


def test_validate_ok_and_errors(served):
    port, _, _ = served
    out = request_json(port, "POST", "/api/validate", {"yaml": tiny_yaml()})
    assert out["ok"] is True and out["errors"] == []
    # v0.8, additive: a VALID scenario also carries the fleet it describes,
    # which is what the editor's shape preview draws
    assert set(out) == {"ok", "errors", "fleet"}
    assert out["fleet"]["total_chips"] > 0

    out = request_json(port, "POST", "/api/validate", {"yaml": "just a string"})
    assert out["ok"] is False and out["errors"]

    out = request_json(port, "POST", "/api/validate", {"yaml": "a: [unclosed"})
    assert out["ok"] is False
    assert any("YAML" in e for e in out["errors"])

    doc = tiny_doc()
    doc["scheduler"]["name"] = "no_such_scheduler"
    out = request_json(
        port, "POST", "/api/validate", {"yaml": yaml.safe_dump(doc)}
    )
    assert out["ok"] is False and out["errors"]

    # malformed request envelope -> 400 JSON
    out = request_json(port, "POST", "/api/validate", {"nope": 1}, expect=400)
    assert "error" in out


# ---------------------------------------------------------------------------
# run lifecycle over HTTP
# ---------------------------------------------------------------------------


def test_run_lifecycle(served):
    port, _, workspace = served
    out = request_json(
        port, "POST", "/api/runs", {"yaml": tiny_yaml(), "title": "smoke"}
    )
    run_id = out["id"]
    assert set(out) == {"id"}
    assert (workspace / run_id).is_dir()
    assert (workspace / run_id / "scenario.yaml").read_text(
        encoding="utf-8"
    ) == tiny_yaml()

    final = wait_for_status(port, run_id, "done")
    prog = final["progress"]
    assert prog is not None
    assert prog["t_us"] == prog["horizon_us"]
    assert prog["healthy_chips"] == 16

    # listing carries the headline
    listing, _ = get_json(port, "/api/runs")
    row = next(r for r in listing if r["id"] == run_id)
    assert row["title"] == "smoke"
    assert row["status"] == "done"
    assert isinstance(row["created"], int)
    hl = row["headline"]
    # the three pinned keys, plus one frag.<level> per level the run
    # recorded (v0.8: the sweep board's placement metrics)
    assert {"occupancy", "goodput", "jobs_finished"} <= set(hl)
    assert all(
        k in ("occupancy", "goodput", "jobs_finished") or k.startswith("frag.")
        for k in hl
    ), sorted(hl)
    assert isinstance(hl["jobs_finished"], int)

    # detail carries the full summary
    detail, _ = get_json(port, f"/api/runs/{run_id}")
    assert detail["status"] == "done"
    assert detail["summary"]["horizon_us"] == 10 * 60 * 1_000_000

    # outputs landed inside the run dir (out_dir forced by the server)
    for name in ("summary.json", "jobs.parquet", "timeseries.parquet", "meta.json"):
        assert (workspace / run_id / name).is_file(), name

    # model: JSON, pinned top-level keys, cached to disk
    model, headers = get_json(port, f"/api/runs/{run_id}/model")
    assert headers["Content-Type"].startswith("application/json")
    for key in ("meta", "frames", "stints", "gantt", "summary_cards"):
        assert key in model, key
    assert (workspace / run_id / "viz_model.json").is_file()

    # report: HTML with the report CSP, cached to disk
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/runs/{run_id}/report", timeout=30
    ) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        assert resp.headers["Content-Security-Policy"] == CSP_REPORT
        html = resp.read().decode("utf-8")
    assert "const DATA = " in html
    assert (workspace / run_id / "report.html").is_file()

    # done runs cannot be deleted through the API
    out = request_json(port, "DELETE", f"/api/runs/{run_id}", expect=409)
    assert "error" in out


def test_post_invalid_yaml_is_400(served):
    port, _, workspace = served
    out = request_json(
        port, "POST", "/api/runs", {"yaml": "not: [valid"}, expect=400
    )
    assert out["ok"] is False and out["errors"]
    # nothing was created
    listing, _ = get_json(port, "/api/runs")
    assert listing == []


def test_model_and_report_conflict_before_done(served_no_worker):
    port, _, _ = served_no_worker
    out = request_json(port, "POST", "/api/runs", {"yaml": tiny_yaml()})
    run_id = out["id"]
    for sub in ("model", "report"):
        body = request_json(port, "GET", f"/api/runs/{run_id}/{sub}", expect=409)
        assert "error" in body


# ---------------------------------------------------------------------------
# run-id traversal and unknown ids
# ---------------------------------------------------------------------------


def test_run_id_traversal_is_404_and_stays_in_workspace(served, tmp_path):
    port, _, workspace = served
    # a real file OUTSIDE the workspace that traversal would love to hit
    secret = workspace.parent / "secret"
    secret.mkdir(exist_ok=True)
    (secret / "summary.json").write_text("{}", encoding="utf-8")
    for path in (
        "/api/runs/../../etc",
        "/api/runs/../secret",
        "/api/runs/..%2F..%2Fetc",
        "/api/runs/..%2Fsecret/model",
        "/api/runs/../secret/progress",
        "/api/runs/%2e%2e",
        "/api/runs/..",
    ):
        status, body = raw_get(port, path)
        assert status == 404, path
        payload = json.loads(body)
        assert "error" in payload
    # plain unknown id
    get_json(port, "/api/runs/run-19990101-000000-000-zzzz", expect=404)
    request_json(
        port, "DELETE", "/api/runs/run-19990101-000000-000-zzzz", expect=404
    )


def test_dotted_and_separator_ids_rejected_by_manager(tmp_path):
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        for bad in ("", ".", "..", "a/b", "a\\b", ".hidden", "../ws"):
            assert manager.resolve_dir(bad) is None, bad
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# DELETE (dequeue) semantics
# ---------------------------------------------------------------------------


def test_delete_queued_run(served_no_worker):
    port, _, workspace = served_no_worker
    out = request_json(port, "POST", "/api/runs", {"yaml": tiny_yaml()})
    run_id = out["id"]
    prog, _ = get_json(port, f"/api/runs/{run_id}/progress")
    # v0.8: the progress payload also carries the queued run's FIFO
    # position (1-based), so a poller needs one request, not two.
    assert prog == {"status": "queued", "progress": None, "queue_position": 1}

    assert request_json(port, "DELETE", f"/api/runs/{run_id}") == {"ok": True}
    assert not (workspace / run_id).exists()
    listing, _ = get_json(port, "/api/runs")
    assert listing == []
    # a second delete is a 404 (it is gone)
    request_json(port, "DELETE", f"/api/runs/{run_id}", expect=404)


# ---------------------------------------------------------------------------
# external run detection
# ---------------------------------------------------------------------------


def test_external_cli_run_shows_up(served):
    port, _, workspace = served
    api.run_scenario(tiny_doc(), out_dir=workspace / "cli-drop")
    listing, _ = get_json(port, "/api/runs")
    row = next(r for r in listing if r["id"] == "cli-drop")
    assert row["status"] == "done"
    assert row["title"] == "cli-drop"
    assert row["headline"] is not None

    detail, _ = get_json(port, "/api/runs/cli-drop")
    assert detail["summary"] is not None
    prog, _ = get_json(port, "/api/runs/cli-drop/progress")
    assert prog == {
        "status": "done",
        "progress": None,  # external runs spool nothing
        "queue_position": None,
    }
    # model/report work off summary.json alone
    model, _ = get_json(port, "/api/runs/cli-drop/model")
    assert "frames" in model


# ---------------------------------------------------------------------------
# RunManager unit behavior
# ---------------------------------------------------------------------------


def test_slugs_are_server_shaped_and_unique(tmp_path):
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        import re

        ids = {manager.submit(tiny_yaml()) for _ in range(5)}
        assert len(ids) == 5
        for slug in ids:
            assert re.fullmatch(
                r"run-\d{8}-\d{6}-\d{3}-[a-z0-9]{4}", slug
            ), slug
            assert (manager.workspace / slug).is_dir()
    finally:
        manager.shutdown(timeout=5.0)


def test_boot_repair_marks_stale_runs_failed(tmp_path):
    ws = tmp_path / "ws"
    manager = RunManager(ws, start_worker=False)
    slug = manager.submit(tiny_yaml())
    manager.shutdown(timeout=5.0)
    # shutdown marks the still-queued run failed
    meta = json.loads((ws / slug / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"

    # simulate a crash: force meta back to "running", then boot again
    meta["status"] = "running"
    (ws / slug / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manager2 = RunManager(ws, start_worker=False)
    try:
        meta2 = json.loads(
            (ws / slug / "meta.json").read_text(encoding="utf-8")
        )
        assert meta2["status"] == "failed"
        assert "restart" in meta2["error"]
    finally:
        manager2.shutdown(timeout=5.0)


def test_validate_catches_missing_trace_file(tmp_path):
    # trace workload pointing at a missing ABSOLUTE file passes the
    # schema; the feasibility pass still rejects it up front.
    doc = tiny_doc()
    doc["workload"] = {
        "kind": "trace",
        "source": str(tmp_path / "missing-trace.csv"),
    }
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        errors = manager.validate_text(yaml.safe_dump(doc))
        assert errors  # feasibility catches the missing trace up front
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_serve_parser_defaults():
    args = build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.port == 8500
    assert args.workspace == "./fleetsim-runs"
    assert args.host == "127.0.0.1"
    assert args.open_browser is False
    args = build_parser().parse_args(
        ["serve", "-p", "9000", "--workspace", "w", "--host", "0.0.0.0", "--open"]
    )
    assert (args.port, args.workspace, args.host, args.open_browser) == (
        9000,
        "w",
        "0.0.0.0",
        True,
    )


# ---------------------------------------------------------------------------
# terminal hygiene: client disconnects are not errors
# ---------------------------------------------------------------------------


def test_handle_error_suppresses_client_disconnect_noise(tmp_path, capsys):
    """A browser closing a keep-alive socket mid-read (refresh, tab
    close, poll abort) must not dump a traceback to the operator's
    terminal — but real bugs still must."""
    manager = RunManager(tmp_path / "ws3", start_worker=False)
    httpd = FleetsimHTTPServer(("127.0.0.1", 0), manager)
    try:
        try:
            raise ConnectionResetError(54, "Connection reset by peer")
        except ConnectionResetError:
            httpd.handle_error(None, ("127.0.0.1", 12345))
        assert capsys.readouterr().err == ""  # silent: normal churn

        try:
            raise ValueError("a real bug")
        except ValueError:
            httpd.handle_error(None, ("127.0.0.1", 12345))
        assert "ValueError" in capsys.readouterr().err  # still surfaced
    finally:
        httpd.server_close()
        manager.shutdown(timeout=5.0)
