"""Regression guards for the v0.5 server-hardening review fixes.

Each test pins one reviewed defect:

- CACHE-BUILD RACE: concurrent cold-cache model/report requests must all
  succeed (unique temp names + a per-run build lock), never 500 with a
  ``FileNotFoundError`` on a shared ``.tmp`` rename.
- HOST-HEADER PIN (anti DNS-rebinding): a request whose Host is not a
  loopback authority answers 421, never data.
- CSRF: a cross-origin ``Origin`` / cross-site ``Sec-Fetch-Site`` on
  state-changing routes is 403; POST bodies must be application/json
  (text/plain would be a CORS-"simple" no-preflight request).
- VALIDATION BOMB: a tiny YAML declaring a huge fleet is a validation
  error computed arithmetically, not a minutes-long ``build_fleet``.
- 500 HYGIENE: internal errors expose the exception class only — never
  the message (which can embed the operator's filesystem paths).
- JSON EVERYWHERE: unsupported methods and malformed request lines get
  the JSON envelope + hardening headers, not stdlib HTML pages.
- NUL BYTES: percent-encoded NULs in run ids / static paths are 404s.
- KEEP-ALIVE DRAIN: an unread request body never desyncs the next
  request on the same connection.
- NEGATIVE Content-Length is a 400, not a 413.
- CANCEL: a running run can be cancelled cooperatively; the run is
  marked failed with ``cancelled by request``.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from fleetsim import api
from fleetsim.cli import _MAX_FLEET_CHIPS, _MAX_FLEET_NODES
from fleetsim.serve.runs import RunManager
from fleetsim.serve.server import CSP_APP, CSP_REPORT, FleetsimHTTPServer

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def tiny_doc(**sim_over):
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


@pytest.fixture()
def served(tmp_path):
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


def request(port, method, path, body=None, headers=None):
    """(status, headers, bytes) via raw http.client, no error raising."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        hdrs = {"Host": f"127.0.0.1:{port}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def wait_for_status(port, run_id, want, timeout=60.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, _, body = request(port, "GET", f"/api/runs/{run_id}/progress")
        assert status == 200
        last = json.loads(body)
        if last["status"] == want:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run never reached {want!r}; last: {last}")


# ---------------------------------------------------------------------------
# cache-build race (atomic write + per-run lock)
# ---------------------------------------------------------------------------


def test_concurrent_cold_cache_model_and_report_never_500(served):
    port, _, workspace = served
    status, _, body = request(
        port, "POST", "/api/runs", {"yaml": tiny_yaml(), "title": "race"}
    )
    assert status == 200
    run_id = json.loads(body)["id"]
    wait_for_status(port, run_id, "done")

    def hit(sub):
        st, _, payload = request(port, "GET", f"/api/runs/{run_id}/{sub}")
        return st, payload

    for round_no in range(3):  # repeat: the race window is the cold build
        for cache in ("viz_model.json", "report.html"):
            (workspace / run_id / cache).unlink(missing_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(hit, "model") for _ in range(4)]
            futures += [pool.submit(hit, "report") for _ in range(4)]
            results = [f.result() for f in futures]
        for st, payload in results:
            assert st == 200, (round_no, st, payload[:200])
        # all model responses byte-identical (one build won; readers saw it)
        models = {p for st, p in results[:4]}
        assert len(models) == 1


def test_atomic_write_temp_names_are_unique(tmp_path):
    from fleetsim.serve.runs import _atomic_write_text

    target = tmp_path / "out.json"
    _atomic_write_text(target, "one")
    _atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    # no shared "<name>.tmp" and no leftover temp litter
    assert list(tmp_path.iterdir()) == [target]


# ---------------------------------------------------------------------------
# Host-header pin (DNS rebinding)
# ---------------------------------------------------------------------------


def test_foreign_or_empty_host_is_421(served):
    port, _, _ = served
    for host in (
        "evil.example.com",
        f"attacker.test:{port}",
        f"127.0.0.1.nip.io:{port}",
        "",
    ):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            conn.putrequest("GET", "/api/runs", skip_host=True)
            conn.putheader("Host", host)
            conn.putheader("Connection", "close")
            conn.endheaders()
            resp = conn.getresponse()
            body = resp.read()
            assert resp.status == 421, host
            assert "error" in json.loads(body)
        finally:
            conn.close()


def test_loopback_hosts_accepted(served):
    port, _, _ = served
    for host in (f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"):
        status, _, body = request(port, "GET", "/api/runs", headers={"Host": host})
        assert status == 200, host
        assert json.loads(body) == []


# ---------------------------------------------------------------------------
# CSRF: Origin / Sec-Fetch-Site / Content-Type
# ---------------------------------------------------------------------------


def test_cross_origin_post_is_403(served_no_worker):
    port, _, _ = served_no_worker
    for origin in ("https://evil.example.com", "http://evil.test:1234", "null"):
        status, _, body = request(
            port,
            "POST",
            "/api/validate",
            {"yaml": "x"},
            headers={"Origin": origin},
        )
        assert status == 403, origin
        assert "error" in json.loads(body)
    # DELETE too
    status, _, _ = request(
        port,
        "DELETE",
        "/api/runs/nope",
        headers={"Origin": "https://evil.example.com"},
    )
    assert status == 403


def test_same_origin_post_is_accepted(served_no_worker):
    port, _, _ = served_no_worker
    status, _, body = request(
        port,
        "POST",
        "/api/validate",
        {"yaml": tiny_yaml()},
        headers={
            "Origin": f"http://127.0.0.1:{port}",
            "Sec-Fetch-Site": "same-origin",
        },
    )
    assert status == 200
    assert json.loads(body)["ok"] is True


def test_cross_site_sec_fetch_site_is_403(served_no_worker):
    port, _, _ = served_no_worker
    status, _, _ = request(
        port,
        "POST",
        "/api/validate",
        {"yaml": "x"},
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert status == 403


def test_text_plain_post_is_415(served_no_worker):
    # text/plain POSTs are CORS-"simple" (no preflight): rejecting the
    # content type outright closes the no-preflight CSRF channel.
    port, _, _ = served_no_worker
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            "POST",
            "/api/runs",
            body=json.dumps({"yaml": tiny_yaml()}).encode("utf-8"),
            headers={
                "Host": f"127.0.0.1:{port}",
                "Content-Type": "text/plain;charset=UTF-8",
            },
        )
        resp = conn.getresponse()
        assert resp.status == 415
        assert "error" in json.loads(resp.read())
    finally:
        conn.close()
    # and nothing was created
    status, _, body = request(port, "GET", "/api/runs")
    assert status == 200 and json.loads(body) == []


# ---------------------------------------------------------------------------
# validation bomb: fleet size capped arithmetically
# ---------------------------------------------------------------------------


def test_huge_declared_fleet_is_validation_error_and_fast(served_no_worker):
    port, _, _ = served_no_worker
    doc = tiny_doc()
    doc["fleet"]["clusters"][0]["topology"] = {
        "levels": ["rack", "node"],
        "counts": [65536, 65536],  # 4.3 BILLION declared nodes
    }
    t0 = time.monotonic()
    status, _, body = request(
        port, "POST", "/api/validate", {"yaml": yaml.safe_dump(doc)}
    )
    elapsed = time.monotonic() - t0
    assert status == 200
    out = json.loads(body)
    assert out["ok"] is False
    assert any("too large" in e for e in out["errors"])
    assert elapsed < 5.0, f"validation took {elapsed:.1f}s — fleet was materialized"
    # submitting it is rejected the same way
    status, _, body = request(
        port, "POST", "/api/runs", {"yaml": yaml.safe_dump(doc)}
    )
    assert status == 400


def test_fleet_ceiling_admits_the_frontier_example():
    # 04_frontier declares 65,536 nodes / 524,288 chips — well inside.
    assert 65_536 * 4 <= _MAX_FLEET_NODES
    assert 524_288 * 8 <= _MAX_FLEET_CHIPS


def test_cli_validate_rejects_huge_fleet(tmp_path):
    from fleetsim.cli import main

    doc = tiny_doc()
    doc["fleet"]["clusters"][0]["topology"] = {
        "levels": ["rack", "node"],
        "counts": [65536, 65536],
    }
    path = tmp_path / "huge.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    t0 = time.monotonic()
    assert main(["validate", str(path)]) == 1
    assert time.monotonic() - t0 < 5.0


# ---------------------------------------------------------------------------
# 500 hygiene: exception class only, no paths
# ---------------------------------------------------------------------------


def test_500_exposes_class_but_not_message(served, monkeypatch, capsys):
    port, manager, workspace = served

    def boom(run_id):
        raise FileNotFoundError(f"{workspace}/secret/place.tmp")

    monkeypatch.setattr(manager, "model_json", boom)
    api.run_scenario(tiny_doc(), out_dir=workspace / "cli-drop")
    status, _, body = request(port, "GET", "/api/runs/cli-drop/model")
    assert status == 500
    payload = json.loads(body)
    assert payload["error"] == "internal error: FileNotFoundError"
    assert str(workspace) not in body.decode("utf-8")


def test_model_build_error_scrubs_workspace_path(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    run = ws / "cli-drop"
    api.run_scenario(tiny_doc(), out_dir=run)
    (run / "timeseries.parquet").unlink()  # break the build input
    manager = RunManager(ws, start_worker=False)
    try:
        code, msg = manager.model_json("cli-drop")
        assert code == 500
        assert str(ws) not in msg
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# JSON everywhere: no stdlib HTML error pages
# ---------------------------------------------------------------------------


def test_unsupported_methods_are_json_405_with_allow(served_no_worker):
    port, _, _ = served_no_worker
    for method in ("PUT", "PATCH", "OPTIONS"):
        status, headers, body = request(port, method, "/api/runs")
        assert status == 405, method
        assert headers["Content-Type"].startswith("application/json")
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Cache-Control"] == "no-store"
        assert "GET" in headers["Allow"]
        assert "error" in json.loads(body)
        # OPTIONS must NOT grant CORS (preflights have to fail)
        assert not any(k.lower().startswith("access-control-") for k in headers)


def test_unknown_method_and_bad_request_line_are_json(served_no_worker):
    port, _, _ = served_no_worker
    # unknown method -> stdlib 501 path, now JSON with hardening headers
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("FROB", "/api/runs", headers={"Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        headers = dict(resp.getheaders())
        body = resp.read()
        assert resp.status == 501
        assert headers["Content-Type"].startswith("application/json")
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert b"<html" not in body.lower()
        assert "error" in json.loads(body)
    finally:
        conn.close()
    # garbage request line -> 400 JSON
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(b"garbage\r\n\r\n")
        raw = sock.makefile("rb").read()
    head, _, tail = raw.partition(b"\r\n\r\n")
    assert b" 400 " in head.split(b"\r\n")[0]
    assert b"<html" not in raw.lower()
    assert "error" in json.loads(tail)
    # bad HTTP version -> 505 JSON
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(b"GET / HTTP/9.9\r\nHost: x\r\n\r\n")
        raw = sock.makefile("rb").read()
    assert b" 505 " in raw.split(b"\r\n")[0]
    assert b"<html" not in raw.lower()


# ---------------------------------------------------------------------------
# NUL bytes: 404, not 500
# ---------------------------------------------------------------------------


def test_percent_encoded_nul_is_404(served_no_worker):
    port, _, _ = served_no_worker
    for path in (
        "/api/runs/run%00x",
        "/api/runs/a%00b/progress",
        "/static/app%00.js",
        "/static/vendor/%00",
    ):
        status, _, body = request(port, "GET", path)
        assert status == 404, path
        assert "error" in json.loads(body)


# ---------------------------------------------------------------------------
# keep-alive: unread bodies are drained
# ---------------------------------------------------------------------------


def test_unread_body_does_not_desync_keep_alive(served_no_worker):
    port, _, _ = served_no_worker
    shapes = (
        ("POST", "/api/nope", b'{"yaml": "x"}'),
        ("DELETE", "/api/runs/nope", b'{"yaml": "x"}'),
        ("GET", "/api/runs", b'{"yaml": "x"}'),
    )
    for method, path, payload in shapes:
        with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
            req1 = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n"
            ).encode("ascii") + payload
            req2 = (
                f"GET /api/runs HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
            sock.sendall(req1 + req2)
            raw = sock.makefile("rb").read()
        statuses = re.findall(rb"HTTP/1\.1 (\d{3})", raw)
        assert b"400" not in statuses, (method, path, statuses)
        assert len(statuses) == 2, (method, path, statuses)
        assert statuses[-1] == b"200", (method, path, statuses)


def test_negative_content_length_is_400(served_no_worker):
    port, _, _ = served_no_worker
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(
            (
                f"POST /api/validate HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: -1\r\n\r\n"
            ).encode("ascii")
        )
        raw = sock.makefile("rb").read()
    assert b" 400 " in raw.split(b"\r\n")[0]
    assert b"invalid Content-Length" in raw


# ---------------------------------------------------------------------------
# cooperative cancel of a running run
# ---------------------------------------------------------------------------


def test_cancel_running_run(served):
    port, _, workspace = served
    # long horizon, short rounds: many flushes to intercept
    status, _, body = request(
        port,
        "POST",
        "/api/runs",
        {"yaml": tiny_yaml(horizon="30d", round="60s"), "title": "runaway"},
    )
    assert status == 200
    run_id = json.loads(body)["id"]
    wait_for_status(port, run_id, "running")
    # queued-only DELETE still refuses (immutability unchanged) ...
    status, _, _ = request(port, "DELETE", f"/api/runs/{run_id}")
    assert status == 409
    # ... but cancel stops it at the next metrics flush
    status, _, body = request(port, "POST", f"/api/runs/{run_id}/cancel")
    assert status == 200 and json.loads(body) == {"ok": True}
    wait_for_status(port, run_id, "failed", timeout=30.0)
    status, _, body = request(port, "GET", f"/api/runs/{run_id}")
    detail = json.loads(body)
    assert detail["error"] == "cancelled by request"


def test_cancel_non_running_states(served_no_worker):
    port, _, _ = served_no_worker
    status, _, body = request(port, "POST", "/api/runs", {"yaml": tiny_yaml()})
    run_id = json.loads(body)["id"]
    # queued -> 409 pointing at DELETE
    status, _, body = request(port, "POST", f"/api/runs/{run_id}/cancel")
    assert status == 409
    assert "DELETE" in json.loads(body)["error"]
    # unknown -> 404
    status, _, _ = request(port, "POST", "/api/runs/nope/cancel")
    assert status == 404


# ---------------------------------------------------------------------------
# report title uses the submit-time title
# ---------------------------------------------------------------------------


def test_report_title_uses_run_title(served):
    port, _, _ = served
    status, _, body = request(
        port,
        "POST",
        "/api/runs",
        {"yaml": tiny_yaml(), "title": "review journey: smoke"},
    )
    run_id = json.loads(body)["id"]
    wait_for_status(port, run_id, "done")
    status, _, body = request(port, "GET", f"/api/runs/{run_id}/model")
    assert status == 200
    model = json.loads(body)
    assert model["meta"]["title"] == "fleetsim replay — review journey: smoke"
    assert model["meta"]["out_dir"] == run_id  # the slug stays reachable
    status, _, body = request(port, "GET", f"/api/runs/{run_id}/report")
    assert status == 200
    assert "review journey: smoke" in body.decode("utf-8")


# ---------------------------------------------------------------------------
# CSP: frame-ancestors pinned
# ---------------------------------------------------------------------------


def test_csp_pins_frame_ancestors():
    assert "frame-ancestors 'self'" in CSP_APP
    assert "frame-ancestors 'self'" in CSP_REPORT
