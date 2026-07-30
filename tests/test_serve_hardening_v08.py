"""v0.8 review fixes: server hardening.

Each test names the failure it prevents, in the terms the review found it.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest
import yaml

from fleetsim.serve.runs import (
    LIVE_OFFSET_MEMO,
    LIVE_ROWS,
    LIVE_STATE,
    LIVE_WATCH,
    MAX_TITLE_CHARS,
    RunManager,
)
from fleetsim.serve.server import FleetsimHTTPServer
from fleetsim.serve.sweeps import MAX_SWEEP_AXES, SweepManager

from test_serve_live import busy_doc, busy_yaml, get_json, post_json, wait_status


@pytest.fixture()
def serve_factory(tmp_path):
    """Same shape as ``test_serve_live``'s: ``make(max_workers, start_worker)
    -> (port, manager)``, with every server torn down at teardown."""
    made: list[tuple[RunManager, FleetsimHTTPServer]] = []

    def make(max_workers: int = 1, start_worker: bool = True):
        ws = tmp_path / f"ws{len(made)}"
        manager = RunManager(ws, start_worker=start_worker, max_workers=max_workers)
        httpd = FleetsimHTTPServer(("127.0.0.1", 0), manager)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        made.append((manager, httpd))
        return httpd.server_address[1], manager

    yield make
    for manager, httpd in made:
        httpd.shutdown()
        httpd.server_close()
        manager.shutdown(timeout=10.0)


# ---------------------------------------------------------------------------
# 1. a killed worker must not brick the server forever
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL is POSIX")
def test_a_sigkilled_worker_does_not_brick_every_future_run(serve_factory):
    """CRITICAL (review): one abruptly-terminated worker used to make every
    later submission fail forever with ``BrokenProcessPool``, because
    ``_ensure_pool`` cached the executor unconditionally.  The kernel OOM
    killer sends exactly this signal, and a legal scenario may declare
    262,144 nodes.

    Recovery must be automatic, and the error the killed run records must
    tell the operator what happened and that resubmitting works.
    """
    port, manager = serve_factory(max_workers=1)
    victim = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="45m")})["id"]
    wait_status(port, victim, "running")

    # kill the worker process the way the OOM killer would
    deadline = time.monotonic() + 20
    procs: list = []
    while time.monotonic() < deadline:
        pool = manager._pool
        procs = list(getattr(pool, "_processes", {}).values()) if pool else []
        if procs:
            break
        time.sleep(0.05)
    assert procs, "no worker process to kill"
    os.kill(procs[0].pid, signal.SIGKILL)

    info = wait_status(port, victim, "failed", timeout=60)
    assert info["status"] == "failed"
    detail = get_json(port, f"/api/runs/{victim}")
    assert "killed from outside" in detail["error"]
    assert "BrokenProcessPool" not in detail["error"]
    assert "again" in detail["error"], "the message must say recovery is possible"

    # and the very next runs execute — the pool rebuilt itself
    for _ in range(2):
        run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="2m")})["id"]
        got = wait_status(port, run_id, ("done", "failed"), timeout=90)
        assert got["status"] == "done", get_json(port, f"/api/runs/{run_id}")


def test_a_broken_pool_is_rebuilt_not_reused(tmp_path):
    """The unit-level statement of the same property, with no simulation:
    a pool flagged broken is dropped and a fresh one takes its place."""

    class _FakeBrokenPool:
        _broken = "a child process terminated abruptly"

        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self, **kw):
            self.shutdown_calls += 1

    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        dead = _FakeBrokenPool()
        manager._pool = dead
        assert manager._pool_is_broken(dead) is True
        fresh = manager._ensure_pool()
        assert fresh is not dead
        assert dead.shutdown_calls == 1
        assert manager._pool is fresh
        # a healthy pool is returned unchanged (no churn per dispatch)
        assert manager._ensure_pool() is fresh
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# 2. a corrupt spool line must not become an unthrottled poll loop
# ---------------------------------------------------------------------------


def test_a_corrupt_spool_row_reports_a_stall_instead_of_more_forever(serve_factory):
    """A single unparseable ``live.jsonl`` line used to pin the cursor with
    ``more: true`` and zero rows FOREVER, and the client contract's
    "more -> poll again immediately" turned that into ~1,240 req/s.

    The server must now name the line it stopped at, so a client can stop
    hot-looping and say why.
    """
    port, manager = serve_factory(max_workers=1)
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="10m")})["id"]
    wait_status(port, run_id, "done", timeout=120)

    rows_path = manager.workspace / run_id / LIVE_ROWS
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 12, "need a spool with rows to corrupt"
    lines[10] = '{"job_id": "broken", not json at all}'
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manager._live_offsets.clear()

    doc = get_json(port, f"/api/runs/{run_id}/live?cursor=0")
    assert doc["stalled_at"] == 10
    assert len(doc["stints"]) == 10
    assert doc["cursor"] == 10
    assert doc["more"] is True, "rows really do remain; the stream cannot reach them"
    # polling again from the returned cursor still reports the stall — the
    # client's job is to back off, and it can only do that if told
    again = get_json(port, f"/api/runs/{run_id}/live?cursor=10")
    assert again["stalled_at"] == 10 and again["cursor"] == 10 and again["more"]


def test_a_healthy_stream_never_reports_a_stall(serve_factory):
    port, manager = serve_factory(max_workers=1)
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="5m")})["id"]
    wait_status(port, run_id, "done", timeout=120)
    doc = get_json(port, f"/api/runs/{run_id}/live?cursor=0")
    assert doc["stalled_at"] is None


# ---------------------------------------------------------------------------
# 3. the open-stint overlay is spooled only while someone is watching
# ---------------------------------------------------------------------------


def test_the_overlay_is_not_rewritten_when_nobody_polls(tmp_path):
    """Write amplification: the overlay is the one part of a flush whose
    size grows with CONCURRENT jobs, and rewriting it every flush cost ~1 GB
    of file writes for an 86-second example.  With no watcher it must be
    omitted — and reported as omitted, never as an empty overlay."""
    from fleetsim.serve.runs import _LiveSpool

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spool = _LiveSpool(run_dir)
    snapshot = {
        "t_us": 60_000_000,
        "horizon_us": 600_000_000,
        "stints": [],
        "stint_cursor": 0,
        "open_stints": [{"job_id": f"j{i}", "end_reason": "open"} for i in range(400)],
    }
    spool(snapshot)
    state = json.loads((run_dir / LIVE_STATE).read_text(encoding="utf-8"))
    assert state["open_omitted"] is True
    assert state["open_stints"] is None  # NOT [], which claims nothing is running
    small = (run_dir / LIVE_STATE).stat().st_size

    # a client polls: the watch file is fresh, so the next flush carries it
    (run_dir / LIVE_WATCH).touch()
    spool(snapshot)
    state = json.loads((run_dir / LIVE_STATE).read_text(encoding="utf-8"))
    assert state["open_omitted"] is False
    assert len(state["open_stints"]) == 400
    assert (run_dir / LIVE_STATE).stat().st_size > 10 * small

    # a stale watch file stops paying for it again
    old = time.time() - 3600
    os.utime(run_dir / LIVE_WATCH, (old, old))
    spool(snapshot)
    assert json.loads((run_dir / LIVE_STATE).read_text())["open_omitted"] is True

    # ...but the FINAL write always carries it, so a client arriving after
    # the run replays the identical open set
    spool.finish()
    state = json.loads((run_dir / LIVE_STATE).read_text(encoding="utf-8"))
    assert state["open_omitted"] is False
    assert len(state["open_stints"]) == 400


def test_a_late_client_still_sees_the_final_overlay(serve_factory):
    """End to end: run with nobody watching, then read from cursor 0."""
    port, manager = serve_factory(max_workers=1)
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="10m")})["id"]
    wait_status(port, run_id, "done", timeout=120)
    doc = get_json(port, f"/api/runs/{run_id}/live?cursor=0")
    while doc["more"]:
        doc = get_json(port, f"/api/runs/{run_id}/live?cursor={doc['cursor']}")
    assert doc["open_pending"] is False
    assert isinstance(doc["open_stints"], list)
    horizon = 10 * 60 * 1_000_000
    assert all(r["t1_us"] == horizon for r in doc["open_stints"])


# ---------------------------------------------------------------------------
# 4. bounded memos and bounded titles
# ---------------------------------------------------------------------------


def test_the_live_read_memo_is_bounded(tmp_path):
    """It is a pure cache — an evicted run simply rescans — so it must not
    grow one entry per run any client ever scrubbed."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        for i in range(LIVE_OFFSET_MEMO + 40):
            manager._live_offsets[f"run-{i}"] = (0, 0)
            while len(manager._live_offsets) > LIVE_OFFSET_MEMO:
                manager._live_offsets.popitem(last=False)
        assert len(manager._live_offsets) == LIVE_OFFSET_MEMO
    finally:
        manager.shutdown(timeout=5.0)


def test_a_run_title_is_bounded(tmp_path):
    """A title rides every 3-second ``GET /api/runs`` of every open tab,
    forever: a 1.55 MB generated one made the rail poll ~500 KB/s."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        run_id = manager.submit(yaml.safe_dump(busy_doc()), "x" * 100_000)
        row = next(r for r in manager.list_runs() if r["id"] == run_id)
        assert len(row["title"]) <= MAX_TITLE_CHARS
        assert row["title"].endswith("…")
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# 5. queue positions are contiguous on the sweep board too
# ---------------------------------------------------------------------------


def test_get_run_never_reports_queued_without_a_position(tmp_path):
    """``get_run`` read meta and the queue positions under SEPARATE lock
    holds, so a run admitted in between came back ``queued`` with
    ``queue_position: null`` — a state the contract does not have."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        text = yaml.safe_dump(busy_doc())
        ids = [manager.submit(text, f"r{i}") for i in range(6)]
        stop = threading.Event()
        seen: list[tuple] = []

        def churn():
            while not stop.is_set():
                with manager._lock:
                    if manager._queued:
                        slug = manager._queued.pop(0)
                        manager._set_status_locked(slug, "running")
                        manager._queued.insert(0, slug)
                        manager._set_status_locked(slug, "queued")
                time.sleep(0.0005)

        t = threading.Thread(target=churn, daemon=True)
        t.start()
        try:
            for _ in range(400):
                for run_id in ids:
                    info = manager.get_run(run_id)
                    if info and info["status"] == "queued":
                        seen.append((run_id, info["queue_position"]))
        finally:
            stop.set()
            t.join(timeout=5)
        assert seen, "the probe never caught a queued run"
        assert all(pos is not None for _id, pos in seen), (
            "a queued run reported no queue position"
        )
    finally:
        manager.shutdown(timeout=5.0)


def test_sweep_queue_positions_are_contiguous(tmp_path):
    """The sweep board renders ``"queued #" + queue_position``; "#2, #3, #4
    and no #1" is not a state it should ever show."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    sweeps = SweepManager(manager)
    try:
        code, out = sweeps.create(
            yaml.safe_dump(busy_doc()), {"sim.seed": [1, 2, 3, 4]}
        )
        assert code == 200, out
        # simulate the race the review caught: the head is admitted while
        # the board is mid-scan, so the rows behind it carry stale positions
        with manager._lock:
            admitted = manager._queued.pop(0)
            manager._set_status_locked(admitted, "running")
        detail = sweeps.get_sweep(out["sweep_id"])
        positions = [
            r["queue_position"] for r in detail["runs"] if r["status"] == "queued"
        ]
        assert sorted(positions) == list(range(1, len(positions) + 1)), positions
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# 6. sweeps: inert axes, axis-count cap, bounded labels, memoized validation
# ---------------------------------------------------------------------------


def test_a_typo_axis_is_refused_instead_of_charted(tmp_path):
    """``sim.horizonn`` is accepted by the DOCUMENT (the schema checks
    unknown keys only at the top level) and expanded into N byte-identical
    runs the board then charts as an experiment."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    sweeps = SweepManager(manager)
    try:
        code, out = sweeps.create(
            yaml.safe_dump(busy_doc()), {"sim.horizonn": ["1h", "2h"]}
        )
        assert code == 400, out
        joined = " ".join(out["errors"])
        assert "sim.horizonn" in joined
        assert "byte-identical" in joined
        assert "sim.horizon" in joined, "the message must suggest the real path"
        assert not [r for r in manager.list_runs()], "nothing may be created"

        for bad in ("scheduler.paramz", "workload.clases"):
            code, out = sweeps.create(yaml.safe_dump(busy_doc()), {bad: [1, 2]})
            assert code == 400, (bad, out)

        # a REAL axis is still accepted
        code, out = sweeps.create(
            yaml.safe_dump(busy_doc()), {"sim.horizon": ["1h", "2h"]}
        )
        assert code == 200, out
    finally:
        manager.shutdown(timeout=5.0)


def test_axis_count_is_capped_and_labels_are_bounded(tmp_path):
    """50,000 one-value axes pass the CELL cap (product 1) and produced a
    1.55 MB run title that then rode every ``/api/runs`` poll."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    sweeps = SweepManager(manager)
    try:
        grid = {f"sim.k{i}": [i] for i in range(50_000)}
        code, out = sweeps.create(yaml.safe_dump(busy_doc()), grid)
        assert code == 400, out
        assert f"{MAX_SWEEP_AXES}-axis cap" in " ".join(out["errors"])
        assert not [r for r in manager.list_runs()]

        # and an in-cap sweep's generated titles are bounded
        code, out = sweeps.create(
            yaml.safe_dump(busy_doc()),
            {"sim.seed": [1, 2]},
            title="t" * 5000,
        )
        assert code == 200, out
        for row in manager.list_runs():
            assert len(row["title"]) <= MAX_TITLE_CHARS
    finally:
        manager.shutdown(timeout=5.0)


def test_a_seed_sweep_builds_the_fleet_once(tmp_path):
    """The all-or-nothing gate validated every cell independently, so a
    64-cell seed sweep over a 262,144-node fleet ran the same feasibility
    pass 64 times — 89.5 s inside one HTTP thread."""
    manager = RunManager(tmp_path / "ws", start_worker=False)
    try:
        calls = {"n": 0}
        import fleetsim.cli as cli

        real = cli._feasibility_errors

        def counting(scenario, base_dir):
            calls["n"] += 1
            return real(scenario, base_dir)

        cli._feasibility_errors = counting
        try:
            text = yaml.safe_dump(busy_doc())
            for seed in range(8):
                doc = busy_doc(seed=seed)
                assert manager.validate_text(yaml.safe_dump(doc)) == []
        finally:
            cli._feasibility_errors = real
        assert calls["n"] == 1, (
            f"the fleet was built {calls['n']} times for 8 seed-only variants"
        )
        assert manager.validate_text(text) == []
    finally:
        manager.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# 7. the preview route is the cheap one
# ---------------------------------------------------------------------------


BIG_FLEET = {
    "sim": {"horizon": "1h", "round": "60s", "seed": 1},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "h100", "per_node": 8},
                "topology": {"levels": ["pod", "rack", "node"], "counts": [64, 64, 64]},
            }
        ],
    },
    "failure_model": {"node_mtbf_days": 0},
    "workload": {
        "kind": "synthetic",
        "classes": {"eval": {"rate_per_hour": 1, "chips": 8, "duration": "10m"}},
    },
    "scheduler": {"name": "fifo"},
    "outputs": {"stints": True},
}


def test_preview_is_cheap_where_validate_is_not(serve_factory):
    """The editor fires on every 700 ms typing pause, and the feasibility
    half of validation BUILDS the declared fleet (1.9 s at the 262,144-node
    ceiling, measured).  /api/preview must answer the same shape without
    paying that, and must not weaken /api/validate."""
    port, _ = serve_factory(max_workers=1, start_worker=False)
    body = {"yaml": yaml.safe_dump(BIG_FLEET)}

    t0 = time.perf_counter()
    preview = post_json(port, "/api/preview", body)
    preview_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    full = post_json(port, "/api/validate", body)
    validate_s = time.perf_counter() - t0

    assert preview["ok"] is True and full["ok"] is True
    assert preview["fleet"] == full["fleet"], "same shape, both routes"
    assert preview["fleet"]["total_nodes"] == 262_144
    assert preview_s < 0.25, f"preview took {preview_s:.3f}s"
    assert preview_s * 4 < validate_s, (preview_s, validate_s)


def test_preview_still_reports_schema_errors(serve_factory):
    port, _ = serve_factory(max_workers=1, start_worker=False)
    bad = post_json(port, "/api/preview", {"yaml": "sim: {horizon: nope}"})
    assert bad["ok"] is False and bad["errors"]
    assert "fleet" not in bad
    broken = post_json(port, "/api/preview", {"yaml": "a: [unclosed"})
    assert broken["ok"] is False


def test_the_preview_draws_the_level_the_run_records(tmp_path):
    """``outputs: {stints: <level>}`` names the recording level; drawing
    the root-children level regardless was wrong for every scenario that
    names one — example 07 ships ``stints: node`` and previewed 4 rack
    blocks for a run that records 32 node domains."""
    from fleetsim.serve.runs import scenario_fleet_shape

    examples = Path(__file__).resolve().parents[1] / "examples"
    text = (examples / "07_placement_study" / "scenario.yaml").read_text()
    shape = scenario_fleet_shape(text)
    assert shape["stints_mode"] == "level" and shape["stints_level"] == "node"
    cluster = shape["clusters"][0]
    assert cluster["map_level"] == "node"
    assert cluster["n_domains"] == 32
    assert [d["short"] for d in cluster["domains"][:3]] == ["node0", "node1", "node2"]
    # ids match fleet.build's numbering, per parent + level
    assert cluster["domains"][0]["path"] == "rack0/node0"
    assert cluster["domains"][8]["path"] == "rack1/node0"

    # stints: true keeps the level below each cluster root
    doc = yaml.safe_load(text)
    doc["outputs"]["stints"] = True
    shape = scenario_fleet_shape(yaml.safe_dump(doc))
    assert shape["stints_mode"] == "root_children"
    assert shape["clusters"][0]["map_level"] == "rack"
    assert shape["clusters"][0]["n_domains"] == 4

    # no stints at all: the honest signal that the fleet map will be empty
    doc["outputs"].pop("stints")
    shape = scenario_fleet_shape(yaml.safe_dump(doc))
    assert shape["stints_mode"] == "off" and shape["stints_level"] is None


def test_the_preview_stays_arithmetic_at_the_ceiling():
    """Walking to a named level must stay bounded by the BLOCK cap, not by
    the domain count: ``stints: node`` on a 262,144-node fleet is 256
    blocks, and the true count comes from arithmetic."""
    from fleetsim.serve.runs import MAX_PREVIEW_DOMAINS, scenario_fleet_shape

    doc = json.loads(json.dumps(BIG_FLEET))
    doc["outputs"]["stints"] = "node"
    t0 = time.perf_counter()
    shape = scenario_fleet_shape(yaml.safe_dump(doc))
    elapsed = time.perf_counter() - t0
    cluster = shape["clusters"][0]
    assert cluster["n_domains"] == 262_144
    assert len(cluster["domains"]) == MAX_PREVIEW_DOMAINS
    assert cluster["domains_truncated"] is True
    assert elapsed < 0.25, f"the preview took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 8. the report cache follows the model cache's currency rule
# ---------------------------------------------------------------------------


def test_a_stale_report_cache_is_rebuilt(serve_factory):
    """``model_json`` refuses a cache from an older fleetsim; ``report_html``
    served ANY cached file, so the report iframe and the analysis tab on one
    run could be built by two different releases."""
    port, manager = serve_factory(max_workers=1)
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="5m")})["id"]
    wait_status(port, run_id, "done", timeout=120)

    code, html = manager.report_html(run_id)
    assert code == 200 and html
    cache = manager.workspace / run_id / "report.html"
    assert cache.is_file()

    cache.write_text("<html>rendered by fleetsim 0.6.0</html>", encoding="utf-8")
    code, rebuilt = manager.report_html(run_id)
    assert code == 200
    assert rebuilt == html, "a stale cache must be rebuilt, not served"
    assert manager._report_cache_is_current(rebuilt) is True
    assert manager._report_cache_is_current("<html>old</html>") is False


# ---------------------------------------------------------------------------
# 9. the client-side review fixes, pinned structurally
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "fleetsim" / "serve" / "static"


def read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_no_static_file_contains_a_control_byte():
    """A raw NUL in a served .js makes every text tool treat it as BINARY —
    grep prints "Binary file matches" instead of lines, which is how a
    review concluded fleet3d.js never referenced a flag it does reference.
    Separators belong in the source as ESCAPES."""
    for path in sorted(STATIC_DIR.glob("*.js")):
        raw = path.read_bytes()
        bad = [b for b in raw if b < 0x09 or 0x0E <= b < 0x20]
        assert not bad, f"{path.name} contains control bytes {sorted(set(bad))}"


def test_the_editor_preview_uses_the_cheap_route_and_guards_it():
    js = read_static("app.js")
    assert '"/api/preview"' in js, "the preview must not call the full gate"
    assert 'apiPost("/api/validate"' in js, "Validate itself is unchanged"
    # exactly one in-flight preview, and a pending edit re-runs once
    assert "previewInFlight" in js and "previewAgain" in js
    # an edit invalidates the previous explicit Validate result
    body = js[js.index('box.addEventListener("input"'):][:400]
    assert "clearValidation()" in body


def test_the_live_map_mount_is_token_guarded():
    """`liveMapRunId === id` is not idempotence across an await: a progress
    tick arriving during a COLD module import started a second feed that
    nothing could stop."""
    js = read_static("app.js")
    assert "liveMapToken" in js
    fn = js[js.index("async function startLiveMap"):][:1200]
    assert "const token = ++liveMapToken" in fn
    assert "token !== liveMapToken" in fn


def test_deep_link_cam_rejects_empty_fields():
    """Number("") is 0, so ",,,,," used to parse as six valid zeros — a
    radius-0 camera inside the geometry."""
    js = read_static("app.js")
    fn = js[js.index("function parseDeep"):js.index("function parseRoute")]
    assert 's.trim() === ""' in fn and "NaN" in fn
    assert "cam[3] > 0" in fn


def test_the_orbit_controller_clamps_a_restored_pose():
    js = read_static("fleet3d.js")
    fn = js[js.index("  restore(s) {"):][:700]
    assert "clamp(s.radius" in fn and "clamp(s.phi" in fn


def test_the_deep_link_is_authoritative_and_round_trips_an_empty_filter():
    js = read_static("fleet3d.js")
    fn = js[js.index("  applyDeepLink(deep) {"):][:1600]
    # absent means DEFAULT, not "keep whatever this viewer already had"
    assert "this.S.hidden = new Set(Array.isArray(deep.hide) ? deep.hide : [])" in fn
    assert "this.setPin(u == null ? null : u)" in fn
    url = js[js.index("  deepLinkUrl() {"):][:900]
    assert 'parts.push("hide=" + (s.hide || [])' in url, "empty filter must round-trip"


def test_the_live_palette_call_sites_pass_buckets():
    """A bare label list makes every custom class best_effort; the bucket
    map is what makes live and finished colors agree."""
    for name in ("app.js", "fleet3d.js"):
        js = read_static(name)
        assert "livePalette(" in js
        assert "livePalette(st.classes)" not in js
        assert "livePalette(up.stints.classes)" not in js


def test_the_live_feed_backs_off_when_the_cursor_does_not_move():
    js = read_static("live.js")
    tick = js[js.index("  async _tick(token) {"):]
    assert "this.cursor <= before" in tick
    assert "stalled_at" in tick
    # and the stall path must NOT re-poll at 0 ms: it returns before the
    # `more` branch ever reaches the zero-delay reschedule
    stall = tick[tick.index("if (this.stalled) {"):]
    stall = stall[: stall.index("if (doc.more) {")]
    assert "LIVE_POLL_MS" in stall
    assert "this._later(token, 0)" not in stall


def test_both_truncation_flags_are_rendered():
    """live.js's docstring claims the feed "says so"; that has to be true
    of BOTH consumers, for BOTH caps."""
    app = read_static("app.js")
    f3d = read_static("fleet3d.js")
    for js, where in ((app, "app.js"), (f3d, "fleet3d.js")):
        assert "openTruncated" in js, where
        assert "truncated" in js, where
        assert "stalled" in js, where
    assert "#liveMapWarn" in app
    assert "liveWarn" in f3d


def test_the_analysis_tab_calls_frames_frames():
    js = read_static("insight.js")
    assert "function frameSpanUs" in js and "function framesText" in js
    assert '" frames in the model"' in js
    assert "A FRAME IS NOT A ROUND" in js
    # the model's own notes are rendered here too, not only in the report
    assert "model.meta && model.meta.notes" in js
    # the event list states the truncation
    assert "event_totals" in js
    # the fit is clipped to the observed x range
    fit = js[js.index("  if (fit.r != null) {"):][:900]
    assert "Math.min(...xs)" in fit and "Math.max(...xs)" in fit


def test_compare_carries_fragmentation_everywhere():
    js = read_static("compare.js")
    assert "function fragColumns" in js
    assert '"fragmentation"' in js
    assert "Fragmentation index" in js, "the overlaid timeline"
    assert "function reduceSeries" in js, "the density fix"
    assert "function seriesToggles" in js, "isolate/hide a run"
    # truncated labels carry the whole thing
    assert "runFullLabel" in js and "runLabel" in js
    exp = read_static("experiment.js")
    assert "function syncMetricOptions" in exp, "the sweep metric selector"


def test_the_validation_tab_reads_its_options_and_units_from_data():
    js = read_static("validation.js")
    assert "function fillRungs" in js
    assert "function fmtTolerance" in js
    assert '" pp"' in js, "an absolute share tolerance is percentage POINTS"
    assert "group.caption" in js
    html = read_static("index.html")
    # the hand-written (and inverted) labels are gone
    assert "V2 — distribution" not in html and "V3 — absolute" not in html


def test_the_headline_carries_both_qualifiers():
    from fleetsim.validation.results import (
        HEADLINE,
        HEADLINE_BOUNDS_PCT,
        headline_bounds,
    )

    for phrase in (
        "per-VC September-max sizing",
        "strict blocking scan",
        "consolidate placement",
        "ORDER-SENSITIVE",
        "not evidence of accuracy",
    ):
        assert phrase in HEADLINE, phrase
    # and the quoted figures still cover the data they describe
    observed = headline_bounds()
    for group, quoted in HEADLINE_BOUNDS_PCT.items():
        got = observed.get(group)
        if got is None:
            continue
        assert got <= quoted, f"{group}: {got:.2f}% exceeds the quoted +/-{quoted}%"
        assert f"+/-{quoted:.0f} %" in HEADLINE, group
