"""Tests for the v0.8 server core: PARALLEL worker processes and the LIVE
replay stream.

Two contracts are load-bearing here and both are asserted end to end:

1. **The cursor contract.** A client that walks ``/api/runs/{id}/live``
   from ``cursor=0`` to the end of a run must reassemble EXACTLY the
   ``stints.parquet`` the finished run writes — no duplicated row, no
   missing row, no revised row.  ``test_live_stream_reassembles_stints_parquet``
   compares the streamed union (settled rows + the final open overlay)
   against the parquet, as multisets.
2. **Parallelism changes nothing about a run.**  A run is a pure function
   of ``(scenario, seed)``, so moving execution into a worker process must
   leave outputs byte-identical — asserted by running the same sweep twice
   and comparing every output file byte for byte.

Scenarios are seconds-fast (2 nodes, 10-20 min of sim time) so the whole
module runs in CI time; nothing here downloads or simulates at scale.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fleetsim import api
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.serve.runs import (
    LIVE_ROWS,
    LIVE_STATE,
    RunManager,
    default_max_workers,
)
from fleetsim.serve.server import FleetsimHTTPServer

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Column set of one stint row, in both the live stream and stints.parquet.
STINT_COLUMNS = (
    "job_id",
    "class_name",
    "job_class",
    "tier",
    "domain",
    "chips",
    "t0_us",
    "t1_us",
    "end_reason",
)


def busy_doc(*, stints=True, horizon="20m", seed=5, rate=900):
    """A seconds-fast but CONTENDED scenario: 2 nodes of 8 chips against a
    900 job/hour eval stream, so jobs queue, start, and settle many times
    over — the live stream needs stint churn to be worth asserting on."""
    doc = {
        "sim": {"horizon": horizon, "round": "30s", "seed": seed},
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
                    "rate_per_hour": rate,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=1m, p90=4m]",
                }
            },
        },
        "scheduler": {"name": "fifo"},
        "outputs": {},
    }
    if stints:
        doc["outputs"]["stints"] = True
    return doc


def busy_yaml(**kw) -> str:
    return yaml.safe_dump(busy_doc(**kw))


def get_json(port: int, path: str, expect: int = 200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == expect
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        assert err.code == expect, f"{path}: got {err.code}, want {expect}"
        return json.loads(err.read().decode("utf-8"))


def post_json(port: int, path: str, body, expect: int = 200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == expect
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        assert err.code == expect, f"{path}: got {err.code}, want {expect}"
        return json.loads(err.read().decode("utf-8"))


def normalize(row: dict) -> tuple:
    """One stint row as a comparable tuple (parquet dtypes -> Python)."""
    return (
        str(row["job_id"]),
        str(row["class_name"]),
        str(row["job_class"]),
        str(row["tier"]),
        str(row["domain"]),
        int(row["chips"]),
        int(row["t0_us"]),
        int(row["t1_us"]),
        str(row["end_reason"]),
    )


def parquet_rows(run_dir: Path) -> list[tuple]:
    df = pd.read_parquet(run_dir / "stints.parquet")
    assert list(df.columns) == list(STINT_COLUMNS), list(df.columns)
    return sorted(normalize(r) for r in df.to_dict("records"))


def wait_status(port: int, run_id: str, want, timeout: float = 120.0) -> dict:
    """Poll until the run's status is in ``want``."""
    targets = {want} if isinstance(want, str) else set(want)
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = get_json(port, f"/api/runs/{run_id}/progress")
        if last["status"] in targets:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run never reached {targets}; last={last}")


def drain_live(port: int, run_id: str, timeout: float = 120.0) -> dict:
    """Walk ``/live`` from cursor 0 to the end of the run.

    Returns ``{rows, open_final, cursor, fleet, n_requests, statuses}``.
    Asserts the cursor invariants on every hop: never decreases, never
    jumps past what was delivered, and ``fleet`` rides only the first
    (``cursor=0``) response.
    """
    rows: list[dict] = []
    open_final: list[dict] | None = None
    fleet = None
    cursor = 0
    n = 0
    statuses: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = get_json(port, f"/api/runs/{run_id}/live?cursor={cursor}")
        n += 1
        statuses.append(doc["status"])
        assert doc["cursor"] >= cursor, (doc["cursor"], cursor)
        assert doc["cursor"] == cursor + len(doc["stints"])
        # The fleet block rides ``cursor=0`` requests only (it never
        # changes during a run, so a client asks once).
        if cursor == 0:
            fleet = doc["fleet"] if doc["fleet"] is not None else fleet
        else:
            assert doc["fleet"] is None, "fleet must ride cursor=0 only"
        if doc["more"]:
            assert doc["open_stints"] is None
        else:
            assert isinstance(doc["open_stints"], list)
            open_final = doc["open_stints"]
        rows.extend(doc["stints"])
        cursor = doc["cursor"]
        if doc["status"] in ("done", "failed") and not doc["more"]:
            return {
                "rows": rows,
                "open_final": open_final,
                "cursor": cursor,
                "fleet": fleet,
                "n_requests": n,
                "statuses": statuses,
                "final": doc,
            }
        if not doc["more"]:
            time.sleep(0.02)
    raise AssertionError("run never finished while draining /live")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def serve_factory(tmp_path):
    """``make(max_workers=N, start_worker=True) -> (port, manager)``."""
    made: list[tuple[RunManager, FleetsimHTTPServer]] = []

    def make(max_workers: int = 1, start_worker: bool = True):
        ws = tmp_path / f"ws{len(made)}"
        manager = RunManager(
            ws, start_worker=start_worker, max_workers=max_workers
        )
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
# 1. the engine hook: opt-in, byte-compatible
# ---------------------------------------------------------------------------

_V05_KEYS = {
    "t_us",
    "horizon_us",
    "jobs_finished",
    "jobs_running",
    "pending",
    "occupancy_to_date",
    "allocated_chips",
    "healthy_chips",
}
_V08_KEYS = _V05_KEYS | {"stints", "stint_cursor", "open_stints", "stint_fleet"}


def test_progress_stints_is_opt_in():
    """Default OFF: the v0.5 snapshot shape stays pinned (the /progress
    contract).  With ``progress_stints=True`` the four v0.8 keys appear."""
    off: list[dict] = []
    api.run_scenario(busy_doc(horizon="5m"), progress_cb=off.append)
    assert off and all(set(s) == _V05_KEYS for s in off)

    on: list[dict] = []
    api.run_scenario(
        busy_doc(horizon="5m"), progress_cb=on.append, progress_stints=True
    )
    assert on and all(set(s) == _V08_KEYS for s in on)
    # The fleet geometry rides the FIRST snapshot only.
    assert on[0]["stint_fleet"] is not None
    assert all(s["stint_fleet"] is None for s in on[1:])


def test_progress_stints_outputs_are_byte_identical(tmp_path):
    """Observation only: the same scenario written with the hook off, with
    the hook on, and with stint streaming on produces byte-identical
    output files."""
    outs = []
    for name, kwargs in (
        ("off", {}),
        ("cb", {"progress_cb": lambda s: None}),
        ("live", {"progress_cb": lambda s: None, "progress_stints": True}),
    ):
        out = tmp_path / name
        api.run_scenario(busy_doc(horizon="5m"), out_dir=out, **kwargs)
        outs.append(out)
    names = [
        "summary.json",
        "jobs.parquet",
        "timeseries.parquet",
        "stints.parquet",
    ]
    for name in names:
        first = (outs[0] / name).read_bytes()
        for other in outs[1:]:
            assert (other / name).read_bytes() == first, name


def test_stint_fleet_matches_the_finished_model_domain_order(tmp_path):
    """``stint_fleet()`` must agree with the finished-run viz model's
    ``fleet.clusters`` — same clusters, same domains, SAME FLAT ORDER (the
    frontend indexes live rows and model rows with one ``domain_idx``)."""
    from fleetsim.viz import build_viz_model

    doc = busy_doc(horizon="10m")
    doc["fleet"]["clusters"][0]["topology"] = {
        "levels": ["pod", "node"],
        "counts": [3, 4],
    }
    doc["outputs"]["stints"] = "pod"
    snaps: list[dict] = []
    out = tmp_path / "run"
    api.run_scenario(doc, out_dir=out, progress_cb=snaps.append, progress_stints=True)

    live = snaps[0]["stint_fleet"]
    model = build_viz_model(out)["fleet"]
    assert live["map_level"] == "pod" == model["map_level"]
    flat = [d["id"] for c in model["clusters"] for d in c["domains"]]
    assert live["domains"] == flat
    assert live["clusters"] == model["clusters"]


def test_stints_since_stream_equals_stint_rows():
    """Collector level: the union of every ``stints_since`` batch plus the
    LAST open overlay (``open`` -> ``running_at_horizon``) is exactly
    ``stint_rows()`` — the invariant the whole live path rests on."""
    collector: dict[str, MetricsCollector] = {}
    streamed: list[dict] = []
    cursors: list[int] = []
    last_open: list[dict] = []

    def cb(snapshot):
        streamed.extend(snapshot["stints"])
        cursors.append(snapshot["stint_cursor"])
        last_open.clear()
        last_open.extend(snapshot["open_stints"])

    # Re-run the pipeline by hand so the collector object stays reachable.
    from fleetsim.config import load_scenario
    from fleetsim.engine.rng import RngStreams
    from fleetsim.engine.sim import Simulator
    from fleetsim.fleet.build import build_fleet
    from fleetsim.schedulers.base import get_scheduler
    from fleetsim.workload.synthetic import SyntheticSource

    scenario = load_scenario(busy_doc(horizon="20m"), strict=True)
    fleet = build_fleet(scenario)
    rng = RngStreams(scenario.sim.seed)
    source = SyntheticSource(
        scenario.workload, fleet, rng, scenario.sim.horizon_us
    )
    coll = MetricsCollector.from_scenario(scenario, fleet)
    collector["c"] = coll
    Simulator(
        scenario,
        fleet,
        source,
        get_scheduler(scenario.scheduler.name, scenario.scheduler.params),
        coll,
        rng=rng,
        progress_cb=cb,
        progress_stints=True,
    ).run()

    assert cursors == sorted(cursors), "cursor must be monotone"
    assert cursors[-1] == len(streamed), "cursor counts every streamed row once"
    assert streamed, "a contended run must settle stints"

    horizon = scenario.sim.horizon_us
    reassembled = [normalize(r) for r in streamed]
    for row in last_open:
        assert row["end_reason"] == "open"
        assert row["t1_us"] == horizon
        reassembled.append(normalize({**row, "end_reason": "running_at_horizon"}))
    assert sorted(reassembled) == sorted(normalize(r) for r in coll.stint_rows())


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_shipped_examples_are_byte_identical_under_v08(tmp_path):
    """BACKWARD-COMPAT CONTRACT for v0.8, against a RECORDED v0.7 BASELINE.

    Two independent things have to hold, and the earlier version of this
    test only checked the first:

    1. The v0.8 live machinery is read-side only, so a run with stint
       STREAMING on writes the same bytes as one with it off.
    2. v0.8 did not move a recorded byte AT ALL — which cannot be shown by
       comparing v0.8 against itself.  ``tests/data/v07_example_outputs_golden.json``
       is the fingerprint of examples 01 and 04 as produced by git 4c73782
       (fleetsim v0.7.0), captured by running that tree; the comparison is
       against THAT.

    The fingerprint is byte hashes PLUS a ULP-tolerant fallback, the shape
    commit 6eeabab established for cross-platform golden data: a matching
    ``sha256`` settles it outright, and otherwise the parquet must have the
    identical row count, column list and per-column aggregate within 1e-9
    relative — enough for a libm ``exp``/``log`` that differs by an ULP on
    another OS/arch, nowhere near enough for a behaviour change.
    """
    golden = json.loads(
        (Path(__file__).parent / "data" / "v07_example_outputs_golden.json")
        .read_text(encoding="utf-8")
    )["examples"]["01_minimal"]

    outs = []
    for name, kwargs in (
        ("plain", {}),
        ("streamed", {"progress_cb": lambda s: None, "progress_stints": True}),
    ):
        out = tmp_path / name
        api.run_scenario(
            _EXAMPLES / "01_minimal" / "scenario.yaml",
            out_dir=out,
            overrides={"outputs.plots": "false"},
            **kwargs,
        )
        outs.append(out)
    produced = sorted(p.name for p in outs[0].iterdir() if p.is_file())
    assert "stints.parquet" in produced and "summary.json" in produced

    # (1) streaming changes nothing
    for name in produced:
        first = (outs[0] / name).read_bytes()
        for other in outs[1:]:
            assert (other / name).read_bytes() == first, name

    # (2) and neither did v0.8, measured against the v0.7 recording
    assert_matches_golden(outs[0], golden)


@pytest.mark.slow
def test_example_04_matches_the_v07_baseline(tmp_path):
    """The same v0.7 comparison for example 04 — the frontier scenario
    with segmented gangs and pod-level stints, i.e. the paths example 01
    never touches.  ~80 s, so it carries the ``slow`` marker and is
    deselected in the default CI path (``-m "not slow"``); the golden data
    it reads was captured from the same v0.7 tree."""
    golden = json.loads(
        (Path(__file__).parent / "data" / "v07_example_outputs_golden.json")
        .read_text(encoding="utf-8")
    )["examples"]["04_frontier"]
    out = tmp_path / "ex04"
    api.run_scenario(
        _EXAMPLES / "04_frontier" / "scenario.yaml",
        out_dir=out,
        overrides={"outputs.plots": "false"},
        progress_cb=lambda s: None,
        progress_stints=True,
    )
    assert_matches_golden(out, golden)


def test_example_04_read_side_additions_touch_nothing(tmp_path):
    """The CI-fast half of example 04: the v0.8 read-side additions are
    populated and reading them leaves the settlement log exactly where it
    was.  (The byte comparison itself is
    ``test_example_04_matches_the_v07_baseline``, marked slow.)"""
    from fleetsim.config import load_scenario
    from fleetsim.fleet.build import build_fleet

    scn = load_scenario(_EXAMPLES / "04_frontier" / "scenario.yaml")
    fleet = build_fleet(scn)
    coll = MetricsCollector.from_scenario(scn, fleet)
    assert coll.stint_level == "pod"
    live_fleet = coll.stint_fleet()
    assert live_fleet["map_level"] == "pod"
    assert live_fleet["domains"], "example 04 must expose pod domains"
    assert coll.stints_since(0) == ([], 0)
    assert coll.open_stint_rows(0) == []
    assert coll.stint_rows() == []


#: Relative tolerance for a golden aggregate.  A 1-ULP libm difference
#: moves a double by ~2e-16 and these are sums over <= 10^5 rows, so 1e-9
#: is orders of magnitude of headroom; any behaviour change is orders of
#: magnitude the other way.
GOLDEN_RTOL = 1e-9


def _close(got, want):
    if got is None or want is None:
        return got is want
    return abs(got - want) <= GOLDEN_RTOL * max(abs(got), abs(want), 1.0)


def assert_matches_golden(out_dir: Path, golden: dict) -> None:
    """Every recorded file matches the v0.7 fingerprint."""
    produced = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert produced == sorted(golden["files"]), (produced, sorted(golden["files"]))
    for name in produced:
        want = golden["files"][name]
        raw = (out_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() == want["sha256"]:
            continue  # byte-identical to v0.7: nothing left to check
        if name == "summary.json":
            _assert_summary_close(json.loads(raw.decode("utf-8")), want["doc"], name)
            continue
        assert name.endswith(".parquet"), f"{name} differs from the v0.7 bytes"
        df = pd.read_parquet(out_dir / name)
        assert len(df) == want["n_rows"], name
        assert [str(c) for c in df.columns] == want["columns"], name
        for col, stats in want["col_stats"].items():
            series = df[col]
            if "sha256" in stats:
                joined = chr(31).join(str(v) for v in series.astype(str).tolist())
                assert (
                    hashlib.sha256(joined.encode("utf-8")).hexdigest()
                    == stats["sha256"]
                ), f"{name}.{col} (text column) differs from v0.7"
                continue
            s = series.astype("float64").dropna()
            assert len(s) == stats["n_finite"], f"{name}.{col}"
            if not stats["n_finite"]:
                continue
            for key, got in (
                ("sum", float(s.sum())),
                ("min", float(s.min())),
                ("max", float(s.max())),
            ):
                assert _close(got, stats[key]), (name, col, key, got, stats[key])


def _assert_summary_close(got, want, path: str) -> None:
    """summary.json compared structurally, floats within GOLDEN_RTOL."""
    assert type(got) is type(want), path
    if isinstance(want, dict):
        assert sorted(got) == sorted(want), path
        for key in want:
            _assert_summary_close(got[key], want[key], f"{path}.{key}")
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, item in enumerate(want):
            _assert_summary_close(got[i], item, f"{path}[{i}]")
    elif isinstance(want, bool) or want is None:
        assert got == want, path
    elif isinstance(want, (int, float)):
        assert _close(float(got), float(want)), (path, got, want)
    else:
        assert got == want, path


# ---------------------------------------------------------------------------
# 2. /api/runs/{id}/live over HTTP
# ---------------------------------------------------------------------------


def test_live_stream_reassembles_stints_parquet(serve_factory):
    """THE cursor test: walk ``/live`` for a whole run and require the
    streamed union to equal ``stints.parquet`` exactly."""
    port, manager = serve_factory()
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml()})["id"]
    walk = drain_live(port, run_id)
    assert walk["statuses"][-1] == "done", walk["statuses"]

    run_dir = manager.workspace / run_id
    horizon = 20 * 60 * 1_000_000
    reassembled = [normalize(r) for r in walk["rows"]]
    assert len(walk["rows"]) == walk["cursor"]  # no duplicate, no gap
    for row in walk["open_final"]:
        assert row["end_reason"] == "open" and row["t1_us"] == horizon
        reassembled.append(normalize({**row, "end_reason": "running_at_horizon"}))
    assert sorted(reassembled) == parquet_rows(run_dir)
    assert walk["final"]["progress"]["t_us"] == horizon
    assert walk["final"]["open_truncated"] is False

    # The live fleet is the exact stint-level geometry (2 nodes of 8 chips).
    fleet = walk["fleet"]
    assert fleet is not None
    assert fleet["map_level"] == "node"
    assert fleet["domains"] == ["m/c/node0", "m/c/node1"]
    assert [d["chips"] for c in fleet["clusters"] for d in c["domains"]] == [8, 8]


def test_live_replay_from_cursor_zero_after_the_run(serve_factory):
    """A client that arrives late (or reconnects) reads the identical
    stream from ``cursor=0`` — the spool is on disk, not in memory."""
    port, manager = serve_factory()
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="10m")})["id"]
    first = drain_live(port, run_id)
    second = drain_live(port, run_id)
    assert [normalize(r) for r in second["rows"]] == [
        normalize(r) for r in first["rows"]
    ]
    assert second["open_final"] == first["open_final"]
    assert second["fleet"] == first["fleet"]


def test_live_row_cap_forces_the_client_to_loop(serve_factory):
    """With the per-response cap set to 1 row, the payload reports
    ``more: true``, withholds the open overlay, and the loop still
    reassembles the identical row sequence."""
    port, manager = serve_factory()
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="10m")})["id"]
    whole = drain_live(port, run_id)

    rows: list[dict] = []
    cursor = 0
    saw_more = False
    while True:
        code, doc = manager.live_payload(run_id, cursor=cursor, limit=1)
        assert code == 200
        assert len(doc["stints"]) <= 1
        if doc["more"]:
            saw_more = True
            assert doc["open_stints"] is None
        rows.extend(doc["stints"])
        assert doc["cursor"] == cursor + len(doc["stints"])
        cursor = doc["cursor"]
        if not doc["more"]:
            break
    assert saw_more, "a run with >1 stint row must have reported more=True"
    assert [normalize(r) for r in rows] == [normalize(r) for r in whole["rows"]]


def test_live_without_stints_streams_progress_only(serve_factory):
    """``outputs.stints`` off: the live endpoint still reports status and
    progress, but there is no map to build — empty rows, null fleet."""
    port, manager = serve_factory()
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(stints=False, horizon="5m")})[
        "id"
    ]
    walk = drain_live(port, run_id)
    assert walk["rows"] == []
    assert walk["cursor"] == 0
    assert walk["open_final"] == []
    assert walk["fleet"] is None
    assert walk["final"]["progress"]["jobs_finished"] > 0
    assert not (manager.workspace / run_id / LIVE_ROWS).exists()
    assert (manager.workspace / run_id / LIVE_STATE).is_file()


def test_live_queued_run_and_unknown_ids(serve_factory):
    """Before the first flush the payload is well-formed and empty; an
    unknown or hostile id is a 404, never a 500."""
    port, _ = serve_factory(start_worker=False)
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml()})["id"]
    doc = get_json(port, f"/api/runs/{run_id}/live")
    assert doc == {
        "status": "queued",
        "cursor": 0,
        "more": False,
        "progress": None,
        "stints": [],
        "open_stints": [],
        "open_truncated": False,
        "open_pending": False,
        "stalled_at": None,
        "fleet": None,
    }
    get_json(port, "/api/runs/nope/live", expect=404)
    get_json(port, "/api/runs/..%2F..%2Fetc/live", expect=404)


def test_live_cursor_query_param_is_never_a_500(serve_factory):
    """A garbage ``cursor`` falls back to 0 (and a huge one simply yields
    nothing) rather than erroring — query strings are untrusted input."""
    port, _ = serve_factory(start_worker=False)
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml()})["id"]
    for query in ("", "?cursor=", "?cursor=abc", "?cursor=-9", "?cursor=" + "9" * 40):
        doc = get_json(port, f"/api/runs/{run_id}/live{query}")
        assert doc["cursor"] == 0, query
        assert doc["stints"] == []


# ---------------------------------------------------------------------------
# 3. parallel workers: queue positions, throughput, determinism
# ---------------------------------------------------------------------------


def test_default_max_workers_is_bounded():
    n = default_max_workers()
    assert 1 <= n <= 4
    # A manager takes the default, and honours an explicit cap.
    assert RunManager.__init__.__kwdefaults__["max_workers"] is None


def test_cli_workers_flag():
    """``serve --workers N`` parses, defaults to None (= the computed
    default), and rejects zero/negative before binding anything."""
    from fleetsim.cli import build_parser, main

    parser = build_parser()
    assert parser.parse_args(["serve"]).workers is None
    assert parser.parse_args(["serve", "--workers", "3"]).workers == 3
    assert main(["serve", "--workers", "0"]) == 2
    assert main(["serve", "--workers", "-2"]) == 2


def test_queue_positions_are_fifo_and_shrink(serve_factory):
    """With the dispatcher stopped, every submitted run reports its 1-based
    FIFO position; dequeuing one closes the gap."""
    port, manager = serve_factory(start_worker=False)
    ids = [
        post_json(port, "/api/runs", {"yaml": busy_yaml(seed=i)})["id"]
        for i in range(4)
    ]
    listing = {r["id"]: r for r in get_json(port, "/api/runs")}
    assert [listing[i]["queue_position"] for i in ids] == [1, 2, 3, 4]
    assert all(listing[i]["status"] == "queued" for i in ids)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/runs/{ids[1]}", method="DELETE"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
    listing = {r["id"]: r for r in get_json(port, "/api/runs")}
    assert [listing[i]["queue_position"] for i in (ids[0], ids[2], ids[3])] == [
        1,
        2,
        3,
    ]


@pytest.mark.parametrize("workers", (1, 2))
def test_at_most_max_workers_run_at_once(serve_factory, workers):
    """The admission cap is real: sampling the listing while four runs
    drain never shows more than ``max_workers`` in ``running``, and the
    queued tail always carries contiguous 1..N positions."""
    port, manager = serve_factory(max_workers=workers)
    ids = [
        post_json(port, "/api/runs", {"yaml": busy_yaml(seed=i, horizon="10m")})["id"]
        for i in range(4)
    ]
    peak = 0
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        rows = [r for r in get_json(port, "/api/runs") if r["id"] in ids]
        running = [r for r in rows if r["status"] == "running"]
        queued = sorted(
            r["queue_position"] for r in rows if r["status"] == "queued"
        )
        peak = max(peak, len(running))
        assert len(running) <= workers, rows
        assert queued == list(range(1, len(queued) + 1)), rows
        for row in rows:
            if row["status"] not in ("queued",):
                assert row["queue_position"] is None, row
        if all(r["status"] == "done" for r in rows):
            break
        time.sleep(0.05)
    else:  # pragma: no cover - timeout
        raise AssertionError("runs never all finished")
    assert peak >= 1


def test_two_workers_really_run_two_scenarios_at_once(serve_factory):
    """Parallelism, proved rather than inferred: with ``max_workers=2``,
    three long runs settle into exactly two ``running`` and one ``queued``
    at position 1 — and the queued one is not merely slow, since the two
    running ones would take hours."""
    port, manager = serve_factory(max_workers=2)
    ids = [
        post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="8h", seed=i)})["id"]
        for i in range(3)
    ]
    deadline = time.monotonic() + 90
    rows: list[dict] = []
    while time.monotonic() < deadline:
        rows = [r for r in get_json(port, "/api/runs") if r["id"] in ids]
        if sum(1 for r in rows if r["status"] == "running") == 2:
            break
        time.sleep(0.05)
    states = sorted(r["status"] for r in rows)
    assert states == ["queued", "running", "running"], rows
    queued = next(r for r in rows if r["status"] == "queued")
    assert queued["queue_position"] == 1, rows
    # Both in-flight runs are streaming independently (the fleet block
    # appears with the first metrics flush, so give it a moment).
    for row in rows:
        if row["status"] != "running":
            continue
        doc = None
        stop = time.monotonic() + 60
        while time.monotonic() < stop:
            doc = get_json(port, f"/api/runs/{row['id']}/live")
            if doc["fleet"] is not None:
                break
            time.sleep(0.05)
        assert doc is not None and doc["fleet"] is not None, row
        assert doc["progress"]["t_us"] > 0


def test_parallel_execution_is_deterministic(serve_factory):
    """Same scenario + seed in a worker process twice -> byte-identical
    outputs.  Runs execute out of process now; determinism must not care."""
    port, manager = serve_factory(max_workers=2)
    ids = [
        post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="10m", seed=11)})[
            "id"
        ]
        for _ in range(2)
    ]
    for run_id in ids:
        wait_status(port, run_id, "done")
    a, b = (manager.workspace / i for i in ids)
    for name in (
        "summary.json",
        "jobs.parquet",
        "timeseries.parquet",
        "stints.parquet",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_cancel_reaches_the_worker_process(serve_factory):
    """``POST /cancel`` is delivered to the child as ``cancel.flag`` and
    stops it at the next metrics flush, marked ``failed``."""
    port, manager = serve_factory()
    run_id = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="8h")})["id"]
    wait_status(port, run_id, "running", timeout=60)
    assert post_json(port, f"/api/runs/{run_id}/cancel", None) == {"ok": True}
    assert (manager.workspace / run_id / "cancel.flag").is_file()
    detail = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        detail = get_json(port, f"/api/runs/{run_id}")
        if detail["status"] == "failed":
            break
        time.sleep(0.05)
    assert detail["status"] == "failed", detail
    assert detail["error"] == "cancelled by request"


def test_shutdown_is_bounded_and_marks_inflight_failed(tmp_path):
    """Ctrl-C semantics: ``shutdown`` stops a long in-flight run, marks it
    and every queued run ``failed`` with a clear reason, and RETURNS —
    bounded, not "whenever the simulation ends"."""
    manager = RunManager(tmp_path / "ws", max_workers=1)
    running = manager.submit(busy_yaml(horizon="24h"), "long")
    queued = manager.submit(busy_yaml(horizon="24h"), "waiting")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if (manager.get_run(running) or {}).get("status") == "running":
            break
        time.sleep(0.05)
    assert manager.get_run(running)["status"] == "running"

    t0 = time.monotonic()
    manager.shutdown(timeout=15.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 15.0, elapsed

    after_running = manager.get_run(running)
    after_queued = manager.get_run(queued)
    assert after_running["status"] == "failed"
    assert "shut down" in after_running["error"] or "aborted" in after_running["error"]
    assert after_queued["status"] == "failed"
    assert "shut down" in after_queued["error"]
    # Submitting after shutdown is refused, not silently queued.
    with pytest.raises(RuntimeError):
        manager.submit(busy_yaml(), "late")


def test_unguarded_main_is_refused_with_the_fix(tmp_path):
    """Worker processes re-execute the parent's ``__main__``, so a user
    script without an ``if __name__ == "__main__":`` guard would start a
    second server inside every worker.

    Both halves are checked for real, in subprocesses: the GUARDED script
    runs a simulation to completion, and the unguarded one is refused with
    a message naming the guard (instead of a phantom workspace-lock error
    pointing nowhere near the cause)."""
    import subprocess
    import sys

    body = """
import json, pathlib, sys, time, yaml
sys.path.insert(0, {tests!r})
from test_serve_live import busy_doc
from fleetsim.serve.runs import RunManager

def run():
    m = RunManager({ws!r}, max_workers=2)
    rid = m.submit(yaml.safe_dump(busy_doc(horizon="2m")), "guarded")
    for _ in range(600):
        info = m.get_run(rid)
        if info and info["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    m.shutdown(timeout=10)
    print("STATUS", m.get_run(rid)["status"])
"""
    tests_dir = str(Path(__file__).resolve().parent)
    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        body.format(tests=tests_dir, ws=str(tmp_path / "ws_ok"))
        + '\nif __name__ == "__main__":\n    run()\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(guarded)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tests_dir,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "STATUS done" in proc.stdout, (proc.stdout, proc.stderr[-2000:])

    unguarded = tmp_path / "unguarded.py"
    unguarded.write_text(
        body.format(tests=tests_dir, ws=str(tmp_path / "ws_bad")) + "\nrun()\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(unguarded)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tests_dir,
    )
    combined = proc.stdout + proc.stderr
    assert "UnguardedMainError" in combined, combined[-3000:]
    assert '__name__ == "__main__"' in combined, combined[-3000:]
    assert "STATUS done" not in proc.stdout


def test_child_run_reports_failure_as_an_outcome(tmp_path):
    """``_child_run`` never raises across the process boundary: a scenario
    the loader rejects comes back as an outcome dict the parent can act
    on."""
    from fleetsim.serve.runs import _child_run

    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    (run_dir / "scenario.yaml").write_text(
        "sim: {horizon: 1m, round: 30s}\n", encoding="utf-8"
    )
    result = _child_run(str(run_dir))
    assert result["outcome"] == "failed"
    assert result["error"]


def test_a_failing_run_does_not_take_the_pool_down(serve_factory):
    """A run that fails in its worker marks THAT run failed and leaves the
    pool serving the next one.

    The broken scenario is corrupted WHILE QUEUED (behind a long run on a
    single worker), so the child is guaranteed to read the broken file —
    no race with the dispatcher."""
    port, manager = serve_factory(max_workers=1)
    blocker = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="8h")})["id"]
    broken = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="5m")})["id"]
    good = post_json(port, "/api/runs", {"yaml": busy_yaml(horizon="5m")})["id"]
    wait_status(port, blocker, "running", timeout=60)
    assert get_json(port, f"/api/runs/{broken}")["status"] == "queued"
    (manager.workspace / broken / "scenario.yaml").write_text(
        "sim: {horizon: 1m, round: 30s}\n", encoding="utf-8"
    )
    assert post_json(port, f"/api/runs/{blocker}/cancel", None) == {"ok": True}

    wait_status(port, broken, "failed", timeout=120)
    detail = get_json(port, f"/api/runs/{broken}")
    assert detail["error"]
    assert str(manager.workspace) not in detail["error"]
    # The pool survived the failure: the next queued run completes.
    wait_status(port, good, "done", timeout=120)
