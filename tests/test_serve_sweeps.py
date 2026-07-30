"""Tests for the v0.8 sweep API and the two new data endpoints.

The sweep contract has four properties worth pinning, and each has a test
below: the expansion ORDER is deterministic (so run ids line up with cells
a reader can predict), the cell CAP is enforced on the computed product
rather than on an expanded list (a 20-axis request must not be expanded to
be rejected), validation is ALL-OR-NOTHING (one bad cell creates nothing),
and a sweep's cells are ordinary runs (every run endpoint works on them).

``GET /api/validation`` is asserted against
:mod:`fleetsim.validation.results` — the same module the validation tests
import — so app and tests cannot drift.  ``GET /api/examples`` is asserted
to carry all seven shipped starters.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import yaml

from fleetsim.serve.runs import RunManager
from fleetsim.serve.server import FleetsimHTTPServer, list_examples
from fleetsim.serve.sweeps import (
    MAX_SWEEP_RUNS,
    SweepManager,
    expand_grid,
    grid_size,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def base_doc(**sim_over):
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
                    "rate_per_hour": 120,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=1m, p90=4m]",
                }
            },
        },
        "scheduler": {"name": "fifo"},
        "outputs": {"stints": True},
    }


def base_yaml(**kw) -> str:
    return yaml.safe_dump(base_doc(**kw))


def request_json(port, method, path, body=None, expect=200):
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


def get_json(port, path, expect=200):
    return request_json(port, "GET", path, expect=expect)


def wait_status(port, run_id, want, timeout=120.0):
    targets = {want} if isinstance(want, str) else set(want)
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = get_json(port, f"/api/runs/{run_id}")
        if last["status"] in targets:
            return last
        time.sleep(0.05)
    raise AssertionError(f"{run_id} never reached {targets}; last={last}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def serve_factory(tmp_path):
    made = []

    def make(max_workers: int = 1, start_worker: bool = False):
        manager = RunManager(
            tmp_path / f"ws{len(made)}",
            start_worker=start_worker,
            max_workers=max_workers,
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
# 1. expansion (pure functions)
# ---------------------------------------------------------------------------


def test_expand_grid_order_is_nested_loops_last_axis_fastest():
    cells = expand_grid({"a.b": [1, 2], "c.d": ["x", "y", "z"]})
    assert cells == [
        {"a.b": 1, "c.d": "x"},
        {"a.b": 1, "c.d": "y"},
        {"a.b": 1, "c.d": "z"},
        {"a.b": 2, "c.d": "x"},
        {"a.b": 2, "c.d": "y"},
        {"a.b": 2, "c.d": "z"},
    ]


def test_seeds_are_the_last_axis():
    cells = expand_grid({"scheduler.name": ["fifo", "sjf"]}, seeds=[1, 2])
    assert cells == [
        {"scheduler.name": "fifo", "sim.seed": 1},
        {"scheduler.name": "fifo", "sim.seed": 2},
        {"scheduler.name": "sjf", "sim.seed": 1},
        {"scheduler.name": "sjf", "sim.seed": 2},
    ]


def test_grid_size_never_materializes_the_product():
    """A few hundred bytes of JSON must not become 64**20 dicts: the cap is
    enforced on the COMPUTED size."""
    grid = {f"a.k{i}": list(range(64)) for i in range(20)}
    assert grid_size(grid) == 64**20
    assert grid_size({}) == 1
    assert grid_size({}, seeds=[1, 2, 3]) == 3


def test_empty_grid_expands_to_the_base_scenario():
    assert expand_grid({}) == [{}]


# ---------------------------------------------------------------------------
# 2. POST /api/sweeps
# ---------------------------------------------------------------------------


def test_sweep_creates_one_run_per_cell_with_cell_metadata(serve_factory):
    port, manager = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(),
            "title": "seed study",
            "grid": {"scheduler.name": ["fifo", "sjf"]},
            "seeds": [1, 2],
        },
    )
    assert out["n_runs"] == 4
    assert len(out["run_ids"]) == 4
    assert out["sweep_id"].startswith("sweep-")

    cells = []
    for run_id in out["run_ids"]:
        meta = json.loads(
            (manager.workspace / run_id / "meta.json").read_text(encoding="utf-8")
        )
        assert meta["sweep_id"] == out["sweep_id"]
        assert meta["status"] == "queued"
        cells.append(meta["sweep_cell"])
        # The stored scenario is a COMPLETE scenario with the cell applied.
        doc = yaml.safe_load(
            (manager.workspace / run_id / "scenario.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert doc["scheduler"]["name"] == meta["sweep_cell"]["scheduler.name"]
        assert doc["sim"]["seed"] == meta["sweep_cell"]["sim.seed"]
        assert doc["fleet"]["clusters"][0]["chip"]["per_node"] == 8  # untouched
    assert cells == expand_grid({"scheduler.name": ["fifo", "sjf"]}, seeds=[1, 2])

    # Cells are ordinary runs: they carry sweep provenance in the listing.
    listing = {r["id"]: r for r in get_json(port, "/api/runs")}
    for run_id in out["run_ids"]:
        row = listing[run_id]
        assert row["sweep_id"] == out["sweep_id"]
        assert row["status"] == "queued"
        assert row["queue_position"] in (1, 2, 3, 4)
    assert sorted(listing[i]["queue_position"] for i in out["run_ids"]) == [
        1,
        2,
        3,
        4,
    ]


def test_seeds_only_sweep_is_a_valid_one_axis_sweep(serve_factory):
    """The commonest sweep — "the same scenario at three seeds" — needs no
    grid at all."""
    port, manager = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {"yaml": base_yaml(), "grid": {}, "seeds": [3, 4, 5]},
    )
    assert out["n_runs"] == 3
    seeds = []
    for run_id in out["run_ids"]:
        meta = json.loads(
            (manager.workspace / run_id / "meta.json").read_text(encoding="utf-8")
        )
        assert meta["sweep_cell"] == {"sim.seed": meta["sweep_cell"]["sim.seed"]}
        seeds.append(meta["sweep_cell"]["sim.seed"])
    assert seeds == [3, 4, 5]
    detail = get_json(port, f"/api/sweeps/{out['sweep_id']}")
    assert detail["seeds"] == [3, 4, 5]
    assert detail["grid"] == {}


def test_sweep_overrides_apply_nested_paths(serve_factory):
    port, manager = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(),
            "grid": {"workload.classes.eval.rate_per_hour": [60, 600]},
        },
    )
    rates = []
    for run_id in out["run_ids"]:
        doc = yaml.safe_load(
            (manager.workspace / run_id / "scenario.yaml").read_text(
                encoding="utf-8"
            )
        )
        rates.append(doc["workload"]["classes"]["eval"]["rate_per_hour"])
    assert rates == [60, 600]


def test_sweep_cap_is_413_and_creates_nothing(serve_factory):
    port, manager = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(),
            "grid": {
                "sim.seed": list(range(9)),
                "workload.classes.eval.rate_per_hour": [60, 120, 240],
                "scheduler.name": ["fifo", "sjf", "easy_backfill"],
            },
        },
        expect=413,
    )
    assert out["ok"] is False
    assert str(MAX_SWEEP_RUNS) in out["errors"][0]
    assert "81" in out["errors"][0]  # the computed size is reported
    assert get_json(port, "/api/runs") == []
    assert get_json(port, "/api/sweeps") == []


def test_sweep_validation_is_all_or_nothing(serve_factory):
    """One invalid cell -> 400, and NOT ONE run directory is created."""
    port, manager = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(),
            "grid": {"scheduler.name": ["fifo", "no_such_scheduler"]},
        },
        expect=400,
    )
    assert out["ok"] is False
    assert any("no_such_scheduler" in e for e in out["errors"])
    assert any(e.startswith("cell [") for e in out["errors"])
    assert get_json(port, "/api/runs") == []
    assert get_json(port, "/api/sweeps") == []
    assert not (manager.workspace / ".sweeps").exists()


def test_sweep_reports_at_most_a_bounded_number_of_cell_errors(serve_factory):
    port, _ = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(),
            "grid": {"scheduler.name": [f"nope{i}" for i in range(30)]},
        },
        expect=400,
    )
    assert len(out["errors"]) <= 21
    assert "more cell error" in out["errors"][-1]


@pytest.mark.parametrize(
    "body,expect",
    (
        ({"yaml": 5, "grid": {"sim.seed": [1]}}, 400),
        ({"yaml": "a: 1", "grid": []}, 400),
        ({"yaml": "a: 1", "grid": {"sim.seed": [1]}, "seeds": 3}, 400),
        ({"yaml": "a: 1", "grid": {}}, 400),
        ({"yaml": "a: 1", "grid": {"sim.seed": []}}, 400),
        ({"yaml": "a: 1", "grid": {"sim..seed": [1]}}, 400),
        ({"yaml": "a: 1", "grid": {"../etc": [1]}}, 400),
        ({"yaml": "a: 1", "grid": {"sim.seed": "1"}}, 400),
        ({"yaml": "a: 1", "grid": {"sim.seed": [1]}, "seeds": ["x"]}, 400),
        ({"yaml": "a: [unclosed", "grid": {"sim.seed": [1]}}, 400),
        ({"yaml": "just a string", "grid": {"sim.seed": [1]}}, 400),
    ),
)
def test_sweep_request_envelope_errors(serve_factory, body, expect):
    port, _ = serve_factory()
    out = request_json(port, "POST", "/api/sweeps", body, expect=expect)
    assert out.get("errors") or out.get("error")


# ---------------------------------------------------------------------------
# 3. GET/DELETE /api/sweeps
# ---------------------------------------------------------------------------


def test_sweep_listing_and_detail(serve_factory):
    port, manager = serve_factory()
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(),
            "title": "packing",
            "grid": {"scheduler.name": ["fifo", "sjf"]},
        },
    )
    listing = get_json(port, "/api/sweeps")
    assert len(listing) == 1
    row = listing[0]
    assert row["sweep_id"] == out["sweep_id"]
    assert row["title"] == "packing"
    assert (row["n_runs"], row["n_done"]) == (2, 0)
    assert row["grid"] == {"scheduler.name": ["fifo", "sjf"]}
    assert isinstance(row["created"], int)

    detail = get_json(port, f"/api/sweeps/{out['sweep_id']}")
    assert detail["n_runs"] == 2
    assert [r["cell"] for r in detail["runs"]] == [
        {"scheduler.name": "fifo"},
        {"scheduler.name": "sjf"},
    ]
    assert all(r["status"] == "queued" for r in detail["runs"])
    assert all(r["headline"] is None for r in detail["runs"])
    assert [r["queue_position"] for r in detail["runs"]] == [1, 2]

    get_json(port, "/api/sweeps/sweep-19700101-000000-001-aaaa", expect=404)
    get_json(port, "/api/sweeps/nope", expect=404)
    get_json(port, "/api/sweeps/..%2F..%2Fetc%2Fpasswd", expect=404)


def test_sweep_detail_carries_headlines_once_cells_finish(serve_factory):
    port, manager = serve_factory(start_worker=True, max_workers=2)
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {"yaml": base_yaml(horizon="5m"), "grid": {"sim.seed": [1, 2]}},
    )
    for run_id in out["run_ids"]:
        wait_status(port, run_id, "done")
    detail = get_json(port, f"/api/sweeps/{out['sweep_id']}")
    assert detail["n_done"] == 2
    for row in detail["runs"]:
        assert row["status"] == "done"
        # the three pinned keys plus the run's own frag.<level> means
        # (v0.8: the sweep board's placement metrics come from here)
        assert {"occupancy", "goodput", "jobs_finished"} <= set(row["headline"])
        assert all(
            k in ("occupancy", "goodput", "jobs_finished") or k.startswith("frag.")
            for k in row["headline"]
        ), sorted(row["headline"])
        assert row["queue_position"] is None
    listing = get_json(port, "/api/sweeps")
    assert listing[0]["n_done"] == 2
    assert listing[0]["n_failed"] == 0


def test_delete_sweep_dequeues_only_queued_cells(serve_factory):
    """A sweep whose every cell is still queued disappears entirely; one
    with a running or finished cell keeps its record and that cell."""
    port, manager = serve_factory()  # dispatcher stopped: all cells queued
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {"yaml": base_yaml(), "grid": {"sim.seed": [1, 2, 3]}},
    )
    res = request_json(port, "DELETE", f"/api/sweeps/{out['sweep_id']}")
    assert res["ok"] is True
    assert sorted(res["dequeued"]) == sorted(out["run_ids"])
    assert res["kept"] == []
    assert res["removed_record"] is True
    assert get_json(port, "/api/runs") == []
    assert get_json(port, "/api/sweeps") == []
    request_json(port, "DELETE", f"/api/sweeps/{out['sweep_id']}", expect=404)


def test_delete_sweep_keeps_finished_cells(serve_factory):
    port, manager = serve_factory(start_worker=True, max_workers=1)
    out = request_json(
        port,
        "POST",
        "/api/sweeps",
        {
            "yaml": base_yaml(horizon="5m"),
            "grid": {"sim.seed": [1, 2, 3, 4, 5, 6]},
        },
    )
    first = out["run_ids"][0]
    wait_status(port, first, "done")
    res = request_json(port, "DELETE", f"/api/sweeps/{out['sweep_id']}")
    assert first in res["kept"], res
    assert first not in res["dequeued"]
    assert res["removed_record"] is False
    detail = get_json(port, f"/api/sweeps/{out['sweep_id']}")
    assert [r["id"] for r in detail["runs"]] == res["kept"]
    assert first in [r["id"] for r in detail["runs"]]


def test_sweep_records_live_in_a_dot_directory(serve_factory):
    """The sweeps directory must never surface as a run."""
    port, manager = serve_factory()
    request_json(
        port,
        "POST",
        "/api/sweeps",
        {"yaml": base_yaml(), "grid": {"sim.seed": [1]}},
    )
    assert (manager.workspace / ".sweeps").is_dir()
    assert all(r["id"] != ".sweeps" for r in get_json(port, "/api/runs"))


def test_sweep_after_shutdown_is_refused(tmp_path):
    manager = RunManager(tmp_path / "ws", start_worker=False)
    sweeps = SweepManager(manager)
    manager.shutdown(timeout=5.0)
    with pytest.raises(RuntimeError):
        sweeps.create(base_yaml(), {"sim.seed": [1]})
    assert sweeps.list_sweeps() == []


# ---------------------------------------------------------------------------
# 4. parallel determinism across a sweep
# ---------------------------------------------------------------------------


def test_the_same_sweep_twice_is_byte_identical(serve_factory):
    """Two identical sweeps, executed by a 2-worker pool in whatever order
    the OS chooses, produce byte-identical outputs cell for cell — the
    determinism guarantee ``(scenario, seed)`` implies, now that runs
    execute out of process and concurrently."""
    port, manager = serve_factory(start_worker=True, max_workers=2)
    body = {
        "yaml": base_yaml(horizon="5m"),
        "grid": {"scheduler.name": ["fifo", "sjf"]},
        "seeds": [7, 8],
    }
    first = request_json(port, "POST", "/api/sweeps", body)
    second = request_json(port, "POST", "/api/sweeps", body)
    for run_id in first["run_ids"] + second["run_ids"]:
        wait_status(port, run_id, "done")

    names = ("summary.json", "jobs.parquet", "timeseries.parquet", "stints.parquet")
    for a_id, b_id in zip(first["run_ids"], second["run_ids"]):
        a, b = manager.workspace / a_id, manager.workspace / b_id
        a_cell = json.loads((a / "meta.json").read_text())["sweep_cell"]
        b_cell = json.loads((b / "meta.json").read_text())["sweep_cell"]
        assert a_cell == b_cell
        for name in names:
            assert (a / name).read_bytes() == (b / name).read_bytes(), (
                a_cell,
                name,
            )
    # Different cells really are different runs (the sweep is not a no-op).
    summaries = {
        (manager.workspace / rid / "summary.json").read_bytes()
        for rid in first["run_ids"]
    }
    assert len(summaries) > 1


# ---------------------------------------------------------------------------
# 5. GET /api/validation and GET /api/examples
# ---------------------------------------------------------------------------


def test_validation_endpoint_matches_the_results_module(serve_factory):
    from fleetsim.validation import results as R

    port, _ = serve_factory()
    doc = get_json(port, "/api/validation")
    assert doc == R.payload()  # JSON round-trip is lossless for this table
    assert doc["doc"] == "docs/validation.md"
    ids = [row["id"] for row in doc["results"]]
    assert len(ids) == len(set(ids)) == len(R.RESULTS)
    for row in doc["results"]:
        assert row["published"] is not None
        assert row["trace"] in doc["citations"]
        assert row["rung"] in {"V1", "V2", "V3"}
        if row["fleetsim"] is None:
            assert row["in_band"] is None and row["rel_error"] is None
    assert doc["counts"]["in_band"] == doc["counts"]["asserted"], (
        "every asserted, measured row should satisfy its own band"
    )
    assert doc["citations"]["helios"]["license"] == "CC-BY-4.0"
    assert "Hu et al." in doc["citations"]["helios"]["citation"]
    assert doc["ladder"] and doc["anti_goals"] and doc["placer_sweep"]


def test_results_module_is_the_single_source_of_truth():
    """The numbers the validation tests assert are the numbers the endpoint
    serves — derived from one table, not typed twice."""
    from fleetsim.validation import results as R

    assert R.HELIOS_PUBLISHED_RATIOS["Saturn"] == {
        "jct": 6.59,
        "q": 18.5,
        "share": 0.897,
    }
    assert set(R.HELIOS_PUBLISHED_RATIOS) == {"Venus", "Earth", "Saturn", "Uranus"}
    assert R.HELIOS_SATURN_FIFO_JCT_PUBLISHED == 55_984.0
    assert R.PHILLY_PUBLISHED_BY_COUNT == {
        "Passed": 0.693,
        "Killed": 0.135,
        "Unsuccessful": 0.172,
    }
    # Bands are shared, not re-declared, by the tests that assert them.
    ratio_rows = R.results_for(rung="V1", trace="helios")
    for row in ratio_rows:
        if row.id.startswith("helios.jct_ratio."):
            assert row.band == R.HELIOS_JCT_RATIO_BAND
            assert row.in_band is True, row
        elif row.id.startswith("helios.q_ratio."):
            assert row.band == R.HELIOS_Q_RATIO_BAND
            assert row.in_band is True, row
    # Saturn's v0.6 first_fit JCT ratio is the recorded out-of-band value.
    saturn = next(r for r in R.RESULTS if r.id == "helios.jct_ratio.Saturn")
    assert saturn.previous == 8.75
    assert not (saturn.band[0] <= saturn.previous <= saturn.band[1])
    # Absolute FIFO JCT lands within the +/-14 % docs/validation.md §5 claim.
    for row in R.results_for(rung="V2"):
        if row.id.startswith("helios.fifo_jct_s."):
            assert abs(row.rel_error) <= 0.14, row


def test_validation_tests_import_the_shared_constants():
    """The validation suite asserts against the results module's OBJECTS.

    Checked by identity, not by value: if someone re-types the published
    numbers into a test file, these ``is`` comparisons fail — which is the
    whole point of moving them into one table.
    """
    import importlib
    import sys
    from pathlib import Path

    from fleetsim.validation import results as R

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "validation"))
    try:
        helios = importlib.import_module("test_helios_ratio")
        philly = importlib.import_module("test_philly_status")
    finally:
        sys.path.remove(str(root / "validation"))

    assert helios._JCT_RATIO_BAND is R.HELIOS_JCT_RATIO_BAND
    assert helios._Q_RATIO_BAND is R.HELIOS_Q_RATIO_BAND
    assert helios._PUB is R.HELIOS_PUBLISHED_RATIOS
    assert helios._PUB_SATURN_FIFO_JCT == R.HELIOS_SATURN_FIFO_JCT_PUBLISHED
    assert philly._PUB_BY_COUNT is R.PHILLY_PUBLISHED_BY_COUNT
    assert philly._PUB_BY_GPU is R.PHILLY_PUBLISHED_BY_GPU
    assert philly._BY_COUNT_TOL == R.PHILLY_BY_COUNT_TOL
    assert philly._BY_GPU_TOL == R.PHILLY_BY_GPU_TOL


def test_examples_endpoint_returns_all_seven_starters(serve_factory):
    port, _ = serve_factory()
    doc = get_json(port, "/api/examples")
    names = [e["name"] for e in doc]
    assert names == sorted(names)
    assert names == [
        "01_minimal",
        "02_trace_replay",
        "03_custom_scheduler",
        "04_frontier",
        "05_topology_tradeoff",
        "06_economics",
        "07_placement_study",
    ]
    for entry in doc:
        assert entry["yaml"].strip()
        assert isinstance(entry["runnable"], bool)
        if not entry["runnable"]:
            assert entry["note"]
    # The trace-replay starter is the one that cannot run web-submitted.
    assert [e["name"] for e in doc if not e["runnable"]] == ["02_trace_replay"]
    assert doc == list_examples()
