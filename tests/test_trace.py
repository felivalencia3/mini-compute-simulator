"""Trace loading/replay tests: the canonical CSV schema, the sample
trace, round-tripping, the Philly adapter, and verbatim final_status
replay through the engine."""

import json
from pathlib import Path

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.base import NullSink
from fleetsim.model import JobClass, JobStatus, Tier
from fleetsim.schedulers.fifo import FIFOScheduler
from fleetsim.workload.base import JobSource
from fleetsim.workload.philly import convert_philly
from fleetsim.workload.trace import (
    CANONICAL_COLUMNS,
    TraceJob,
    TraceSource,
    load_trace,
    write_trace,
)

S = 1_000_000

SAMPLE = Path(__file__).parent / "data" / "sample_trace.csv"

HEADER = ",".join(CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Canonical CSV loading
# ---------------------------------------------------------------------------


def test_sample_trace_loads():
    jobs = load_trace(SAMPLE)
    assert len(jobs) == 30
    assert all(isinstance(j, TraceJob) for j in jobs)
    times = [j.submit_t for j in jobs]
    assert times == sorted(times)

    j0 = jobs[0]
    assert j0.id == "j000"
    assert j0.tenant == "t0"
    assert j0.job_class is JobClass.EVAL
    assert j0.tier is Tier.BATCH
    assert j0.submit_t == 0
    assert j0.gangs[0].chips == 1
    assert j0.gangs[0].chip_type == "h100"
    assert j0.true_duration_s == 120.0
    assert j0.walltime_est_s == 600.0
    assert j0.terminal_status_override is None  # COMPLETED

    by_id = {j.id: j for j in jobs}
    assert by_id["j002"].terminal_status_override is JobStatus.FAILED
    assert by_id["j002"].walltime_est_s is None  # empty walltime_limit_s
    assert by_id["j004"].terminal_status_override is JobStatus.CANCELED
    assert by_id["j011"].terminal_status_override is JobStatus.TIMEOUT
    assert by_id["j014"].terminal_status_override is JobStatus.NODE_FAIL
    assert by_id["j010"].job_class is JobClass.PRETRAIN
    assert by_id["j010"].tier is Tier.PROD
    assert by_id["j010"].gangs[0].chips == 32
    assert by_id["j017"].job_class is JobClass.INFER_REPLICA
    assert by_id["j017"].tier is Tier.PROD
    # ~30% killed/failed, per the trace mix.
    non_completed = [j for j in jobs if j.terminal_status_override is not None]
    assert len(non_completed) == 9


def test_round_trip_write_then_load(tmp_path):
    jobs = load_trace(SAMPLE)
    rows = []
    for j in jobs:
        rows.append(
            {
                "job_id": j.id,
                "user": "u",
                "tenant": j.tenant,
                "class": j.job_class.name.lower(),
                "submit_time": j.submit_t,
                "num_chips": j.gangs[0].chips,
                "chip_type": j.gangs[0].chip_type or "",
                "num_nodes": 1,
                "duration_s": j.true_duration_s,
                "walltime_limit_s": j.walltime_est_s,
                "final_status": (
                    "COMPLETED"
                    if j.terminal_status_override is None
                    else j.terminal_status_override.name
                ),
            }
        )
    out = tmp_path / "rt.csv"
    write_trace(out, rows)
    reloaded = load_trace(out)
    assert len(reloaded) == len(jobs)
    for a, b in zip(jobs, reloaded):
        assert (a.id, a.tenant, a.job_class, a.submit_t) == (
            b.id,
            b.tenant,
            b.job_class,
            b.submit_t,
        )
        assert a.gangs[0].chips == b.gangs[0].chips
        assert a.true_duration_s == b.true_duration_s
        assert a.walltime_est_s == b.walltime_est_s
        assert a.terminal_status_override == b.terminal_status_override


def test_trace_source_protocol(tmp_path):
    src = TraceSource(SAMPLE)
    assert isinstance(src, JobSource)
    seen = []
    while (nxt := src.next_arrival()) is not None:
        seen.append(nxt)
    assert len(seen) == 30
    assert [t for t, _ in seen] == sorted(t for t, _ in seen)
    assert all(j.submit_t == t for t, j in seen)
    assert src.next_arrival() is None  # sticky exhaustion
    # From an unsorted job list: TraceSource sorts.
    jobs = load_trace(SAMPLE)
    src2 = TraceSource(reversed(jobs))
    first = src2.next_arrival()
    assert first is not None and first[1].id == "j000"


def write_lines(tmp_path, *rows):
    p = tmp_path / "t.csv"
    p.write_text("\n".join([HEADER, *rows]) + "\n")
    return p


def test_loader_errors(tmp_path):
    # Missing column.
    p = tmp_path / "bad.csv"
    p.write_text("job_id,user,tenant\nj0,u,t\n")
    with pytest.raises(ValueError, match="missing column"):
        load_trace(p)
    # Unknown status.
    p2 = write_lines(tmp_path, "j0,u,t0,eval,0,1,,1,60,,EXPLODED")
    with pytest.raises(ValueError, match="final_status"):
        load_trace(p2)
    # Unknown class.
    p3 = write_lines(tmp_path, "j0,u,t0,mining,0,1,,1,60,,COMPLETED")
    with pytest.raises(ValueError, match="class"):
        load_trace(p3)
    # Duplicate id.
    p4 = write_lines(
        tmp_path,
        "j0,u,t0,eval,0,1,,1,60,,COMPLETED",
        "j0,u,t0,eval,1,1,,1,60,,COMPLETED",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_trace(p4)
    # Bad chip count.
    p5 = write_lines(tmp_path, "j0,u,t0,eval,0,0,,1,60,,COMPLETED")
    with pytest.raises(ValueError, match="num_chips"):
        load_trace(p5)
    # Empty file.
    p6 = tmp_path / "empty.csv"
    p6.write_text("# only comments\n")
    with pytest.raises(ValueError, match="header"):
        load_trace(p6)


def test_loader_tolerates_comments_blanks_extra_columns(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "# comment\n\n"
        + HEADER
        + ",extra\n"
        + "# mid-file comment\n"
        + "j0,u,t0,EVAL,1000000,2,h100,1,60.5,300,completed,ignored\n"
    )
    jobs = load_trace(p)
    assert len(jobs) == 1
    assert jobs[0].submit_t == 1_000_000
    assert jobs[0].true_duration_s == 60.5
    assert jobs[0].gangs[0].chip_type == "h100"
    assert jobs[0].terminal_status_override is None


# ---------------------------------------------------------------------------
# Philly adapter
# ---------------------------------------------------------------------------


def philly_record(
    jobid,
    submitted,
    status="Pass",
    attempts=(),
    vc="vc1",
    user="alice",
):
    return {
        "jobid": jobid,
        "vc": vc,
        "user": user,
        "status": status,
        "submitted_time": submitted,
        "attempts": list(attempts),
    }


def attempt(start, end, gpus_per_server):
    return {
        "start_time": start,
        "end_time": end,
        "detail": [
            {"ip": f"m{i}", "gpus": [f"gpu{k}" for k in range(n)]}
            for i, n in enumerate(gpus_per_server)
        ],
    }


def test_convert_philly_mapping(tmp_path):
    records = [
        # Two attempts: 10 min on 4 GPUs, then 20 min on 8 GPUs across 2
        # servers -> duration 1800 s, widest attempt 8 GPUs / 2 nodes.
        philly_record(
            "job_a",
            "2017-10-01 00:00:00",
            status="Pass",
            attempts=[
                attempt("2017-10-01 00:05:00", "2017-10-01 00:15:00", [4]),
                attempt("2017-10-01 01:00:00", "2017-10-01 01:20:00", [4, 4]),
            ],
        ),
        # Killed 1-GPU job, submitted 1 hour later.
        philly_record(
            "job_b",
            "2017-10-01 01:00:00",
            status="Killed",
            attempts=[attempt("2017-10-01 01:01:00", "2017-10-01 01:31:00", [1])],
        ),
        # Failed job with one unparseable attempt time: duration counts
        # only the valid attempt.
        philly_record(
            "job_c",
            "2017-10-01 02:00:00",
            status="Failed",
            attempts=[
                {"start_time": None, "end_time": None, "detail": [{"ip": "m0", "gpus": ["g0", "g1"]}]},
                attempt("2017-10-01 02:10:00", "2017-10-01 02:15:00", [2]),
            ],
        ),
        # Skipped: never ran on hardware (no GPUs).
        philly_record("job_d", "2017-10-01 03:00:00", status="Pass", attempts=[]),
        # Skipped: unknown status.
        philly_record(
            "job_e",
            "2017-10-01 04:00:00",
            status="Queued",
            attempts=[attempt("2017-10-01 04:01:00", "2017-10-01 04:02:00", [1])],
        ),
        # Skipped: no submitted_time.
        philly_record("job_f", None, attempts=[attempt("2017-10-01 05:00:00", "2017-10-01 05:01:00", [1])]),
    ]
    p = tmp_path / "philly.json"
    p.write_text(json.dumps(records))
    rows = convert_philly(p)
    assert [r["job_id"] for r in rows] == ["job_a", "job_b", "job_c"]

    a, b, c = rows
    assert a["submit_time"] == 0  # epoch = earliest kept submit
    assert a["duration_s"] == 1800.0
    assert a["num_chips"] == 8
    assert a["num_nodes"] == 2
    assert a["final_status"] == "COMPLETED"
    assert a["tenant"] == "vc1" and a["user"] == "alice"
    assert a["class"] == "finetune" and a["chip_type"] == ""

    assert b["submit_time"] == 3600 * S
    assert b["duration_s"] == 1800.0
    assert b["final_status"] == "CANCELED"

    assert c["submit_time"] == 7200 * S
    assert c["duration_s"] == 300.0
    assert c["num_chips"] == 2
    assert c["final_status"] == "FAILED"


def test_convert_philly_round_trip_to_source(tmp_path):
    p = tmp_path / "philly.json"
    p.write_text(
        json.dumps(
            [
                philly_record(
                    "job_a",
                    "2017-10-01 00:00:00",
                    attempts=[attempt("2017-10-01 00:01:00", "2017-10-01 00:11:00", [8])],
                ),
                philly_record(
                    "job_b",
                    "2017-10-01 00:30:00",
                    status="Failed",
                    attempts=[attempt("2017-10-01 00:31:00", "2017-10-01 00:41:00", [1])],
                ),
            ]
        )
    )
    out = tmp_path / "canon.csv"
    write_trace(out, convert_philly(p))
    src = TraceSource(out)
    t0, j0 = src.next_arrival()
    t1, j1 = src.next_arrival()
    assert j0.id == "job_a" and t0 == 0
    assert j0.gangs[0].chips == 8
    assert j0.gangs[0].chip_type is None  # philly is unlabeled
    assert j0.terminal_status_override is None
    assert j1.id == "job_b" and t1 == 1800 * S
    assert j1.terminal_status_override is JobStatus.FAILED
    assert src.next_arrival() is None


def test_convert_philly_rejects_non_list(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"not_jobs": 1}')
    with pytest.raises(ValueError, match="list"):
        convert_philly(p)


# ---------------------------------------------------------------------------
# Engine replay: final_status verbatim
# ---------------------------------------------------------------------------


def test_engine_replays_final_status_verbatim():
    scn = load_scenario(
        {
            "sim": {"horizon": "1d", "round": "60s", "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": 8},
                        "topology": {"levels": ["node"], "counts": [8]},
                    }
                ],
            },
            "failure_model": {
                "node_mtbf_days": 0,
                "maintenance_rate_per_node_month": 0,
            },
            "workload": {"kind": "trace", "source": str(SAMPLE)},
        }
    )
    fleet = build_fleet(scn)
    src = TraceSource(SAMPLE)
    expected = {
        j.id: (
            JobStatus.COMPLETED
            if j.terminal_status_override is None
            else j.terminal_status_override
        )
        for j in src.jobs
    }
    sim = Simulator(scn, fleet, src, FIFOScheduler(strict=False), NullSink())
    sim.run()
    fleet.check_invariants()
    got = {j.id: j.status for j in src.jobs}
    assert got == expected
