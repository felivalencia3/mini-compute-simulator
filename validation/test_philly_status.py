"""V3 — Philly job-status split (plan §2 V3): converter-fidelity validation.

The single most directly replayable scheduler-trace fact — the fraction of
jobs in each terminal status, by **count** and by **GPU-time** — is a
property of the *converted* rows, so it tests the converter's fidelity to
the Philly paper (Jeon et al., USENIX ATC '19, Table 6), not the scheduler.
No simulation is involved.

Two rungs:

- **CI smoke** (always runs, < 5 s, no network): the vendored ~2,000-row
  Philly slice (``tests/validation_traces/philly_slice.csv``).  Checks the
  converter's status mapping (Pass->COMPLETED / Killed->CANCELED /
  Failed->FAILED) end to end on a hand-built record, and that on the slice
  the three status shares sum to 1 and hold the paper's ordering
  invariants (by-count ``Passed > Unsuccessful > Killed``; the
  Killed+Unsuccessful GPU-time share exceeds their job-count share).  The
  vendored slice is SYNTHETIC — it honours the ordering the paper reports
  but does NOT reproduce the paper's exact percentages, so the smoke rung
  asserts only structure, never point values.
- **Full trace** (opt-in, ``@pytest.mark.trace_full`` + ``FLEETSIM_PHILLY_FULL``):
  downloads the real 1 GB Git-LFS ``trace-data.tar.gz``, converts the whole
  ``cluster_job_log``, and asserts the Table-6 shares within the plan's
  tolerance (by-count ±5 pp, by-GPU-time ±8 pp) plus the ordering
  invariant.  This rung is written to the plan spec but is **UNVERIFIED on
  the real trace in this build** (the 1 GB LFS artifact was not fetched
  here); it requires ``git lfs install && git lfs pull`` before it can run.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from fleetsim.validation.philly_status import (
    PHILLY_STATUS_LABELS,
    status_split_by_count,
    status_split_by_gpu_time,
)
from fleetsim.workload.philly import convert_philly

# Repo layout: this file is <root>/validation/test_philly_status.py.
_ROOT = Path(__file__).resolve().parents[1]
_SLICE = _ROOT / "tests" / "validation_traces" / "philly_slice.csv"


# ---------------------------------------------------------------------------
# CI smoke — converter mapping + vendored-slice structure (always runs)
# ---------------------------------------------------------------------------


def _mk_philly_record(jobid: str, status: str, gpus: int, dur_s: int) -> dict:
    """A minimal raw Philly ``cluster_job_log`` record: one attempt on one
    server with ``gpus`` GPUs, running ``dur_s`` seconds."""
    return {
        "jobid": jobid,
        "user": "u",
        "vc": "vcTEST",
        "submitted_time": "2017-10-01 00:00:00",
        "status": status,
        "attempts": [
            {
                "start_time": "2017-10-01 00:00:00",
                "end_time": f"2017-10-01 {dur_s // 3600:02d}:00:00",
                "detail": [{"gpus": [f"gpu{i}" for i in range(gpus)]}],
            }
        ],
    }


def test_convert_philly_status_mapping(tmp_path: Path) -> None:
    """CI smoke: :func:`convert_philly` maps the three Philly raw states to
    canonical statuses exactly (plan §2 V3(c) / DESIGN §10): Pass ->
    COMPLETED, Killed -> CANCELED, Failed -> FAILED.  This is the mapping
    the whole by-count / by-GPU-time split relies on."""
    log = tmp_path / "cluster_job_log"
    log.write_text(
        json.dumps(
            [
                _mk_philly_record("p", "Pass", gpus=1, dur_s=3600),
                _mk_philly_record("k", "Killed", gpus=1, dur_s=3600),
                _mk_philly_record("f", "Failed", gpus=1, dur_s=3600),
            ]
        ),
        encoding="utf-8",
    )
    rows = convert_philly(log)
    by_id = {r["job_id"]: r["final_status"] for r in rows}
    assert by_id == {"p": "COMPLETED", "k": "CANCELED", "f": "FAILED"}
    # And the paper-bucket relabel the split helpers apply is the inverse.
    assert PHILLY_STATUS_LABELS == {
        "COMPLETED": "Passed",
        "CANCELED": "Killed",
        "FAILED": "Unsuccessful",
    }


def _load_slice() -> pd.DataFrame:
    """The vendored Philly slice as a canonical DataFrame (``#`` header
    comments skipped)."""
    return pd.read_csv(_SLICE, comment="#")


def test_philly_slice_status_split_smoke() -> None:
    """CI smoke: on the vendored Philly slice, both splits sum to 1 and
    hold the paper's ordering invariants (plan §2 V3(f)) — by-count
    ``Passed > Unsuccessful > Killed`` and Killed+Unsuccessful consuming a
    LARGER share of GPU-time than of the headcount.  Structure only: the
    slice is synthetic, so no point-value assertion (that is the opt-in
    full-trace rung)."""
    df = _load_slice()

    by_count = status_split_by_count(df)
    by_gpu = status_split_by_gpu_time(df)

    # All three Philly buckets present, shares sum to 1 (both splits).
    assert set(by_count) == {"Passed", "Killed", "Unsuccessful"}
    assert set(by_gpu) == {"Passed", "Killed", "Unsuccessful"}
    assert abs(sum(by_count.values()) - 1.0) < 1e-9
    assert abs(sum(by_gpu.values()) - 1.0) < 1e-9

    # By-count ordering (Table 6): most jobs Pass; more fail than are killed.
    assert by_count["Passed"] > by_count["Unsuccessful"] > by_count["Killed"]

    # The paper's actual finding: the unsuccessful/killed jobs are a
    # MINORITY of the headcount but a MAJORITY-tilted share of GPU-time
    # (wasted work) — so their GPU-time share strictly exceeds their count
    # share.  Held exactly (plan §2 V3(f) ordering invariant).
    ku_count = by_count["Killed"] + by_count["Unsuccessful"]
    ku_gpu = by_gpu["Killed"] + by_gpu["Unsuccessful"]
    assert ku_gpu > ku_count, (ku_gpu, ku_count)


# ---------------------------------------------------------------------------
# Full trace — opt-in (1 GB Git-LFS; asserts §2 V3 Table-6 shares)
# ---------------------------------------------------------------------------

#: Published Table 6 shares (Jeon et al., ATC '19, 96,260-job paper window).
_PUB_BY_COUNT = {"Passed": 0.693, "Killed": 0.135, "Unsuccessful": 0.172}
_PUB_BY_GPU = {"Passed": 0.4453, "Killed": 0.3769, "Unsuccessful": 0.1776}

#: plan §2 V3(f) tolerances: the released 117,325-job / 137-day trace is not
#: the paper's 96,260-job / ~75-day window (which is not precisely
#: published), so the bands are loose enough to absorb the window residual.
_BY_COUNT_TOL = 0.05  # +/- 5 percentage points
_BY_GPU_TOL = 0.08  # +/- 8 percentage points


def _resolve_philly_log() -> Path:
    """Fetch + extract the real Philly trace and return the path to
    ``cluster_job_log`` (the JSON array, no file extension).

    Skips (with the ``git lfs pull`` remediation) when the fetched artifact
    is a Git-LFS pointer rather than the real 1 GB tarball — the common
    opt-in failure mode.
    """
    from fleetsim.validation.fetch import LFSPointerError, fetch_trace, is_lfs_pointer

    try:
        tarball = fetch_trace("philly")
    except LFSPointerError as exc:  # pragma: no cover - opt-in path
        pytest.skip(f"Philly trace is a Git-LFS pointer, not the data: {exc}")
    if is_lfs_pointer(tarball):  # pragma: no cover - opt-in path
        pytest.skip(
            "Philly trace-data.tar.gz is a Git-LFS pointer; run "
            "`git lfs install && git lfs pull` in the philly-traces checkout"
        )
    extract_root = tarball.parent / "extracted"
    hits = (
        list(extract_root.rglob("cluster_job_log")) if extract_root.is_dir() else []
    )
    if not hits:  # pragma: no cover - opt-in path (extract once, idempotently)
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(extract_root)
        hits = list(extract_root.rglob("cluster_job_log"))
    if not hits:  # pragma: no cover - opt-in path
        pytest.skip(f"cluster_job_log not found under {extract_root}")
    return hits[0]


@pytest.mark.trace_full
@pytest.mark.skipif(
    not os.environ.get("FLEETSIM_PHILLY_FULL"),
    reason="set FLEETSIM_PHILLY_FULL=1 (and `git lfs pull`) to fetch and "
    "convert the full 1 GB Philly trace",
)
def test_philly_full_status_split_table6() -> None:
    """Opt-in full replay: convert the whole released Philly trace and
    assert the Table-6 status split (Jeon et al., ATC '19).

    Asserts (plan §2 V3(f)): each by-count share within ±5 pp of Passed
    69.3% / Killed 13.5% / Unsuccessful 17.2%; each by-GPU-time share
    within ±8 pp of Passed 44.53% / Killed 37.69% / Unsuccessful 17.76%;
    and the ordering invariant that Killed+Unsuccessful consume a larger
    GPU-time share than their headcount share.

    UNVERIFIED-ON-REAL-DATA NOTE: the 1 GB Git-LFS artifact was not fetched
    when this test was written, so the exact fleetsim-vs-published deltas
    are not yet recorded (unlike the Helios V1 rung).  The tolerances follow
    the plan; a maintainer with the trace should record the measured shares
    in docs/validation.md and tighten if warranted.
    """
    log = _resolve_philly_log()
    rows = convert_philly(log)
    assert rows, "convert_philly returned no rows from the real trace"

    by_count = status_split_by_count(rows)
    by_gpu = status_split_by_gpu_time(rows)

    for label, target in _PUB_BY_COUNT.items():
        assert abs(by_count.get(label, 0.0) - target) <= _BY_COUNT_TOL, (
            f"by-count {label} {by_count.get(label, 0.0):.3f} outside "
            f"{target:.3f} +/- {_BY_COUNT_TOL} (Table 6); all={by_count}"
        )
    for label, target in _PUB_BY_GPU.items():
        assert abs(by_gpu.get(label, 0.0) - target) <= _BY_GPU_TOL, (
            f"by-GPU-time {label} {by_gpu.get(label, 0.0):.3f} outside "
            f"{target:.3f} +/- {_BY_GPU_TOL} (Table 6); all={by_gpu}"
        )

    # Ordering invariant (the paper's actual finding), held exactly.
    assert by_count["Passed"] > by_count["Unsuccessful"] > by_count["Killed"]
    ku_count = by_count["Killed"] + by_count["Unsuccessful"]
    ku_gpu = by_gpu["Killed"] + by_gpu["Unsuccessful"]
    assert ku_gpu > ku_count, (ku_gpu, ku_count)
