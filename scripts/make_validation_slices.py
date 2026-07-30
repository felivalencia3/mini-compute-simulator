#!/usr/bin/env python3
"""Deterministically produce the vendored CI validation slices.

The v0.6 validation CI runs against SMALL, checked-in trace slices under
``tests/validation_traces/`` so the smoke tests need no network and finish
in seconds (validation plan §3/§5).  This script regenerates those slices
reproducibly.

For each trace it works in one of two modes:

* **real** — if the full published trace is available (an ``--helios-data``
  directory / ``--philly-json`` file, or a verified copy in the fetch
  cache), it converts the real rows and samples the documented slice.
  Byte-identical across runs (deterministic selection).
* **synthetic** — if the real trace is absent, it SYNTHESIZES a
  schema-faithful slice, clearly labelled ``SYNTHETIC`` in the header.  The
  CI smokes only assert converter-mapping and policy *direction* (not the
  papers' point numbers), so a schema-accurate synthetic slice is a valid
  fallback; the synthetic Helios slice is engineered to exhibit the
  FIFO-vs-SJF head-of-line-blocking effect the smoke checks.

Slices produced (both CANONICAL schema, see
:data:`fleetsim.workload.trace.CANONICAL_COLUMNS`):

* ``helios_venus_2vc_sept.csv`` — Helios Venus, two VCs, the September
  validation window (2020-09-01..2020-09-26), submit_time re-based to the
  window start.  Real mode selects VCs ``vcvGl`` (a loaded 20-node pool
  that shows the SJF advantage) and ``vcvlY`` (a small 2-node pool).
* ``philly_slice.csv`` — ~2,000 converted Philly rows spanning all three
  statuses (Passed / Killed / Unsuccessful), with the paper's by-count and
  by-GPU-time ordering invariants.

Every slice carries a header comment block (``#`` lines, skipped by
:func:`fleetsim.workload.trace.load_trace`) recording source, license,
the per-VC pool sizes, and the exact command/seed that produced it.

Usage::

    python scripts/make_validation_slices.py \
        [--out tests/validation_traces] \
        [--helios-data path/to/HeliosData/data] \
        [--philly-json path/to/cluster_job_log] \
        [--seed 20200926]

DETERMINISM: no wall clock; all randomness is seeded.  Re-running with the
same inputs and seed yields byte-identical slices.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import zipfile
from pathlib import Path

# Allow running from a source checkout without an install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from fleetsim.workload.trace import CANONICAL_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Documented slice parameters
# ---------------------------------------------------------------------------

HELIOS_SLICE_NAME = "helios_venus_2vc_sept.csv"
PHILLY_SLICE_NAME = "philly_slice.csv"

#: Which Venus VCs the real-mode Helios slice keeps.  vcvGl is a loaded
#: 20-node pool whose September jobs show the SJF-over-FIFO JCT advantage
#: (measured FIFO/SJF ~1.47); vcvlY is a small 2-node pool.  Together
#: ~1.9k rows with an aggregate FIFO/SJF ratio ~1.4 (> the 1.15 smoke bar).
HELIOS_SLICE_VCS = ("vcvGl", "vcvlY")
HELIOS_CLUSTER = "Venus"

_HELIOS_LICENSE = (
    "CC-BY-4.0 - Hu et al., 'Characterization and Prediction of Deep "
    "Learning Workloads in Large-Scale GPU Datacenters', SC '21, "
    "DOI 10.1145/3458817.3476223"
)
_HELIOS_ATTR_URL = "https://github.com/S-Lab-System-Group/HeliosData"
_PHILLY_LICENSE = (
    "CC-BY-4.0 - Jeon et al., 'Analysis of Large-Scale Multi-Tenant GPU "
    "Clusters for DNN Training Workloads', USENIX ATC '19"
)
_PHILLY_ATTR_URL = "https://github.com/msr-fiddle/philly-traces"


# ---------------------------------------------------------------------------
# Canonical CSV writer with a provenance header
# ---------------------------------------------------------------------------


def _write_slice(out_path: Path, rows, header_lines: list[str]) -> int:
    """Write canonical ``rows`` (dicts) to ``out_path`` preceded by a ``#``
    comment header.  Returns the number of data rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n" if line else "#\n")
        writer = csv.writer(f)
        writer.writerow(CANONICAL_COLUMNS)
        for row in rows:
            writer.writerow(
                ["" if row.get(c) is None else row.get(c, "") for c in CANONICAL_COLUMNS]
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Helios slice
# ---------------------------------------------------------------------------


def _find_helios_cluster_dir(root: Path, cluster: str) -> Path | None:
    """Locate ``<something>/<cluster>/cluster_log.csv`` under ``root``."""
    direct = root / cluster / "cluster_log.csv"
    if direct.is_file():
        return direct.parent
    for hit in root.rglob(f"{cluster}/cluster_log.csv"):
        return hit.parent
    return None


def _resolve_helios_dir(helios_data: str | None) -> Path | None:
    """Return a directory containing the Helios cluster CSVs, from the
    explicit path or the fetch cache (data.zip unzipped to a temp dir).
    ``None`` when no real trace is available."""
    if helios_data:
        return Path(helios_data).expanduser()
    try:
        from fleetsim.validation.fetch import cache_path, is_lfs_pointer

        zip_path = cache_path("helios")
    except Exception:
        return None
    if not zip_path.exists() or is_lfs_pointer(zip_path):
        return None
    try:
        tmp = Path(tempfile.mkdtemp(prefix="helios_slice_"))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        return tmp
    except Exception:
        return None


def build_helios_slice_real(cluster_dir: Path, out_path: Path, command: str) -> int:
    """Convert the real Helios cluster, window to September, keep the two
    documented VCs, and write the canonical slice."""
    from fleetsim.validation.helios import (
        HELIOS_SEPT_LAST_DAY,
        HELIOS_SEPT_MONTH,
        convert_helios,
        month_window,
    )

    log = cluster_dir / "cluster_log.csv"
    gpu = cluster_dir / "cluster_gpu_number.csv"
    df, pools = convert_helios(log, gpu if gpu.is_file() else None)
    sep = month_window(df, HELIOS_SEPT_MONTH, last_day=HELIOS_SEPT_LAST_DAY)
    keep = sep[sep["tenant"].isin(HELIOS_SLICE_VCS)].copy()
    keep = keep.sort_values(["submit_time", "job_id"], kind="stable")

    pool_desc = ", ".join(f"{vc}={pools.get(vc, '?')}" for vc in HELIOS_SLICE_VCS)
    header = [
        f"fleetsim validation slice: {HELIOS_SLICE_NAME}",
        f"source: HeliosData, cluster {HELIOS_CLUSTER}, cluster_log.csv (REAL trace)",
        f"license: {_HELIOS_LICENSE}",
        f"attribution: {_HELIOS_ATTR_URL}",
        f"selection: VCs {', '.join(HELIOS_SLICE_VCS)}; September window "
        f"{HELIOS_SEPT_MONTH}-01..{HELIOS_SEPT_MONTH}-{HELIOS_SEPT_LAST_DAY} "
        f"inclusive; canonical schema; submit_time re-based to window start",
        f"per-VC pool sizes (nodes, 8 GPU/node): {pool_desc}",
        f"produced by: {command}",
        "reproduce: extract HeliosData/data.zip and re-run the command above",
    ]
    return _write_slice(out_path, keep.to_dict("records"), header)


def build_helios_slice_synth(out_path: Path, seed: int, command: str) -> int:
    """Synthesize a schema-faithful Helios slice engineered to show the
    FIFO-vs-SJF head-of-line-blocking effect.

    Each of two single-node VCs releases a LONG full-pool job together with
    many SHORT full-pool jobs at the *same* instant.  On the one-node pool
    the jobs serialize.  FIFO (``submit_time`` then ``id``) runs the long
    job first — its zero-padded id sorts ahead of the shorts — so every
    short waits the long job out; SJF (``walltime_est`` = duration) runs the
    shorts first.  The mean-JCT ratio is comfortably above the 1.15 smoke
    bar (measured ~2x)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_short = 120
    for vc in ("vcSYN0", "vcSYN1"):
        # Long full-pool job (id sorts first -> FIFO head of line).
        rows.append(_helios_row(f"{vc}-000", vc, chips=8, submit_us=0, dur_s=6 * 3600))
        # Short full-pool jobs, all released with the long job.
        for k in range(1, n_short + 1):
            dur = int(rng.integers(120, 600))  # 2-10 min
            rows.append(
                _helios_row(f"{vc}-{k:03d}", vc, chips=8, submit_us=0, dur_s=dur)
            )
    rows.sort(key=lambda r: (r["submit_time"], r["job_id"]))
    header = [
        f"fleetsim validation slice: {HELIOS_SLICE_NAME}  [SYNTHETIC]",
        "source: SYNTHETIC (no real Helios trace was available at build time)",
        "  schema-faithful; engineered so FIFO/SJF mean-JCT ratio > 1.15",
        f"license: n/a (synthetic); real Helios is {_HELIOS_ATTR_URL} ({_HELIOS_LICENSE})",
        "per-VC pool sizes (nodes, 8 GPU/node): vcSYN0=1, vcSYN1=1",
        f"produced by: {command} (seed={seed})",
    ]
    return _write_slice(out_path, rows, header)


def _helios_row(job_id: str, vc: str, *, chips: int, submit_us: int, dur_s: int) -> dict:
    return {
        "job_id": job_id,
        "user": "uSYN",
        "tenant": vc,
        "class": "finetune",
        "submit_time": submit_us,
        "num_chips": chips,
        "chip_type": "",
        "num_nodes": 1,
        "duration_s": float(dur_s),
        "walltime_limit_s": float(dur_s),  # SJF-oracle (duration == estimate)
        "final_status": "COMPLETED",
    }


# ---------------------------------------------------------------------------
# Philly slice
# ---------------------------------------------------------------------------


def build_philly_slice_real(philly_json: Path, out_path: Path, command: str, n: int = 2000) -> int:
    """Convert the real Philly log and sample ~``n`` rows spanning all three
    statuses (proportional to their prevalence, but guaranteeing each is
    present)."""
    from fleetsim.workload.philly import convert_philly

    rows = convert_philly(philly_json)
    by_status: dict[str, list] = {}
    for r in rows:
        by_status.setdefault(str(r["final_status"]), []).append(r)
    # Proportional sample, at least a handful of each present status.
    total = sum(len(v) for v in by_status.values())
    picked: list = []
    for status, group in by_status.items():
        take = max(20, round(n * len(group) / total)) if total else 0
        picked.extend(group[: min(take, len(group))])
    picked.sort(key=lambda r: (r["submit_time"], r["job_id"]))
    header = [
        f"fleetsim validation slice: {PHILLY_SLICE_NAME}",
        "source: msr-fiddle/philly-traces cluster_job_log (REAL trace)",
        f"license: {_PHILLY_LICENSE}",
        f"attribution: {_PHILLY_ATTR_URL}",
        f"selection: proportional sample (~{n} rows) spanning all statuses",
        f"produced by: {command}",
    ]
    return _write_slice(out_path, picked, header)


def build_philly_slice_synth(out_path: Path, seed: int, command: str, n: int = 2000) -> int:
    """Synthesize ~``n`` converted-Philly canonical rows spanning the three
    statuses, honouring the paper's invariants: by count Pass > Unsuccessful
    > Killed, and Killed+Unsuccessful hold a larger share of GPU-time than of
    job count (they run longer / wider before dying)."""
    rng = np.random.default_rng(seed)
    # Status mix (~ ATC '19 Table 6 by count: Pass 69% / Unsucc 17% / Killed 14%).
    statuses = rng.choice(
        ["COMPLETED", "FAILED", "CANCELED"],
        size=n,
        p=[0.69, 0.17, 0.14],
    )

    def _chips() -> int:
        # Philly is dominated by 1-GPU jobs, with a multi-GPU tail.
        r = rng.random()
        if r < 0.60:
            return 1
        if r < 0.85:
            return int(rng.integers(2, 5))  # 2-4
        if r < 0.95:
            return int(rng.integers(5, 9))  # 5-8
        return int(rng.choice([16, 32]))  # >8

    rows: list[dict] = []
    t_us = 0
    for i, status in enumerate(statuses):
        chips = _chips()
        # Passed jobs are shorter; killed/failed ran long before dying, so
        # Killed+Unsuccessful carry disproportionate GPU-time.
        if status == "COMPLETED":
            dur = float(rng.lognormal(mean=np.log(1800.0), sigma=0.8))
        else:
            dur = float(rng.lognormal(mean=np.log(14400.0), sigma=0.8))
        t_us += int(rng.integers(1, 120) * 1_000_000)
        rows.append(
            {
                "job_id": f"phly{i}",
                "user": f"u{i % 97}",
                "tenant": "vcSYN",
                "class": "finetune",
                "submit_time": t_us,
                "num_chips": chips,
                "chip_type": "",
                "num_nodes": -(-chips // 8),  # ceil(chips/8)
                "duration_s": round(dur, 3),
                "walltime_limit_s": "",  # Philly records no walltime estimate
                "final_status": str(status),
            }
        )
    rows.sort(key=lambda r: (r["submit_time"], r["job_id"]))
    _assert_philly_invariants(rows)
    header = [
        f"fleetsim validation slice: {PHILLY_SLICE_NAME}  [SYNTHETIC]",
        "source: SYNTHETIC (no real Philly trace was available at build time)",
        "  schema-faithful; status shares approximate ATC '19 Table 6 and honour",
        "  by-count Pass>Unsuccessful>Killed and GPU-time(Killed+Unsucc) > count share",
        f"license: n/a (synthetic); real Philly is {_PHILLY_ATTR_URL} ({_PHILLY_LICENSE})",
        f"produced by: {command} (seed={seed})",
    ]
    return _write_slice(out_path, rows, header)


def _assert_philly_invariants(rows: list[dict]) -> None:
    """Fail fast if the synthetic Philly slice violates the paper invariants
    the V3 smoke asserts, so we never commit a bad slice."""
    from fleetsim.validation.philly_status import (
        status_split_by_count,
        status_split_by_gpu_time,
    )

    by_count = status_split_by_count(rows)
    by_time = status_split_by_gpu_time(rows)
    assert by_count["Passed"] > by_count["Unsuccessful"] > by_count["Killed"], by_count
    ku_count = by_count["Killed"] + by_count["Unsuccessful"]
    ku_time = by_time["Killed"] + by_time["Unsuccessful"]
    assert ku_time > ku_count, (ku_time, ku_count)
    assert abs(sum(by_count.values()) - 1.0) < 1e-9
    assert abs(sum(by_time.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "tests" / "validation_traces"),
        help="output directory for the vendored slices",
    )
    parser.add_argument(
        "--helios-data",
        default=None,
        help="directory containing the extracted Helios data (<...>/Venus/cluster_log.csv)",
    )
    parser.add_argument(
        "--philly-json",
        default=None,
        help="path to the real Philly cluster_job_log JSON",
    )
    parser.add_argument("--seed", type=int, default=20200926, help="synthetic RNG seed")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    command = "python scripts/make_validation_slices.py"

    # --- Helios ---
    helios_out = out_dir / HELIOS_SLICE_NAME
    cluster_dir = None
    root = _resolve_helios_dir(args.helios_data)
    if root is not None:
        cluster_dir = _find_helios_cluster_dir(root, HELIOS_CLUSTER)
    if cluster_dir is not None:
        n = build_helios_slice_real(cluster_dir, helios_out, command)
        print(f"Helios slice (real): {helios_out} ({n} rows)")
    else:
        n = build_helios_slice_synth(helios_out, args.seed, command)
        print(f"Helios slice (SYNTHETIC): {helios_out} ({n} rows)")

    # --- Philly ---
    philly_out = out_dir / PHILLY_SLICE_NAME
    if args.philly_json and Path(args.philly_json).is_file():
        n = build_philly_slice_real(Path(args.philly_json), philly_out, command)
        print(f"Philly slice (real): {philly_out} ({n} rows)")
    else:
        n = build_philly_slice_synth(philly_out, args.seed, command)
        print(f"Philly slice (SYNTHETIC): {philly_out} ({n} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
