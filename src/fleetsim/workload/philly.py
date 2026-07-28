"""Adapter for the Microsoft Philly trace (msr-fiddle/philly-traces).

:func:`convert_philly` reads the published ``cluster_job_log`` JSON (a
list of job records; a mapping with a ``"jobs"`` list is also accepted)
and emits rows in fleetsim's canonical trace schema
(:data:`fleetsim.workload.trace.CANONICAL_COLUMNS`), ready for
:func:`fleetsim.workload.trace.write_trace` /
:class:`fleetsim.workload.trace.TraceSource`.

PHILLY -> CANONICAL MAPPING (pinned)
------------------------------------
=================  ====================================================
canonical column   source
=================  ====================================================
job_id             ``jobid``
user               ``user`` (empty string when absent)
tenant             ``vc`` (the virtual cluster; ``"unknown"`` if absent)
class              ``"finetune"`` for every row (Philly does not record
                   a class; its jobs are DL training jobs)
submit_time        ``submitted_time`` parsed as ``%Y-%m-%d %H:%M:%S``,
                   converted to int MICROSECONDS relative to the
                   earliest kept row (the trace epoch)
num_chips          GPUs of the WIDEST attempt: max over ``attempts`` of
                   the total ``gpus`` across that attempt's ``detail``
                   servers
chip_type          empty (Philly is unlabeled -> unpinned gang)
num_nodes          server count (``len(detail)``) of that widest attempt
duration_s         SUM over attempts of ``end_time - start_time`` in
                   seconds (per-attempt records -> total runtime across
                   retries); attempts with missing/unparseable times
                   contribute 0
walltime_limit_s   empty (not recorded by Philly)
final_status       ``Pass`` -> COMPLETED, ``Killed`` -> CANCELED,
                   ``Failed`` -> FAILED
=================  ====================================================

Rows are SKIPPED (not errors) when: ``submitted_time`` is missing or
unparseable, no attempt reports any GPUs (the job never ran on
hardware), or ``status`` is not one of the three known values.  Negative
per-attempt durations (clock skew in the raw trace) contribute 0.

INVARIANTS: a pure function of the file — no randomness, no wall clock;
rows are returned sorted by ``(submit_time, job_id)``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

__all__ = ["convert_philly"]

_TIME_FMT = "%Y-%m-%d %H:%M:%S"

_STATUS = {"Pass": "COMPLETED", "Killed": "CANCELED", "Failed": "FAILED"}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), _TIME_FMT)
    except ValueError:
        return None


def convert_philly(path: str | Path) -> list[dict[str, object]]:
    """Convert a Philly ``cluster_job_log`` JSON file to canonical rows.

    Returns a list of dicts keyed by the canonical column names (see the
    module docstring for the mapping), sorted by ``(submit_time,
    job_id)``.  Raises ``ValueError`` if the document is not a list of
    job records (or a mapping containing one under ``"jobs"``).
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, Mapping):
        doc = doc.get("jobs")
    if not isinstance(doc, list):
        raise ValueError(
            f"{path}: expected a JSON list of Philly job records"
            f" (or a mapping with a 'jobs' list)"
        )

    kept: list[tuple[datetime, dict[str, object]]] = []
    for rec in doc:
        if not isinstance(rec, Mapping):
            continue
        submitted = _parse_time(rec.get("submitted_time"))
        if submitted is None:
            continue
        status_raw = rec.get("status")
        status = _STATUS.get(status_raw) if isinstance(status_raw, str) else None
        if status is None:
            continue

        duration_s = 0.0
        best_gpus = 0
        best_nodes = 0
        for attempt in rec.get("attempts") or []:
            if not isinstance(attempt, Mapping):
                continue
            detail = attempt.get("detail") or []
            gpus = sum(
                len(server.get("gpus") or [])
                for server in detail
                if isinstance(server, Mapping)
            )
            if gpus > best_gpus:
                best_gpus = gpus
                best_nodes = len(detail)
            start = _parse_time(attempt.get("start_time"))
            end = _parse_time(attempt.get("end_time"))
            if start is not None and end is not None:
                duration_s += max(0.0, (end - start).total_seconds())
        if best_gpus <= 0:
            continue

        kept.append(
            (
                submitted,
                {
                    "job_id": str(rec.get("jobid")),
                    "user": str(rec.get("user") or ""),
                    "tenant": str(rec.get("vc") or "unknown"),
                    "class": "finetune",
                    "submit_time": 0,  # filled in below, relative to epoch
                    "num_chips": best_gpus,
                    "chip_type": "",
                    "num_nodes": best_nodes,
                    "duration_s": round(duration_s, 6),
                    "walltime_limit_s": "",
                    "final_status": status,
                },
            )
        )

    if not kept:
        return []
    epoch = min(sub for sub, _ in kept)
    rows: list[dict[str, object]] = []
    for sub, row in kept:
        row["submit_time"] = int(round((sub - epoch).total_seconds() * 1e6))
        rows.append(row)
    rows.sort(key=lambda r: (r["submit_time"], r["job_id"]))
    return rows
