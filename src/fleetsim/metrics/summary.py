"""Summary building and output writing for the metrics pipeline.

Consumes a :class:`~fleetsim.metrics.collector.MetricsCollector` (its
read-side API only) and produces DESIGN §9's outputs: ``jobs.parquet``,
``timeseries.parquet``, ``summary.json``, plus a pretty console table.

Every distributional metric is reported BOTH job-weighted and
chip-hour-weighted (mice vs hogs); the chip-hour weight of a job is
``chips * running_elapsed_s / 3600`` (wall time holding an allocation).
Summaries are computed over the FULL run and over the steady-state
WINDOW.  Scope membership (window = closed ``[w0, w1]``):

- queue-wait stats: jobs whose ``first_start_t`` is in scope (reported
  per JobClass, per SOURCE CLASS — the workload-class label, so a
  closed-loop backlog reusing another JobClass never pollutes it — and
  bucketed by gang chip count, DESIGN §9).  BEST_EFFORT-tier jobs are
  EXCLUDED from every queue-wait/JCT distribution (DESIGN §16.1 /
  traffic-math §2.1: report best-effort goodput, never best-effort
  wait — undefined under a saturated closed loop);
- JCT stats: COMPLETED jobs whose ``end_t`` is in scope;
- ETTR / status counts: terminal jobs whose ``end_t`` is in scope;
- goodput: the collector's productive chip-time integral for the scope
  (stint-interval-spread, so window goodput <= 1 by construction and
  still-running jobs' checkpointed progress counts) over the
  scope-clipped allocated integral;
- per-tenant ``n_jobs_submitted``: ``submit_t`` in scope.

Percentiles use the weighted inverted-CDF rule: the smallest value whose
cumulative weight reaches ``q * total_weight`` (job-weighted p50 of
``[1, 2]`` is 1).  Deterministic: values sort ascending, ties stable.

UNITS: all reported times are float seconds (``*_s``), chip-hours where
named; rates are per minute.  INVARIANTS: pure functions of the collector
state — no wall clock, no randomness; JSON keys serialize sorted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .collector import MetricsCollector

__all__ = [
    "build_summary",
    "write_outputs",
    "jobs_dataframe",
    "timeseries_dataframe",
    "format_summary_table",
]

_JOB_STR_COLS = frozenset(
    {"job_id", "tenant", "job_class", "tier", "source_class", "chip_type", "status"}
)
_JOB_NULLABLE_INT_COLS = frozenset(
    {"first_start_t_us", "end_t_us", "n_domains_spanned"}
)
_JOB_FLOAT_COLS = frozenset(
    {
        "productive_chip_s",
        "lost_chip_s",
        "running_elapsed_s",
        "queue_wait_s",
        "jct_s",
        "ettr",
    }
)

#: Fixed timeseries columns (per-level frag columns are appended by the
#: collector after the first flush).
_TS_BASE_COLS = (
    "t_us",
    "allocated_chips",
    "healthy_chips",
    "pending_jobs",
    "running_jobs",
    "occupancy_to_date",
    "allocation_rate_to_date",
    "goodput_to_date",
    "cum_preemptions",
    "cum_failure_kills",
    "cum_node_failures",
    "stranded_chips",
)

_TERMINAL_STATUSES = frozenset(
    {"COMPLETED", "FAILED", "CANCELED", "TIMEOUT", "NODE_FAIL"}
)


# ---------------------------------------------------------------------------
# Weighted distribution statistics
# ---------------------------------------------------------------------------


def _dist_stats(
    pairs: list[tuple[float, float]], percentiles: tuple[int, ...]
) -> dict[str, float] | None:
    """Stats over ``(value, weight)`` pairs: ``n``, weighted ``mean`` and
    the requested percentiles (weighted inverted CDF).  Pairs with
    non-positive weight or ``None`` value are dropped; returns ``None``
    when nothing remains."""
    pts = [(float(v), float(w)) for v, w in pairs if v is not None and w > 0]
    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    total_w = sum(w for _, w in pts)
    out: dict[str, float] = {
        "n": len(pts),
        "mean": sum(v * w for v, w in pts) / total_w,
    }
    for q in percentiles:
        target = total_w * (q / 100.0)
        acc = 0.0
        val = pts[-1][0]
        for v, w in pts:
            acc += w
            if acc >= target - 1e-12:
                val = v
                break
        out[f"p{q}"] = val
    return out


def _both_weightings(
    rows: list[dict[str, Any]], value_key: str, percentiles: tuple[int, ...]
) -> dict[str, dict[str, float] | None]:
    """The job-weighted and chip-hour-weighted stats of one value column.
    Chip-hour weight = ``chips * running_elapsed_s / 3600``; jobs that
    never held chips drop out of the chip-hour weighting."""
    return {
        "job_weighted": _dist_stats(
            [(r[value_key], 1.0) for r in rows], percentiles
        ),
        "chip_hour_weighted": _dist_stats(
            [
                (r[value_key], r["chips"] * r["running_elapsed_s"] / 3600.0)
                for r in rows
            ],
            percentiles,
        ),
    }


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


#: Gang chip-count buckets for the DESIGN §9 "queue wait bucketed by chip
#: count" dimension (upper bound inclusive; None = unbounded).
_CHIP_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("1-8", 8),
    ("9-64", 64),
    ("65-512", 512),
    ("513+", None),
)


def _chip_bucket(chips: int) -> str:
    for name, hi in _CHIP_BUCKETS:
        if hi is None or chips <= hi:
            return name
    return _CHIP_BUCKETS[-1][0]  # pragma: no cover - last bucket is open


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _scope_summary(collector: "MetricsCollector", scope: str) -> dict[str, Any]:
    ints = collector.integral_report()[scope]
    counts = collector.event_counts()[scope]
    frag = collector.frag_stats()[scope]
    rows = collector.job_rows()
    if scope == "full":
        w0, w1 = 0, collector.horizon_us
    else:
        w0, w1 = collector.window

    def in_scope(t: int | None) -> bool:
        return t is not None and w0 <= t <= w1

    started = [r for r in rows if in_scope(r["first_start_t_us"])]
    ended = [r for r in rows if in_scope(r["end_t_us"])]
    completed = [r for r in ended if r["status"] == "COMPLETED"]
    # DESIGN §16.1 / traffic-math §2.1: never report best-effort wait/JCT
    # distributions (undefined under a saturated closed loop; blending
    # them into the JobClass they reuse misstates the open-loop class).
    # Their goodput/occupancy contributions remain in the integrals.
    started_open = [r for r in started if r["tier"] != "BEST_EFFORT"]
    completed_open = [r for r in completed if r["tier"] != "BEST_EFFORT"]

    duration_s = ints["duration_s"]
    minutes = duration_s / 60.0
    alloc_chip_s = ints["allocated_chip_s"]
    # Goodput numerator: the interval-spread productive integral (includes
    # still-running jobs' checkpointed progress; never exceeds allocated).
    productive = ints["productive_chip_s"]

    def per_class(
        pool: list[dict[str, Any]], key: str, pcts: tuple[int, ...]
    ) -> dict[str, Any]:
        classes = sorted({r["job_class"] for r in pool})
        return {
            c: _both_weightings([r for r in pool if r["job_class"] == c], key, pcts)
            for c in classes
        }

    def _src(r: dict[str, Any]) -> str:
        # Workload-class label; hand-built/trace jobs (source_class None)
        # fall back to the JobClass name.
        return r["source_class"] if r["source_class"] is not None else r["job_class"]

    def per_source_class(
        pool: list[dict[str, Any]], key: str, pcts: tuple[int, ...]
    ) -> dict[str, Any]:
        classes = sorted({_src(r) for r in pool})
        return {
            c: _both_weightings([r for r in pool if _src(r) == c], key, pcts)
            for c in classes
        }

    trig = counts["preemptions_by_trigger"]
    preempt_total = sum(trig.values())
    preemptions_per_min = {k: v / minutes for k, v in sorted(trig.items())}
    preemptions_per_min["node_failure"] = counts["failure_kills"] / minutes
    preemptions_per_min["total"] = (
        preempt_total + counts["failure_kills"]
    ) / minutes

    tenants = sorted({r["tenant"] for r in rows})
    per_tenant: dict[str, Any] = {}
    for tenant in tenants:
        t_started = [r for r in started if r["tenant"] == tenant]
        mqw = _dist_stats(
            [(r["queue_wait_s"], 1.0) for r in t_started], (50,)
        )
        per_tenant[tenant] = {
            "chip_hours": ints["allocated_chip_s_by_tenant"].get(tenant, 0.0)
            / 3600.0,
            "median_queue_wait_s": mqw["p50"] if mqw else None,
            "n_jobs_submitted": sum(
                1 for r in rows if r["tenant"] == tenant and in_scope(r["submit_t_us"])
            ),
        }

    by_status: dict[str, int] = {}
    for r in ended:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    return {
        "duration_s": duration_s,
        "occupancy": _ratio(alloc_chip_s, ints["healthy_chip_s"]),
        "allocation_rate": _ratio(alloc_chip_s, ints["total_chip_s"]),
        "goodput": _ratio(productive, alloc_chip_s),
        "allocated_chip_hours": {
            "total": alloc_chip_s / 3600.0,
            "by_chip_type": {
                k: v / 3600.0
                for k, v in ints["allocated_chip_s_by_type"].items()
            },
            "by_class": {
                k: v / 3600.0
                for k, v in ints["allocated_chip_s_by_class"].items()
            },
        },
        "queue_wait_s": per_class(started_open, "queue_wait_s", (50, 90, 99)),
        "queue_wait_s_by_source_class": per_source_class(
            started_open, "queue_wait_s", (50, 90, 99)
        ),
        "queue_wait_s_by_chips": {
            bucket: _both_weightings(pool, "queue_wait_s", (50, 90, 99))
            for bucket, _ in _CHIP_BUCKETS
            if (
                pool := [
                    r for r in started_open if _chip_bucket(r["chips"]) == bucket
                ]
            )
        },
        "jct_s": per_class(completed_open, "jct_s", (50, 90, 99)),
        "jct_s_by_source_class": per_source_class(
            completed_open, "jct_s", (50, 90, 99)
        ),
        "ettr": _both_weightings(
            [r for r in ended if r["ettr"] is not None], "ettr", (10, 50, 90)
        ),
        "preemptions_per_min": preemptions_per_min,
        "counts": {
            "node_failures": counts["node_failures"],
            "node_failures_by_cause": counts["node_failures_by_cause"],
            "node_repairs": counts["node_repairs"],
            "drains_started": counts["drains_started"],
            "failure_kills": counts["failure_kills"],
            "preemptions": preempt_total,
            "jobs_started": len(started),
            "jobs_finished": len(ended),
            "jobs_by_status": dict(sorted(by_status.items())),
        },
        "fragmentation": frag,
        "mean_pending_by_class": {
            k: v / duration_s if duration_s > 0 else 0.0
            for k, v in ints["pending_job_s_by_class"].items()
        },
        "per_tenant": per_tenant,
        "replica_availability": _ratio(
            ints["replica_running_s"], ints["replica_desired_s"]
        ),
    }


def build_summary(collector: "MetricsCollector") -> dict[str, Any]:
    """The full summary dict: run metadata plus one scope dict each for
    the full run and the steady-state window (see module docstring for
    the exact scoping rules)."""
    w0, w1 = collector.window
    return {
        "horizon_us": collector.horizon_us,
        "steady_state_window": {
            "warmup_frac": collector.warmup_frac,
            "drain_frac": collector.drain_frac,
            "start_us": w0,
            "end_us": w1,
        },
        "full": _scope_summary(collector, "full"),
        "window": _scope_summary(collector, "window"),
    }


# ---------------------------------------------------------------------------
# DataFrames and file outputs
# ---------------------------------------------------------------------------


def jobs_dataframe(collector: "MetricsCollector") -> pd.DataFrame:
    """Per-job records as a typed DataFrame (rows sorted by
    ``(submit_t, job_id)``; nullable ints use pandas ``Int64``)."""
    rows = collector.job_rows()
    trigger_cols = [f"n_preempt_{t}" for t in collector.preempt_triggers()]
    cols = [
        "job_id",
        "tenant",
        "job_class",
        "tier",
        "source_class",
        "chips",
        "chip_type",
        "submit_t_us",
        "first_start_t_us",
        "end_t_us",
        "status",
        "n_starts",
        "n_restarts",
        "n_domains_spanned",
        "n_preemptions",
        *trigger_cols,
        "n_node_failures",
        "productive_chip_s",
        "lost_chip_s",
        "running_elapsed_s",
        "queue_wait_s",
        "jct_s",
        "ettr",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c in _JOB_STR_COLS:
            df[c] = df[c].astype("string")
        elif c in _JOB_NULLABLE_INT_COLS:
            df[c] = df[c].astype("Int64")
        elif c in _JOB_FLOAT_COLS:
            df[c] = df[c].astype("float64")
        else:
            df[c] = df[c].astype("int64")
    return df


def timeseries_dataframe(collector: "MetricsCollector") -> pd.DataFrame:
    """Flush samples as a typed DataFrame in flush order."""
    rows = collector.timeseries_rows()
    cols = list(rows[0].keys()) if rows else list(_TS_BASE_COLS)
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c.endswith("_to_date") or c.startswith("frag_index_"):
            df[c] = df[c].astype("float64")
        else:
            df[c] = df[c].astype("int64")
    return df


def write_outputs(collector: "MetricsCollector", out_dir: str | Path) -> dict[str, Any]:
    """Write ``jobs.parquet``, ``timeseries.parquet`` and ``summary.json``
    into ``out_dir`` (created if missing) and return the summary dict.

    INVARIANT: byte-identical outputs for identical collector state (JSON
    keys sorted; DataFrame column order pinned)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs_dataframe(collector).to_parquet(out / "jobs.parquet", index=False)
    timeseries_dataframe(collector).to_parquet(
        out / "timeseries.parquet", index=False
    )
    summary = build_summary(collector)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = "{:.3f}") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):  # pragma: no cover - defensive
        return str(value)
    if isinstance(value, int):
        return str(value)
    return spec.format(value)


def format_summary_table(summary: dict[str, Any]) -> str:
    """Render the summary as a fixed-width console table (full vs window
    columns, then per-class distribution blocks from the full scope)."""
    full = summary["full"]
    win = summary["window"]
    w = summary["steady_state_window"]
    lines: list[str] = []
    lines.append("FleetSim summary")
    lines.append(
        f"horizon {summary['horizon_us'] / 1e6:.0f} s | steady-state window "
        f"{w['start_us'] / 1e6:.0f}-{w['end_us'] / 1e6:.0f} s "
        f"(warmup {w['warmup_frac']:.0%}, drain {w['drain_frac']:.0%})"
    )
    lines.append("")
    lines.append(f"{'metric':<30}{'full':>12}{'window':>12}")
    lines.append("-" * 54)

    def row(name: str, f: Any, wv: Any, spec: str = "{:.3f}") -> None:
        lines.append(f"{name:<30}{_fmt(f, spec):>12}{_fmt(wv, spec):>12}")

    row("occupancy", full["occupancy"], win["occupancy"])
    row("allocation rate", full["allocation_rate"], win["allocation_rate"])
    row("goodput", full["goodput"], win["goodput"])
    ettr_f = full["ettr"]["job_weighted"]
    ettr_w = win["ettr"]["job_weighted"]
    row(
        "ETTR p50 (job-weighted)",
        ettr_f["p50"] if ettr_f else None,
        ettr_w["p50"] if ettr_w else None,
    )
    row(
        "replica availability",
        full["replica_availability"],
        win["replica_availability"],
    )
    row(
        "preemptions/min (total)",
        full["preemptions_per_min"]["total"],
        win["preemptions_per_min"]["total"],
    )
    row("node failures", full["counts"]["node_failures"], win["counts"]["node_failures"])
    row("drains started", full["counts"]["drains_started"], win["counts"]["drains_started"])
    row("jobs finished", full["counts"]["jobs_finished"], win["counts"]["jobs_finished"])

    blocks = (
        ("queue wait", "queue_wait_s", "class", True),
        (
            "queue wait by source class",
            "queue_wait_s_by_source_class",
            "class",
            True,
        ),
        ("queue wait by gang size", "queue_wait_s_by_chips", "chips", False),
        ("JCT", "jct_s", "class", True),
        ("JCT by source class", "jct_s_by_source_class", "class", True),
    )
    for title, key, col0, sort_keys in blocks:
        block = full.get(key) or {}
        if not block:
            continue
        lines.append("")
        lines.append(f"{title} (s), full run — job-weighted / chip-hour-weighted:")
        lines.append(
            f"  {col0:<14}{'p50':>10}{'p90':>10}{'p99':>10}"
            f"{'p50(ch)':>10}{'p90(ch)':>10}{'n':>6}"
        )
        for cls in sorted(block) if sort_keys else block:
            jw = block[cls]["job_weighted"]
            cw = block[cls]["chip_hour_weighted"]
            lines.append(
                f"  {cls:<14}"
                f"{_fmt(jw and jw['p50'], '{:.1f}'):>10}"
                f"{_fmt(jw and jw['p90'], '{:.1f}'):>10}"
                f"{_fmt(jw and jw['p99'], '{:.1f}'):>10}"
                f"{_fmt(cw and cw['p50'], '{:.1f}'):>10}"
                f"{_fmt(cw and cw['p90'], '{:.1f}'):>10}"
                f"{_fmt(jw and jw['n']):>6}"
            )

    tenants = full["per_tenant"]
    if tenants:
        lines.append("")
        lines.append("per-tenant (full run):")
        lines.append(
            f"  {'tenant':<14}{'chip-hours':>12}{'med wait (s)':>14}{'jobs':>7}"
        )
        for tenant in sorted(tenants):
            info = tenants[tenant]
            lines.append(
                f"  {tenant:<14}"
                f"{_fmt(info['chip_hours'], '{:.2f}'):>12}"
                f"{_fmt(info['median_queue_wait_s'], '{:.1f}'):>14}"
                f"{_fmt(info['n_jobs_submitted']):>7}"
            )
    return "\n".join(lines)
