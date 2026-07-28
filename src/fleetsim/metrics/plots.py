"""Standard plots from the fleetsim parquet outputs (DESIGN §9, §13).

Four charts, rendered from ``jobs.parquet`` / ``timeseries.parquet`` in a
run's output directory into ``<out_dir>/plots/*.png``:

1. ``jct_cdf.png``          — JCT CDF per job class (COMPLETED jobs);
2. ``queue_wait_cdf.png``   — queue-wait CDF per job class (started jobs);
3. ``occupancy_timeline.png`` — allocated vs healthy chips over time,
   with occupancy-to-date and allocation-rate-to-date on a second axis;
4. ``goodput_timeline.png`` — goodput-to-date plus the interruption rate
   (scheduler/maintenance preemptions + node-failure kills) per minute.

matplotlib is imported LAZILY: importing this module never touches it,
and a missing matplotlib raises a clear ``RuntimeError`` only when a plot
function is actually called.

UNITS: time axes are hours (converted from int µs); rates per minute.
INVARIANTS: pure functions of the input frames — no wall clock, no
randomness; deterministic output for identical inputs.  Empty frames
still produce (empty) charts rather than raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "render_plots",
    "plot_jct_cdf",
    "plot_queue_wait_cdf",
    "plot_occupancy_timeline",
    "plot_goodput_timeline",
]

_US_PER_HOUR = 3_600_000_000.0


def _require_matplotlib() -> Any:
    """Import and return ``matplotlib.pyplot`` (Agg backend), or raise a
    clear RuntimeError when matplotlib is not installed."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "matplotlib is required for fleetsim plots but is not installed;"
            " run `pip install matplotlib` (it is an optional dependency)"
        ) from exc
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    _require_matplotlib().close(fig)
    return path


def _cdf_by_class(
    plt: Any, df: pd.DataFrame, value_col: str, title: str, xlabel: str
) -> Any:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = df[df[value_col].notna()]
    for cls in sorted(sub["job_class"].dropna().unique()):
        vals = sub.loc[sub["job_class"] == cls, value_col].sort_values()
        if len(vals) == 0:
            continue
        y = [(i + 1) / len(vals) for i in range(len(vals))]
        (line,) = ax.step(
            vals.to_list(), y, where="post", label=f"{cls} (n={len(vals)})"
        )
        if len(vals) < 3:
            # A 1-2 point step draws (nearly) nothing: add visible markers
            # so singleton classes (one completed pretrain) still show up.
            ax.plot(
                vals.to_list(), y, "o", color=line.get_color(), markersize=5
            )
    if len(sub) and (sub[value_col] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ax.has_data():
        ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_jct_cdf(jobs: pd.DataFrame, path: str | Path) -> Path:
    """JCT CDF per class over COMPLETED jobs -> PNG at ``path``."""
    plt = _require_matplotlib()
    completed = jobs[jobs["status"] == "COMPLETED"]
    fig = _cdf_by_class(
        plt, completed, "jct_s", "JCT CDF per class (completed jobs)", "JCT (s)"
    )
    return _save(fig, Path(path))


def plot_queue_wait_cdf(jobs: pd.DataFrame, path: str | Path) -> Path:
    """Queue-wait CDF per class over jobs that started -> PNG."""
    plt = _require_matplotlib()
    fig = _cdf_by_class(
        plt, jobs, "queue_wait_s", "Queue-wait CDF per class", "queue wait (s)"
    )
    return _save(fig, Path(path))


def plot_occupancy_timeline(ts: pd.DataFrame, path: str | Path) -> Path:
    """Allocated/healthy chips (left axis) with occupancy-to-date and
    allocation-rate-to-date (right axis, 0-1) -> PNG."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(ts):
        hours = ts["t_us"] / _US_PER_HOUR
        ax.fill_between(
            hours, ts["allocated_chips"], step="post", alpha=0.4,
            label="allocated chips",
        )
        ax.step(
            hours, ts["healthy_chips"], where="post", label="healthy chips"
        )
        ax2 = ax.twinx()
        ax2.plot(
            hours, ts["occupancy_to_date"], linestyle="--",
            label="occupancy (to date)",
        )
        ax2.plot(
            hours, ts["allocation_rate_to_date"], linestyle=":",
            label="allocation rate (to date)",
        )
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("ratio")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("chips")
    ax.set_title("Occupancy and allocation timeline")
    ax.grid(True, alpha=0.3)
    return _save(fig, Path(path))


def plot_goodput_timeline(ts: pd.DataFrame, path: str | Path) -> Path:
    """Goodput-to-date (left axis) and interruptions/min — preemptions
    plus node-failure kills, differenced per flush interval — -> PNG."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(ts):
        hours = ts["t_us"] / _US_PER_HOUR
        ax.plot(hours, ts["goodput_to_date"], label="goodput (to date)")
        ax.set_ylim(0.0, 1.05)
        if len(ts) > 1:
            cum = ts["cum_preemptions"] + ts["cum_failure_kills"]
            d_events = cum.diff()
            d_min = ts["t_us"].diff() / 60e6
            rate = (d_events / d_min).iloc[1:]
            ax2 = ax.twinx()
            ax2.step(
                hours.iloc[1:], rate, where="pre", alpha=0.7, color="tab:red",
                label="interruptions/min",
            )
            ax2.set_ylabel("interruptions / min")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        else:
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("goodput")
    ax.set_title("Goodput and interruption timeline")
    ax.grid(True, alpha=0.3)
    return _save(fig, Path(path))


def render_plots(out_dir: str | Path) -> list[Path]:
    """Render the four standard charts from ``<out_dir>/jobs.parquet``
    and ``<out_dir>/timeseries.parquet`` into ``<out_dir>/plots/``.

    Returns the written paths in a fixed order: jct_cdf.png,
    queue_wait_cdf.png, occupancy_timeline.png, goodput_timeline.png.
    """
    out = Path(out_dir)
    jobs = pd.read_parquet(out / "jobs.parquet")
    ts = pd.read_parquet(out / "timeseries.parquet")
    plots = out / "plots"
    return [
        plot_jct_cdf(jobs, plots / "jct_cdf.png"),
        plot_queue_wait_cdf(jobs, plots / "queue_wait_cdf.png"),
        plot_occupancy_timeline(ts, plots / "occupancy_timeline.png"),
        plot_goodput_timeline(ts, plots / "goodput_timeline.png"),
    ]
