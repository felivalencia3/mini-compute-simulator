"""The ``fleetsim`` command-line interface (DESIGN §13).

Subcommands::

    fleetsim run scenario.yaml [-o out/] [--seed 42] [--override k=v ...]
    fleetsim validate scenario.yaml
    fleetsim plot out/
    fleetsim compare out_a/ out_b/ [...]

``run`` executes the scenario and prints the summary table; ``validate``
checks schema + feasibility (fleet buildable, scheduler resolvable,
trace file present) and exits nonzero on any error; ``plot`` renders the
standard charts from an output directory; ``compare`` prints headline
metrics of two or more runs side by side.

Exit codes: 0 success, 1 validation/comparison failure, 2 usage or
runtime error.  All output is deterministic given the inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import api
from .config import ScenarioError, load_scenario, validate

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fleetsim",
        description="Discrete-event simulator for ML accelerator fleets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a scenario and write outputs")
    p_run.add_argument("scenario", help="scenario YAML file")
    p_run.add_argument(
        "-o",
        "--out",
        default=None,
        help="output directory (default: the scenario's outputs.dir, else 'out')",
    )
    p_run.add_argument("--seed", type=int, default=None, help="override sim.seed")
    p_run.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-path document override, repeatable"
        " (e.g. --override scheduler.name=fifo)",
    )

    p_val = sub.add_parser("validate", help="check a scenario without running it")
    p_val.add_argument("scenario", help="scenario YAML file")

    p_plot = sub.add_parser("plot", help="render plots from an output directory")
    p_plot.add_argument("out_dir", help="directory containing jobs/timeseries parquet")

    p_cmp = sub.add_parser("compare", help="compare summaries of two or more runs")
    p_cmp.add_argument("out_dirs", nargs="+", help="output directories to compare")
    return parser


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ScenarioError(
                [f"--override expects KEY=VALUE, got {pair!r}"]
            )
        overrides[key.strip()] = value
    return overrides


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from .metrics.summary import format_summary_table

    overrides = _parse_overrides(args.override)
    doc = api.load_document(args.scenario)
    api.apply_overrides(doc, overrides)
    out_raw = doc.get("outputs") if isinstance(doc.get("outputs"), dict) else {}
    out_dir = args.out or (out_raw or {}).get("dir") or "out"
    summary = api.run_scenario(
        args.scenario,
        out_dir=out_dir,
        seed_override=args.seed,
        overrides=overrides,
    )
    print(format_summary_table(summary))
    print(f"\noutputs written to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _feasibility_errors(scenario, scenario_dir: Path) -> list[str]:
    """Checks beyond the schema: fleet buildable, scheduler resolvable,
    trace source present.  Run only on schema-clean scenarios."""
    errors: list[str] = []
    try:
        from .fleet.build import build_fleet

        build_fleet(scenario)
    except ValueError as exc:
        errors.append(f"fleet: {exc}")
    try:
        from .schedulers.base import get_scheduler

        get_scheduler(scenario.scheduler.name, scenario.scheduler.params)
    except (ValueError, TypeError) as exc:
        errors.append(f"scheduler: {exc}")
    if scenario.workload.kind == "trace" and scenario.workload.source:
        src = Path(scenario.workload.source)
        if not src.is_absolute():
            src = scenario_dir / src
        if not src.is_file():
            errors.append(f"workload.source: trace file not found: {src}")
    return errors


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.scenario)
    try:
        scenario = load_scenario(path, strict=False)
    except ScenarioError as exc:
        for err in exc.errors:
            print(f"error: {err}")
        print(f"\n{args.scenario}: INVALID ({len(exc.errors)} error(s))")
        return 1
    errors = validate(scenario)
    if not errors:
        errors = _feasibility_errors(scenario, path.resolve().parent)
    if errors:
        for err in errors:
            print(f"error: {err}")
        print(f"\n{args.scenario}: INVALID ({len(errors)} error(s))")
        return 1
    print(f"{args.scenario}: OK")
    return 0


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def _cmd_plot(args: argparse.Namespace) -> int:
    from .metrics.plots import render_plots

    out = Path(args.out_dir)
    missing = [
        name
        for name in ("jobs.parquet", "timeseries.parquet")
        if not (out / name).is_file()
    ]
    if missing:
        print(
            f"error: {args.out_dir} is not a fleetsim output directory"
            f" (missing {', '.join(missing)}); pass the -o directory of a"
            f" previous `fleetsim run`",
            file=sys.stderr,
        )
        return 2
    paths = render_plots(args.out_dir)
    for p in paths:
        print(p)
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _get(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _fmt(value: Any, spec: str = "{:.3f}") -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return spec.format(value)


def _compare_rows(summaries: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """(row label, one formatted cell per run) for the headline metrics."""
    rows: list[tuple[str, list[str]]] = []

    def add(label: str, *keys: str, spec: str = "{:.3f}") -> None:
        rows.append((label, [_fmt(_get(s, *keys), spec) for s in summaries]))

    add("occupancy (window)", "window", "occupancy")
    add("allocation rate (window)", "window", "allocation_rate")
    add("goodput (window)", "window", "goodput")
    add("ETTR p50 (window)", "window", "ettr", "job_weighted", "p50")
    classes = sorted(
        {c for s in summaries for c in (_get(s, "full", "queue_wait_s") or {})}
    )
    for cls in classes:
        add(
            f"queue wait p50 {cls} (s)",
            "full",
            "queue_wait_s",
            cls,
            "job_weighted",
            "p50",
            spec="{:.1f}",
        )
    for cls in sorted(
        {c for s in summaries for c in (_get(s, "full", "jct_s") or {})}
    ):
        add(
            f"JCT p50 {cls} (s)",
            "full",
            "jct_s",
            cls,
            "job_weighted",
            "p50",
            spec="{:.1f}",
        )
    add("preemptions/min (full)", "full", "preemptions_per_min", "total")
    add("node failures (full)", "full", "counts", "node_failures")
    add("jobs finished (full)", "full", "counts", "jobs_finished")
    add("jobs started (full)", "full", "counts", "jobs_started")
    return rows


def _cmd_compare(args: argparse.Namespace) -> int:
    if len(args.out_dirs) < 2:
        print("compare needs at least two output directories", file=sys.stderr)
        return 2
    summaries: list[dict[str, Any]] = []
    names: list[str] = []
    for d in args.out_dirs:
        path = Path(d) / "summary.json"
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"error: invalid JSON in {path}: {exc}", file=sys.stderr)
            return 1
        name = Path(d).name or str(d)
        names.append(name)
    # Disambiguate colliding basenames (a/out vs b/out): fall back to the
    # argument as given for every member of a duplicated name.
    dupes = {n for n in names if names.count(n) > 1}
    names = [
        str(d) if n in dupes else n for n, d in zip(names, args.out_dirs)
    ]
    rows = _compare_rows(summaries)
    width = max(12, *(len(n) + 2 for n in names))
    label_w = max(28, *(len(label) + 2 for label, _ in rows))
    header = f"{'metric':<{label_w}}" + "".join(f"{n:>{width}}" for n in names)
    print(header)
    print("-" * len(header))
    for label, cells in rows:
        print(
            f"{label:<{label_w}}" + "".join(f"{c:>{width}}" for c in cells)
        )
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "plot":
            return _cmd_plot(args)
        if args.command == "compare":
            return _cmd_compare(args)
    except ScenarioError as exc:
        for err in exc.errors:
            print(f"error: {err}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
