"""The ``fleetsim`` command-line interface (DESIGN §13).

Subcommands::

    fleetsim run scenario.yaml [-o out/] [--seed 42] [--override k=v ...]
    fleetsim validate scenario.yaml
    fleetsim plot out/
    fleetsim compare out_a/ out_b/ [...]
    fleetsim viz out/ [out_b/] [-o report.html] [--title T]
                     [--map-level L] [--open]
    fleetsim serve [-p 8500] [--workspace DIR] [--host H] [--open]
    fleetsim validation cite [trace]
    fleetsim validation run

``run`` executes the scenario and prints the summary table; ``validate``
checks schema + feasibility (fleet buildable, scheduler resolvable,
trace file present) and exits nonzero on any error; ``plot`` renders the
standard charts from an output directory; ``compare`` prints headline
metrics of two or more runs side by side; ``viz`` renders one run (or an
A/B pair) into a single self-contained interactive HTML replay
(docs/visualizer.md); ``serve`` starts the local web app (v0.5): browse
workspace runs, launch scenarios with live progress, open the 2D report
and the three.js 3D fleet replay (docs/webapp.md); ``validation cite``
prints the license + citation each published trace must be attributed
under, and ``validation run`` runs the vendored-slice validation checks
(no network) — the v0.6 validation suite (docs/validation.md).

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
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="fleetsim",
        description="Discrete-event simulator for ML accelerator fleets.",
    )
    parser.add_argument(
        "--version", action="version", version=f"fleetsim {__version__}"
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

    p_viz = sub.add_parser(
        "viz",
        help="render a run into one self-contained interactive HTML replay",
    )
    p_viz.add_argument("out_dir", help="fleetsim run output directory (run A)")
    p_viz.add_argument(
        "out_dir_b",
        nargs="?",
        default=None,
        help="optional second run: dashed overlays + side-by-side cards"
        " (compare mode; must share run A's horizon)",
    )
    p_viz.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="REPORT",
        help="report file to write (default: OUT_DIR/report.html)",
    )
    p_viz.add_argument(
        "--title",
        default=None,
        help="report title (default: 'fleetsim replay — <dir name(s)>')",
    )
    p_viz.add_argument(
        "--map-level",
        default=None,
        metavar="LEVEL",
        help="fleet-map level name, e.g. pod"
        " (default: inferred from the stint domain ids)",
    )
    p_viz.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="open the written report in the default browser",
    )

    p_srv = sub.add_parser(
        "serve",
        help="start the local fleetsim web app (browse, launch, replay runs)",
    )
    p_srv.add_argument(
        "-p", "--port", type=int, default=8500, help="port (default 8500)"
    )
    p_srv.add_argument(
        "--workspace",
        default="./fleetsim-runs",
        help="run workspace directory, created if missing"
        " (default ./fleetsim-runs)",
    )
    p_srv.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; NON-LOOPBACK VALUES EXPOSE THE APP TO YOUR"
        " NETWORK and print a warning (default 127.0.0.1)",
    )
    p_srv.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="open the app in the default browser",
    )

    p_vld = sub.add_parser(
        "validation",
        help="v0.6 trace-validation suite: cite trace attributions or run"
        " the vendored-slice checks",
    )
    vsub = p_vld.add_subparsers(dest="validation_command", required=True)
    p_cite = vsub.add_parser(
        "cite",
        help="print the license + citation each published trace must be"
        " attributed under",
    )
    p_cite.add_argument(
        "trace",
        nargs="?",
        default=None,
        help="trace name (e.g. helios, philly, pai_task_table);"
        " omit to print every registered trace",
    )
    vsub.add_parser(
        "run",
        help="run the vendored-slice validation checks (Helios FIFO-vs-SJF"
        " direction + Philly status split); no network, no full trace",
    )
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


#: Ceiling on the DECLARED fleet size, checked arithmetically from the
#: topology counts BEFORE any node object is materialized.  Building
#: costs ~30 µs and ~1.4 KB per node, so an unbounded declaration (a
#: 200-byte YAML can declare billions of nodes) would pin a core for
#: minutes or OOM the process — including the `fleetsim serve` HTTP
#: handler that validates submissions.  The ceiling is 4x the frontier
#: example (65,536 nodes / 524,288 chips).
_MAX_FLEET_NODES = 262_144  # 2**18
_MAX_FLEET_CHIPS = 4_194_304  # 2**22


def _feasibility_errors(scenario, scenario_dir: Path) -> list[str]:
    """Checks beyond the schema: fleet buildable (and bounded), scheduler
    resolvable, trace source present.  Run only on schema-clean
    scenarios."""
    errors: list[str] = []
    clusters = scenario.fleet.clusters()
    total_nodes = sum(cl.total_nodes() for cl in clusters)
    total_chips = sum(cl.total_chips() for cl in clusters)
    if total_nodes > _MAX_FLEET_NODES or total_chips > _MAX_FLEET_CHIPS:
        errors.append(
            f"fleet: declared fleet is too large to build"
            f" ({total_nodes:,} nodes / {total_chips:,} chips; the"
            f" ceiling is {_MAX_FLEET_NODES:,} nodes /"
            f" {_MAX_FLEET_CHIPS:,} chips)"
        )
    else:
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
    # Queue-wait / JCT rows are FULL-RUN values (labeled): the summary's
    # windowed variants live under summary["window"].  Source-class rows
    # (workload-class label, e.g. a `frontier` class distinct from other
    # PRETRAINs) are shown when the summaries carry them.
    classes = sorted(
        {c for s in summaries for c in (_get(s, "full", "queue_wait_s") or {})}
    )
    for cls in classes:
        add(
            f"queue wait p50 {cls} (s, full)",
            "full",
            "queue_wait_s",
            cls,
            "job_weighted",
            "p50",
            spec="{:.1f}",
        )
    for cls in sorted(
        {
            c
            for s in summaries
            for c in (_get(s, "full", "queue_wait_s_by_source_class") or {})
        }
    ):
        add(
            f"queue wait p50 [{cls}] (s, full)",
            "full",
            "queue_wait_s_by_source_class",
            cls,
            "job_weighted",
            "p50",
            spec="{:.1f}",
        )
    for cls in sorted(
        {c for s in summaries for c in (_get(s, "full", "jct_s") or {})}
    ):
        add(
            f"JCT p50 {cls} (s, full)",
            "full",
            "jct_s",
            cls,
            "job_weighted",
            "p50",
            spec="{:.1f}",
        )
    for cls in sorted(
        {
            c
            for s in summaries
            for c in (_get(s, "full", "jct_s_by_source_class") or {})
        }
    ):
        add(
            f"JCT p50 [{cls}] (s, full)",
            "full",
            "jct_s_by_source_class",
            cls,
            "job_weighted",
            "p50",
            spec="{:.1f}",
        )
    add("preemptions/min (full)", "full", "preemptions_per_min", "total")
    add("node failures (full)", "full", "counts", "node_failures")
    add("jobs finished (full)", "full", "counts", "jobs_finished")
    add("jobs started (full)", "full", "counts", "jobs_started")
    # v0.7 placement rows: shown only when at least one run NAMED a
    # placement policy (a run that named none has neither key, and a
    # pre-v0.7 output directory therefore compares exactly as before).
    # These are what actually separate placers — occupancy and goodput can
    # be bit-identical across them (docs/placement.md), so a placement A/B
    # read off the rows above alone would look like a null result.
    if any(_get(s, "full", "counts", "placement_policy") for s in summaries):
        add(
            "stranded whole nodes (full mean)",
            "full",
            "fragmentation",
            "stranded_whole_nodes",
            "mean",
            spec="{:.2f}",
        )
        rows.append(
            (
                "placement policy",
                [
                    str(_get(s, "full", "counts", "placement_policy") or "-")
                    for s in summaries
                ],
            )
        )
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
# viz
# ---------------------------------------------------------------------------


def _viz_horizon_mismatch(dir_a: str, dir_b: str) -> str | None:
    """A helpful message when the two runs cannot be overlaid honestly.

    Compare mode draws run B's frames on run A's time axis, so the runs
    must cover the same horizon.  Unreadable summaries return ``None``
    here — ``build_viz_model`` raises its own (more specific) error."""
    horizons: list[Any] = []
    for d in (dir_a, dir_b):
        try:
            summary = json.loads(
                (Path(d) / "summary.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        horizons.append(summary.get("horizon_us"))
    ha, hb = horizons
    if isinstance(ha, int) and isinstance(hb, int) and ha != hb:
        return (
            f"compare runs have different horizons ({dir_a}: {ha / 1e6:.0f} s"
            f" vs {dir_b}: {hb / 1e6:.0f} s), so their timelines cannot be"
            f" overlaid on one axis; compare two runs of the same scenario"
            f" horizon (e.g. the same scenario at two seeds or schedulers)"
        )
    return None


def _cmd_viz(args: argparse.Namespace) -> int:
    from .viz import build_viz_model, render_html

    for d in (args.out_dir, args.out_dir_b):
        if d is not None and not Path(d).is_dir():
            print(
                f"error: {d} is not a directory; pass the -o directory of a"
                f" previous `fleetsim run`",
                file=sys.stderr,
            )
            return 2
    if args.out_dir_b is not None:
        mismatch = _viz_horizon_mismatch(args.out_dir, args.out_dir_b)
        if mismatch is not None:
            print(f"error: {mismatch}", file=sys.stderr)
            return 1

    model = build_viz_model(
        args.out_dir,
        compare_dir=args.out_dir_b,
        map_level_hint=args.map_level,
    )
    if args.title:
        model["meta"]["title"] = args.title
    # The model's reconstruction notes are the degraded-mode contract:
    # anything the report could not take verbatim from the run outputs
    # (no stints.parquet -> no fleet map, no scenario copy -> inferred
    # round, ...) is disclosed here AND inside the report header.
    for note in model["meta"]["notes"]:
        print(f"note: {note}")

    report = Path(args.out) if args.out else Path(args.out_dir) / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_html(model), encoding="utf-8")
    print(f"report written to {report} ({report.stat().st_size / 1e6:.1f} MB)")
    if args.open_browser:
        import webbrowser

        webbrowser.open(str(report.resolve()))
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from .serve.server import serve

    return serve(
        port=args.port,
        workspace=args.workspace,
        host=args.host,
        open_browser=args.open_browser,
    )


# ---------------------------------------------------------------------------
# validation (v0.6 suite: cite trace attributions / run vendored-slice checks)
# ---------------------------------------------------------------------------


def _format_citation(spec) -> str:
    """The attribution block for one trace: citation, license, source,
    and the extraction hint — everything the trace's license requires when
    a paper's numbers are reproduced from it."""
    lines = [
        spec.name,
        f"  citation: {spec.citation}",
        f"  license:  {spec.license}",
        f"  source:   {spec.attribution_url}",
        f"  extract:  {spec.extract_hint}",
    ]
    return "\n".join(lines)


def _cmd_validation_cite(args: argparse.Namespace) -> int:
    from .validation.registry import TRACE_REGISTRY, get_spec

    if args.trace is not None:
        try:
            spec = get_spec(args.trace)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(_format_citation(spec))
        return 0
    # No trace named: print every registered trace, in registry order.
    print(_format_citation(next(iter(TRACE_REGISTRY.values()))), end="")
    for spec in list(TRACE_REGISTRY.values())[1:]:
        print("\n")
        print(_format_citation(spec), end="")
    print()
    return 0


def _cmd_validation_run(_args: argparse.Namespace) -> int:
    """Run the vendored-slice validation checks (no network, no full
    trace): the Helios FIFO-vs-SJF direction on the 2-VC Venus slice and
    the Philly status split on the ~2k-row slice — the same checks CI runs.
    Prints a PASS/FAIL line per check and exits nonzero if any fail."""
    import pandas as pd

    from .validation.harness import replay_canonical
    from .validation.philly_status import (
        status_split_by_count,
        status_split_by_gpu_time,
    )

    root = Path(__file__).resolve().parents[2]
    traces = root / "tests" / "validation_traces"
    helios_slice = traces / "helios_venus_2vc_sept.csv"
    philly_slice = traces / "philly_slice.csv"
    for p in (helios_slice, philly_slice):
        if not p.is_file():
            print(f"error: vendored slice not found: {p}", file=sys.stderr)
            return 2

    results: list[tuple[str, bool, str]] = []

    # Helios: strict FIFO mean JCT is clearly worse than strict SJF.
    # The validation model's placer is passed EXPLICITLY (the harness default
    # is the engine default, first_fit), exactly as the pytest rungs do.
    df = pd.read_csv(helios_slice, comment="#")
    pools = {"vcvGl": 20, "vcvlY": 2}
    fifo = replay_canonical(
        df, pools, "fifo", cluster="Venus-slice", placement="consolidate"
    )
    sjf = replay_canonical(
        df, pools, "sjf", cluster="Venus-slice", placement="consolidate"
    )
    ratio = fifo["avg_jct"] / sjf["avg_jct"]
    ok = ratio > 1.15 and fifo["n_terminal"] == fifo["n_jobs"] == len(df)
    results.append(
        (
            "Helios (SC '21) FIFO-vs-SJF direction [vendored 2-VC Venus slice]",
            ok,
            f"FIFO/SJF mean-JCT ratio = {ratio:.2f}x (> 1.15), "
            f"{fifo['n_terminal']}/{len(df)} jobs terminal, "
            f"placement={fifo['placement']}",
        )
    )

    # Helios placement model (v0.7) — WIRING ONLY.  The gate is specifically
    # first_fit != consolidate: `spread` alone would satisfy a weaker
    # "not all equal" check, so degrading `consolidate` (the policy carrying
    # the whole v0.7 result) back to first-fit would pass it.  This 2-VC
    # slice does NOT exhibit the Saturn stranding pathology at any meaningful
    # magnitude (the spread is ~1 s on ~28,500 s), so no direction or
    # magnitude is claimed here; that lives in the opt-in full-trace rung
    # (docs/validation.md §4.2, ~26 % on Saturn) and in
    # tests/test_placement.py's unit-scale mechanism test.
    spread = replay_canonical(
        df, pools, "fifo", cluster="Venus-slice", placement="spread"
    )
    ff = replay_canonical(
        df, pools, "fifo", cluster="Venus-slice", placement="first_fit"
    )
    jcts = (fifo["avg_jct"], ff["avg_jct"], spread["avg_jct"])
    ok = (
        fifo["placement"] == "consolidate"
        and jcts[0] != jcts[1]  # consolidate != first_fit
        and jcts[1] != jcts[2]  # first_fit != spread
    )
    results.append(
        (
            "Helios placement selection reaches the engine [wiring, 2-VC slice]",
            ok,
            f"FIFO mean JCT differs by placer: consolidate={jcts[0]:.1f}s, "
            f"first_fit={jcts[1]:.1f}s, spread={jcts[2]:.1f}s "
            f"(slice is too small to show the effect's magnitude)",
        )
    )

    # Philly: status shares sum to 1 and hold the paper's ordering.
    pdf = pd.read_csv(philly_slice, comment="#")
    bc = status_split_by_count(pdf)
    bg = status_split_by_gpu_time(pdf)
    ku_count = bc.get("Killed", 0.0) + bc.get("Unsuccessful", 0.0)
    ku_gpu = bg.get("Killed", 0.0) + bg.get("Unsuccessful", 0.0)
    ok = (
        abs(sum(bc.values()) - 1.0) < 1e-9
        and abs(sum(bg.values()) - 1.0) < 1e-9
        and bc["Passed"] > bc["Unsuccessful"] > bc["Killed"]
        and ku_gpu > ku_count
    )
    results.append(
        (
            "Philly (ATC '19) status split [SYNTHETIC slice - structure only]",
            ok,
            f"by-count Pass {bc['Passed']:.1%} > Unsucc {bc['Unsuccessful']:.1%}"
            f" > Killed {bc['Killed']:.1%}; Killed+Unsucc GPU-time"
            f" {ku_gpu:.1%} > headcount {ku_count:.1%}",
        )
    )

    width = max(len(name) for name, _, _ in results)
    all_ok = True
    for name, ok, detail in results:
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<{width}}  {detail}")
    print()
    print(
        "vendored-slice checks: "
        + ("all passed" if all_ok else "FAILURES above")
        + " (full-trace replays are opt-in: FLEETSIM_HELIOS_FULL /"
        " FLEETSIM_PHILLY_FULL; see docs/validation.md)"
    )
    return 0 if all_ok else 1


def _cmd_validation(args: argparse.Namespace) -> int:
    if args.validation_command == "cite":
        return _cmd_validation_cite(args)
    if args.validation_command == "run":
        return _cmd_validation_run(args)
    raise AssertionError(  # pragma: no cover - argparse requires the subcommand
        f"unhandled validation command {args.validation_command!r}"
    )


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
        if args.command == "viz":
            return _cmd_viz(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "validation":
            return _cmd_validation(args)
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
