"""One-call simulation entry point: scenario in, summary dict out.

:func:`run_scenario` wires every subsystem together exactly the way the
CLI does — load + validate config, build the fleet tree, construct the
job source (synthetic generator or trace replay), instantiate the
scheduler from the registry, attach a :class:`MetricsCollector`, run the
:class:`Simulator` to the horizon, and write ``jobs.parquet`` /
``timeseries.parquet`` / ``summary.json`` (plus plots when requested and
``stints.parquet`` when ``outputs.stints`` is set).

UNITS: all times inside the pipeline are int microseconds; the returned
summary reports float seconds / chip-hours (see fleetsim.metrics.summary).

DETERMINISM: a run is a pure function of ``(scenario, seed)`` — the same
input document and seed produce byte-identical Parquet/JSON outputs.
``seed_override`` and dotted-path ``overrides`` mutate a deep copy of the
raw document BEFORE parsing, so an overridden run is indistinguishable
from a run of the edited file.

INVARIANTS: the input mapping is never mutated; relative
``workload.source`` paths resolve against the scenario file's directory
(or ``base_dir`` for mapping inputs); ``out_dir=None`` falls back to the
scenario's ``outputs.dir`` and, when that is also unset, skips file
output entirely (the summary dict is still returned).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .config import Scenario, ScenarioError, load_scenario
from .engine.rng import RngStreams
from .engine.sim import QuotaAdmission, Simulator
from .fleet.build import build_fleet
from .metrics.collector import MetricsCollector
from .metrics.summary import build_summary, write_outputs
from .schedulers.base import get_scheduler
from .workload.base import JobSource
from .workload.synthetic import SyntheticSource
from .workload.trace import TraceSource

__all__ = ["run_scenario", "apply_overrides", "load_document"]


def load_document(path_or_dict: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load the raw scenario document as a plain dict.

    A mapping input is deep-copied (callers' objects are never mutated);
    a path is read and YAML-parsed.  Raises :class:`ScenarioError` on
    unreadable files, invalid YAML, or non-mapping documents.
    """
    if isinstance(path_or_dict, Mapping):
        return copy.deepcopy(dict(path_or_dict))
    path = Path(path_or_dict)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError([f"cannot read scenario file {path}: {exc}"]) from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError([f"invalid YAML in {path}: {exc}"]) from exc
    if not isinstance(doc, Mapping):
        raise ScenarioError(
            [f"scenario must be a mapping, got {type(doc).__name__}"]
        )
    return copy.deepcopy(dict(doc))


def apply_overrides(
    doc: dict[str, Any], overrides: Mapping[str, str] | None
) -> dict[str, Any]:
    """Apply dotted-path overrides to a raw scenario dict, in place.

    Keys are dotted paths into the document (``scheduler.name``,
    ``sim.seed``, ``workload.classes.eval.rate_per_hour``); values are
    parsed as YAML scalars/collections, so ``"7"`` becomes int 7 and
    ``"{}"`` an empty mapping.  Intermediate mappings are created as
    needed; a path that traverses a non-mapping raises
    :class:`ScenarioError`.  Returns ``doc`` for chaining.
    """
    for dotted, raw in (overrides or {}).items():
        parts = dotted.split(".")
        if not all(parts):
            raise ScenarioError([f"override {dotted!r}: empty path segment"])
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ScenarioError(
                [f"override {dotted!r}: invalid value {raw!r}: {exc}"]
            ) from exc
        node: dict[str, Any] = doc
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                raise ScenarioError(
                    [
                        f"override {dotted!r}: {part!r} is not a mapping"
                        f" (got {type(child).__name__})"
                    ]
                )
            node = child
        node[parts[-1]] = value
    return doc


def _make_source(
    scenario: Scenario, fleet, rng: RngStreams, base_dir: Path | None
) -> JobSource:
    """Build the job source: synthetic generator (plus any expanded
    inference services as initial jobs) or trace replay (quantized
    against the fleet's node sizes, DESIGN §4.1)."""
    workload = scenario.workload
    if workload.kind == "trace":
        src = Path(workload.source or "")
        if not src.is_absolute() and base_dir is not None:
            src = base_dir / src
        return TraceSource(src, fleet=fleet)
    initial: list = []
    if scenario.services:
        from .model import GangSpec, Service
        from .workload.services import expand_services

        services = [
            Service(
                id=sc.id,
                tenant=sc.tenant,
                replica_spec=GangSpec(
                    chips=1,  # superseded: v1 replicas are one whole node
                    chip_type=sc.chip_type,
                    within=sc.within,
                ),
                min_replicas=sc.replicas,
                max_replicas=sc.replicas,
                tier=sc.tier,
            )
            for sc in scenario.services
        ]
        initial = expand_services(services, fleet, scenario.sim.horizon_us)
    return SyntheticSource(
        workload, fleet, rng, scenario.sim.horizon_us, initial_jobs=initial
    )


def run_scenario(
    path_or_dict: str | Path | Mapping[str, Any],
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    overrides: Mapping[str, str] | None = None,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one scenario end to end and return its summary dict.

    Parameters
    ----------
    path_or_dict:
        Scenario YAML path, or an already-parsed mapping (deep-copied).
    out_dir:
        Where to write ``jobs.parquet`` / ``timeseries.parquet`` /
        ``summary.json`` (created if missing).  ``None`` falls back to the
        scenario's ``outputs.dir``; if that is also unset, nothing is
        written and only the summary dict is returned.
    seed_override:
        Replaces ``sim.seed`` (wins over ``overrides``).
    overrides:
        Dotted-path document edits applied before parsing, e.g.
        ``{"scheduler.name": "fifo", "sim.seed": "7"}`` (values are YAML).
    progress_cb:
        Optional observer forwarded to the :class:`Simulator` — invoked
        at every metrics flush with the engine's progress snapshot dict
        (see the Simulator docstring).  ``None`` (default) changes
        nothing; outputs stay byte-identical.

    Raises :class:`ScenarioError` (listing every problem) for invalid
    scenarios and ``ValueError`` for unknown scheduler names.
    """
    base_dir: Path | None = None
    if not isinstance(path_or_dict, Mapping):
        base_dir = Path(path_or_dict).resolve().parent
    doc = load_document(path_or_dict)
    apply_overrides(doc, overrides)
    if seed_override is not None:
        doc.setdefault("sim", {})
        if not isinstance(doc["sim"], dict):
            raise ScenarioError([f"sim: expected a mapping, got {doc['sim']!r}"])
        doc["sim"]["seed"] = int(seed_override)

    scenario = load_scenario(doc, strict=True)
    fleet = build_fleet(scenario)
    rng = RngStreams(scenario.sim.seed)
    source = _make_source(scenario, fleet, rng, base_dir)
    scheduler = get_scheduler(scenario.scheduler.name, scenario.scheduler.params)
    collector = MetricsCollector.from_scenario(scenario, fleet)
    admission = (
        QuotaAdmission(scenario.quota) if scenario.quota is not None else None
    )
    sim = Simulator(
        scenario,
        fleet,
        source,
        scheduler,
        collector,
        admission,
        rng=rng,
        progress_cb=progress_cb,
    )
    sim.run()

    target = out_dir if out_dir is not None else scenario.outputs.dir
    if target is None:
        return build_summary(collector)
    summary = write_outputs(collector, target)
    if scenario.outputs.plots:
        from .metrics.plots import render_plots

        render_plots(target)
    return summary
