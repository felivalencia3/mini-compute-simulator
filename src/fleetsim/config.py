"""Scenario configuration: loading, template expansion, and validation.

Loads a scenario from YAML (path) or a plain dict into a typed
:class:`Scenario` tree.  Two fleet-spec forms are accepted and expanded to
the same internal representation:

- **template form** (DESIGN 3.3): top-level ``chip_types`` + ``templates``
  with ``children: {template, count}``, and ``fleet:`` as a list of metros
  with datacenters and clusters, each cluster declaring a ``levels`` list.
- **compact form** (DESIGN 13): ``fleet: {metro: ..., clusters: [...]}``
  where each cluster gives ``chip: {type, per_node}`` and
  ``topology: {levels, counts}`` (one count per level, outermost first).

Internal representation: a cluster is a declared level list plus a tree of
:class:`NodeGroup` **counts** (declarative — nodes are instantiated by the
fleet layer, not here); leaves are nodes carrying ``chips`` of one
``chip_type``.  The compact form's level list is normalized by prepending
``"cluster"`` when absent.

Distribution expressions (``pow2[1, 8]``, ``lognormal[median=2m, p90=30m]``,
``exponential[mean=30s]``, ``uniform[a, b]``, ``fixed[x]``) parse into
declarative :class:`DistSpec` records; sampling is the workload phase's job.
NOTE: inside YAML *flow* mappings ``{...}`` the brackets must be quoted
(``chips: "pow2[1, 8]"``); block-context plain scalars need no quotes.

UNITS: every ``*_us`` field is int microseconds; every ``*_s`` field is
float seconds.  DistSpec params written as durations (``"2m"``, ``"30s"``)
are stored as **int microseconds**; bare numbers stay numbers (so
``pow2[1, 8]`` keeps chip counts as ints).

INVARIANTS: loading is a pure function of its input — no wall clock, no
randomness, deterministic iteration (YAML mapping insertion order is
preserved).  ``load_scenario(strict=True)`` (the default) raises
:class:`ScenarioError` when ``validate`` finds any error; with
``strict=False`` the scenario is returned and ``validate(scenario)``
reports the full error list.  Per DESIGN principle 5, schema-accepted but
v0.1-unimplemented features are rejected with messages containing
"not implemented in v0.1" — never silently ignored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .model import CapacityClass, ChipType, Constraint, JobClass, Tier
from .units import HOUR, S, parse_duration

__all__ = [
    "DistSpec",
    "NodeGroup",
    "ClusterConfig",
    "DatacenterConfig",
    "MetroConfig",
    "FleetConfig",
    "FailureModelConfig",
    "WorkloadClassConfig",
    "WorkloadConfig",
    "SchedulerConfig",
    "ServiceConfig",
    "OutputsConfig",
    "SimConfig",
    "Scenario",
    "ScenarioError",
    "parse_dist",
    "load_scenario",
    "validate",
]

_NOT_V01 = "not implemented in v0.1"


class ScenarioError(Exception):
    """A scenario failed to load or validate.  ``errors`` lists every
    problem found (deterministic order)."""

    def __init__(self, errors: list[str] | str):
        self.errors: list[str] = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


# ---------------------------------------------------------------------------
# Distribution expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistSpec:
    """Declarative distribution: ``kind`` plus numeric params.

    Duration-valued params are stored as int microseconds; other numbers
    keep their parsed int/float type.  Sampling semantics live in the
    workload phase; this module only parses and validates.
    """

    kind: str
    params: dict[str, int | float] = field(default_factory=dict)


#: Positional parameter names per known distribution kind.
_DIST_POSITIONAL: dict[str, tuple[str, ...]] = {
    "pow2": ("lo", "hi"),
    "uniform": ("lo", "hi"),
    "fixed": ("value",),
    "exponential": ("mean",),
    "lognormal": ("median", "p90"),
}

KNOWN_DIST_KINDS = frozenset(_DIST_POSITIONAL)

#: Sentinel used when an expression failed to parse (the parse error was
#: already recorded in load_errors); validate() skips it.
_INVALID_DIST = DistSpec("invalid", {})

_DIST_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]\s*$", re.DOTALL)


def _parse_scalar(text: str) -> int | float:
    """Parse a dist-expression value: int, then float, then duration
    (returned as int microseconds).  Raises ValueError."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return parse_duration(text)  # raises ValueError with its own message


def parse_dist(expr: int | float | str) -> DistSpec:
    """Parse a distribution expression into a :class:`DistSpec`.

    Grammar: ``name[a, b]`` or ``name[k=v, k=v]``.  A bare number (or bare
    numeric/duration string) becomes ``fixed``.  Positional args map to
    kind-specific names (``pow2[1, 8]`` -> ``{lo: 1, hi: 8}``); for unknown
    kinds they are stored as ``p0, p1, ...`` and flagged by ``validate``.
    Raises ``ValueError`` on malformed syntax.
    """
    if isinstance(expr, bool):
        raise ValueError(f"cannot parse distribution expression: {expr!r}")
    if isinstance(expr, (int, float)):
        return DistSpec("fixed", {"value": expr})
    if not isinstance(expr, str):
        raise ValueError(f"cannot parse distribution expression: {expr!r}")
    m = _DIST_RE.match(expr)
    if m is None:
        try:
            return DistSpec("fixed", {"value": _parse_scalar(expr)})
        except (ValueError, TypeError):
            raise ValueError(f"cannot parse distribution expression: {expr!r}") from None
    kind, body = m.group(1), m.group(2).strip()
    positional = _DIST_POSITIONAL.get(kind, ())
    params: dict[str, int | float] = {}
    if body:
        for i, raw in enumerate(body.split(",")):
            raw = raw.strip()
            if not raw:
                raise ValueError(f"empty parameter in distribution expression: {expr!r}")
            if "=" in raw:
                key, _, val = raw.partition("=")
                key, val = key.strip(), val.strip()
                if not key or not val:
                    raise ValueError(f"malformed parameter {raw!r} in {expr!r}")
            else:
                key = positional[i] if i < len(positional) else f"p{i}"
                val = raw
            if key in params:
                raise ValueError(f"duplicate parameter {key!r} in {expr!r}")
            try:
                params[key] = _parse_scalar(val)
            except (ValueError, TypeError):
                raise ValueError(
                    f"cannot parse value {val!r} for parameter {key!r} in {expr!r}"
                ) from None
    return DistSpec(kind, params)


# ---------------------------------------------------------------------------
# Config dataclass tree
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Simulation horizon and scheduler round, both int microseconds."""

    horizon_us: int
    round_us: int = 60 * S
    seed: int = 0


@dataclass
class NodeGroup:
    """A declarative subtree of the fleet: ``count`` copies of this group
    under its parent.  Leaves (``children == []``) are nodes carrying
    ``chips`` chips of ``chip_type``.  Purely counts — the fleet layer
    instantiates actual :class:`~fleetsim.model.Domain` objects."""

    level: str
    count: int
    chips: int = 0
    chip_type: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list["NodeGroup"] = field(default_factory=list)

    def total_nodes(self) -> int:
        """Number of leaf nodes this subtree expands to."""
        if not self.children:
            return self.count
        return self.count * sum(c.total_nodes() for c in self.children)

    def total_chips(self) -> int:
        """Number of chips this subtree expands to."""
        if not self.children:
            return self.count * self.chips
        return self.count * sum(c.total_chips() for c in self.children)


@dataclass
class FailureModelConfig:
    """Failure/repair/maintenance parameters (defaults per DESIGN 14).

    ``repair_auto_min`` is a uniform range in **minutes**;
    ``repair_manual_days`` a uniform range in **days**;
    ``drain_grace_us`` int microseconds (default 1 h).

    LEMON NODES (DESIGN §8): ``lemon_frac`` of leaves (chosen
    deterministically by a stable hash of the leaf id at fleet build time
    — seed-independent) get ``lemon_factor = lemon_multiplier``, an MTBF
    multiplier consumed by the failure sampler.  Defaults: no lemons."""

    node_mtbf_days: float = 42.0
    repair_auto_min: tuple[float, float] = (60.0, 180.0)
    repair_manual_frac: float = 0.1
    repair_manual_days: tuple[float, float] = (1.0, 3.0)
    maintenance_rate_per_node_month: float = 1.0
    drain_grace_us: int = HOUR
    lemon_frac: float = 0.0
    lemon_multiplier: float = 1.0


@dataclass
class ClusterConfig:
    """One cluster: its level vocabulary (outermost first; ``levels[0]`` is
    the cluster's own level) plus the declarative count tree."""

    id: str
    levels: list[str]
    children: list[NodeGroup]
    attrs: dict[str, Any] = field(default_factory=dict)
    failure_model: FailureModelConfig = field(default_factory=FailureModelConfig)

    def total_nodes(self) -> int:
        return sum(c.total_nodes() for c in self.children)

    def total_chips(self) -> int:
        return sum(c.total_chips() for c in self.children)


@dataclass
class DatacenterConfig:
    id: str
    clusters: list[ClusterConfig] = field(default_factory=list)


@dataclass
class MetroConfig:
    name: str
    datacenters: list[DatacenterConfig] = field(default_factory=list)


@dataclass
class FleetConfig:
    """Chip registry plus the metro/datacenter/cluster hierarchy.
    Iteration order is YAML document order (deterministic)."""

    chip_types: dict[str, ChipType] = field(default_factory=dict)
    metros: list[MetroConfig] = field(default_factory=list)

    def clusters(self) -> list[ClusterConfig]:
        """All clusters in deterministic (document) order."""
        return [
            cl
            for metro in self.metros
            for dc in metro.datacenters
            for cl in dc.clusters
        ]


@dataclass
class WorkloadClassConfig:
    """One synthetic workload class.  ``rate_per_hour`` is the normalized
    arrival rate (from rate_per_hour|rate_per_day|rate_per_week).
    ``checkpoint_interval_s``/``min_runtime_s``/``max_lifetime_s`` are
    float seconds, matching the Job fields they populate.  Per-class
    defaults (DESIGN §5.1/§14) when the YAML omits the key:
    ``min_runtime`` 2 h for pretrain / 0 otherwise; ``abort_prob`` 0.3
    (Philly 30.7% / Acme ~40% killed-or-failed) except 0 for
    infer_replica — write ``abort_prob: 0`` to opt out.  ``max_lifetime``
    defaults to None (no cap; Meta's 7-day policy is one line of YAML).
    ``chip_type`` pins the class's gangs to one chip type — REQUIRED when
    the fleet has more than one leaf chip type (DESIGN §11: chips pinned
    per job).  ``capacity``, ``n_gangs``, ``shape`` and ``twisted`` are
    schema-carried; validate() rejects non-v0.1 values."""

    name: str
    job_class: JobClass | None
    rate_per_hour: float
    chips: DistSpec
    duration: DistSpec
    tier: Tier
    diurnal: bool = False
    checkpoint_interval_s: float = 3600.0
    min_runtime_s: float = 0.0
    max_lifetime_s: float | None = None
    within: Constraint | None = None
    abort_prob: float = 0.0
    n_tenants: int = 8
    chip_type: str | None = None
    capacity: CapacityClass = CapacityClass.ON_DEMAND
    n_gangs: int = 1
    shape: tuple[int, int, int] | None = None
    twisted: bool = False


@dataclass
class WorkloadConfig:
    """``kind`` is "synthetic" (uses ``classes``) or "trace" (uses
    ``source``).  ``n_tenants`` is the default a class inherits when it
    does not set its own."""

    kind: str = "synthetic"
    classes: list[WorkloadClassConfig] = field(default_factory=list)
    source: str | None = None
    n_tenants: int = 8


@dataclass
class SchedulerConfig:
    """Scheduler selection.  ``name`` is looked up in the scheduler
    registry at run time — unknown names are NOT a validation error here."""

    name: str = "fifo"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceConfig:
    """One inference service (DESIGN §5; v1 freezes replica counts).

    Expanded by the run pipeline into ``replicas`` frozen one-node
    INFER_REPLICA jobs at t=0 (:func:`fleetsim.workload.services.
    expand_services`).  ``within`` is the optional hard containment level;
    ``tier`` defaults to PROD (Borg/MAST: serving is prod-tier jobs).
    v0.1 supports services only with synthetic workloads."""

    id: str
    tenant: str
    replicas: int
    chip_type: str | None = None
    within: Constraint | None = None
    tier: Tier = Tier.PROD


@dataclass
class OutputsConfig:
    dir: str | None = None
    events: str = "parquet"
    plots: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """The fully-parsed scenario.  ``load_errors`` holds problems found
    while parsing (bad expressions, unknown keys, ambiguous rates);
    ``validate`` prepends them to its own findings."""

    sim: SimConfig
    fleet: FleetConfig
    workload: WorkloadConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    failure_model: FailureModelConfig = field(default_factory=FailureModelConfig)
    services: list[ServiceConfig] = field(default_factory=list)
    reservations: Any = None
    load_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _duration_us(value: Any, ctx: str, errors: list[str], default: int = 0) -> int:
    try:
        return parse_duration(value)
    except (ValueError, TypeError) as exc:
        errors.append(f"{ctx}: {exc}")
        return default


def _range2(value: Any, ctx: str, errors: list[str]) -> tuple[float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value), float(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            pass
    errors.append(f"{ctx}: expected a number or [lo, hi] pair, got {value!r}")
    return (0.0, 0.0)


#: Accepted failure-model keys (DESIGN 3.3 spelling + DESIGN 13 compact
#: aliases + lemon nodes).  Anything else is an error — never a silent
#: no-op (DESIGN principle 5).
_FM_KEYS = frozenset(
    {
        "node_mtbf_days",
        "mtbf_node_days",
        "repair",
        "repair_auto_min",
        "repair_manual_frac",
        "repair_manual_days",
        "maintenance_rate_per_node_month",
        "drain_grace",
        "lemon_frac",
        "lemon_multiplier",
    }
)
_FM_REPAIR_KEYS = frozenset({"auto_min", "manual_frac", "manual_days"})


def _parse_failure_model(
    m: Mapping[str, Any] | None,
    base: FailureModelConfig,
    ctx: str,
    errors: list[str],
) -> FailureModelConfig:
    """Parse a failure-model mapping, inheriting unset fields from ``base``.
    Accepts DESIGN 3.3 spelling (node_mtbf_days, repair: {auto_min,
    manual_frac, manual_days}) and DESIGN 13 compact aliases
    (mtbf_node_days, repair_auto_min); unknown keys are errors."""
    if not m:
        return FailureModelConfig(
            node_mtbf_days=base.node_mtbf_days,
            repair_auto_min=base.repair_auto_min,
            repair_manual_frac=base.repair_manual_frac,
            repair_manual_days=base.repair_manual_days,
            maintenance_rate_per_node_month=base.maintenance_rate_per_node_month,
            drain_grace_us=base.drain_grace_us,
            lemon_frac=base.lemon_frac,
            lemon_multiplier=base.lemon_multiplier,
        )
    unknown = sorted(set(m) - _FM_KEYS)
    if unknown:
        errors.append(
            f"{ctx}: unknown key(s): {', '.join(unknown)}"
            f" (known: {', '.join(sorted(_FM_KEYS))})"
        )
    repair = m.get("repair") or {}
    if isinstance(repair, Mapping):
        unknown_r = sorted(set(repair) - _FM_REPAIR_KEYS)
        if unknown_r:
            errors.append(
                f"{ctx}.repair: unknown key(s): {', '.join(unknown_r)}"
                f" (known: {', '.join(sorted(_FM_REPAIR_KEYS))})"
            )
    else:
        errors.append(f"{ctx}.repair: expected a mapping, got {repair!r}")
        repair = {}
    mtbf = m.get("node_mtbf_days", m.get("mtbf_node_days", base.node_mtbf_days))
    auto_min = m.get("repair_auto_min", repair.get("auto_min", base.repair_auto_min))
    manual_frac = m.get(
        "repair_manual_frac", repair.get("manual_frac", base.repair_manual_frac)
    )
    manual_days = m.get(
        "repair_manual_days", repair.get("manual_days", base.repair_manual_days)
    )
    maint = m.get(
        "maintenance_rate_per_node_month", base.maintenance_rate_per_node_month
    )
    drain = m.get("drain_grace")
    drain_us = (
        _duration_us(drain, f"{ctx}.drain_grace", errors, base.drain_grace_us)
        if drain is not None
        else base.drain_grace_us
    )
    return FailureModelConfig(
        node_mtbf_days=float(mtbf),
        repair_auto_min=_range2(auto_min, f"{ctx}.repair.auto_min", errors),
        repair_manual_frac=float(manual_frac),
        repair_manual_days=_range2(manual_days, f"{ctx}.repair.manual_days", errors),
        maintenance_rate_per_node_month=float(maint),
        drain_grace_us=drain_us,
        lemon_frac=float(m.get("lemon_frac", base.lemon_frac)),
        lemon_multiplier=float(m.get("lemon_multiplier", base.lemon_multiplier)),
    )


def _expand_template(
    name: str,
    templates: Mapping[str, Any],
    stack: tuple[str, ...],
    errors: list[str],
) -> NodeGroup | None:
    """Expand template ``name`` into a NodeGroup with count=1 (the caller
    sets the count).  Returns None (with an error recorded) for unknown
    names or cycles."""
    if name in stack:
        errors.append(
            "templates: cycle detected: " + " -> ".join(stack + (name,))
        )
        return None
    t = templates.get(name)
    if not isinstance(t, Mapping):
        errors.append(f"templates: unknown template {name!r}")
        return None
    level = t.get("level")
    if not isinstance(level, str):
        errors.append(f"templates.{name}: missing required key 'level'")
        return None
    group = NodeGroup(
        level=level,
        count=1,
        chips=int(t.get("chips", 0) or 0),
        chip_type=t.get("chip_type"),
        attrs=dict(t.get("attrs") or {}),
    )
    children = t.get("children")
    if children:
        refs = children if isinstance(children, list) else [children]
        for ref in refs:
            if not isinstance(ref, Mapping) or "template" not in ref:
                errors.append(
                    f"templates.{name}: children entries must be "
                    f"{{template, count}} mappings, got {ref!r}"
                )
                continue
            child = _expand_template(
                str(ref["template"]), templates, stack + (name,), errors
            )
            if child is not None:
                child.count = int(ref.get("count", 1))
                group.children.append(child)
    return group


def _parse_cluster_template_form(
    c: Mapping[str, Any],
    templates: Mapping[str, Any],
    base_fm: FailureModelConfig,
    errors: list[str],
) -> ClusterConfig | None:
    cid = c.get("id") or c.get("name")
    if not cid:
        errors.append("fleet: cluster missing 'id' (or 'name')")
        return None
    ctx = f"fleet.clusters.{cid}"
    levels = c.get("levels")
    if not isinstance(levels, list) or not levels:
        errors.append(f"{ctx}: template-form cluster requires a non-empty 'levels' list")
        levels = []
    children: list[NodeGroup] = []
    refs = c.get("children") or []
    if isinstance(refs, Mapping):
        refs = [refs]
    for ref in refs:
        if not isinstance(ref, Mapping) or "template" not in ref:
            errors.append(
                f"{ctx}: children entries must be {{template, count}} mappings,"
                f" got {ref!r}"
            )
            continue
        group = _expand_template(str(ref["template"]), templates, (), errors)
        if group is not None:
            group.count = int(ref.get("count", 1))
            children.append(group)
    fm = _parse_failure_model(
        c.get("failures") or c.get("failure_model"), base_fm, ctx, errors
    )
    return ClusterConfig(
        id=str(cid),
        levels=[str(x) for x in levels],
        children=children,
        attrs=dict(c.get("attrs") or {}),
        failure_model=fm,
    )


def _parse_cluster_compact_form(
    c: Mapping[str, Any],
    base_fm: FailureModelConfig,
    errors: list[str],
) -> ClusterConfig | None:
    cid = c.get("name") or c.get("id")
    if not cid:
        errors.append("fleet: cluster missing 'name' (or 'id')")
        return None
    ctx = f"fleet.clusters.{cid}"
    chip = c.get("chip") or {}
    topo = c.get("topology") or {}
    levels = topo.get("levels")
    counts = topo.get("counts")
    if (
        not isinstance(levels, list)
        or not isinstance(counts, list)
        or not levels
        or len(levels) != len(counts)
    ):
        errors.append(
            f"{ctx}: topology requires same-length non-empty 'levels' and"
            f" 'counts' lists"
        )
        return None
    chip_type = chip.get("type")
    per_node = int(chip.get("per_node", 0) or 0)
    if not chip_type:
        errors.append(f"{ctx}: compact-form cluster requires chip: {{type, per_node}}")
    levels = [str(x) for x in levels]
    counts = [int(x) for x in counts]
    group = NodeGroup(
        level=levels[-1], count=counts[-1], chips=per_node, chip_type=chip_type
    )
    for lvl, cnt in zip(reversed(levels[:-1]), reversed(counts[:-1])):
        group = NodeGroup(level=lvl, count=cnt, children=[group])
    cluster_levels = levels if levels[0] == "cluster" else ["cluster", *levels]
    fm = _parse_failure_model(
        c.get("failures") or c.get("failure_model"), base_fm, ctx, errors
    )
    return ClusterConfig(
        id=str(cid),
        levels=cluster_levels,
        children=[group],
        attrs=dict(c.get("attrs") or {}),
        failure_model=fm,
    )


def _parse_cluster(
    c: Any,
    templates: Mapping[str, Any],
    base_fm: FailureModelConfig,
    errors: list[str],
) -> ClusterConfig | None:
    if not isinstance(c, Mapping):
        errors.append(f"fleet: cluster entries must be mappings, got {c!r}")
        return None
    if "topology" in c or "chip" in c:
        return _parse_cluster_compact_form(c, base_fm, errors)
    return _parse_cluster_template_form(c, templates, base_fm, errors)


def _parse_fleet(
    doc: Mapping[str, Any],
    base_fm: FailureModelConfig,
    errors: list[str],
) -> FleetConfig:
    chip_types_decl = doc.get("chip_types")
    declared = isinstance(chip_types_decl, Mapping)
    registry: dict[str, ChipType] = {}
    if declared:
        for cname, spec in chip_types_decl.items():
            spec = spec or {}
            registry[str(cname)] = ChipType(
                name=str(cname),
                vendor=str(spec.get("vendor", "unknown")),
                hbm_gib=float(spec.get("hbm_gib", 0.0)),
                peak_tflops_bf16=float(spec.get("peak_tflops_bf16", 0.0)),
                generation=int(spec.get("generation", 1)),
            )
    templates = doc.get("templates") or {}
    fleet_node = doc.get("fleet")
    metros: list[MetroConfig] = []

    def parse_metro(m: Mapping[str, Any]) -> None:
        mname = str(m.get("metro", m.get("name", "default")))
        dcs: list[DatacenterConfig] = []
        for i, d in enumerate(m.get("datacenters") or []):
            if not isinstance(d, Mapping):
                errors.append(f"fleet.{mname}: datacenter entries must be mappings")
                continue
            dc = DatacenterConfig(id=str(d.get("id", f"dc{i}")))
            for c in d.get("clusters") or []:
                cl = _parse_cluster(c, templates, base_fm, errors)
                if cl is not None:
                    dc.clusters.append(cl)
            dcs.append(dc)
        # compact form: clusters directly under the metro, implicit dc0
        if "clusters" in m:
            dc = DatacenterConfig(id="dc0")
            for c in m.get("clusters") or []:
                cl = _parse_cluster(c, templates, base_fm, errors)
                if cl is not None:
                    dc.clusters.append(cl)
            dcs.append(dc)
        metros.append(MetroConfig(name=mname, datacenters=dcs))

    if fleet_node is None:
        errors.append("fleet: section is required")
    elif isinstance(fleet_node, list):
        for m in fleet_node:
            if isinstance(m, Mapping):
                parse_metro(m)
            else:
                errors.append(f"fleet: metro entries must be mappings, got {m!r}")
    elif isinstance(fleet_node, Mapping):
        parse_metro(fleet_node)
    else:
        errors.append(f"fleet: expected a mapping or list, got {fleet_node!r}")

    fleet = FleetConfig(chip_types=registry, metros=metros)
    # Auto-register referenced chip types only when the scenario declared no
    # registry at all (compact form); with a declared registry, a missing
    # reference is a validation error (typo protection).
    if not declared:
        for ct in _referenced_chip_types(fleet):
            if ct not in registry:
                registry[ct] = ChipType(
                    name=ct, vendor="unknown", hbm_gib=0.0, peak_tflops_bf16=0.0
                )
    return fleet


def _walk_groups(cluster: ClusterConfig):
    """Yield (group, parent) pairs in deterministic pre-order."""
    stack = [(g, None) for g in reversed(cluster.children)]
    while stack:
        group, parent = stack.pop()
        yield group, parent
        for child in reversed(group.children):
            stack.append((child, group))


def _referenced_chip_types(fleet: FleetConfig) -> list[str]:
    seen: dict[str, None] = {}
    for cl in fleet.clusters():
        for g, _ in _walk_groups(cl):
            if g.chip_type is not None:
                seen.setdefault(g.chip_type, None)
    return list(seen)


_JOB_CLASS_BY_NAME = {
    "pretrain": JobClass.PRETRAIN,
    "finetune": JobClass.FINETUNE,
    "eval": JobClass.EVAL,
    "infer_replica": JobClass.INFER_REPLICA,
}

_TIER_BY_NAME = {
    "free": Tier.FREE,
    "batch": Tier.BATCH,
    "prod": Tier.PROD,
    "monitoring": Tier.MONITORING,
}

_CLASS_KEYS = {
    "class",
    "rate_per_hour",
    "rate_per_day",
    "rate_per_week",
    "chips",
    "chip_type",
    "duration",
    "tier",
    "diurnal",
    "checkpoint_interval",
    "min_runtime",
    "max_lifetime",
    "within",
    "abort_prob",
    "n_tenants",
    "capacity",
    "gangs",
    "shape",
    "twisted",
}

#: Per-class abort_prob defaults (DESIGN §5.1: 30-40% of jobs end
#: killed/failed; inference replicas do not abort).  ``abort_prob: 0``
#: in YAML is the explicit opt-out.
_ABORT_PROB_DEFAULT = 0.3
_MIN_RUNTIME_DEFAULT_S = {JobClass.PRETRAIN: 7200.0}  # DESIGN §14 guard


def _parse_dist_field(
    spec: Mapping[str, Any], key: str, ctx: str, errors: list[str]
) -> DistSpec:
    if key not in spec:
        errors.append(f"{ctx}: missing required key {key!r}")
        return _INVALID_DIST
    try:
        return parse_dist(spec[key])
    except ValueError as exc:
        errors.append(f"{ctx}.{key}: {exc}")
        return _INVALID_DIST


def _parse_workload_class(
    name: str,
    spec: Mapping[str, Any],
    default_tenants: int,
    errors: list[str],
) -> WorkloadClassConfig:
    ctx = f"workload.classes.{name}"
    unknown = sorted(set(spec) - _CLASS_KEYS)
    if unknown:
        errors.append(f"{ctx}: unknown key(s): {', '.join(unknown)}")

    class_name = str(spec.get("class", name)).lower()
    job_class = _JOB_CLASS_BY_NAME.get(class_name)
    if job_class is None:
        errors.append(
            f"{ctx}: unknown job class {class_name!r}"
            f" (known: {', '.join(sorted(_JOB_CLASS_BY_NAME))})"
        )

    rate_keys = [
        k for k in ("rate_per_hour", "rate_per_day", "rate_per_week") if k in spec
    ]
    rate_per_hour = 0.0
    if len(rate_keys) != 1:
        errors.append(
            f"{ctx}: exactly one of rate_per_hour | rate_per_day | rate_per_week"
            f" is required (got {rate_keys or 'none'})"
        )
    else:
        k = rate_keys[0]
        try:
            raw = float(spec[k])
        except (TypeError, ValueError):
            errors.append(f"{ctx}.{k}: expected a number, got {spec[k]!r}")
            raw = 0.0
        divisor = {"rate_per_hour": 1.0, "rate_per_day": 24.0, "rate_per_week": 168.0}[k]
        rate_per_hour = raw / divisor
        if raw <= 0:
            errors.append(f"{ctx}.{k}: arrival rate must be positive, got {raw}")

    chips = _parse_dist_field(spec, "chips", ctx, errors)
    duration = _parse_dist_field(spec, "duration", ctx, errors)

    tier_raw = spec.get("tier")
    if tier_raw is None:
        tier = (
            Tier.PROD
            if job_class in (JobClass.PRETRAIN, JobClass.INFER_REPLICA)
            else Tier.BATCH
        )
    else:
        tier = _TIER_BY_NAME.get(str(tier_raw).lower())
        if tier is None:
            errors.append(
                f"{ctx}: unknown tier {tier_raw!r}"
                f" (known: {', '.join(_TIER_BY_NAME)})"
            )
            tier = Tier.BATCH

    ckpt_s = 3600.0
    if "checkpoint_interval" in spec:
        ckpt_s = _duration_us(
            spec["checkpoint_interval"], f"{ctx}.checkpoint_interval", errors, 0
        ) / S
    # DESIGN §5.1/§14 per-class default: pretrain 2 h preemption guard.
    min_runtime_s = _MIN_RUNTIME_DEFAULT_S.get(job_class, 0.0)
    if "min_runtime" in spec:
        min_runtime_s = _duration_us(
            spec["min_runtime"], f"{ctx}.min_runtime", errors, 0
        ) / S
    max_lifetime_s: float | None = None
    if "max_lifetime" in spec and spec["max_lifetime"] is not None:
        max_lifetime_s = _duration_us(
            spec["max_lifetime"], f"{ctx}.max_lifetime", errors, 0
        ) / S
        if max_lifetime_s <= 0:
            errors.append(
                f"{ctx}.max_lifetime: must be a positive duration"
                f" (omit the key for no cap)"
            )
            max_lifetime_s = None

    within_raw = spec.get("within")
    within: Constraint | None = None
    if within_raw is not None:
        if isinstance(within_raw, str):
            within = Constraint(level=within_raw)
        elif isinstance(within_raw, Mapping) and "level" in within_raw:
            within = Constraint(
                level=str(within_raw["level"]),
                required=bool(within_raw.get("required", True)),
                relax_after_s=float(within_raw.get("relax_after_s", 300.0)),
            )
        else:
            errors.append(
                f"{ctx}.within: expected a level name or {{level, required}},"
                f" got {within_raw!r}"
            )

    capacity = CapacityClass.ON_DEMAND
    if "capacity" in spec:
        cap_name = str(spec["capacity"]).upper()
        if cap_name in CapacityClass.__members__:
            capacity = CapacityClass[cap_name]
        else:
            errors.append(f"{ctx}: unknown capacity class {spec['capacity']!r}")

    shape_raw = spec.get("shape")
    shape: tuple[int, int, int] | None = None
    if shape_raw is not None:
        if isinstance(shape_raw, (list, tuple)) and len(shape_raw) == 3:
            shape = (int(shape_raw[0]), int(shape_raw[1]), int(shape_raw[2]))
        else:
            errors.append(f"{ctx}.shape: expected [a, b, c], got {shape_raw!r}")

    abort_default = (
        0.0 if job_class is JobClass.INFER_REPLICA else _ABORT_PROB_DEFAULT
    )
    chip_type_raw = spec.get("chip_type")
    return WorkloadClassConfig(
        name=name,
        job_class=job_class,
        rate_per_hour=rate_per_hour,
        chips=chips,
        duration=duration,
        tier=tier,
        diurnal=bool(spec.get("diurnal", False)),
        checkpoint_interval_s=ckpt_s,
        min_runtime_s=min_runtime_s,
        max_lifetime_s=max_lifetime_s,
        within=within,
        abort_prob=float(spec.get("abort_prob", abort_default)),
        n_tenants=int(spec.get("n_tenants", default_tenants)),
        chip_type=str(chip_type_raw) if chip_type_raw is not None else None,
        capacity=capacity,
        n_gangs=int(spec.get("gangs", 1)),
        shape=shape,
        twisted=bool(spec.get("twisted", False)),
    )


def _parse_workload(doc: Mapping[str, Any], errors: list[str]) -> WorkloadConfig:
    w = doc.get("workload")
    if w is None:
        errors.append("workload: section is required")
        return WorkloadConfig()
    if not isinstance(w, Mapping):
        errors.append(f"workload: expected a mapping, got {w!r}")
        return WorkloadConfig()
    kind = str(w.get("kind", "synthetic"))
    n_tenants = int(w.get("n_tenants", 8))
    source = w.get("source")
    classes: list[WorkloadClassConfig] = []
    raw_classes = w.get("classes") or {}
    if not isinstance(raw_classes, Mapping):
        errors.append("workload.classes: expected a mapping of class name -> spec")
        raw_classes = {}
    for cname, cspec in raw_classes.items():
        if not isinstance(cspec, Mapping):
            errors.append(f"workload.classes.{cname}: expected a mapping")
            continue
        classes.append(_parse_workload_class(str(cname), cspec, n_tenants, errors))
    return WorkloadConfig(
        kind=kind,
        classes=classes,
        source=str(source) if source is not None else None,
        n_tenants=n_tenants,
    )


_SERVICE_KEYS = {"id", "tenant", "replicas", "chip_type", "within", "tier"}


def _parse_services(doc: Mapping[str, Any], errors: list[str]) -> list[ServiceConfig]:
    raw = doc.get("services")
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"services: expected a list of mappings, got {raw!r}")
        return []
    out: list[ServiceConfig] = []
    for i, s in enumerate(raw):
        if not isinstance(s, Mapping):
            errors.append(f"services[{i}]: expected a mapping, got {s!r}")
            continue
        ctx = f"services[{i}]"
        unknown = sorted(set(s) - _SERVICE_KEYS)
        if unknown:
            errors.append(f"{ctx}: unknown key(s): {', '.join(unknown)}")
        sid = s.get("id")
        if not sid:
            errors.append(f"{ctx}: 'id' is required")
            continue
        ctx = f"services.{sid}"
        try:
            replicas = int(s["replicas"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{ctx}: 'replicas' must be an integer >= 0")
            replicas = 0
        tier_raw = s.get("tier")
        tier = Tier.PROD
        if tier_raw is not None:
            tier = _TIER_BY_NAME.get(str(tier_raw).lower(), None)
            if tier is None:
                errors.append(
                    f"{ctx}: unknown tier {tier_raw!r}"
                    f" (known: {', '.join(_TIER_BY_NAME)})"
                )
                tier = Tier.PROD
        within_raw = s.get("within")
        within = Constraint(level=str(within_raw)) if within_raw is not None else None
        ct = s.get("chip_type")
        out.append(
            ServiceConfig(
                id=str(sid),
                tenant=str(s.get("tenant", sid)),
                replicas=replicas,
                chip_type=str(ct) if ct is not None else None,
                within=within,
                tier=tier,
            )
        )
    return out


_TOP_LEVEL_KEYS = {
    "sim",
    "chip_types",
    "templates",
    "fleet",
    "failure_model",
    "workload",
    "scheduler",
    "services",
    "outputs",
    "reservations",
}


def _build_scenario(doc: Mapping[str, Any]) -> Scenario:
    errors: list[str] = []

    unknown = sorted(set(doc) - _TOP_LEVEL_KEYS)
    if unknown:
        errors.append(f"unknown top-level key(s): {', '.join(unknown)}")

    sim_raw = doc.get("sim") or {}
    if not isinstance(sim_raw, Mapping):
        errors.append(f"sim: expected a mapping, got {sim_raw!r}")
        sim_raw = {}
    horizon_us = 0
    if "horizon" in sim_raw:
        horizon_us = _duration_us(sim_raw["horizon"], "sim.horizon", errors, 0)
    round_us = 60 * S
    if "round" in sim_raw:
        round_us = _duration_us(sim_raw["round"], "sim.round", errors, 60 * S)
    seed = int(sim_raw.get("seed", 0))
    sim = SimConfig(horizon_us=horizon_us, round_us=round_us, seed=seed)

    failure_model = _parse_failure_model(
        doc.get("failure_model"), FailureModelConfig(), "failure_model", errors
    )
    fleet = _parse_fleet(doc, failure_model, errors)
    workload = _parse_workload(doc, errors)

    sched_raw = doc.get("scheduler") or {}
    if not isinstance(sched_raw, Mapping):
        errors.append(f"scheduler: expected a mapping, got {sched_raw!r}")
        sched_raw = {}
    scheduler = SchedulerConfig(
        name=str(sched_raw.get("name", "fifo")),
        params=dict(sched_raw.get("params") or {}),
    )

    out_raw = doc.get("outputs") or {}
    if not isinstance(out_raw, Mapping):
        errors.append(f"outputs: expected a mapping, got {out_raw!r}")
        out_raw = {}
    outputs = OutputsConfig(
        dir=out_raw.get("dir"),
        events=str(out_raw.get("events", "parquet")),
        plots=bool(out_raw.get("plots", False)),
        extra={k: v for k, v in out_raw.items() if k not in ("dir", "events", "plots")},
    )

    services = _parse_services(doc, errors)
    reservations = doc["reservations"] if "reservations" in doc else None

    return Scenario(
        sim=sim,
        fleet=fleet,
        workload=workload,
        scheduler=scheduler,
        outputs=outputs,
        failure_model=failure_model,
        services=services,
        reservations=reservations,
        load_errors=errors,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_pow2(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n > 0 and n & (n - 1) == 0


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_dist(spec: DistSpec, ctx: str, errors: list[str]) -> None:
    if spec.kind == "invalid":
        return  # the parse failure was already recorded in load_errors
    if spec.kind not in KNOWN_DIST_KINDS:
        errors.append(
            f"{ctx}: unknown distribution kind {spec.kind!r}"
            f" (known: {', '.join(sorted(KNOWN_DIST_KINDS))})"
        )
        return
    required = _DIST_POSITIONAL[spec.kind]
    missing = [k for k in required if k not in spec.params]
    extra = sorted(set(spec.params) - set(required))
    if missing:
        errors.append(
            f"{ctx}: {spec.kind} requires parameter(s) {', '.join(missing)}"
        )
    if extra:
        errors.append(
            f"{ctx}: unexpected parameter(s) for {spec.kind}: {', '.join(extra)}"
        )
    if missing or extra:
        return
    p = spec.params
    if spec.kind == "pow2":
        lo, hi = p["lo"], p["hi"]
        if not (_is_pow2(lo) and _is_pow2(hi)):
            errors.append(
                f"{ctx}: pow2 bounds must be powers of two, got lo={lo!r}, hi={hi!r}"
            )
        elif lo > hi:
            errors.append(f"{ctx}: pow2 requires lo <= hi, got lo={lo}, hi={hi}")
    elif spec.kind == "uniform":
        lo, hi = p["lo"], p["hi"]
        if not (_is_number(lo) and _is_number(hi)) or lo > hi:
            errors.append(f"{ctx}: uniform requires numeric lo <= hi, got {lo!r}, {hi!r}")
    elif spec.kind == "exponential":
        if not _is_number(p["mean"]) or p["mean"] <= 0:
            errors.append(f"{ctx}: exponential mean must be positive, got {p['mean']!r}")
    elif spec.kind == "lognormal":
        median, p90 = p["median"], p["p90"]
        if not (_is_number(median) and _is_number(p90)) or median <= 0 or p90 <= 0:
            errors.append(
                f"{ctx}: lognormal median and p90 must be positive,"
                f" got median={median!r}, p90={p90!r}"
            )
        elif p90 < median:
            errors.append(
                f"{ctx}: lognormal requires p90 >= median,"
                f" got median={median}, p90={p90}"
            )
    elif spec.kind == "fixed":
        if not _is_number(p["value"]):
            errors.append(f"{ctx}: fixed value must be a number, got {p['value']!r}")


def validate(scenario: Scenario) -> list[str]:
    """Validate a scenario, returning every problem as a human-readable
    error string (empty list = valid).  Includes ``scenario.load_errors``.

    Unknown scheduler names are deliberately NOT an error — the scheduler
    registry is checked at run time.  Schema-accepted but v0.1-unimplemented
    features are rejected with messages containing "not implemented in
    v0.1" (DESIGN principle 5).
    """
    errors: list[str] = list(scenario.load_errors)

    # --- sim ---
    if scenario.sim.horizon_us <= 0:
        errors.append("sim.horizon: must be a positive duration")
    if scenario.sim.round_us <= 0:
        errors.append("sim.round: must be a positive duration")

    # --- failure models (global + per-cluster lemons) ---
    def check_fm(fm: FailureModelConfig, ctx: str) -> None:
        if not 0.0 <= fm.lemon_frac <= 1.0:
            errors.append(
                f"{ctx}.lemon_frac: must be in [0, 1], got {fm.lemon_frac}"
            )
        if fm.lemon_multiplier <= 0:
            errors.append(
                f"{ctx}.lemon_multiplier: must be positive,"
                f" got {fm.lemon_multiplier}"
            )

    check_fm(scenario.failure_model, "failure_model")

    # --- fleet ---
    clusters = scenario.fleet.clusters()
    if not clusters:
        errors.append("fleet: at least one cluster is required")
    level_vocab: set[str] = set()
    for cl in clusters:
        check_fm(cl.failure_model, f"fleet.clusters.{cl.id}.failure_model")
    for cl in clusters:
        ctx = f"fleet.clusters.{cl.id}"
        level_vocab.update(cl.levels)
        if len(set(cl.levels)) != len(cl.levels):
            errors.append(f"{ctx}: duplicate level names in {cl.levels}")
        if cl.attrs.get("ocs_pool"):
            errors.append(f"{ctx}: attr 'ocs_pool' (TPU OCS predicates) is {_NOT_V01}")
        if not cl.children:
            errors.append(f"{ctx}: cluster has no children")
        level_index = {lvl: i for i, lvl in enumerate(cl.levels)}
        for group, parent in _walk_groups(cl):
            gctx = f"{ctx} (level {group.level!r})"
            if group.count <= 0:
                errors.append(f"{gctx}: count must be positive, got {group.count}")
            if group.level not in level_index:
                errors.append(
                    f"{gctx}: level not in this cluster's levels list {cl.levels}"
                )
            elif parent is not None and parent.level in level_index:
                if level_index[group.level] <= level_index[parent.level]:
                    errors.append(
                        f"{ctx}: level {group.level!r} must be deeper than its"
                        f" parent {parent.level!r} in {cl.levels}"
                    )
            if group.attrs.get("ocs_pool"):
                errors.append(
                    f"{gctx}: attr 'ocs_pool' (TPU OCS predicates) is {_NOT_V01}"
                )
            if not group.children:  # leaf = node
                if group.chips <= 0:
                    errors.append(
                        f"{gctx}: leaf nodes need a positive chip count,"
                        f" got {group.chips}"
                    )
                if group.chip_type is None:
                    errors.append(f"{gctx}: leaf nodes need a chip_type")
                elif group.chip_type not in scenario.fleet.chip_types:
                    errors.append(
                        f"{gctx}: unknown chip_type {group.chip_type!r}"
                        f" (declared: {', '.join(scenario.fleet.chip_types) or 'none'})"
                    )

    # --- workload ---
    w = scenario.workload
    if w.kind not in ("synthetic", "trace"):
        errors.append(
            f"workload.kind: expected 'synthetic' or 'trace', got {w.kind!r}"
        )
    if w.kind == "trace" and not w.source:
        errors.append("workload.source: required when kind is 'trace'")
    if w.kind == "synthetic" and not w.classes:
        errors.append("workload.classes: at least one class is required")
    # Leaf chip types actually present in the fleet: with >1, synthetic
    # classes must pin (DESIGN §11 — unpinned chips on a heterogeneous
    # fleet would be fake heterogeneity).
    fleet_leaf_types = _referenced_chip_types(scenario.fleet)
    heterogeneous = len(fleet_leaf_types) > 1
    for c in w.classes:
        ctx = f"workload.classes.{c.name}"
        _validate_dist(c.chips, f"{ctx}.chips", errors)
        _validate_dist(c.duration, f"{ctx}.duration", errors)
        if c.chip_type is not None and c.chip_type not in scenario.fleet.chip_types:
            errors.append(
                f"{ctx}.chip_type: unknown chip_type {c.chip_type!r}"
                f" (declared: {', '.join(scenario.fleet.chip_types) or 'none'})"
            )
        elif c.chip_type is not None and c.chip_type not in fleet_leaf_types:
            errors.append(
                f"{ctx}.chip_type: no fleet leaves carry {c.chip_type!r}"
                f" (present: {', '.join(fleet_leaf_types) or 'none'})"
            )
        if heterogeneous and c.chip_type is None:
            errors.append(
                f"{ctx}: chip_type is required — the fleet has multiple chip"
                f" types ({', '.join(fleet_leaf_types)}) and v0.1 pins chips"
                f" per job (DESIGN §11)"
            )
        if c.within is not None:
            if c.within.level not in level_vocab:
                errors.append(
                    f"{ctx}.within: unknown level {c.within.level!r}"
                    f" (known: {', '.join(sorted(level_vocab)) or 'none'})"
                )
            if not c.within.required:
                errors.append(
                    f"{ctx}.within: preferred (relaxable) constraints are {_NOT_V01}"
                )
        if not 0.0 <= c.abort_prob <= 1.0:
            errors.append(
                f"{ctx}.abort_prob: must be in [0, 1], got {c.abort_prob}"
            )
        if c.n_tenants < 1:
            errors.append(f"{ctx}.n_tenants: must be >= 1, got {c.n_tenants}")
        if c.capacity is not CapacityClass.ON_DEMAND:
            errors.append(
                f"{ctx}.capacity: capacity class"
                f" {c.capacity.name.lower()!r} is {_NOT_V01} (only on_demand)"
            )
        if c.n_gangs < 1:
            errors.append(f"{ctx}.gangs: must be positive, got {c.n_gangs}")
        elif c.n_gangs > 1:
            errors.append(f"{ctx}.gangs: multi-gang jobs are {_NOT_V01}")
        if c.shape is not None:
            errors.append(f"{ctx}.shape: TPU shape requests are {_NOT_V01}")
        if c.twisted:
            errors.append(f"{ctx}.twisted: twisted TPU slices are {_NOT_V01}")

    # --- services (DESIGN §5; v1 frozen replicas, synthetic only) ---
    if scenario.services and w.kind == "trace":
        errors.append(
            f"services: combined with workload.kind 'trace' is {_NOT_V01}"
            " (services need the synthetic pipeline)"
        )
    seen_ids: set[str] = set()
    for svc in scenario.services:
        ctx = f"services.{svc.id}"
        if svc.id in seen_ids:
            errors.append(f"{ctx}: duplicate service id")
        seen_ids.add(svc.id)
        if svc.replicas < 0:
            errors.append(f"{ctx}: replicas must be >= 0, got {svc.replicas}")
        if svc.chip_type is not None and svc.chip_type not in scenario.fleet.chip_types:
            errors.append(
                f"{ctx}.chip_type: unknown chip_type {svc.chip_type!r}"
                f" (declared: {', '.join(scenario.fleet.chip_types) or 'none'})"
            )
        if svc.chip_type is None and len(_referenced_chip_types(scenario.fleet)) > 1:
            errors.append(
                f"{ctx}: chip_type is required — the fleet has multiple chip"
                f" types and v0.1 pins chips per job (DESIGN §11)"
            )
        if svc.within is not None and svc.within.level not in level_vocab:
            errors.append(
                f"{ctx}.within: unknown level {svc.within.level!r}"
                f" (known: {', '.join(sorted(level_vocab)) or 'none'})"
            )

    # --- outputs ---
    if scenario.outputs.events != "parquet":
        errors.append(
            f"outputs.events: {scenario.outputs.events!r} is {_NOT_V01}"
            " (v0.1 writes jobs.parquet/timeseries.parquet/summary.json;"
            " only 'parquet' is accepted)"
        )

    # --- reservations ---
    if scenario.reservations is not None:
        errors.append(f"reservations: are {_NOT_V01}")

    # scheduler.name deliberately unchecked (registry is a run-time concern)
    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_scenario(
    source: str | Path | Mapping[str, Any], *, strict: bool = True
) -> Scenario:
    """Load a scenario from a YAML file path or an already-parsed mapping.

    With ``strict=True`` (default), raises :class:`ScenarioError` carrying
    the full ``validate`` error list if anything is wrong.  With
    ``strict=False``, returns the best-effort :class:`Scenario`; call
    ``validate(scenario)`` for the error list.  Structurally unreadable
    input (unreadable YAML, non-mapping document) always raises.
    """
    if isinstance(source, Mapping):
        doc: Any = source
    else:
        path = Path(source)
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
    scenario = _build_scenario(doc)
    if strict:
        errors = validate(scenario)
        if errors:
            raise ScenarioError(errors)
    return scenario
