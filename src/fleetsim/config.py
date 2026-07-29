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

Distribution expressions (``pow2[1, 8]``, ``lognormal[median=2m, p90=30m]``
or ``p99=...``, ``exponential[mean=30s]``, ``pareto[alpha=1.5, xm=1h]``,
``weibull[shape=1.5, scale=10m]``, ``uniform[a, b]``, ``fixed[x]``) parse
into declarative :class:`DistSpec` records; sampling is the workload
phase's job.  Two v0.2 MAPPING forms exist: ``chips: {pmf: {1: .55, ...}}``
(or a preset name from :data:`PMF_PRESETS`) and ``duration: {body:
lognormal[...], tail: {alpha, splice, cap}}`` (the lognormal-body /
Pareto-tail splice).  NOTE: inside YAML *flow* mappings ``{...}`` the
brackets must be quoted (``chips: "pow2[1, 8]"``); block-context plain
scalars need no quotes.

v0.2 traffic surface (docs/traffic-math.md): per class,
``arrival: {process: poisson|nhpp|mmpp2|hawkes|closed_loop,
rate_per_hour|rate_per_day|rate_per_week: <mean rate, inside the block>,
seasonality: null | helios_v01 | v01_steps | {daily: [[a,b],...],
weekly: [[A,B],...]}, hawkes: {branching, kernel_tau},
mmpp2: {rate_ratio, burst_frac, switch_tau},
closed_loop: {target_pending | concurrency}}``.  The v0.1 sugar
(top-level ``rate_*`` + ``diurnal``) normalizes to poisson / nhpp on the
pinned ``v01_steps`` curve.  ``workload.preset: google_fleet`` (with
optional ``scale`` chips and per-class ``classes`` overrides) expands to
the stylized multi-tenant mix.  ``workload.tenant_zipf_s`` (default 1.2)
sets the finite-Zipf tenant exponent, per-class overridable.

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

import math
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
    "SeasonalityConfig",
    "ArrivalProcessConfig",
    "ARRIVAL_PROCESSES",
    "SEASONALITY_PRESETS",
    "HELIOS_V01_DAILY",
    "PMF_PRESETS",
    "TENANT_ZIPF_S_DEFAULT",
    "WORKLOAD_PRESETS",
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
    "PenaltiesConfig",
    "QuotaConfig",
    "ReservationConfig",
    "SimConfig",
    "Scenario",
    "ScenarioError",
    "parse_dist",
    "load_scenario",
    "validate",
]

_NOT_V01 = "not implemented in v0.1"
_NOT_V04 = "not implemented in v0.4"


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


#: Positional parameter names per known distribution kind.  ``backlog``
#: is an ARRIVAL spec only (closed-loop standing backlog, v0.2); it is
#: rejected for chips/duration by :func:`_validate_dist`.  ``lognormal``
#: also accepts a named ``p99`` INSTEAD of ``p90`` (exactly one).
_DIST_POSITIONAL: dict[str, tuple[str, ...]] = {
    "pow2": ("lo", "hi"),
    "uniform": ("lo", "hi"),
    "fixed": ("value",),
    "exponential": ("mean",),
    "lognormal": ("median", "p90"),
    "pareto": ("alpha", "xm"),
    "weibull": ("shape", "scale"),
    "backlog": ("target_pending",),
}

#: ``pmf`` and ``splice`` are MAPPING-ONLY kinds (v0.2): written as YAML
#: mappings (``chips: {pmf: {...}}``, ``duration: {body: ..., tail:
#: {...}}``), never as bracket expressions.
KNOWN_DIST_KINDS = frozenset(_DIST_POSITIONAL) | {"pmf", "splice"}

#: Named size pmfs (traffic-math §2.2, trace-derived).  Keys are chip
#: counts; weights are normalized at load.  ``tpu_isca23`` follows the
#: TPU v4 ISCA '23 slice histogram anchors (29% < 64 chips, 14% at 64,
#: 18% at 128–192, 8% at 2–3K).
PMF_PRESETS: dict[str, dict[int, float]] = {
    "eval_v02": {1: 0.55, 2: 0.15, 4: 0.15, 8: 0.15},
    "finetune_v02": {8: 0.30, 16: 0.25, 32: 0.20, 64: 0.15, 128: 0.10},
    "pretrain_v02": {
        256: 0.28, 512: 0.22, 1024: 0.18, 2048: 0.14,
        4096: 0.09, 8192: 0.06, 16384: 0.03,
    },
    "tpu_isca23": {
        16: 0.12, 32: 0.17, 64: 0.14, 128: 0.18,
        256: 0.14, 512: 0.09, 1024: 0.08, 2048: 0.08,
    },
}

#: Standard-normal quantiles used to derive the splice point from the
#: body's p90/p99 (pinned; must match fleetsim.workload.distributions).
_Z_BY_QUANTILE = {0.90: 1.2816, 0.99: 2.3263}

#: Default finite-Zipf tenant exponent (traffic-math §2.4: s = 1.19
#: reproduces PAI's top-5% -> 77% of jobs; 0.9 is the Helios GPU-time
#: preset).  Replaces the v0.1 hardcoded 1.5 — a pinned behavior change.
TENANT_ZIPF_S_DEFAULT = 1.2

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
# Arrival-process configuration (v0.2, traffic-math §2.1)
# ---------------------------------------------------------------------------


#: Harmonic coefficients of the ``helios_v01`` seasonality preset: the
#: least-squares fit of K=2 daily harmonics to the log of the v0.1
#: pinned step curve (traffic-math §2.1).  Weekly part is flat.
HELIOS_V01_DAILY: tuple[tuple[float, float], ...] = (
    (-0.205, -0.199),
    (-0.001, -0.221),
)

#: Known seasonality preset names.  ``v01_steps`` is the v0.1 pinned
#: 3-step diurnal curve (byte-identical arrivals to v0.1 ``diurnal:
#: true``); ``helios_v01`` is its smooth harmonic fit.
SEASONALITY_PRESETS = ("helios_v01", "v01_steps")

ARRIVAL_PROCESSES = ("poisson", "nhpp", "mmpp2", "hawkes", "closed_loop")


@dataclass(frozen=True)
class SeasonalityConfig:
    """Log-linear harmonic seasonality (traffic-math §2.1).

    ``daily`` holds up to 3 ``(a_k, b_k)`` cos/sin pairs on the log-rate
    at the 24 h fundamental; ``weekly`` up to 2 pairs at 168 h.  When
    ``preset`` is set the pairs are empty and the preset defines the
    curve.  The normalization ``theta0`` is DERIVED at sampling time
    (weekly-mean normalization to the configured rate), never stored.
    """

    daily: tuple[tuple[float, float], ...] = ()
    weekly: tuple[tuple[float, float], ...] = ()
    preset: str | None = None


@dataclass(frozen=True)
class ArrivalProcessConfig:
    """One class's arrival process (v0.2).  ``process`` is one of
    ``poisson | nhpp | mmpp2 | hawkes`` (``closed_loop`` normalizes to a
    ``backlog`` DistSpec at parse and never appears here).  The
    process-specific fields carry doc-pinned defaults; ``*_s`` fields
    are float seconds.  The stationary MEAN rate lives on the class's
    ``rate_per_hour`` (all processes normalize to it)."""

    process: str
    seasonality: SeasonalityConfig | None = None
    hawkes_branching: float = 0.4        # n = alpha/beta, stationary iff < 1
    hawkes_kernel_tau_s: float = 900.0   # 1/beta
    mmpp2_rate_ratio: float = 4.0        # lambda_burst / lambda_quiet
    mmpp2_burst_frac: float = 0.25       # stationary burst-state fraction
    mmpp2_switch_tau_s: float = 172800.0  # 1/(sigma1+sigma2)


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
    per job).

    v0.2 additions: ``arrival`` (a ``backlog[target_pending=N]``
    DistSpec) declares a CLOSED-LOOP standing-backlog class instead of a
    Poisson rate — mutually exclusive with the rate keys; such classes
    default to ``tier: best_effort`` and ``min_runtime: 0`` and
    ``rate_per_hour`` is 0.  ``arrival`` may instead be a MAPPING
    (traffic-math §2.1) selecting a process (poisson | nhpp | mmpp2 |
    hawkes | closed_loop) with the rate key INSIDE the mapping; the
    parsed result lands in ``arrival_process`` (closed_loop normalizes
    to the backlog DistSpec).  The v0.1 sugar — top-level ``rate_*``
    plus ``diurnal`` — is normalized at parse to ``arrival_process``
    poisson (diurnal false) or nhpp with the ``v01_steps`` seasonality
    (diurnal true, byte-identical arrivals to v0.1).  ``tenant_zipf_s``
    is the finite-Zipf tenant exponent (inherits the workload-level
    value, default 1.2).  ``segment_nodes`` + ``segment_level``
    (both-or-neither) declare a segmented gang: whole-node blocks of
    ``segment_nodes`` nodes, each block within one ``segment_level``
    domain, with ``within`` as the OUTER constraint at a higher level.

    ``capacity``, ``n_gangs``, ``shape`` and ``twisted`` are
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
    arrival: DistSpec | None = None
    arrival_process: ArrivalProcessConfig | None = None
    segment_nodes: int | None = None
    segment_level: str | None = None
    abort_prob: float = 0.0
    n_tenants: int = 8
    tenant_zipf_s: float = TENANT_ZIPF_S_DEFAULT
    #: Fixed tenant name for EVERY job of this class (v0.4) — bypasses
    #: the Zipf tenant draw entirely (the ``tenant/<class>`` stream is
    #: reserved but not consumed).  None = Zipf marking over n_tenants.
    tenant: str | None = None
    chip_type: str | None = None
    capacity: CapacityClass = CapacityClass.ON_DEMAND
    n_gangs: int = 1
    shape: tuple[int, int, int] | None = None
    twisted: bool = False


@dataclass
class WorkloadConfig:
    """``kind`` is "synthetic" (uses ``classes``) or "trace" (uses
    ``source``).  ``n_tenants`` and ``tenant_zipf_s`` are the defaults a
    class inherits when it does not set its own.  ``preset`` records the
    workload preset name that was expanded (e.g. ``google_fleet``), or
    None; expansion happens at parse, so ``classes`` always holds the
    final merged class list."""

    kind: str = "synthetic"
    classes: list[WorkloadClassConfig] = field(default_factory=list)
    source: str | None = None
    n_tenants: int = 8
    tenant_zipf_s: float = TENANT_ZIPF_S_DEFAULT
    preset: str | None = None


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
    """File-output selection.  ``stints`` (v0.3 visualizer) opts into
    ``stints.parquet`` — one row per (allocation stint x domain): a level
    name records domains at that level, ``True`` means the level directly
    below each cluster root, and ``None`` (the default, key absent)
    disables the file entirely (existing scenarios' outputs stay
    byte-identical)."""

    dir: str | None = None
    events: str = "parquet"
    plots: bool = False
    stints: str | bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PenaltiesConfig:
    """Placement-quality speed penalties (v0.4, the v0.3-designed
    relax/penalty matched pair).

    ``xover`` maps a level name to a speed MULTIPLIER in (0, 1]: a gang
    whose leaves do NOT all sit under one domain at that level runs at
    ``multiplier`` x speed (multipliers of several configured levels
    multiply).  This covers both segmented multi-pod gangs and RELAXED
    ``within`` placements — relaxing a constraint is never free once a
    penalty is configured (DESIGN §4.2: the matched pair).  Absent
    section = every speed stays exactly 1.0 (byte-identical outputs).
    """

    xover: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QuotaConfig:
    """Tenant chip quota (v0.4; MAST-INSPIRED admission-time model).

    ``tenants`` maps tenant name -> concurrent in-quota chip cap.  At
    ADMISSION, a job whose tenant's committed in-quota chips (pending +
    running in-quota jobs) would exceed the cap is OVER-QUOTA:
    ``over_quota: best_effort`` (default) demotes it to the BEST_EFFORT
    band (it runs as preemptible scavenger work, ``Job.quota_demoted``
    set); ``over_quota: reject`` fails it at admission.  Tenants absent
    from the table are unlimited; ``validate`` rejects capped names no
    configured workload class or service can produce.

    HONEST SEMANTICS (pinned; DESIGN §17.3): commitment is charged over
    QUEUED DEMAND (pending + running) at admission time and demotion is
    IRREVERSIBLE — deliberately simpler than MAST/HyperPod, which
    evaluate in-quota vs over-quota against RUNNING usage at scheduling
    time and treat over-quota as a dynamic, reclaimable state.  A burst
    of short jobs therefore demotes everything past the cap even on an
    idle fleet.  Scheduling-time evaluation is future work.
    """

    tenants: dict[str, int] = field(default_factory=dict)
    over_quota: str = "best_effort"


@dataclass(frozen=True)
class ReservationConfig:
    """One calendar capacity block (v0.4; CapacityClass.CALENDAR
    semantics — reservation-as-meta-job).

    The engine claims ``chips`` worth of WHOLE nodes (rounded up) inside
    one ``level`` domain at ``start_us``, evicting residents (REQUEUE,
    trigger ``"reservation"``); during ``[start, end)`` only ``tenant``'s
    jobs may be placed on the held nodes.  With ``hard_end`` (default
    True — capacity blocks end hard) residents still on the hold at
    ``end_us`` are evicted; the marker always lifts at ``end_us``.
    ``chip_type`` is required on heterogeneous fleets.
    """

    id: str
    tenant: str
    chips: int
    start_us: int
    end_us: int
    level: str | None = None
    chip_type: str | None = None
    hard_end: bool = True


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
    reservations: list[ReservationConfig] = field(default_factory=list)
    penalties: PenaltiesConfig | None = None
    quota: QuotaConfig | None = None
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

#: YAML tier spellings.  ``best_effort`` is the canonical band-0 name
#: (v0.2); ``free`` is the accepted legacy spelling for the same band.
_TIER_BY_NAME = {
    "best_effort": Tier.BEST_EFFORT,
    "free": Tier.BEST_EFFORT,  # legacy alias
    "batch": Tier.BATCH,
    "prod": Tier.PROD,
    "monitoring": Tier.MONITORING,
}

_CLASS_KEYS = {
    "class",
    "rate_per_hour",
    "rate_per_day",
    "rate_per_week",
    "arrival",
    "chips",
    "chip_type",
    "duration",
    "tier",
    "diurnal",
    "checkpoint_interval",
    "min_runtime",
    "max_lifetime",
    "within",
    "segment_nodes",
    "segment_level",
    "abort_prob",
    "n_tenants",
    "tenant",
    "tenant_zipf_s",
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


def _parse_pmf_mapping(raw: Any, ctx: str, errors: list[str]) -> DistSpec:
    """Parse ``{pmf: {chips: weight, ...}}`` or ``{pmf: <preset name>}``
    into a ``pmf`` DistSpec (params keyed by str(chips), weights
    normalized here).  Records errors and returns the invalid sentinel
    on failure."""
    if isinstance(raw, str):
        preset = PMF_PRESETS.get(raw)
        if preset is None:
            errors.append(
                f"{ctx}: unknown pmf preset {raw!r}"
                f" (known: {', '.join(sorted(PMF_PRESETS))})"
            )
            return _INVALID_DIST
        raw = preset
    if not isinstance(raw, Mapping) or not raw:
        errors.append(
            f"{ctx}: pmf expects a non-empty {{chips: weight}} mapping"
            f" or a preset name, got {raw!r}"
        )
        return _INVALID_DIST
    entries: list[tuple[int, float]] = []
    ok = True
    for k, v in raw.items():
        if isinstance(k, bool) or not isinstance(k, int):
            errors.append(f"{ctx}: pmf keys must be integers, got {k!r}")
            ok = False
            continue
        if not _is_number(v) or float(v) <= 0:
            errors.append(
                f"{ctx}: pmf weight for {k} must be a positive number,"
                f" got {v!r}"
            )
            ok = False
            continue
        entries.append((k, float(v)))
    if not ok or not entries:
        return _INVALID_DIST
    entries.sort()
    total = sum(w for _, w in entries)
    return DistSpec("pmf", {str(k): w / total for k, w in entries})


_SPLICE_TAIL_KEYS = frozenset({"alpha", "splice", "cap"})


def _parse_splice_mapping(
    m: Mapping[str, Any], ctx: str, errors: list[str]
) -> DistSpec:
    """Parse ``{body: lognormal[...], tail: {alpha, splice, cap}}`` into
    a ``splice`` DistSpec (traffic-math §2.3).  Params: ``median`` +
    ``p90``|``p99`` (int µs, from the body), ``alpha`` (float > 1,
    validated later), ``cap`` (int µs), and the splice point as either
    ``splice_q`` (0.90 | 0.99) or ``splice_at`` (int µs)."""
    unknown = sorted(set(m) - {"body", "tail"})
    if unknown:
        errors.append(
            f"{ctx}: unknown key(s) in splice duration: {', '.join(unknown)}"
            f" (known: body, tail)"
        )
    body_raw = m.get("body")
    tail = m.get("tail")
    if body_raw is None or not isinstance(tail, Mapping):
        errors.append(
            f"{ctx}: splice duration requires 'body' (a lognormal"
            f" expression) and 'tail' (a mapping with alpha, cap)"
        )
        return _INVALID_DIST
    try:
        body = parse_dist(body_raw)
    except ValueError as exc:
        errors.append(f"{ctx}.body: {exc}")
        return _INVALID_DIST
    if body.kind != "lognormal":
        errors.append(
            f"{ctx}.body: splice body must be lognormal, got {body.kind!r}"
        )
        return _INVALID_DIST
    unknown_t = sorted(set(tail) - _SPLICE_TAIL_KEYS)
    if unknown_t:
        errors.append(
            f"{ctx}.tail: unknown key(s): {', '.join(unknown_t)}"
            f" (known: {', '.join(sorted(_SPLICE_TAIL_KEYS))})"
        )
    params: dict[str, int | float] = {}
    for k in ("median", "p90", "p99"):
        if k in body.params:
            params[k] = body.params[k]
    alpha = tail.get("alpha")
    if not _is_number(alpha):
        errors.append(f"{ctx}.tail.alpha: expected a number, got {alpha!r}")
        return _INVALID_DIST
    params["alpha"] = float(alpha)
    if "cap" not in tail:
        errors.append(
            f"{ctx}.tail: 'cap' is required (untruncated Pareto tails"
            f" make sample means non-convergent, traffic-math §2.3)"
        )
        return _INVALID_DIST
    params["cap"] = _duration_us(tail["cap"], f"{ctx}.tail.cap", errors, 0)
    splice = tail.get("splice", "p90")
    if splice in ("p90", "p99"):
        params["splice_q"] = 0.90 if splice == "p90" else 0.99
    else:
        params["splice_at"] = _duration_us(
            splice, f"{ctx}.tail.splice", errors, 0
        )
    return DistSpec("splice", params)


def _parse_dist_field(
    spec: Mapping[str, Any], key: str, ctx: str, errors: list[str]
) -> DistSpec:
    if key not in spec:
        errors.append(f"{ctx}: missing required key {key!r}")
        return _INVALID_DIST
    raw = spec[key]
    if isinstance(raw, Mapping):
        if "pmf" in raw:
            unknown = sorted(set(raw) - {"pmf"})
            if unknown:
                errors.append(
                    f"{ctx}.{key}: unknown key(s) beside pmf:"
                    f" {', '.join(unknown)}"
                )
            return _parse_pmf_mapping(raw["pmf"], f"{ctx}.{key}", errors)
        if "body" in raw or "tail" in raw:
            return _parse_splice_mapping(raw, f"{ctx}.{key}", errors)
        errors.append(
            f"{ctx}.{key}: mapping form must be {{pmf: ...}} or"
            f" {{body: ..., tail: ...}}, got keys {sorted(raw)}"
        )
        return _INVALID_DIST
    try:
        return parse_dist(raw)
    except ValueError as exc:
        errors.append(f"{ctx}.{key}: {exc}")
        return _INVALID_DIST


def _parse_harmonic_pairs(
    raw: Any, ctx: str, max_pairs: int, errors: list[str]
) -> tuple[tuple[float, float], ...]:
    """Parse ``[[a, b], ...]`` harmonic coefficient pairs (≤ max_pairs)."""
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        errors.append(f"{ctx}: expected a list of [a, b] pairs, got {raw!r}")
        return ()
    if len(raw) > max_pairs:
        errors.append(
            f"{ctx}: at most {max_pairs} harmonic pairs allowed,"
            f" got {len(raw)}"
        )
        return ()
    out: list[tuple[float, float]] = []
    for i, pair in enumerate(raw):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(_is_number(x) for x in pair)
        ):
            errors.append(
                f"{ctx}[{i}]: expected a [a, b] number pair, got {pair!r}"
            )
            return ()
        out.append((float(pair[0]), float(pair[1])))
    return tuple(out)


def _parse_seasonality(
    raw: Any, ctx: str, errors: list[str]
) -> SeasonalityConfig | None:
    """Parse an ``arrival.seasonality`` value: null (flat), a preset
    name, or ``{daily: [[a,b],...], weekly: [[A,B],...]}`` with ≤3 daily
    and ≤2 weekly pairs (traffic-math config surface)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw not in SEASONALITY_PRESETS:
            errors.append(
                f"{ctx}: unknown seasonality preset {raw!r}"
                f" (known: {', '.join(SEASONALITY_PRESETS)})"
            )
            return None
        return SeasonalityConfig(preset=raw)
    if not isinstance(raw, Mapping):
        errors.append(
            f"{ctx}: expected null, a preset name, or"
            f" {{daily, weekly}}, got {raw!r}"
        )
        return None
    unknown = sorted(set(raw) - {"daily", "weekly"})
    if unknown:
        errors.append(
            f"{ctx}: unknown key(s): {', '.join(unknown)}"
            f" (known: daily, weekly)"
        )
    daily = _parse_harmonic_pairs(raw.get("daily"), f"{ctx}.daily", 3, errors)
    weekly = _parse_harmonic_pairs(raw.get("weekly"), f"{ctx}.weekly", 2, errors)
    if not daily and not weekly:
        errors.append(
            f"{ctx}: at least one daily or weekly harmonic pair required"
            f" (write 'seasonality: null' for a flat rate)"
        )
        return None
    return SeasonalityConfig(daily=daily, weekly=weekly)


_ARRIVAL_KEYS = frozenset(
    {
        "process",
        "rate_per_hour",
        "rate_per_day",
        "rate_per_week",
        "seasonality",
        "hawkes",
        "mmpp2",
        "closed_loop",
    }
)
_RATE_DIVISOR = {"rate_per_hour": 1.0, "rate_per_day": 24.0, "rate_per_week": 168.0}


def _parse_arrival_mapping(
    m: Mapping[str, Any], ctx: str, errors: list[str]
) -> tuple[ArrivalProcessConfig | None, DistSpec | None, float]:
    """Parse the mapping form of ``arrival`` (traffic-math §2.1).

    Returns ``(arrival_process, backlog_spec, rate_per_hour)``:
    open-loop processes yield an :class:`ArrivalProcessConfig` plus the
    normalized mean rate; ``process: closed_loop`` yields a ``backlog``
    DistSpec (reusing the standing-backlog machinery) and rate 0.
    All structural and range validation happens here (recorded into
    ``errors``)."""
    unknown = sorted(set(m) - _ARRIVAL_KEYS)
    if unknown:
        errors.append(
            f"{ctx}: unknown key(s): {', '.join(unknown)}"
            f" (known: {', '.join(sorted(_ARRIVAL_KEYS))})"
        )
    process = m.get("process")
    if process not in ARRIVAL_PROCESSES:
        errors.append(
            f"{ctx}.process: unknown arrival process {process!r}"
            f" (known: {', '.join(ARRIVAL_PROCESSES)})"
        )
        return None, None, 0.0

    rate_keys = [k for k in _RATE_DIVISOR if k in m]
    rate_per_hour = 0.0
    if process == "closed_loop":
        if rate_keys:
            errors.append(
                f"{ctx}: rate keys ({', '.join(rate_keys)}) are forbidden"
                f" with process 'closed_loop' (a saturated closed source"
                f" has no arrival rate)"
            )
    elif len(rate_keys) != 1:
        errors.append(
            f"{ctx}: exactly one of rate_per_hour | rate_per_day |"
            f" rate_per_week is required inside the arrival block"
            f" (got {rate_keys or 'none'})"
        )
    else:
        k = rate_keys[0]
        try:
            raw_rate = float(m[k])
        except (TypeError, ValueError):
            errors.append(f"{ctx}.{k}: expected a number, got {m[k]!r}")
            raw_rate = 0.0
        rate_per_hour = raw_rate / _RATE_DIVISOR[k]
        if raw_rate <= 0:
            errors.append(
                f"{ctx}.{k}: arrival rate must be positive, got {raw_rate}"
            )

    seasonality: SeasonalityConfig | None = None
    if "seasonality" in m:
        if process in ("poisson", "closed_loop"):
            if m["seasonality"] is not None:
                errors.append(
                    f"{ctx}.seasonality: not accepted with process"
                    f" {process!r} (use 'nhpp' for a seasonal rate)"
                )
        else:
            seasonality = _parse_seasonality(
                m["seasonality"], f"{ctx}.seasonality", errors
            )
    if process == "nhpp" and seasonality is None:
        errors.append(
            f"{ctx}: process 'nhpp' requires a seasonality (a flat nhpp"
            f" IS poisson — write process: poisson instead)"
        )

    for block, proc in (("hawkes", "hawkes"), ("mmpp2", "mmpp2"),
                        ("closed_loop", "closed_loop")):
        if block in m and process != proc:
            errors.append(
                f"{ctx}.{block}: only accepted with process {proc!r}"
                f" (got process {process!r})"
            )

    if process == "closed_loop":
        cl = m.get("closed_loop") or {}
        if not isinstance(cl, Mapping):
            errors.append(f"{ctx}.closed_loop: expected a mapping, got {cl!r}")
            cl = {}
        unknown_cl = sorted(set(cl) - {"target_pending", "concurrency", "think_time"})
        if unknown_cl:
            errors.append(
                f"{ctx}.closed_loop: unknown key(s): {', '.join(unknown_cl)}"
                f" (known: target_pending, concurrency, think_time)"
            )
        if "target_pending" in cl and "concurrency" in cl:
            errors.append(
                f"{ctx}.closed_loop: give target_pending OR concurrency"
                f" (aliases), not both"
            )
        target = cl.get("target_pending", cl.get("concurrency", 4))
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            errors.append(
                f"{ctx}.closed_loop: target_pending must be an integer"
                f" >= 1, got {target!r}"
            )
            target = 1
        if "think_time" in cl:
            try:
                tt = parse_dist(cl["think_time"])
            except ValueError as exc:
                errors.append(f"{ctx}.closed_loop.think_time: {exc}")
                tt = None
            if tt is not None and (
                tt.kind != "fixed" or tt.params.get("value") not in (0, 0.0)
            ):
                errors.append(
                    f"{ctx}.closed_loop.think_time: only fixed[0s] is"
                    f" implemented in v0.2 (omit the key)"
                )
        return None, DistSpec("backlog", {"target_pending": target}), 0.0

    cfg = ArrivalProcessConfig(process=str(process), seasonality=seasonality)
    if process == "hawkes":
        hk = m.get("hawkes") or {}
        if not isinstance(hk, Mapping):
            errors.append(f"{ctx}.hawkes: expected a mapping, got {hk!r}")
            hk = {}
        unknown_h = sorted(set(hk) - {"branching", "kernel_tau"})
        if unknown_h:
            errors.append(
                f"{ctx}.hawkes: unknown key(s): {', '.join(unknown_h)}"
                f" (known: branching, kernel_tau)"
            )
        branching = hk.get("branching", 0.4)
        if not _is_number(branching) or not 0.0 <= float(branching) < 1.0:
            errors.append(
                f"{ctx}.hawkes.branching: must be in [0, 1) — the mean"
                f" children per event; >= 1 is supercritical (non-"
                f"stationary cascade) — got {branching!r}"
            )
            branching = 0.4
        tau_us = (
            _duration_us(hk["kernel_tau"], f"{ctx}.hawkes.kernel_tau", errors, 0)
            if "kernel_tau" in hk
            else 900_000_000
        )
        if tau_us <= 0:
            errors.append(
                f"{ctx}.hawkes.kernel_tau: must be a positive duration"
            )
            tau_us = 900_000_000
        cfg = ArrivalProcessConfig(
            process="hawkes",
            seasonality=seasonality,
            hawkes_branching=float(branching),
            hawkes_kernel_tau_s=tau_us / S,
        )
    elif process == "mmpp2":
        mp = m.get("mmpp2") or {}
        if not isinstance(mp, Mapping):
            errors.append(f"{ctx}.mmpp2: expected a mapping, got {mp!r}")
            mp = {}
        unknown_m = sorted(set(mp) - {"rate_ratio", "burst_frac", "switch_tau"})
        if unknown_m:
            errors.append(
                f"{ctx}.mmpp2: unknown key(s): {', '.join(unknown_m)}"
                f" (known: burst_frac, rate_ratio, switch_tau)"
            )
        ratio = mp.get("rate_ratio", 4.0)
        if not _is_number(ratio) or float(ratio) <= 1.0:
            errors.append(
                f"{ctx}.mmpp2.rate_ratio: must be > 1"
                f" (lambda_burst/lambda_quiet), got {ratio!r}"
            )
            ratio = 4.0
        frac = mp.get("burst_frac", 0.25)
        if not _is_number(frac) or not 0.0 < float(frac) < 1.0:
            errors.append(
                f"{ctx}.mmpp2.burst_frac: must be in (0, 1), got {frac!r}"
            )
            frac = 0.25
        tau_us = (
            _duration_us(mp["switch_tau"], f"{ctx}.mmpp2.switch_tau", errors, 0)
            if "switch_tau" in mp
            else 2 * 24 * 3600 * S
        )
        if tau_us <= 0:
            errors.append(f"{ctx}.mmpp2.switch_tau: must be a positive duration")
            tau_us = 2 * 24 * 3600 * S
        cfg = ArrivalProcessConfig(
            process="mmpp2",
            seasonality=seasonality,
            mmpp2_rate_ratio=float(ratio),
            mmpp2_burst_frac=float(frac),
            mmpp2_switch_tau_s=tau_us / S,
        )
    return cfg, None, rate_per_hour


def _parse_workload_class(
    name: str,
    spec: Mapping[str, Any],
    default_tenants: int,
    errors: list[str],
    default_zipf_s: float = TENANT_ZIPF_S_DEFAULT,
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

    arrival: DistSpec | None = None
    arrival_process: ArrivalProcessConfig | None = None
    mapping_rate = 0.0
    if "arrival" in spec:
        if isinstance(spec["arrival"], Mapping):
            arrival_process, arrival, mapping_rate = _parse_arrival_mapping(
                spec["arrival"], f"{ctx}.arrival", errors
            )
        else:
            try:
                arrival = parse_dist(spec["arrival"])
            except ValueError as exc:
                errors.append(f"{ctx}.arrival: {exc}")
                arrival = _INVALID_DIST

    rate_keys = [
        k for k in ("rate_per_hour", "rate_per_day", "rate_per_week") if k in spec
    ]
    rate_per_hour = 0.0
    if arrival_process is not None:
        rate_per_hour = mapping_rate
        if rate_keys:
            errors.append(
                f"{ctx}: with an explicit arrival block, rate keys go"
                f" INSIDE the block (got top-level {', '.join(rate_keys)})"
            )
        if spec.get("diurnal"):
            errors.append(
                f"{ctx}.diurnal: has no effect with an explicit arrival"
                f" block (seasonality is configured there)"
            )
    elif arrival is not None:
        if rate_keys:
            errors.append(
                f"{ctx}: 'arrival' (closed-loop backlog) cannot be combined"
                f" with rate keys ({', '.join(rate_keys)}) — pick one"
            )
    elif len(rate_keys) != 1:
        errors.append(
            f"{ctx}: exactly one of rate_per_hour | rate_per_day |"
            f" rate_per_week | arrival is required (got {rate_keys or 'none'})"
        )
    else:
        k = rate_keys[0]
        try:
            raw = float(spec[k])
        except (TypeError, ValueError):
            errors.append(f"{ctx}.{k}: expected a number, got {spec[k]!r}")
            raw = 0.0
        rate_per_hour = raw / _RATE_DIVISOR[k]
        if raw <= 0:
            errors.append(f"{ctx}.{k}: arrival rate must be positive, got {raw}")
        # v0.1 sugar normalization: top-level rate + diurnal maps to the
        # nhpp process on the pinned v0.1 step curve (byte-identical
        # arrivals) or plain poisson (traffic-math §2.1).
        arrival_process = (
            ArrivalProcessConfig(
                process="nhpp", seasonality=SeasonalityConfig(preset="v01_steps")
            )
            if spec.get("diurnal")
            else ArrivalProcessConfig(process="poisson")
        )

    chips = _parse_dist_field(spec, "chips", ctx, errors)
    duration = _parse_dist_field(spec, "duration", ctx, errors)

    tier_raw = spec.get("tier")
    if tier_raw is None:
        if arrival is not None:
            tier = Tier.BEST_EFFORT  # backlog-class default (v0.2)
        else:
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
    # Backlog (closed-loop) classes default to 0 regardless of job class —
    # standing best-effort work must be freely reclaimable.
    min_runtime_s = (
        0.0 if arrival is not None else _MIN_RUNTIME_DEFAULT_S.get(job_class, 0.0)
    )
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
    # Splice tail cap <-> lifetime coupling (traffic-math §2.3): the
    # Pareto tail is truncated at its physical cap; a max_lifetime BELOW
    # the cap would silently re-truncate, so it is an error, and an
    # omitted max_lifetime inherits the cap.
    if duration.kind == "splice" and "cap" in duration.params:
        cap_s = duration.params["cap"] / S
        if max_lifetime_s is None:
            max_lifetime_s = cap_s
        elif cap_s > max_lifetime_s:
            errors.append(
                f"{ctx}.duration.tail.cap: exceeds max_lifetime"
                f" ({cap_s} s > {max_lifetime_s} s) — the tail cap must"
                f" not outlive the job's lifetime cap"
            )

    within_raw = spec.get("within")
    within: Constraint | None = None
    if within_raw is not None:
        if isinstance(within_raw, str):
            within = Constraint(level=within_raw)
        elif isinstance(within_raw, Mapping) and "level" in within_raw:
            unknown_w = sorted(
                set(within_raw) - {"level", "required", "relax_after", "relax_after_s"}
            )
            if unknown_w:
                errors.append(
                    f"{ctx}.within: unknown key(s): {', '.join(unknown_w)}"
                    f" (known: level, required, relax_after, relax_after_s)"
                )
            if "relax_after" in within_raw and "relax_after_s" in within_raw:
                errors.append(
                    f"{ctx}.within: give relax_after (a duration) OR"
                    f" relax_after_s (float seconds), not both"
                )
            if "relax_after" in within_raw:
                relax_s = (
                    _duration_us(
                        within_raw["relax_after"],
                        f"{ctx}.within.relax_after",
                        errors,
                        300 * S,
                    )
                    / S
                )
            else:
                relax_s = float(within_raw.get("relax_after_s", 300.0))
            required_raw = within_raw.get("required", True)
            if not isinstance(required_raw, bool):
                # Truthiness would turn e.g. the quoted string "false"
                # (or any typo) into a HARD constraint, silently killing
                # the relax/penalty pair the user configured.
                errors.append(
                    f"{ctx}.within.required: expected true/false,"
                    f" got {required_raw!r}"
                )
                required_raw = True
            within = Constraint(
                level=str(within_raw["level"]),
                required=required_raw,
                relax_after_s=relax_s,
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

    segment_nodes: int | None = None
    if "segment_nodes" in spec:
        raw_sn = spec["segment_nodes"]
        if isinstance(raw_sn, bool) or not isinstance(raw_sn, int):
            errors.append(
                f"{ctx}.segment_nodes: expected an integer, got {raw_sn!r}"
            )
        else:
            segment_nodes = raw_sn
    segment_level_raw = spec.get("segment_level")
    segment_level = str(segment_level_raw) if segment_level_raw is not None else None

    abort_default = (
        0.0 if job_class is JobClass.INFER_REPLICA else _ABORT_PROB_DEFAULT
    )
    zipf_raw = spec.get("tenant_zipf_s", default_zipf_s)
    if not _is_number(zipf_raw) or float(zipf_raw) < 0:
        errors.append(
            f"{ctx}.tenant_zipf_s: must be a number >= 0, got {zipf_raw!r}"
        )
        zipf_raw = default_zipf_s
    chip_type_raw = spec.get("chip_type")
    tenant_raw = spec.get("tenant")
    if tenant_raw is not None and (
        not isinstance(tenant_raw, str) or not tenant_raw.strip()
    ):
        # Empty/whitespace names (e.g. an unset templating variable)
        # would silently key jobs, quota, and reservations on "".
        errors.append(
            f"{ctx}.tenant: expected a non-empty tenant name string,"
            f" got {tenant_raw!r}"
        )
        tenant_raw = None
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
        arrival=arrival,
        arrival_process=arrival_process,
        segment_nodes=segment_nodes,
        segment_level=segment_level,
        abort_prob=float(spec.get("abort_prob", abort_default)),
        n_tenants=int(spec.get("n_tenants", default_tenants)),
        tenant_zipf_s=float(zipf_raw),
        tenant=tenant_raw,
        chip_type=str(chip_type_raw) if chip_type_raw is not None else None,
        capacity=capacity,
        n_gangs=int(spec.get("gangs", 1)),
        shape=shape,
        twisted=bool(spec.get("twisted", False)),
    )


#: Known workload preset names (expanded by :func:`_expand_workload_preset`).
WORKLOAD_PRESETS = ("google_fleet",)

#: google_fleet rate anchors, PER 1024 FLEET CHIPS (tuned to the
#: examples/01_minimal utilization target of rho ~ 0.9 at 2,048 chips).
_GF_EVAL_PER_HOUR_PER_1K = 20.0
_GF_FINETUNE_PER_DAY_PER_1K = 45.0
_GF_PRETRAIN_PER_WEEK_PER_1K = 1.0


def _google_fleet_classes(
    scale: int, fleet: FleetConfig
) -> dict[str, dict[str, Any]]:
    """The ``google_fleet`` preset class specs (traffic-math §2).

    A stylized multi-tenant ML-fleet mix, scaled by ``scale`` (total
    fleet chips): Hawkes-burst eval floods on a diurnal baseline, MMPP-2
    crunch/normal finetunes under the same envelope, rare Poisson
    pretrains with the lognormal-body/Pareto-tail duration splice, and a
    standing best-effort backlog sized to the fleet.  Tier shares are
    Borg-2019-flavored: pretrain PROD, finetune+eval BATCH, filler
    BEST_EFFORT (band 0).

    Deterministic fleet-shape inspection (first cluster only): with >= 3
    declared levels, pretrains are SEGMENTED at the next-to-leaf level
    (one segment = one full such domain); with >= 4 levels they also get
    a hard ``within`` at ``levels[1]``.  The pretrain size pmf is
    truncated to entries <= max(256, min(scale // 4, placeable)) and
    renormalized, where ``placeable`` is the capacity the derived
    placement constraints actually allow: the chips of ONE ``levels[1]``
    domain when the hard ``within`` applies (an 8,192-chip draw can
    never place inside a 4,096-chip pod, even on an empty fleet), else
    the first cluster's total chips — so "huge" really stays placeable
    at the declared scale.
    """
    f = scale / 1024.0
    clusters = fleet.clusters()
    # Capacity ceiling implied by the derived placement constraints (the
    # `within`/segment logic below): pretrains are searched under the
    # first cluster's roots, and hard-`within` pins them inside one
    # levels[1] domain.  Computed BEFORE building the class specs so the
    # pmf truncation below can use it.
    placeable: int | None = None
    if clusters:
        cl0 = clusters[0]
        placeable = cl0.total_chips()
        if len(cl0.levels) >= 4:
            for group, _ in _walk_groups(cl0):
                if group.level == cl0.levels[1]:
                    per_domain = (
                        sum(c.total_chips() for c in group.children)
                        if group.children
                        else group.chips
                    )
                    if per_domain > 0:
                        placeable = per_domain
                    break
    # Pretrain sizes: truncate the doc pmf to the fleet's own scale AND
    # its per-domain placement ceiling.
    cap_chips = max(256, scale // 4)
    if placeable is not None:
        cap_chips = max(256, min(cap_chips, placeable))
    pre_pmf = {
        k: w for k, w in PMF_PRESETS["pretrain_v02"].items() if k <= cap_chips
    }
    if not pre_pmf:
        pre_pmf = {256: 1.0}

    pretrain: dict[str, Any] = {
        "class": "pretrain",
        "arrival": {
            "process": "poisson",
            "rate_per_week": _GF_PRETRAIN_PER_WEEK_PER_1K * f,
        },
        "chips": {"pmf": pre_pmf},
        "duration": {
            "body": "lognormal[median=12d, p90=30d]",
            "tail": {"alpha": 1.5, "splice": "p90", "cap": "54d"},
        },
        "tier": "prod",
        "checkpoint_interval": "1h",
        "min_runtime": "2h",
    }
    if clusters and len(clusters[0].levels) >= 3:
        cl = clusters[0]
        seg_level = cl.levels[-2]
        seg_nodes = 0
        for group, _ in _walk_groups(cl):
            if group.level == seg_level and group.children:
                seg_nodes = sum(c.total_nodes() for c in group.children)
                break
        if seg_nodes >= 1:
            pretrain["segment_nodes"] = seg_nodes
            pretrain["segment_level"] = seg_level
            if len(cl.levels) >= 4:
                pretrain["within"] = cl.levels[1]

    return {
        "pretrain": pretrain,
        "finetune": {
            "class": "finetune",
            "arrival": {
                "process": "mmpp2",
                "rate_per_day": _GF_FINETUNE_PER_DAY_PER_1K * f,
                "seasonality": "helios_v01",
                "mmpp2": {
                    "rate_ratio": 4,
                    "burst_frac": 0.25,
                    "switch_tau": "2d",
                },
            },
            "chips": {"pmf": "finetune_v02"},
            "duration": "lognormal[median=4h, p90=24h]",
            "tier": "batch",
        },
        "eval": {
            "class": "eval",
            "arrival": {
                "process": "hawkes",
                "rate_per_hour": _GF_EVAL_PER_HOUR_PER_1K * f,
                "seasonality": "helios_v01",
                "hawkes": {"branching": 0.4, "kernel_tau": "15m"},
            },
            "chips": {"pmf": "eval_v02"},
            "duration": "lognormal[median=2m, p99=1h]",
            "tier": "batch",
        },
        "best_effort": {
            "class": "finetune",
            "arrival": {
                "process": "closed_loop",
                "closed_loop": {"target_pending": max(4, scale // 512)},
            },
            "chips": {"pmf": {8: 0.5, 16: 0.3, 32: 0.2}},
            "duration": "lognormal[median=1h, p90=6h]",
            "checkpoint_interval": "0s",
        },
    }


def _expand_workload_preset(
    w: Mapping[str, Any], fleet: FleetConfig, errors: list[str]
) -> Mapping[str, Any]:
    """Expand ``workload: {preset: ..., scale: ..., classes: ...}`` into
    a plain classes mapping (pure, deterministic).  Per-class overrides
    shallow-merge over the preset's class spec (key by key); an override
    of ``null`` REMOVES the preset class; override-only names append in
    document order.  ``scale`` defaults to the fleet's total chips."""
    preset = str(w["preset"])
    if preset not in WORKLOAD_PRESETS:
        errors.append(
            f"workload.preset: unknown preset {preset!r}"
            f" (known: {', '.join(WORKLOAD_PRESETS)})"
        )
        return {}
    scale = w.get("scale")
    if scale is None:
        scale = sum(cl.total_chips() for cl in fleet.clusters())
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        errors.append(
            f"workload.scale: must be a positive integer chip count,"
            f" got {scale!r}"
        )
        return {}
    classes = _google_fleet_classes(scale, fleet)
    overrides = w.get("classes") or {}
    if not isinstance(overrides, Mapping):
        errors.append("workload.classes: expected a mapping of class name -> spec")
        overrides = {}
    for cname, cspec in overrides.items():
        cname = str(cname)
        if cspec is None:
            classes.pop(cname, None)
            continue
        if not isinstance(cspec, Mapping):
            errors.append(f"workload.classes.{cname}: expected a mapping")
            continue
        classes[cname] = {**classes.get(cname, {}), **cspec}
    return classes


def _parse_workload(
    doc: Mapping[str, Any], fleet: FleetConfig, errors: list[str]
) -> WorkloadConfig:
    w = doc.get("workload")
    if w is None:
        errors.append("workload: section is required")
        return WorkloadConfig()
    if not isinstance(w, Mapping):
        errors.append(f"workload: expected a mapping, got {w!r}")
        return WorkloadConfig()
    kind = str(w.get("kind", "synthetic"))
    n_tenants = int(w.get("n_tenants", 8))
    zipf_raw = w.get("tenant_zipf_s", TENANT_ZIPF_S_DEFAULT)
    if not _is_number(zipf_raw) or float(zipf_raw) < 0:
        errors.append(
            f"workload.tenant_zipf_s: must be a number >= 0, got {zipf_raw!r}"
        )
        zipf_raw = TENANT_ZIPF_S_DEFAULT
    tenant_zipf_s = float(zipf_raw)
    source = w.get("source")
    preset_name: str | None = None
    if "preset" in w:
        preset_name = str(w["preset"])
        if kind != "synthetic":
            errors.append(
                f"workload.preset: only valid with kind 'synthetic',"
                f" got {kind!r}"
            )
        raw_classes: Mapping[str, Any] = _expand_workload_preset(w, fleet, errors)
    else:
        if "scale" in w:
            errors.append("workload.scale: only valid together with 'preset'")
        raw_classes = w.get("classes") or {}
    classes: list[WorkloadClassConfig] = []
    if not isinstance(raw_classes, Mapping):
        errors.append("workload.classes: expected a mapping of class name -> spec")
        raw_classes = {}
    for cname, cspec in raw_classes.items():
        if not isinstance(cspec, Mapping):
            errors.append(f"workload.classes.{cname}: expected a mapping")
            continue
        classes.append(
            _parse_workload_class(
                str(cname), cspec, n_tenants, errors, tenant_zipf_s
            )
        )
    return WorkloadConfig(
        kind=kind,
        classes=classes,
        source=str(source) if source is not None else None,
        n_tenants=n_tenants,
        tenant_zipf_s=tenant_zipf_s,
        preset=preset_name,
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
    "penalties",
    "quota",
}


def _parse_penalties(
    doc: Mapping[str, Any], errors: list[str]
) -> PenaltiesConfig | None:
    """Parse the top-level ``penalties`` section (v0.4).  Structure:
    ``penalties: {xover: {<level>: <multiplier>}}``."""
    raw = doc.get("penalties")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        errors.append(f"penalties: expected a mapping, got {raw!r}")
        return None
    unknown = sorted(set(raw) - {"xover"})
    if unknown:
        errors.append(
            f"penalties: unknown key(s): {', '.join(unknown)} (known: xover)"
        )
    xover_raw = raw.get("xover")
    xover: dict[str, float] = {}
    if xover_raw is None:
        errors.append("penalties: 'xover' mapping is required")
    elif not isinstance(xover_raw, Mapping):
        errors.append(
            f"penalties.xover: expected a {{level: multiplier}} mapping,"
            f" got {xover_raw!r}"
        )
    else:
        for level, mult in xover_raw.items():
            if not _is_number(mult):
                errors.append(
                    f"penalties.xover.{level}: expected a number in (0, 1],"
                    f" got {mult!r}"
                )
                continue
            xover[str(level)] = float(mult)
    return PenaltiesConfig(xover=xover)


_QUOTA_OVER_MODES = ("best_effort", "reject")


def _parse_quota(doc: Mapping[str, Any], errors: list[str]) -> QuotaConfig | None:
    """Parse the top-level ``quota`` section (v0.4).  Structure:
    ``quota: {tenants: {<name>: <chips> | {chips: <chips>}},
    over_quota: best_effort | reject}``."""
    raw = doc.get("quota")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        errors.append(f"quota: expected a mapping, got {raw!r}")
        return None
    unknown = sorted(set(raw) - {"tenants", "over_quota"})
    if unknown:
        errors.append(
            f"quota: unknown key(s): {', '.join(unknown)}"
            f" (known: tenants, over_quota)"
        )
    tenants_raw = raw.get("tenants")
    tenants: dict[str, int] = {}
    if not isinstance(tenants_raw, Mapping) or not tenants_raw:
        errors.append(
            "quota.tenants: a non-empty {tenant: chips} mapping is required"
        )
    else:
        for name, spec in tenants_raw.items():
            ctx = f"quota.tenants.{name}"
            chips: Any = spec
            if isinstance(spec, Mapping):
                unknown_t = sorted(set(spec) - {"chips"})
                if unknown_t:
                    errors.append(
                        f"{ctx}: unknown key(s): {', '.join(unknown_t)}"
                        f" (known: chips)"
                    )
                chips = spec.get("chips")
            if isinstance(chips, bool) or not isinstance(chips, int) or chips < 1:
                errors.append(
                    f"{ctx}: chips must be an integer >= 1, got {chips!r}"
                )
                continue
            tenants[str(name)] = chips
    over = str(raw.get("over_quota", "best_effort"))
    if over not in _QUOTA_OVER_MODES:
        errors.append(
            f"quota.over_quota: expected one of"
            f" {', '.join(_QUOTA_OVER_MODES)}, got {over!r}"
        )
        over = "best_effort"
    return QuotaConfig(tenants=tenants, over_quota=over)


_RESERVATION_KEYS = frozenset(
    {"id", "tenant", "chips", "start", "end", "level", "chip_type", "hard_end"}
)


def _parse_reservations(
    doc: Mapping[str, Any], errors: list[str]
) -> list[ReservationConfig]:
    """Parse the top-level ``reservations`` list (v0.4 calendar blocks)."""
    raw = doc.get("reservations")
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"reservations: expected a list of mappings, got {raw!r}")
        return []
    out: list[ReservationConfig] = []
    for i, r in enumerate(raw):
        if not isinstance(r, Mapping):
            errors.append(f"reservations[{i}]: expected a mapping, got {r!r}")
            continue
        rid = str(r.get("id", f"reservation-{i}"))
        ctx = f"reservations.{rid}"
        unknown = sorted(set(r) - _RESERVATION_KEYS)
        if unknown:
            errors.append(
                f"{ctx}: unknown key(s): {', '.join(unknown)}"
                f" (known: {', '.join(sorted(_RESERVATION_KEYS))})"
            )
        tenant = r.get("tenant")
        if not isinstance(tenant, str) or not tenant:
            errors.append(f"{ctx}: 'tenant' (owner name) is required")
            tenant = ""
        chips = r.get("chips")
        if isinstance(chips, bool) or not isinstance(chips, int) or chips < 1:
            errors.append(f"{ctx}: chips must be an integer >= 1, got {chips!r}")
            chips = 1
        start_us = _duration_us(r.get("start"), f"{ctx}.start", errors, 0)
        end_us = _duration_us(r.get("end"), f"{ctx}.end", errors, 0)
        level_raw = r.get("level")
        ct_raw = r.get("chip_type")
        out.append(
            ReservationConfig(
                id=rid,
                tenant=str(tenant),
                chips=chips,
                start_us=start_us,
                end_us=end_us,
                level=str(level_raw) if level_raw is not None else None,
                chip_type=str(ct_raw) if ct_raw is not None else None,
                hard_end=bool(r.get("hard_end", True)),
            )
        )
    return out


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
    workload = _parse_workload(doc, fleet, errors)

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
    stints_raw = out_raw.get("stints")
    stints: str | bool | None
    if stints_raw is None or stints_raw is False:
        stints = None  # absent / explicit opt-out: no stints.parquet
    elif stints_raw is True or isinstance(stints_raw, str):
        stints = stints_raw
    else:
        errors.append(
            f"outputs.stints: expected a level name or true"
            f" (true = the level directly below each cluster root),"
            f" got {stints_raw!r}"
        )
        stints = None
    outputs = OutputsConfig(
        dir=out_raw.get("dir"),
        events=str(out_raw.get("events", "parquet")),
        plots=bool(out_raw.get("plots", False)),
        stints=stints,
        extra={
            k: v
            for k, v in out_raw.items()
            if k not in ("dir", "events", "plots", "stints")
        },
    )

    services = _parse_services(doc, errors)
    reservations = _parse_reservations(doc, errors)
    penalties = _parse_penalties(doc, errors)
    quota = _parse_quota(doc, errors)

    return Scenario(
        sim=sim,
        fleet=fleet,
        workload=workload,
        scheduler=scheduler,
        outputs=outputs,
        failure_model=failure_model,
        services=services,
        reservations=reservations,
        penalties=penalties,
        quota=quota,
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
    if spec.kind == "backlog":
        errors.append(
            f"{ctx}: 'backlog' is an arrival spec (closed-loop standing"
            f" backlog), not a value distribution — use it under 'arrival'"
        )
        return
    if spec.kind not in KNOWN_DIST_KINDS:
        errors.append(
            f"{ctx}: unknown distribution kind {spec.kind!r}"
            f" (known: {', '.join(sorted(KNOWN_DIST_KINDS))})"
        )
        return
    if spec.kind == "pmf":
        _validate_pmf_params(spec, ctx, errors)
        return
    if spec.kind == "splice":
        _validate_splice_params(spec, ctx, errors)
        return
    if spec.kind == "lognormal":
        _validate_lognormal_params(spec, ctx, errors)
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
    elif spec.kind == "pareto":
        alpha, xm = p["alpha"], p["xm"]
        if not (_is_number(alpha) and _is_number(xm)) or alpha <= 0 or xm <= 0:
            errors.append(
                f"{ctx}: pareto alpha and xm must be positive,"
                f" got alpha={alpha!r}, xm={xm!r}"
            )
    elif spec.kind == "weibull":
        shape_p, scale_p = p["shape"], p["scale"]
        if (
            not (_is_number(shape_p) and _is_number(scale_p))
            or shape_p <= 0
            or scale_p <= 0
        ):
            errors.append(
                f"{ctx}: weibull shape and scale must be positive,"
                f" got shape={shape_p!r}, scale={scale_p!r}"
            )
    elif spec.kind == "fixed":
        if not _is_number(p["value"]):
            errors.append(f"{ctx}: fixed value must be a number, got {p['value']!r}")


def _validate_lognormal_params(spec: DistSpec, ctx: str, errors: list[str]) -> None:
    """Lognormal takes ``median`` plus EXACTLY ONE of ``p90`` | ``p99``."""
    p = spec.params
    extra = sorted(set(p) - {"median", "p90", "p99"})
    if extra:
        errors.append(
            f"{ctx}: unexpected parameter(s) for lognormal: {', '.join(extra)}"
        )
        return
    if "median" not in p:
        errors.append(f"{ctx}: lognormal requires parameter(s) median")
        return
    has90, has99 = "p90" in p, "p99" in p
    if has90 and has99:
        errors.append(
            f"{ctx}: lognormal takes p90 OR p99, not both (exactly one)"
        )
        return
    if not (has90 or has99):
        errors.append(f"{ctx}: lognormal requires parameter(s) p90 (or p99)")
        return
    qname = "p90" if has90 else "p99"
    median, q = p["median"], p[qname]
    if not (_is_number(median) and _is_number(q)) or median <= 0 or q <= 0:
        errors.append(
            f"{ctx}: lognormal median and {qname} must be positive,"
            f" got median={median!r}, {qname}={q!r}"
        )
    elif q < median:
        errors.append(
            f"{ctx}: lognormal requires {qname} >= median,"
            f" got median={median}, {qname}={q}"
        )


def _validate_pmf_params(spec: DistSpec, ctx: str, errors: list[str]) -> None:
    """pmf params are ``{str(chips): weight}``: keys must be powers of
    two (DESIGN 4.1 sizes cluster at powers of two); weights positive."""
    if not spec.params:
        errors.append(f"{ctx}: pmf requires at least one entry")
        return
    for k, w in spec.params.items():
        try:
            chips = int(k)
        except (TypeError, ValueError):
            errors.append(f"{ctx}: pmf key {k!r} is not an integer chip count")
            continue
        if not _is_pow2(chips):
            errors.append(
                f"{ctx}: pmf chip counts must be powers of two, got {chips}"
            )
        if not _is_number(w) or w <= 0:
            errors.append(
                f"{ctx}: pmf weight for {chips} must be positive, got {w!r}"
            )


def _validate_splice_params(spec: DistSpec, ctx: str, errors: list[str]) -> None:
    """Splice body/tail parameter checks (traffic-math §2.3): lognormal
    body params, ``alpha > 1``, and splice point strictly below cap."""
    p = spec.params
    body = DistSpec(
        "lognormal", {k: p[k] for k in ("median", "p90", "p99") if k in p}
    )
    _validate_lognormal_params(body, f"{ctx}.body", errors)
    alpha = p.get("alpha")
    if not _is_number(alpha) or alpha <= 1.0:
        errors.append(
            f"{ctx}.tail.alpha: must be > 1 (alpha <= 1 has infinite"
            f" mean even truncated in practice), got {alpha!r}"
        )
        return
    cap = p.get("cap")
    if not _is_number(cap) or cap <= 0:
        errors.append(f"{ctx}.tail.cap: must be a positive duration, got {cap!r}")
        return
    # Locate the splice point to check it sits below the cap.
    median = p.get("median")
    if "splice_at" in p:
        theta = p["splice_at"]
        if not _is_number(theta) or theta <= 0:
            errors.append(
                f"{ctx}.tail.splice: must be a positive duration, got {theta!r}"
            )
            return
    elif _is_number(median) and median > 0:
        q = p.get("splice_q")
        z = _Z_BY_QUANTILE.get(q)
        if z is None:
            errors.append(
                f"{ctx}.tail.splice: must be 'p90', 'p99', or a duration,"
                f" got splice_q={q!r}"
            )
            return
        qv = p.get("p90") if "p90" in p else p.get("p99")
        zq = _Z_BY_QUANTILE[0.90] if "p90" in p else _Z_BY_QUANTILE[0.99]
        if not _is_number(qv) or qv <= 0:
            return  # body errors already recorded
        sigma = (math.log(qv) - math.log(median)) / zq
        theta = median * math.exp(sigma * z)
    else:
        return  # body errors already recorded
    if theta >= cap:
        errors.append(
            f"{ctx}.tail: splice point ({theta:.6g} us) must sit strictly"
            f" below the cap ({cap} us)"
        )


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
        if c.chips.kind == "splice":
            errors.append(
                f"{ctx}.chips: 'splice' is a duration distribution,"
                f" not a chip-count one"
            )
        if c.duration.kind == "pmf":
            errors.append(
                f"{ctx}.duration: 'pmf' is a chip-count distribution,"
                f" not a duration one"
            )
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
                # Relaxable (preferred) constraints are v0.4 semantics.
                if c.within.relax_after_s < 0:
                    errors.append(
                        f"{ctx}.within: relax_after must be >= 0,"
                        f" got {c.within.relax_after_s}"
                    )
                if c.segment_nodes is not None:
                    errors.append(
                        f"{ctx}.within: a relaxable (required: false) OUTER"
                        f" constraint on a segmented gang is {_NOT_V04}"
                        f" (segments already span domains; relax the shape"
                        f" by choosing a higher within level instead)"
                    )
        # --- closed-loop backlog arrival (v0.2) ---
        if c.arrival is not None and c.arrival.kind != "invalid":
            if c.arrival.kind != "backlog":
                errors.append(
                    f"{ctx}.arrival: unknown arrival kind {c.arrival.kind!r}"
                    f" (known: backlog)"
                )
            else:
                extra = sorted(set(c.arrival.params) - {"target_pending"})
                if extra:
                    errors.append(
                        f"{ctx}.arrival: unexpected parameter(s) for backlog:"
                        f" {', '.join(extra)}"
                    )
                tp = c.arrival.params.get("target_pending")
                if (
                    tp is None
                    or isinstance(tp, bool)
                    or not isinstance(tp, int)
                    or tp < 1
                ):
                    errors.append(
                        f"{ctx}.arrival: backlog requires an integer"
                        f" target_pending >= 1, got {tp!r}"
                    )
                if c.diurnal:
                    errors.append(
                        f"{ctx}.diurnal: has no effect with a backlog arrival"
                        f" (closed-loop classes have no arrival process)"
                    )
        # --- segmented gangs (v0.2) ---
        if (c.segment_nodes is None) != (c.segment_level is None):
            errors.append(
                f"{ctx}: segment_nodes and segment_level must be given"
                f" together (got only one)"
            )
        elif c.segment_nodes is not None:
            if c.segment_nodes < 1:
                errors.append(
                    f"{ctx}.segment_nodes: must be >= 1, got {c.segment_nodes}"
                )
            if c.segment_level not in level_vocab:
                errors.append(
                    f"{ctx}.segment_level: unknown level {c.segment_level!r}"
                    f" (known: {', '.join(sorted(level_vocab)) or 'none'})"
                )
            elif c.within is not None and c.within.level in level_vocab:
                # within is the OUTER constraint: it must sit strictly
                # above segment_level in every cluster declaring both.
                n_both = 0
                for cl in clusters:
                    if (
                        c.segment_level in cl.levels
                        and c.within.level in cl.levels
                    ):
                        n_both += 1
                        if cl.levels.index(c.within.level) >= cl.levels.index(
                            c.segment_level
                        ):
                            errors.append(
                                f"{ctx}: within level {c.within.level!r} must"
                                f" be strictly above segment_level"
                                f" {c.segment_level!r} in cluster {cl.id!r}"
                                f" (levels {cl.levels})"
                            )
                if n_both == 0:
                    errors.append(
                        f"{ctx}: no cluster declares both within level"
                        f" {c.within.level!r} and segment_level"
                        f" {c.segment_level!r}"
                    )
        if not 0.0 <= c.abort_prob <= 1.0:
            errors.append(
                f"{ctx}.abort_prob: must be in [0, 1], got {c.abort_prob}"
            )
        if c.n_tenants < 1:
            errors.append(f"{ctx}.n_tenants: must be >= 1, got {c.n_tenants}")
        if c.capacity not in (CapacityClass.ON_DEMAND, CapacityClass.SPOT):
            # SPOT is v0.4 (zero-notice kill + checkpoint restart);
            # reserved/flex_start/calendar job capacity classes remain
            # unimplemented (CALENDAR capacity is expressed through the
            # top-level `reservations` section instead).
            errors.append(
                f"{ctx}.capacity: capacity class"
                f" {c.capacity.name.lower()!r} is {_NOT_V04}"
                f" (only on_demand and spot)"
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
    if isinstance(scenario.outputs.stints, str):
        # Every leaf must have an ancestor at the configured level, so the
        # level must be declared by EVERY cluster (True — "directly below
        # each cluster root" — always resolves and needs no check).
        for cl in clusters:
            if scenario.outputs.stints not in cl.levels:
                errors.append(
                    f"outputs.stints: level {scenario.outputs.stints!r} is"
                    f" not declared by cluster {cl.id!r}"
                    f" (its levels: {', '.join(cl.levels) or 'none'})"
                )

    # --- penalties (v0.4) ---
    if scenario.penalties is not None:
        for level, mult in scenario.penalties.xover.items():
            pctx = f"penalties.xover.{level}"
            if level not in level_vocab:
                errors.append(
                    f"{pctx}: unknown level {level!r}"
                    f" (known: {', '.join(sorted(level_vocab)) or 'none'})"
                )
            if not 0.0 < mult <= 1.0:
                errors.append(
                    f"{pctx}: speed multiplier must be in (0, 1],"
                    f" got {mult}"
                )

    # --- quota (v0.4): every capped tenant must be producible ---
    # Zipf marking yields t0..t{n_tenants-1} per class; fixed `tenant:`
    # pins and service tenants are declared explicitly.  A name no
    # configured source can ever produce means the cap is DEAD CONFIG
    # (unlisted tenants are unlimited) — error, never a silent no-op
    # (DESIGN principle 5).  Trace workloads are exempt: their tenant
    # space comes from the trace file, unknown at validate time.
    if scenario.quota is not None and w.kind == "synthetic":
        fixed_tenants = {c.tenant for c in w.classes if c.tenant is not None}
        fixed_tenants.update(svc.tenant for svc in scenario.services)
        max_zipf = max(
            (c.n_tenants for c in w.classes if c.tenant is None), default=0
        )
        for name in scenario.quota.tenants:
            if name in fixed_tenants:
                continue
            digits = name[1:]
            if (
                name.startswith("t")
                and digits.isdigit()
                and (digits == "0" or not digits.startswith("0"))
                and int(digits) < max_zipf
            ):
                continue  # reachable via Zipf marking
            reachable = sorted(fixed_tenants)
            if max_zipf > 0:
                reachable.append(f"t0..t{max_zipf - 1}")
            errors.append(
                f"quota.tenants.{name}: no configured workload class or"
                f" service can ever produce tenant {name!r}"
                f" (reachable: {', '.join(reachable) or 'none'}) — the cap"
                f" would be silently dead config"
            )

    # --- reservations (v0.4 calendar blocks) ---
    if scenario.reservations:
        # Largest single-domain chip capacity per level (static topology).
        level_cap: dict[str, int] = {}
        for cl in clusters:
            if cl.levels:
                cap0 = cl.total_chips()
                level_cap[cl.levels[0]] = max(level_cap.get(cl.levels[0], 0), cap0)
            for group, _ in _walk_groups(cl):
                per_instance = (
                    sum(ch.total_chips() for ch in group.children)
                    if group.children
                    else group.chips
                )
                level_cap[group.level] = max(
                    level_cap.get(group.level, 0), per_instance
                )
        root_cap = max((cl.total_chips() for cl in clusters), default=0)
        seen_res: set[str] = set()
        fleet_leaf_types_r = _referenced_chip_types(scenario.fleet)
        for res in scenario.reservations:
            rctx = f"reservations.{res.id}"
            if res.id in seen_res:
                errors.append(f"{rctx}: duplicate reservation id")
            seen_res.add(res.id)
            if res.start_us >= res.end_us:
                errors.append(
                    f"{rctx}: start must be strictly before end"
                    f" (got {res.start_us} us >= {res.end_us} us)"
                )
            if res.level is not None and res.level not in level_vocab:
                errors.append(
                    f"{rctx}.level: unknown level {res.level!r}"
                    f" (known: {', '.join(sorted(level_vocab)) or 'none'})"
                )
            if res.chip_type is not None:
                if res.chip_type not in scenario.fleet.chip_types:
                    errors.append(
                        f"{rctx}.chip_type: unknown chip_type {res.chip_type!r}"
                        f" (declared:"
                        f" {', '.join(scenario.fleet.chip_types) or 'none'})"
                    )
            elif len(fleet_leaf_types_r) > 1:
                errors.append(
                    f"{rctx}: chip_type is required — the fleet has multiple"
                    f" chip types ({', '.join(fleet_leaf_types_r)})"
                )
            cap = (
                level_cap.get(res.level, 0)
                if res.level is not None
                else root_cap
            )
            if cap and res.chips > cap:
                errors.append(
                    f"{rctx}: {res.chips} chips can never fit inside one"
                    f" {res.level or 'cluster'} domain ({cap} chips)"
                )

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
