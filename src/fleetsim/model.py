"""Core data model: chips, domain tree, allocations, jobs, services.

This module is DESIGN.md sections 3-5 rendered as code.  It is purely
declarative — engine-side runtime bookkeeping (queues, event handles,
checkpoint clocks) lives in the engine, not here.  The only mutable
runtime-ish fields kept on :class:`Job` are the ones DESIGN puts there:
``status``, ``attained_service_chip_s``, ``goodput_chip_s``.

UNITS
-----
All time fields are **int microseconds** unless the field name ends in
``_s``, in which case they are **float seconds** (e.g.
``checkpoint_interval_s``).  ``submit_t`` and ``valid_until`` on
:class:`Job` are int microseconds since sim epoch.

INVARIANTS
----------
- A gang is atomic: :class:`Allocation` either exists in full or not at
  all; partial allocation of a job is unrepresentable.
- :class:`Domain` leaves (``children == []``) carry ``chips``/``state``;
  interior domains carry only the derived counters
  ``total_chips``/``free_chips``.
- ``Tier`` comparisons are band rules: a higher band may preempt a lower
  band; never preempt within PROD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any

__all__ = [
    "ChipType",
    "NodeState",
    "Domain",
    "Constraint",
    "GangSpec",
    "GangAlloc",
    "Allocation",
    "PreemptMode",
    "JobClass",
    "Tier",
    "CapacityClass",
    "JobStatus",
    "Job",
    "Service",
]


# ---------------------------------------------------------------------------
# 3.1 Chip types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChipType:
    """A chip model in the flat registry (e.g. ``h100``, ``tpu_v5p``).

    Jobs constrain on ``name``; the v0.4 performance model keys its
    throughput matrix on ``(job_profile, chip_type)``.  ``generation``
    exists for affinity/compat rules; defaults to 1 when a config omits it.
    """

    name: str
    vendor: str
    hbm_gib: float
    peak_tflops_bf16: float
    generation: int = 1


# ---------------------------------------------------------------------------
# 3.2 The domain tree
# ---------------------------------------------------------------------------


class NodeState(Enum):
    """Health state of a leaf domain (node). Leaves only."""

    HEALTHY = auto()
    DRAINING = auto()
    FAILED = auto()
    MAINTENANCE = auto()


@dataclass(slots=True)
class Domain:
    """One domain in the capacity tree, metro down to node.

    INVARIANTS: leaves (``children == []``) have ``chips > 0`` and a
    ``chip_type``; ``total_chips``/``free_chips`` are derived counters
    maintained incrementally by the fleet layer on alloc/free — never
    recomputed by tree walks in the hot path.  ``level`` comes from the
    cluster's config-declared level list, never from code.
    """

    id: str
    level: str
    parent: str | None
    children: list[str]
    chip_type: str | None
    chips: int = 0
    state: NodeState = NodeState.HEALTHY
    lemon_factor: float = 1.0
    total_chips: int = 0
    free_chips: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4.2 Constraints and allocations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Constraint:
    """Placement constraint: the gang must fit within one ``level`` domain.

    v0.1 supports only ``required=True`` (hard constraints).  Preferred
    (relaxable) constraints and ``relax_after_s`` (float seconds; Slurm
    max_switch_wait default) arrive in v0.3 — the fields exist so configs
    carry them without migration.
    """

    level: str
    required: bool = True
    relax_after_s: float = 300.0


@dataclass(slots=True)
class GangSpec:
    """Declarative request for one gang (all-or-nothing chip set).

    ``chips`` is either sub-node (fits one leaf) or a whole-node multiple.
    ``chip_type`` must be pinned in v1.  ``segments`` (Slurm block) and
    ``shape``/``twisted`` (TPU slice) are schema-carried, v0.3 semantics.
    """

    chips: int
    chip_type: str | None = None
    within: Constraint | None = None
    segments: tuple[int, str] | None = None
    shape: tuple[int, int, int] | None = None
    twisted: bool = False


@dataclass(slots=True)
class GangAlloc:
    """A realized gang placement.

    ``nodes`` is a list of leaf domain ids for whole-node gangs, or a
    ``{leaf_id: chips}`` dict for a sub-node gang sharing one leaf.
    ``anchor`` is the LCA domain that satisfied the ``within`` constraint
    (placement-quality score and audit trail).
    """

    nodes: list[str] | dict[str, int]
    anchor: str
    relaxed: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Allocation:
    """Atomic allocation for a job: all gangs placed, or the object never
    exists.  ``len(gangs) > 1`` is Multislice-style gang-of-gangs (v0.3);
    one gang's node failing kills every gang in the allocation."""

    job_id: str
    gangs: list[GangAlloc]


class PreemptMode(Enum):
    """How a preemption ends: CANCEL kills the job; REQUEUE re-enters the
    queue at original priority.  SUSPEND is deliberately absent (undefined
    on GPUs — held memory)."""

    CANCEL = auto()
    REQUEUE = auto()


# ---------------------------------------------------------------------------
# 5. Job and workload model
# ---------------------------------------------------------------------------


class JobClass(Enum):
    """Workload class of a job (drives generator defaults and reporting)."""

    PRETRAIN = auto()
    FINETUNE = auto()
    EVAL = auto()
    INFER_REPLICA = auto()


class Tier(IntEnum):
    """Borg priority bands.  Band rules, not raw integers: a higher band
    preempts a lower band; no preemption within PROD."""

    FREE = 0
    BATCH = 1
    PROD = 2
    MONITORING = 3


class CapacityClass(Enum):
    """Cloud capacity semantics.  v1 implements ON_DEMAND only; the others
    are schema-carried for v0.2+ (fleetsim validate rejects them)."""

    RESERVED = auto()
    ON_DEMAND = auto()
    SPOT = auto()
    FLEX_START = auto()
    CALENDAR = auto()


class JobStatus(Enum):
    """Queue states plus Helios terminal states."""

    PENDING = auto()
    ADMITTED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()
    TIMEOUT = auto()
    NODE_FAIL = auto()
    PREEMPTED = auto()


@dataclass(slots=True)
class Job:
    """One schedulable unit — training, eval, or an inference replica.

    UNITS: ``submit_t`` and ``valid_until`` are int microseconds since sim
    epoch; every ``*_s`` field is float seconds.  ``true_duration_s`` is
    hidden from schedulers (they may see ``walltime_est_s``).

    INVARIANTS: declarative except ``status``, ``attained_service_chip_s``
    and ``goodput_chip_s``, which the engine mutates.  v1 generators emit
    ``len(gangs) == 1``.  A job preempted or failed before its first
    checkpoint loses all progress.
    """

    id: str
    tenant: str
    job_class: JobClass
    submit_t: int
    gangs: list[GangSpec]
    tier: Tier
    capacity: CapacityClass = CapacityClass.ON_DEMAND
    preemptible: bool = True
    min_runtime_s: float = 0.0
    max_lifetime_s: float | None = None
    walltime_est_s: float | None = None
    true_duration_s: float = 0.0
    checkpoint_interval_s: float = 3600.0
    checkpoint_save_s: float = 60.0
    restart_overhead_s: float = 900.0
    valid_until: int | None = None
    service_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    attained_service_chip_s: float = 0.0
    goodput_chip_s: float = 0.0


@dataclass(slots=True)
class Service:
    """An inference service: emits/cancels INFER_REPLICA jobs above the
    scheduler.  v1 freezes replica counts (``min_replicas == max_replicas``);
    ``load`` becomes a DiurnalCurve (target QPS -> desired replicas) in
    v0.2 and is untyped until then."""

    id: str
    tenant: str
    replica_spec: GangSpec
    min_replicas: int
    max_replicas: int
    load: Any = None
    tier: Tier = Tier.PROD
