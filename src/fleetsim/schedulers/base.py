"""Scheduler API (DESIGN §7): immutable views, actions, the Scheduler ABC,
placement-policy protocol, and the scheduler registry.

The engine mutates state; policies emit intents.  A :class:`Scheduler` is
invoked with an immutable :class:`ClusterView` (coalesced wake, DESIGN 6.1)
and returns a list of :class:`Action`\\ s that the engine validates and
applies — illegal intents raise in the engine's strict mode.

Ordering and placement are separate axes (Blox's decomposition): a
Scheduler decides *who next*; a :class:`PlacementPolicy` decides *where*.
``view.find_placement(job, policy)`` runs the policy inside the engine's
indexes so ordering code never touches node internals.

RESERVATION SEMANTICS (pinned): within one wake, every placement returned
by ``view.find_placement`` is **tentatively reserved** — later
``find_placement`` calls in the same wake see the remaining capacity, so a
loop of find-then-Place (the FIFO pattern) can never double-book.
Reservations for jobs the scheduler does NOT ultimately ``Place`` are
rolled back when ``schedule()`` returns.  ``view.search_first_fit`` is the
raw, reservation-free search primitive placement policies build on.

UNITS: ``now``/``submit_time`` int microseconds; ``*_s`` float seconds.

INVARIANTS: views are frozen snapshots computed fresh per wake;
``pending()`` is sorted by ``(submit_time, id)`` — requeued jobs keep
their ORIGINAL ``submit_time`` (Borg requeue ordering), so they re-enter
ahead of later arrivals.  ``running()`` is sorted the same way.  Jobs in a
preemption grace window appear in neither list.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..fleet.tree import Placement
from ..model import GangSpec, JobClass, PreemptMode, Tier
from ..units import S

__all__ = [
    "JobView",
    "DomainView",
    "ClusterView",
    "PlacementPolicy",
    "Place",
    "Preempt",
    "Action",
    "Scheduler",
    "register",
    "get_scheduler",
    "registered_schedulers",
]

#: Entry-points group consulted by :func:`get_scheduler` for out-of-tree
#: scheduler plugins (DESIGN §13).
ENTRY_POINT_GROUP = "fleetsim.schedulers"


# ---------------------------------------------------------------------------
# Immutable views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobView:
    """Scheduler-visible snapshot of one job.

    ``true_duration_s`` is deliberately absent (hidden from schedulers);
    ``walltime_est_s`` is the visible estimate.  ``attained_service_chip_s``
    is checkpointed progress plus the current stint's work, x chips
    (LAS/Tiresias input).  ``checkpoint_age_s`` is the work-seconds of
    progress that would be LOST if the job were killed right now (0 for
    queued jobs; everything since the last checkpoint boundary, or since
    start when checkpointing is disabled).  ``within`` is the hard
    containment level (``GangSpec.within.level``) or ``None``.
    """

    id: str
    submit_time: int
    chips: int
    chip_type: str | None
    tier: Tier
    job_class: JobClass
    preemptible: bool
    min_runtime_s: float
    attained_service_chip_s: float
    checkpoint_age_s: float
    walltime_est_s: float | None
    within: str | None
    tenant: str


@dataclass(frozen=True, slots=True)
class DomainView:
    """Scheduler-visible snapshot of one domain's capacity counters.

    ``free_chips`` counts unallocated chips on HEALTHY leaves only;
    ``healthy_chips`` is HEALTHY capacity allocated or not; ``total_chips``
    includes failed/drained capacity (the allocation-rate denominator).
    """

    id: str
    level: str
    chip_type: str | None
    total_chips: int
    free_chips: int
    healthy_chips: int


@runtime_checkable
class PlacementPolicy(Protocol):
    """The *where* axis: find a concrete placement for one job, or None."""

    def place(self, job: JobView, view: "ClusterView") -> Placement | None: ...


@runtime_checkable
class ClusterView(Protocol):
    """Immutable scheduler-facing snapshot of queue + fleet state.

    ``now`` is the wake time (int microseconds).  See the module docstring
    for the tentative-reservation semantics of ``find_placement``.
    """

    @property
    def now(self) -> int: ...

    def pending(self) -> Sequence[JobView]:
        """Queued, admission-passed jobs, sorted ``(submit_time, id)``."""
        ...

    def running(self) -> Sequence[JobView]:
        """Running jobs (attained service and checkpoint age included),
        sorted ``(submit_time, id)``."""
        ...

    def free_capacity(self, domain_id: str) -> int:
        """Free healthy chips under one domain."""
        ...

    def domains(self, level: str) -> Sequence[DomainView]:
        """All domains at ``level``, ascending id order."""
        ...

    def find_placement(
        self, job: JobView, policy: PlacementPolicy
    ) -> Placement | None:
        """Run ``policy`` for ``job`` and tentatively reserve the result
        (rolled back after ``schedule()`` unless a matching ``Place`` is
        emitted).  Returns the cached placement on repeat calls."""
        ...

    def search_first_fit(self, spec: GangSpec) -> Placement | None:
        """Raw first-fit capacity search (no reservation side effect) —
        the primitive placement policies compose."""
        ...

    def throughput(self, job: JobView, chip_type: str) -> float:
        """Gavel-matrix hook (v0.4).  Always 1.0 in v1."""
        ...


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Place:
    """Place ``job_id`` on a complete gang node-set.  Atomic."""

    job_id: str
    placement: Placement


@dataclass(frozen=True, slots=True)
class Preempt:
    """Preempt ``job_id`` (CANCEL | REQUEUE).  ``preemptor`` optionally
    names the job this preemption makes room for; when present the engine
    enforces Borg band rules (victim band must be strictly lower; never
    within PROD).  MONITORING victims are always refused."""

    job_id: str
    mode: PreemptMode
    preemptor: str | None = None


Action = Place | Preempt


# ---------------------------------------------------------------------------
# Scheduler ABC
# ---------------------------------------------------------------------------


class Scheduler(ABC):
    """Base class for ordering policies.

    ``wake_interval`` (int microseconds, or ``None``) requests periodic
    wakes: the engine chains a wake every ``wake_interval`` after each
    invocation, in addition to the dirty-flag wakes it schedules at round
    boundaries after state changes.  ``None`` = event-triggered wakes only.
    The scheduler is never invoked more than once per timestamp.

    A scheduler that leaves ``wake_interval`` untouched (neither the
    subclass nor the instance sets it) follows the scenario's ``sim.round``
    — the engine rewires the inherited base default at construction, so
    ``sim: {round: 10m}`` really means a 10-minute cadence.  Set your own
    class or instance ``wake_interval`` (including ``None``) to opt out.
    """

    wake_interval: int | None = 60 * S  # 60 s in µs; engine follows sim.round

    @abstractmethod
    def schedule(self, view: ClusterView) -> list[Action]:
        """Map an immutable view to a list of intents."""
        ...


# ---------------------------------------------------------------------------
# Registry (decorator + entry points)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Scheduler]] = {}

#: Built-in scheduler modules imported (lazily) before any lookup so their
#: ``@register`` decorators have run.  New built-ins append here.
_BUILTIN_MODULES: tuple[str, ...] = (
    "fleetsim.schedulers.fifo",
    "fleetsim.schedulers.tiered_priority",
)


def register(name: str):
    """Class decorator registering a Scheduler under ``name``.

    Re-registering the same class under the same name is a no-op;
    registering a different class under a taken name raises ValueError.
    """

    def deco(cls: type[Scheduler]) -> type[Scheduler]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"scheduler name {name!r} already registered to"
                f" {existing.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return deco


def _load_builtins() -> None:
    for mod in _BUILTIN_MODULES:
        importlib.import_module(mod)


def registered_schedulers() -> tuple[str, ...]:
    """Names of all registered (built-in + already-imported) schedulers,
    sorted."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def get_scheduler(name: str, params: Mapping[str, object] | None = None) -> Scheduler:
    """Instantiate the scheduler registered under ``name`` with ``params``
    as keyword arguments.

    Lookup order: the in-process registry (built-ins are imported first),
    then the ``fleetsim.schedulers`` entry-points group (out-of-tree
    plugins; the loaded object must be a Scheduler subclass, and is
    registered for subsequent lookups).  Unknown names raise ``ValueError``
    listing both registered schedulers and installed-but-unloaded entry
    points.  Params the scheduler's ``__init__`` does not accept also
    raise ``ValueError`` (naming the scheduler and the params), so
    ``--override scheduler.name=...`` failures surface as clean config
    errors, not tracebacks.
    """
    _load_builtins()
    cls = _REGISTRY.get(name)
    if cls is None:
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == name:
                loaded = ep.load()
                if not (isinstance(loaded, type) and issubclass(loaded, Scheduler)):
                    raise TypeError(
                        f"entry point {name!r} in group {ENTRY_POINT_GROUP!r}"
                        f" is not a Scheduler subclass: {loaded!r}"
                    )
                cls = loaded
                _REGISTRY.setdefault(name, cls)
                break
    if cls is None:
        ep_names = sorted(
            {
                ep.name
                for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
            }
            - set(_REGISTRY)
        )
        extra = f"; entry points: {', '.join(ep_names)}" if ep_names else ""
        raise ValueError(
            f"unknown scheduler {name!r}"
            f" (registered: {', '.join(sorted(_REGISTRY)) or 'none'}{extra})"
        )
    kwargs = dict(params or {})
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"scheduler {name!r} rejected params {kwargs!r}: {exc}"
            f" (when switching schedulers via --override, also override"
            f" scheduler.params, e.g. --override \"scheduler.params={{}}\")"
        ) from exc
