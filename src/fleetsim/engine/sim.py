"""The discrete-event simulation engine (DESIGN §6, §8).

:class:`Simulator` owns all mutable runtime state — the event queue, the
job queue, per-job stint bookkeeping, failure/maintenance samplers — and
drives the pluggable pieces: a :class:`~fleetsim.workload.base.JobSource`
(pulled lazily), a :class:`~fleetsim.schedulers.base.Scheduler` (invoked
via coalesced wakes), a :class:`~fleetsim.metrics.base.MetricsSink`
(called at every state transition), and an :class:`AdmissionPolicy`
(no-op pass-through in v1, the v0.2 quota seam).

UNITS: all engine times are int microseconds; all ``*_s`` quantities are
float seconds; "work" is measured in seconds-at-speed-1 (a job's total
work is its ``true_duration_s``).

WORK / CHECKPOINT MATH (pinned)
-------------------------------
While running, wall-clock progress rate is ``speed * eff`` where ``speed``
is 1.0 in v1 (:meth:`Simulator.speed` is the v0.3 hook) and ``eff =
interval / (interval + save)`` amortizes checkpoint-save overhead into the
rate (``interval`` = ``checkpoint_interval_s``, ``save`` =
``checkpoint_save_s``; ``interval`` of 0/None disables checkpointing and
means ``eff = 1``).  The amortization only applies when the stint's
REMAINING work exceeds one interval — a job that will finish before its
first checkpoint boundary (e.g. a 2-minute eval under a 1 h interval)
writes no checkpoints and pays no save tax (``eff = 1`` for that stint).
On a stint start at ``t`` with remaining work ``W``: completion is
scheduled at ``t + overhead + W/(speed*eff)`` where ``overhead`` is
``restart_overhead_s`` on resumes (never the first start).
On interruption after ``dt`` wall-seconds:
``work = max(0, dt - overhead_this_stint) * speed * eff`` (capped at the
remaining work); ``cum = kept + work``;
``kept' = max(kept, floor(cum / interval) * interval)`` when checkpointing
is enabled, else ``kept' = kept`` (which stays 0 — never-checkpointed jobs
lose everything); ``lost += cum - kept'``; remaining work is
``total - kept'``.  Progress never regresses.  EXCEPTION (drain grace,
DESIGN §8): a resident preempted because its node's maintenance drain
grace expired had the whole grace window to checkpoint, so its
interruption banks the full ``cum`` (``kept' = min(cum, total)``) when
checkpointing is enabled — an out-of-band checkpoint, not the floor.  A
node FAILURE during the save window cancels the bank (floor semantics).

ATTAINED SERVICE vs GOODPUT (pinned)
------------------------------------
``job.goodput_chip_s`` is surviving (checkpointed) work x chips only.
``job.attained_service_chip_s`` is ALL service consumed — surviving plus
lost work — x chips (the LAS/Tiresias input; a job that lost 9 h of
GPU-time is not "brand new").  Overheads (restart, checkpoint save) count
in neither.  ``JobView.attained_service_chip_s`` additionally includes
the current stint's in-flight work.

EVENT PAYLOAD CONVENTIONS
-------------------------
``JOB_ARRIVAL``: the :class:`~fleetsim.model.Job`.  ``JOB_COMPLETION`` /
``PREEMPTION_DONE``: job id.  ``NODE_REPAIR``: node id.  ``NODE_FAILURE``:
``None`` for a sampled failure (victim picked at fire time, next failure
chained), or a node id to force that node down (tests; no chaining).
``MAINTENANCE_DRAIN``: ``None`` sampled, node id forced,
``("grace", node_id)`` for a drain-grace expiry, or — v0.4 calendar
reservations, which ride this event channel (DESIGN §17.5) —
``("res_start", index)`` / ``("res_end", index)`` where ``index`` is the
position in the engine's ``(start_us, id)``-sorted reservation list.
``JOB_TIMEOUT``: ``("lifetime", job_id)`` or ``("valid_until", job_id)``.

DETERMINISM: a run is a pure function of (scenario, fleet, source,
scheduler, seed).  RNG streams used: ``"failures"`` and ``"maintenance"``
(one exponential gap per armed event at the STATIC fleet-wide maximum
aggregate rate, then at fire time one thinning-accept uniform plus one
victim-pick uniform — the draw pattern is exogenous, so paired A/B runs
stay aligned even when the healthy set differs); ``"maintenance"``
additionally draws each node's maintenance duration AT DRAIN START
(workload-independent), never at the drain->MAINTENANCE transition;
``"repair"`` (repair kind + delay); ``"failure_causes"`` (one uniform per
realized node failure, DESIGN §8 cause mix).  Sampled events are thinned:
accepted with probability ``current_rate / max_rate`` (exact for a
piecewise-constant hazard, by memorylessness), so the failure hazard
tracks the healthy set with no stale-rate bias.  Iteration order is
sorted or insertion order everywhere.

INVARIANTS
----------
- The scheduler is invoked at most once per timestamp, always after every
  same-timestamp state change (event-type ranks), and sees tentative
  reservations made through its own ``find_placement`` calls.
- At every SCHED_WAKE the engine first calls the source's optional
  ``refill(now_us, pending_by_class)`` hook (closed-loop backlog, v0.2)
  and submits the returned jobs at ``now`` through the normal admission
  path, BEFORE the scheduler view is built — refilled jobs are visible
  to the same wake.
- A job holds an allocation exactly while RUNNING or in a preemption
  grace window; every transition calls the matching sink method.
- Node failure is never terminal for a job in v1 — victims requeue with
  their ORIGINAL ``submit_t`` and retry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..config import FailureModelConfig, QuotaConfig, ReservationConfig, Scenario
from ..fleet.tree import FleetTree, Placement
from ..model import (
    Allocation,
    CapacityClass,
    Constraint,
    GangSpec,
    Job,
    JobStatus,
    NodeState,
    PreemptMode,
    Tier,
)
from ..metrics.base import MetricsSink
from ..schedulers.base import (
    Action,
    DomainView,
    JobView,
    Place,
    PlacementPolicy,
    Preempt,
    ReservationView,
    Scheduler,
)
from ..units import DAY
from ..workload.base import JobSource
from .events import EventQueue, EventType
from .rng import RngStreams

__all__ = [
    "AdmissionPolicy",
    "PassThrough",
    "QuotaAdmission",
    "Simulator",
    "FAILURE_CAUSE_MIX",
]

_DAYS_PER_MONTH = 30.0  # maintenance "per node-month" denominator (pinned)

#: DESIGN §8 default failure-cause mix for reporting (Llama-3 paper):
#: ~60% GPU/HBM, ~10% network, ~13% software, remainder "other".  Sampled
#: per realized node failure from the ``"failure_causes"`` stream and
#: threaded to ``sink.node_failed(..., cause=...)``.
FAILURE_CAUSE_MIX: tuple[tuple[str, float], ...] = (
    ("gpu_hbm", 0.60),
    ("network", 0.10),
    ("software", 0.13),
    ("other", 0.17),
)

_TERMINAL = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELED,
        JobStatus.TIMEOUT,
        JobStatus.NODE_FAIL,
    }
)


def _s_to_us(seconds: float) -> int:
    """Convert float seconds to int microseconds (round-half-even)."""
    return int(round(seconds * 1_000_000))


# ---------------------------------------------------------------------------
# Admission (v1 no-op seam, DESIGN §5 / §11)
# ---------------------------------------------------------------------------


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Quota/admission pipeline stage, upstream of the queue."""

    def admit(self, job: Job, t: int) -> bool: ...


class PassThrough:
    """v1 default: admit everything."""

    def admit(self, job: Job, t: int) -> bool:
        return True


class QuotaAdmission:
    """Tenant chip-quota admission (v0.4; MAST-INSPIRED, deliberately
    simpler — see below).

    A tenant's COMMITTED in-quota chips are the summed gang chips of its
    non-terminal in-quota jobs (pending, running, or graced — commitment
    is taken at admission and released only at the terminal transition,
    which the engine reports through :meth:`job_terminal`).  A job that
    would push its tenant past the cap is OVER-QUOTA: with ``over_quota:
    best_effort`` (default) it is DEMOTED to the BEST_EFFORT band
    (``job.tier`` overwritten, ``job.quota_demoted`` set) and admitted as
    preemptible scavenger work; with ``over_quota: reject`` it is refused
    (the engine records it FAILED).  Tenants absent from the table are
    unlimited.

    HONEST SEMANTICS NOTE (pinned): this is ADMISSION-TIME commitment
    over QUEUED DEMAND (pending + running) with IRREVERSIBLE demotion —
    a burst of short jobs from one tenant charges the cap while queued,
    and a demoted job stays best_effort even after the tenant's usage
    drops to zero.  MAST (OSDI '24) and HyperPod task governance instead
    evaluate in-quota vs over-quota against RUNNING usage at scheduling
    time and treat over-quota work as a dynamic, reclaimable state.
    Scheduling-time evaluation is future work (DESIGN §17.3); demoted
    jobs also keep their ``min_runtime_s`` shield, so an in-quota tenant
    cannot instantly reclaim from long-guarded over-quota work.

    INVARIANT (the quota-conservation validation rung): at every instant,
    each tenant's in-quota jobs — running included — sum to at most its
    cap, because commitment is checked at admission and the committed set
    only shrinks between admissions.
    """

    def __init__(self, quota: QuotaConfig):
        self._caps: dict[str, int] = dict(quota.tenants)
        self._reject = quota.over_quota == "reject"
        self._committed: dict[str, int] = {}
        self._in_quota: dict[str, tuple[str, int]] = {}  # job id -> (tenant, chips)

    def admit(self, job: Job, t: int) -> bool:
        cap = self._caps.get(job.tenant)
        if cap is None:
            return True  # unlisted tenant: unlimited
        chips = sum(g.chips for g in job.gangs)
        used = self._committed.get(job.tenant, 0)
        if used + chips <= cap:
            self._committed[job.tenant] = used + chips
            self._in_quota[job.id] = (job.tenant, chips)
            return True
        if self._reject:
            return False
        job.tier = Tier.BEST_EFFORT
        job.quota_demoted = True
        return True

    def job_terminal(self, job: Job, t: int) -> None:
        """Engine hook: release the job's in-quota commitment (no-op for
        over-quota / unlisted-tenant jobs)."""
        entry = self._in_quota.pop(job.id, None)
        if entry is not None:
            tenant, chips = entry
            self._committed[tenant] -= chips

    def committed(self, tenant: str) -> int:
        """Currently committed in-quota chips of ``tenant`` (tests)."""
        return self._committed.get(tenant, 0)

    def is_in_quota(self, job_id: str) -> bool:
        """True iff ``job_id`` holds an in-quota commitment (tests)."""
        return job_id in self._in_quota


# ---------------------------------------------------------------------------
# Weighted victim sampling (failure/maintenance hazards)
# ---------------------------------------------------------------------------


class _Hazard:
    """Fenwick (binary indexed) tree of per-leaf event rates over the
    STATIC sorted leaf order — the weighted-victim sampler for the
    failure and maintenance streams.

    Non-HEALTHY leaves carry rate 0.0; health transitions update one
    leaf in O(log N) and victim picks are O(log N), so the failure path
    costs O(log N) per event instead of an O(fleet) rescan (DESIGN §6.3
    envelope at 100K-node fleets; v0.2 perf fix).  Per-leaf HEALTHY
    rates are fixed at construction (model + lemon factor — they never
    change at runtime).

    UNITS: rates are events per microsecond.  INVARIANTS: deterministic
    — the leaf order is the fleet's static sorted order; ``pick`` mirrors
    "smallest index whose cumulative rate exceeds u" (ties move past,
    matching ``bisect_right``); the float-edge fallback (u at/above the
    total) returns the LAST positive-rate leaf, never a zero-rate one.
    """

    __slots__ = ("_leaves", "_index", "_healthy_rate", "_rate", "_tree", "_n")

    def __init__(
        self,
        leaves: tuple[str, ...],
        rate_fn: Callable[[str], float],
        is_healthy: Callable[[str], bool],
    ):
        self._leaves = leaves
        self._index = {lid: i for i, lid in enumerate(leaves)}
        self._healthy_rate = [rate_fn(lid) for lid in leaves]
        self._rate = [
            hr if is_healthy(lid) else 0.0
            for lid, hr in zip(leaves, self._healthy_rate)
        ]
        n = len(leaves)
        self._n = n
        tree = [0.0] * (n + 1)
        for i, r in enumerate(self._rate):  # O(N) build
            j = i + 1
            tree[j] += r
            parent = j + (j & -j)
            if parent <= n:
                tree[parent] += tree[j]
        self._tree = tree

    @property
    def total(self) -> float:
        """Sum of all current (healthy) rates."""
        acc = 0.0
        j = self._n
        tree = self._tree
        while j > 0:
            acc += tree[j]
            j -= j & -j
        return acc

    def set_healthy(self, leaf_id: str, healthy: bool) -> None:
        """Set one leaf's rate to its healthy value or 0.0 (O(log N))."""
        i = self._index[leaf_id]
        new = self._healthy_rate[i] if healthy else 0.0
        delta = new - self._rate[i]
        if delta == 0.0:
            return
        self._rate[i] = new
        j = i + 1
        tree = self._tree
        n = self._n
        while j <= n:
            tree[j] += delta
            j += j & -j

    def pick(self, u: float) -> str:
        """The leaf at the smallest index whose cumulative rate exceeds
        ``u`` (Fenwick lower-bound descent)."""
        pos = 0
        rem = u
        k = 1
        while (k << 1) <= self._n:
            k <<= 1
        tree = self._tree
        while k > 0:
            nxt = pos + k
            if nxt <= self._n and tree[nxt] <= rem:
                pos = nxt
                rem -= tree[nxt]
            k >>= 1
        if pos >= self._n or self._rate[pos] == 0.0:
            # Float edge (u at/above the total) or landed past the last
            # positive rate: fall back to the last positive-rate leaf.
            pos = min(pos, self._n - 1)
            while pos > 0 and self._rate[pos] == 0.0:
                pos -= 1
        return self._leaves[pos]


# ---------------------------------------------------------------------------
# Per-job runtime bookkeeping (engine-side; Job is slots=True and stays
# declarative)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _JobRt:
    """Engine-side mutable state for one job.  All work quantities are in
    seconds-at-speed-1; ``kept_work_s`` never regresses."""

    job: Job
    spec: GangSpec
    total_work_s: float
    #: Segment count for a segmented gang (v0.2): chips /
    #: (nodes_per_segment * node_size); 0 for non-segmented jobs.
    n_segments: int = 0
    kept_work_s: float = 0.0
    lost_work_s: float = 0.0
    bank_next_interrupt: bool = False  # drain-grace out-of-band checkpoint
    started_ever: bool = False
    first_start_us: int | None = None
    stint_start_us: int = 0
    stint_overhead_s: float = 0.0
    stint_speed: float = 1.0
    stint_eff: float = 1.0
    n_failures: int = 0
    #: True iff the MOST RECENT return to the pending queue was caused by
    #: a node failure (reset on every successful (re)start) — the
    #: "failure_second_order" preemption-trigger input (DESIGN §8/§9).
    #: Deliberately NOT the lifetime ``n_failures`` counter: a job that
    #: failed once on day 2 must not have its day-20 priority evictions
    #: tagged as failure fallout.
    failure_requeued: bool = False
    allocation: Allocation | None = None
    placed_chips: int = 0
    placed_chip_type: str | None = None
    completion_seq: int | None = None
    preemption_seq: int | None = None
    lifetime_seq: int | None = None
    valid_until_seq: int | None = None

    def stint_work_s(self, t_us: int) -> float:
        """Work done so far in the current stint at wall time ``t_us``
        (capped at the remaining work)."""
        dt = (t_us - self.stint_start_us) / 1e6
        work = max(0.0, dt - self.stint_overhead_s) * self.stint_speed * self.stint_eff
        return min(work, self.total_work_s - self.kept_work_s)


@dataclass(slots=True)
class _ReservationRt:
    """Engine-side runtime state for one calendar reservation (v0.4).

    ``leaves`` is the claimed node set (ascending id) once the claim
    succeeds; ``used_cur``/``used_acc_chip_us`` track the OWNER tenant's
    allocated chips on those leaves as an exact int chip-µs integral
    (advanced on every owner alloc/free touching the hold)."""

    cfg: ReservationConfig
    leaves: tuple[str, ...] = ()
    leaf_set: frozenset[str] = frozenset()
    chips_reserved: int = 0
    active: bool = False
    reported: bool = False
    claim_failed: bool = False
    n_evicted_start: int = 0
    n_evicted_end: int = 0
    used_cur: int = 0
    used_acc_chip_us: int = 0
    last_us: int = 0

    def advance(self, t: int) -> None:
        if t > self.last_us:
            self.used_acc_chip_us += self.used_cur * (t - self.last_us)
            self.last_us = t


# ---------------------------------------------------------------------------
# ClusterView adapter over live engine state
# ---------------------------------------------------------------------------


class _EngineView:
    """Concrete :class:`~fleetsim.schedulers.base.ClusterView` for one wake.

    Job views are computed LAZILY on first ``pending()`` / ``running()``
    access and cached for the wake (still frozen snapshots of the wake's
    settled state — nothing between wake start and the scheduler call
    mutates queues).  Sorting is O(P log P) on the accessed list only, so
    schedulers that touch neither list pay nothing — a v0.2 perf
    guardrail for 100K+-node fleets.  ``find_placement`` tentatively
    applies found placements to the fleet tree so subsequent searches in
    the same wake see remaining capacity; the engine confirms or rolls
    back after ``schedule()`` returns.
    """

    __slots__ = ("_sim", "_now", "_pending", "_running", "tentative")

    def __init__(self, sim: "Simulator", now: int):
        self._sim = sim
        self._now = now
        self.tentative: dict[str, Placement] = {}
        self._pending: tuple[JobView, ...] | None = None
        self._running: tuple[JobView, ...] | None = None

    @property
    def now(self) -> int:
        return self._now

    def pending(self) -> tuple[JobView, ...]:
        if self._pending is None:
            key = lambda jv: (jv.submit_time, jv.id)  # noqa: E731
            self._pending = tuple(
                sorted(
                    (
                        self._sim._job_view(rt, self._now)
                        for rt in self._sim._pending.values()
                    ),
                    key=key,
                )
            )
        return self._pending

    def running(self) -> tuple[JobView, ...]:
        if self._running is None:
            key = lambda jv: (jv.submit_time, jv.id)  # noqa: E731
            self._running = tuple(
                sorted(
                    (
                        self._sim._job_view(rt, self._now)
                        for rt in self._sim._running.values()
                    ),
                    key=key,
                )
            )
        return self._running

    def free_capacity(self, domain_id: str) -> int:
        return self._sim.fleet.free_chips(domain_id)

    def domains(self, level: str) -> tuple[DomainView, ...]:
        fleet = self._sim.fleet
        return tuple(
            DomainView(
                id=did,
                level=level,
                chip_type=fleet.domain(did).chip_type,
                total_chips=fleet.total_chips(did),
                free_chips=fleet.free_chips(did),
                healthy_chips=fleet.healthy_chips(did),
            )
            for did in fleet.domains_at(level)
        )

    def search_first_fit(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        return self._sim.fleet.search_first_fit(spec, tenant)

    def search_segmented(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        return self._sim.fleet.search_segmented(spec, tenant)

    def search_best_fit(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """Raw tightest-fit search (v0.7; see
        :meth:`fleetsim.fleet.tree.FleetTree.search_best_fit`)."""
        return self._sim.fleet.search_best_fit(spec, tenant)

    def search_consolidate(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """Raw fewest-domains-touched search (v0.7; see
        :meth:`fleetsim.fleet.tree.FleetTree.search_consolidate`)."""
        return self._sim.fleet.search_consolidate(spec, tenant)

    def search_spread(
        self, spec: GangSpec, tenant: str | None = None
    ) -> Placement | None:
        """Raw maximum-spread search (v0.7; see
        :meth:`fleetsim.fleet.tree.FleetTree.search_spread`)."""
        return self._sim.fleet.search_spread(spec, tenant)

    def graced_job_ids(self) -> tuple[str, ...]:
        """Ids of jobs currently in a preemption grace window (PREEMPTED,
        still holding their chips), sorted.  These jobs appear in neither
        ``pending()`` nor ``running()``; preempting schedulers use this to
        recognize their own in-flight reclaim claims (v0.2)."""
        return tuple(sorted(self._sim._graced))

    def reclaim_feasible(
        self,
        job: JobView,
        victim_ids: Sequence[str],
        *,
        mode: str = "first_fit",
    ) -> bool:
        """Dry-run: would releasing the allocations of ``victim_ids`` let
        ``job`` place right now?  Runs the real placement search on the
        fleet tree with the victims' chips hypothetically freed (leaf
        health respected: chips on DRAINING/FAILED leaves free nothing)
        and restores the tree exactly (v0.2 reclaim planning).  Tentative
        reservations made earlier in this wake stay in force.

        RELAXABLE constraints (v0.4) are honored exactly as FirstFit
        places them: the constrained search runs first; when the job's
        ``within`` is relaxable (``within_required == False``, no
        segments) and its ``relax_after_s`` has elapsed at ``view.now``,
        a failed constrained search retries UNCONSTRAINED — so reclaim
        planning never reports infeasible for a victim set the very next
        wake's relaxed placement would use.

        ``mode`` (v0.7, keyword-only) is the PACKING MODE of the caller's
        placement policy (``PlacementPolicy.search_mode``).  It must match
        what the scheduler actually places with, or a reclaim plan
        predicts a placement the policy would never make.  It defaults to
        ``"first_fit"`` — the v0.2 behavior — and preempting schedulers
        pass it only when their policy is not FirstFit, so existing
        two-argument callers and custom views are untouched."""
        con = (
            Constraint(
                level=job.within,
                required=job.within_required,
                relax_after_s=job.relax_after_s,
            )
            if job.within is not None
            else None
        )
        spec = GangSpec(
            chips=job.chips,
            chip_type=job.chip_type,
            within=con,
            segments=job.segments,
        )
        if (
            self._sim.fleet.search_after_release(
                spec, victim_ids, job.tenant, mode=mode
            )
            is not None
        ):
            return True
        if (
            con is not None
            and not con.required
            and job.segments is None
            and (self._now - job.submit_time) / 1e6 >= con.relax_after_s
        ):
            relaxed = GangSpec(chips=job.chips, chip_type=job.chip_type)
            return (
                self._sim.fleet.search_after_release(
                    relaxed, victim_ids, job.tenant, mode=mode
                )
                is not None
            )
        return False

    def reserved_free_chips(self, domain_id: str, tenant: str | None) -> int:
        """Free chips under ``domain_id`` sitting on leaves held by a
        calendar reservation for a DIFFERENT tenant — capacity ``tenant``
        can never place on while the hold is active (v0.4).  0 when no
        hold is active.  Chip-count-honest schedulers (e.g. EASY's shadow
        accounting) subtract this from ``DomainView.free_chips``."""
        return self._sim.fleet.reserved_free_chips(domain_id, tenant)

    def reservations(self) -> tuple[ReservationView, ...]:
        """Calendar reservations visible to schedulers (v0.4): every
        configured block that has not finished (``end_us > now``) and did
        not fail its claim, in ``(start_us, id)`` order.  ``active`` is
        True once the hold is claimed, and ``leaves`` then names the held
        nodes — so topology-aware policies can avoid placing long jobs
        onto imminent or active holds (DESIGN §17.4)."""
        out: list[ReservationView] = []
        for res in self._sim._reservations:
            cfg = res.cfg
            if cfg.end_us <= self._now or res.claim_failed:
                continue
            out.append(
                ReservationView(
                    id=cfg.id,
                    tenant=cfg.tenant,
                    chips=cfg.chips,
                    level=cfg.level,
                    chip_type=cfg.chip_type,
                    start_us=cfg.start_us,
                    end_us=cfg.end_us,
                    hard_end=cfg.hard_end,
                    active=res.active,
                    leaves=res.leaves if res.active else (),
                )
            )
        return tuple(out)

    def placement_speed(self, placement: Placement) -> float:
        """The wall-clock speed multiplier the cost model would charge
        this placement (v0.4 ``penalties.xover``); exactly the ``speed``
        the engine will use at stint start.  1.0 without penalties.
        Estimate-honest schedulers divide walltime promises by this."""
        return self._sim.placement_speed(placement)

    def find_placement(
        self, job: JobView, policy: PlacementPolicy
    ) -> Placement | None:
        cached = self.tentative.get(job.id)
        if cached is not None:
            return cached
        if job.id not in self._sim._pending:
            return None
        placement = policy.place(job, self)
        if placement is None:
            return None
        self._sim.fleet.apply(Allocation(job.id, [placement.to_gang_alloc()]))
        self.tentative[job.id] = placement
        return placement

    def release_tentative(self, job_id: str) -> None:
        """Roll back a tentative reservation made by ``find_placement``
        THIS wake: the scheduler examined the placement and decided not
        to ``Place`` the job, and wants the chips back for later
        searches in the same wake (v0.4; unclaimed tentatives are rolled
        back after ``schedule()`` returns anyway).  Unknown/unreserved
        ids are a no-op."""
        if job_id in self.tentative:
            self._sim.fleet.release(job_id)
            del self.tentative[job_id]

    def throughput(self, job: JobView, chip_type: str) -> float:
        return 1.0  # Gavel-matrix hook, v0.4


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class Simulator:
    """Event loop + state machine for one simulation run.

    ``strict=True`` (the default) makes illegal scheduler intents raise
    (DESIGN §7); ``strict=False`` skips them silently.  Regardless of
    ``strict``, the following are always refused: MONITORING victims;
    PROD-preempting-PROD (when ``Preempt.preemptor`` is named); and
    anonymous Preempts (``preemptor=None``) against PROD victims — an
    anonymous preemption may only target sub-PROD bands, so no scheduler
    can evade the no-preemption-within-PROD guardrail by omitting the
    preemptor.  ``Place`` actions are validated against the job's
    :class:`~fleetsim.model.GangSpec`: chip count, pinned chip type, and
    the hard ``within`` constraint (all leaves under ONE domain at that
    level) must hold or the action is refused.

    WAKE CADENCE: when the scheduler instance inherits the base-class
    default ``wake_interval`` (i.e. neither the subclass nor the instance
    set its own), the engine overwrites it with the scenario's
    ``sim.round`` so the configured round actually drives the scheduling
    cadence (DESIGN §6.1/§13).  A scheduler that declares its own
    ``wake_interval`` (including ``None`` = event-triggered) keeps it.

    ``rng`` overrides the seed-derived stream factory (e.g. to vary only
    the ``"failures"`` stream between paired runs).

    ``progress_cb`` (v0.5, opt-in) is invoked at every METRICS_FLUSH —
    including the final flush at the horizon — with one dict::

        {t_us, horizon_us, jobs_finished, jobs_running, pending,
         occupancy_to_date, allocated_chips, healthy_chips}

    The last three mirror the sink's flush-sampled timeseries row and are
    ``None`` when the sink does not expose ``last_flush_sample`` (custom
    sinks).  The callback only OBSERVES: with ``progress_cb=None`` (the
    default) no code path changes and outputs stay byte-identical.  An
    exception raised by the callback aborts the run (``fleetsim serve``
    uses exactly this for cooperative cancellation).

    ``progress_stints`` (v0.8, opt-in and OFF by default so the v0.5
    snapshot shape above stays pinned for existing callers) adds three
    keys to every snapshot, read off the sink's live stint API
    (:meth:`~fleetsim.metrics.collector.MetricsCollector.stints_since` /
    ``open_stint_rows``; a sink without them yields ``[]``/``[]``/``0``)::

        {stints: [row, ...],       # SETTLED since the previous flush
         stint_cursor: int,        # rows consumed so far (monotone)
         open_stints: [row, ...],  # overlay of stints still open, t1_us=t
         stint_fleet: {...}|None}  # domain geometry, FIRST snapshot only

    Still pure observation — the engine reads the sink, never mutates it,
    and outputs are byte-identical with the flag on or off.
    """

    def __init__(
        self,
        scenario: Scenario,
        fleet: FleetTree,
        source: JobSource,
        scheduler: Scheduler,
        sink: MetricsSink,
        admission: AdmissionPolicy | None = None,
        *,
        rng: RngStreams | None = None,
        strict: bool = True,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
        progress_stints: bool = False,
    ):
        self.scenario = scenario
        self.fleet = fleet
        self.source = source
        self.scheduler = scheduler
        self.sink = sink
        self.admission: AdmissionPolicy = (
            admission if admission is not None else PassThrough()
        )
        self.rng = rng if rng is not None else RngStreams(scenario.sim.seed)
        self.strict = bool(strict)
        self.progress_cb = progress_cb
        self._progress_stints = bool(progress_stints)
        self._stint_cursor = 0
        self._stint_fleet_sent = False
        self._n_finished = 0  # sink.job_finished calls (terminal jobs)
        self.queue = EventQueue()
        self.now: int = 0

        self._round_us: int = scenario.sim.round_us
        self._horizon_us: int = scenario.sim.horizon_us
        if self._round_us <= 0:
            raise ValueError(f"sim.round must be positive, got {self._round_us}")
        self._wire_wake_interval()

        self._jobs: dict[str, _JobRt] = {}
        self._pending: dict[str, _JobRt] = {}  # insertion order; views sort
        self._running: dict[str, _JobRt] = {}
        self._graced: dict[str, _JobRt] = {}  # PREEMPTED, grace pending

        self._dirty = False
        self._wake_times: set[int] = set()
        self._last_wake_us: int | None = None
        # v0.4: placement-quality speed penalties + quota terminal hook +
        # calendar reservations.  All default to inert when the scenario
        # does not use the features (byte-identical outputs).
        self._penalties = scenario.penalties
        self._admission_terminal = getattr(self.admission, "job_terminal", None)
        self._res_report_sink = getattr(self.sink, "reservation_report", None)
        self._reservations: list[_ReservationRt] = [
            _ReservationRt(cfg=cfg)
            for cfg in sorted(
                scenario.reservations, key=lambda r: (r.start_us, r.id)
            )
        ]
        self._res_active: list[_ReservationRt] = []
        self._failure_seq: int | None = None
        self._maint_seq: int | None = None
        self._maint_wait: set[str] = set()  # DRAINING past grace, owners pending
        self._maint_duration: dict[str, float] = {}  # node -> duration (s)
        self._leaf_model: dict[str, FailureModelConfig] = self._map_leaf_models()
        # Leaf (node) sizes present per chip type — segmented-gang specs
        # are validated against this SET (a spec is accepted when ANY
        # size decomposes it into whole segments, exactly matching
        # FleetTree.search_segmented's per-size search; a max-size-only
        # quantum would reject placeable jobs on mixed-node-size fleets).
        self._node_sizes: dict[str, set[int]] = {}
        for lid in self.fleet.leaves():
            leaf = self.fleet.domain(lid)
            self._node_sizes.setdefault(leaf.chip_type, set()).add(leaf.chips)
        # Static fleet-wide maximum aggregate rates (all leaves healthy);
        # sampled events are thinned against these — exact by memorylessness.
        self._failure_rate_max: float = sum(
            self._failure_rate_us(lid) for lid in self.fleet.leaves()
        )
        self._maint_rate_max: float = sum(
            self._maint_rate_us(lid) for lid in self.fleet.leaves()
        )
        # Weighted-victim samplers per stream: Fenwick trees over the
        # static leaf order, updated O(log N) per health transition and
        # queried O(log N) per sampled event (v0.2 perf fix — the
        # previous per-event healthy-leaf rescan was O(fleet), which is
        # quadratic-ish at 100K-node scale with failures on).
        def _is_healthy(lid: str) -> bool:
            return self.fleet.domain(lid).state is NodeState.HEALTHY

        self._hazards: dict[str, _Hazard] = {
            "failures": _Hazard(
                self.fleet.leaves(), self._failure_rate_us, _is_healthy
            ),
            "maintenance": _Hazard(
                self.fleet.leaves(), self._maint_rate_us, _is_healthy
            ),
        }

    def _wire_wake_interval(self) -> None:
        """Point an un-overridden scheduler ``wake_interval`` at the
        scenario's ``sim.round`` (see the class docstring)."""
        sched = self.scheduler
        if "wake_interval" in vars(sched):
            return  # instance-level override wins
        for klass in type(sched).__mro__:
            if "wake_interval" in vars(klass):
                if klass is Scheduler:  # inherited the base default only
                    sched.wake_interval = self._round_us
                return

    # -- hooks ----------------------------------------------------------

    def speed(self, job: Job, placement: Placement) -> float:
        """Wall-clock progress multiplier for a placement.

        v0.4: the ``penalties.xover`` cost model — for every configured
        level, a placement whose leaves do NOT all sit under one domain
        at that level runs at the level's multiplier (several configured
        levels multiply).  This penalizes BOTH segmented multi-pod gangs
        and relaxed ``within`` placements with the same rule.  A leaf
        with no ancestor at the level counts as its own singleton domain.
        Without a ``penalties`` section every speed is exactly 1.0
        (byte-identical outputs to pre-v0.4).  The multiplier depends
        only on the placement, never the job — schedulers may price a
        candidate placement via the view's ``placement_speed`` and get
        exactly this value."""
        return self.placement_speed(placement)

    def placement_speed(self, placement: Placement) -> float:
        """The ``penalties.xover`` multiplier for ``placement`` (see
        :meth:`speed`); exposed to schedulers via the engine view."""
        pen = self._penalties
        if pen is None or not pen.xover:
            return 1.0
        speed = 1.0
        for level in sorted(pen.xover):
            if self._spans_level(placement, level):
                speed *= pen.xover[level]
        return speed

    def _spans_level(self, placement: Placement, level: str) -> bool:
        """True iff the placement's leaves sit under MORE than one domain
        at ``level`` (leaves lacking an ancestor at the level are their
        own singleton domains)."""
        first: str | None = None
        for lid, _ in placement.leaves:
            dom = lid  # fallback: no ancestor at `level`
            for aid in self.fleet.ancestors(lid, include_self=True):
                if self.fleet.domain(aid).level == level:
                    dom = aid
                    break
            if first is None:
                first = dom
            elif dom != first:
                return True
        return False

    # -- setup ----------------------------------------------------------

    def _map_leaf_models(self) -> dict[str, FailureModelConfig]:
        """Per-leaf failure model: the leaf's cluster's (inherited) model,
        falling back to the scenario-global model."""
        out: dict[str, FailureModelConfig] = {}
        for metro in self.scenario.fleet.metros:
            for dc in metro.datacenters:
                for cluster in dc.clusters:
                    root = f"{metro.name}/{cluster.id}"
                    if root in self.fleet:
                        for lid in self.fleet.leaves_under(root):
                            out[lid] = cluster.failure_model
        return out

    def _model_for(self, leaf_id: str) -> FailureModelConfig:
        return self._leaf_model.get(leaf_id, self.scenario.failure_model)

    # -- main loop ------------------------------------------------------

    def run(self) -> None:
        """Run to the horizon: pop events while ``time <= horizon``, then
        emit the final metrics flush at exactly the horizon."""
        self._schedule_next_arrival()
        self._arm_failure()
        self._arm_maintenance()
        # Calendar reservations (v0.4): start/end ride the engine's
        # maintenance event channel (rank 5 — after same-timestamp
        # completions/failures/arrivals, BEFORE the wake) with tagged
        # payloads, so claims and hard-end cliffs settle before the
        # scheduler sees the state.
        for i, res in enumerate(self._reservations):
            if res.cfg.start_us <= self._horizon_us:
                self.queue.push(
                    res.cfg.start_us, EventType.MAINTENANCE_DRAIN, ("res_start", i)
                )
                if res.cfg.end_us <= self._horizon_us:
                    self.queue.push(
                        res.cfg.end_us, EventType.MAINTENANCE_DRAIN, ("res_end", i)
                    )
        # Seed a wake at t=0 for periodic schedulers AND whenever the
        # source implements the closed-loop ``refill`` hook: refill runs
        # only inside wakes, so an event-triggered scheduler
        # (wake_interval=None) paired with a pure closed-loop workload
        # would otherwise silently generate zero jobs (no arrivals, no
        # dirty marks, no wakes — ever).
        if (
            self.scheduler.wake_interval is not None
            or getattr(self.source, "refill", None) is not None
        ):
            self._ensure_wake(0)
        if self._round_us < self._horizon_us:
            self.queue.push(self._round_us, EventType.METRICS_FLUSH, None)
        while True:
            t = self.queue.peek_time()
            if t is None or t > self._horizon_us:
                break
            ev = self.queue.pop()
            self.now = ev.time
            self._dispatch(ev)
        self.now = self._horizon_us
        self._finalize_reservations()
        self._report_live_progress()
        self.sink.flush(
            self._horizon_us, self.fleet, len(self._pending), len(self._running)
        )
        self._emit_progress(self._horizon_us)

    def _report_live_progress(self) -> None:
        """At the horizon, credit the checkpoint-banked (durable) work of
        still-allocated jobs to the sink via ``job_progress`` — metrics
        only; engine state is left untouched.  Work past the last
        checkpoint boundary is at-risk, not durable, and is not credited
        (jobs without checkpointing bank nothing)."""
        horizon = self._horizon_us
        for jid in sorted(set(self._running) | set(self._graced)):
            rt = self._jobs[jid]
            interval = rt.job.checkpoint_interval_s
            if not (interval and interval > 0):
                continue
            cum = rt.kept_work_s + rt.stint_work_s(horizon)
            banked = math.floor(cum / interval) * interval
            banked = min(max(banked, rt.kept_work_s), rt.total_work_s)
            delta = banked - rt.kept_work_s
            if delta > 0:
                self.sink.job_progress(
                    rt.job, rt.stint_start_us, horizon, delta * rt.spec.chips, 0.0
                )

    def _emit_progress(self, t: int) -> None:
        """Invoke ``progress_cb`` (if set) with the pinned snapshot dict.

        Called immediately after every ``sink.flush``.  The chip/occupancy
        fields are read from the sink's just-sampled timeseries row via the
        optional ``last_flush_sample`` accessor (:class:`MetricsCollector`
        provides it; other sinks yield ``None`` fields).  Pure observation:
        no engine state is touched."""
        cb = self.progress_cb
        if cb is None:
            return
        row: dict[str, Any] | None = None
        probe = getattr(self.sink, "last_flush_sample", None)
        if probe is not None:
            row = probe()
        snapshot: dict[str, Any] = {
            "t_us": t,
            "horizon_us": self._horizon_us,
            "jobs_finished": self._n_finished,
            "jobs_running": len(self._running),
            "pending": len(self._pending),
            "occupancy_to_date": (
                row["occupancy_to_date"] if row is not None else None
            ),
            "allocated_chips": (
                row["allocated_chips"] if row is not None else None
            ),
            "healthy_chips": (row["healthy_chips"] if row is not None else None),
        }
        if self._progress_stints:
            since = getattr(self.sink, "stints_since", None)
            if since is not None:
                rows, self._stint_cursor = since(self._stint_cursor)
            else:  # a custom sink without the live stint API
                rows = []
            snap_open = getattr(self.sink, "open_stint_rows", None)
            snapshot["stints"] = rows
            snapshot["stint_cursor"] = self._stint_cursor
            snapshot["open_stints"] = snap_open(t) if snap_open is not None else []
            # The stint level's domain geometry is static, so it rides the
            # FIRST snapshot only (and is null in every later one).
            fleet_probe = getattr(self.sink, "stint_fleet", None)
            snapshot["stint_fleet"] = (
                fleet_probe()
                if (fleet_probe is not None and not self._stint_fleet_sent)
                else None
            )
            self._stint_fleet_sent = True
        cb(snapshot)

    def _job_finished(
        self,
        job: Job,
        t: int,
        status: JobStatus,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None:
        """The single funnel for terminal job reports: counts the job for
        ``progress_cb`` snapshots, then forwards to the sink verbatim."""
        self._n_finished += 1
        self.sink.job_finished(job, t, status, productive_chip_s, lost_chip_s)

    def _dispatch(self, ev) -> None:
        et = ev.type
        if et is EventType.NODE_REPAIR:
            self._on_repair(ev)
        elif et is EventType.JOB_COMPLETION:
            self._on_completion(ev)
        elif et is EventType.NODE_FAILURE:
            self._on_failure(ev)
        elif et is EventType.PREEMPTION_DONE:
            self._on_preemption_done(ev)
        elif et is EventType.JOB_ARRIVAL:
            self._on_arrival(ev)
        elif et is EventType.MAINTENANCE_DRAIN:
            self._on_maintenance(ev)
        elif et is EventType.JOB_TIMEOUT:
            self._on_timeout(ev)
        elif et is EventType.SCHED_WAKE:
            self._on_wake(ev)
        elif et is EventType.METRICS_FLUSH:
            self._on_flush(ev)
        else:  # pragma: no cover - EventType is closed
            raise AssertionError(f"unhandled event type {et!r}")

    # -- wake coalescing ------------------------------------------------

    def _mark_dirty(self, t: int) -> None:
        """Record a state change and ensure one wake at the next round
        boundary (a boundary that already ran gets the following one)."""
        self._dirty = True
        boundary = -(-t // self._round_us) * self._round_us
        if self._last_wake_us is not None and boundary <= self._last_wake_us:
            boundary = self._last_wake_us + self._round_us
        self._ensure_wake(boundary)

    def _ensure_wake(self, t: int) -> None:
        if t in self._wake_times:
            return
        self._wake_times.add(t)
        self.queue.push(t, EventType.SCHED_WAKE, None)

    def _on_wake(self, ev) -> None:
        t = self.now
        self._wake_times.discard(t)
        self._last_wake_us = t
        self._dirty = False
        self._refill(t)
        view = _EngineView(self, t)
        actions = self.scheduler.schedule(view)
        self._apply_actions(actions, view, t)
        wi = self.scheduler.wake_interval
        if wi is not None and t + wi <= self._horizon_us:
            self._ensure_wake(t + wi)

    def _refill(self, t: int) -> None:
        """Closed-loop backlog hook (v0.2): if the job source implements
        ``refill``, call it with the settled pending counts and submit the
        returned jobs at ``t`` through the normal admission path (sink
        calls included) — BEFORE the scheduler view is built, so the wake
        sees them.  See :class:`fleetsim.workload.base.JobSource`."""
        refill = getattr(self.source, "refill", None)
        if refill is None:
            return
        pending_by_class: dict[str, int] = {}
        for rt in self._pending.values():  # insertion order (deterministic)
            job = rt.job
            key = job.source_class if job.source_class is not None else (
                job.job_class.name
            )
            pending_by_class[key] = pending_by_class.get(key, 0) + 1
        for job in refill(t, pending_by_class):
            if job.submit_t != t:
                raise ValueError(
                    f"refill job {job.id!r} has submit_t {job.submit_t}"
                    f" != wake time {t}"
                )
            self._submit_job(job, t, wake=False)

    # -- action application ---------------------------------------------

    def _apply_actions(self, actions: list[Action], view: _EngineView, t: int) -> None:
        placed: set[str] = set()
        for action in actions:
            if isinstance(action, Place):
                self._apply_place(action, view, t, placed)
            elif isinstance(action, Preempt):
                self._apply_preempt(action, t)
            elif self.strict:
                raise TypeError(f"unknown action {action!r}")
        # Roll back tentative reservations the scheduler didn't confirm.
        for jid in list(view.tentative):
            if jid not in placed:
                self.fleet.release(jid)
                del view.tentative[jid]

    def _apply_place(
        self, action: Place, view: _EngineView, t: int, placed: set[str]
    ) -> None:
        jid = action.job_id
        try:
            rt = self._jobs.get(jid)
            if rt is None:
                raise ValueError(f"Place for unknown job {jid!r}")
            if jid in placed or jid not in self._pending:
                raise ValueError(f"Place for non-pending job {jid!r}")
            self._validate_placement_spec(rt, action.placement, t)
            tent = view.tentative.get(jid)
            if tent is not None and tent == action.placement:
                alloc = self.fleet.allocation(jid)
            else:
                if tent is not None:  # scheduler substituted its own placement
                    self.fleet.release(jid)
                    del view.tentative[jid]
                alloc = Allocation(jid, [action.placement.to_gang_alloc()])
                self.fleet.apply(alloc)  # raises on any conflict
                view.tentative[jid] = action.placement
        except ValueError:
            if self.strict:
                raise
            return
        placed.add(jid)
        self._start_job(rt, alloc, action.placement, t)

    def _validate_placement_spec(
        self, rt: _JobRt, placement: Placement, t: int
    ) -> None:
        """Check a Place action's placement against the job's GangSpec:
        exact chip count, pinned chip type, the ``within`` constraint
        (every leaf under ONE domain at that level — or, for a RELAXED
        placement, that the constraint really was relaxable and its
        ``relax_after`` timeout elapsed), reservation-hold ownership
        (v0.4), and — for segmented specs — whole-node leaves grouped at
        the segment level into multiples of ``nodes_per_segment``.
        Raises ``ValueError`` (refused in strict mode, skipped
        otherwise)."""
        spec = rt.spec
        jid = rt.job.id
        if self.fleet.has_reservations:
            for lid, _ in placement.leaves:
                owner = self.fleet.reserved_owner(lid)
                if owner is not None and owner != rt.job.tenant:
                    raise ValueError(
                        f"Place for job {jid!r}: leaf {lid!r} is reserved"
                        f" for tenant {owner!r} (job tenant"
                        f" {rt.job.tenant!r})"
                    )
        if placement.relaxed:
            if spec.within is None or spec.within.required:
                raise ValueError(
                    f"Place for job {jid!r}: relaxed placement for a job"
                    f" without a relaxable within constraint"
                )
            waited_s = (t - rt.job.submit_t) / 1e6
            if waited_s < spec.within.relax_after_s:
                raise ValueError(
                    f"Place for job {jid!r}: within={spec.within.level!r}"
                    f" may only relax after {spec.within.relax_after_s:.0f}s"
                    f" pending ({waited_s:.0f}s elapsed)"
                )
        if spec.segments is not None:
            nodes_per_seg, seg_level = spec.segments
            if not placement.whole_node:
                raise ValueError(
                    f"Place for job {jid!r}: segmented gangs are whole-node"
                    f" only, got a sub-node placement"
                )
            per_dom: dict[str, int] = {}
            for lid, _ in placement.leaves:
                dom: str | None = None
                for aid in self.fleet.ancestors(lid, include_self=True):
                    if self.fleet.domain(aid).level == seg_level:
                        dom = aid
                        break
                if dom is None:
                    raise ValueError(
                        f"Place for job {jid!r}: leaf {lid!r} has no"
                        f" ancestor at segment level {seg_level!r}"
                    )
                per_dom[dom] = per_dom.get(dom, 0) + 1
            for dom in sorted(per_dom):
                if per_dom[dom] % nodes_per_seg:
                    raise ValueError(
                        f"Place for job {jid!r}: {per_dom[dom]} nodes under"
                        f" segment domain {dom!r} is not a multiple of"
                        f" nodes_per_segment {nodes_per_seg}"
                    )
        if placement.chips != spec.chips:
            raise ValueError(
                f"Place for job {jid!r}: placement covers {placement.chips}"
                f" chips, gang spec wants {spec.chips}"
            )
        if spec.chip_type is not None and placement.chip_type != spec.chip_type:
            raise ValueError(
                f"Place for job {jid!r}: placement chip type"
                f" {placement.chip_type!r} != pinned spec type"
                f" {spec.chip_type!r}"
            )
        if spec.within is not None and not placement.relaxed:
            level = spec.within.level
            anchor: str | None = None
            for lid, _ in placement.leaves:
                dom: str | None = None
                for aid in self.fleet.ancestors(lid, include_self=True):
                    if self.fleet.domain(aid).level == level:
                        dom = aid
                        break
                if dom is None:
                    raise ValueError(
                        f"Place for job {jid!r}: leaf {lid!r} has no ancestor"
                        f" at within-level {level!r}"
                    )
                if anchor is None:
                    anchor = dom
                elif dom != anchor:
                    raise ValueError(
                        f"Place for job {jid!r} violates within={level!r}:"
                        f" leaves span {anchor!r} and {dom!r}"
                    )

    def _apply_preempt(self, action: Preempt, t: int) -> None:
        try:
            rt = self._jobs.get(action.job_id)
            if rt is None:
                raise ValueError(f"Preempt for unknown job {action.job_id!r}")
            job = rt.job
            if job.status is not JobStatus.RUNNING:
                raise ValueError(
                    f"Preempt victim {job.id!r} is not running"
                    f" (status {job.status.name})"
                )
            if job.tier is Tier.MONITORING:
                raise ValueError(f"cannot preempt MONITORING job {job.id!r}")
            if not job.preemptible:
                raise ValueError(f"job {job.id!r} is not preemptible")
            elapsed_s = (t - rt.stint_start_us) / 1e6
            if elapsed_s < job.min_runtime_s:
                raise ValueError(
                    f"job {job.id!r} min_runtime not elapsed this stint"
                    f" ({elapsed_s:.0f}s < {job.min_runtime_s:.0f}s)"
                )
            trigger = "scheduler"
            if action.preemptor is None:
                if job.tier is Tier.PROD:
                    raise ValueError(
                        f"anonymous Preempt (preemptor=None) cannot target"
                        f" PROD job {job.id!r}; name the preemptor so band"
                        f" rules can be enforced"
                    )
            else:
                preemptor_rt = self._jobs.get(action.preemptor)
                if preemptor_rt is None:
                    raise ValueError(f"unknown preemptor {action.preemptor!r}")
                p_tier = preemptor_rt.job.tier
                if job.tier is Tier.PROD and p_tier is Tier.PROD:
                    raise ValueError(
                        f"no preemption within PROD ({job.id!r} by"
                        f" {action.preemptor!r})"
                    )
                if job.tier >= p_tier:
                    raise ValueError(
                        f"victim {job.id!r} tier {job.tier.name} is not below"
                        f" preemptor {action.preemptor!r} tier {p_tier.name}"
                    )
                if preemptor_rt.failure_requeued:
                    # DESIGN §8/§9: a job whose MOST RECENT requeue was a
                    # node failure evicting others is a second-order
                    # failure cost, tagged distinctly.  (Keyed on the
                    # last-requeue cause, not the lifetime failure count:
                    # ordinary priority evictions by a long job that
                    # failed once weeks ago are "scheduler".)
                    trigger = "failure_second_order"
        except ValueError:
            if self.strict:
                raise
            return
        self._preempt(rt, t, action.mode, trigger=trigger)

    # -- job lifecycle --------------------------------------------------

    def _job_view(self, rt: _JobRt, t: int) -> JobView:
        job = rt.job
        in_stint = job.status in (JobStatus.RUNNING, JobStatus.PREEMPTED)
        stint = rt.stint_work_s(t) if in_stint else 0.0
        cum = rt.kept_work_s + stint  # surviving progress
        interval = job.checkpoint_interval_s
        if not in_stint:
            ckpt_age = 0.0
        elif interval and interval > 0:
            ckpt_age = cum - math.floor(cum / interval) * interval
        else:
            ckpt_age = cum
        con = rt.spec.within
        return JobView(
            id=job.id,
            submit_time=job.submit_t,
            chips=rt.spec.chips,
            chip_type=rt.spec.chip_type,
            tier=job.tier,
            job_class=job.job_class,
            preemptible=job.preemptible,
            min_runtime_s=job.min_runtime_s,
            # Attained service = ALL work consumed (surviving + lost).
            attained_service_chip_s=(cum + rt.lost_work_s) * rt.spec.chips,
            checkpoint_age_s=ckpt_age,
            walltime_est_s=job.walltime_est_s,
            within=con.level if con is not None else None,
            tenant=job.tenant,
            segments=rt.spec.segments,
            n_segments=rt.n_segments,
            within_required=con.required if con is not None else True,
            relax_after_s=con.relax_after_s if con is not None else 300.0,
        )

    def _schedule_next_arrival(self) -> None:
        nxt = self.source.next_arrival()
        if nxt is None:
            return
        t_arr, job = nxt
        if t_arr < self.now:
            raise ValueError(
                f"job source went backwards: arrival at {t_arr} < now {self.now}"
            )
        self.queue.push(t_arr, EventType.JOB_ARRIVAL, job)

    def _on_arrival(self, ev) -> None:
        self._submit_job(ev.payload, self.now, wake=True)
        self._schedule_next_arrival()

    def _segment_count(self, job: Job, spec: GangSpec) -> int:
        """Validate a segmented spec and return its segment count.

        Segmented gangs (v0.2) are WHOLE-NODE only: ``chips`` must
        decompose into whole segments of exactly ``nodes_per_segment``
        nodes for AT LEAST ONE leaf size of the pinned chip type —
        matching ``FleetTree.search_segmented``, which tries every leaf
        size.  On mixed-node-size fleets the returned count derives from
        the LARGEST decomposable size and may differ from a realized
        placement on a smaller size (rare; documented v0.2 limitation).
        Raises ``ValueError`` on any violation — the caller
        (:meth:`_submit_job`) turns that into a per-job FAILED terminal
        status, never a crash of the run."""
        nodes_per_seg, seg_level = spec.segments  # type: ignore[misc]
        if not isinstance(nodes_per_seg, int) or nodes_per_seg < 1:
            raise ValueError(
                f"job {job.id!r}: segments nodes_per_segment must be an"
                f" integer >= 1, got {nodes_per_seg!r}"
            )
        if spec.chip_type is not None:
            sizes = self._node_sizes.get(spec.chip_type)
            if sizes is None:
                raise ValueError(
                    f"job {job.id!r}: no fleet leaves of chip_type"
                    f" {spec.chip_type!r}"
                )
        elif len(self._node_sizes) == 1:
            sizes = next(iter(self._node_sizes.values()))
        else:
            raise ValueError(
                f"job {job.id!r}: segmented gangs need a pinned chip_type"
                f" on a heterogeneous fleet"
            )
        # Accept the largest leaf size that decomposes the request into
        # whole segments (deterministic; matches the generator's
        # quantization on uniform fleets).
        for node_size in sorted(sizes, reverse=True):
            if spec.chips % node_size:
                continue
            total_nodes = spec.chips // node_size
            if total_nodes % nodes_per_seg:
                continue
            return total_nodes // nodes_per_seg
        if not any(spec.chips % s == 0 for s in sizes):
            raise ValueError(
                f"job {job.id!r}: segmented gangs are whole-node only —"
                f" {spec.chips} chips is not a multiple of any node size"
                f" in {sorted(sizes)}"
            )
        raise ValueError(
            f"job {job.id!r}: {spec.chips} chips do not divide into"
            f" segments of exactly {nodes_per_seg} nodes for any node"
            f" size in {sorted(sizes)}"
        )

    def _submit_job(self, job: Job, t: int, *, wake: bool) -> None:
        """Submit one job through the admission path (shared by
        ``JOB_ARRIVAL`` events and the refill hook).  ``wake=False``
        skips the dirty-mark — refilled jobs are submitted inside the
        wake that will schedule them."""
        if len(job.gangs) != 1:
            raise ValueError(
                f"job {job.id!r} has {len(job.gangs)} gangs; multi-gang jobs"
                " are not implemented in v0.1"
            )
        if job.id in self._jobs:
            raise ValueError(f"duplicate job id {job.id!r}")
        spec = job.gangs[0]
        try:
            n_segments = (
                self._segment_count(job, spec) if spec.segments is not None else 0
            )
        except ValueError:
            # Spec violation at submission is a PER-JOB terminal failure
            # (recorded in metrics), never an uncaught exception that
            # kills the whole run.
            self._jobs[job.id] = _JobRt(
                job=job, spec=spec, total_work_s=float(job.true_duration_s)
            )
            self.sink.job_submitted(job, t)
            job.status = JobStatus.FAILED
            self._job_finished(job, t, JobStatus.FAILED, 0.0, 0.0)
            return
        rt = _JobRt(
            job=job,
            spec=spec,
            total_work_s=float(job.true_duration_s),
            n_segments=n_segments,
        )
        self._jobs[job.id] = rt
        self.sink.job_submitted(job, t)
        if self.admission.admit(job, t):
            job.status = JobStatus.ADMITTED
            self._pending[job.id] = rt
            self.sink.job_admitted(job, t)
            if job.valid_until is not None:
                rt.valid_until_seq = self.queue.push(
                    max(t, job.valid_until),
                    EventType.JOB_TIMEOUT,
                    ("valid_until", job.id),
                )
            if wake:
                self._mark_dirty(t)
        else:
            job.status = JobStatus.FAILED
            self._job_finished(job, t, JobStatus.FAILED, 0.0, 0.0)

    def _start_job(
        self, rt: _JobRt, alloc: Allocation, placement: Placement, t: int
    ) -> None:
        job = rt.job
        resumed = rt.started_ever
        overhead_s = job.restart_overhead_s if resumed else 0.0
        interval = job.checkpoint_interval_s
        ckpt_on = interval is not None and interval > 0
        remaining = rt.total_work_s - rt.kept_work_s
        # Amortize save overhead only when this stint will actually write
        # a checkpoint (remaining work exceeds one interval) — short jobs
        # pay no tax for checkpoints they never take.
        eff = (
            interval / (interval + job.checkpoint_save_s)
            if ckpt_on and job.checkpoint_save_s > 0 and remaining > interval
            else 1.0
        )
        speed = self.speed(job, placement)
        dur_s = overhead_s + remaining / (speed * eff)

        rt.stint_start_us = t
        rt.stint_overhead_s = overhead_s
        rt.stint_speed = speed
        rt.stint_eff = eff
        rt.failure_requeued = False  # back on chips: failure fallout over
        rt.allocation = alloc
        rt.placed_chips = placement.chips
        rt.placed_chip_type = placement.chip_type
        rt.completion_seq = self.queue.push(
            t + _s_to_us(dur_s), EventType.JOB_COMPLETION, job.id
        )
        del self._pending[job.id]
        self._running[job.id] = rt
        job.status = JobStatus.RUNNING
        if not resumed:
            rt.started_ever = True
            rt.first_start_us = t
            if rt.valid_until_seq is not None:
                self.queue.cancel(rt.valid_until_seq)
                rt.valid_until_seq = None
            if job.max_lifetime_s is not None:
                rt.lifetime_seq = self.queue.push(
                    t + _s_to_us(job.max_lifetime_s),
                    EventType.JOB_TIMEOUT,
                    ("lifetime", job.id),
                )
        if self._res_active:
            self._res_note_alloc(job, alloc, +1, t)
        self.sink.job_started(job, alloc, t)
        self.sink.chips_allocated(placement.chips, placement.chip_type, t)

    def _interrupt(self, rt: _JobRt, t: int) -> None:
        """Apply the pinned checkpoint math for an interruption at ``t``
        and report the stint's surviving/lost work to the sink.  A pending
        ``bank_next_interrupt`` flag (drain grace, DESIGN §8) banks the
        full ``cum`` instead of flooring to the checkpoint boundary."""
        job = rt.job
        work = rt.stint_work_s(t)
        cum = rt.kept_work_s + work
        interval = job.checkpoint_interval_s
        if interval and interval > 0:
            if rt.bank_next_interrupt:
                kept = min(cum, rt.total_work_s)  # out-of-band checkpoint
            else:
                kept = math.floor(cum / interval) * interval
                kept = max(kept, rt.kept_work_s)
                kept = min(kept, rt.total_work_s)
        else:
            kept = rt.kept_work_s  # checkpointing disabled: nothing banked
        rt.bank_next_interrupt = False
        delta_kept = kept - rt.kept_work_s
        delta_lost = cum - kept
        rt.lost_work_s += delta_lost
        rt.kept_work_s = kept
        job.attained_service_chip_s = (kept + rt.lost_work_s) * rt.spec.chips
        job.goodput_chip_s = kept * rt.spec.chips
        if delta_kept > 0 or delta_lost > 0:
            self.sink.job_progress(
                job,
                rt.stint_start_us,
                t,
                delta_kept * rt.spec.chips,
                delta_lost * rt.spec.chips,
            )

    def _release(self, rt: _JobRt, t: int) -> None:
        """Release the job's allocation; report freed chips; complete any
        pending drain->maintenance transitions its leaves were blocking."""
        alloc = rt.allocation
        if alloc is None:
            return
        if self._res_active:
            self._res_note_alloc(rt.job, alloc, -1, t)
        leaf_ids: dict[str, None] = {}
        for gang in alloc.gangs:
            nodes = gang.nodes
            for lid in nodes if isinstance(nodes, list) else nodes.keys():
                leaf_ids.setdefault(lid)
        self.fleet.release(rt.job.id)
        rt.allocation = None
        self.sink.chips_freed(rt.placed_chips, rt.placed_chip_type, t)
        for lid in leaf_ids:
            self._check_maint_transition(lid, t)

    def _tombstone(self, rt: _JobRt) -> None:
        """Cancel every pending event owned by this job."""
        for attr in ("completion_seq", "preemption_seq", "lifetime_seq",
                     "valid_until_seq"):
            seq = getattr(rt, attr)
            if seq is not None:
                self.queue.cancel(seq)
                setattr(rt, attr, None)

    def _finish(self, rt: _JobRt, t: int, status: JobStatus) -> None:
        """Terminal transition: release, tombstone, report, mark dirty
        whenever the schedulable state changed (capacity freed OR a job
        left the pending queue — either can unblock a strict-FIFO head)."""
        job = rt.job
        freed = rt.allocation is not None
        was_pending = job.id in self._pending
        if freed:
            self._release(rt, t)
        self._tombstone(rt)
        self._pending.pop(job.id, None)
        self._running.pop(job.id, None)
        self._graced.pop(job.id, None)
        job.status = status
        if self._admission_terminal is not None:
            self._admission_terminal(job, t)  # quota commitment release
        productive = rt.kept_work_s * rt.spec.chips
        lost = rt.lost_work_s * rt.spec.chips
        self._job_finished(job, t, status, productive, lost)
        if freed or was_pending:
            self._mark_dirty(t)

    def _on_completion(self, ev) -> None:
        jid: str = ev.payload
        rt = self._jobs[jid]
        rt.completion_seq = None
        delta_kept = rt.total_work_s - rt.kept_work_s
        rt.kept_work_s = rt.total_work_s  # all work delivered
        rt.job.attained_service_chip_s = (
            rt.total_work_s + rt.lost_work_s
        ) * rt.spec.chips
        rt.job.goodput_chip_s = rt.total_work_s * rt.spec.chips
        if delta_kept > 0:
            self.sink.job_progress(
                rt.job, rt.stint_start_us, self.now, delta_kept * rt.spec.chips, 0.0
            )
        override = getattr(rt.job, "terminal_status_override", None)
        status = override if isinstance(override, JobStatus) else JobStatus.COMPLETED
        self._finish(rt, self.now, status)

    def _preempt(self, rt: _JobRt, t: int, mode: PreemptMode, trigger: str) -> None:
        """Begin a (validated or engine-initiated) preemption."""
        job = rt.job
        self.sink.job_preempted(job, t, trigger)
        if rt.completion_seq is not None:
            self.queue.cancel(rt.completion_seq)
            rt.completion_seq = None
        self._running.pop(job.id, None)
        if mode is PreemptMode.CANCEL:
            self._interrupt(rt, t)
            self._finish(rt, t, JobStatus.CANCELED)
        else:
            job.status = JobStatus.PREEMPTED
            self._graced[job.id] = rt
            # SPOT capacity (v0.4): zero-notice kill — no checkpoint-save
            # grace window; the job restarts from its last periodic
            # checkpoint (floor semantics; ckpt-off spot loses everything).
            grace_us = (
                0
                if job.capacity is CapacityClass.SPOT
                else _s_to_us(job.checkpoint_save_s)
            )
            rt.preemption_seq = self.queue.push(
                t + grace_us, EventType.PREEMPTION_DONE, job.id
            )

    def _on_preemption_done(self, ev) -> None:
        jid: str = ev.payload
        rt = self._jobs[jid]
        rt.preemption_seq = None
        self._graced.pop(jid, None)
        t = self.now
        self._interrupt(rt, t)
        self._release(rt, t)
        rt.job.status = JobStatus.PENDING
        self._pending[jid] = rt  # original submit_t preserved on the Job
        self.sink.job_requeued(rt.job, t)
        self._mark_dirty(t)

    def _on_timeout(self, ev) -> None:
        kind, jid = ev.payload
        rt = self._jobs[jid]
        job = rt.job
        t = self.now
        if kind == "valid_until":
            rt.valid_until_seq = None
            if rt.started_ever or job.status in _TERMINAL:
                return
            self._finish(rt, t, JobStatus.FAILED)
            return
        # kind == "lifetime"
        rt.lifetime_seq = None
        if job.status in _TERMINAL:
            return
        if job.status is JobStatus.RUNNING:
            if rt.completion_seq is not None:
                self.queue.cancel(rt.completion_seq)
                rt.completion_seq = None
            self._interrupt(rt, t)
        elif job.status is JobStatus.PREEMPTED:
            self._interrupt(rt, t)
        self._finish(rt, t, JobStatus.TIMEOUT)

    # -- failures (DESIGN §8) -------------------------------------------

    def _failure_rate_us(self, leaf_id: str) -> float:
        model = self._model_for(leaf_id)
        if model.node_mtbf_days <= 0:
            return 0.0
        return self.fleet.domain(leaf_id).lemon_factor / (
            model.node_mtbf_days * DAY
        )

    def _maint_rate_us(self, leaf_id: str) -> float:
        model = self._model_for(leaf_id)
        if model.maintenance_rate_per_node_month <= 0:
            return 0.0
        return model.maintenance_rate_per_node_month / (_DAYS_PER_MONTH * DAY)

    def _arm(self, which: str) -> None:
        """(Re)arm the sampled failure or maintenance event if disarmed.

        The gap is exponential at the STATIC fleet-wide maximum aggregate
        rate; the fired event is thinned in :meth:`_thinned_victim`
        (accepted with probability ``current/max``).  Thinning keeps the
        hazard exact under any healthy-set trajectory while the per-stream
        draw pattern (gap, accept, victim) stays exogenous — paired A/B
        runs stay aligned."""
        if which == "failures":
            if self._failure_seq is not None:
                return
            total, etype = self._failure_rate_max, EventType.NODE_FAILURE
        else:
            if self._maint_seq is not None:
                return
            total, etype = self._maint_rate_max, EventType.MAINTENANCE_DRAIN
        if total <= 0.0:
            return
        delay = self.rng.stream(which).exponential(1.0 / total)
        seq = self.queue.push(self.now + max(1, round(delay)), etype, None)
        if which == "failures":
            self._failure_seq = seq
        else:
            self._maint_seq = seq

    def _arm_failure(self) -> None:
        self._arm("failures")

    def _arm_maintenance(self) -> None:
        self._arm("maintenance")

    def _thinned_victim(self, stream_name: str, rate_fn, max_total: float) -> str | None:
        """Thinning-accept the fired event and pick a healthy victim
        weighted by the stream's per-leaf rates (static sorted-id
        order).  Draw order is pinned: one accept uniform, then one
        victim uniform (the victim draw happens even for rejected
        events, keeping paired runs aligned).  Returns ``None`` when
        rejected or nothing is eligible.

        The rates live in the incrementally-maintained :class:`_Hazard`
        Fenwick tree (O(log N) per pick; O(log N) per health
        transition), never an O(fleet) rescan.  ``rate_fn`` is retained
        in the signature for forced-path parity/tests; the hazard tree
        was built from it at construction."""
        hz = self._hazards[stream_name]
        total = hz.total
        if total <= 0.0 or max_total <= 0.0:
            return None
        stream = self.rng.stream(stream_name)
        accepted = stream.random() * max_total < total
        u = stream.random() * total
        victim = hz.pick(u)
        return victim if accepted else None

    def _on_failure(self, ev) -> None:
        t = self.now
        if ev.payload is None:  # sampled
            self._failure_seq = None
            victim = self._thinned_victim(
                "failures", self._failure_rate_us, self._failure_rate_max
            )
            if victim is not None:
                self._fail_node(victim, t)
            self._arm_failure()
        else:  # forced (tests): no chaining
            leaf = self.fleet.domain(ev.payload)
            if leaf.state in (NodeState.HEALTHY, NodeState.DRAINING):
                self._fail_node(ev.payload, t)

    def _sample_failure_cause(self) -> str:
        """One cause label from :data:`FAILURE_CAUSE_MIX`, drawn from the
        dedicated ``"failure_causes"`` stream (one uniform per failure)."""
        u = self.rng.stream("failure_causes").random()
        acc = 0.0
        for cause, frac in FAILURE_CAUSE_MIX:
            acc += frac
            if u < acc:
                return cause
        return FAILURE_CAUSE_MIX[-1][0]  # float-edge fallback

    def _fail_node(self, node_id: str, t: int) -> None:
        leaf = self.fleet.domain(node_id)
        was_healthy = leaf.state is NodeState.HEALTHY
        victims = self.fleet.fail_node(node_id)  # sorted job ids
        if was_healthy:
            for hz in self._hazards.values():
                hz.set_healthy(node_id, False)
        self.sink.node_failed(node_id, t, victims, cause=self._sample_failure_cause())
        if was_healthy:
            self.sink.healthy_delta(-leaf.chips, leaf.chip_type, t)
        self._maint_wait.discard(node_id)  # failure repair supersedes drain
        self._maint_duration.pop(node_id, None)
        for jid in victims:
            rt = self._jobs[jid]
            job = rt.job
            if job.status is JobStatus.RUNNING:
                if rt.completion_seq is not None:
                    self.queue.cancel(rt.completion_seq)
                    rt.completion_seq = None
                self._running.pop(jid, None)
            elif job.status is JobStatus.PREEMPTED:
                if rt.preemption_seq is not None:
                    self.queue.cancel(rt.preemption_seq)
                    rt.preemption_seq = None
                self._graced.pop(jid, None)
            else:  # pragma: no cover - owners are always running/graced
                continue
            rt.bank_next_interrupt = False  # a crash voids the drain bank
            self._interrupt(rt, t)
            rt.n_failures += 1
            rt.failure_requeued = True  # this requeue IS failure-caused
            self._release(rt, t)
            job.status = JobStatus.PENDING
            self._pending[jid] = rt
            self.sink.job_requeued(job, t)
        self._mark_dirty(t)
        # Schedule the node's repair.
        model = self._model_for(node_id)
        r = self.rng.stream("repair")
        if r.random() < model.repair_manual_frac:
            lo, hi = model.repair_manual_days
            delay_s = r.uniform(lo, hi) * 86_400.0
        else:
            lo, hi = model.repair_auto_min
            delay_s = r.uniform(lo, hi) * 60.0
        self.queue.push(t + _s_to_us(delay_s), EventType.NODE_REPAIR, node_id)

    def _on_repair(self, ev) -> None:
        node_id: str = ev.payload
        t = self.now
        leaf = self.fleet.domain(node_id)
        if leaf.state not in (NodeState.FAILED, NodeState.MAINTENANCE):
            return  # e.g. drain cancelled by a forced-failure repair race
        self.fleet.repair_node(node_id)
        for hz in self._hazards.values():
            hz.set_healthy(node_id, True)
        self.sink.node_repaired(node_id, t)
        self.sink.healthy_delta(leaf.chips, leaf.chip_type, t)
        self._mark_dirty(t)
        self._arm_failure()
        self._arm_maintenance()

    # -- maintenance drains (DESIGN §8) ---------------------------------

    def _on_maintenance(self, ev) -> None:
        t = self.now
        p = ev.payload
        if p is None:  # sampled drain
            self._maint_seq = None
            victim = self._thinned_victim(
                "maintenance", self._maint_rate_us, self._maint_rate_max
            )
            if victim is not None:
                self._drain_node(victim, t)
            self._arm_maintenance()
        elif isinstance(p, tuple) and p[0] == "grace":
            self._on_drain_grace(p[1], t)
        elif isinstance(p, tuple) and p[0] == "res_start":
            self._on_reservation_start(self._reservations[p[1]], t)
        elif isinstance(p, tuple) and p[0] == "res_end":
            self._on_reservation_end(self._reservations[p[1]], t)
        else:  # forced (tests): no chaining
            if self.fleet.domain(p).state is NodeState.HEALTHY:
                self._drain_node(p, t)

    def _drain_node(self, node_id: str, t: int) -> None:
        self.fleet.drain_node(node_id)
        for hz in self._hazards.values():
            hz.set_healthy(node_id, False)
        leaf = self.fleet.domain(node_id)
        self.sink.node_drain_started(node_id, t)
        self.sink.healthy_delta(-leaf.chips, leaf.chip_type, t)
        # Draw the maintenance duration NOW (drain start is exogenous —
        # workload-independent), not at the drain->MAINTENANCE transition
        # whose timing depends on when residents release.
        lo, hi = self._model_for(node_id).repair_auto_min
        self._maint_duration[node_id] = (
            self.rng.stream("maintenance").uniform(lo, hi) * 60.0
        )
        grace_us = self._model_for(node_id).drain_grace_us
        self.queue.push(
            t + grace_us, EventType.MAINTENANCE_DRAIN, ("grace", node_id)
        )
        self._mark_dirty(t)

    def _on_drain_grace(self, node_id: str, t: int) -> None:
        leaf = self.fleet.domain(node_id)
        if leaf.state is not NodeState.DRAINING:
            return  # failed (and possibly repaired) during the grace window
        for jid in sorted(self.fleet.owners(node_id)):
            rt = self._jobs[jid]
            if rt.job.status is JobStatus.RUNNING:
                # Engine-initiated: bypasses preemptibility/min-runtime
                # validation — the machine is going away.  The whole grace
                # window was checkpoint notice (DESIGN §8), so the coming
                # interruption banks full progress (out-of-band checkpoint)
                # when checkpointing is enabled.
                rt.bank_next_interrupt = True
                self._preempt(rt, t, PreemptMode.REQUEUE, trigger="maintenance")
            # PREEMPTED residents are already leaving; nothing to do.
        self._maint_wait.add(node_id)
        self._check_maint_transition(node_id, t)

    def _check_maint_transition(self, leaf_id: str, t: int) -> None:
        """DRAINING -> MAINTENANCE once past-grace and empty; schedules the
        node's return to HEALTHY after the maintenance duration drawn at
        drain start (see :meth:`_drain_node`)."""
        if leaf_id not in self._maint_wait:
            return
        if self.fleet.domain(leaf_id).state is not NodeState.DRAINING:
            self._maint_wait.discard(leaf_id)
            return
        if self.fleet.owners(leaf_id):
            return
        self._maint_wait.discard(leaf_id)
        self.fleet.to_maintenance(leaf_id)
        delay_s = self._maint_duration.pop(leaf_id, None)
        if delay_s is None:  # defensive: never drawn (hand-driven state)
            lo, hi = self._model_for(leaf_id).repair_auto_min
            delay_s = (lo + hi) / 2.0 * 60.0
        self.queue.push(t + _s_to_us(delay_s), EventType.NODE_REPAIR, leaf_id)

    # -- calendar reservations (v0.4; DESIGN v0.4 addendum) ---------------

    def _on_reservation_start(self, res: _ReservationRt, t: int) -> None:
        """Claim the hold: pick whole HEALTHY nodes of the reservation's
        chip type inside ONE domain at its level, choosing the domain
        that needs the FEWEST EVICTIONS (distinct non-owner running jobs
        displaced) — ties broken by ascending domain id.  Within a
        domain, leaves are taken free-first, then owner-occupied (zero
        evictions), then foreign-occupied, ascending id within each
        group.  A completely idle domain therefore always beats a busy
        lower-id one — the claim never evicts when any candidate domain
        can host the block eviction-free (DESIGN §17.4; Slurm advance
        reservations pick non-conflicting nodes).  Chosen leaves are
        marked reserved and non-owner residents are evicted (REQUEUE,
        trigger ``"reservation"``, engine-initiated: preemptibility and
        min-runtime guards do NOT apply — the capacity is contractually
        gone).  A fleet that cannot host the hold anywhere records a
        failed claim (reported in the summary) and holds nothing."""
        cfg = res.cfg
        ct = cfg.chip_type
        if ct is None:
            ct = next(iter(sorted(self._node_sizes)))  # homogeneous fleet
        domains = (
            self.fleet.domains_at(cfg.level)
            if cfg.level is not None
            else self.fleet.cluster_roots
        )
        chosen: list[str] | None = None
        got = 0
        best_victims: int | None = None
        for did in domains:  # ascending id order (first minimum wins ties)
            free_leaves: list[str] = []
            owner_leaves: list[str] = []  # residents all belong to the owner
            foreign_leaves: list[str] = []  # >= 1 non-owner resident
            for lid in self.fleet.leaves_under(did):  # ascending id order
                leaf = self.fleet.domain(lid)
                if (
                    leaf.chip_type != ct
                    or leaf.state is not NodeState.HEALTHY
                    or self.fleet.reserved_owner(lid) is not None
                ):
                    continue
                residents = self.fleet.owners(lid)
                if not residents:
                    free_leaves.append(lid)
                elif all(
                    self._jobs[jid].job.tenant == cfg.tenant
                    for jid in residents
                ):
                    owner_leaves.append(lid)
                else:
                    foreign_leaves.append(lid)
            trial: list[str] = []
            trial_got = 0
            trial_victims: set[str] = set()  # distinct RUNNING non-owner jobs
            for lid in free_leaves + owner_leaves + foreign_leaves:
                if trial_got >= cfg.chips:
                    break
                trial.append(lid)
                trial_got += self.fleet.domain(lid).chips
                for jid in sorted(self.fleet.owners(lid)):
                    rt = self._jobs[jid]
                    if (
                        rt.job.tenant != cfg.tenant
                        and rt.job.status is JobStatus.RUNNING
                    ):
                        trial_victims.add(jid)
            if trial_got < cfg.chips:
                continue  # this domain cannot host the block at all
            if best_victims is None or len(trial_victims) < best_victims:
                best_victims = len(trial_victims)
                chosen = trial
                got = trial_got
                if best_victims == 0:
                    break  # nothing beats an eviction-free claim
        if chosen is None:
            res.claim_failed = True
            self._emit_reservation_report(res, "claim_failed", t)
            return
        res.leaves = tuple(sorted(chosen))
        res.leaf_set = frozenset(chosen)
        res.chips_reserved = got
        self.fleet.reserve_leaves(res.leaves, cfg.tenant)
        # Evict non-owner residents (owner jobs already on the hold stay).
        victims: dict[str, None] = {}
        for lid in res.leaves:
            for jid in sorted(self.fleet.owners(lid)):
                victims.setdefault(jid)
        for jid in victims:
            rt = self._jobs[jid]
            if rt.job.tenant == cfg.tenant:
                continue
            if rt.job.status is JobStatus.RUNNING:
                res.n_evicted_start += 1
                self._preempt(rt, t, PreemptMode.REQUEUE, trigger="reservation")
            # PREEMPTED residents are already leaving; nothing to do.
        # Start the owner-usage integral (owner jobs may already be here).
        res.last_us = t
        res.used_cur = 0
        for lid in res.leaves:
            for jid, chips in sorted(self.fleet.owners(lid).items()):
                if self._jobs[jid].job.tenant == cfg.tenant:
                    res.used_cur += chips
        res.active = True
        self._res_active.append(res)
        self._mark_dirty(t)

    def _on_reservation_end(self, res: _ReservationRt, t: int) -> None:
        """Lift the hold.  With ``hard_end`` (the capacity-block cliff),
        residents still on the held nodes are evicted first (REQUEUE,
        trigger ``"reservation"``); they lose work back to their last
        checkpoint and requeue for placement elsewhere."""
        if not res.active:
            return  # claim failed (already reported)
        res.advance(t)
        res.active = False
        self._res_active.remove(res)
        self.fleet.release_reservation(res.leaves)
        if res.cfg.hard_end:
            victims: dict[str, None] = {}
            for lid in res.leaves:
                for jid in sorted(self.fleet.owners(lid)):
                    victims.setdefault(jid)
            for jid in victims:
                rt = self._jobs[jid]
                if rt.job.status is JobStatus.RUNNING:
                    res.n_evicted_end += 1
                    self._preempt(
                        rt, t, PreemptMode.REQUEUE, trigger="reservation"
                    )
        self._emit_reservation_report(res, "completed", t)
        self._mark_dirty(t)

    def _res_note_alloc(self, job: Job, alloc: Allocation, sign: int, t: int) -> None:
        """Advance active holds' owner-usage integrals for an allocation
        (de)applied at ``t`` (called on every start/release; fast-path
        no-op when no hold is active)."""
        for res in self._res_active:
            if job.tenant != res.cfg.tenant:
                continue
            overlap = 0
            for gang in alloc.gangs:
                nodes = gang.nodes
                if isinstance(nodes, dict):
                    for lid, chips in nodes.items():
                        if lid in res.leaf_set:
                            overlap += chips
                else:
                    for lid in nodes:
                        if lid in res.leaf_set:
                            overlap += self.fleet.domain(lid).chips
            if overlap:
                res.advance(t)
                res.used_cur += sign * overlap

    def _emit_reservation_report(
        self, res: _ReservationRt, status: str, t: int
    ) -> None:
        """Send the reservation's final accounting to the sink (probed —
        sinks without ``reservation_report`` are skipped)."""
        res.reported = True
        cfg = res.cfg
        window_end = min(cfg.end_us, self._horizon_us)
        util: float | None = None
        if res.chips_reserved > 0 and window_end > cfg.start_us:
            util = res.used_acc_chip_us / (
                res.chips_reserved * (window_end - cfg.start_us)
            )
        if self._res_report_sink is not None:
            self._res_report_sink(
                {
                    "id": cfg.id,
                    "tenant": cfg.tenant,
                    "level": cfg.level,
                    "chip_type": cfg.chip_type,
                    "hard_end": cfg.hard_end,
                    "start_us": cfg.start_us,
                    "end_us": cfg.end_us,
                    "status": status,
                    "chips_requested": cfg.chips,
                    "chips_reserved": res.chips_reserved,
                    "n_nodes": len(res.leaves),
                    "nodes": list(res.leaves),
                    "n_evicted_at_start": res.n_evicted_start,
                    "n_evicted_at_end": res.n_evicted_end,
                    "utilization": util,
                }
            )

    def _finalize_reservations(self) -> None:
        """At the horizon: settle and report every unreported reservation
        (active holds report ``active_at_horizon`` with the usage integral
        clipped at the horizon; never-started ones report
        ``not_started``)."""
        for res in self._reservations:
            if res.reported:
                continue
            if res.active:
                res.advance(self._horizon_us)
                self._emit_reservation_report(
                    res, "active_at_horizon", self._horizon_us
                )
            else:
                self._emit_reservation_report(res, "not_started", self._horizon_us)

    # -- metrics flush ---------------------------------------------------

    def _on_flush(self, ev) -> None:
        t = self.now
        self.sink.flush(t, self.fleet, len(self._pending), len(self._running))
        self._emit_progress(t)
        nxt = t + self._round_us
        if nxt < self._horizon_us:
            self.queue.push(nxt, EventType.METRICS_FLUSH, None)
