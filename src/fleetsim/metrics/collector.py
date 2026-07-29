"""Metrics collection: time-weighted integrals + per-job event records.

:class:`MetricsCollector` implements the
:class:`~fleetsim.metrics.base.MetricsSink` protocol (DESIGN §9) with the
two mechanisms that section pins:

- **Time-weighted integrals** (:class:`TimeWeighted`): on every state
  change ``acc += value * dt`` — O(1) per change and *exact* (integer
  chip-microsecond arithmetic, no float accumulation error).  Tracked:
  allocated chips (overall, per chip type, per job class, per tenant),
  healthy chips (overall), pending queue depth per class, and
  INFER_REPLICA desired/running replica counts.
- **Event-sourced records**: one row per job accumulating identity,
  timing, preemption/failure counts, and the engine-reported
  productive/lost chip-seconds.

PRODUCTIVE CHIP-TIME (goodput numerator)
----------------------------------------
``job_progress`` calls report each stint's surviving-work delta together
with the stint interval it accrued over.  The collector keeps the full-run
sum AND a window-clipped sum that spreads each delta uniformly over its
stint interval — so the windowed goodput numerator never exceeds the
window-clipped allocated integral (goodput <= 1 by construction) and
still-running jobs' checkpoint-banked progress counts (the engine reports
it at the horizon).

STEADY-STATE WINDOW
-------------------
``warmup_frac`` / ``drain_frac`` (default 0.1 / 0.1) define the window
``[w0, w1] = [warmup_frac * horizon, horizon - drain_frac * horizon]``.
Every integral is maintained BOTH over the full run and clipped to the
window; point events (preemptions, failures, drains, flush samples) belong
to the window iff ``w0 <= t <= w1`` (closed interval).  Job-level window
membership is decided downstream (``fleetsim.metrics.summary``) per
metric: queue waits by ``first_start_t``, JCT/ETTR by ``end_t``.

DERIVED RATIOS (reported by :meth:`integral_report` / the summary layer)
- occupancy       = ∫ allocated dt / ∫ healthy dt
- allocation rate = ∫ allocated dt / ∫ total dt  (total incl. failed/drained)
- goodput         = Σ productive_chip_s / ∫ allocated dt
- ETTR (per job)  = productive seconds / wall seconds holding an
  allocation (stints incl. restart overhead and preemption grace windows)
- replica availability = ∫ running INFER_REPLICA jobs dt / ∫ desired dt,
  where a replica is "desired" from submission until its terminal event.

UNITS
-----
All ``t`` / ``*_us`` are int microseconds since sim epoch; ``*_chip_s``
are float chip-seconds; integrals are stored internally as exact int
chip-microseconds and reported as float chip-seconds.

INVARIANTS
----------
- Sink methods never mutate the engine's ``job`` / ``fleet`` objects.
- Emission order is deterministic: job rows sort by ``(submit_t, id)``,
  keyed reports sort keys lexicographically, timeseries rows are in flush
  order.
- Fleet-derived statics (total chips, node quantum = min leaf chips,
  level list, initial healthy count) come from the ``fleet`` passed at
  construction, or are learned on the first ``flush``; in the latter case
  the healthy integral is back-corrected exactly (valid because every
  healthy-capacity change since t=0 arrives as a ``healthy_delta`` call).
- ``preemptions by trigger`` counts ``job_preempted`` calls keyed by the
  engine's trigger string ("scheduler" | "maintenance" |
  "failure_second_order"); node-failure gang kills are counted separately
  (one per victim) as ``failure_kills`` and surfaced by the summary layer
  under the "node_failure" trigger.  Node failures are additionally
  bucketed by the engine-sampled ``cause`` label (DESIGN §8 mix).
- The ``stranded_chips`` timeseries uses the smallest gang quantum
  (DESIGN §9): the minimum gang chip count observed among submitted jobs
  so far, capped at the node quantum (before any job arrives, the node
  quantum) — free chips a gang of that size cannot use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..model import Allocation, Job, JobStatus, NodeState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Scenario
    from ..fleet.tree import FleetTree

__all__ = ["TimeWeighted", "MetricsCollector"]

_US = 1_000_000  # microseconds per second

#: JobClass name whose jobs feed the replica-availability integrals.
_REPLICA_CLASS = "INFER_REPLICA"


class TimeWeighted:
    """Exact time-weighted integral of an integer-valued step function.

    ``value`` is the current level (int); :meth:`add` advances the
    integral to ``t`` and then applies a delta.  Two accumulators are
    kept: the full-run integral and the integral clipped to the
    steady-state ``window`` — both in exact int value-microseconds
    (chip-µs, job-µs, ...).

    UNITS: times int microseconds; integrals int value-µs.
    INVARIANTS: the accumulator epoch is t=0; times passed to
    :meth:`add`/:meth:`advance` must be non-decreasing (earlier times are
    clamped, never accumulated negatively).  Reads are non-mutating.
    """

    __slots__ = ("value", "_last", "_acc", "_acc_w", "_w0", "_w1")

    def __init__(self, window: tuple[int, int], value: int = 0):
        self._w0, self._w1 = window
        self.value = value
        self._last = 0
        self._acc = 0
        self._acc_w = 0

    def advance(self, t: int) -> None:
        """Accumulate ``value`` over ``[last, t]`` and move the cursor."""
        if t <= self._last:
            return
        self._acc += self.value * (t - self._last)
        lo = max(self._last, self._w0)
        hi = min(t, self._w1)
        if hi > lo:
            self._acc_w += self.value * (hi - lo)
        self._last = t

    def add(self, t: int, delta: int) -> None:
        """Advance to ``t`` then change the level by ``delta``."""
        self.advance(t)
        self.value += delta

    def retro_add(self, delta: int) -> None:
        """Apply ``delta`` retroactively since t=0 (exact back-correction
        for a constant offset that was unknown at construction, e.g. the
        initial healthy-chip count learned at the first flush)."""
        self._acc += delta * self._last
        hi = min(self._last, self._w1)
        if hi > self._w0:
            self._acc_w += delta * (hi - self._w0)
        self.value += delta

    def integral(self, upto: int) -> int:
        """Full-run integral over ``[0, upto]`` in value-µs (non-mutating;
        assumes no further changes between the cursor and ``upto``)."""
        return self._acc + self.value * max(0, upto - self._last)

    def window_integral(self, upto: int) -> int:
        """Window-clipped integral over ``[w0, w1] ∩ [0, upto]``."""
        lo = max(self._last, self._w0)
        hi = min(upto, self._w1)
        extra = self.value * (hi - lo) if hi > lo else 0
        return self._acc_w + extra


@dataclass(slots=True)
class _JobAcc:
    """Accumulating per-job record (identity + lifecycle counters).

    UNITS: ``*_t`` int µs; ``running_elapsed_us`` int µs (wall time
    holding an allocation, incl. restart overhead and grace windows);
    ``*_chip_s`` float chip-seconds.
    """

    job_id: str
    tenant: str
    job_class: str
    tier: str
    #: Workload-class label of the generating source (``Job.source_class``;
    #: None for hand-built/trace jobs).  Distinguishes e.g. a closed-loop
    #: best-effort backlog class from the open-loop class sharing its
    #: JobClass — the summary layer keys honest per-class stats on it.
    source_class: str | None
    chips: int
    chip_type: str | None
    submit_t: int
    status: str
    first_start_t: int | None = None
    end_t: int | None = None
    n_starts: int = 0
    #: Distinct segment-level domains of the LATEST placement (v0.2,
    #: segmented gangs; 1 for plain gangs) — None until first start.
    n_domains_spanned: int | None = None
    n_node_failures: int = 0
    preempts: dict[str, int] = field(default_factory=dict)
    productive_chip_s: float = 0.0
    lost_chip_s: float = 0.0
    running_elapsed_us: int = 0
    alloc_start: int | None = None
    queued: bool = False


class MetricsCollector:
    """The real metrics sink: integrals, records, counters, timeseries.

    Construct with the simulation ``horizon_us`` and (ideally) the
    :class:`~fleetsim.fleet.tree.FleetTree`, so the initial healthy count
    and fleet statics are known from t=0; without a fleet they are learned
    at the first ``flush`` (see module docstring).  ``warmup_frac`` and
    ``drain_frac`` configure the steady-state window.

    INVARIANTS: this class only *receives* engine callbacks — it owns no
    randomness and no wall-clock; identical call sequences produce
    identical reports.
    """

    def __init__(
        self,
        horizon_us: int,
        *,
        fleet: "FleetTree | None" = None,
        warmup_frac: float = 0.1,
        drain_frac: float = 0.1,
    ):
        if horizon_us <= 0:
            raise ValueError(f"horizon_us must be positive, got {horizon_us}")
        warmup_frac = float(warmup_frac)
        drain_frac = float(drain_frac)
        if warmup_frac < 0 or drain_frac < 0 or warmup_frac + drain_frac >= 1.0:
            raise ValueError(
                "steady-state fractions must satisfy 0 <= warmup_frac,"
                f" 0 <= drain_frac, warmup_frac + drain_frac < 1;"
                f" got {warmup_frac}/{drain_frac}"
            )
        self._horizon = int(horizon_us)
        self._warmup_frac = warmup_frac
        self._drain_frac = drain_frac
        w0 = round(self._horizon * warmup_frac)
        w1 = self._horizon - round(self._horizon * drain_frac)
        self._w: tuple[int, int] = (w0, w1)

        # Time-weighted integrals.
        self._alloc = TimeWeighted(self._w)
        self._healthy = TimeWeighted(self._w)
        self._alloc_by_type: dict[str, TimeWeighted] = {}
        self._alloc_by_class: dict[str, TimeWeighted] = {}
        self._alloc_by_tenant: dict[str, TimeWeighted] = {}
        self._pending_by_class: dict[str, TimeWeighted] = {}
        self._replica_desired = TimeWeighted(self._w)
        self._replica_running = TimeWeighted(self._w)

        # Event-sourced state.
        self._recs: dict[str, _JobAcc] = {}
        self._preempts_full: dict[str, int] = {}
        self._preempts_win: dict[str, int] = {}
        self._failure_kills_full = 0
        self._failure_kills_win = 0
        self._node_failures_full = 0
        self._node_failures_win = 0
        self._causes_full: dict[str, int] = {}
        self._causes_win: dict[str, int] = {}
        self._node_repairs_full = 0
        self._node_repairs_win = 0
        self._drains_full = 0
        self._drains_win = 0
        # Productive chip-seconds from job_progress (full + window-spread).
        self._productive_full = 0.0
        self._productive_win = 0.0
        self._lost_full = 0.0
        # Smallest gang chip count seen among submitted jobs (stranded
        # quantum, DESIGN §9); None until the first submission.
        self._min_gang: int | None = None

        # Flush-sampled timeseries + fragmentation stats.
        self._ts: list[dict[str, Any]] = []
        self._frag_full: dict[str, list[float]] = {}  # level -> [sum, max, n]
        self._frag_win: dict[str, list[float]] = {}

        # Fleet statics (filled at construction or first flush).
        self._statics_ready = False
        self._total_chips = 0
        self._node_quantum = 0
        self._fleet_levels: tuple[str, ...] = ()
        if fleet is not None:
            self._init_fleet_statics(fleet)

    # -- construction helpers -------------------------------------------

    @classmethod
    def from_scenario(
        cls, scenario: "Scenario", fleet: "FleetTree | None" = None
    ) -> "MetricsCollector":
        """Build from a scenario: horizon from ``sim``, window fractions
        from optional ``outputs`` keys ``warmup_frac`` / ``drain_frac``
        (they land in ``OutputsConfig.extra``)."""
        extra = scenario.outputs.extra or {}
        return cls(
            scenario.sim.horizon_us,
            fleet=fleet,
            warmup_frac=float(extra.get("warmup_frac", 0.1)),
            drain_frac=float(extra.get("drain_frac", 0.1)),
        )

    def _init_fleet_statics(self, fleet: "FleetTree") -> None:
        """Learn total chips, node quantum, level list, and correct the
        healthy integral for the (constant) initial healthy count."""
        if self._statics_ready:
            return
        self._statics_ready = True
        total = 0
        healthy = 0
        quantum: int | None = None
        for lid in fleet.leaves():
            leaf = fleet.domain(lid)
            total += leaf.chips
            if leaf.state is NodeState.HEALTHY:
                healthy += leaf.chips
            quantum = leaf.chips if quantum is None else min(quantum, leaf.chips)
        self._total_chips = total
        self._node_quantum = quantum or 0
        self._fleet_levels = fleet.levels()
        offset = healthy - self._healthy.value
        if offset:
            self._healthy.retro_add(offset)

    # -- small internal helpers -----------------------------------------

    def _in_window(self, t: int) -> bool:
        return self._w[0] <= t <= self._w[1]

    def _stranded_quantum(self) -> int:
        """Smallest gang quantum (DESIGN §9): min gang chips observed so
        far, capped at the node quantum; the node quantum before any job
        has been submitted."""
        if self._min_gang is None:
            return self._node_quantum
        return min(self._node_quantum, self._min_gang) if self._node_quantum else 0

    def _tw(self, table: dict[str, TimeWeighted], key: str) -> TimeWeighted:
        tw = table.get(key)
        if tw is None:
            tw = table[key] = TimeWeighted(self._w)
        return tw

    def _rec(self, job: Job) -> _JobAcc:
        rec = self._recs.get(job.id)
        if rec is None:
            rec = _JobAcc(
                job_id=job.id,
                tenant=job.tenant,
                job_class=job.job_class.name,
                tier=job.tier.name,
                source_class=getattr(job, "source_class", None),
                chips=sum(g.chips for g in job.gangs),
                chip_type=job.gangs[0].chip_type if job.gangs else None,
                submit_t=job.submit_t,
                status=job.status.name,
            )
            self._recs[job.id] = rec
        return rec

    def _close_alloc(self, rec: _JobAcc, t: int) -> None:
        if rec.alloc_start is None:
            return
        rec.running_elapsed_us += t - rec.alloc_start
        rec.alloc_start = None
        self._tw(self._alloc_by_class, rec.job_class).add(t, -rec.chips)
        self._tw(self._alloc_by_tenant, rec.tenant).add(t, -rec.chips)
        if rec.job_class == _REPLICA_CLASS:
            self._replica_running.add(t, -1)

    def _dequeue(self, rec: _JobAcc, t: int) -> None:
        if rec.queued:
            rec.queued = False
            self._tw(self._pending_by_class, rec.job_class).add(t, -1)

    # -- MetricsSink protocol -------------------------------------------

    def job_submitted(self, job: Job, t: int) -> None:
        rec = self._rec(job)
        rec.status = job.status.name
        for gang in job.gangs:
            if gang.chips > 0 and (
                self._min_gang is None or gang.chips < self._min_gang
            ):
                self._min_gang = gang.chips
        if rec.job_class == _REPLICA_CLASS:
            self._replica_desired.add(t, 1)

    def job_admitted(self, job: Job, t: int) -> None:
        rec = self._rec(job)
        rec.status = "ADMITTED"
        if not rec.queued:
            rec.queued = True
            self._tw(self._pending_by_class, rec.job_class).add(t, 1)

    def job_started(self, job: Job, alloc: Allocation, t: int) -> None:
        rec = self._rec(job)
        rec.status = "RUNNING"
        rec.n_starts += 1
        if rec.first_start_t is None:
            rec.first_start_t = t
        # Segmented placements record their span in GangAlloc.attrs; a
        # plain gang always spans exactly one domain.  Latest start wins
        # (restarts may land elsewhere).
        rec.n_domains_spanned = max(
            (int(g.attrs.get("n_domains_spanned", 1)) for g in alloc.gangs),
            default=1,
        )
        self._dequeue(rec, t)
        if rec.alloc_start is None:
            rec.alloc_start = t
            self._tw(self._alloc_by_class, rec.job_class).add(t, rec.chips)
            self._tw(self._alloc_by_tenant, rec.tenant).add(t, rec.chips)
            if rec.job_class == _REPLICA_CLASS:
                self._replica_running.add(t, 1)

    def job_preempted(self, job: Job, t: int, trigger: str) -> None:
        rec = self._rec(job)
        rec.status = "PREEMPTED"
        rec.preempts[trigger] = rec.preempts.get(trigger, 0) + 1
        self._preempts_full[trigger] = self._preempts_full.get(trigger, 0) + 1
        if self._in_window(t):
            self._preempts_win[trigger] = self._preempts_win.get(trigger, 0) + 1
        # The allocation stays live through the grace window; it is closed
        # by the matching job_requeued (REQUEUE) or job_finished (CANCEL).

    def job_requeued(self, job: Job, t: int) -> None:
        rec = self._rec(job)
        rec.status = "PENDING"
        self._close_alloc(rec, t)
        if not rec.queued:
            rec.queued = True
            self._tw(self._pending_by_class, rec.job_class).add(t, 1)

    def job_progress(
        self,
        job: Job,
        start_us: int,
        end_us: int,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None:
        """Accumulate a settled stint's surviving/lost work — into the
        fleet aggregates AND the per-job record, so jobs still RUNNING at
        the horizon carry their checkpoint-banked progress in
        ``jobs.parquet`` (the engine reports it at the horizon; deltas
        sum exactly to the terminal totals, which ``job_finished``
        re-asserts authoritatively).  The window share spreads the delta
        uniformly over ``[start_us, end_us]`` (point credit when the
        interval is empty)."""
        productive = float(productive_chip_s)
        lost = float(lost_chip_s)
        self._productive_full += productive
        self._lost_full += lost
        rec = self._recs.get(job.id)
        if rec is not None:
            rec.productive_chip_s += productive
            rec.lost_chip_s += lost
        w0, w1 = self._w
        if end_us > start_us:
            lo = max(start_us, w0)
            hi = min(end_us, w1)
            if hi > lo:
                self._productive_win += productive * (hi - lo) / (end_us - start_us)
        elif w0 <= end_us <= w1:
            self._productive_win += productive

    def job_finished(
        self,
        job: Job,
        t: int,
        status: JobStatus,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None:
        rec = self._rec(job)
        if rec.end_t is not None:  # double-terminal guard
            return
        self._close_alloc(rec, t)
        self._dequeue(rec, t)
        rec.end_t = t
        rec.status = status.name
        rec.productive_chip_s = float(productive_chip_s)
        rec.lost_chip_s = float(lost_chip_s)
        if rec.job_class == _REPLICA_CLASS:
            self._replica_desired.add(t, -1)

    def node_failed(
        self,
        node_id: str,
        t: int,
        killed_alloc_ids: Sequence[str],
        cause: str = "unknown",
    ) -> None:
        self._node_failures_full += 1
        self._causes_full[cause] = self._causes_full.get(cause, 0) + 1
        win = self._in_window(t)
        if win:
            self._node_failures_win += 1
            self._causes_win[cause] = self._causes_win.get(cause, 0) + 1
        for jid in killed_alloc_ids:
            rec = self._recs.get(jid)
            if rec is not None:
                rec.n_node_failures += 1
            self._failure_kills_full += 1
            if win:
                self._failure_kills_win += 1

    def node_repaired(self, node_id: str, t: int) -> None:
        self._node_repairs_full += 1
        if self._in_window(t):
            self._node_repairs_win += 1

    def node_drain_started(self, node_id: str, t: int) -> None:
        self._drains_full += 1
        if self._in_window(t):
            self._drains_win += 1

    def chips_allocated(self, n: int, chip_type: str, t: int) -> None:
        self._alloc.add(t, n)
        self._tw(self._alloc_by_type, chip_type).add(t, n)

    def chips_freed(self, n: int, chip_type: str, t: int) -> None:
        self._alloc.add(t, -n)
        self._tw(self._alloc_by_type, chip_type).add(t, -n)

    def healthy_delta(self, n_chips: int, chip_type: str, t: int) -> None:
        self._healthy.add(t, n_chips)

    def flush(
        self, t: int, fleet: "FleetTree", n_pending: int, n_running: int
    ) -> None:
        """Sample the O(1)-readable state into one timeseries row and
        accumulate fragmentation stats (mean/max per level)."""
        self._init_fleet_statics(fleet)
        alloc_int = self._alloc.integral(t)
        healthy_int = self._healthy.integral(t)
        total_int = self._total_chips * t
        cum_preempt = sum(self._preempts_full.values())
        row: dict[str, Any] = {
            "t_us": t,
            "allocated_chips": self._alloc.value,
            "healthy_chips": self._healthy.value,
            "pending_jobs": n_pending,
            "running_jobs": n_running,
            "occupancy_to_date": alloc_int / healthy_int if healthy_int > 0 else 0.0,
            "allocation_rate_to_date": alloc_int / total_int if total_int > 0 else 0.0,
            "goodput_to_date": (
                self._productive_full / (alloc_int / _US) if alloc_int > 0 else 0.0
            ),
            "cum_preemptions": cum_preempt,
            "cum_failure_kills": self._failure_kills_full,
            "cum_node_failures": self._node_failures_full,
            "stranded_chips": (
                fleet.stranded_chips(self._stranded_quantum())
                if self._stranded_quantum() > 0
                else 0
            ),
        }
        in_win = self._in_window(t)
        for level in self._fleet_levels:
            largest = fleet.largest_placeable(level)
            frag = fleet.fragmentation_index(level)
            row[f"largest_placeable_{level}"] = largest
            row[f"frag_index_{level}"] = frag
            for table, hit in ((self._frag_full, True), (self._frag_win, in_win)):
                if not hit:
                    continue
                acc = table.get(level)
                if acc is None:
                    table[level] = [frag, frag, 1]
                else:
                    acc[0] += frag
                    acc[1] = max(acc[1], frag)
                    acc[2] += 1
        self._ts.append(row)

    # -- read-side API (consumed by fleetsim.metrics.summary) -----------

    @property
    def horizon_us(self) -> int:
        return self._horizon

    @property
    def window(self) -> tuple[int, int]:
        """The steady-state window ``(w0, w1)`` in int microseconds."""
        return self._w

    @property
    def warmup_frac(self) -> float:
        return self._warmup_frac

    @property
    def drain_frac(self) -> float:
        return self._drain_frac

    def preempt_triggers(self) -> tuple[str, ...]:
        """All preemption trigger strings observed, sorted."""
        return tuple(sorted(self._preempts_full))

    def job_rows(self) -> list[dict[str, Any]]:
        """One dict per job, sorted by ``(submit_t, job_id)``.

        Columns: ``job_id, tenant, job_class, tier, source_class, chips,
        chip_type, submit_t_us, first_start_t_us, end_t_us, status,
        n_starts, n_restarts, n_domains_spanned, n_preemptions,
        n_preempt_<trigger>..., n_node_failures, productive_chip_s,
        lost_chip_s, running_elapsed_s, queue_wait_s, jct_s, ettr``.
        ``productive_chip_s``/``lost_chip_s`` accumulate as stints settle
        (still-running jobs show their checkpoint-banked progress after
        the horizon flush, not 0.0).
        ``n_domains_spanned`` is the latest placement's distinct
        segment-level domain count (1 for plain gangs, ``None`` before
        the first start).  Nullable fields are ``None`` when undefined
        (never started / not terminal).
        ``jct_s`` is filled for every terminal job; JCT *statistics*
        downstream restrict to COMPLETED.  Still-open allocations are
        clamped at the horizon for ``running_elapsed_s``.
        """
        triggers = self.preempt_triggers()
        rows: list[dict[str, Any]] = []
        for rec in sorted(self._recs.values(), key=lambda r: (r.submit_t, r.job_id)):
            elapsed_us = rec.running_elapsed_us
            if rec.alloc_start is not None:
                elapsed_us += max(0, self._horizon - rec.alloc_start)
            elapsed_s = elapsed_us / _US
            terminal = rec.end_t is not None
            queue_wait_s = (
                (rec.first_start_t - rec.submit_t) / _US
                if rec.first_start_t is not None
                else None
            )
            jct_s = (rec.end_t - rec.submit_t) / _US if terminal else None
            ettr = (
                (rec.productive_chip_s / rec.chips) / elapsed_s
                if terminal and elapsed_s > 0 and rec.chips > 0
                else None
            )
            row: dict[str, Any] = {
                "job_id": rec.job_id,
                "tenant": rec.tenant,
                "job_class": rec.job_class,
                "tier": rec.tier,
                "source_class": rec.source_class,
                "chips": rec.chips,
                "chip_type": rec.chip_type,
                "submit_t_us": rec.submit_t,
                "first_start_t_us": rec.first_start_t,
                "end_t_us": rec.end_t,
                "status": rec.status,
                "n_starts": rec.n_starts,
                "n_restarts": max(0, rec.n_starts - 1),
                "n_domains_spanned": rec.n_domains_spanned,
                "n_preemptions": sum(rec.preempts.values()),
            }
            for trig in triggers:
                row[f"n_preempt_{trig}"] = rec.preempts.get(trig, 0)
            row.update(
                {
                    "n_node_failures": rec.n_node_failures,
                    "productive_chip_s": rec.productive_chip_s,
                    "lost_chip_s": rec.lost_chip_s,
                    "running_elapsed_s": elapsed_s,
                    "queue_wait_s": queue_wait_s,
                    "jct_s": jct_s,
                    "ettr": ettr,
                }
            )
            rows.append(row)
        return rows

    def timeseries_rows(self) -> list[dict[str, Any]]:
        """Flush samples in flush order (shallow copies)."""
        return [dict(r) for r in self._ts]

    def integral_report(self) -> dict[str, dict[str, Any]]:
        """All time-weighted integrals as float (chip-/job-)seconds, for
        both scopes: ``{"full": {...}, "window": {...}}``.

        ``total_chip_s`` is 0.0 when the fleet was never seen (no fleet at
        construction and no flush) — consumers must treat a zero
        denominator as "unknown".
        """
        upto = self._horizon
        w0, w1 = self._w
        out: dict[str, dict[str, Any]] = {}
        for scope, full in (("full", True), ("window", False)):
            dur_us = upto if full else (w1 - w0)

            def rd(tw: TimeWeighted) -> float:
                raw = tw.integral(upto) if full else tw.window_integral(upto)
                return raw / _US

            out[scope] = {
                "duration_s": dur_us / _US,
                "allocated_chip_s": rd(self._alloc),
                "healthy_chip_s": rd(self._healthy),
                "total_chip_s": self._total_chips * dur_us / _US,
                "productive_chip_s": (
                    self._productive_full if full else self._productive_win
                ),
                "allocated_chip_s_by_type": {
                    k: rd(tw) for k, tw in sorted(self._alloc_by_type.items())
                },
                "allocated_chip_s_by_class": {
                    k: rd(tw) for k, tw in sorted(self._alloc_by_class.items())
                },
                "allocated_chip_s_by_tenant": {
                    k: rd(tw) for k, tw in sorted(self._alloc_by_tenant.items())
                },
                "pending_job_s_by_class": {
                    k: rd(tw) for k, tw in sorted(self._pending_by_class.items())
                },
                "replica_running_s": rd(self._replica_running),
                "replica_desired_s": rd(self._replica_desired),
            }
        return out

    def event_counts(self) -> dict[str, dict[str, Any]]:
        """Discrete event counters for both scopes (window = closed
        ``[w0, w1]`` membership of the event time)."""
        return {
            "full": {
                "preemptions_by_trigger": dict(sorted(self._preempts_full.items())),
                "failure_kills": self._failure_kills_full,
                "node_failures": self._node_failures_full,
                "node_failures_by_cause": dict(sorted(self._causes_full.items())),
                "node_repairs": self._node_repairs_full,
                "drains_started": self._drains_full,
            },
            "window": {
                "preemptions_by_trigger": dict(sorted(self._preempts_win.items())),
                "failure_kills": self._failure_kills_win,
                "node_failures": self._node_failures_win,
                "node_failures_by_cause": dict(sorted(self._causes_win.items())),
                "node_repairs": self._node_repairs_win,
                "drains_started": self._drains_win,
            },
        }

    def frag_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        """Fragmentation-index stats over flush samples, per level:
        ``{"full"|"window": {level: {"mean", "max", "n_samples"}}}``.
        Levels with no in-scope samples are absent."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for scope, table in (("full", self._frag_full), ("window", self._frag_win)):
            out[scope] = {
                level: {
                    "mean": acc[0] / acc[2],
                    "max": acc[1],
                    "n_samples": int(acc[2]),
                }
                for level, acc in sorted(table.items())
            }
        return out
