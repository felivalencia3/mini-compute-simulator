"""Tiered-priority preempting scheduler (DESIGN §5, §7; Borg band semantics).

fleetsim v0.1's second policy: pending jobs are served in **band order**
(higher :class:`~fleetsim.model.Tier` first), and a pending job that
cannot be placed may evict strictly-lower-band running work via
``Preempt`` actions.  The engine applies grace windows, checkpoint-loss
accounting, and re-validates every guardrail; this policy simply never
emits an action the engine would refuse.

PINNED SEMANTICS
----------------
- **Ordering**: the pending queue is scanned by ``(tier DESC,
  submit_time ASC, id ASC)``.  Requeued jobs keep their original
  ``submit_t`` (engine invariant), so preempted work re-enters at its
  original priority.  Jobs that fit are placed; unplaceable jobs that
  cannot trigger preemption are skipped (best-effort — a stuck job with
  nothing to evict never blocks the jobs behind it).
- **Victim eligibility**: a running job is a preemption candidate for
  pending job ``J`` iff its tier is STRICTLY below ``J``'s band, it is
  ``preemptible``, it is not MONITORING (structurally implied by the
  strict band test, kept explicit), and its current stint has run at
  least ``min_runtime_s``.  PROD never evicts PROD: strictly-lower-band
  means a PROD victim requires a MONITORING preemptor, which the engine
  permits (Borg's monitoring band sits above prod).
- **Domain scan**: victims must free capacity where ``J`` could actually
  place — inside ONE domain at ``J.within``'s level, or at level
  ``"cluster"`` when ``J`` is unconstrained (gangs never span cluster
  roots).  Domains are scanned in ascending id order; the FIRST domain
  whose ``free + preemptable >= J.chips`` wins.  A domain is skipped when
  its ``chip_type`` does not match the job's (a homogeneous domain of the
  wrong type can never host the gang; heterogeneous domains — interior
  ``chip_type is None`` — are a documented v0.1 preemption blind spot).
- **Greedy victim order** within the winning domain: ``(tier ASC,
  attained_service_chip_s ASC, submit_time DESC, id ASC)`` — lowest band
  first, least attained service first, youngest first — accumulating
  until ``freed + free >= J.chips``.  If the domain's free chips alone
  already cover the request (placement failed on shape, e.g. a
  whole-node gang facing sub-node fragmentation), preempting cannot be
  shown to help by this accounting and nothing is emitted.
- **The pending job is NOT placed in the preempting wake**: REQUEUE
  victims hold their chips until the checkpoint grace elapses, so ``J``
  places at a later wake.  After committing a preemption the scan STOPS
  for this wake — lower-priority pending jobs must not steal the free
  chips counted toward ``J``'s claim (``J`` outranks them, so it re-scans
  first at every subsequent wake until it fits).
- **Storm cap**: at most ``max_preemptions_per_wake`` Preempts are
  emitted per wake.  A victim set larger than the cap is worked through
  incrementally across consecutive wakes (the stop-scan rule protects the
  partially-freed chips in between).  ``0`` disables preemption entirely,
  reducing the policy to best-effort priority-FIFO.
- **Statefulness**: the scheduler records ``(stint start, placement)``
  for every ``Place`` it emits — the engine starts stints exactly at the
  emitting wake's timestamp, and every stint begins with a scheduler
  ``Place``, so the records are exact.  They provide the per-stint
  min-runtime guard and the victim-location map (the ClusterView exposes
  neither).  Use a FRESH scheduler instance per Simulator run.
- **Id convention**: leaf-in-domain containment is tested by path prefix
  (``leaf == domain or leaf.startswith(domain + "/")``), matching the
  hierarchical ids produced by :func:`fleetsim.fleet.build.build_fleet`.

UNITS: ``view.now`` / ``submit_time`` are int microseconds; every
``*_s`` field is float seconds; ``attained_service_chip_s`` is
chip-seconds.

INVARIANTS: deterministic — all scans are over sorted sequences, no
randomness, no wall clock; emitted ``Preempt`` actions always carry
``preemptor`` so the engine can enforce band rules; a single wake emits
either placements, or placements followed by one job's (possibly
truncated) victim set, never both placements and preemptions for the
same job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..fleet.tree import Placement
from ..model import PreemptMode, Tier
from .base import (
    Action,
    ClusterView,
    JobView,
    Place,
    PlacementPolicy,
    Preempt,
    Scheduler,
    register,
)
from .placement import FirstFit

__all__ = ["TieredPriorityScheduler"]

#: YAML/param spellings of the two preemption modes (DESIGN §13:
#: ``params: {preempt: requeue}``).
_PREEMPT_MODES: dict[str, PreemptMode] = {
    "requeue": PreemptMode.REQUEUE,
    "cancel": PreemptMode.CANCEL,
}


@dataclass(frozen=True, slots=True)
class _Stint:
    """Scheduler-side record of a stint this scheduler started.

    ``start_us`` (int microseconds) equals the engine's stint start — the
    timestamp of the wake whose ``Place`` action began the stint.
    ``placement`` is the immutable gang placement, used to locate the job
    in the domain tree when hunting victims.
    """

    start_us: int
    placement: Placement


def _placement_within(placement: Placement, domain_id: str) -> bool:
    """True iff every leaf of ``placement`` lies under ``domain_id``
    (path-prefix containment over build_fleet's hierarchical ids)."""
    prefix = domain_id + "/"
    return all(
        lid == domain_id or lid.startswith(prefix) for lid, _ in placement.leaves
    )


@register("tiered_priority")
class TieredPriorityScheduler(Scheduler):
    """Borg-band priority scheduler with REQUEUE/CANCEL preemption.

    Parameters (all reachable via ``scheduler.params``):

    - ``placement``: the :class:`~fleetsim.schedulers.base.PlacementPolicy`
      (default :class:`~fleetsim.schedulers.placement.FirstFit`);
      programmatic only — not expressible in YAML.
    - ``preempt``: ``"requeue"`` (default) or ``"cancel"`` — the
      :class:`~fleetsim.model.PreemptMode` used for every eviction.  A
      ``PreemptMode`` value is also accepted.
    - ``max_preemptions_per_wake``: cap on Preempt actions per wake
      (default 50); ``0`` disables preemption.
    """

    def __init__(
        self,
        placement: PlacementPolicy | None = None,
        preempt: str | PreemptMode = "requeue",
        max_preemptions_per_wake: int = 50,
    ):
        self.placement: PlacementPolicy = (
            placement if placement is not None else FirstFit()
        )
        if isinstance(preempt, PreemptMode):
            self.preempt_mode = preempt
        else:
            mode = _PREEMPT_MODES.get(str(preempt).lower())
            if mode is None:
                raise ValueError(
                    f"preempt must be one of {sorted(_PREEMPT_MODES)},"
                    f" got {preempt!r}"
                )
            self.preempt_mode = mode
        self.max_preemptions_per_wake = int(max_preemptions_per_wake)
        if self.max_preemptions_per_wake < 0:
            raise ValueError(
                "max_preemptions_per_wake must be >= 0,"
                f" got {max_preemptions_per_wake!r}"
            )
        #: job id -> stint record for jobs this scheduler placed; pruned to
        #: the running set at each wake (grace-window and finished jobs
        #: drop out; re-placed jobs get fresh records).
        self._stints: dict[str, _Stint] = {}

    # ------------------------------------------------------------------

    def schedule(self, view: ClusterView) -> list[Action]:
        running = view.running()
        running_ids = {jv.id for jv in running}
        self._stints = {
            jid: st for jid, st in self._stints.items() if jid in running_ids
        }
        actions: list[Action] = []
        order = sorted(
            view.pending(), key=lambda j: (-int(j.tier), j.submit_time, j.id)
        )
        for job in order:
            placement = view.find_placement(job, self.placement)
            if placement is not None:
                actions.append(Place(job.id, placement))
                self._stints[job.id] = _Stint(
                    start_us=view.now, placement=placement
                )
                continue
            if self.max_preemptions_per_wake <= 0:
                continue  # preemption disabled: best-effort priority-FIFO
            victims = self._plan_preemption(job, view, running)
            if not victims:
                continue  # nothing this job may evict; let others through
            for victim in victims[: self.max_preemptions_per_wake]:
                actions.append(
                    Preempt(victim.id, self.preempt_mode, preemptor=job.id)
                )
            break  # stop-scan: protect the free chips counted for `job`
        return actions

    # ------------------------------------------------------------------

    def _plan_preemption(
        self, job: JobView, view: ClusterView, running: Sequence[JobView]
    ) -> list[JobView]:
        """The full (uncapped) victim set for ``job``, or ``[]``.

        Scans candidate domains in ascending id order and returns the
        greedy victim list of the first domain that can yield enough
        chips; empty when no domain can (or when free chips alone already
        cover the request, i.e. preemption provably would not help under
        chip-count accounting).
        """
        now = view.now
        eligible: list[tuple[JobView, _Stint]] = []
        for victim in running:
            if victim.tier >= job.tier or victim.tier is Tier.MONITORING:
                continue  # strictly-lower-band only; MONITORING untouchable
            if not victim.preemptible:
                continue
            stint = self._stints.get(victim.id)
            if stint is None:
                continue  # unknown stint: cannot verify guards or location
            if (
                victim.min_runtime_s > 0.0
                and (now - stint.start_us) / 1e6 < victim.min_runtime_s
            ):
                continue  # young stint: engine would refuse (doomed action)
            eligible.append((victim, stint))
        if not eligible:
            return []

        level = job.within if job.within is not None else "cluster"
        for dom in view.domains(level):  # ascending id order
            chip_type = job.chip_type if job.chip_type is not None else dom.chip_type
            if chip_type is None or dom.chip_type != chip_type:
                continue  # wrong-typed or heterogeneous domain (v0.1 skip)
            candidates = [
                victim
                for victim, stint in eligible
                if stint.placement.chip_type == chip_type
                and _placement_within(stint.placement, dom.id)
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda v: (
                    int(v.tier),
                    v.attained_service_chip_s,
                    -v.submit_time,
                    v.id,
                )
            )
            freed = dom.free_chips
            chosen: list[JobView] = []
            for victim in candidates:
                if freed >= job.chips:
                    break
                chosen.append(victim)
                freed += victim.chips
            if chosen and freed >= job.chips:
                return chosen
        return []
