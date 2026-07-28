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
- **Claims memory** (v0.2): the scheduler remembers each emitted victim
  set as an in-flight CLAIM for its preemptor.  While any claimed victim
  is still in its grace window (invisible in both ``pending()`` and
  ``running()``), the preemptor plans NO further victims — a grace
  window longer than the scheduler round must not trigger a second,
  redundant eviction — and the scan stops at it (same chip-protection
  rationale as the emitting wake).  A claim resolves when its victims
  reappear (requeued/re-placed), when the preemptor places, or when the
  view reports no claimed victim in grace (``view.graced_job_ids()``,
  exact; CANCELed victims resolve immediately).  Consequence for the
  storm cap: a victim set larger than the cap is now worked through one
  tranche per GRACE WINDOW rather than per wake when the grace exceeds
  the round.
- **Feasibility dry-run** (v0.2): chip-count accounting alone cannot see
  node shapes (a 7-chip sub-node victim frees no whole node) or leaf
  health (a victim on a DRAINING node frees nothing).  Every planned
  victim set is therefore verified with ``view.reclaim_feasible`` — the
  engine's real placement search with the victims' chips hypothetically
  freed — before being emitted.  If the chip-count plan is insufficient,
  it is extended with further eligible victims and pruned back to an
  inclusion-minimal feasible set (dropping least-preferred victims
  first); if no eligible set can make the job placeable, NOTHING is
  emitted — no eviction whose freed chips provably cannot host the job
  (this kills the evict/re-place thrash loop and draining-node churn).
  Views without ``reclaim_feasible`` fall back to trusting the
  chip-count plan (v0.1 behavior).
- **Segmented reclaim** (v0.2): a pending job with ``segments`` plans
  victims PER SEGMENT-DOMAIN instead of per single domain — greedy over
  segment-level domains under each outer (``within``) domain, hosting
  each segment where it costs the fewest additional preemptions and
  aggregating one victim set across all segments.  This lets one large
  gang empty multiple pods of best-effort mice in a single wake.  Chip
  types must be pinned (unpinned segmented jobs are skipped).
- **Storm cap**: at most ``max_preemptions_per_wake`` Preempts are
  emitted per wake (default 512 — sized for multi-pod segmented
  reclaim).  A victim set larger than the cap is worked through
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


def _chips_within(placement: Placement, domain_id: str) -> int:
    """Chips of ``placement`` sitting under ``domain_id`` (path-prefix
    containment) — a segmented victim may straddle several domains, and
    preempting it frees only its in-domain chips toward each."""
    prefix = domain_id + "/"
    return sum(
        chips
        for lid, chips in placement.leaves
        if lid == domain_id or lid.startswith(prefix)
    )


def _id_within(child_id: str, ancestor_id: str) -> bool:
    """True iff ``child_id`` equals or lies under ``ancestor_id``
    (path-prefix containment over build_fleet's hierarchical ids)."""
    return child_id == ancestor_id or child_id.startswith(ancestor_id + "/")


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
      (default 512 — raised in v0.2 so a segmented gang can empty
      multiple pods of best-effort mice in one wake); ``0`` disables
      preemption.
    """

    def __init__(
        self,
        placement: PlacementPolicy | None = None,
        preempt: str | PreemptMode = "requeue",
        max_preemptions_per_wake: int = 512,
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
        #: preemptor job id -> ids of its emitted victims still believed
        #: to be in their grace window (the in-flight CLAIM; see the
        #: module docstring's "Claims memory").
        self._claims: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------

    def schedule(self, view: ClusterView) -> list[Action]:
        running = view.running()
        pending = view.pending()
        running_ids = {jv.id for jv in running}
        pending_ids = {jv.id for jv in pending}
        self._stints = {
            jid: st for jid, st in self._stints.items() if jid in running_ids
        }
        self._prune_claims(view, pending_ids, running_ids)
        actions: list[Action] = []
        order = sorted(
            pending, key=lambda j: (-int(j.tier), j.submit_time, j.id)
        )
        for job in order:
            placement = view.find_placement(job, self.placement)
            if placement is not None:
                actions.append(Place(job.id, placement))
                self._claims.pop(job.id, None)  # claim (if any) resolved
                self._stints[job.id] = _Stint(
                    start_us=view.now, placement=placement
                )
                continue
            if self.max_preemptions_per_wake <= 0:
                continue  # preemption disabled: best-effort priority-FIFO
            if job.id in self._claims:
                # In-flight claim: this job's previous victims are still
                # in their grace window, holding the chips it is waiting
                # for.  Never evict a second, redundant set — and stop the
                # scan so lower-priority jobs cannot steal the chips being
                # freed (same rule as the emitting wake).
                break
            victims = self._plan_preemption(job, view, running)
            if not victims:
                continue  # nothing this job may evict; let others through
            emitted = victims[: self.max_preemptions_per_wake]
            for victim in emitted:
                actions.append(
                    Preempt(victim.id, self.preempt_mode, preemptor=job.id)
                )
            self._claims[job.id] = frozenset(v.id for v in emitted)
            break  # stop-scan: protect the free chips counted for `job`
        return actions

    def _prune_claims(
        self,
        view: ClusterView,
        pending_ids: set[str],
        running_ids: set[str],
    ) -> None:
        """Resolve in-flight claims (see the module docstring).

        A claim drops when its preemptor is no longer pending (placed or
        terminal) or when none of its victims remains in a grace window.
        Grace membership comes from ``view.graced_job_ids()`` when the
        view provides it (exact — the engine view does); otherwise a
        victim absent from BOTH pending and running is assumed graced (a
        conservative fallback for foreign views)."""
        if not self._claims:
            return
        graced_fn = getattr(view, "graced_job_ids", None)
        graced: set[str] | None = (
            set(graced_fn()) if graced_fn is not None else None
        )
        for pid in list(self._claims):
            if pid not in pending_ids:
                del self._claims[pid]
                continue
            if graced is not None:
                live = self._claims[pid] & graced
            else:
                live = {
                    v
                    for v in self._claims[pid]
                    if v not in pending_ids and v not in running_ids
                }
            if live:
                self._claims[pid] = frozenset(live)
            else:
                del self._claims[pid]

    def _verify_and_refine(
        self,
        job: JobView,
        view: ClusterView,
        chosen: list[JobView],
        extras: list[JobView],
    ) -> list[JobView]:
        """Dry-run-verify a chip-count victim plan, or repair it.

        Returns ``chosen`` unchanged when the engine confirms that
        evicting it lets ``job`` place (the common, shape-clean case —
        one search).  Otherwise the plan is extended with ``extras``
        (remaining eligible victims in greedy order); if even the full
        set cannot make the job placeable, returns ``[]`` (evicting
        provably cannot help — no thrash).  A repaired plan is pruned to
        an inclusion-minimal feasible set, dropping least-preferred
        victims first, and returned in greedy order.  Views without
        ``reclaim_feasible`` trust the chip-count plan (v0.1 behavior)."""
        feasible = getattr(view, "reclaim_feasible", None)
        if feasible is None:
            return chosen
        if feasible(job, [v.id for v in chosen]):
            return chosen
        full = chosen + [v for v in extras if v not in chosen]
        if not feasible(job, [v.id for v in full]):
            return []
        kept = list(full)
        for victim in reversed(full):  # drop least-preferred first
            if len(kept) == 1:
                break
            trial = [v for v in kept if v.id != victim.id]
            if feasible(job, [v.id for v in trial]):
                kept = trial
        return kept

    # ------------------------------------------------------------------

    def _plan_preemption(
        self, job: JobView, view: ClusterView, running: Sequence[JobView]
    ) -> list[JobView]:
        """The full (uncapped) victim set for ``job``, or ``[]``.

        Scans candidate domains in ascending id order; in the first
        domain whose ``free + preemptable`` chips cover the request, the
        greedy chip-count plan is verified (and, when node shapes or leaf
        health defeat it, repaired or rejected) by
        :meth:`_verify_and_refine`.  Empty when no domain yields a
        feasible set, or when free chips alone already cover the request
        (preemption provably would not help under chip-count accounting —
        pinned v0.1 rule).
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

        if job.segments is not None:
            return self._plan_segmented_preemption(job, view, eligible)

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
                extras = [v for v in candidates[len(chosen):]]
                refined = self._verify_and_refine(job, view, chosen, extras)
                if refined:
                    return refined
                # This domain's victims cannot make the job placeable
                # (shape/health); try the next domain.
        return []

    # ------------------------------------------------------------------
    # Segmented reclaim (v0.2): empty multiple pods for one pending gang
    # ------------------------------------------------------------------

    def _plan_segmented_preemption(
        self,
        job: JobView,
        view: ClusterView,
        eligible: list[tuple[JobView, _Stint]],
    ) -> list[JobView]:
        """Victim set letting a SEGMENTED pending job place, or ``[]``.

        Outer domains (``job.within``'s level, or ``"cluster"``) are
        tried in ascending id order; inside one, segment-level domains of
        the job's chip type are packed GREEDILY: each of the job's
        ``n_segments`` segments goes to the domain that can host its next
        segment with the FEWEST additional preemptions (ties by ascending
        domain id), accumulating one aggregate victim set across all
        segments — this is how one pending gang empties multiple pods of
        best-effort mice.  Victim ordering inside a domain is the same
        greedy key as the flat path; a victim straddling several segment
        domains (itself segmented) is chosen once and its freed chips are
        credited to every domain it touches.  Packing is by CHIP COUNTS
        on domain counters (never fleet-wide leaf scans); the resulting
        plan is then verified/repaired against real node shapes and leaf
        health by :meth:`_verify_and_refine` before being returned.
        Returns ``[]`` when the job's chip type is unpinned, no outer
        domain can be made to fit, or zero preemptions would already
        suffice (preempting cannot be shown to help).
        """
        if job.chip_type is None or job.n_segments <= 0:
            return []  # unpinned segmented jobs: documented blind spot
        seg_chips = job.chips // job.n_segments
        if seg_chips <= 0:
            return []
        assert job.segments is not None
        seg_level = job.segments[1]
        outer_level = job.within if job.within is not None else "cluster"
        seg_views = view.domains(seg_level)  # ascending id order
        for outer in view.domains(outer_level):  # ascending id order
            doms = [
                dv
                for dv in seg_views
                if _id_within(dv.id, outer.id) and dv.chip_type == job.chip_type
            ]
            if not doms:
                continue
            victims = self._pack_reclaim(job, doms, eligible, seg_chips)
            if victims:
                chosen_ids = {v.id for v in victims}
                extras = [
                    victim
                    for victim, stint in eligible
                    if victim.id not in chosen_ids
                    and stint.placement.chip_type == job.chip_type
                    and _chips_within(stint.placement, outer.id) > 0
                ]
                extras.sort(
                    key=lambda v: (
                        int(v.tier),
                        v.attained_service_chip_s,
                        -v.submit_time,
                        v.id,
                    )
                )
                refined = self._verify_and_refine(job, view, victims, extras)
                if refined:
                    return refined
                # Shape/health defeat this outer domain; try the next.
        return []

    def _pack_reclaim(
        self,
        job: JobView,
        doms: list,
        eligible: list[tuple[JobView, _Stint]],
        seg_chips: int,
    ) -> list[JobView]:
        """Greedy fewest-preemptions segment packing under one outer
        domain (see :meth:`_plan_segmented_preemption`).  Returns the
        aggregate victim list in choice order, or ``[]``."""
        # Per-domain victim candidates with their in-domain chips, in the
        # flat path's greedy victim order.
        lists: dict[str, list[tuple[JobView, int]]] = {}
        vmap: dict[str, dict[str, int]] = {}
        for dv in doms:
            lst = [
                (victim, chips)
                for victim, stint in eligible
                if stint.placement.chip_type == job.chip_type
                and (chips := _chips_within(stint.placement, dv.id)) > 0
            ]
            lst.sort(
                key=lambda pair: (
                    int(pair[0].tier),
                    pair[0].attained_service_chip_s,
                    -pair[0].submit_time,
                    pair[0].id,
                )
            )
            lists[dv.id] = lst
            vmap[dv.id] = {victim.id: chips for victim, chips in lst}
        cap = {dv.id: dv.free_chips for dv in doms}
        idx = {dv.id: 0 for dv in doms}
        hosted = {dv.id: 0 for dv in doms}
        chosen: dict[str, JobView] = {}  # insertion order = emission order
        for _ in range(job.n_segments):
            best: tuple[int, str, list[tuple[JobView, int]], int] | None = None
            for dv in doms:  # ascending id order (tie-break)
                d = dv.id
                need = (hosted[d] + 1) * seg_chips
                cap_d = cap[d]
                j = idx[d]
                lst = lists[d]
                new: list[tuple[JobView, int]] = []
                while cap_d < need and j < len(lst):
                    victim, chips = lst[j]
                    j += 1
                    if victim.id in chosen:
                        continue  # chips already credited into cap[d]
                    new.append((victim, chips))
                    cap_d += chips
                if cap_d >= need and (
                    best is None or len(new) < best[0]
                ):
                    best = (len(new), d, new, j)
                    if best[0] == 0:
                        break  # cannot beat zero new preemptions
            if best is None:
                return []  # this outer domain cannot host every segment
            _, d, new, j = best
            hosted[d] += 1
            idx[d] = j
            for victim, _chips in new:
                chosen[victim.id] = victim
                # Credit the victim's freed chips everywhere it runs.
                for dv2 in doms:
                    credit = vmap[dv2.id].get(victim.id)
                    if credit:
                        cap[dv2.id] += credit
        # All segments hosted with zero preemptions -> placement failed on
        # shape alone; preempting cannot be shown to help (flat-path rule).
        return list(chosen.values())
