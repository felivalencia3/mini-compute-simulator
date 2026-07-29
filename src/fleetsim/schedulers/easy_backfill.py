"""EASY backfill (v0.4): FIFO with a head-of-line reservation and
conservative backfill against it.

SEMANTICS (pinned)
------------------
- **Ordering**: pending jobs are scanned by ``(submit_time, id)`` —
  plain FIFO.  Jobs that fit are placed immediately.
- **The head**: the FIRST job in FIFO order that cannot be placed
  becomes the head of line.  The scheduler computes the head's **shadow
  time** — the earliest time at which, by scheduler-visible walltime
  estimates, enough chips free up inside ONE candidate domain for the
  head — and lets later jobs run NOW only if they provably finish by
  then.
- **Shadow computation** (chip-count accounting): candidate domains are
  the domains at the head's ``within`` level (or ``"cluster"`` when
  unconstrained), ascending id, skipping wrong-typed / heterogeneous
  domains (interior ``chip_type is None`` — same documented blind spot
  as tiered_priority's victim scan).  Free chips sitting on leaves HELD
  by a calendar reservation for another tenant (v0.4) are SUBTRACTED
  first — the head can never claim them, so they neither collapse the
  shadow to ``now`` nor seed the accumulation (releases are still
  counted hold-blind: a chip-count approximation, like the shape
  approximation below).  Inside a domain, running jobs placed by THIS
  scheduler release ``chips`` at ``stint_start + walltime_est /
  placement_speed`` (estimates are speed-1 walltimes; a penalized
  cross-domain placement runs 1/speed longer — see PENALTIES); jobs
  with NO estimate never release.  ``shadow(domain)`` is the earliest
  est-release time by which cumulative freed + currently-free eligible
  chips cover the head; the shadow is the MINIMUM over domains (``now``
  when a domain's eligible free chips already cover the request — the
  placement failed on shape, so no backfill window exists).  If no
  domain can ever cover the head by estimates, there is NO shadow and
  NO backfill this wake (never gamble against an unbounded
  reservation).
- **Backfill rule**: a later pending job may start now iff it has a
  walltime estimate, it places, and ``now + walltime_est /
  placement_speed <= shadow`` at the speed of the placement it actually
  found.  This is the conservative EASY rule (no "extra nodes" clause —
  a documented divergence from canonical EASY, see DESIGN §17.2):
  every backfilled job is gone — by its speed-adjusted estimate —
  before the head's reservation.
- **THE GUARANTEE (exact statement)**: with EXACT estimates
  (``est == true remaining``) the head — and, on a fungible fleet,
  every job — starts no later than under strict FIFO (the CI rung
  asserts this pointwise).  With merely HONEST over-estimates
  (``est >= true``) the pointwise property does NOT hold: an inflated
  estimate widens the shadow window and a backfilled job may outlive
  the head's true FIFO start.  What holds then is the canonical EASY
  property (Mu'alem & Feitelson, IEEE TPDS 2001): the head never
  starts later than the shadow time computed when backfill was
  granted.  Both are asserted in
  validation/test_backfill_property.py.
- **ESTIMATE-ERROR HONESTY**: all reasoning uses ``walltime_est_s``,
  the scheduler-visible estimate.  The engine does NOT kill a job at
  its estimate — a job that UNDERESTIMATED (true > est) keeps running
  past its promised release and CAN delay the head past the shadow,
  exactly as on a real cluster whose operators refuse to enforce
  walltime kills.  Estimates of RESUMED jobs are measured from the
  stint start (restart overhead and checkpoint rewind make the
  estimate optimistic by up to one checkpoint interval — a documented,
  real-world-shaped error source).
- **PENALTIES (v0.4)**: ``penalties.xover`` placements run at
  ``1/speed`` their estimate; the scheduler prices every placement it
  makes via the engine view's ``placement_speed`` (1.0 when the view
  lacks the hook or no penalties are configured), both for its own
  release accounting and for the backfill admission test — a relaxed
  cross-pod backfill cannot overstay its promise by the penalty
  factor.  A candidate whose found placement is too slow to keep its
  promise is skipped (its tentative reservation is rolled back so the
  chips stay available to later candidates).
- **Statefulness**: like tiered_priority, the scheduler records
  ``(stint start, placement, speed)`` for every ``Place`` it emits
  (the ClusterView exposes none of these); running jobs without a
  record are treated as never releasing.  Use a fresh instance per
  Simulator run.
- **Id convention**: leaf-in-domain containment is tested by path
  prefix over :func:`fleetsim.fleet.build.build_fleet`'s hierarchical
  ids.

UNITS: ``view.now`` / ``submit_time`` / shadow are int microseconds;
``walltime_est_s`` float seconds.  INVARIANTS: deterministic (sorted
scans, no randomness); emits only ``Place`` actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..fleet.tree import Placement
from .base import (
    Action,
    ClusterView,
    JobView,
    Place,
    PlacementPolicy,
    Scheduler,
    register,
)
from .placement import FirstFit

__all__ = ["EasyBackfillScheduler"]


@dataclass(frozen=True, slots=True)
class _Stint:
    """Scheduler-side record of a stint this scheduler started.

    ``speed`` is the placement's cost-model multiplier at start (v0.4
    penalties; 1.0 unpenalized) — the stint's promised release is
    ``start_us + walltime_est / speed``."""

    start_us: int
    placement: Placement
    speed: float = 1.0


def _placement_within(placement: Placement, domain_id: str) -> bool:
    """True iff every leaf of ``placement`` lies under ``domain_id``
    (path-prefix containment over build_fleet's hierarchical ids)."""
    prefix = domain_id + "/"
    return all(
        lid == domain_id or lid.startswith(prefix) for lid, _ in placement.leaves
    )


@register("easy_backfill")
class EasyBackfillScheduler(Scheduler):
    """EASY backfill: FIFO + head-of-line shadow reservation +
    conservative walltime-estimate backfill (see module docstring).

    Parameters:

    - ``placement``: the :class:`~fleetsim.schedulers.base.PlacementPolicy`
      (default :class:`~fleetsim.schedulers.placement.FirstFit`);
      programmatic only — not expressible in YAML.
    """

    def __init__(self, placement: PlacementPolicy | None = None):
        self.placement: PlacementPolicy = (
            placement if placement is not None else FirstFit()
        )
        #: job id -> stint record for jobs this scheduler placed; pruned
        #: to the running set at each wake.
        self._stints: dict[str, _Stint] = {}

    # ------------------------------------------------------------------

    def schedule(self, view: ClusterView) -> list[Action]:
        running = view.running()
        running_ids = {jv.id for jv in running}
        self._stints = {
            jid: st for jid, st in self._stints.items() if jid in running_ids
        }
        # Engine-view extras (probed like tiered_priority's
        # reclaim_feasible): the v0.4 cost-model price of a placement and
        # same-wake rollback of examined-but-rejected candidates.
        speed_fn = getattr(view, "placement_speed", None)
        release_fn = getattr(view, "release_tentative", None)
        actions: list[Action] = []
        order = sorted(view.pending(), key=lambda j: (j.submit_time, j.id))
        # Jobs releasing capacity in the future: already-running jobs with
        # known stints PLUS jobs placed earlier in THIS wake (their stints
        # start now) — the same-wake placements must count, or the shadow
        # would lag one round behind the tentative reservations.
        active: list[tuple[JobView, _Stint]] = [
            (jv, self._stints[jv.id]) for jv in running if jv.id in self._stints
        ]
        head: JobView | None = None
        for i, job in enumerate(order):
            placement = view.find_placement(job, self.placement)
            if placement is not None:
                actions.append(Place(job.id, placement))
                speed = speed_fn(placement) if speed_fn is not None else 1.0
                stint = _Stint(
                    start_us=view.now, placement=placement, speed=speed
                )
                self._stints[job.id] = stint
                active.append((job, stint))
                continue
            head = job
            rest = order[i + 1 :]
            break
        if head is None:
            return actions

        shadow = self._shadow(head, view, active)
        if shadow is None:
            return actions  # unbounded reservation: never backfill against it

        for job in rest:
            est = job.walltime_est_s
            if est is None:
                continue  # no estimate, no backfill (EASY needs a promise)
            if view.now + round(est * 1e6) > shadow:
                continue  # would outlive the head even at full speed
            placement = view.find_placement(job, self.placement)
            if placement is None:
                continue
            speed = speed_fn(placement) if speed_fn is not None else 1.0
            if speed < 1.0 and view.now + round(est / speed * 1e6) > shadow:
                # The placement actually found is PENALIZED and would run
                # 1/speed past its speed-1 promise, overstaying the
                # head's reservation — skip, and free the tentatively
                # reserved chips for later candidates this wake.
                if release_fn is not None:
                    release_fn(job.id)
                continue
            actions.append(Place(job.id, placement))
            self._stints[job.id] = _Stint(
                start_us=view.now, placement=placement, speed=speed
            )
        return actions

    # ------------------------------------------------------------------

    def _shadow(
        self,
        head: JobView,
        view: ClusterView,
        active: list[tuple[JobView, _Stint]],
    ) -> int | None:
        """The head's shadow time (int µs), or None when no candidate
        domain can ever cover it by estimates (see module docstring).
        ``active`` pairs every capacity-holding job (running or placed
        this wake) with its stint record.  Free chips on calendar-held
        leaves the head's tenant cannot use are subtracted (v0.4);
        promised releases are speed-adjusted (penalties)."""
        level = head.within if head.within is not None else "cluster"
        reserved_fn = getattr(view, "reserved_free_chips", None)
        best: int | None = None
        for dom in view.domains(level):  # ascending id order
            chip_type = (
                head.chip_type if head.chip_type is not None else dom.chip_type
            )
            if chip_type is None or dom.chip_type != chip_type:
                continue  # wrong-typed or heterogeneous domain (skip)
            free = dom.free_chips
            if reserved_fn is not None:
                # Held-for-another-tenant free chips are invisible to the
                # head: they must not collapse the shadow to `now` (which
                # would disable backfill entirely) nor pad the
                # accumulation with capacity the head can never claim.
                free -= reserved_fn(dom.id, head.tenant)
            if free >= head.chips:
                # Eligible chip count already suffices — the placement
                # failed on shape, so there is no time window to
                # backfill into.
                return view.now
            releases: list[tuple[int, int]] = []  # (est_end_us, chips)
            for jv, stint in active:
                if jv.walltime_est_s is None:
                    continue  # never releases (scheduler-visible truth)
                if stint.placement.chip_type != chip_type:
                    continue
                if not _placement_within(stint.placement, dom.id):
                    continue
                releases.append(
                    (
                        stint.start_us
                        + round(jv.walltime_est_s / stint.speed * 1e6),
                        jv.chips,
                    )
                )
            releases.sort()
            shadow_d: int | None = None
            for est_end, chips in releases:
                free += chips
                if free >= head.chips:
                    shadow_d = max(est_end, view.now)
                    break
            if shadow_d is not None and (best is None or shadow_d < best):
                best = shadow_d
        return best
