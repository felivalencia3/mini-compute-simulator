"""SJF: shortest-job-first ordering, the Helios Table-3 reference policy.

Orders pending jobs by their *estimated* service time
(``walltime_est_s``) ascending, then breaks ties by ``(submit_time, id)``
so the policy stays deterministic and requeued jobs keep their arrival
order behind equal-length peers.  Jobs with no estimate
(``walltime_est_s is None``) sort LAST (treated as +inf — the least
information, so the most conservative position).

ORACLE SEMANTICS.  fleetsim's trace converter writes each job's replayed
``duration`` into BOTH ``duration_s`` (the hidden true work) and
``walltime_limit_s`` -> ``walltime_est_s`` (the scheduler-visible
estimate).  When the estimate equals the true duration the SJF ordering
is *perfect* — this is **SJF-oracle**, a scheduler with a flawless
service-time predictor.  That is exactly the policy the Helios reference
simulator (Hu et al., "Characterization and Prediction of Deep Learning
Workloads in Large-Scale GPU Datacenters", SC '21, arXiv:2109.01313)
keys on when it reports the SJF column of Table 3, and it is a strict
upper bound on what any real (imperfect) shortest-job predictor such as
Helios QSSF can achieve.  ``SJFScheduler`` is therefore the reproducible
reference used by the v0.6 validation suite (V1/V2) to replay the
FIFO-vs-SJF JCT and queuing ratios.

PLACEMENT + SCAN.  Placement is FirstFit by default (the same
consolidating first-fit the FIFO baseline uses).  ``strict`` defaults to
**False** (best-effort): an unplaceable job is skipped and the scan
continues down the ordered queue, so a short job flows around a stuck
gang.  With ``strict=True`` the head of the (shortest-first) line blocks,
exposing gang fragmentation exactly as ``fifo`` strict does.  The Helios
validation runs ``strict=True``: the reference sim is BLOCKing
(head-of-line), not the non-blocking scan the plan guessed — best-effort
collapses FIFO's queuing and destroys the FIFO-vs-SJF ratio, so
``strict=True`` is the load-bearing fidelity choice there (see
:mod:`fleetsim.validation.harness`).  The best-effort default is for
general workloads where flowing small jobs around a stall is the desired
policy, not for the Helios replay.

INVARIANTS: emits only ``Place`` actions; deterministic — the ordering
key ``(walltime_est_s or +inf, submit_time, id)`` and the deterministic
placement policy make the schedule a pure function of the view.
"""

from __future__ import annotations

import math

from .base import Action, ClusterView, Place, PlacementPolicy, Scheduler, register
from .placement import FirstFit

__all__ = ["SJFScheduler"]


@register("sjf")
class SJFScheduler(Scheduler):
    """Shortest-job-first over ``(walltime_est_s, submit_time, id)`` with
    pluggable placement.  ``strict=True`` blocks on the (shortest) head of
    line; the default best-effort mode skips unplaceable jobs and
    continues.  See the module docstring for the SJF-oracle semantics."""

    def __init__(self, placement: PlacementPolicy | None = None, strict: bool = False):
        self.placement: PlacementPolicy = placement if placement is not None else FirstFit()
        self.strict = bool(strict)

    @staticmethod
    def _order_key(job) -> tuple[float, int, str]:
        """Sort key: estimated service time (missing -> +inf) then the
        deterministic ``(submit_time, id)`` FIFO tie-break."""
        est = job.walltime_est_s
        return (est if est is not None else math.inf, job.submit_time, job.id)

    def schedule(self, view: ClusterView) -> list[Action]:
        actions: list[Action] = []
        for job in sorted(view.pending(), key=self._order_key):
            p = view.find_placement(job, self.placement)
            if p is not None:
                actions.append(Place(job.id, p))
            elif self.strict:
                break  # strict SJF: the shortest head-of-line blocks
            # else best-effort: skip the unplaceable job and continue
        return actions
