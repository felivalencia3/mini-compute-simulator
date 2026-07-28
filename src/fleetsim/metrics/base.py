"""The metrics sink protocol: every engine state transition, as callbacks.

The engine (``fleetsim.engine.sim``) calls exactly one sink method at each
corresponding transition; the metrics phase implements a real collector
against this protocol (time-weighted integrals + event-sourced records,
DESIGN §9).  :class:`NullSink` is the do-nothing implementation.

UNITS: every ``t`` is int microseconds since sim epoch; ``*_chip_s``
quantities are float chip-seconds; ``n``/``n_chips`` are chip counts.

CALL SEMANTICS (pinned by the engine)
-------------------------------------
- ``job_submitted`` fires at arrival, before admission; ``job_admitted``
  only if admission passed.
- ``job_started`` fires at every stint start (first start and resumes),
  immediately followed by ``chips_allocated``.
- ``job_preempted`` fires when a preemption begins (scheduler action or
  engine maintenance drain; ``trigger`` is ``"scheduler"`` or
  ``"maintenance"``).  Node-failure kills are reported via ``node_failed``
  instead, never ``job_preempted``.
- ``job_requeued`` fires when a preempted or failure-killed job re-enters
  the queue (it keeps its original ``submit_t``).
- ``job_progress`` fires whenever a stint's work is settled — at every
  interruption, at completion, and once at the horizon for the banked
  (checkpointed) work of still-allocated jobs.  ``start_us``/``end_us``
  bracket the stint interval the work accrued over;
  ``productive_chip_s`` is the SURVIVING work delta x chips and
  ``lost_chip_s`` the lost delta x chips.  Summing ``productive_chip_s``
  over all calls gives the run's goodput numerator (DESIGN §9),
  including live jobs' checkpointed progress.
- ``job_finished`` fires once per job at its terminal transition;
  ``productive_chip_s`` is checkpointed forward progress x chips,
  ``lost_chip_s`` is work lost to interruptions x chips (cumulative
  totals — the same quantities ``job_progress`` reported as deltas).
- ``node_failed`` carries a sampled failure ``cause`` label (DESIGN §8
  mix: gpu_hbm / network / software / other; default "unknown" for
  callers that do not sample one).
- ``chips_allocated`` / ``chips_freed`` bracket every allocation's
  lifetime (freed is called even when the leaf is down — it tracks
  *allocation*, while ``healthy_delta`` tracks *capacity health*).
- ``healthy_delta`` fires with negative ``n_chips`` when a node leaves
  HEALTHY (failure or drain start) and positive when it returns.
- ``flush`` fires every scheduler round (rank after SCHED_WAKE) and once
  at the horizon: the O(1) sampling hook for time series.

INVARIANTS: sink methods must not mutate ``job``/``fleet`` — they are the
engine's live objects, shared for speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..model import Allocation, Job, JobStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..fleet.tree import FleetTree

__all__ = ["MetricsSink", "NullSink"]


@runtime_checkable
class MetricsSink(Protocol):
    """Receiver of every engine state transition (see module docstring)."""

    def job_submitted(self, job: Job, t: int) -> None: ...

    def job_admitted(self, job: Job, t: int) -> None: ...

    def job_started(self, job: Job, alloc: Allocation, t: int) -> None: ...

    def job_preempted(self, job: Job, t: int, trigger: str) -> None: ...

    def job_requeued(self, job: Job, t: int) -> None: ...

    def job_progress(
        self,
        job: Job,
        start_us: int,
        end_us: int,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None: ...

    def job_finished(
        self,
        job: Job,
        t: int,
        status: JobStatus,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None: ...

    def node_failed(
        self,
        node_id: str,
        t: int,
        killed_alloc_ids: Sequence[str],
        cause: str = "unknown",
    ) -> None: ...

    def node_repaired(self, node_id: str, t: int) -> None: ...

    def node_drain_started(self, node_id: str, t: int) -> None: ...

    def chips_allocated(self, n: int, chip_type: str, t: int) -> None: ...

    def chips_freed(self, n: int, chip_type: str, t: int) -> None: ...

    def healthy_delta(self, n_chips: int, chip_type: str, t: int) -> None: ...

    def flush(
        self, t: int, fleet: "FleetTree", n_pending: int, n_running: int
    ) -> None: ...


class NullSink:
    """A sink that ignores everything (the default when metrics are off)."""

    def job_submitted(self, job: Job, t: int) -> None:
        pass

    def job_admitted(self, job: Job, t: int) -> None:
        pass

    def job_started(self, job: Job, alloc: Allocation, t: int) -> None:
        pass

    def job_preempted(self, job: Job, t: int, trigger: str) -> None:
        pass

    def job_requeued(self, job: Job, t: int) -> None:
        pass

    def job_progress(
        self,
        job: Job,
        start_us: int,
        end_us: int,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None:
        pass

    def job_finished(
        self,
        job: Job,
        t: int,
        status: JobStatus,
        productive_chip_s: float,
        lost_chip_s: float,
    ) -> None:
        pass

    def node_failed(
        self,
        node_id: str,
        t: int,
        killed_alloc_ids: Sequence[str],
        cause: str = "unknown",
    ) -> None:
        pass

    def node_repaired(self, node_id: str, t: int) -> None:
        pass

    def node_drain_started(self, node_id: str, t: int) -> None:
        pass

    def chips_allocated(self, n: int, chip_type: str, t: int) -> None:
        pass

    def chips_freed(self, n: int, chip_type: str, t: int) -> None:
        pass

    def healthy_delta(self, n_chips: int, chip_type: str, t: int) -> None:
        pass

    def flush(
        self, t: int, fleet: "FleetTree", n_pending: int, n_running: int
    ) -> None:
        pass
