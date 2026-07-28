"""Inference services -> frozen INFER_REPLICA jobs (DESIGN §5, v1 scope).

v1 freezes replica counts (``min_replicas == max_replicas``); autoscaling
and the DiurnalCurve load model arrive in v0.2.  :func:`expand_services`
turns each :class:`~fleetsim.model.Service` into ``min_replicas``
INFER_REPLICA jobs submitted at t=0 that run for the whole horizon —
Borg/MAST-style "serving is just prod-tier jobs", so every scheduler
handles the training/inference mix with no special casing.

Replica shape (pinned for v1): each replica requests exactly ONE whole
node — ``chips`` = the fleet's node size for the replica spec's chip
type (``replica_spec.chips`` is superseded; sub-node or multi-node
replicas are future surface).  The ``within`` constraint on the replica
spec is copied through.

UNITS: ``horizon_us`` int microseconds; job ``true_duration_s`` is float
seconds (== the horizon, i.e. open-ended for the run).

INVARIANTS: pure function of its inputs (no randomness); job ids are
``<service_id>-r<i>`` with ``i`` in ``0..min_replicas-1``; jobs are
returned in service order then replica order (deterministic); replicas
disable checkpointing (``checkpoint_interval_s = 0`` — an interrupted
replica has no progress to bank).
"""

from __future__ import annotations

from typing import Sequence

from ..fleet.tree import FleetTree
from ..model import Constraint, GangSpec, Job, JobClass, Service
from .synthetic import node_sizes

__all__ = ["expand_services"]


def expand_services(
    services: Sequence[Service], fleet: FleetTree, horizon_us: int
) -> list[Job]:
    """Expand services into frozen one-node INFER_REPLICA jobs at t=0.

    Raises ``ValueError`` for a non-positive horizon, ``min_replicas !=
    max_replicas`` (autoscaling is not implemented in v0.1), negative
    replica counts, or a pinned chip type with no matching leaves.
    """
    if horizon_us <= 0:
        raise ValueError(f"horizon must be positive, got {horizon_us}")
    sizes = node_sizes(fleet)
    jobs: list[Job] = []
    for svc in services:
        if svc.min_replicas != svc.max_replicas:
            raise ValueError(
                f"service {svc.id!r}: autoscaling (min_replicas !="
                f" max_replicas) is not implemented in v0.1"
            )
        if svc.min_replicas < 0:
            raise ValueError(
                f"service {svc.id!r}: min_replicas must be >= 0,"
                f" got {svc.min_replicas}"
            )
        spec = svc.replica_spec
        if spec.chip_type is not None:
            if spec.chip_type not in sizes:
                raise ValueError(
                    f"service {svc.id!r}: no leaves of chip_type"
                    f" {spec.chip_type!r} in the fleet"
                )
            node = sizes[spec.chip_type]
        else:
            node = max(sizes.values())
        for i in range(svc.min_replicas):
            within = (
                Constraint(
                    level=spec.within.level,
                    required=spec.within.required,
                    relax_after_s=spec.within.relax_after_s,
                )
                if spec.within is not None
                else None
            )
            jobs.append(
                Job(
                    id=f"{svc.id}-r{i}",
                    tenant=svc.tenant,
                    job_class=JobClass.INFER_REPLICA,
                    submit_t=0,
                    gangs=[
                        GangSpec(chips=node, chip_type=spec.chip_type, within=within)
                    ],
                    tier=svc.tier,
                    min_runtime_s=0.0,
                    checkpoint_interval_s=0.0,
                    true_duration_s=horizon_us / 1e6,
                    service_id=svc.id,
                )
            )
    return jobs
