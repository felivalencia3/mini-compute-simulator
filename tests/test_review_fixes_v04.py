"""Regression guards for the v0.4 review fixes.

Each test pins one reviewed defect:

- Reservation claims pick the FEWEST-EVICTIONS domain (an idle domain
  beats a busy lower-id one) and prefer free / owner-occupied leaves
  inside it — no gratuitous policy evictions (DESIGN §17.4).
- EASY backfill stays alive while a calendar hold is active: the shadow
  subtracts held-for-another-tenant free chips instead of collapsing to
  ``now`` (DESIGN §17.2).
- EASY backfill prices penalized placements: a candidate whose found
  placement runs at 1/speed cannot overstay its speed-1 promise
  (DESIGN §17.2).
- ``reclaim_feasible`` honors relaxable ``within`` constraints exactly
  as FirstFit places them (constrained first, relaxed retry after the
  timeout).
- The engine view exposes calendar reservations (``reservations()``) so
  schedulers can see holds coming, and the tree prices held-but-free
  capacity (``reserved_free_chips``).
- Config honesty: ``within.required`` must be a real boolean, per-class
  ``tenant:`` pins must be non-empty, and quota caps on tenants no
  configured source can produce are rejected as dead config.
"""

from dataclasses import replace

import pytest

from fleetsim.config import load_scenario, validate
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.model import GangSpec, Job, JobClass, JobStatus, Tier
from fleetsim.schedulers.base import (
    JobView,
    Place,
    Scheduler,
    get_scheduler,
)
from fleetsim.schedulers.placement import FirstFit
from fleetsim.workload.base import ListSource

S = 1_000_000


def make_scenario(
    *,
    pods=2,
    nodes=2,
    per_node=8,
    reservations=None,
    penalties=None,
    horizon="2h",
):
    doc = {
        "sim": {"horizon": horizon, "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": per_node},
                    "topology": {"levels": ["pod", "node"], "counts": [pods, nodes]},
                }
            ],
        },
        "failure_model": {
            "node_mtbf_days": 0,
            "maintenance_rate_per_node_month": 0,
        },
        "workload": {
            "kind": "synthetic",
            "classes": {
                "eval": {
                    "rate_per_hour": 1,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=2m, p90=30m]",
                }
            },
        },
    }
    if reservations is not None:
        doc["reservations"] = reservations
    if penalties is not None:
        doc["penalties"] = penalties
    return load_scenario(doc)


def mk_job(jid, chips, dur_s, tenant="t0", est_s=None, submit_s=0.0):
    return Job(
        id=jid,
        tenant=tenant,
        job_class=JobClass.FINETUNE,
        submit_t=int(round(submit_s * S)),
        gangs=[GangSpec(chips=chips, chip_type="h100")],
        tier=Tier.BATCH,
        walltime_est_s=est_s,
        true_duration_s=dur_s,
        checkpoint_interval_s=0.0,
        checkpoint_save_s=0.0,
        restart_overhead_s=0.0,
    )


class Sink:
    def __init__(self):
        self.first_start = {}
        self.preempted = []
        self.reports = []

    def job_started(self, job, alloc, t):
        self.first_start.setdefault(job.id, t)

    def job_preempted(self, job, t, trigger):
        self.preempted.append((job.id, t, trigger))

    def reservation_report(self, report):
        self.reports.append(report)

    def __getattr__(self, name):
        return lambda *a, **k: None


# ---------------------------------------------------------------------------
# Reservation claim: fewest evictions across domains, gentlest leaves within
# ---------------------------------------------------------------------------


def test_reservation_claim_prefers_the_idle_domain_over_low_id():
    """A running non-owner job fills half of pod0; pod1 is idle.  The
    claim must take pod1 (zero evictions), never evict the pod0
    resident just because pod0 sorts first."""
    scenario = make_scenario(
        reservations=[
            {
                "id": "r",
                "tenant": "acme",
                "chips": 16,
                "level": "pod",
                "start": "600s",
                "end": "1200s",
            }
        ]
    )
    fleet = build_fleet(scenario)
    sink = Sink()
    resident = mk_job("resident", 8, 5000.0, tenant="other")
    sim = Simulator(
        scenario, fleet, ListSource([resident]), get_scheduler("fifo", {}), sink
    )
    sim.run()
    (report,) = sink.reports
    assert report["n_evicted_at_start"] == 0
    assert all(n.startswith("m/c/pod1/") for n in report["nodes"]), report["nodes"]
    assert not sink.preempted  # the resident was never touched
    assert resident.status is JobStatus.COMPLETED


def test_reservation_claim_prefers_owner_occupied_leaves_inside_a_domain():
    """One pod, three nodes: node0 hosts the OWNER's job, node1 a
    foreign job, node2 is free.  A 16-chip claim takes the free node and
    the owner-occupied node — zero evictions."""
    scenario = make_scenario(
        pods=1,
        nodes=3,
        reservations=[
            {
                "id": "r",
                "tenant": "acme",
                "chips": 16,
                "level": "pod",
                "start": "600s",
                "end": "1200s",
            }
        ],
    )
    fleet = build_fleet(scenario)
    sink = Sink()
    # FIFO order is (submit, id): a_own -> node0, b_for -> node1;
    # node2 stays free.
    owner = mk_job("a_own", 8, 5000.0, tenant="acme")
    foreign = mk_job("b_for", 8, 5000.0, tenant="other")
    sim = Simulator(
        scenario,
        fleet,
        ListSource([owner, foreign]),
        get_scheduler("fifo", {}),
        sink,
    )
    sim.run()
    (report,) = sink.reports
    assert report["n_evicted_at_start"] == 0
    assert report["nodes"] == ["m/c/pod0/node0", "m/c/pod0/node2"]
    # No eviction at the claim; the hard-end cliff at 1200 s correctly
    # cuts through the owner's own still-running job (DESIGN §17.4).
    assert [p for p in sink.preempted if p[1] == 600 * S] == []
    assert [(p[0], p[2]) for p in sink.preempted] == [("a_own", "reservation")]
    assert report["n_evicted_at_end"] == 1


def test_reservation_claim_still_evicts_when_unavoidable():
    """Both pods carry foreign residents and the claim needs a full
    pod: evictions happen (REQUEUE, trigger 'reservation'), and the
    chosen domain is the fewest-evictions one (pod1: one 16-chip job
    beats pod0's two 8-chip jobs)."""
    scenario = make_scenario(
        reservations=[
            {
                "id": "r",
                "tenant": "acme",
                "chips": 16,
                "level": "pod",
                "start": "600s",
                "end": "1200s",
            }
        ]
    )
    fleet = build_fleet(scenario)
    sink = Sink()
    jobs = [
        mk_job("a", 8, 5000.0, tenant="other"),   # pod0/node0
        mk_job("b", 8, 5000.0, tenant="other"),   # pod0/node1
        mk_job("c", 16, 5000.0, tenant="other"),  # pod1 (both nodes)
    ]
    sim = Simulator(
        scenario, fleet, ListSource(jobs), get_scheduler("fifo", {}), sink
    )
    sim.run()
    (report,) = sink.reports
    assert report["n_evicted_at_start"] == 1  # c, not a+b
    assert all(n.startswith("m/c/pod1/") for n in report["nodes"])
    assert [(p[0], p[2]) for p in sink.preempted] == [("c", "reservation")]


# ---------------------------------------------------------------------------
# EASY backfill: hold-aware shadow, penalty-aware promises
# ---------------------------------------------------------------------------


def test_backfill_survives_an_active_reservation_hold():
    """16 x 1-chip fleet, 8 nodes held for 'acme' the whole run.  The
    held-but-free chips must not collapse the shadow to `now` (which
    silently disabled ALL backfill): the 1-chip mouse with a 10 s
    estimate backfills at t=0 against the 600 s shadow."""
    scenario = make_scenario(
        pods=1,
        nodes=16,
        per_node=1,
        reservations=[
            {
                "id": "r",
                "tenant": "acme",
                "chips": 8,
                "start": "0s",
                "end": "3600s",
            }
        ],
    )
    fleet = build_fleet(scenario)
    sink = Sink()
    jobs = [
        mk_job("a0", 6, 600.0, tenant="other", est_s=600.0),
        mk_job("a1", 4, 900.0, tenant="other", est_s=900.0),  # head
        mk_job("b2", 1, 10.0, tenant="other", est_s=10.0),    # mouse
    ]
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        get_scheduler("easy_backfill", {}),
        sink,
    )
    sim.run()
    assert sink.first_start["a0"] == 0
    assert sink.first_start["b2"] == 0  # backfilled, not FIFO-blocked
    assert sink.first_start["a1"] == 600 * S  # head at its shadow


def test_backfill_prices_penalized_placements():
    """xover.pod = 0.5: a relaxable 16-chip job with an exact 550 s
    estimate would land cross-pod at half speed (true release 1100 s) —
    admitting it against the head's 600 s shadow would double-overstay
    its promise.  The scheduler must refuse the backfill and start the
    head exactly at its shadow."""
    scenario = make_scenario(
        pods=2, nodes=3, penalties={"xover": {"pod": 0.5}}, horizon="3h"
    )
    fleet = build_fleet(scenario)
    sink = Sink()

    def relaxable(jid, chips, dur_s, est_s):
        from fleetsim.model import Constraint

        j = mk_job(jid, chips, dur_s, est_s=est_s)
        j.gangs = [
            GangSpec(
                chips=chips,
                chip_type="h100",
                within=Constraint(level="pod", required=False, relax_after_s=0.0),
            )
        ]
        return j

    def hard(jid, chips, dur_s, est_s):
        from fleetsim.model import Constraint

        j = mk_job(jid, chips, dur_s, est_s=est_s)
        j.gangs = [
            GangSpec(
                chips=chips,
                chip_type="h100",
                within=Constraint(level="pod", required=True),
            )
        ]
        return j

    jobs = [
        hard("a0", 16, 600.0, 600.0),        # fills pod0 (2 of 3 nodes)
        hard("a1", 16, 600.0, 600.0),        # fills pod1 (2 of 3 nodes)
        # FIFO order is (submit, id): "a2head" precedes "c3", so c3 is a
        # BACKFILL candidate against the head's 600 s shadow.
        mk_job("a2head", 24, 900.0, est_s=900.0),  # needs 3 nodes: blocked
        relaxable("c3", 16, 550.0, 550.0),   # would go cross-pod at 0.5x
    ]
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        get_scheduler("easy_backfill", {}),
        sink,
    )
    sim.run()
    assert sink.first_start["c3"] != 0  # refused at t=0 (would overstay)
    assert sink.first_start["a2head"] == 600 * S  # head lands at its shadow


# ---------------------------------------------------------------------------
# reclaim_feasible honors relaxable constraints
# ---------------------------------------------------------------------------


def test_reclaim_feasible_honors_relaxable_within():
    """Victims v1+v2 fill pod0; pod1 is free.  A 24-chip job cannot fit
    one pod even after releasing v1 (16 max per pod), but the RELAXED
    (elapsed-timeout) search spans pods: reclaim_feasible must say True
    for the relaxable spec and False for the hard one."""
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    result = {}

    class Probe(Scheduler):
        def __init__(self):
            self.placement = FirstFit()

        def schedule(self, view):
            actions = []
            for job in view.pending():
                p = view.find_placement(job, self.placement)
                if p is not None:
                    actions.append(Place(job.id, p))
            if view.now >= 60 * S and not result:
                jv = JobView(
                    id="R",
                    submit_time=0,
                    chips=24,
                    chip_type="h100",
                    tier=Tier.PROD,
                    job_class=JobClass.PRETRAIN,
                    preemptible=True,
                    min_runtime_s=0.0,
                    attained_service_chip_s=0.0,
                    checkpoint_age_s=0.0,
                    walltime_est_s=None,
                    within="pod",
                    tenant="t0",
                    within_required=False,
                    relax_after_s=0.0,
                )
                result["relaxed"] = view.reclaim_feasible(jv, ["v1"])
                result["hard"] = view.reclaim_feasible(
                    replace(jv, within_required=True), ["v1"]
                )
            return actions

    jobs = [mk_job("v1", 8, 5000.0), mk_job("v2", 8, 5000.0)]
    sim = Simulator(scenario, fleet, ListSource(jobs), Probe(), Sink())
    sim.run()
    assert result == {"relaxed": True, "hard": False}


# ---------------------------------------------------------------------------
# Reservation visibility on the engine view; held-free pricing on the tree
# ---------------------------------------------------------------------------


def test_engine_view_exposes_upcoming_and_active_reservations():
    scenario = make_scenario(
        reservations=[
            {
                "id": "r",
                "tenant": "acme",
                "chips": 8,
                "level": "pod",
                "start": "120s",
                "end": "600s",
            }
        ]
    )
    fleet = build_fleet(scenario)
    seen = {}

    class Probe(Scheduler):
        def schedule(self, view):
            seen[view.now] = view.reservations()
            return []

    sim = Simulator(scenario, fleet, ListSource([]), Probe(), Sink())
    sim.run()
    before = seen[60 * S]
    assert len(before) == 1 and not before[0].active and before[0].leaves == ()
    assert (before[0].id, before[0].tenant, before[0].chips) == ("r", "acme", 8)
    assert (before[0].start_us, before[0].end_us) == (120 * S, 600 * S)
    active = seen[180 * S]
    assert len(active) == 1 and active[0].active and len(active[0].leaves) == 1
    assert seen[660 * S] == ()  # finished blocks disappear


def test_tree_reserved_free_chips_prices_held_capacity():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    assert fleet.reserved_free_chips("m/c", None) == 0  # no holds: free fast-path
    fleet.reserve_leaves(["m/c/pod0/node0"], "acme")
    assert fleet.reserved_free_chips("m/c", None) == 8
    assert fleet.reserved_free_chips("m/c", "other") == 8
    assert fleet.reserved_free_chips("m/c", "acme") == 0  # the owner may use it
    assert fleet.reserved_free_chips("m/c/pod1", "other") == 0  # not under pod1
    fleet.release_reservation(["m/c/pod0/node0"])
    assert fleet.reserved_free_chips("m/c", "other") == 0


# ---------------------------------------------------------------------------
# Config honesty: within.required, tenant pins, quota tenant reachability
# ---------------------------------------------------------------------------


def _base_doc():
    return {
        "sim": {"horizon": "1h", "round": "60s", "seed": 0},
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["pod", "node"], "counts": [2, 2]},
                }
            ],
        },
        "workload": {
            "kind": "synthetic",
            "n_tenants": 3,
            "classes": {
                "eval": {
                    "rate_per_hour": 1,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=2m, p90=30m]",
                }
            },
        },
    }


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, "maybe"])
def test_within_required_must_be_a_real_boolean(bad):
    doc = _base_doc()
    doc["workload"]["classes"]["eval"]["within"] = {
        "level": "pod",
        "required": bad,
        "relax_after": "10m",
    }
    errs = validate(load_scenario(doc, strict=False))
    assert any("within.required" in e and "true/false" in e for e in errs), errs


def test_within_required_real_booleans_still_accepted():
    doc = _base_doc()
    doc["workload"]["classes"]["eval"]["within"] = {
        "level": "pod",
        "required": False,
        "relax_after": "10m",
    }
    scenario = load_scenario(doc)
    (cls,) = scenario.workload.classes
    assert cls.within is not None and cls.within.required is False


@pytest.mark.parametrize("bad", ["", "   "])
def test_class_tenant_pin_must_be_nonempty(bad):
    doc = _base_doc()
    doc["workload"]["classes"]["eval"]["tenant"] = bad
    errs = validate(load_scenario(doc, strict=False))
    assert any("tenant" in e and "non-empty" in e for e in errs), errs


def test_quota_unreachable_tenant_is_dead_config():
    doc = _base_doc()
    doc["quota"] = {"tenants": {"tenant0": 64}}  # typo: Zipf yields t0..t2
    errs = validate(load_scenario(doc, strict=False))
    assert any(
        "quota.tenants.tenant0" in e and "dead config" in e for e in errs
    ), errs


def test_quota_reachable_tenants_accepted():
    doc = _base_doc()
    doc["workload"]["classes"]["train"] = {
        "class": "finetune",
        "rate_per_hour": 1,
        "chips": "pow2[8, 16]",
        "duration": "lognormal[median=1h, p90=4h]",
        "tenant": "acme",
    }
    doc["quota"] = {"tenants": {"t0": 8, "t2": 8, "acme": 16}}
    assert validate(load_scenario(doc, strict=False)) == []


def test_quota_zipf_edge_names_rejected():
    doc = _base_doc()  # n_tenants: 3 -> t0..t2 reachable
    doc["quota"] = {"tenants": {"t3": 8}}  # one past the end
    errs = validate(load_scenario(doc, strict=False))
    assert any("quota.tenants.t3" in e for e in errs), errs
    doc["quota"] = {"tenants": {"t01": 8}}  # zero-padded: never emitted
    errs = validate(load_scenario(doc, strict=False))
    assert any("quota.tenants.t01" in e for e in errs), errs
