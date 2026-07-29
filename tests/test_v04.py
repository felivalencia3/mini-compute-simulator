"""v0.4 feature mechanics: SPOT zero-notice grace, engine relax gating,
reservation-hold search filtering and Place refusal, quota reject mode,
and the --version flag.  (The end-to-end behavior of each feature is
validated in the validation/ rungs; these are the unit-level guardrails.)
"""

import pytest

from fleetsim.config import load_scenario
from fleetsim.engine.sim import QuotaAdmission, Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.fleet.tree import Placement
from fleetsim.model import (
    CapacityClass,
    Constraint,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    PreemptMode,
    Tier,
)
from fleetsim.schedulers.base import Place, Preempt, Scheduler, get_scheduler
from fleetsim.workload.base import ListSource

S = 1_000_000


def make_scenario(quota=None, penalties=None, horizon="1h"):
    doc = {
        "sim": {"horizon": horizon, "round": "60s", "seed": 0},
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
    if quota is not None:
        doc["quota"] = quota
    if penalties is not None:
        doc["penalties"] = penalties
    return load_scenario(doc)


def mk_job(jid, chips=8, dur_s=600.0, tenant="t0", capacity=CapacityClass.ON_DEMAND,
           tier=Tier.BATCH, within=None, save_s=60.0):
    return Job(
        id=jid,
        tenant=tenant,
        job_class=JobClass.FINETUNE,
        submit_t=0,
        gangs=[GangSpec(chips=chips, chip_type="h100", within=within)],
        tier=tier,
        capacity=capacity,
        true_duration_s=dur_s,
        checkpoint_interval_s=0.0,
        checkpoint_save_s=save_s,
        restart_overhead_s=0.0,
    )


class RecordingSink:
    def __init__(self):
        self.calls = []

    def job_preempted(self, job, t, trigger):
        self.calls.append(("preempted", job.id, t, trigger))

    def job_requeued(self, job, t):
        self.calls.append(("requeued", job.id, t))

    def job_finished(self, job, t, status, p, l):
        self.calls.append(("finished", job.id, t, status.name))

    def __getattr__(self, name):
        return lambda *a, **k: None


class PreemptOnce(Scheduler):
    """Places everything FIFO; at the first wake >= trigger_us preempts
    the named victim once (REQUEUE)."""

    def __init__(self, victim, trigger_us):
        from fleetsim.schedulers.placement import FirstFit

        self.placement = FirstFit()
        self.victim = victim
        self.trigger_us = trigger_us
        self.done = False

    def schedule(self, view):
        actions = []
        if not self.done and view.now >= self.trigger_us:
            running = {j.id for j in view.running()}
            if self.victim in running:
                actions.append(Preempt(self.victim, PreemptMode.REQUEUE))
                self.done = True
        for job in view.pending():
            p = view.find_placement(job, self.placement)
            if p is not None:
                actions.append(Place(job.id, p))
        return actions


def test_spot_preemption_has_zero_grace():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    sink = RecordingSink()
    spot = mk_job("spot", chips=8, capacity=CapacityClass.SPOT, save_s=60.0)
    sim = Simulator(
        scenario, fleet, ListSource([spot]),
        PreemptOnce("spot", trigger_us=120 * S), sink,
    )
    sim.run()
    pre = [c for c in sink.calls if c[0] == "preempted"]
    req = [c for c in sink.calls if c[0] == "requeued"]
    assert pre and req
    # Zero-notice: requeue at the SAME timestamp as the preemption,
    # despite checkpoint_save_s = 60 (an on-demand job would wait 60 s).
    assert req[0][2] == pre[0][2]


def test_on_demand_preemption_keeps_checkpoint_save_grace():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    sink = RecordingSink()
    job = mk_job("od", chips=8, capacity=CapacityClass.ON_DEMAND, save_s=60.0)
    sim = Simulator(
        scenario, fleet, ListSource([job]),
        PreemptOnce("od", trigger_us=120 * S), sink,
    )
    sim.run()
    pre = [c for c in sink.calls if c[0] == "preempted"]
    req = [c for c in sink.calls if c[0] == "requeued"]
    assert req[0][2] == pre[0][2] + 60 * S


class PlaceRelaxedTooEarly(Scheduler):
    """Emits a hand-built RELAXED cross-pod placement at t=0 for a job
    whose relax_after has not elapsed — the engine must refuse it."""

    def schedule(self, view):
        for job in view.pending():
            leaves = (
                ("m/c/pod0/node0", 8), ("m/c/pod0/node1", 8),
                ("m/c/pod1/node0", 8),
            )
            placement = Placement(
                leaves=leaves, anchor="m/c", chip_type="h100",
                whole_node=True, relaxed=True,
            )
            return [Place(job.id, placement)]
        return []


def test_engine_refuses_relaxed_placement_before_timeout():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    job = mk_job(
        "j", chips=24,
        within=Constraint(level="pod", required=False, relax_after_s=600.0),
    )
    sim = Simulator(
        scenario, fleet, ListSource([job]), PlaceRelaxedTooEarly(),
        RecordingSink(),
    )
    with pytest.raises(ValueError, match="may only relax after"):
        sim.run()


def test_engine_refuses_relaxed_placement_for_hard_constraint():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    job = mk_job("j", chips=24, within=Constraint(level="pod", required=True))
    sim = Simulator(
        scenario, fleet, ListSource([job]), PlaceRelaxedTooEarly(),
        RecordingSink(),
    )
    with pytest.raises(ValueError, match="without a relaxable"):
        sim.run()


def test_reserved_leaves_filtered_from_search_and_place_refused():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    fleet.reserve_leaves(["m/c/pod0/node0", "m/c/pod0/node1"], "acme")

    # Non-owner searches skip the hold entirely.
    spec = GangSpec(chips=16, chip_type="h100")
    p = fleet.search_first_fit(spec, "someone_else")
    assert p is not None
    assert all(lid.startswith("m/c/pod1/") for lid, _ in p.leaves)
    # 32 chips can no longer be found for a non-owner...
    assert fleet.search_first_fit(GangSpec(chips=32, chip_type="h100"),
                                  "someone_else") is None
    # ...but the owner sees the whole fleet.
    assert fleet.search_first_fit(GangSpec(chips=32, chip_type="h100"),
                                  "acme") is not None
    # Segmented search honors holds too (2 x 2-node pod segments need
    # both pods; pod0 has only reserved nodes free for non-owners).
    seg = GangSpec(chips=32, chip_type="h100", segments=(2, "pod"))
    assert fleet.search_segmented(seg, "someone_else") is None
    assert fleet.search_segmented(seg, "acme") is not None
    fleet.release_reservation(["m/c/pod0/node0", "m/c/pod0/node1"])
    assert fleet.search_first_fit(spec, "someone_else") is not None


class PlaceOnHold(Scheduler):
    def schedule(self, view):
        for job in view.pending():
            placement = Placement(
                leaves=(("m/c/pod0/node0", 8),), anchor="m/c",
                chip_type="h100", whole_node=True,
            )
            return [Place(job.id, placement)]
        return []


def test_engine_refuses_non_owner_place_on_hold():
    scenario = make_scenario()
    fleet = build_fleet(scenario)
    fleet.reserve_leaves(["m/c/pod0/node0"], "acme")
    job = mk_job("j", chips=8, tenant="not_acme")
    sim = Simulator(
        scenario, fleet, ListSource([job]), PlaceOnHold(), RecordingSink()
    )
    with pytest.raises(ValueError, match="reserved for tenant"):
        sim.run()


def test_quota_reject_mode_fails_over_quota_jobs():
    scenario = make_scenario(
        quota={"tenants": {"t0": 8}, "over_quota": "reject"}
    )
    fleet = build_fleet(scenario)
    sink = RecordingSink()
    jobs = [mk_job("a", chips=8, dur_s=3000.0), mk_job("b", chips=8, dur_s=60.0)]
    admission = QuotaAdmission(scenario.quota)
    sim = Simulator(
        scenario, fleet, ListSource(jobs), get_scheduler("fifo", {}), sink,
        admission,
    )
    sim.run()
    finished = {c[1]: c[3] for c in sink.calls if c[0] == "finished"}
    assert finished["b"] == "FAILED"  # rejected at admission
    assert finished["a"] == "COMPLETED"
    assert not jobs[1].quota_demoted  # reject mode never demotes


def test_quota_commitment_releases_at_terminal():
    quota = {"tenants": {"t0": 8}, "over_quota": "best_effort"}
    scenario = make_scenario(quota=quota, horizon="2h")
    fleet = build_fleet(scenario)
    admission = QuotaAdmission(scenario.quota)
    # a fills the cap; b (arrives same wake) demotes; after a finishes,
    # c (arrives later) is in-quota again.
    a = mk_job("a", chips=8, dur_s=100.0)
    b = mk_job("b", chips=8, dur_s=100.0)
    c = mk_job("c", chips=8, dur_s=100.0)
    c.submit_t = 600 * S
    sim = Simulator(
        scenario, fleet, ListSource([a, b, c]), get_scheduler("fifo", {}),
        RecordingSink(), admission,
    )
    sim.run()
    assert not a.quota_demoted and a.tier is Tier.BATCH
    assert b.quota_demoted and b.tier is Tier.BEST_EFFORT
    assert not c.quota_demoted
    assert admission.committed("t0") == 0  # all commitments released


def test_version_flag(capsys):
    import fleetsim
    from fleetsim.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"fleetsim {fleetsim.__version__}" in capsys.readouterr().out
