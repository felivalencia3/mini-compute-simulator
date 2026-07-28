"""Tests for fleetsim.model: enum vocabularies, defaults, invariant plumbing."""

import dataclasses

import pytest

from fleetsim.model import (
    Allocation,
    CapacityClass,
    ChipType,
    Constraint,
    Domain,
    GangAlloc,
    GangSpec,
    Job,
    JobClass,
    JobStatus,
    NodeState,
    PreemptMode,
    Service,
    Tier,
)


class TestEnums:
    def test_node_state_members(self):
        assert {m.name for m in NodeState} == {
            "HEALTHY",
            "DRAINING",
            "FAILED",
            "MAINTENANCE",
        }

    def test_preempt_mode_members(self):
        # SUSPEND is deliberately absent (undefined on GPUs)
        assert {m.name for m in PreemptMode} == {"CANCEL", "REQUEUE"}

    def test_job_class_members(self):
        assert {m.name for m in JobClass} == {
            "PRETRAIN",
            "FINETUNE",
            "EVAL",
            "INFER_REPLICA",
        }

    def test_tier_is_ordered_int_enum(self):
        assert Tier.FREE == 0
        assert Tier.BATCH == 1
        assert Tier.PROD == 2
        assert Tier.MONITORING == 3
        assert Tier.FREE < Tier.BATCH < Tier.PROD < Tier.MONITORING

    def test_capacity_class_members(self):
        assert {m.name for m in CapacityClass} == {
            "RESERVED",
            "ON_DEMAND",
            "SPOT",
            "FLEX_START",
            "CALENDAR",
        }

    def test_job_status_members(self):
        assert {m.name for m in JobStatus} == {
            "PENDING",
            "ADMITTED",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELED",
            "TIMEOUT",
            "NODE_FAIL",
            "PREEMPTED",
        }


class TestChipType:
    def test_frozen(self):
        ct = ChipType(name="h100", vendor="nvidia", hbm_gib=80, peak_tflops_bf16=989)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ct.name = "h200"

    def test_generation_defaults_to_1(self):
        ct = ChipType(name="h100", vendor="nvidia", hbm_gib=80, peak_tflops_bf16=989)
        assert ct.generation == 1


class TestDomain:
    def test_defaults(self):
        d = Domain(id="n0", level="node", parent="rack0", children=[], chip_type="h100")
        assert d.chips == 0
        assert d.state is NodeState.HEALTHY
        assert d.lemon_factor == 1.0
        assert d.total_chips == 0
        assert d.free_chips == 0
        assert d.attrs == {}

    def test_attrs_not_shared_between_instances(self):
        a = Domain(id="a", level="node", parent=None, children=[], chip_type=None)
        b = Domain(id="b", level="node", parent=None, children=[], chip_type=None)
        a.attrs["pooled"] = True
        assert b.attrs == {}

    def test_slots(self):
        d = Domain(id="n0", level="node", parent=None, children=[], chip_type=None)
        assert not hasattr(d, "__dict__")
        with pytest.raises(AttributeError):
            d.bogus_field = 1


class TestConstraintAndGang:
    def test_constraint_defaults(self):
        c = Constraint(level="pod")
        assert c.required is True
        assert c.relax_after_s == 300.0

    def test_gang_spec_defaults(self):
        g = GangSpec(chips=8)
        assert g.chip_type is None
        assert g.within is None
        assert g.segments is None
        assert g.shape is None
        assert g.twisted is False

    def test_gang_alloc_whole_node_and_sub_node_forms(self):
        whole = GangAlloc(nodes=["n0", "n1"], anchor="rack0")
        sub = GangAlloc(nodes={"n0": 2}, anchor="n0")
        assert whole.relaxed is False
        assert whole.attrs == {}
        assert sub.nodes == {"n0": 2}

    def test_allocation(self):
        alloc = Allocation(job_id="j1", gangs=[GangAlloc(nodes=["n0"], anchor="n0")])
        assert alloc.job_id == "j1"
        assert len(alloc.gangs) == 1


class TestJob:
    def _job(self, **kw):
        base = dict(
            id="j1",
            tenant="t0",
            job_class=JobClass.EVAL,
            submit_t=0,
            gangs=[GangSpec(chips=1, chip_type="h100")],
            tier=Tier.BATCH,
        )
        base.update(kw)
        return Job(**base)

    def test_defaults_match_design(self):
        j = self._job()
        assert j.capacity is CapacityClass.ON_DEMAND
        assert j.preemptible is True
        assert j.min_runtime_s == 0.0
        assert j.max_lifetime_s is None
        assert j.walltime_est_s is None
        assert j.true_duration_s == 0.0
        assert j.checkpoint_interval_s == 3600.0
        assert j.checkpoint_save_s == 60.0
        assert j.restart_overhead_s == 900.0
        assert j.valid_until is None
        assert j.service_id is None
        assert j.status is JobStatus.PENDING
        assert j.attained_service_chip_s == 0.0
        assert j.goodput_chip_s == 0.0

    def test_submit_t_is_int_us(self):
        j = self._job(submit_t=5_000_000)
        assert isinstance(j.submit_t, int)

    def test_slots(self):
        j = self._job()
        assert not hasattr(j, "__dict__")
        with pytest.raises(AttributeError):
            j.extra = 1

    def test_engine_mutable_fields(self):
        # status / attained service / goodput are the engine's to mutate
        j = self._job()
        j.status = JobStatus.RUNNING
        j.attained_service_chip_s += 10.0
        j.goodput_chip_s += 8.0
        assert j.status is JobStatus.RUNNING


class TestService:
    def test_defaults(self):
        s = Service(
            id="svc1",
            tenant="t0",
            replica_spec=GangSpec(chips=8, chip_type="h100"),
            min_replicas=3,
            max_replicas=3,
        )
        assert s.tier is Tier.PROD
        assert s.load is None
