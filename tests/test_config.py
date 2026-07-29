"""Tests for fleetsim.config: both fleet YAML forms, dist expressions,
validation, and v0.1 feature gating."""

import copy

import pytest
import yaml

from fleetsim.config import (
    DistSpec,
    ScenarioError,
    load_scenario,
    parse_dist,
    validate,
)
from fleetsim.model import CapacityClass, JobClass, Tier
from fleetsim.units import DAY, HOUR, MIN, S, WEEK

# ---------------------------------------------------------------------------
# Fixture documents
# ---------------------------------------------------------------------------

COMPACT_YAML = """
sim: {horizon: 14d, round: 60s, seed: 42}
fleet:
  metro: us-central
  clusters:
    - name: h100-main
      chip: {type: h100, per_node: 8}
      topology: {levels: [pod, rack, node], counts: [2, 16, 8]}   # 2,048 chips
      failures: {mtbf_node_days: 42, repair_auto_min: [60, 180]}
workload:
  kind: synthetic
  classes:
    pretrain:
      rate_per_week: 2
      chips: pow2[256, 2048]
      duration: lognormal[median=10d, p90=30d]
      tier: prod
      checkpoint_interval: 1h
      min_runtime: 2h
      within: pod
    finetune:
      rate_per_day: 30
      chips: pow2[8, 64]
      duration: lognormal[median=4h, p90=24h]
      tier: batch
    eval:
      rate_per_hour: 40
      chips: pow2[1, 8]
      duration: lognormal[median=2m, p90=30m]
      tier: batch
      diurnal: true
scheduler: {name: tiered_priority, params: {preempt: requeue}}
outputs: {events: parquet, plots: true}
"""

TEMPLATE_YAML = """
sim: {horizon: 14d, round: 60s, seed: 42}
chip_types:
  h100:    {vendor: nvidia, hbm_gib: 80,  peak_tflops_bf16: 989}
  tpu_v5p: {vendor: google, hbm_gib: 95,  peak_tflops_bf16: 459}
templates:
  h100_node: {level: node, chips: 8, chip_type: h100, attrs: {pooled: true}}
  h100_su:   {level: su,  attrs: {rails: 8},
              children: {template: h100_node, count: 32}}
  h100_pod:  {level: pod, attrs: {oversub: 7, xover_bw_gbps: 400},
              children: {template: h100_su, count: 16}}
  v5p_host:  {level: host, chips: 4, chip_type: tpu_v5p}
  v5p_cube:  {level: cube, attrs: {geometry: [4, 4, 4]},
              children: {template: v5p_host, count: 16}}
  v5p_pod:   {level: pod, attrs: {dcn_gbps_per_host: 50},
              children: {template: v5p_cube, count: 140}}
fleet:
  - metro: us-east
    datacenters:
      - id: dc1
        clusters:
          - id: hopper-a
            levels: [cluster, pod, su, node]
            children: [{template: h100_pod, count: 4}]
          - id: tpu-a
            levels: [cluster, pod, cube, host]
            children: [{template: v5p_pod, count: 2}]
failure_model:
  node_mtbf_days: 42
  repair: {auto_min: [60, 180], manual_frac: 0.1, manual_days: [1, 3]}
  maintenance_rate_per_node_month: 1.0
  drain_grace: 1h
workload:
  kind: synthetic
  classes:
    eval:
      rate_per_hour: 40
      chips: pow2[1, 8]
      chip_type: h100     # heterogeneous fleet: classes must pin (DESIGN 11)
      duration: lognormal[median=2m, p90=30m]
      tier: batch
scheduler: {name: fifo}
outputs: {events: parquet}
"""

# Template-form fleet describing exactly the same cluster as COMPACT_YAML.
TEMPLATE_EQUIV_YAML = """
sim: {horizon: 14d, round: 60s, seed: 42}
chip_types:
  h100: {vendor: nvidia, hbm_gib: 80, peak_tflops_bf16: 989}
templates:
  node_t: {level: node, chips: 8, chip_type: h100}
  rack_t: {level: rack, children: {template: node_t, count: 8}}
  pod_t:  {level: pod,  children: {template: rack_t, count: 16}}
fleet:
  - metro: us-central
    datacenters:
      - id: dc1
        clusters:
          - id: h100-main
            levels: [cluster, pod, rack, node]
            children: [{template: pod_t, count: 2}]
workload:
  kind: synthetic
  classes:
    eval:
      rate_per_hour: 40
      chips: pow2[1, 8]
      duration: lognormal[median=2m, p90=30m]
"""


def compact_doc() -> dict:
    return yaml.safe_load(COMPACT_YAML)


# ---------------------------------------------------------------------------
# Distribution expressions
# ---------------------------------------------------------------------------


class TestParseDist:
    def test_pow2_positional(self):
        assert parse_dist("pow2[1, 8]") == DistSpec("pow2", {"lo": 1, "hi": 8})

    def test_uniform_positional(self):
        assert parse_dist("uniform[4, 64]") == DistSpec("uniform", {"lo": 4, "hi": 64})

    def test_fixed(self):
        assert parse_dist("fixed[8]") == DistSpec("fixed", {"value": 8})

    def test_lognormal_kwargs_with_durations(self):
        d = parse_dist("lognormal[median=2m, p90=30m]")
        assert d == DistSpec("lognormal", {"median": 2 * MIN, "p90": 30 * MIN})

    def test_exponential_duration_mean(self):
        assert parse_dist("exponential[mean=30s]") == DistSpec(
            "exponential", {"mean": 30 * S}
        )

    def test_bare_number_becomes_fixed(self):
        assert parse_dist(8) == DistSpec("fixed", {"value": 8})
        assert parse_dist(2.5) == DistSpec("fixed", {"value": 2.5})
        assert parse_dist("8") == DistSpec("fixed", {"value": 8})

    def test_bare_duration_string_becomes_fixed_us(self):
        assert parse_dist("2m") == DistSpec("fixed", {"value": 2 * MIN})

    def test_plain_numbers_stay_numbers(self):
        # chip counts must not be reinterpreted as durations
        d = parse_dist("pow2[256, 2048]")
        assert d.params == {"lo": 256, "hi": 2048}
        assert all(type(v) is int for v in d.params.values())

    def test_float_params(self):
        assert parse_dist("uniform[0.5, 1.5]") == DistSpec(
            "uniform", {"lo": 0.5, "hi": 1.5}
        )

    def test_whitespace_tolerant(self):
        assert parse_dist("  pow2[ 1 , 8 ]  ") == DistSpec("pow2", {"lo": 1, "hi": 8})

    def test_unknown_kind_parses_with_positional_names(self):
        d = parse_dist("zipf[1, 8]")
        assert d.kind == "zipf"
        assert d.params == {"p0": 1, "p1": 8}

    @pytest.mark.parametrize(
        "bad",
        [
            "pow2[",
            "pow2[1,]",
            "pow2[,1]",
            "[1, 8]",
            "pow2[a, b]",
            "lognormal[=2m]",
            "hello world",
            "",
            None,
            True,
        ],
    )
    def test_malformed_raises(self, bad):
        with pytest.raises(ValueError):
            parse_dist(bad)

    def test_duplicate_param_raises(self):
        with pytest.raises(ValueError):
            parse_dist("uniform[lo=1, lo=2]")


# ---------------------------------------------------------------------------
# Compact form (DESIGN section 13)
# ---------------------------------------------------------------------------


class TestCompactForm:
    def test_loads_strict(self):
        s = load_scenario(compact_doc(), strict=True)
        assert validate(s) == []

    def test_sim(self):
        s = load_scenario(compact_doc())
        assert s.sim.horizon_us == 14 * DAY
        assert s.sim.round_us == 60 * S
        assert s.sim.seed == 42

    def test_fleet_expansion(self):
        s = load_scenario(compact_doc())
        clusters = s.fleet.clusters()
        assert [c.id for c in clusters] == ["h100-main"]
        cl = clusters[0]
        assert cl.levels == ["cluster", "pod", "rack", "node"]
        assert cl.total_chips() == 2048
        assert cl.total_nodes() == 2 * 16 * 8
        # tree shape: pod(2) -> rack(16) -> node(8, 8 chips)
        pod = cl.children[0]
        assert (pod.level, pod.count) == ("pod", 2)
        rack = pod.children[0]
        assert (rack.level, rack.count) == ("rack", 16)
        node = rack.children[0]
        assert (node.level, node.count, node.chips, node.chip_type) == (
            "node",
            8,
            8,
            "h100",
        )

    def test_undeclared_chip_type_auto_registered(self):
        s = load_scenario(compact_doc())
        assert "h100" in s.fleet.chip_types
        assert s.fleet.chip_types["h100"].vendor == "unknown"

    def test_cluster_failure_model_overrides_and_inherits(self):
        s = load_scenario(compact_doc())
        fm = s.fleet.clusters()[0].failure_model
        assert fm.node_mtbf_days == 42.0
        assert fm.repair_auto_min == (60.0, 180.0)
        # unset fields inherit defaults
        assert fm.repair_manual_frac == 0.1
        assert fm.repair_manual_days == (1.0, 3.0)
        assert fm.maintenance_rate_per_node_month == 1.0
        assert fm.drain_grace_us == HOUR

    def test_workload_classes(self):
        s = load_scenario(compact_doc())
        by_name = {c.name: c for c in s.workload.classes}
        assert list(by_name) == ["pretrain", "finetune", "eval"]  # document order

        pre = by_name["pretrain"]
        assert pre.job_class is JobClass.PRETRAIN
        assert pre.rate_per_hour == pytest.approx(2 / 168)
        assert pre.chips == DistSpec("pow2", {"lo": 256, "hi": 2048})
        assert pre.duration == DistSpec(
            "lognormal", {"median": 10 * DAY, "p90": 30 * DAY}
        )
        assert pre.tier is Tier.PROD
        assert pre.checkpoint_interval_s == 3600.0
        assert pre.min_runtime_s == 7200.0
        assert pre.within is not None and pre.within.level == "pod"
        assert pre.within.required is True
        assert pre.abort_prob == 0.3  # DESIGN 5.1 default: 30-40% aborts
        assert pre.max_lifetime_s is None  # no cap unless configured
        assert pre.n_tenants == 8
        assert pre.chip_type is None  # homogeneous fleet: pin optional
        assert pre.capacity is CapacityClass.ON_DEMAND
        assert pre.n_gangs == 1

        fin = by_name["finetune"]
        assert fin.rate_per_hour == pytest.approx(30 / 24)
        assert fin.job_class is JobClass.FINETUNE
        assert fin.min_runtime_s == 0.0  # per-class default
        assert fin.within is None
        assert fin.diurnal is False

        ev = by_name["eval"]
        assert ev.rate_per_hour == 40.0
        assert ev.diurnal is True
        assert ev.tier is Tier.BATCH

    def test_scheduler_and_outputs(self):
        s = load_scenario(compact_doc())
        assert s.scheduler.name == "tiered_priority"
        assert s.scheduler.params == {"preempt": "requeue"}
        assert s.outputs.events == "parquet"
        assert s.outputs.plots is True


# ---------------------------------------------------------------------------
# Template form (DESIGN section 3.3)
# ---------------------------------------------------------------------------


class TestTemplateForm:
    def test_loads_strict(self):
        s = load_scenario(yaml.safe_load(TEMPLATE_YAML), strict=True)
        assert validate(s) == []

    def test_chip_types_registered(self):
        s = load_scenario(yaml.safe_load(TEMPLATE_YAML))
        assert set(s.fleet.chip_types) == {"h100", "tpu_v5p"}
        h100 = s.fleet.chip_types["h100"]
        assert h100.vendor == "nvidia"
        assert h100.hbm_gib == 80.0
        assert h100.peak_tflops_bf16 == 989.0
        assert h100.generation == 1

    def test_expansion_totals(self):
        s = load_scenario(yaml.safe_load(TEMPLATE_YAML))
        by_id = {c.id: c for c in s.fleet.clusters()}
        assert list(by_id) == ["hopper-a", "tpu-a"]
        assert by_id["hopper-a"].total_chips() == 4 * 16 * 32 * 8  # 16,384
        assert by_id["hopper-a"].total_nodes() == 4 * 16 * 32
        assert by_id["tpu-a"].total_chips() == 2 * 140 * 16 * 4  # 17,920

    def test_levels_and_attrs_carried(self):
        s = load_scenario(yaml.safe_load(TEMPLATE_YAML))
        hopper = s.fleet.clusters()[0]
        assert hopper.levels == ["cluster", "pod", "su", "node"]
        pod = hopper.children[0]
        assert pod.attrs == {"oversub": 7, "xover_bw_gbps": 400}
        su = pod.children[0]
        assert su.attrs == {"rails": 8}
        node = su.children[0]
        assert node.attrs == {"pooled": True}  # pooled is fine in v0.1
        assert (node.chips, node.chip_type, node.count) == (8, "h100", 32)

    def test_metro_datacenter_structure(self):
        s = load_scenario(yaml.safe_load(TEMPLATE_YAML))
        assert [m.name for m in s.fleet.metros] == ["us-east"]
        assert [d.id for d in s.fleet.metros[0].datacenters] == ["dc1"]

    def test_global_failure_model(self):
        s = load_scenario(yaml.safe_load(TEMPLATE_YAML))
        fm = s.failure_model
        assert fm.node_mtbf_days == 42.0
        assert fm.repair_auto_min == (60.0, 180.0)
        assert fm.repair_manual_frac == 0.1
        assert fm.repair_manual_days == (1.0, 3.0)
        assert fm.maintenance_rate_per_node_month == 1.0
        assert fm.drain_grace_us == HOUR
        # clusters inherit the global model when they set nothing
        assert s.fleet.clusters()[0].failure_model == fm


class TestBothFormsEquivalent:
    def test_same_internal_representation(self):
        compact = load_scenario(compact_doc()).fleet.clusters()[0]
        template = load_scenario(
            yaml.safe_load(TEMPLATE_EQUIV_YAML)
        ).fleet.clusters()[0]
        assert compact.id == template.id == "h100-main"
        assert compact.levels == template.levels
        assert compact.total_chips() == template.total_chips() == 2048
        assert compact.total_nodes() == template.total_nodes() == 256

        def shape(groups):
            return [
                (g.level, g.count, g.chips, g.chip_type, shape(g.children))
                for g in groups
            ]

        assert shape(compact.children) == shape(template.children)


# ---------------------------------------------------------------------------
# Loading mechanics
# ---------------------------------------------------------------------------


class TestLoadScenario:
    def test_from_file_path(self, tmp_path):
        p = tmp_path / "scenario.yaml"
        p.write_text(COMPACT_YAML, encoding="utf-8")
        from_file = load_scenario(p)
        from_dict = load_scenario(compact_doc())
        assert from_file.sim == from_dict.sim
        assert from_file.fleet.clusters()[0].total_chips() == 2048
        # str path works too
        assert load_scenario(str(p)).sim.seed == 42

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ScenarioError):
            load_scenario(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("a: [unclosed", encoding="utf-8")
        with pytest.raises(ScenarioError):
            load_scenario(p)

    def test_non_mapping_document_raises_even_non_strict(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ScenarioError):
            load_scenario(p, strict=False)

    def test_strict_raises_with_error_list(self):
        # capacity: spot became legal in v0.4; twisted TPU slices remain
        # the canonical not-yet-implemented trigger.
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["twisted"] = True
        with pytest.raises(ScenarioError) as exc_info:
            load_scenario(doc, strict=True)
        assert any("not implemented in v0.1" in e for e in exc_info.value.errors)

    def test_non_strict_returns_scenario(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["twisted"] = True
        s = load_scenario(doc, strict=False)
        errors = validate(s)
        assert any("not implemented in v0.1" in e for e in errors)

    def test_input_dict_not_mutated(self):
        doc = compact_doc()
        snapshot = copy.deepcopy(doc)
        load_scenario(doc)
        assert doc == snapshot

    def test_defaults_when_sections_missing(self):
        s = load_scenario({"sim": {"horizon": "1d"}}, strict=False)
        assert s.scheduler.name == "fifo"
        assert s.scheduler.params == {}
        assert s.outputs.events == "parquet"
        assert s.outputs.plots is False
        errors = validate(s)
        assert any(e.startswith("fleet:") for e in errors)
        assert any(e.startswith("workload") for e in errors)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def errors_for(doc) -> list[str]:
    return validate(load_scenario(doc, strict=False))


class TestValidation:
    def test_valid_scenario_no_errors(self):
        assert errors_for(compact_doc()) == []

    def test_unknown_within_level(self):
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["within"] = "superpod"
        errs = errors_for(doc)
        assert any("superpod" in e and "unknown level" in e for e in errs)

    def test_known_within_levels_include_cluster_and_node(self):
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["within"] = "cluster"
        assert errors_for(doc) == []
        doc["workload"]["classes"]["pretrain"]["within"] = "node"
        assert errors_for(doc) == []

    def test_non_positive_topology_count(self):
        doc = compact_doc()
        doc["fleet"]["clusters"][0]["topology"]["counts"] = [2, 0, 8]
        errs = errors_for(doc)
        assert any("count must be positive" in e for e in errs)

    def test_non_positive_template_count(self):
        doc = yaml.safe_load(TEMPLATE_YAML)
        doc["fleet"][0]["datacenters"][0]["clusters"][0]["children"][0]["count"] = -1
        errs = errors_for(doc)
        assert any("count must be positive" in e for e in errs)

    def test_pow2_bounds_must_be_powers_of_two(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["chips"] = "pow2[3, 8]"
        errs = errors_for(doc)
        assert any("power" in e and "two" in e for e in errs)

    def test_pow2_lo_must_not_exceed_hi(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["chips"] = "pow2[16, 8]"
        errs = errors_for(doc)
        assert any("lo <= hi" in e for e in errs)

    def test_unknown_dist_kind(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["chips"] = "zipf[1, 8]"
        errs = errors_for(doc)
        assert any("unknown distribution kind" in e and "zipf" in e for e in errs)

    def test_missing_dist_params(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["duration"] = "lognormal[median=2m]"
        errs = errors_for(doc)
        assert any("requires parameter" in e and "p90" in e for e in errs)

    def test_lognormal_p90_below_median(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["duration"] = "lognormal[median=30m, p90=2m]"
        errs = errors_for(doc)
        assert any("p90 >= median" in e for e in errs)

    def test_exponential_mean_positive(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["duration"] = "exponential[mean=0]"
        errs = errors_for(doc)
        assert any("must be positive" in e for e in errs)

    def test_unknown_scheduler_name_is_not_an_error(self):
        doc = compact_doc()
        doc["scheduler"]["name"] = "totally_bogus_policy"
        assert errors_for(doc) == []

    def test_missing_arrival_rate(self):
        doc = compact_doc()
        del doc["workload"]["classes"]["eval"]["rate_per_hour"]
        errs = errors_for(doc)
        assert any("exactly one of" in e for e in errs)

    def test_multiple_arrival_rates(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["rate_per_day"] = 5
        errs = errors_for(doc)
        assert any("exactly one of" in e for e in errs)

    def test_negative_rate(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["rate_per_hour"] = -1
        errs = errors_for(doc)
        assert any("must be positive" in e for e in errs)

    def test_abort_prob_range(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["abort_prob"] = 1.5
        errs = errors_for(doc)
        assert any("abort_prob" in e for e in errs)

    def test_trace_requires_source(self):
        doc = compact_doc()
        doc["workload"] = {"kind": "trace"}
        errs = errors_for(doc)
        assert any("source" in e for e in errs)

    def test_trace_with_source_is_valid(self):
        doc = compact_doc()
        doc["workload"] = {"kind": "trace", "source": "traces/philly.csv"}
        s = load_scenario(doc, strict=True)
        assert s.workload.kind == "trace"
        assert s.workload.source == "traces/philly.csv"
        assert s.workload.classes == []

    def test_synthetic_requires_classes(self):
        doc = compact_doc()
        doc["workload"] = {"kind": "synthetic"}
        errs = errors_for(doc)
        assert any("at least one class" in e for e in errs)

    def test_unknown_job_class_name(self):
        doc = compact_doc()
        doc["workload"]["classes"]["mystery"] = {
            "rate_per_hour": 1,
            "chips": "fixed[1]",
            "duration": "fixed[60]",
        }
        errs = errors_for(doc)
        assert any("unknown job class" in e and "mystery" in e for e in errs)

    def test_explicit_class_key_maps_name(self):
        doc = compact_doc()
        doc["workload"]["classes"]["nightly-evals"] = {
            "class": "eval",
            "rate_per_hour": 1,
            "chips": "fixed[1]",
            "duration": "fixed[60]",
        }
        s = load_scenario(doc, strict=True)
        by_name = {c.name: c for c in s.workload.classes}
        assert by_name["nightly-evals"].job_class is JobClass.EVAL

    def test_unknown_class_key_flagged(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["chps"] = "pow2[1, 8]"  # typo
        errs = errors_for(doc)
        assert any("unknown key" in e and "chps" in e for e in errs)

    def test_missing_horizon(self):
        doc = compact_doc()
        del doc["sim"]["horizon"]
        errs = errors_for(doc)
        assert any("sim.horizon" in e for e in errs)

    def test_n_tenants_defaults_and_overrides(self):
        doc = compact_doc()
        doc["workload"]["n_tenants"] = 4
        doc["workload"]["classes"]["eval"]["n_tenants"] = 16
        s = load_scenario(doc, strict=True)
        by_name = {c.name: c for c in s.workload.classes}
        assert by_name["pretrain"].n_tenants == 4  # inherits workload default
        assert by_name["eval"].n_tenants == 16  # class override

    def test_declared_registry_flags_unknown_chip_type(self):
        doc = yaml.safe_load(TEMPLATE_YAML)
        del doc["chip_types"]["tpu_v5p"]
        errs = errors_for(doc)
        assert any("unknown chip_type" in e and "tpu_v5p" in e for e in errs)


class TestNotImplementedInV01:
    """DESIGN principle 5: schema accepts, validate rejects loudly."""

    MARKER = "not implemented in v0.1"

    def check(self, mutate, needle: str | None = None):
        doc = compact_doc()
        mutate(doc)
        errs = errors_for(doc)
        matching = [e for e in errs if self.MARKER in e]
        assert matching, f"expected a '{self.MARKER}' error, got: {errs}"
        if needle is not None:
            assert any(needle in e for e in matching)

    def test_capacity_class_spot_is_fine_in_v04(self):
        # SPOT (zero-notice kill + checkpoint restart) landed in v0.4.
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["capacity"] = "spot"
        assert errors_for(doc) == []

    def test_capacity_class_reserved(self):
        # reserved/flex_start/calendar job capacity classes remain
        # unimplemented in v0.4 (CALENDAR capacity is the top-level
        # `reservations` section, not a per-class knob).
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["capacity"] = "reserved"
        errs = errors_for(doc)
        assert any(
            "not implemented in v0.4" in e and "reserved" in e for e in errs
        )

    def test_capacity_on_demand_is_fine(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["capacity"] = "on_demand"
        assert errors_for(doc) == []

    def test_multi_gang_jobs(self):
        self.check(
            lambda d: d["workload"]["classes"]["pretrain"].update(gangs=2),
            "multi-gang",
        )

    def test_tpu_shape(self):
        self.check(
            lambda d: d["workload"]["classes"]["pretrain"].update(shape=[4, 4, 4]),
            "shape",
        )

    def test_twisted(self):
        self.check(
            lambda d: d["workload"]["classes"]["pretrain"].update(twisted=True),
            "twisted",
        )

    def test_preferred_relaxable_constraint_is_fine_in_v04(self):
        # Relaxable constraints + the crossing penalty are the v0.4
        # matched pair; `required: false` now validates cleanly.
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["within"] = {
            "level": "pod",
            "required": False,
            "relax_after": "10m",
        }
        assert errors_for(doc) == []
        s = load_scenario(doc, strict=True)
        by_name = {c.name: c for c in s.workload.classes}
        con = by_name["pretrain"].within
        assert con is not None and con.required is False
        assert con.relax_after_s == 600.0

    def test_relaxable_outer_constraint_on_segmented_gang_rejected(self):
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["within"] = {
            "level": "pod",
            "required": False,
        }
        doc["workload"]["classes"]["pretrain"]["segment_nodes"] = 2
        doc["workload"]["classes"]["pretrain"]["segment_level"] = "rack"
        errs = errors_for(doc)
        assert any(
            "relaxable" in e and "not implemented in v0.4" in e for e in errs
        )

    def test_hard_within_dict_form_is_fine(self):
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["within"] = {
            "level": "pod",
            "required": True,
        }
        assert errors_for(doc) == []

    def test_reservations_block_is_fine_in_v04(self):
        # Calendar reservations landed in v0.4: a well-formed block
        # validates cleanly...
        doc = compact_doc()
        doc["reservations"] = [
            {
                "id": "block-1",
                "tenant": "t0",
                "chips": 512,
                "level": "pod",
                "start": "2d",
                "end": "4d",
            }
        ]
        assert errors_for(doc) == []
        s = load_scenario(doc, strict=True)
        (res,) = s.reservations
        assert res.hard_end is True  # calendar blocks end hard by default
        assert res.start_us == 2 * 24 * 3600 * 1_000_000

    def test_reservations_block_rejects_bad_fields(self):
        doc = compact_doc()
        doc["reservations"] = [
            {
                "id": "block-1",
                "tenant": "t0",
                "chips": 4096,  # > one pod (1,024) — can never fit
                "level": "pod",
                "start": "4d",
                "end": "2d",  # end before start
            }
        ]
        errs = errors_for(doc)
        assert any("start must be strictly before end" in e for e in errs)
        assert any("can never fit" in e for e in errs)

    def test_ocs_pool_attr(self):
        doc = yaml.safe_load(TEMPLATE_YAML)
        doc["templates"]["v5p_pod"]["attrs"]["ocs_pool"] = True
        errs = errors_for(doc)
        assert any("ocs_pool" in e and self.MARKER in e for e in errs)


# ---------------------------------------------------------------------------
# Review-fix coverage: per-class defaults, chip_type pinning, lemons,
# outputs.events gating, services section
# ---------------------------------------------------------------------------


class TestPerClassDefaults:
    def test_pretrain_min_runtime_defaults_to_2h(self):
        doc = compact_doc()
        del doc["workload"]["classes"]["pretrain"]["min_runtime"]
        s = load_scenario(doc, strict=True)
        by_name = {c.name: c for c in s.workload.classes}
        assert by_name["pretrain"].min_runtime_s == 7200.0  # DESIGN 14
        assert by_name["eval"].min_runtime_s == 0.0

    def test_abort_prob_defaults_and_opt_out(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["abort_prob"] = 0
        s = load_scenario(doc, strict=True)
        by_name = {c.name: c for c in s.workload.classes}
        assert by_name["finetune"].abort_prob == 0.3  # DESIGN 5.1 default
        assert by_name["eval"].abort_prob == 0.0  # explicit opt-out

    def test_max_lifetime_key_parses(self):
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["max_lifetime"] = "7d"
        s = load_scenario(doc, strict=True)
        by_name = {c.name: c for c in s.workload.classes}
        assert by_name["pretrain"].max_lifetime_s == 7 * 86400.0
        assert by_name["eval"].max_lifetime_s is None

    def test_max_lifetime_must_be_positive(self):
        doc = compact_doc()
        doc["workload"]["classes"]["pretrain"]["max_lifetime"] = "0s"
        errs = errors_for(doc)
        assert any("max_lifetime" in e for e in errs)


class TestChipTypePinning:
    def test_heterogeneous_fleet_requires_class_chip_type(self):
        doc = yaml.safe_load(TEMPLATE_YAML)
        del doc["workload"]["classes"]["eval"]["chip_type"]
        errs = errors_for(doc)
        assert any("chip_type is required" in e for e in errs)

    def test_pinned_class_on_heterogeneous_fleet_is_valid(self):
        assert errors_for(yaml.safe_load(TEMPLATE_YAML)) == []

    def test_unknown_class_chip_type_flagged(self):
        doc = compact_doc()
        doc["workload"]["classes"]["eval"]["chip_type"] = "b200"
        errs = errors_for(doc)
        assert any("chip_type" in e and "b200" in e for e in errs)


class TestFailureModelKeys:
    def test_unknown_failure_model_key_rejected(self):
        doc = compact_doc()
        doc["failure_model"] = {"node_mtbf_days": 42, "lemon_fraction": 0.05}
        errs = errors_for(doc)
        assert any("unknown key" in e and "lemon_fraction" in e for e in errs)

    def test_lemon_keys_parse_and_validate(self):
        doc = compact_doc()
        doc["failure_model"] = {
            "node_mtbf_days": 42,
            "lemon_frac": 0.05,
            "lemon_multiplier": 10,
        }
        s = load_scenario(doc, strict=True)
        assert s.failure_model.lemon_frac == 0.05
        assert s.failure_model.lemon_multiplier == 10.0
        # clusters inherit
        assert s.fleet.clusters()[0].failure_model.lemon_multiplier == 10.0

    def test_lemon_range_validated(self):
        doc = compact_doc()
        doc["failure_model"] = {"lemon_frac": 1.5}
        errs = errors_for(doc)
        assert any("lemon_frac" in e for e in errs)
        doc["failure_model"] = {"lemon_frac": 0.1, "lemon_multiplier": 0}
        errs = errors_for(doc)
        assert any("lemon_multiplier" in e for e in errs)


class TestOutputsEventsGating:
    def test_non_parquet_events_rejected(self):
        doc = compact_doc()
        doc["outputs"]["events"] = "chrome_trace"
        errs = errors_for(doc)
        assert any(
            "outputs.events" in e and "not implemented in v0.1" in e for e in errs
        )

    def test_parquet_events_ok(self):
        assert errors_for(compact_doc()) == []


class TestServicesSection:
    def svc_doc(self, **over):
        doc = compact_doc()
        svc = {"id": "chat", "tenant": "t9", "replicas": 3}
        svc.update(over)
        doc["services"] = [svc]
        return doc

    def test_parses(self):
        s = load_scenario(self.svc_doc(within="pod"), strict=True)
        (svc,) = s.services
        assert (svc.id, svc.tenant, svc.replicas) == ("chat", "t9", 3)
        assert svc.within is not None and svc.within.level == "pod"
        assert svc.tier.name == "PROD"

    def test_tenant_defaults_to_id(self):
        doc = self.svc_doc()
        del doc["services"][0]["tenant"]
        s = load_scenario(doc, strict=True)
        assert s.services[0].tenant == "chat"

    def test_unknown_key_and_bad_level_flagged(self):
        errs = errors_for(self.svc_doc(qps=100))
        assert any("services" in e and "qps" in e for e in errs)
        errs = errors_for(self.svc_doc(within="superpod"))
        assert any("superpod" in e for e in errs)

    def test_services_with_trace_workload_rejected(self):
        doc = self.svc_doc()
        doc["workload"] = {"kind": "trace", "source": "t.csv"}
        errs = errors_for(doc)
        assert any("services" in e and "not implemented in v0.1" in e for e in errs)
