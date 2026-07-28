"""Unit tests for the integration layer: fleetsim.api, fleetsim.cli, and
the package's public exports.

Scenarios here are tiny (2 nodes, minutes-long horizons) — the heavier
end-to-end behavior lives in validation/.
"""

import json
from pathlib import Path

import pytest
import yaml

import fleetsim
from fleetsim import api
from fleetsim.cli import main
from fleetsim.config import ScenarioError

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def tiny_doc(**sim_over):
    sim = {"horizon": "20m", "round": "60s", "seed": 5}
    sim.update(sim_over)
    return {
        "sim": sim,
        "fleet": {
            "metro": "m",
            "clusters": [
                {
                    "name": "c",
                    "chip": {"type": "h100", "per_node": 8},
                    "topology": {"levels": ["node"], "counts": [2]},
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
                    "rate_per_hour": 60,
                    "chips": "pow2[1, 8]",
                    "duration": "lognormal[median=1m, p90=5m]",
                }
            },
        },
        "scheduler": {"name": "fifo"},
    }


# ---------------------------------------------------------------------------
# api.apply_overrides / api.load_document
# ---------------------------------------------------------------------------


def test_apply_overrides_types_and_nesting():
    doc = {"sim": {"seed": 1}}
    api.apply_overrides(
        doc,
        {
            "sim.seed": "7",
            "scheduler.name": "fifo",
            "scheduler.params": "{}",
            "outputs.plots": "false",
        },
    )
    assert doc["sim"]["seed"] == 7  # YAML-typed, not the string "7"
    assert doc["scheduler"] == {"name": "fifo", "params": {}}
    assert doc["outputs"] == {"plots": False}


def test_apply_overrides_rejects_non_mapping_traversal_and_empty_segments():
    with pytest.raises(ScenarioError):
        api.apply_overrides({"sim": 3}, {"sim.seed": "7"})
    with pytest.raises(ScenarioError):
        api.apply_overrides({}, {"sim..seed": "7"})


def test_load_document_deep_copies_mapping_input():
    doc = tiny_doc()
    loaded = api.load_document(doc)
    loaded["sim"]["seed"] = 999
    assert doc["sim"]["seed"] == 5  # caller's object untouched


def test_load_document_errors():
    with pytest.raises(ScenarioError):
        api.load_document("/nonexistent/scenario.yaml")


# ---------------------------------------------------------------------------
# api.run_scenario
# ---------------------------------------------------------------------------


def test_run_scenario_returns_summary_without_writing(tmp_path):
    summary = api.run_scenario(tiny_doc())  # no out_dir, no outputs.dir
    assert summary["full"]["counts"]["jobs_finished"] > 0
    assert list(tmp_path.iterdir()) == []


def test_run_scenario_writes_outputs(tmp_path):
    out = tmp_path / "out"
    summary = api.run_scenario(tiny_doc(), out_dir=out)
    for name in ("summary.json", "jobs.parquet", "timeseries.parquet"):
        assert (out / name).is_file()
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk == summary


def test_run_scenario_seed_override_wins_over_overrides():
    a = api.run_scenario(tiny_doc(), seed_override=11, overrides={"sim.seed": "99"})
    b = api.run_scenario(tiny_doc(seed=11))
    assert a == b


def test_run_scenario_invalid_scenario_lists_all_errors():
    doc = tiny_doc()
    doc["workload"]["classes"]["eval"]["chips"] = "pow2[3, 8]"
    doc["reservations"] = []
    with pytest.raises(ScenarioError) as exc:
        api.run_scenario(doc)
    joined = "; ".join(exc.value.errors)
    assert "pow2" in joined and "not implemented in v0.1" in joined


def test_run_scenario_resolves_trace_source_relative_to_scenario_file():
    # Uses the bundled example from an unrelated cwd (pytest's).
    summary = api.run_scenario(EXAMPLES / "02_trace_replay" / "scenario.yaml")
    assert summary["full"]["counts"]["jobs_finished"] == 30


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_tiny(tmp_path: Path) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(tiny_doc()), encoding="utf-8")
    return p


def test_cli_run_prints_table_and_writes(tmp_path, capsys):
    scn = write_tiny(tmp_path)
    out = tmp_path / "o"
    rc = main(["run", str(scn), "-o", str(out), "--override", "sim.seed=8"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "FleetSim summary" in printed
    assert str(out) in printed
    assert (out / "summary.json").is_file()


def test_cli_validate_ok_and_broken(tmp_path, capsys):
    scn = write_tiny(tmp_path)
    assert main(["validate", str(scn)]) == 0
    assert "OK" in capsys.readouterr().out

    broken = tmp_path / "broken.yaml"
    doc = tiny_doc()
    doc["scheduler"] = {"name": "no_such_scheduler"}
    doc["workload"]["classes"]["eval"]["gangs"] = 2
    broken.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert main(["validate", str(broken)]) == 1
    printed = capsys.readouterr().out
    assert "not implemented in v0.1" in printed
    assert "INVALID" in printed


def test_cli_validate_flags_unknown_scheduler(tmp_path, capsys):
    scn = tmp_path / "s.yaml"
    doc = tiny_doc()
    doc["scheduler"] = {"name": "definitely_not_registered_xyz"}
    scn.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert main(["validate", str(scn)]) == 1
    assert "unknown scheduler" in capsys.readouterr().out


def test_cli_run_invalid_scenario_exits_2(tmp_path, capsys):
    scn = tmp_path / "s.yaml"
    doc = tiny_doc()
    doc["sim"]["horizon"] = "-1d"
    scn.write_text(yaml.safe_dump(doc), encoding="utf-8")
    rc = main(["run", str(scn), "-o", str(tmp_path / "o")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_cli_plot_and_compare(tmp_path, capsys):
    scn = write_tiny(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(scn), "-o", str(out_a)]) == 0
    assert main(["run", str(scn), "-o", str(out_b), "--seed", "9"]) == 0
    capsys.readouterr()

    assert main(["plot", str(out_a)]) == 0
    printed = capsys.readouterr().out
    assert (out_a / "plots" / "jct_cdf.png").is_file()
    assert "jct_cdf.png" in printed

    assert main(["compare", str(out_a), str(out_b)]) == 0
    printed = capsys.readouterr().out
    assert "occupancy (window)" in printed
    assert "a" in printed and "b" in printed


def test_cli_compare_error_paths(tmp_path, capsys):
    assert main(["compare", str(tmp_path / "solo")]) == 2
    capsys.readouterr()
    assert main(["compare", str(tmp_path / "x"), str(tmp_path / "y")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_cli_plot_missing_dir_exits_2(tmp_path, capsys):
    assert main(["plot", str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_public_api_exports():
    assert fleetsim.__version__ == "0.1.0"
    for name in fleetsim.__all__:
        assert getattr(fleetsim, name, None) is not None, name
    # The documented plugin surface is importable from the package root.
    assert fleetsim.run_scenario is api.run_scenario
    assert issubclass(fleetsim.Place, object)
    assert callable(fleetsim.register)
