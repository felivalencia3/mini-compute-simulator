"""Validation rung (v0.4): reservation exclusivity, via stints
cross-check.

One calendar block (tenant ``acme``, 64 chips = one FULL pod, hard end)
cuts through a run whose fleet is otherwise saturated by a SPOT
best-effort filler.  The claim picks the fewest-evictions domain and
takes free / owner-occupied leaves before foreign ones (DESIGN §17.4),
so a partial-pod hold could land eviction-free on nodes the owner (or
nobody) occupies — reserving the WHOLE pod on a saturated fleet makes
evictions unavoidable, which is what this rung exercises.  The run
writes ``stints.parquet`` at NODE level, so every stint row names the
exact leaf it occupied; the summary's reservation report names the
claimed leaves and the hold window.  Cross-checks:

1. **Exclusivity**: no stint of any NON-owner job STARTS on a held leaf
   inside ``[start, end)`` (stints straddling the start are the claim's
   evictees — they may linger through their preemption grace, which for
   the SPOT filler is zero).
2. **Eviction spike at claim**: the report counts evictions at start,
   the filler's stints on held leaves end in reason ``preempted``
   exactly at the claim time, and the ``reservation`` preemption trigger
   shows in the summary.
3. **Owner really used the hold**: at least one ``acme`` stint runs on
   held leaves inside the window, and the reported utilization is
   consistent with a direct integral over the owner's stint rows
   (exact: both are chip-µs integrals of the same step function).
4. **Hard-end cliff**: owner jobs still on the hold at ``end`` are
   evicted (report count > 0 and matching stint settlements at ``end``).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fleetsim.api import run_scenario

#: The claim lands MID-ROUND (6 h + 90 s), 30 s after the last wake
#: packed the fleet — so the hold must actually evict the filler (a
#: round-boundary claim can luck into same-timestamp-freed nodes).
START_US = (6 * 3600 + 90) * 1_000_000
END_US = 18 * 3600 * 1_000_000
#: acme_train keeps the default 60 s checkpoint-save, so its REQUEUE
#: settlements (stint t1) land one grace window after the eviction.
OWNER_GRACE_US = 60 * 1_000_000

SCENARIO = {
    "sim": {"horizon": "1d", "round": "60s", "seed": 3},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "h100", "per_node": 8},
                # 2 pods x 8 nodes x 8 chips = 128 chips
                "topology": {"levels": ["pod", "node"], "counts": [2, 8]},
            }
        ],
    },
    "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
    "workload": {
        "kind": "synthetic",
        "classes": {
            # SPOT filler: a standing backlog of zero-notice-killable
            # 8-16 chip jobs saturating the fleet.
            "spot_filler": {
                "class": "finetune",
                "arrival": {
                    "process": "closed_loop",
                    "closed_loop": {"target_pending": 16},
                },
                "chips": {"pmf": {8: 0.5, 16: 0.5}},
                "duration": "lognormal[median=2h, p90=8h]",
                "checkpoint_interval": "0s",
                "capacity": "spot",
                "tenant": "spot",
                "abort_prob": 0,
            },
            # The reservation owner's work: arrives all run long; inside
            # the window it may (only it) use the held nodes.
            "acme_train": {
                "class": "finetune",
                "rate_per_hour": 1,
                "chips": "pow2[8, 16]",
                "duration": "lognormal[median=4h, p90=12h]",
                "tier": "batch",
                "tenant": "acme",
                "abort_prob": 0,
            },
        },
    },
    "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
    "reservations": [
        {
            "id": "block-1",
            "tenant": "acme",
            "chips": 64,  # one FULL pod: eviction-free claims impossible
            "level": "pod",
            "start": "21690s",  # 6 h + 90 s: mid-round (see START_US)
            "end": "18h",
            "hard_end": True,
        }
    ],
    "outputs": {"events": "parquet", "stints": "node"},
}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("res_excl")
    summary = run_scenario(SCENARIO, out_dir=out)
    stints = pd.read_parquet(Path(out) / "stints.parquet")
    on_disk = json.loads((Path(out) / "summary.json").read_text())
    assert on_disk == summary
    return summary, stints


def test_reservation_report_shape(outputs):
    summary, _ = outputs
    (report,) = summary["reservations"]
    assert report["id"] == "block-1"
    assert report["tenant"] == "acme"
    assert report["status"] == "completed"
    assert report["chips_reserved"] >= report["chips_requested"] == 64
    assert report["n_nodes"] == len(report["nodes"]) == 8  # 64 chips / 8


def test_exclusivity_no_foreign_stint_starts_inside_the_hold(outputs):
    summary, stints = outputs
    (report,) = summary["reservations"]
    held = set(report["nodes"])
    inside = stints[
        stints["domain"].isin(held)
        & (stints["t0_us"] >= START_US)
        & (stints["t0_us"] < END_US)
    ]
    assert not inside.empty  # the hold was used at all
    foreign = inside[inside["class_name"] != "acme_train"]
    assert foreign.empty, f"foreign stints inside the hold:\n{foreign}"
    # ... and the owner really ran there (non-vacuous).
    assert (inside["class_name"] == "acme_train").any()


def test_eviction_spike_at_claim(outputs):
    summary, stints = outputs
    (report,) = summary["reservations"]
    held = set(report["nodes"])
    assert report["n_evicted_at_start"] > 0
    # The filler's stints on held leaves settle exactly at the claim
    # (SPOT: zero-notice, so the grace is zero and t1 == start).
    evicted = stints[
        stints["domain"].isin(held)
        & (stints["t1_us"] == START_US)
        & (stints["end_reason"] == "preempted")
        & (stints["class_name"] == "spot_filler")
    ]
    assert len(evicted["job_id"].unique()) == report["n_evicted_at_start"]
    # The dedicated preemption trigger is visible in the summary.
    assert summary["full"]["preemptions_per_min"].get("reservation", 0) > 0


def test_hard_end_cliff_evicts_owner_residents(outputs):
    summary, stints = outputs
    (report,) = summary["reservations"]
    held = set(report["nodes"])
    assert report["n_evicted_at_end"] > 0
    # REQUEUE settlements land one checkpoint-save grace after the cliff.
    cut = stints[
        stints["domain"].isin(held)
        & (stints["t1_us"] == END_US + OWNER_GRACE_US)
        & (stints["end_reason"] == "preempted")
        & (stints["class_name"] == "acme_train")
    ]
    assert len(cut["job_id"].unique()) == report["n_evicted_at_end"]


def test_reported_utilization_matches_stint_integral(outputs):
    summary, stints = outputs
    (report,) = summary["reservations"]
    held = set(report["nodes"])
    own = stints[
        stints["domain"].isin(held) & (stints["class_name"] == "acme_train")
    ]
    # Exact chip-µs integral of owner stints clipped to the window.
    used = 0
    for _, r in own.iterrows():
        lo = max(int(r["t0_us"]), START_US)
        hi = min(int(r["t1_us"]), END_US)
        if hi > lo:
            used += int(r["chips"]) * (hi - lo)
    expected = used / (report["chips_reserved"] * (END_US - START_US))
    assert report["utilization"] == pytest.approx(expected, abs=1e-12)
    assert 0.0 < report["utilization"] <= 1.0
