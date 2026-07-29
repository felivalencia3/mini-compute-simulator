"""Validation rung (v0.4): quota conservation.

The QuotaAdmission invariant — commitment is taken at ADMISSION over a
tenant's non-terminal in-quota jobs and released only at the terminal
transition — implies the fleet-level conservation law this rung checks
at EVERY metrics flush of a contended run:

    for every quota'd tenant:  Σ chips of its RUNNING in-quota jobs
                               <= Σ chips of its committed jobs
                               <= its configured cap.

The run is deliberately over-subscribed (3 Zipf-marked tenants against a
64-chip pool, the heaviest tenant capped far below its demand), so
demotions actually happen: the rung also asserts a nonzero demotion
count, that every demoted job carries ``tier == BEST_EFFORT`` +
``quota_demoted``, and that in-quota usage actually reached each cap at
some flush (the check is not vacuously below the bound).
"""

from fleetsim.config import load_scenario
from fleetsim.engine.rng import RngStreams
from fleetsim.engine.sim import QuotaAdmission, Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.metrics.collector import MetricsCollector
from fleetsim.workload.synthetic import SyntheticSource
from fleetsim.schedulers.base import get_scheduler

HORIZON_US = 24 * 3600 * 1_000_000

SCENARIO = {
    "sim": {"horizon": "1d", "round": "60s", "seed": 11},
    "fleet": {
        "metro": "m",
        "clusters": [
            {
                "name": "c",
                "chip": {"type": "h100", "per_node": 8},
                # 8 nodes x 8 chips = 64 chips
                "topology": {"levels": ["rack", "node"], "counts": [1, 8]},
            }
        ],
    },
    "failure_model": {"node_mtbf_days": 0, "maintenance_rate_per_node_month": 0},
    "workload": {
        "kind": "synthetic",
        "n_tenants": 3,  # Zipf marking over t0 (heavy), t1, t2
        "classes": {
            "train": {
                "class": "finetune",
                "rate_per_hour": 12,
                "chips": "pow2[8, 16]",
                "duration": "lognormal[median=1h, p90=4h]",
                "tier": "batch",
                "abort_prob": 0,
            },
            "eval": {
                "rate_per_hour": 30,
                "chips": "pow2[1, 4]",
                "duration": "lognormal[median=5m, p90=30m]",
                "tier": "batch",
                "abort_prob": 0,
            },
        },
    },
    "scheduler": {"name": "tiered_priority", "params": {"preempt": "requeue"}},
    "quota": {
        "tenants": {"t0": 24, "t1": 32, "t2": 32},
        "over_quota": "best_effort",
    },
}

CAPS = SCENARIO["quota"]["tenants"]


class ConservationSink:
    """Wraps the collector; at every flush cross-checks the engine's live
    running set against the admission ledger and the configured caps."""

    def __init__(self, inner: MetricsCollector, admission: QuotaAdmission):
        self.inner = inner
        self.admission = admission
        self.sim: Simulator | None = None  # late-bound by the test
        self.n_checks = 0
        self.max_in_quota_running: dict[str, int] = {t: 0 for t in CAPS}

    def flush(self, t, fleet, n_pending, n_running):
        assert self.sim is not None
        in_quota_running: dict[str, int] = {t_: 0 for t_ in CAPS}
        for jid, rt in self.sim._running.items():
            job = rt.job
            if job.tenant not in CAPS:
                continue
            if self.admission.is_in_quota(jid):
                assert not job.quota_demoted
                in_quota_running[job.tenant] += rt.spec.chips
            else:
                # Admitted despite the cap -> must be a marked demotion.
                assert job.quota_demoted
                assert job.tier.name == "BEST_EFFORT"
        for tenant, cap in CAPS.items():
            running = in_quota_running[tenant]
            committed = self.admission.committed(tenant)
            assert running <= committed <= cap, (
                f"t={t}: tenant {tenant} running={running}"
                f" committed={committed} cap={cap}"
            )
            self.max_in_quota_running[tenant] = max(
                self.max_in_quota_running[tenant], running
            )
        self.n_checks += 1
        self.inner.flush(t, fleet, n_pending, n_running)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_quota_conservation_at_every_flush():
    scenario = load_scenario(SCENARIO)
    fleet = build_fleet(scenario)
    rng = RngStreams(scenario.sim.seed)
    source = SyntheticSource(
        scenario.workload, fleet, rng, scenario.sim.horizon_us
    )
    collector = MetricsCollector.from_scenario(scenario, fleet)
    admission = QuotaAdmission(scenario.quota)
    sink = ConservationSink(collector, admission)
    sim = Simulator(
        scenario,
        fleet,
        source,
        get_scheduler("tiered_priority", {"preempt": "requeue"}),
        sink,
        admission,
        rng=rng,
    )
    sink.sim = sim
    sim.run()

    assert sink.n_checks > 1000  # one per round over a day

    # Non-vacuous: the heavy tenant hit its cap, and demotions happened.
    assert sink.max_in_quota_running["t0"] == CAPS["t0"]
    rows = collector.job_rows()
    demoted = [r for r in rows if r["quota_demoted"]]
    assert demoted, "no job was ever demoted — the rung tested nothing"
    assert all(r["tier"] == "BEST_EFFORT" for r in demoted)
    # Demotions concentrate on the over-subscriber under Zipf marking.
    by_tenant: dict[str, int] = {}
    for r in demoted:
        by_tenant[r["tenant"]] = by_tenant.get(r["tenant"], 0) + 1
    assert by_tenant.get("t0", 0) == max(by_tenant.values())
    # The summary carries the demotion counter (feature-keyed schema).
    counts = collector.event_counts()["full"]
    assert counts["quota_demotions"] == len(demoted)
