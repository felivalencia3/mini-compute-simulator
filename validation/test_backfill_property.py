"""Validation rung (v0.4): EASY backfill head-job non-delay — a
paired-run property.

Setup: a flat 16-chip pool (16 x 1-chip nodes — chips are fully
fungible, so chip-count reasoning is exact), hand-built jobs with EXACT
walltime estimates (``walltime_est_s == true_duration_s``), no failures,
no checkpointing, no aborts.  The same job list runs under strict FIFO
and under ``easy_backfill`` (fresh Simulator + scheduler per run —
paired experiment, byte-identical inputs).

Properties asserted (the EASY contract under honest estimates):

1. **Head-job non-delay**: whenever a job is the FIFO head, backfill
   never starts it later than strict FIFO does.  With exact estimates
   and fungible chips this extends pointwise: EVERY job starts no later
   under EASY than under strict FIFO (induction over release times —
   each backfilled job returns its chips by the head's shadow time).
2. **Backfill actually happens**: at least one job starts strictly
   earlier (the rung is not vacuously true).
3. **Work conservation**: both runs complete the same job set with the
   same per-job durations (backfill reorders starts, never work).

The dual honesty property — that a LYING estimate (true > est) CAN
delay the head, because the engine never kills at the estimate — is
asserted in the second test.

The third test is the documented NEGATIVE property (DESIGN §17.2): with
merely HONEST over-estimates (est >= true, not exact) the pointwise
non-delay property does NOT hold — an inflated estimate widens the
shadow window and a backfilled job can outlive the head's true FIFO
start.  What survives is the canonical EASY guarantee (Mu'alem &
Feitelson, IEEE TPDS 2001): the head never starts later than the
shadow computed when backfill was granted.
"""

from fleetsim.config import load_scenario
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.model import GangSpec, Job, JobClass, Tier
from fleetsim.schedulers.base import get_scheduler
from fleetsim.workload.base import ListSource

S = 1_000_000  # one second in microseconds


def make_scenario(horizon="6h"):
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": "60s", "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": 1},
                        "topology": {"levels": ["node"], "counts": [16]},
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
    )


def mk_job(jid, submit_s, chips, dur_s, est_s=None):
    return Job(
        id=jid,
        tenant="t0",
        job_class=JobClass.FINETUNE,
        submit_t=int(round(submit_s * S)),
        gangs=[GangSpec(chips=chips, chip_type="h100")],
        tier=Tier.BATCH,
        walltime_est_s=est_s if est_s is not None else dur_s,
        true_duration_s=dur_s,
        checkpoint_interval_s=0.0,
        checkpoint_save_s=0.0,
        restart_overhead_s=0.0,
    )


class StartSink:
    """Records first starts and finishes."""

    def __init__(self):
        self.first_start = {}
        self.finish = {}

    def job_started(self, job, alloc, t):
        self.first_start.setdefault(job.id, t)

    def job_finished(self, job, t, status, productive_chip_s, lost_chip_s):
        self.finish[job.id] = (t, status.name)

    def __getattr__(self, name):
        return lambda *a, **k: None


def run(jobs, scheduler_name, horizon="6h"):
    scenario = make_scenario(horizon)
    fleet = build_fleet(scenario)
    sink = StartSink()
    sim = Simulator(
        scenario,
        fleet,
        ListSource(jobs),
        get_scheduler(scheduler_name, {}),
        sink,
    )
    sim.run()
    return sink


def job_list():
    """A contended, hand-shaped mix: wide jobs that block the strict-FIFO
    head, plus mice with short exact estimates that fit the shadow."""
    jobs = [
        mk_job("a00", 0, 12, 600.0),     # takes 12/16 chips
        mk_job("a01", 0, 8, 900.0),      # head: blocks (12+8 > 16)
        mk_job("a02", 0, 4, 120.0),      # backfills (ends well before shadow)
        mk_job("a03", 0, 4, 3000.0),     # too long to backfill
        mk_job("a04", 60, 2, 60.0),      # backfills
        mk_job("a05", 120, 16, 1200.0),  # full-pool job
        mk_job("a06", 120, 1, 300.0),
        mk_job("a07", 180, 6, 240.0),
        mk_job("a08", 240, 3, 60.0),
        mk_job("a09", 300, 10, 600.0),
        mk_job("a10", 300, 2, 120.0),
        mk_job("a11", 360, 5, 180.0),
        mk_job("a12", 420, 8, 2400.0),
        mk_job("a13", 480, 1, 60.0),
        mk_job("a14", 540, 4, 300.0),
        mk_job("a15", 600, 12, 900.0),
        mk_job("a16", 660, 2, 60.0),
        mk_job("a17", 720, 7, 420.0),
        mk_job("a18", 780, 3, 120.0),
        mk_job("a19", 840, 16, 600.0),
    ]
    return jobs


def test_backfill_never_delays_any_job_under_exact_estimates():
    jobs_a = job_list()
    jobs_b = job_list()
    fifo = run(jobs_a, "fifo")
    easy = run(jobs_b, "easy_backfill")

    ids = {j.id for j in job_list()}
    # Work conservation: same completion set, same statuses.
    assert set(fifo.finish) == set(easy.finish) == ids
    assert all(fifo.finish[j][1] == "COMPLETED" for j in ids)
    assert all(easy.finish[j][1] == "COMPLETED" for j in ids)

    # Pointwise start dominance (includes head-job non-delay).
    assert set(fifo.first_start) == set(easy.first_start) == ids
    for jid in sorted(ids):
        assert easy.first_start[jid] <= fifo.first_start[jid], (
            f"{jid}: EASY start {easy.first_start[jid]} > FIFO start"
            f" {fifo.first_start[jid]}"
        )

    # Not vacuous: backfill moved at least one mouse ahead.
    strictly_earlier = [
        jid
        for jid in ids
        if easy.first_start[jid] < fifo.first_start[jid]
    ]
    assert strictly_earlier, "no job was backfilled — the rung tested nothing"
    # And the very first designed backfill actually happened at the head's
    # blocking wake.
    assert easy.first_start["a02"] < fifo.first_start["a02"]


def test_honest_overestimates_delay_the_head_but_never_past_the_shadow():
    """The over-estimate counterexample (DESIGN §17.2): every estimate
    is an honest UPPER bound (a0 estimates 1000 s for 100 s of true
    work), yet the head is delayed 780 s past its strict-FIFO start —
    a0's inflated estimate widens the shadow to 1000 s, z2 (est = true
    900 s) backfills at t=0 and holds 4 chips until 900 s, while FIFO
    would have started the 14-chip head at the 120 s round boundary.
    The canonical guarantee survives: the head starts at 900 s, before
    the 1000 s shadow that granted the backfill."""
    jobs = lambda: [  # noqa: E731
        mk_job("a0", 0, 12, 100.0, est_s=1000.0),  # honest 10x over-estimate
        mk_job("a1", 0, 14, 900.0),                # head (exact estimate)
        mk_job("z2", 0, 4, 900.0),                 # backfills on a0's promise
    ]
    fifo = run(jobs(), "fifo")
    easy = run(jobs(), "easy_backfill")
    assert all(v[1] == "COMPLETED" for v in fifo.finish.values())
    assert all(v[1] == "COMPLETED" for v in easy.finish.values())
    # Strict FIFO: a0 done at 100 s, head placed at the next round.
    assert fifo.first_start["a1"] == 120 * S
    # EASY: z2 was granted backfill against the 1000 s shadow...
    assert easy.first_start["z2"] == 0
    # ...and the head IS delayed past its FIFO start (the pointwise
    # property fails under honest-but-inexact estimates)...
    assert easy.first_start["a1"] == 900 * S > fifo.first_start["a1"]
    # ...but never past the shadow computed when backfill was granted.
    assert easy.first_start["a1"] <= 1000 * S


def test_underestimates_can_delay_the_head_estimate_error_honesty():
    """A job that LIES about its walltime (true 3000 s, est 120 s) is
    backfilled and then keeps running past its promise — the engine
    refuses to kill at the estimate, so the head IS delayed.  This is
    the documented estimate-error honesty of the EASY implementation
    (real clusters without walltime enforcement behave this way)."""
    truthful = [
        mk_job("big", 0, 12, 600.0),
        mk_job("head", 0, 14, 900.0),  # needs nearly the whole pool
        mk_job("liar", 0, 4, 120.0, est_s=120.0),  # honest here
    ]
    lying = [
        mk_job("big", 0, 12, 600.0),
        mk_job("head", 0, 14, 900.0),
        mk_job("liar", 0, 4, 3000.0, est_s=120.0),  # runs 25x its promise
    ]
    honest = run(truthful, "easy_backfill")
    dishonest = run(lying, "easy_backfill")
    # Both runs backfill the liar at t=0 (est fits the 600 s shadow)...
    assert honest.first_start["liar"] == dishonest.first_start["liar"] == 0
    # ...but only the dishonest run delays the head past its shadow.
    assert honest.first_start["head"] == 600 * S
    assert dishonest.first_start["head"] > 600 * S
