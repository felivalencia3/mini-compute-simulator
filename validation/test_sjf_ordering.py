"""Validation rung (v0.6): SJF is SPT-optimal on a fungible pool.

The classic single-machine result (Smith 1956): scheduling independent
jobs shortest-processing-time first minimizes mean completion time.  Our
``sjf`` scheduler, keyed on ``walltime_est_s`` with *exact* estimates
(``walltime_est_s == true_duration_s`` -> SJF-oracle), must reproduce it.

Setup: a flat 4-chip pool (4 x 1-chip nodes -> chips fully fungible),
four jobs each requesting the WHOLE pool so they serialize on one
"machine", all released at t=0 with distinct durations that are multiples
of the 60 s round (so completions land on round boundaries and the next
placement happens at that boundary — no quantization slack).  The job ids
are deliberately in longest-first order, so FIFO's arrival order is the
worst case and differs maximally from SJF.

Asserted:

1. **Shortest-first start order** — SJF starts the jobs in ascending
   duration order (the SJF ordering property, end to end through the real
   engine, not just the scheduler's sort key).
2. **SPT optimality** — SJF's mean JCT <= FIFO's mean JCT (strictly less
   here: the rung is not vacuous).  Both conserve total work (identical
   makespan), so the win is pure reordering.
"""

from fleetsim.config import load_scenario
from fleetsim.engine.sim import Simulator
from fleetsim.fleet.build import build_fleet
from fleetsim.model import GangSpec, Job, JobClass, Tier
from fleetsim.schedulers.base import get_scheduler
from fleetsim.workload.base import ListSource

S = 1_000_000  # one second in microseconds
POOL_CHIPS = 4


def make_scenario(horizon="1h"):
    return load_scenario(
        {
            "sim": {"horizon": horizon, "round": "60s", "seed": 0},
            "fleet": {
                "metro": "m",
                "clusters": [
                    {
                        "name": "c",
                        "chip": {"type": "h100", "per_node": 1},
                        "topology": {"levels": ["node"], "counts": [POOL_CHIPS]},
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
                        "chips": "pow2[1, 4]",
                        "duration": "lognormal[median=2m, p90=30m]",
                    }
                },
            },
        }
    )


def mk_job(jid, dur_s):
    """A whole-pool job released at t=0 with an EXACT walltime estimate
    (SJF-oracle: est == true duration), checkpointing/overheads off so
    service time is exactly ``dur_s``."""
    return Job(
        id=jid,
        tenant="t0",
        job_class=JobClass.FINETUNE,
        submit_t=0,
        gangs=[GangSpec(chips=POOL_CHIPS, chip_type="h100")],
        tier=Tier.BATCH,
        walltime_est_s=dur_s,
        true_duration_s=dur_s,
        checkpoint_interval_s=0.0,
        checkpoint_save_s=0.0,
        restart_overhead_s=0.0,
    )


class StartSink:
    """Records first starts and finishes (like the backfill rung)."""

    def __init__(self):
        self.first_start = {}
        self.finish = {}

    def job_started(self, job, alloc, t):
        self.first_start.setdefault(job.id, t)

    def job_finished(self, job, t, status, productive_chip_s, lost_chip_s):
        self.finish[job.id] = (t, status.name)

    def __getattr__(self, name):
        return lambda *a, **k: None


def run(jobs, scheduler_name, horizon="1h"):
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


#: id -> duration (s); ids intentionally in LONGEST-first order so FIFO's
#: arrival order is the worst case for mean JCT.
DURATIONS = {"j0": 480, "j1": 120, "j2": 360, "j3": 240}


def job_list():
    return [mk_job(jid, dur) for jid, dur in DURATIONS.items()]


def mean_jct(sink):
    # All jobs submit at t=0, so JCT == finish time.
    return sum(t for (t, _status) in sink.finish.values()) / len(sink.finish)


def test_sjf_starts_shortest_first_and_beats_fifo_mean_jct():
    sjf = run(job_list(), "sjf")
    fifo = run(job_list(), "fifo")

    ids = set(DURATIONS)
    # Both runs complete every job successfully (work conservation).
    assert set(sjf.finish) == set(fifo.finish) == ids
    assert all(sjf.finish[j][1] == "COMPLETED" for j in ids)
    assert all(fifo.finish[j][1] == "COMPLETED" for j in ids)

    # (1) SJF start order is shortest-first: sorting jobs by their actual
    # start time yields ascending duration order.
    by_start = sorted(ids, key=lambda j: sjf.first_start[j])
    assert by_start == sorted(ids, key=lambda j: DURATIONS[j])
    assert by_start == ["j1", "j3", "j2", "j0"]

    # (2) SPT optimality: SJF mean JCT <= FIFO mean JCT (strictly less
    # here — the rung is not vacuous).
    sjf_mean, fifo_mean = mean_jct(sjf), mean_jct(fifo)
    assert sjf_mean <= fifo_mean
    assert sjf_mean < fifo_mean

    # Total work is conserved: identical makespan (last completion), so
    # the advantage is pure reordering, not lost/gained work.
    assert max(t for t, _ in sjf.finish.values()) == max(
        t for t, _ in fifo.finish.values()
    )
