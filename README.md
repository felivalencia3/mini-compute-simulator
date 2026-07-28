# fleetsim

[![CI](https://github.com/felivalencia3/mini-compute-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/felivalencia3/mini-compute-simulator/actions/workflows/ci.yml)

**FleetSim simulates fleets, not jobs.** Given a fleet of heterogeneous
GPU/TPU clusters described in YAML, a stochastic mix of pretraining gangs,
fine-tunes, evals, and inference — plus node failures — it answers: *what
occupancy, queue wait, preemption rate, and goodput does a given
scheduling policy deliver?* A lab describes its fleet, plugs in a
scheduler as a small Python class (or replays a public trace), and gets
reproducible, seeded results.

No open-source tool combines *hierarchical multi-cluster fleets +
heterogeneous chips + gang scheduling + preemption cost + failures +
mixed training/inference traffic*. Single-cluster simulators (Blox,
Gavel, Pollux) have flat GPU counts and no failures; per-job simulators
(ASTRA-sim, Vidur) don't know what a queue is; HPC batch simulators lack
ML gang/preemption-cost semantics. That combination is the product — see
[DESIGN.md](DESIGN.md) §1 for the full survey.

## What v0.1 does (honest scope)

- **Fleet model**: one metro, N clusters, config-defined level trees
  (`[cluster, pod, rack, node]` — vocabulary from config, not code),
  heterogeneous chip types across clusters, chip type pinned per job.
- **Gang allocation**: atomic and immutable; chip-granular sharing inside
  a node, whole-node multiples above; hard `within: <level>` constraints
  (the load-bearing ingredient of gang-scheduling fragmentation).
- **Workloads**: synthetic per-class generator with trace-derived defaults
  (diurnal Poisson arrivals, pow2 sizes, lognormal durations, tenant
  skew, ~30% aborts by default — `abort_prob: 0` opts out) and
  canonical-CSV trace replay with a Philly converter
  (`fleetsim.workload.philly`); trace chip counts are quantized to the
  fleet's node grammar so replay can never wedge on an unplaceable gang.
- **Schedulers**: `fifo` (strict + best-effort) and `tiered_priority`
  (Borg bands, REQUEUE preemption, no preemption within PROD, per-class
  min-runtime guard) — plus out-of-tree plugins via entry points.
- **Failures**: node MTBF sampling, auto/manual repair, planned
  maintenance drains with grace windows, checkpoint/restart accounting
  (lost work, restart overhead, checkpoint-save overhead).
- **Metrics**: allocation rate vs occupancy vs goodput split, ETTR,
  queue-wait/JCT distributions (job- and chip-hour-weighted),
  fragmentation index, per-tenant shares, steady-state windowing;
  Parquet + JSON + plots.
- **Engine**: int-microsecond event core, coalesced scheduler rounds
  (cadence follows `sim.round`). Measured on the shipped 2,048-chip
  example at ρ≈0.9: 14 simulated days in ~4 s, 56 days in ~45 s, 6
  months in ~9 minutes on a laptop. Cost scales with events × queue
  depth, so hotter or longer runs cost proportionally more; the DESIGN
  §6.3 100K-GPU envelope is a target, not yet a measurement.

**Not in v0.1** (schema present, rejected by `fleetsim validate` with
"not implemented in v0.1"): backfill/`Reserve`, quota, capacity classes
beyond on-demand, TPU OCS predicates, relaxable constraints, multi-gang
jobs, autoscaling inference, throughput matrices. Roadmap: DESIGN.md §11.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'          # numpy, pandas, pyarrow, pyyaml (+ matplotlib, pytest)

fleetsim run examples/01_minimal/scenario.yaml -o out_tiered
fleetsim run examples/01_minimal/scenario.yaml -o out_fifo \
    --override scheduler.name=fifo --override "scheduler.params={}"
fleetsim compare out_fifo out_tiered
fleetsim validate examples/01_minimal/scenario.yaml
fleetsim plot out_tiered
```

`run` prints a summary table and writes to the output directory:

| File | Contents |
|---|---|
| `summary.json` | every metric below, full-run and steady-state window |
| `jobs.parquet` | one row per job: timings, status, preemptions, productive/lost chip-seconds |
| `timeseries.parquet` | per-round samples: allocated/healthy chips, queue depth, fragmentation |
| `plots/*.png` | JCT + queue-wait CDFs, occupancy + goodput timelines (`outputs: {plots: true}`) |

Python one-liner (same pipeline):

```python
import fleetsim
summary = fleetsim.run_scenario("examples/01_minimal/scenario.yaml", "out")
```

## Custom schedulers

```python
from fleetsim import Place, Scheduler, register
from fleetsim.schedulers.placement import FirstFit

@register("smallest_first")
class SmallestFirst(Scheduler):
    def __init__(self):
        self.placement = FirstFit()

    def schedule(self, view):          # called once per 60 s round
        actions = []
        for job in sorted(view.pending(), key=lambda j: (j.chips, j.submit_time, j.id)):
            p = view.find_placement(job, self.placement)   # tentatively reserves
            if p is not None:
                actions.append(Place(job.id, p))
        return actions
```

The engine validates and applies the returned intents (illegal preemptions
raise in strict mode), so every policy inherits correct gang atomicity and
preemption cost. Ship it out-of-tree by declaring an entry point in the
`fleetsim.schedulers` group — `examples/03_custom_scheduler/` is a
complete plugin package to copy.

## Metrics (DESIGN §9)

| Metric | Definition |
|---|---|
| **Allocation rate** | allocated / total chips (incl. failed/drained) |
| **Occupancy** | ∫ allocated dt / ∫ *healthy* dt — the gap vs allocation rate = capacity lost to failures |
| **Goodput** | ∫ productive chip-time / ∫ allocated chip-time; excludes lost-since-checkpoint work, restart latency, checkpoint saves. Still-running jobs' checkpointed progress counts; windowed goodput spreads each stint's surviving work over its interval, so it is ≤ 1 by construction |
| **ETTR** | per-job productive/elapsed (fleet-healthy benchmark > 0.9) |
| Queue wait | first_start − submit; per class **and** bucketed by gang chip count (1–8 / 9–64 / 65–512 / 513+), P50/P90/P99 |
| JCT | end − submit for completed jobs; job- **and** chip-hour-weighted |
| Preemptions/min | split by trigger: scheduler, maintenance, failure_second_order (a failure-requeued job evicting others), node_failure |
| Node failures | counted overall and by sampled cause (gpu_hbm / network / software / other, the Llama-3 mix) |
| **Fragmentation** | largest placeable gang per level; frag index = 1 − largest/free; stranded chips (below the smallest gang quantum seen) |
| Replica availability | inference replica-time running / desired (`services:` section) |
| Per-tenant share | chip-hours, queue waits, submissions by tenant |

All distributional metrics are reported both job-weighted and
chip-hour-weighted (mice vs hogs), over the full run and a configurable
steady-state window.

## Determinism contract

Every run is a pure function of `(fleet config, workload config | trace,
seed)`. Named independent RNG streams (`arrivals/<class>`,
`size/<class>`, `failures`, `repair`, `maintenance`, ...) mean enabling
failure injection never perturbs the arrival sequence — A/B scheduler
comparisons are paired experiments, not noise. Identical `(scenario,
seed)` produce **byte-identical** Parquet and JSON outputs
(`validation/test_determinism.py` enforces this in CI, alongside an
M/M/c Erlang-C check and invariant property tests).

## License

Apache-2.0 — see [LICENSE](LICENSE).
