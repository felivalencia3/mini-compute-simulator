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

## What it does (honest scope, v0.7)

- **Fleet model**: one metro, N clusters, config-defined level trees
  (`[cluster, pod, rack, node]` — vocabulary from config, not code),
  heterogeneous chip types across clusters, chip type pinned per job.
- **Gang allocation**: atomic and immutable; chip-granular sharing inside
  a node, whole-node multiples above; hard `within: <level>` constraints
  (the load-bearing ingredient of gang-scheduling fragmentation).
- **Workloads**: synthetic per-class generator with trace-derived defaults
  (diurnal Poisson arrivals, pow2 sizes, lognormal durations, tenant
  skew, ~30% aborts by default — `abort_prob: 0` opts out) and
  canonical-CSV trace replay with Philly (`fleetsim.workload.philly`) and
  Helios (`fleetsim.validation.helios`, v0.6) converters; trace chip
  counts are quantized to the fleet's node grammar so replay can never
  wedge on an unplaceable gang.
  v0.2 adds CLOSED-LOOP standing-backlog classes
  (`arrival: backlog[target_pending=N]` instead of a rate: the engine
  tops the class back up to N pending jobs at every scheduler wake —
  default `tier: best_effort`, freely reclaimable) and SEGMENTED gangs
  (`segment_nodes` + `segment_level` with `within` as the outer
  constraint: whole-node blocks bin-packed across e.g. pods, placed
  atomically; `jobs.parquet` reports `n_domains_spanned`).
  v0.2's generative traffic model
  ([docs/traffic-math.md](docs/traffic-math.md)) adds per-class arrival
  processes (log-linear harmonic NHPP diurnality, MMPP-2 crunch/normal
  regimes, Hawkes self-excited eval bursts), weighted pow2 size pmfs
  with trace-derived presets, lognormal-body + truncated-Pareto-tail
  duration splices, finite-Zipf tenant skew (`tenant_zipf_s`, default
  1.2), and a `google_fleet` workload preset that scales a full
  four-class mix to any fleet, with per-class overrides.
- **Schedulers**: `fifo` (strict + best-effort), `tiered_priority`
  (Borg bands `best_effort < batch < prod < monitoring` — band 0's
  canonical name is `best_effort` as of v0.2, with `free` accepted as a
  legacy spelling; REQUEUE preemption, no preemption within PROD,
  per-class min-runtime guard, and v0.2 segmented reclaim that can empty
  multiple pods for one large pending gang), and v0.4's
  `easy_backfill` (FIFO + head-of-line shadow reservation +
  conservative walltime-estimate backfill; the engine never kills at
  the estimate, so lying estimates delay the head exactly like real
  clusters), and v0.6's `sjf` (shortest-job-first; keyed on the walltime
  estimate, so an exact estimate is the SJF-oracle the Helios validation
  replays) — plus out-of-tree plugins via entry points. Every one of them
  takes a **placement policy** (v0.7, below) on the orthogonal *where*
  axis.
- **Placement physics (v0.4)**: relaxable `within` constraints
  (`within: {level: pod, required: false, relax_after: 10m}`) shipped
  as a matched pair with the crossing penalty (`penalties: {xover:
  {pod: 0.7}}` — cross-domain placements, relaxed or segmented, run at
  the configured speed multiplier), so relaxing is a measurable
  tradeoff, not a free lunch (`examples/05_topology_tradeoff/`).
- **Capacity economics (v0.4)**: per-tenant chip quota with
  admission-time over-quota demotion to the best_effort band (or
  rejection) — a deliberately simplified, MAST-inspired model (DESIGN
  §17.3 states the divergence exactly); calendar reservations that
  claim whole nodes inside the fewest-evictions domain, evict
  residents, exclude other tenants for the window, and cut through
  their own tenant at a hard end (per-block `utilization` reported in
  the summary); `capacity: spot` = zero-notice kill + checkpoint
  restart, the worst-case spot model (`examples/06_economics/`).
- **Failures**: node MTBF sampling, auto/manual repair, planned
  maintenance drains with grace windows, checkpoint/restart accounting
  (lost work, restart overhead, checkpoint-save overhead).
- **Metrics**: allocation rate vs occupancy vs goodput split, ETTR,
  queue-wait/JCT distributions (job- and chip-hour-weighted),
  fragmentation index, per-tenant shares, steady-state windowing;
  Parquet + JSON + plots.
- **Visualizer** (v0.3): `fleetsim viz out/` renders any run into one
  self-contained interactive HTML replay — scrub the fleet map through
  time, watch preemption waves and the reclaim of 32 pods for a
  131K-chip gang — with zero external requests, every pixel traceable
  to the output files ([docs/visualizer.md](docs/visualizer.md)).
- **Web app** (v0.5): `fleetsim serve --open` runs a local,
  pure-stdlib web app — browse workspace runs, open any run as the
  interactive 2D report, submit scenario YAML from an editor with
  validation and live progress, and replay the fleet in a **three.js
  3D view** (vendored, zero external requests). Loopback-only by
  default, strict CSP, path-contained workspace, `yaml.safe_load`
  only, in-process runs ([docs/webapp.md](docs/webapp.md)).
- **Validation suite** (v0.6, extended in v0.7): reproduces PUBLISHED
  cluster-trace results, so the occupancy / queue-wait / JCT numbers are
  checked against reality, not just internally consistent. fleetsim
  reproduces the **Helios (SC '21) FIFO-vs-SJF policy effect** across all
  four clusters (under September-max per-VC sizing, a strict scan, and
  `consolidate` placement) — the SJF advantage, **all four** JCT ratios
  inside fleetsim's [1.3–8]× tolerance band, all four queuing ratios inside
  [3–25]×, the Saturn-strongest → Uranus-weakest JCT-ratio rank and the
  queuing-share ordering — with **no `xfail`** (see the Validation section
  below and [docs/validation.md](docs/validation.md)). Downloads are
  stdlib-only and integrity-gated by exact byte size + Git-LFS-pointer
  detection; CI runs vendored slices only, full replays are opt-in.
- **Pluggable placement** (v0.7): ordering and placement are separate axes,
  and the *where* axis is now selectable per scenario —
  `scheduler: {name: fifo, params: {placement: best_fit}}` on any scheduler.
  `first_fit` stays the default (existing scenarios are byte-identical —
  examples 01 and 04 were verified byte-for-byte against v0.6, and the
  validation harness's own `placement` default is `first_fit` too, so no
  pre-v0.7 caller's numbers move);
  `best_fit` packs sub-node gangs tightest-first so whole nodes stay whole
  for gangs, `consolidate` minimizes the number of leaf-*parent* domains a
  spanning gang touches (not crossings at coarser levels — see
  [docs/placement.md](docs/placement.md)), and `spread` is the deliberate
  anti-policy for control arms. Naming a policy also switches on
  a `stranded_whole_nodes` fleet metric, so the mechanism is observable in
  `timeseries.parquet` instead of inferred from JCT. This is what closed
  the last Helios deviation; semantics, selection and measured effects in
  [docs/placement.md](docs/placement.md), a 10-second worked study in
  `examples/07_placement_study/`.
- **Engine**: int-microsecond event core, coalesced scheduler rounds
  (cadence follows `sim.round`). Measured on the shipped 2,048-chip
  example at ρ≈0.9: 14 simulated days in ~4 s, 56 days in ~45 s, 6
  months in ~9 minutes on a laptop. At frontier scale
  (`examples/04_frontier/`): 2 simulated days of a **524,288-chip**
  fleet held at 99.5% occupancy by a closed-loop backlog — ~36K jobs,
  including a 131,072-chip segmented gang reclaiming 32 pods — in
  ~80 s. Cost scales with events × queue depth, so hotter or longer
  runs cost proportionally more; the DESIGN §6.3 "6 months of 100K
  GPUs" envelope is still a target, not yet a measurement.

**Not yet in** (schema present, rejected by `fleetsim validate` with
"not implemented"): `reserved`/`flex_start`/`calendar` as *per-class*
capacity classes (calendar capacity is the top-level `reservations`
section), TPU OCS predicates, multi-gang (Multislice) jobs, autoscaling
inference, Gavel throughput matrices / unpinned chip types, multi-metro
two-stage scheduling. Roadmap: DESIGN.md §11 plus the v0.2 (§16), v0.4
(§17), v0.6 (§18), and v0.7 (§19) addenda. **Still deferred after v0.7**
(v0.6 had pencilled both in for v0.7; v0.7 spent its budget on placement
instead): fractional / sub-chip GPU allocation (unlocks the Alibaba PAI
sharing result) and per-job delay-cause attribution (unlocks Philly
Table 2's fair-share vs fragmentation split).

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
fleetsim viz out_tiered -o replay.html --open       # interactive replay
fleetsim viz out_tiered out_fifo -o ab.html         # A/B overlay

# frontier scale: 524,288 chips, google_fleet preset traffic, a 131K-chip
# 32-pod gang reclaiming a cluster (measured walkthrough: examples/04_frontier/)
fleetsim run examples/04_frontier/scenario.yaml -o out_frontier
fleetsim viz out_frontier -o frontier.html

# v0.4 physics & economics (measured walkthroughs in each README)
fleetsim run examples/05_topology_tradeoff/scenario.yaml -o out_penalty
fleetsim run examples/05_topology_tradeoff/scenario.yaml \
    --override penalties.xover.pod=1.0 -o out_free
fleetsim compare out_penalty out_free               # relax vs pay, measured
fleetsim run examples/06_economics/scenario.yaml -o out_econ   # quota + calendar block + spot

# v0.7 placement: same ordering, same seed, different WHERE (~10 s total)
for P in first_fit best_fit consolidate spread; do
  fleetsim run examples/07_placement_study/scenario.yaml \
      --override scheduler.params.placement=$P -o out_$P
done
fleetsim compare out_first_fit out_best_fit out_spread

# the local web app: browse runs, launch scenarios, 2D report + 3D replay
fleetsim serve --open

# validation: reproduce published trace results (docs/validation.md)
fleetsim validation run                 # vendored-slice checks, no network
fleetsim validation cite helios         # trace attribution (SC '21)

fleetsim --version
```

`run` prints a summary table and writes to the output directory:

| File | Contents |
|---|---|
| `summary.json` | every metric below, full-run and steady-state window |
| `jobs.parquet` | one row per job: timings, status, preemptions, productive/lost chip-seconds |
| `timeseries.parquet` | per-round samples: allocated/healthy chips, queue depth, fragmentation |
| `stints.parquet` | who-ran-where-when: one row per allocation stint × domain (`outputs: {stints: pod}`) — the visualizer's replay input |
| `plots/*.png` | JCT + queue-wait CDFs, occupancy + goodput timelines (`outputs: {plots: true}`) |

Python one-liner (same pipeline):

```python
import fleetsim
summary = fleetsim.run_scenario("examples/01_minimal/scenario.yaml", "out")
```

## Visualizer

```
fleetsim viz OUT_DIR [OUT_DIR_B] [-o REPORT] [--title T] [--map-level L] [--open]
```

One self-contained HTML file (no CDN, no fonts, no fetch — opens from
`file://`): a playable fleet map colored by workload class with
failure/drain pulses, occupancy/allocation/pending/preemption
timelines, a top-hogs gantt, queue-wait + JCT CDFs, and event ticks
for preemption waves, failures, and frontier-gang launches. Pass a
second run directory for dashed A/B overlays and side-by-side summary
cards. The fleet map needs `outputs: {stints: pod}` in the scenario
(examples 01 and 04 set it); without stints the report degrades to
fleet-level replay and says so. Every pixel maps to a documented
column of the run outputs — the full honesty contract, controls, and
performance notes live in [docs/visualizer.md](docs/visualizer.md).

## Web app

```
fleetsim serve [-p 8500] [--workspace ./fleetsim-runs] [--host 127.0.0.1] [--open]
```

A local web app on the same pipeline (pure stdlib `http.server`, no new
dependencies, no build step): a runs rail with status and headline
metrics, a YAML editor with the bundled examples as templates,
validate-before-run with CLI-parity errors, live per-round progress
while a run executes, the 2D report inline (and downloadable), and a
three.js **3D fleet replay** — halls of pods as stacks of node slabs,
colored by class, with failure/drain pulses and camera poses (three.js
is vendored into the package; the app makes zero external requests).
Binds 127.0.0.1 only unless you explicitly widen it; run ids are
server-generated, every path is containment-checked, scenarios are
`yaml.safe_load`-parsed and executed in-process. Full API contract,
workspace layout, security posture, 3D controls, and troubleshooting:
[docs/webapp.md](docs/webapp.md).

## Validation

Are the numbers **real**? The suite replays published cluster traces and
checks fleetsim against the papers' reported results — not just internal
consistency.

**Headline (stated exactly as strongly as the evidence supports):**
fleetsim reproduces the **Helios (SC '21, Hu et al.) FIFO-vs-SJF
average-JCT policy effect** across all four production clusters, under
per-VC **September-max** capacity sizing (§4.4 of the docs), a strict
(blocking) scan (§4.3) and **`consolidate`** placement (§4.2) — the SJF
advantage, **all four** JCT ratios inside fleetsim's **[1.3–8]×**
tolerance band, all four queuing ratios inside **[3–25]×**, the
**Saturn-strongest → Uranus-weakest** JCT-ratio rank, and the
queuing-share ordering. Absolute FIFO JCT lands within ±14 % on every
cluster and `#Queuing` within ±12 %.

As of v0.7 that rung is **complete — no out-of-band number, no `xfail`**
(the four-cluster test passes in 266 s against the real 36 MB trace).
"Complete" is bounded, and the docs say how: it holds under the three
modeling choices named above; the *absolute* rung is "right ballpark"
(±14 %), not reproduced, and does not reproduce the published absolute-JCT
rank because two of its values are a dead tie; and passing the ratio bands
is not by itself evidence for the placement model, since `spread` passes
them too.

FIFO / SJF average-JCT ratio, real Helios September trace (per-VC replay,
strict scan, September-max sizing, `consolidate` placement — deterministic,
seed 0):

| Cluster | Published | fleetsim v0.7 | v0.6 (`first_fit`) | In 1.3–8× band |
|---|---|---|---|---|
| Saturn | 6.59× | **6.87×** | 8.75× ✗ | ✅ |
| Venus | 3.07× | **3.21×** | 4.21× | ✅ |
| Earth | 2.87× | **2.95×** | 2.11× | ✅ |
| Uranus | 1.49× | **1.51×** | 1.69× | ✅ |

The tolerance bands are **not** tightened to match, even though every point
value is now within 5 %: the paper's analysis window is unpublished and the
per-VC capacity model is a choice. The result is also order-sensitive —
35.5 % of Saturn's jobs share an exact submit second, and reordering within
those seconds moves its FIFO JCT by 17 %. So the agreement is reported, not
asserted, and the placer selection rests on the four-cluster aggregate rather
than on any single number (§4.5 of the docs).

And a limit on what that table proves, measured rather than assumed: the
same bands and ranks are also satisfied by `spread`, the deliberate
**anti**-policy, which is 45 % off Saturn's absolute FIFO JCT and 99 % off
Earth's. Choosing a placer takes the *absolute* numbers, not the band — so
the validation asserts both (§4.2.3).

v0.6's one out-of-band number (Saturn 8.75×) was blamed on "FirstFit
fragmenting large gangs." **That was wrong** — the replay fleet is
single-level, so which nodes a gang takes cannot matter, and the VC carrying
84 % of the gap has no gang above 56 GPU. The real cause was *sub-node*
stranding: 97 % of that VC's jobs are 1-GPU, and first-fit-by-id placement
leaves free chips as 1–7-GPU remainders that no ≥ 8-GPU job can use, so the
FIFO head idles behind them. Packing small jobs tightest-first fixed it, and
the wrong diagnosis is kept in the record. Full story, counterfactuals, and
the hypotheses that were measured and rejected:
[docs/validation.md](docs/validation.md) §4.2.

Want the same mechanism without a 36 MB download? `examples/07_placement_study/`
runs all four placers on 256 chips in ~10 s and reports the measured
deltas — stranded whole nodes 6.55 → 5.31 of 32, 8-node-gang mean queue
wait 19,057 s → 16,195 s, and `spread` never placing 80 of 109 of them —
plus the seed where `best_fit` loses.

```bash
fleetsim validation run             # vendored-slice checks, no network (~5 s)
fleetsim validation cite helios     # the SC '21 attribution the trace requires

# full replays are opt-in (excluded from CI); Helios data.zip is 36 MB:
FLEETSIM_HELIOS_FULL=1 pytest -m trace_full validation/test_helios_ratio.py
```

CI runs only the vendored slices (real 2-VC Venus + synthetic Philly): the
full-trace rungs self-skip unless their `FLEETSIM_*_FULL` env var is set, so
the whole suite runs with no network in **~80 s with exactly 4 skips** (two
Helios `trace_full`, one Philly `trace_full`, one `FLEETSIM_FRONTIER_BUDGET`
gate — anything else skipping is a real signal), and `-m "not trace_full"`
deselects them outright. Trace downloads are
stdlib-only (`urllib` + `hashlib`) and integrity-gated by exact byte size
plus Git-LFS-pointer detection (a full SHA-256 is also verified when the
registry carries one; the two shipped real traces are size-gated) — so a
truncated or wrong-size download is refused, never silently used.

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

`self.placement` is the orthogonal *where* axis: swap `FirstFit()` for
`BestFit()`, `Consolidate()`, `Spread()`, or your own one-method policy —
and accept `placement=None` in `__init__` so YAML can select one by name.
[docs/placement.md](docs/placement.md) has the exact semantics of each,
when to use which, and the measured differences.

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
| v0.4 feature-keyed | `relaxed` / `quota_demoted` job columns, `counts.relaxed_placements` / `counts.quota_demotions`, and the per-reservation report (`reservations` in summary.json: nodes, evictions at claim/hard-end, `utilization`) — present only when the matching config section is used, so feature-off runs stay byte-identical to pre-v0.4 |
| **Stranded whole nodes** (v0.7) | count of HEALTHY nodes that are *partially* occupied — free capacity no whole-node gang can claim — plus the chips on them (`stranded_whole_nodes` / `stranded_whole_node_chips` timeseries columns, `fragmentation.stranded_whole_nodes` mean/max, `counts.placement_policy`). Gated on `scheduler.params.placement` naming a policy, so a scenario that names none keeps the exact pre-v0.7 schema |

All distributional metrics are reported both job-weighted and
chip-hour-weighted (mice vs hogs), over the full run and a configurable
steady-state window.

## Determinism contract

Every run is a pure function of `(fleet config, workload config | trace,
seed)`. Named independent RNG streams (`arrivals/<class>`,
`size/<class>`, `failures`, `repair`, `maintenance`, ...) mean enabling
failure injection never perturbs the arrival sequence — A/B scheduler
comparisons are paired experiments, not noise. Identical `(scenario,
seed)` produce **byte-identical** Parquet and JSON outputs on a given
platform (`validation/test_determinism.py` enforces this in CI). Across
OS/architectures, results may differ by float ULPs: the lognormal and
exponential samplers route through the platform libm, whose `exp`/`log`
differ in the last bit between e.g. glibc and Apple's libm — so
cross-platform regression tests compare with tolerances, never golden
hashes (`tests/test_traffic_v02.py::TestBackwardCompat`). CI also runs
closed-form queueing rungs — M/M/c vs Erlang-C, Pollaczek–Khinchine
M/G/1 under lognormal service, preemptive-resume priority M/G/1
per-class sojourns, best-effort backlog shielding — invariant property
tests, and the v0.4 rungs: backfill head-job non-delay (paired run,
exact estimates), quota conservation at every flush, reservation
exclusivity via stints cross-check, and the analytic speed-penalty
completion identity.

## License

Apache-2.0 — see [LICENSE](LICENSE).
