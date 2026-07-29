# Example 04 — frontier scale

**The question**: a 131,072-chip pretraining job — 32 data-parallel pod
segments, the Llama-3 8×3072 / TPU-Multislice shape — lands on a
524,288-chip fleet that is already >99% occupied. What happens?

**The measured answer** (everything below is from the two runs at seed
42, not projected): the scheduler empties an entire cluster of
best-effort work in one wake — a spike of **345 preemptions**, ~117K
chips freed and refilled within two rounds — and the frontier job
starts **233 seconds** after submission, spanning **32 pods**. Prod
pretrain waits stay at a couple of rounds. The fleet never dips below
~99% occupancy except for the one reclaim dip.

## The setup

- **Fleet**: 4 clusters × 32 pods × 16 SUs × 32 nodes × 8 chips =
  **524,288 H100s** (DESIGN §3.3 quanta: SU = 256 GPUs per the SuperPOD
  RA, pod = 4,096, cluster = 131,072).
- **Traffic**: the `google_fleet` preset supplies the class structure
  (docs/traffic-math.md §2) — Hawkes-burst evals + MMPP-2 diurnal
  finetunes + rare Poisson pretrains with the splice duration tail +
  a closed-loop best-effort backlog — with per-class overrides
  re-anchoring rates and sizes to this fleet (see scenario comments).
  Open-loop classes offer ~48% load; the backlog fills the rest.
- **The frontier class**: one 131,072-chip gang (`chips:
  fixed[131072]`, `within: cluster`, `segment_level: pod`,
  `segment_nodes: 512`), prod tier, Poisson-timed so exactly one lands
  inside the 2-day horizon at this seed (hour 20.6).
- **Scheduler**: `tiered_priority` with REQUEUE preemption (segmented
  reclaim plans victims per pod, dry-run-verified against real node
  shapes and leaf health, one aggregate victim set).
- **Failures are off**, deliberately: at the Meta-RSC 42 node-day MTBF
  a 16,384-node gang would be interrupted every ~23 minutes (the
  Llama-3 405B job, 8× smaller, measured one per ~3.1 h), and the
  restart thrash would swamp the scheduling story this example
  isolates. Failure amplification is exercised in examples 01/02.

## Reproduce (two runs + compare, ~2 minutes total)

```bash
fleetsim run examples/04_frontier/scenario.yaml -o out_frontier
fleetsim run examples/04_frontier/scenario.yaml -o out_baseline \
    --override workload.classes.best_effort=null
fleetsim compare out_baseline out_frontier
fleetsim plot out_frontier      # occupancy timeline shows the reclaim dip
fleetsim viz out_frontier -o frontier.html   # replay the 32-pod reclaim
```

The `viz` report (from `outputs: {stints: pod}`) is the reclaim as a
movie: 128 pods across 4 clusters, the best-effort band vanishing from
one cluster in a single round and the frontier gang landing across all
32 of its pods — with the preemption-wave tick and the frontier
submit/start events marked on the scrubber (docs/visualizer.md).

Measured wall time: ~80 s for the backlog run, ~40 s for the baseline
(Apple-silicon laptop, pure Python). The only difference between the
runs is deleting the backlog class — arrivals, sizes, durations and
outcomes of every other class are byte-identical (named per-class RNG
streams, DESIGN §6.2), so this is a paired experiment.

## Measured results (seed 42; scope labeled per row)

From `fleetsim compare out_baseline out_frontier`. Occupancy/goodput
rows are steady-state **window** (middle 80%) values; queue-wait, JCT,
preemption and job-count rows are **full-run** values (that is what
`compare` prints — the windowed variants live in `summary.json` under
`"window"` and differ, e.g. batch-finetune wait p50 103.0 s windowed
vs 116.4 s full). Wait/JCT rows are per **source class** — the
workload class that generated the job — and, per the DESIGN §16.1
contract, the best-effort backlog is *excluded* from every wait/JCT
distribution: best-effort wait is undefined under a saturated closed
loop (report best-effort goodput, never its wait).

| metric                                   | no backlog | with backlog |
|------------------------------------------|-----------:|-------------:|
| occupancy (window)                       |      0.477 |    **0.995** |
| goodput (window)                         |      0.966 |        0.880 |
| **productive fraction** (occ × goodput)  |      0.461 |    **0.876** |
| queue wait p50 evals (s, full run)       |       61.3 |         77.8 |
| queue wait p50 batch finetunes (s, full run) |   66.9 |        116.4 |
| queue wait p50 pretrains (s, full run)   |       51.7 |        163.8 |
| JCT p50 evals (s, full run)              |      190.4 |        237.2 |
| JCT p50 batch finetunes (s, full run)    |    9,794.8 |     10,105.4 |
| preemptions/min (full run)               |      0.068 |        0.565 |
| jobs finished (full run)                 |     23,538 |       34,422 |

And the frontier job itself (`jobs.parquet`, `job_id == "frontier-0"`):

| | no backlog | with backlog |
|---|---:|---:|
| submitted (paired streams)    | hour 20.57 | hour 20.57 |
| queue wait                    | 113 s | **233 s** |
| `n_domains_spanned`           | 32 pods | 32 pods |
| status at horizon             | RUNNING (27.4 h elapsed) | RUNNING (27.4 h elapsed) |

## How to read that

- **The backlog buys ~52 points of occupancy** (0.477 → 0.995) and
  1.9× the productive work per fleet-hour (0.461 → 0.876 of capacity,
  goodput-discounted). Goodput *drops* from 0.966 to 0.880 because
  reclaimed best-effort stints are genuine waste — the backlog class
  runs with `checkpoint_interval: 0s`, so every eviction loses the
  stint (that is what makes it instantly reclaimable: zero-length
  grace). Occupancy ≠ goodput is the point, and both are reported.
- **The frontier job's arrival is visible in the timeline**
  (`timeseries.parquet` around hour 20.6): the cumulative-preemption
  counter jumps by 345 in a single wake, `allocated_chips` dips
  523,946 → 406,406 and recovers to 522,998 one round later (pending
  spikes 116 → 441 as the victims requeue) — one cluster emptied for
  the 32-segment gang, the hole refilled by the backlog. The frontier
  job waits 233 s on a 99.5%-occupied fleet vs 113 s on a half-empty
  one: the price of full was **one extra scheduler round** (233.4 −
  113.4 = 120.0 s, exactly one `round: 120s`; `first_start` moves one
  wake boundary, 74,160 → 74,280 s).
- **Prod pretrain waits stay at rounds, not hours**: p50 moves 52 s →
  164 s — one vs two-ish 120 s rounds, unchanged in kind. **Batch
  finetunes pay about one eviction cycle at the median and more in the
  tail**: p50 wait 66.9 s → 116.4 s, but mean 181 → 381 s and p90
  269 → 1,282 s, because on a full fleet a finetune must evict
  best-effort work first (preempt wake → grace → place at a later
  wake, and the scheduler commits one job's victim set per wake).
  Their JCT p50 moves 9,795 s → 10,105 s — ~3% worse; minutes of
  eviction latency against hours of runtime.
- **Eval medians barely move, eval tails stretch**: p50 61 → 78 s
  (sub-node gangs slip into leaf-level holes even at 99.5% occupancy),
  but p99 goes 236 s → 1,292 s (~4 min → ~22 min) and the mean 64 →
  141 s. Still minutes, not hours — and reported, not hidden.
- **The backlog itself** (11,252 closed-loop jobs in the frontier run)
  appears only in occupancy/goodput and the preemption counters —
  summary.json deliberately reports no best-effort wait/JCT stats
  (DESIGN §16.1 / traffic-math §2.1: undefined under saturation).

Every number above is deterministic for `(scenario, seed 42)`; rerun
the commands and diff.
