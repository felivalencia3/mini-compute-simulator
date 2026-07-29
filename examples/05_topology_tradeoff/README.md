# Example 05 — the topology tradeoff: relax the constraint, pay the crossing

**Question**: when a pod-sized pretrain can't find a pod-shaped hole,
should it *wait* for one or *run ugly* across pods at reduced speed?

Before v0.4 the simulator could only wait: `within: pod` was a hard
constraint. Relaxing it **without** a cost model would be strictly
dominant (free speed, no downside), which is why DESIGN §4.2 pinned
relax-after-timeout and the placement-quality penalty as a matched pair
to ship together. This example runs the pair:

```yaml
penalties:
  xover: {pod: 0.7}                 # cross-pod placements run at 0.7x
workload:
  classes:
    pretrain:
      within: {level: pod, required: false, relax_after: 10m}
```

- **Fleet**: 8 pods × 16 racks × 8 nodes × 8 chips = 8,192 H100s
  (pod = 1,024). Failures off — this example isolates placement quality.
- **Workload**: pretrains draw {256, 512, 1024} chips; batch finetunes
  and diurnal eval mice keep the fleet ~86% utilized so pod-shaped holes
  are scarce. Scheduler: best-effort FIFO with first-fit (ascending-id
  bin packing).
- **Run A** (the scenario as written): crossings cost 0.7×.
- **Run B** (`--override penalties.xover.pod=1.0`): crossings are free.
  Penalties never change *feasibility* at any instant — but slower jobs
  hold chips longer, so the two schedules diverge downstream.

## Run it

```bash
fleetsim run examples/05_topology_tradeoff/scenario.yaml -o out_penalty
fleetsim run examples/05_topology_tradeoff/scenario.yaml \
    --override penalties.xover.pod=1.0 -o out_free
fleetsim compare out_penalty out_free
fleetsim viz out_penalty out_free -o report.html --open
```

The viz report (stints at pod level) shows relaxed pretrains as jobs
painting more than one pod row; in compare mode run B's frames overlay
run A's timeline.

## What it measures (seed 42, 7-day horizon)

Both runs, byte-for-byte from `fleetsim compare` and the parquet outputs
(all numbers below were produced by exactly the commands above):

| metric | A: penalty 0.7 | B: no penalty |
|---|---|---|
| relaxed placements (full run) | **35** of 112 pretrains | 17 of 112 |
| relax rate by size (A) | 256: 18% · 512: 33% · 1024: **61%** | — |
| occupancy (window) | **0.862** | 0.826 |
| goodput (window) | **0.861** | 0.983 |
| pretrain queue wait p50 (s) | 484.5 | 47.9 |
| pretrain JCT p50 (s) | 51,002 | 43,136 |
| completed pretrains | 99 | 103 |
| eval wait p50 (s) | 31.9 | 31.4 |

Inside run A, the price of relaxing is visible per job (completed
pretrains, `jobs.parquet` `relaxed` column):

| completed pretrains (run A) | JCT p50 (s) | JCT p90 (s) | JCT mean (s) |
|---|---|---|---|
| clean (in one pod), n=73 | 48,515 | 103,498 | 55,722 |
| relaxed (cross-pod), n=26 | 67,339 | 127,538 | 81,184 |

## How to read it

1. **The relax rate tracks shape scarcity.** A 1024-chip draw needs a
   fully empty pod: 61% of them give up waiting and go cross-pod. A
   256-chip draw usually finds a hole: only 18% relax.
2. **Occupancy and goodput split — in opposite directions.** Run A
   *looks* busier (0.862 vs 0.826 occupancy) because penalized jobs hold
   chips ~1.43× longer for the same work; its goodput is 12 points
   *lower* (0.861 vs 0.983). Allocation is not useful work — the
   penalty makes the classic trace-paper error measurable in one A/B.
3. **The penalty feeds back into the queue.** Slower relaxed jobs keep
   pods fragmented longer, so run A's pretrains wait 10× longer at the
   median (484 s vs 48 s — most of it the 10-minute relax timer) and
   more of them end up relaxing in the first place (35 vs 17): congestion
   begets crossings begets congestion.
4. **Mice don't care.** Eval waits are unchanged (~32 s both runs) —
   the topology tradeoff is a big-job phenomenon, exactly as the
   fragmentation literature says.

Goodput in run B is 0.983 rather than 1.0 because of the checkpoint-save
amortization (1 h interval / 60 s save ⇒ eff ≈ 0.984) — the remaining
gap in run A (0.861) is the crossing penalty.
