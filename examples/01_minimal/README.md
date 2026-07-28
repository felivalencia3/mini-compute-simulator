# Example 01 — minimal fleet, synthetic mix

2,048 H100s (2 pods × 16 racks × 8 nodes × 8 chips), a trace-derived
pretrain/finetune/eval mix, node failures + maintenance drains, and
tiered-priority scheduling with REQUEUE preemption. 14 simulated days run
in a few seconds.

Traffic is tuned to a **utilization target of ρ ≈ 0.9** (offered load /
capacity at this seed, with the DESIGN §5.1 default 30% abort mix): hot
enough for real contention, but *stable* — an overloaded system (ρ > 1)
grows its backlog forever and every queue metric diverges with horizon
length. Because pretrain durations are heavy-tailed (lognormal, p90 =
30 d), other seeds can transiently exceed 1 — that is the tail at work,
not the mean.

```bash
fleetsim run scenario.yaml -o out_tiered
fleetsim run scenario.yaml -o out_fifo \
    --override scheduler.name=fifo --override "scheduler.params={}"
fleetsim compare out_fifo out_tiered
```

What to look at (numbers from the shipped scenario, seed 42):

- **The console table**: occupancy vs allocation rate (the gap = capacity
  lost to failures/drains) and goodput (work that survived checkpoints,
  including still-running jobs' checkpointed progress).
- **`out_tiered/summary.json`**: `full` vs steady-state `window` scopes;
  `preemptions_per_min` split by trigger — including
  `failure_second_order` (a failure-requeued pretrain evicting BATCH
  work, DESIGN §8's failure-amplification cost); `queue_wait_s_by_chips`
  (small vs large gangs); `jobs_by_status` showing the ~30% killed/failed
  mix; `mean_pending_by_class`.
- **The compare table**: strict FIFO head-of-line blocks the whole queue
  behind large gangs — tiered priority cuts the mean EVAL backlog from
  ~11 pending jobs to ~1.3 (**~9×**) and EVAL p99 queue wait from ~5 h to
  ~31 min (**~10×**), while finishing ~1% more jobs overall.
- **`out_tiered/plots/`**: JCT + queue-wait CDFs, occupancy and goodput
  timelines (from `outputs: {plots: true}`).
