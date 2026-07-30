# Example 07 — the placement study: where you put the mice decides how long the gangs wait

**Question**: two runs can pick the *same job next* and still get very
different fleets. Ordering (*which* job) and placement (*which chips*)
are separate axes — how much does the second one actually matter?

This scenario holds ordering fixed (one best-effort FIFO, one seed, one
workload) and varies **only** `scheduler.params.placement`.

## The mechanism in one paragraph

fleetsim's allocation rule (DESIGN §4.1) is that a node with **any**
owner is invisible to a request for whole nodes. So every sub-node job
that lands on a *fully free* node converts 8 whole-node chips into a
1–7-chip remainder that no gang can ever claim. Which node a 1-chip job
lands on is a **placement** decision — and it is the gangs, not the
mice, that pay for it. `first_fit` takes the lowest-id node with room
(which is a fully-free node whenever the low-id nodes happen to be
busy). `best_fit` takes the *tightest sufficient* node, so mice fill
existing remainders instead of manufacturing new ones. `spread` takes
the **emptiest** node — the deliberate anti-policy.

This is the same mechanism that closed the last Helios deviation on the
real SC '21 trace ([docs/validation.md](../../docs/validation.md) §4.2);
this example is the 10-second version of it, with no download. Policy
semantics: [docs/placement.md](../../docs/placement.md).

- **Fleet**: 4 racks × 8 nodes × 8 chips = **256 H100s** (32 nodes).
  Deliberately **two** levels, so `best_fit` and `consolidate` are
  distinguishable here — on a single-level fleet they are bit-identical.
- **Workload**: `mice` (1/2/4-chip, hours long, ~40 % of offered load)
  + `gangs` (16/32/64-chip = 2/4/8 **whole nodes**, ~38 %). Offered load
  ≈ **0.78** by construction (0.81 realized at seed 42 — Poisson arrivals
  and a lognormal duration tail land above the mean here) — deliberately
  *not* near saturation, so the queue numbers measure placement rather
  than how close each arm sits to instability.
- Failures **off**, `abort_prob: 0` everywhere: every difference between
  the runs is a placement difference.
- Sizes are picked so the summary's chip-count buckets split the two
  populations exactly: mice are all of `1-8`, gangs all of `9-64`.

## Run it

```bash
for P in first_fit best_fit consolidate spread; do
  fleetsim run examples/07_placement_study/scenario.yaml \
      --override scheduler.params.placement=$P -o out_$P
done
fleetsim compare out_first_fit out_best_fit out_spread
fleetsim viz out_first_fit out_best_fit -o ab.html --open
```

Measured wall time on a laptop: **1.0 s** each for `first_fit`,
`best_fit` and `consolidate`, **6.8 s** for `spread` (its queue is 30×
deeper, and scheduler cost scales with queue depth) — **~10 s** for the
whole study.

Because the scenario **names** a placement policy, each run also emits
the v0.7 placement diagnostics: a `stranded whole nodes (mean)` row in
the console summary, `fragmentation.stranded_whole_nodes` in
`summary.json`, `counts.placement_policy`, and
`stranded_whole_nodes` / `stranded_whole_node_chips` columns in
`timeseries.parquet`. `fleetsim compare` picks them up too, so the
three-way comparison ends with the rows that actually separate the arms:

```
stranded whole nodes (full mean)           6.55        5.31       25.31
placement policy                      first_fit    best_fit      spread
```

(Naming `first_fit` explicitly changes no scheduling — it is the default
either way — it only switches the extra columns on. Examples 01 and 04
name no policy, which is why their outputs stay byte-identical to v0.6,
and why a pre-v0.7 output directory compares exactly as it always did.)

## What it measures (seed 42, 14-day horizon, full run)

Every number below came out of exactly the commands above.

| metric | `first_fit` | `best_fit` | `consolidate` | `spread` |
|---|---|---|---|---|
| **stranded whole nodes, mean (of 32)** | 6.55 | **5.31** | 5.37 | 25.31 |
| stranded whole nodes, max | 16 | **13** | 14 | 32 |
| free chips on partial nodes, mean | 16.95 | 16.49 | 16.70 | 104.47 |
| ⇒ whole-node capacity denied (chips) | 52.4 | **42.5** | 42.9 | 202.5 |
| **gang queue wait, mean (s)** | 9,635.6 | **8,778.4** | 8,875.9 | 213,961.3 |
| **64-chip (8-node) gang wait, mean (s)** | 19,056.8 | **16,195.3** | 16,314.7 | 739,842.1 |
| 64-chip gang wait, p50 (s) | 14,246.8 | 12,128.4 | **10,766.8** | 785,962.1 |
| 64-chip gangs ever started (of 109) | 107 | 107 | 107 | **29** |
| mice queue wait, p50 (s) | 31.0 | 31.4 | 31.3 | 30.1 |
| node frag index, mean | 0.766 | **0.745** | 0.751 | 0.939 |
| occupancy, full run | 0.8141 | 0.8141 | 0.8141 | **0.5829** |
| occupancy, steady-state window | 0.8491 | **0.8511** | 0.8489 | 0.5951 |
| goodput, full run | 0.9831 | 0.9831 | 0.9831 | 0.9820 |
| allocated chip-hours | 70,021.35 | 70,021.35 | 70,021.35 | **50,139.90** |
| jobs completed | 3,436 | 3,436 | 3,436 | **3,309** |
| gangs completed (of 285) | 279 | 279 | 279 | **152** |
| mean pending jobs | 2.38 | **2.22** | 2.23 | 68.50 |

Per gang size (queue wait, s). The first three arms average over the same
**88 / 88 / 107** gangs; `spread` only ever started **86 / 39 / 29** of
them, so its column averages a smaller and luckier set — its true cost is
the 80 gangs missing from it, not the numbers shown:

| gang size | | `first_fit` | `best_fit` | `consolidate` | `spread` |
|---|---|---|---|---|---|
| 16 chips (2 nodes) | mean | 3,745.8 | 3,647.6 | 4,177.4 | 26,563.7 |
| 32 chips (4 nodes) | mean | 4,070.0 | 4,890.9 | 4,529.6 | 236,157.3 |
| **64 chips (8 nodes)** | mean | 19,056.8 | **16,195.3** | 16,314.7 | 739,842.1 |
| 64 chips | p90 | 44,911.9 | 39,822.7 | **39,318.7** | 804,651.4 |

## How to read it

1. **Occupancy and goodput cannot tell you which placer you ran.** The
   `first_fit`, `best_fit` and `consolidate` arms deliver
   **bit-identical** full-run occupancy (0.8140502977503281), goodput
   (0.9831464030337717) and allocated chip-hours (70,021.35041129222),
   because the same 3,436 jobs complete in all three and each runs for its
   own duration. The placement difference is entirely in **when** they ran.
   (The steady-state-*window* occupancy does move — 0.8491 / 0.8511 /
   0.8489 — because the window clips the timeline, but 0.2 pp is inside
   what any reader would call noise.) A placement study that reports only
   occupancy has measured almost nothing; this table is the proof.
2. **The mechanism metric is the one that moves cleanly.**
   `stranded_whole_nodes` drops 6.55 → 5.31 nodes (−19 %) from
   `first_fit` to `best_fit`, and *that* is what the gangs feel: 6.55
   partial nodes deny 52.4 chips of whole-node capacity where 5.31 deny
   42.5. Note that the **chip** count barely changes (16.95 → 16.49):
   both placers strand about the same number of free chips, but
   `first_fit` smears them over more nodes. For a whole-node gang the
   node count is the quantity that matters, which is exactly why v0.7
   added it as a metric.
3. **The biggest gangs pay the most.** 64-chip gangs need 8 nodes no
   mouse has touched; their mean wait improves 15 % (19,057 s →
   16,195 s) while 16-chip gangs move −2.6 %. Placement is a large-gang
   phenomenon — the same conclusion example 05 reaches from the
   topology side.
4. **Mice are indifferent** (p50 wait 31.0 s vs 31.4 s). The cost of
   sloppy small-job placement is not paid by the small jobs. Their mean
   wait actually gets slightly *worse* under `best_fit` (80.7 s →
   95.3 s): a mouse sent to the tightest node sometimes has to wait for
   one, where first-fit would have opened a fresh node immediately. That
   is the trade being made, and it is a good one here — 15 s of mouse
   wait buys ~2,900 s of gang wait.
5. **`spread` is the control arm, and it fails loudly.** It strands 25
   of 32 nodes, never places 80 of the 109 8-node gangs at all, finishes
   127 fewer jobs and delivers **28.4 % fewer chip-hours** — on
   identical hardware with identical arrival times. It is in the library
   precisely so a placement study can show that its effect is real and
   signed; "spread for blast-radius or thermal reasons" is also a policy
   real operators run, and this is its bill.
6. **`best_fit` ≠ `consolidate` on this fleet.** They differ only when a
   whole-node gang fits inside no single parent domain, and this fleet
   has 4 racks — hence 16,195 s vs 16,315 s on the 64-chip mean and
   12,128 s vs 10,767 s on its median. On a single-level fleet
   (`levels: ["node"]`) they are bit-identical; see
   [docs/placement.md](../../docs/placement.md).

## Four things this example does **not** show

Stated because a teaching example that only shows its own thesis is
advertising.

1. **`best_fit` does not win on every seed.** Re-running with
   `--override sim.seed=N` (everything else identical):

   | seed | occupancy | stranded nodes ff → bf | gang wait mean ff → bf | 64-chip mean ff → bf |
   |---|---|---|---|---|
   | 42 | 0.81 | 6.55 → 5.31 (−19 %) | 9,636 → 8,778 (−9 %) | 19,057 → 16,195 (−15 %) |
   | 7 | 0.71 | 7.03 → 5.66 (−19 %) | 2,839 → 2,511 (−12 %) | 4,869 → 4,179 (−14 %) |
   | 99 | 0.82 | 6.73 → 5.41 (−20 %) | 15,038 → 11,841 (−21 %) | 27,758 → 20,142 (−27 %) |
   | 1234 | 0.78 | 6.63 → 5.31 (−20 %) | 8,205 → **9,407 (+15 %)** | 14,881 → **16,039 (+8 %)** |

   The **mechanism** metric moves by −19…−20 % on all four seeds; the
   **outcome** improves on three and regresses on one. That is the same
   shape as the Helios finding, where `consolidate` closed the cluster
   aggregate while making three of the five Saturn VCs that carry 97 % of
   the FIFO−SJF gap *worse* — one of them by 18.5 %
   (docs/validation.md §4.2). *Reproduces the reference placer's
   aggregate behaviour* is a claim placement policies support;
   *best-fit is a better scheduler* is not.
2. **One pooled quantile moves the "wrong" way, and it is a mix
   artifact.** Pooled gang wait p50 is 1,737 s under `first_fit` vs
   2,939 s under `best_fit` — while the per-size medians differ by under a
   second on 16- and 32-chip gangs (57.2 vs 57.8; 54.1 vs 54.8) and improve
   15 % on 64-chip ones. Both arms place exactly **41.3 %** of gangs within
   120 s, so the pooled median sits precisely on the knife edge between
   "placed immediately" and "waited": it is a quantile of a mixture of
   three size populations, not a statement about any of them. `fleetsim
   compare` shows that p50 — read the per-size table above instead.
3. **Hierarchical constraints can invert the ranking.** Add
   `--override workload.classes.gangs.within=rack` (making every gang
   rack-local, so a 64-chip gang needs a *fully empty rack*) and at
   seed 42 `best_fit` becomes far **worse** than `first_fit` on those
   gangs: 64-chip mean wait 39,677 s → 121,486 s, with 99 rather than
   107 of them ever starting. Node-level tightest-fit spreads *gangs*
   across racks, so no rack ever drains — the opposite of what a
   rack-shaped job needs. Placement policy interacts with topology
   constraints; measure your own fleet rather than assuming best-fit.
4. **This is a floor, not a best case.** The scheduler here is
   **best-effort** (`strict: false`), so a gang that cannot place does
   not hold up the mice behind it. Head-of-line blocking amplifies the
   same mechanism substantially — that is what produces the 26 % mean-JCT
   swing on the Helios Saturn replay (docs/validation.md §4.3).
