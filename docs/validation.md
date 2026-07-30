# The validation suite (`fleetsim validation`, v0.7)

Every prior release proved fleetsim was **internally** consistent —
closed-form queueing rungs (M/M/c, P–K M/G/1), conservation invariants,
determinism. v0.6 asked the harder question: are the occupancy, queue-wait,
and JCT numbers **real** — do they match what a published cluster trace
actually produced? This suite downloads real traces (Helios SC '21, Philly
ATC '19, Alibaba PAI NSDI '22), replays them, and asserts the papers'
reported ratios and distributions. v0.7 closes the one deviation v0.6 could
not — and corrects v0.6's explanation of it (§4.2).

```bash
fleetsim validation run                 # vendored-slice checks, no network
fleetsim validation cite helios         # the attribution the trace requires
FLEETSIM_HELIOS_FULL=1 pytest -m trace_full validation/test_helios_ratio.py
```

The headline result, stated exactly as strongly as the evidence supports:

> **fleetsim reproduces the Helios (SC '21) FIFO-vs-SJF average-JCT policy
> effect across all four clusters** (under per-VC September-max sizing §4.4,
> a strict blocking scan §4.3, and `consolidate` placement §4.2) — the SJF
> advantage, **all four** JCT ratios inside fleetsim's **[1.3–8]×**
> tolerance band, **all four** queuing
> ratios inside its **[3–25]×** band, the **Saturn-strongest →
> Uranus-weakest** JCT-ratio cross-cluster rank, and the
> queuing-share-of-JCT ordering. Absolute FIFO JCT lands within **±14 %**
> on every cluster (Saturn to 0.01 %) and `#Queuing` within **±12 %**.
>
> The tolerance bands are **not** tightened to match, because the paper's
> analysis window is unpublished and the capacity model is a choice (§1,
> §4.4). Separately, the result is **order-sensitive**: 35.5 % of Saturn's
> jobs share an exact submit second, and reordering *within* those seconds
> moves its FIFO JCT by 17 % (§4.5). So agreement closer than ~20 % on any
> single point value here is not evidence of accuracy, and the bands stay
> bands.

**v0.6 → v0.7.** v0.6 had one out-of-band number (Saturn's JCT ratio 8.75×
vs a published 6.59×) and blamed "FirstFit fragments large gangs more than
the reference consolidate placer." **That diagnosis was wrong**, and §4.2
now records what the measurement actually showed and why the corrected fix
(sub-node best-fit packing) closes it. The wrong-but-plausible version is
kept in the record rather than quietly overwritten.

### Is the four-cluster reproduction complete? — the direct answer

**Yes for V1, the policy-effect rung, and it is no longer partial.** Every
assertion V1 makes now passes on the real trace on all four clusters with
**no `xfail`**: all four JCT ratios in `[1.3, 8]×`, all four queuing
ratios in `[3, 25]×`, `n_dropped == 0`, the Saturn-highest /
Uranus-lowest JCT-ratio rank, and the queuing-share ordering
(`test_helios_four_cluster_ratio_bands_and_ranks`, **1 passed in ~260 s** on
a laptop against the cached 36 MB trace). v0.6's single out-of-band number is
gone, and the fix is itself tested (V1p).

**Three things that "complete" does not mean**, each measured rather than
hedged:

1. **It is complete under three stated modeling choices**, not
   unconditionally: strict scan (§4.3), September-max per-VC sizing
   (§4.4), and `consolidate` placement (§4.2). All three are in the
   headline, and all three are options a reader could disagree with.
2. **V2, the absolute rung, is "right ballpark", not reproduced.** ±14 %
   on FIFO JCT and ±12 % on `#Queuing` (§5), with the absolute-JCT *rank*
   explicitly not reproduced because two published values are a dead tie.
3. **V1 passing is not by itself evidence for the placement model** —
   `spread`, the anti-policy, also passes every V1 assertion (§4.2.3).
   What selects `consolidate` is the **four-cluster mean absolute error
   against the published ratios** (3.4 % vs 27.4 % for `first_fit` and
   29.5 % for `spread`) together with the mechanism metric; V2's absolute
   Saturn number corroborates it, but on Saturn alone that number is
   order-sensitive at the same magnitude as the placer effect (§4.5). The
   suite asserts all of it.

Policy semantics and the general (non-Helios) placement story:
[placement.md](placement.md). A 10-second synthetic version of the same
mechanism: `examples/07_placement_study/`.

---

## 1. Thesis: two kinds of claim, weighted differently

A trace replay **never** reproduces a published headline to the last digit:
the released trace differs from the paper's analysis window, timestamps
carry no timezone, and every simulator makes placement choices the original
scheduler did not. So we validate two distinct kinds of claim and weight
them differently.

- **Policy-effect validations (strongest signal).** A *ratio* between two
  policies on the *same* trace — SJF's average JCT vs FIFO's. Ratios cancel
  absolute-scale error: if our load runs 10 % light, both policies are 10 %
  light and the ratio survives. This is the primary target (V1).
- **Distribution-match validations.** Reproduce a shape the trace itself
  carries — Philly's killed/failed %, its GPU-hour split by status. These
  test *converter fidelity* more than the scheduler, and their tolerance is
  dominated by the unpublished analysis window (V3).

## 2. §0 — What fleetsim deliberately does NOT reproduce

Stated up front, never asserted, so a reader never mistakes an anti-goal for
a failure:

| Published number | Why fleetsim cannot reproduce it | Disposition |
|---|---|---|
| Philly ~52.3 % GPU **utilization** (Table 3–5) | A hardware **SM-cycle** counter (`cluster_gpu_util`), not scheduler occupancy. A DES produces *allocation* occupancy — a different, higher quantity. | Out of scope. Documented anti-goal. |
| Philly Table 2 fair-share vs **fragmentation-delay** split | Needs per-job **delay-cause attribution** fleetsim does not compute. | **Still deferred after v0.7** (cause-labeling pass). v0.7's `stranded_whole_nodes` is a fleet-level down payment, not a substitute — §11. |
| Helios **QSSF** column | Needs a `jobname` column (absent from the public CSV) + a GBDT duration predictor. | Out of scope; **SJF-oracle is the reproducible upper-bound proxy** (V1). |
| Alibaba 50 % GPU-sharing saving, median 0.042 GPU/inst | Needs **fractional / sub-chip GPU** allocation; v0.7 still shares only whole chips within a node, and no placement policy substitutes for it. | **Still deferred after v0.7** (fractional-GPU packing) — §11. |
| Borg absolute occupancy / JCT | Resources are normalized `[0,1]`, BigQuery-only, 8 independent 12k-machine cells. | Never a replay target; generator-distribution check only. |
| CPU contention, co-location interference | Not modeled (analytical speed model, no memory-bandwidth contention). | Out of scope. |

Note the two "deferred" rows: v0.6 wrote them down as *v0.7 targets*, and
v0.7 spent its budget on placement instead. They are recorded here as
still-open rather than quietly re-dated.

## 3. The validation ladder

| # | Validation | Kind | Shipped as | CI (vendored) | Full run |
|---|---|---|---|---|---|
| **V1** | Helios FIFO-vs-SJF JCT & queuing **ratio** | policy-effect | `validation/test_helios_ratio.py` | ✅ 2-VC Venus slice, direction | opt-in `FLEETSIM_HELIOS_FULL` |
| **V2** | Helios FIFO **absolute** Table 3 | distribution | reported below (harness rung) | — | opt-in (same trace) |
| **V3** | Philly job-**status** split | distribution | `validation/test_philly_status.py` | ✅ ~2k-row slice | opt-in `FLEETSIM_PHILLY_FULL` |
| **SPT** | SJF is SPT-optimal on a fungible pool | analytic | `validation/test_sjf_ordering.py` | ✅ always | n/a |
| **V1p** | Helios Saturn: the placement model is load-bearing | policy-effect | `validation/test_helios_ratio.py` | ✅ 3-placer slice inequality | opt-in `FLEETSIM_HELIOS_FULL` |
| V4 | Philly fragmentation-delay direction | policy-effect | deferred | — | later |
| V5 | Alibaba PAI run-time / gang distribution | distribution | deferred | — | later |
| V6 | Borg generator heavy-tail | generator | deferred | — | later |

The four shipped rungs (V1, V1p, V3, SPT) plus the V2 absolute reporting are
the v0.7 suite. **V1p** is new: it replays Saturn under `first_fit`,
`consolidate` and `spread` and asserts the v0.6 baseline really is out of
band while the shipped policy is in it, that only the shipped policy lands
within ±20 % of the published **absolute** FIFO JCT, and that the ratio band
alone does *not* separate them (`spread` passes it — §4.2.3). So §4.2's
modeling choice is tested rather than asserted, and a future placement
regression fails loudly. V4–V6 wait on
fractional-GPU allocation and a delay-cause pass (see §11 and DESIGN §18).

---

## 4. V1 — Helios FIFO-vs-SJF ratio (the flagship)

**Source.** Hu et al., *Characterization and Prediction of Deep Learning
Workloads in Large-Scale GPU Datacenters*, **SC '21** (arXiv:2109.01313),
Table 3. Four production clusters — Venus, Earth, Saturn, Uranus — over the
September window **2020-09-01 → 2020-09-26**, GPU jobs only.

**Setup.**

- **Per-VC replay.** The reference simulator schedules **each virtual
  cluster (VC) independently** (one worker process per VC; jobs never cross
  VC boundaries). fleetsim schedules a single global pool, so V1 is a
  *harness* (`fleetsim.validation.harness.per_vc_replay`) that runs one
  simulation per VC on a fleet sized to that VC's node count (8 GPU/node)
  and aggregates job-weighted to cluster totals — no engine change (§9).
- **SJF-oracle.** `convert_helios` writes each job's trace `duration` into
  both `duration_s` and `walltime_limit_s`, so the `sjf` scheduler orders on
  a *perfect* service-time estimate — the exact analogue of the reference
  sim keying on `duration`, and a strict upper bound on what the
  duration-predicting QSSF policy can achieve.
- **Strict (blocking) scan.** Both FIFO and SJF run with head-of-line
  blocking. This is the load-bearing fidelity choice — see §4.3.
- **September-max capacity.** Each VC is sized to its **peak** GPU quota over
  the window — see §4.4.
- **`consolidate` placement.** Sub-node jobs are packed into the fullest
  node that still has room, not the lowest-id one. fleetsim's *engine*
  default is `first_fit`; this is a stated validation-model choice on the
  same footing as the capacity snapshot — see §4.2.
- **Service time == duration.** `failure_model` off and checkpointing
  disabled for replay jobs, so a job's wall time is exactly its trace
  `duration_s` (no checkpoint amortization on top).

### 4.1 Results — the policy effect (all four clusters, real trace)

Measured on the full Helios trace, Sept-max sizing, strict scan,
`consolidate` placement. These are the exact numbers `pytest -m trace_full`
produces on a dev machine (deterministic: seed 0, a pure function of the
trace bytes). The v0.6 column (`first_fit` placement, everything else
identical) is kept so the size and direction of the placement-model effect
are visible rather than asserted.

**FIFO / SJF average-JCT ratio** — band `[1.3, 8]×`:

| Cluster | Published | fleetsim v0.7 | v0.6 (`first_fit`) | In band? |
|---|---|---|---|---|
| Saturn | 6.59× | **6.87×** | 8.75× ✗ | ✅ |
| Venus | 3.07× | **3.21×** | 4.21× | ✅ |
| Earth | 2.87× | **2.95×** | 2.11× | ✅ |
| Uranus | 1.49× | **1.51×** | 1.69× | ✅ |

**FIFO / SJF average-queuing ratio** — band `[3, 25]×`:

| Cluster | Published | fleetsim v0.7 | v0.6 (`first_fit`) | In band? |
|---|---|---|---|---|
| Saturn | 18.5× | **19.55×** | 22.91× | ✅ |
| Earth | 16.4× | **10.67×** | 5.73× | ✅ |
| Venus | 5.68× | **7.78×** | 9.97× | ✅ |
| Uranus | 4.51× | **5.08×** | 6.10× | ✅ |

**Queuing share of FIFO JCT** (`avg_queuing / avg_jct`):

| Cluster | Published | fleetsim v0.7 | v0.6 (`first_fit`) |
|---|---|---|---|
| Saturn | 89.7 % | **90.1 %** | 92.6 % |
| Venus | 81.8 % | **79.1 %** | 84.7 % |
| Earth | 69.3 % | **73.0 %** | 63.8 % |
| Uranus | 42.5 % | **42.2 %** | 48.7 % |

**What holds:**

- **Direction** (exact) — SJF beats FIFO on JCT and queuing on all four
  clusters.
- **JCT-ratio tolerance band** `[1.3, 8]×` — **all four** clusters, no
  `xfail` (v0.6: three of four). Every point value is within 5 % of
  published, which is *better than the measurement's own noise floor*
  (§4.5) and is therefore reported, not asserted.
- **Queuing-ratio tolerance band** `[3, 25]×` — all four. Still a wide
  *tolerance band*, not an exact reproduction: Earth is 10.67× against a
  published 16.4× (~35 % low). The band has teeth — an SJF-vs-SJF null
  gives ratio 1.0 and fails it.
- **Cross-cluster JCT-ratio rank** (exact, asserted) — Saturn > Venus >
  Earth > Uranus, in fleetsim *and* in the published table.
- **Queuing-share ordering** (exact, asserted) — Saturn > Venus > Earth >
  Uranus, in both.
- **Queuing-ratio cross-cluster rank** — now also matches published
  exactly (Saturn > Earth > Venus > Uranus; v0.6 had Saturn > Venus >
  Uranus > Earth, the wrong order). **Not asserted**: the Venus/Earth gap
  that decides it is only ~2× the §4.5 tie-break noise floor, so this is
  reported as a gain, not banked as an invariant.

### 4.2 The v0.6 Saturn gap: a wrong diagnosis, and the real cause

v0.6 reported Saturn's JCT ratio at **8.75×** against a published 6.59× —
its one out-of-band number — and attributed it to *"FirstFit placement
fragments Saturn's large gangs more than the reference consolidate placer."*
**That attribution was mechanically impossible**, and saying so is more
useful than silently replacing it.

**Why the v0.6 story cannot be right.** The replay fleet is *single-level*:
`harness._vc_scenario` emits `topology: {levels: ["node"], counts: [N]}`, so
`build_fleet` produces one cluster domain, one metro, and N node leaves —
every leaf a direct child of the cluster root. Replay jobs carry
`within=None` and `segments=None`, and no `penalties.xover` is configured.
So there is **no higher domain for a gang to consolidate within and no
crossing penalty to pay**: *which* free nodes a multi-node gang takes
provably cannot change any outcome. And the victims were never giants —
Saturn's largest gang is 200 GPU (25 nodes), and the single VC carrying
**84 %** of the FIFO−SJF gap tops out at 56 GPU with 0.7 % of its jobs
above one node.

**The real mechanism — sub-node stranding of whole-node capacity**, in three
measured steps:

1. **`first_fit` places sub-node gangs by ascending leaf id, with no
   preference for already-partially-used nodes.** Saturn is 70.8 %
   single-GPU (97.1 % in its dominant VC), so first-fit-by-id opens a
   *fully free* node for a 1-GPU job whenever the lower-id nodes are
   momentarily full — and immediately re-dirties any node a whole-node gang
   just released (both allocators grow from the same low-id end).
   Long-lived 1-GPU jobs then pin those nodes partial for days.
2. **A partially-used node is invisible to every whole-node request**
   (`tree.py`: a leaf with ANY owner is ineligible). That rule is a
   faithful *approximation* of the trace, not an identity: Helios's own
   `node_num` column satisfies `node_num == ceil(gpu_num/8)` for
   **99.66 %** of windowed multi-GPU jobs (Saturn 129 exceptions of 30,966
   = 0.42 %, Venus 30/10,482, Earth 44/8,361, Uranus 22/17,077). Every
   single exception runs in the *spreading* direction — **more** nodes than
   the ceiling (on Saturn: 50 jobs at `gpu_num=2` on 2 nodes, 28 at 4-on-4,
   13 at 8-on-2, 9 at 16-on-4) — i.e. Helios sometimes scattered a small job
   across hosts, which fleetsim's one-leaf sub-node model cannot express.
   **Zero** exceptions run the other way: the trace never packs a job into
   *fewer* nodes than the ceiling, which is the half the whole-node rule
   depends on. So the rule is right and the *placement* feeding it was not:
   free chips accumulate as 1–7-GPU remainders no job ≥ 8 GPU can ever use.
   (Reproduce: `gpu_num > 1` rows of each `cluster_log.csv` inside
   2020-09-01..09-26.)
3. **Under the strict scan the FIFO head is very often such a job.**
   **100 %** of measured blocked-idle chip-seconds come from heads needing
   whole nodes (8/16/32/40/56 GPU); sub-node heads contribute **0.0 %**
   because they only block when the pool is at literally zero free chips.
   During blocks the dominant VC averages **0.40 of 19** whole-free nodes
   with 25.9 free chips, and **87.7 %** of all free chips sit on
   partially-used nodes. That standing idleness sets a large backlog on a
   pool at 94.4 % offered load, and FIFO's mean JCT *is* that backlog. SJF
   is nearly immune: its head is the shortest job, almost always a 1-GPU
   job that fits in a remainder.

**The fix and its size.** `consolidate` (and `best_fit`, identical here —
§4.2.1) chooses the *tightest sufficient* node for a sub-node gang, so
best-fit never opens a fully-free node while a partial node has room. Whole
free nodes stop being consumed by mice.

| placement model | Saturn FIFO JCT | SJF JCT | JCT ratio | q ratio | #Queuing |
|---|---|---|---|---|---|
| `first_fit` (engine default, v0.6) | 75,329 | 8,613 | 8.75× | 22.91× | 74,160 |
| **`consolidate` (shipped, v0.7)** | **55,978** | 8,146 | **6.87×** | **19.55×** | 72,936 |
| fungible pool (node granularity removed) | 33,181 | 7,611 | 4.36× | 13.51× | 61,283 |
| **published (Hu et al. Table 3)** | **55,984** | 8,495 (implied) | **6.59×** | **18.5×** | **65,991** |

Node-granularity fragmentation accounts for **56.0 %** of the `first_fit`
FIFO JCT; best-fit recovers **45.9 %** of that, and the fungible-pool row
shows why recovering all of it would be *wrong* — it lands **below**
published. The reference simulator fragments too; the target was never zero
fragmentation, and best-fit happens to reproduce the residual.

> **Which of these numbers you can re-derive from this repository.** The
> `first_fit` and `consolidate` rows, and the mechanism table in §4.2.3, are
> **suite output**: the placer rows come from `per_vc_replay` (asserted in
> `test_helios_saturn_placement_model_is_load_bearing`) and the mechanism
> table from `scripts/helios_stranding_table.py`. The **fungible-pool** row
> and the blocked-idle chip-second accounting in step 3 above (100 % from
> whole-node heads, 87.7 % of free chips on partial nodes, 0.40 of 19
> whole-free nodes, 25.9 free chips, time-average queue length 2,258) are
> **one-off ad-hoc measurements** taken with throwaway instrumentation — a
> patched allocator that ignores node granularity, and a per-wake probe of
> the blocked head — neither of which is in the tree. Read them as the
> diagnostic work that found the mechanism, not as numbers the suite
> regenerates.

**One change moved several deviations at once — but they are not
independent, and two moved the wrong way.** Beyond Saturn's ratio: every
cluster's absolute FIFO JCT moved inside ±14 % (from ±35 %), Earth's
`#Queuing` deviation flipped from −27 % to +7.3 %, and the queuing-ratio
cross-cluster rank became the published one.

Resist the temptation to count that as five independent confirmations. It
is not: `avg_jct = service + avg_queuing`, the ratio is
`avg_jct(FIFO)/avg_jct(SJF)`, `#Queuing` counts jobs whose wait exceeds a
round, and q-share is `avg_queuing/avg_jct` — all functions of the same
per-cluster FIFO queue-wait distribution. Shrinking FIFO queue wait moves
them together *by construction*, so their joint movement is close to one
degree of freedom. The evidence that actually adds degrees of freedom is
(a) the **mechanism metric** (`stranded_whole_nodes` — fleet state, not a
queue statistic, §4.2.3), and (b) the **cross-cluster ordering**, which a
single scalar shift need not fix. Both reproduce. And 2 of the 8 §5
quantities got *worse* (§5), which a "one change fixed everything" framing
would hide.

**HONEST SCOPE — `consolidate` does not dominate `first_fit`.** The win is
a *cluster-level aggregate*, not a per-VC one. Of the five Saturn VCs
carrying **97 % of the FIFO−SJF gap** (the §4.2.2 criterion), `consolidate`
makes three **worse** — `vcQ4H` 79,783 → 94,548 s (**+18.5 %**, not
"slightly"), `vcBLw` 73,007 → 77,085 (+5.6 %), `vcOIr` 52,250 → 54,432
(+4.2 %) — and the cluster result is carried by `vczIT`
(137,116 → 88,293 s, −35.6 %, 41 % of Saturn's jobs). Note those five are
*not* Saturn's five largest VCs under either ordering: by nodes the top
five are `vcBLw` 34 / `vcOIr` 28 / `vcofO` 28 / `vcikv` 20 / `vczIT` 19
(no `vcQ4H`, which has 6); by job count `vczIT` 43,873 / `vcdoX` 13,383 /
`vcOIr` 9,524 / `vcQ4H` 5,160 / `vciN1` 4,982 (no `vcBLw`, which has
2,692). The claim this suite supports is *"`consolidate` reproduces the
reference placer's aggregate behavior"* — never *"best-fit is a better
scheduler."*

**Two residuals no placement policy will remove**, stated rather than
chased: `vcofO`'s offered load is **1.013** (genuinely over capacity, so its
backlog is a trace property), and `vcOIr`'s 112–200-GPU gangs (14–25 of its
28 nodes) genuinely block ~38 % of its blocked-idle chip-seconds. Together
those two VCs are 7.3 % of Saturn's gap.

#### 4.2.1 Why the policy is named `consolidate` when it is best-fit

`consolidate` and `best_fit` are **bit-for-bit identical on this fleet**, and
a test asserts it. They differ only when a whole-node gang fits in no single
parent domain, and a single-level fleet has exactly one parent domain. The
name follows the Helios reference implementation's own term for its placer;
`best_fit` is the name for the mechanism that actually matters here. The
degeneracy is documented in `Consolidate`'s docstring precisely because
mistaking the *name* for the *mechanism* is what produced the v0.6
misdiagnosis.

#### 4.2.2 Hypotheses measured and rejected

The placement conclusion is only worth as much as the alternatives ruled
out. Each of these was measured on the real trace, not argued:

- **Capacity model (Sept-max vs Sept-1) — REFUTED for Saturn.** Only 3 of
  Saturn's 23 VCs drift within September, and the five VCs carrying 97 % of
  the gap are flat all month (152/48/224/224/272 GPU). Unlike Uranus (§4.4),
  sizing is not Saturn's problem.
- **Large gangs — REAL BUT CONFINED.** Heads ≥ 112 GPU account for ~38 % of
  `vcOIr`'s and ~11 % of `vcofO`'s blocked-idle chip-seconds, but **0.3 %**
  of the dominant VC's. Those two VCs are 7.3 % of the gap.
- **Throughput — NOT the mechanism.** The dominant VC's FIFO and SJF runs
  have identical total busy chip-seconds and near-identical makespan (30.45
  vs 30.30 days). The effect is pure *ordering* on a pool at 94.4 % offered
  load; fragmentation sets the standing backlog level (time-average queue
  length 2,258 jobs under FIFO) and mean JCT measures it.
- **Tie-breaking — REAL, and it bounds every claim here.** See §4.5.

#### 4.2.3 The full placer sweep, and what the ratio band cannot decide

The placement conclusion was re-measured from scratch as its own step:
**every** placement policy × all four clusters × FIFO and SJF, on the real
trace, Sept-max sizing, strict scan — 32 cluster replays. Every figure in
§4.1 and §5 reproduced to the digit (Saturn FIFO 55,978.50 s under
`consolidate`, 75,329.05 s under `first_fit`), `n_dropped == 0` and zero
truncated jobs throughout.

| placer | Saturn | Venus | Earth | Uranus | mean abs err vs published ratios |
|---|---|---|---|---|---|
| published | 6.59× | 3.07× | 2.87× | 1.49× | — |
| `first_fit` (engine default) | 8.75× ✗ | 4.21× | 2.11× | 1.69× | 27.4 % |
| **`consolidate`** | **6.87×** | **3.21×** | **2.95×** | **1.51×** | **3.4 %** |
| `best_fit` | 6.87× | 3.21× | 2.95× | 1.51× | 3.4 % |
| `spread` (control arm) | 7.48× | 4.36× | 3.06× | 2.32× | 29.5 % |

Two results worth stating plainly:

- **`best_fit` and `consolidate` are bit-identical on the real trace**, on
  every cluster metric under both policies — not merely "expected to be" by
  the §4.2.1 argument, and not merely asserted at unit scale. The
  single-level degeneracy is a measured fact.
- **The ratio band does not select a placer.** `spread` — the deliberate
  anti-policy — puts all four clusters inside `[1.3, 8]×`, keeps
  Saturn-highest and Uranus-lowest, keeps every queuing ratio inside
  `[3, 25]×`, *and* preserves the q-share ordering. Every V1 invariant
  passes under a placer that is 45 % off Saturn's absolute FIFO JCT and
  99 % off Earth's (39,269 s vs 19,754 s).

**What does select the placer, in the order of how much weight it carries:**

1. The **mean absolute error against the four published ratios** — the
   right-hand column above: 3.4 % for `consolidate`/`best_fit`, 27.4 % for
   `first_fit`, 29.5 % for `spread`. It aggregates four clusters, so no
   single cluster's ordering noise decides it. Asserted (for the shipped
   placer) in `test_helios_four_cluster_ratio_bands_and_ranks`.
2. The **mechanism metric** (`stranded_whole_nodes`, below) — fleet state
   rather than a queue statistic, and it orders the three placers exactly as
   their FIFO JCTs do.
3. Saturn's **absolute** FIFO JCT (§5): `consolidate` −0.01 %, `first_fit`
   +35 %, `spread` +45 %. This is corroboration, **not** an independent
   discriminator: under the descending-id tie-break of §4.5 the same
   comparison inverts (`first_fit` +11.7 % vs `consolidate` +13.2 %), so
   Saturn's absolute number on its own would select the wrong placer under a
   perturbation of a knob that has nothing to do with placement. The same is
   true of Saturn's *ratio* under that perturbation (`first_fit` 7.26 vs
   `consolidate` 7.78 against a published 6.59).

**What is NOT measured, stated so nobody assumes it:** the §4.5 perturbation
was run on **Saturn FIFO only** (`first_fit` and `consolidate`, 2 replays).
Whether the four-cluster mean ratio error of item (1) would still separate
the placers under a descending-id order is **unmeasured** — extending it
means 24 more replays. So item (1) is the discriminator *within the shipped,
faithful configuration*; its robustness to intra-second reordering is an open
question, not a claim.

`test_helios_saturn_placement_model_is_load_bearing` asserts (3) under the
shipped configuration and asserts `spread` passes the band, so "in band" can
never be read as "reproduces the paper". The band is a limit on V1's
discriminating power, not on the placement finding.

**The mechanism metric, measured on the real trace — and reproducible.**
`stranded_whole_nodes` (partially-occupied HEALTHY leaves — free capacity no
whole-node gang can claim; emitted by the collector whenever a scenario
names a placement policy) was sampled at every flush of every Saturn FIFO
replay and summed across the cluster's VCs, so it reads as *expected
partially-occupied nodes cluster-wide at a random instant*. It is averaged
over a **fixed 26-day prefix** (the replay window), not over each run's own
horizon: a worse placer gets a longer adaptive horizon, and the extra
near-idle tail samples would dilute its mean and understate the effect. Both
denominators are shown for exactly that reason.

Regenerate the table below with the shipped script (needs the cached
`data.zip`; ~9 min):

```bash
.venv/bin/python scripts/helios_stranding_table.py
```

It drives `per_vc_replay(..., frag_prefix_s=26*86400)`, which is the hook
that surfaces the metric out of the harness.

| placer | Saturn FIFO JCT | partial nodes, 26-day window (of 265) | stranded chips | partial nodes, whole run |
|---|---|---|---|---|
| `consolidate` | 55,978 s | **24.99** | 98.4 | 12.04 |
| `first_fit` | 75,329 s | 27.43 | 107.2 | 12.66 |
| `spread` | 81,096 s | 41.13 | 166.6 | 19.00 |

The metric orders the three placers **exactly as their FIFO JCTs order
them** — the mechanism is visible in the fleet state, not only inferred from
JCT. But note the leverage, which is the honest caveat: an 8.9 % drop in
time-average stranded nodes buys a 25.7 % drop in FIFO mean JCT. The
time-average is therefore a **directional indicator, not a proportional
explanatory variable** — what costs the FIFO head is stranding *at the
moments it needs a whole node*, and a mean over all instants (including the
long stretches where the head is a 1-GPU job that fits anywhere) dilutes
precisely those moments. The per-moment version of this measurement is the
blocked-idle chip-second accounting in §4.2 (100 % of blocked-idle
chip-seconds come from whole-node heads; 87.7 % of free chips during blocks
sit on partial nodes).

### 4.3 Why strict (blocking) scan — a corrected plan assumption

The validation plan (§2 V1(c)) guessed the reference sim was **non-blocking**
(best-effort, small jobs flow around a stuck gang). It is not. A best-effort
scan *collapses* FIFO's queuing (the whole point of the published huge FIFO
numbers — Saturn averages 50,202 s of queuing, 65,991 jobs queued — is
head-of-line blocking of large gangs) and yields aggregate JCT ratios ~1.4×
with the **wrong** cross-cluster rank. **Strict scan is required** and flips
the result from non-reproducing to reproducing. Both policies share the scan
mode, so SJF's win is purely its shortest-first ordering clearing short jobs
before they pile up.

### 4.4 Why September-max sizing — a corrected plan assumption

The plan specified a fixed **Sept-1** per-VC capacity snapshot. That
mis-sizes clusters whose per-VC quota drifts within the month: on Uranus,
`vcUV3` grows 208 → 328 GPU and `vc7hD` spins up 0 → 416 GPU during
September, and 3,117 September jobs sit in VCs with **zero** Sept-1 quota.
Those jobs have no pool under a Sept-1 snapshot and are dropped — the
harness surfaces this as `n_dropped` / `dropped_vcs` (so the loss is
visible, not silent), and the V1 test asserts `n_dropped == 0`. A fixed
Sept-1 pool over-congests Uranus (JCT ratio 3.41×, breaking both the
Uranus-lowest rank and the q-share ordering). Sizing each VC to its
**September-max** quota recovers Uranus almost exactly (1.69× vs a published
1.49×; FIFO JCT 20,833 s vs 19,758 s) and restores both invariants.

This is a validation-harness capacity-model choice (`pool_snapshot="max"`),
not an engine fault — and it is **one modeling option among alternatives,
not unambiguously "more faithful"** than a snapshot. `max` carries its own
bias: it makes each VC's *peak* quota available from day 1, so it
over-provisions VCs early in the month and under-counts early-month
queuing. It is chosen here for two concrete reasons — it loses no windowed
jobs (`n_dropped == 0`, unlike any single-day snapshot on the drifting VCs)
and it reproduces the published cross-cluster rank — and that choice is
stated in the headline rather than buried. `convert_helios`'s default
remains the Sept-1 snapshot the plan specified.

---

### 4.5 Scheduler-order sensitivity (and why the bands stay bands)

**35.5 %** of Saturn's 105,969 windowed jobs share an *exact* submit second
(83,726 distinct timestamps, 15,404 tied-second groups; 92.9 % share a 60-s
scheduler round), so FIFO's `(submit_time, id)` tie-break decides the order
of more than a third of the queue. Flipping that tie-break from ascending to
**descending** job id and changing nothing else moves Saturn's FIFO JCT
**75,329 → 62,549 s (−17.0 %)** under `first_fit`.

**This is a sensitivity probe, not an uncertainty band** — an earlier
version of this section called ascending id an "arbitrary choice", and the
trace says otherwise:

- every windowed job id in a cluster has the **same length** (Venus 6
  characters, Earth / Saturn / Uranus 7), so fleetsim's string compare in
  `sorted(..., key=lambda j: (j.submit_time, j.id))` coincides with numeric
  order: across all tied-second groups on all four clusters there is **not
  one** group where lexicographic and numeric id order differ;
- numeric ids are **monotone in submit time** on Saturn, Venus and Earth
  (Slurm assigns ids at submission), so ascending id reconstructs the true
  FCFS order *within* a second. Uranus is the exception — one inversion.

So descending id is a deliberately **wrong** arrival order, and the −17 %
measures how much the result depends on intra-second ordering, not how
uncertain the faithful configuration is.

**What the probe does bound: Saturn's absolute number, alone, does not
select a placer.** Under the (unfaithful) descending order, Saturn FIFO JCT
is **62,549 s = +11.7 %** vs published under `first_fit` and **63,347 s =
+13.2 %** under `consolidate` — `consolidate` is *worse* on both the
absolute deviation and the ratio (7.78× vs 7.26×), and both sit inside
`[1.3, 8]` and inside the ±20 % bar V1p uses. That is why the placer
selection in §4.2.3 rests on the **four-cluster mean absolute ratio error**
(3.4 % vs 27.4 % / 29.5 %, which survives this perturbation) and on the
mechanism metric, with Saturn's absolute number as corroboration rather than
as the discriminator.

The bands stay bands for a different reason: the analysis window is
unpublished (§1) and the capacity model is a choice (§4.4), so agreement
inside ~5 % on a point value cannot be claimed as accuracy even where it is
measured. V1p's ±20 % absolute bar is derived from *those* two, plus the
Uranus id non-monotonicity — not from the tie-break number above.

## 5. V2 — Helios FIFO absolute Table 3 (secondary, distribution-match)

The same machinery, asserting the **absolute** per-cluster FIFO numbers — the
weaker "faithful replay" claim (absolute JCT is dominated by a few multi-day
TIMEOUT/CANCELLED jobs and by the tie-break noise of §4.5, so the honest bar
is "right ballpark and correct rank").

| Cluster | Published FIFO JCT (s) | fleetsim v0.7 | Δ | v0.6 Δ | Published #Queuing | fleetsim v0.7 | Δ | v0.6 Δ |
|---|---|---|---|---|---|---|---|---|
| Venus | 64,702 | 55,714 | −13.9 % | +18 % | 15,336 | 13,624 | −11.2 % | −9 % |
| Earth | 19,754 | 22,463 | +13.7 % | −15 % | 30,030 | 32,232 | **+7.3 %** | **−27 %** |
| Saturn | 55,984 | **55,978** | **−0.01 %** | +35 % | 65,991 | 72,936 | +10.5 % | +12 % |
| Uranus | 19,758 | 18,492 | −6.4 % | +5 % | 16,917 | 18,449 | +9.1 % | +10 % |

Under `consolidate` placement all four absolute FIFO JCTs land within
**±14 %** (v0.6: ±35 %, Saturn at the boundary) and all four `#Queuing`
counts within **±12 %** (v0.6: Earth was −27 %, outside the ±25 % bar).
Saturn's 0.01 % agreement is **not** claimed as precision — the
scheduler-order sensitivity of §4.5 is an order of magnitude wider than it,
so that digit is luck. What the table supports is the ±14 % band and the
fact that **6 of these 8 quantities moved toward published**.

**Three properties got worse, and they are worth naming** — the table above
already shows two of them, so counting "eight improvements" would contradict
it:

1. **Uranus's absolute FIFO JCT overshoots.** v0.6 was **+5 %** (20,833 s,
   the number §4.4 quotes); v0.7 is **−6.4 %** (18,492 s). Slightly larger
   in magnitude and on the other side — `consolidate` did not move it
   *toward* published, it moved it past.
2. **Venus's `#Queuing` drifts further out**: **−9 % → −11.2 %** (13,624
   against a published 15,336).
3. **The absolute-JCT rank.** v0.6 reproduced the published FIFO-JCT rank
   (Venus > Saturn > Uranus > Earth); v0.7 gives Saturn > Venus > Earth >
   Uranus. Both swaps are between near-ties on one side or the other:
   fleetsim's Venus (55,714) and Saturn (55,978) differ by 0.5 %, and the
   *published* Uranus (19,758) and Earth (19,754) differ by 0.02 % — a dead
   tie whose order no simulator can meaningfully reproduce.

The six that improved, as **absolute** deviations read off the table's own
columns: Venus JCT 18 → 13.9 %, Earth JCT 15 → 13.7 %, Saturn JCT 35 →
0.01 %, Earth `#Queuing` 27 → 7.3 %, Saturn `#Queuing` 12 → 10.5 %, Uranus
`#Queuing` 10 → 9.1 %.

Point 3 is why the asserted rank in V1 is the **JCT-ratio** rank (a
policy-effect quantity with real separation: 6.87 / 3.21 / 2.95 / 1.51), not
the absolute-JCT rank. The absolute rank is reported, never asserted.

---

## 6. V3 — Philly job-status split (converter fidelity)

**Source.** Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters
for DNN Training Workloads*, **USENIX ATC '19**, Table 6. The single most
directly replayable trace fact: the fraction of jobs in each terminal status,
by **count** and by **GPU-time**. This is a property of the *converted rows* —
no simulation — so it tests `convert_philly`'s fidelity, not the scheduler.

`convert_philly` maps Philly's raw states to canonical statuses; the split
helpers relabel to the paper's buckets: `Pass → COMPLETED → "Passed"`,
`Killed → CANCELED → "Killed"`, `Failed → FAILED → "Unsuccessful"`.

**Published targets** (96,260-job paper window):

- **By count**: Passed 69.3 % / Killed 13.5 % / Unsuccessful 17.2 %.
- **By GPU-time** (`num_chips × run_time`): Passed 44.53 % / Killed 37.69 % /
  Unsuccessful 17.76 % — Killed+Unsuccessful are ~30.7 % of *jobs* but
  ~55.45 % of *GPU-time*.

**CI smoke** (`test_philly_slice_status_split_smoke`, vendored ~2k-row slice):
asserts both splits sum to 1, the by-count ordering `Passed > Unsuccessful >
Killed`, and the paper's actual finding — Killed+Unsuccessful consume a
**larger GPU-time share than their headcount share**. The vendored slice is
*synthetic* (no real Philly trace is checked in), so the smoke rung asserts
only structure, never point values. A second smoke rung
(`test_convert_philly_status_mapping`) drives a hand-built raw record through
`convert_philly` to pin the `Pass/Killed/Failed → COMPLETED/CANCELED/FAILED`
mapping end to end.

**Full trace** (`test_philly_full_status_split_table6`, opt-in
`FLEETSIM_PHILLY_FULL`): converts the whole released trace and asserts by-count
within ±5 pp and by-GPU-time within ±8 pp of Table 6, plus the ordering
invariant. The tolerances are loose because the released 117,325-job /
137-day trace is **not** the paper's 96,260-job / ~75-day window (which is not
precisely published).

> **UNVERIFIED-on-real-data.** The full-Philly rung is written to spec but was
> **not** run against the real 1 GB Git-LFS trace in this build — a plain
> HTTP fetch of the LFS path yields a ~135-byte pointer, so the artifact needs
> `git lfs install && git lfs pull`. `fetch_trace("philly")` detects the
> pointer and skips with that remediation. A maintainer who fetches the trace
> should record the measured deltas here and tighten if warranted.

---

## 7. SPT — SJF optimality (analytic, always in CI)

`validation/test_sjf_ordering.py` proves the classic single-machine result
(Smith 1956) end to end through the real engine: four whole-pool jobs
released at `t=0` in longest-first order, replayed under `sjf` and `fifo`. It
asserts SJF **starts jobs shortest-first** and its **mean JCT ≤ FIFO's**
(strictly less here — the rung is not vacuous), with identical makespan (work
conserved; the win is pure reordering). This is the theoretical floor under
V1's SJF-oracle.

---

## 8. The `sjf` scheduler and SJF-oracle

V1 required a scheduler v0.5 did not have. `fleetsim.schedulers.sjf`
(`@register("sjf")`) orders pending jobs by
`(walltime_est_s, submit_time, id)` — ascending, so the shortest estimate
runs first; a `None` estimate sorts **last** (`+inf`); ties break by
`(submit_time, id)`. With `walltime_est_s == trace duration` (what
`convert_helios` writes) this is **SJF-oracle**: a perfect service-time
estimate, the reproducible upper-bound proxy for Helios's QSSF. `strict=True`
(the harness default) blocks on the shortest head-of-line job;
`strict=False` is a best-effort scan. Placement is whatever the harness
selects — `consolidate` for the shipped V1 — the **same policy FIFO uses**,
so the FIFO-vs-SJF ratio is never a placement comparison in disguise.

Note the placement fix helped FIFO far more than SJF (Saturn FIFO
75,329 → 55,978 s, −26 %; SJF 8,613 → 8,146 s, −5 %), which is itself
evidence for the §4.2 mechanism: SJF's head is almost always a 1-GPU job
that fits in a remainder, so it barely notices whole-node stranding.

---

## 9. The per-VC replay harness

fleetsim schedules a single global pool and never routes jobs to a cluster by
tenant; the Helios reference sim schedules each VC independently. So V1/V2 are
a harness, not an engine change:

```
per_vc_replay(cluster, month) :
  pools = per-VC node counts (GPUs / 8) from cluster_gpu_number.csv
  for vc in pools:                       # one independent sim per VC
      rows = convert_helios(log)[tenant == vc, submit in window, chips > 0]
      run one simulation on a 1-cluster fleet of pools[vc] nodes
  aggregate: job-weighted mean JCT / queuing over the union of VCs' jobs,
             summed #Queuing                          # -> cluster totals
```

**The three modeling choices are arguments, not defaults.** `per_vc_replay` /
`replay_canonical` default to `strict=True` (§4.3) but to
`pool_snapshot="2020-09-01"` (the plan's snapshot) and
`placement="first_fit"` (the *engine* default) — so an unqualified call
reproduces what the same call produced in v0.6, and no pre-v0.7 script's
numbers move under your feet. Every shipped rung passes
`pool_snapshot="max"` and `placement="consolidate"` **explicitly**, which is
where a reader should see them. `harness.VALIDATION_PLACEMENT` names the
validation model's placer; `harness.DEFAULT_PLACEMENT` is the function
default and is `"first_fit"`. (Passing `placement="consolidate"` to a v0.6
call changes Saturn's FIFO mean JCT from 75,329 s to 55,978 s — a 26 % move,
which is exactly why it must not be implicit.)

Per-VC `jobs.parquet` frames are concatenated into one cluster frame and the
adapters (§10) applied to the union, so the job-weighting is automatic and
exact. The harness runs event-driven (schedulers wake only on
arrivals/completions), which is **byte-identical** to periodic waking for
these work-conserving list schedulers but ~3.7× faster; the horizon is
adaptive and **verified** so every windowed job reaches a terminal status
(a truncated long-waiter would bias the mean exactly on the jobs carrying the
FIFO-vs-SJF signal). Full 4-cluster × 2-scheduler run ≈ 250 s single-process.

## 10. Metric adapters (why we read `jobs.parquet`, not `summary.json`)

fleetsim's per-job records carry everything the papers report, but the
*summary-level* aggregation differs on two axes, so a faithful replay
recomputes from the raw rows (`fleetsim.validation.adapters`):

| Quantity | Paper definition | `summary.json` | Adapter |
|---|---|---|---|
| Average JCT | mean over **all** terminal jobs of `end − submit` (failed/cancelled/timeout **included**) | filters to COMPLETED only | `jct_over_all_terminal` |
| # Queuing jobs | count of jobs that waited | not a summary field | `n_queuing_jobs(round_s=60)` — waits > one scheduler round |
| GPU-time by status | `chips × runtime`, right-censored rows excluded | — | `gpu_time_by_status` / `status_split_by_gpu_time` |

Trace jobs map to class `finetune` → tier **BATCH**, so they are *included*
in the papers' distributions (a BEST_EFFORT job would be excluded by
`metrics.summary`); the harness never routes replay jobs to best-effort.

---

## 11. Landed in v0.7 / still deferred

**Landed in v0.7:**

- **Opt-in placement policies** (`best_fit`, `consolidate`, `spread`;
  `first_fit` remains the default) → brought Saturn's JCT ratio into the
  `[1.3, 8]` band and closed four other deviations (§4.2). Selected per
  scenario with `scheduler: {params: {placement: best_fit}}`, on any
  scheduler. A `stranded_whole_nodes` fleet metric (count of
  partially-occupied nodes) is sampled at flush when a policy is named, so
  the §4.2 mechanism is directly observable in `timeseries.parquet` rather
  than inferred from JCT, and `counts.placement_policy` records which placer
  produced a run. Semantics, selection and the non-Helios measured effects:
  [placement.md](placement.md).
- **`examples/07_placement_study/`** — the same mechanism at 10 seconds
  instead of 9 minutes: 256 chips, mice + whole-node gangs, all four
  placers, with measured stranded-node / large-gang-wait / occupancy /
  goodput deltas and four counter-results (including a seed where
  `best_fit` loses and a `within: rack` variant where it loses badly). It
  exists so the placement claim is reproducible without downloading a
  trace.

**Still deferred:**

- **Fractional / sub-chip GPU allocation** → unlocks the Alibaba PAI 50 %
  GPU-sharing saving and the 113 s V100 median queueing (V5's headline).
- **Per-job delay-cause attribution** → unlocks Philly Table 2's fair-share
  vs fragmentation-delay split (V4). The `stranded_whole_nodes` metric is a
  fleet-level down payment on this, not a substitute: it says how much
  capacity is stranded, not which job was delayed by it.
- **Threading the placement policy through preemption reclaim** is *done*
  (`search_after_release(..., mode=...)`, forwarded by `tiered_priority`),
  but no shipped validation exercises a preempting scheduler with a
  non-default placement policy yet.

---

## 12. How to run

**CI / no network** — the vendored slices only (this is what CI runs):

```bash
pytest tests validation -q                      # what CI runs: 4 skipped,
                                                # everything else passes (~80 s
                                                # on a laptop; the pass count
                                                # churns with every new test)
pytest -m "not trace_full" validation/          # deselects the opt-in rungs
fleetsim validation run                          # same checks, human-readable
```

The four skips are the env-gated rungs — two Helios `trace_full`, one Philly
`trace_full`, and `tests/test_viz_data.py`'s `FLEETSIM_FRONTIER_BUDGET` gate.
Anything else skipping is a real signal.

Two independent mechanisms keep CI offline, and the distinction matters if
you are reading exit codes: each `trace_full` rung **self-skips** unless its
`FLEETSIM_*_FULL` env var is set (that is what CI relies on — three
`SKIPPED` lines, no network), *and* the `trace_full` marker lets you
deselect them outright. Neither ever downloads implicitly. The vendored
slices live in `tests/validation_traces/` — a REAL 2-VC Venus September
slice for Helios and a synthetic ~2k-row Philly slice — each carrying a
header comment with its source, license, and exact sampling command.

**Full traces** — opt-in, downloads real data:

```bash
FLEETSIM_HELIOS_FULL=1 pytest -m trace_full validation/test_helios_ratio.py
FLEETSIM_PHILLY_FULL=1 pytest -m trace_full validation/test_philly_status.py
```

The Helios `data.zip` (36 MB, a plain Git object) fetches and replays in
~4 minutes. The Philly `trace-data.tar.gz` (1 GB, **Git LFS**) needs
`git lfs install && git lfs pull` first — a plain fetch yields a pointer.

**Attribution** — every trace's required citation:

```bash
fleetsim validation cite            # all registered traces
fleetsim validation cite helios     # one trace
```

Downloads cache under `$FLEETSIM_TRACE_CACHE` or `~/.cache/fleetsim/traces/`,
and `fetch_trace` (stdlib `urllib` + `hashlib` only — no new runtime
dependency) refuses a file whose byte size (or full SHA-256, when the
registry carries one) does not match, and refuses a Git-LFS pointer — so a
truncated or wrong-size download can never silently pass a validation. The
two shipped real traces (Helios, Philly) are size-gated; SHA-256 gating
runs only for an entry carrying a full 64-hex digest.

## 13. Trace licenses & citations

Reproducing a paper's numbers from its trace carries the trace's attribution
obligation. `fleetsim validation cite` prints these; do not omit them.

| Trace | License | Citation | Source |
|---|---|---|---|
| **Helios** | CC-BY-4.0 | Hu et al., *Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters*, **SC '21**, DOI 10.1145/3458817.3476223 | github.com/S-Lab-System-Group/HeliosData |
| **Philly** | CC-BY-4.0 | Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, **USENIX ATC '19** | github.com/msr-fiddle/philly-traces |
| **Alibaba PAI** | free public use | Weng et al., *MLaaS in the Wild*, **NSDI '22** | github.com/alibaba/clusterdata |

## 14. What each validation actually proves (so we don't over-claim)

- **V1 (primary)** — fleetsim's scheduler produces the right **policy
  ordering effect**: the SJF advantage over FIFO, in the published magnitude
  band and correct cross-cluster ranking. This is the real "our numbers are
  real."
- **V2** — the **absolute** replay lands in the right ballpark (±14 % on FIFO
  JCT, ±12 % on `#Queuing`); weaker, and honest that the absolute-JCT *rank*
  is not reproduced because two of the published values are a dead tie.
- **V3** — the converter **faithfully preserves the trace's own status /
  GPU-time structure**; fidelity, not scheduling.
- **SPT** — the `sjf` scheduler realizes the textbook optimality property
  through the real engine; the floor under V1.

The bottom line: the fleetsim engine and converters faithfully reproduce the
Helios policy effect and cross-cluster structure, and as of v0.7 they do so
with **no out-of-band number and no `xfail`**. Three modeling choices carry
that result and all three are stated in the headline rather than buried: the
strict scan (§4.3), September-max per-VC sizing (§4.4), and `consolidate`
placement (§4.2). The remaining uncertainty is dominated by things no fix
addresses — an unpublished analysis window, a capacity model that is a
choice, a 17 % sensitivity to intra-second scheduler order (§4.5), one VC
genuinely over capacity, and one VC whose 200-GPU gangs really do block it.
That is why the tolerance bands stay wide even though the point values now
agree to within 5 %.
