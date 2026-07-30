# The validation suite (`fleetsim validation`, v0.6)

Every prior release proved fleetsim was **internally** consistent —
closed-form queueing rungs (M/M/c, P–K M/G/1), conservation invariants,
determinism. v0.6 asks the harder question: are the occupancy, queue-wait,
and JCT numbers **real** — do they match what a published cluster trace
actually produced? This suite downloads real traces (Helios SC '21, Philly
ATC '19, Alibaba PAI NSDI '22), replays them, and asserts the papers'
reported ratios and distributions.

```bash
fleetsim validation run                 # vendored-slice checks, no network
fleetsim validation cite helios         # the attribution the trace requires
FLEETSIM_HELIOS_FULL=1 pytest -m trace_full validation/test_helios_ratio.py
```

The headline result, stated exactly as strongly as the evidence supports:

> **fleetsim reproduces the Helios (SC '21) FIFO-vs-SJF average-JCT policy
> effect across all four clusters** (under per-VC September-max sizing,
> §4.4) — the SJF advantage, all four queuing ratios inside fleetsim's
> **[3–25]× tolerance band** (which brackets the published point ratios),
> and the **Saturn-strongest → Uranus-weakest** JCT-ratio cross-cluster
> rank — with **three of four** JCT ratios inside fleetsim's [1.3–8]×
> tolerance band. Saturn overshoots (8.75× vs a published 6.59×) because
> fleetsim's FirstFit placement fragments its large gangs more than the
> reference "consolidate" placer; that single deviation is documented, not
> hidden behind a widened tolerance.

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
| Philly Table 2 fair-share vs **fragmentation-delay** split | Needs per-job **delay-cause attribution** fleetsim does not compute. | v0.7 (cause-labeling pass). |
| Helios **QSSF** column | Needs a `jobname` column (absent from the public CSV) + a GBDT duration predictor. | Out of scope; **SJF-oracle is the reproducible upper-bound proxy** (V1). |
| Alibaba 50 % GPU-sharing saving, median 0.042 GPU/inst | Needs **fractional / sub-chip GPU** allocation; v0.6 shares only whole chips within a node. | v0.7 (fractional-GPU packing). |
| Borg absolute occupancy / JCT | Resources are normalized `[0,1]`, BigQuery-only, 8 independent 12k-machine cells. | Never a replay target; generator-distribution check only. |
| CPU contention, co-location interference | Not modeled (analytical speed model, no memory-bandwidth contention). | Out of scope. |

## 3. The validation ladder

| # | Validation | Kind | Shipped as | CI (vendored) | Full run |
|---|---|---|---|---|---|
| **V1** | Helios FIFO-vs-SJF JCT & queuing **ratio** | policy-effect | `validation/test_helios_ratio.py` | ✅ 2-VC Venus slice, direction | opt-in `FLEETSIM_HELIOS_FULL` |
| **V2** | Helios FIFO **absolute** Table 3 | distribution | reported below (harness rung) | — | opt-in (same trace) |
| **V3** | Philly job-**status** split | distribution | `validation/test_philly_status.py` | ✅ ~2k-row slice | opt-in `FLEETSIM_PHILLY_FULL` |
| **SPT** | SJF is SPT-optimal on a fungible pool | analytic | `validation/test_sjf_ordering.py` | ✅ always | n/a |
| V4 | Philly fragmentation-delay direction | policy-effect | deferred | — | v0.7 |
| V5 | Alibaba PAI run-time / gang distribution | distribution | deferred | — | v0.7 |
| V6 | Borg generator heavy-tail | generator | deferred | — | v0.7 |

The three shipped rungs (V1, V3, SPT) plus the V2 absolute reporting are the
v0.6 suite. V4–V6 wait on fractional-GPU allocation and a delay-cause pass
(see §11 and DESIGN §18).

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
- **Service time == duration.** `failure_model` off and checkpointing
  disabled for replay jobs, so a job's wall time is exactly its trace
  `duration_s` (no checkpoint amortization on top).

### 4.1 Results — the policy effect (all four clusters, real trace)

Measured on the full Helios trace, Sept-max sizing, strict scan. These are
the exact numbers `pytest -m trace_full` produces on a dev machine
(deterministic: seed 0, a pure function of the trace bytes).

**FIFO / SJF average-JCT ratio** — band `[1.3, 8]×`:

| Cluster | Published | fleetsim | In band? |
|---|---|---|---|
| Saturn | 6.59× | **8.75×** | ✗ overshoot (documented, §4.2) |
| Venus | 3.07× | **4.21×** | ✅ |
| Earth | 2.87× | **2.11×** | ✅ |
| Uranus | 1.49× | **1.69×** | ✅ |

**FIFO / SJF average-queuing ratio** — band `[3, 25]×`:

| Cluster | Published | fleetsim | In band? |
|---|---|---|---|
| Venus | 5.68× | **9.97×** | ✅ |
| Earth | 16.4× | **5.73×** | ✅ |
| Saturn | 18.5× | **22.91×** | ✅ |
| Uranus | 4.51× | **6.10×** | ✅ |

**Queuing share of FIFO JCT** (`avg_queuing / avg_jct`):

| Cluster | Published | fleetsim |
|---|---|---|
| Saturn | 89.7 % | **92.6 %** |
| Venus | 81.8 % | **84.7 %** |
| Earth | 69.3 % | **63.8 %** |
| Uranus | 42.5 % | **48.7 %** |

**What holds — the invariants that reproduce exactly, plus the tolerance
band all four land in:**

- **Direction** (exact) — SJF beats FIFO on JCT and queuing on all four
  clusters.
- **Queuing-ratio tolerance band** `[3, 25]×` — all four clusters land in
  band. This is a wide *tolerance band*, **not** an exact reproduction: the
  point values diverge substantially (Earth 5.73× vs a published 16.4×, ~65
  % low; Venus 9.97× vs 5.68×, ~75 % high) and the queuing-*ratio*
  cross-cluster rank is **not** preserved (fleetsim Saturn > Venus > Uranus
  > Earth vs published Saturn > Earth > Venus > Uranus). The band has teeth
  — an SJF-vs-SJF null gives ratio 1.0 and fails it — but only the direction,
  the JCT-ratio rank, and the q-share ordering below reproduce *exactly*.
- **JCT-ratio lower bound** (SJF advantage present, ≥ 1.3×) — all four.
- **Cross-cluster JCT-ratio rank** (exact) — Saturn > Venus > Earth >
  Uranus, in fleetsim *and* in the published table.
- **Queuing-share ordering** (exact) — Saturn > Venus > Earth > Uranus, in
  both.
- **JCT-ratio upper bound** `≤ 8×` — Venus, Earth, Uranus (3 of 4).

### 4.2 The one documented gap: Saturn

Saturn's JCT ratio lands at **8.75×**, ~9 % past the plan's 8× ceiling
(published 6.59×). This is a **modeling gap, not a bug**: fleetsim's
FirstFit placement fragments Saturn's large gangs more than the reference
"consolidate" placer, so Saturn's FIFO blocking — and thus its absolute FIFO
JCT (75,329 s vs a published 55,984 s, ~1.35×) — is over-inflated on the most
gang-heavy cluster. The V1 opt-in test asserts the `[1.3, 8]` band for
Venus/Earth/Uranus and **`xfail`s Saturn with exactly this diagnosis** — the
band is *not* widened to pass. A consolidate placer (a v0.7 fix) is expected
to bring Saturn into band.

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

## 5. V2 — Helios FIFO absolute Table 3 (secondary, distribution-match)

The same machinery, asserting the **absolute** per-cluster FIFO numbers — the
weaker "faithful replay" claim (absolute JCT is dominated by a few multi-day
TIMEOUT/CANCELLED jobs and by the consolidate-vs-firstfit fragmentation
delta, so the honest bar is "right ballpark and correct rank," ±35 % on JCT).

| Cluster | Published FIFO JCT (s) | fleetsim (s) | Δ | Published #Queuing | fleetsim | Δ |
|---|---|---|---|---|---|---|
| Venus | 64,702 | 76,394 | +18 % | 15,336 | 13,902 | −9 % |
| Earth | 19,754 | 16,769 | −15 % | 30,030 | 21,861 | **−27 %** |
| Saturn | 55,984 | 75,329 | +35 % | 65,991 | 74,160 | +12 % |
| Uranus | 19,758 | 20,833 | +5 % | 16,917 | 18,535 | +10 % |

All four absolute FIFO JCTs land within ±35 % (Saturn at the boundary, the
same fragmentation over-inflation as §4.2), and the FIFO-JCT rank
(Venus > Saturn > Uranus > Earth) is preserved. Three of four `#Queuing`
counts land within ±25 %; **Earth is ~27 % low** — just outside — because
FirstFit under-fragments Earth's small-VC mix relative to the trace. Reported
honestly rather than banded to pass.

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
`strict=False` is a best-effort scan. FirstFit placement, same as FIFO.

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

## 11. Deferred to v0.7

- **Fractional / sub-chip GPU allocation** → unlocks the Alibaba PAI 50 %
  GPU-sharing saving and the 113 s V100 median queueing (V5's headline).
- **Per-job delay-cause attribution** → unlocks Philly Table 2's fair-share
  vs fragmentation-delay split (V4).
- **A consolidate placement policy** → expected to bring Saturn's JCT ratio
  into the `[1.3, 8]` band (§4.2).

---

## 12. How to run

**CI / no network** — the vendored slices only (this is what CI runs):

```bash
pytest -m "not trace_full" validation/          # excludes the opt-in rungs
fleetsim validation run                          # same checks, human-readable
```

`pytest` **excludes** the `trace_full` rungs by default via the marker, so CI
stays fast (~seconds) and never touches the network. The vendored slices live
in `tests/validation_traces/` — a REAL 2-VC Venus September slice for Helios
and a synthetic ~2k-row Philly slice — each carrying a header comment with its
source, license, and exact sampling command.

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
- **V2** — the **absolute** replay lands in the right ballpark and rank;
  weaker (loose tolerance), honest about the fragmentation-model divergence.
- **V3** — the converter **faithfully preserves the trace's own status /
  GPU-time structure**; fidelity, not scheduling.
- **SPT** — the `sjf` scheduler realizes the textbook optimality property
  through the real engine; the floor under V1.

The bottom line: the fleetsim engine and converters faithfully reproduce the
Helios policy effect and cross-cluster structure. The residual deviations are
a **placement-model difference** (FirstFit vs consolidate) inflating the most
gang-heavy cluster past the band ceiling, and a capacity-snapshot choice for
one drifting cluster. Both are actionable in v0.7; neither is concealed by a
widened tolerance.
