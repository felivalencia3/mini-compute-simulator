# Traffic Math — the fleetsim v0.2 generative traffic model

*Authoritative spec for synthetic multi-tenant ML fleet traffic. Companion to
DESIGN.md §5/§6; supersedes the v0.1 pinned 3-step diurnal curve and
uniform-over-exponents sizing. Conventions: times are int64 microseconds; any
`_s` quantity is float seconds; config durations use unit-suffixed strings
("15m", "6h"). All randomness comes from named `RngStreams` (DESIGN §6.2): per
class, `arrivals/<class>` (gaps, thinning accepts, regime switches),
`size/<class>`, `duration/<class>`, `tenant/<class>`, `outcome/<class>`, each
with a fixed per-job draw count. No wall-clock; every run is a pure function
of `(scenario, seed)`.*

---

## 1. What real multi-tenant ML fleet traffic looks like

Twelve stylized facts the generator must reproduce (or knowingly abstract):

| # | Fact | Source |
|---|---|---|
| F1 | Poisson arrivals are valid only for human-initiated session starts in ~1 h fixed-rate windows; machine-driven arrivals are bursty at all scales (Fano ≫ 1, Hurst 0.7–0.9) | Paxson & Floyd, ToN 1995; Leland et al. |
| F2 | Submission rate is strongly diurnal/weekly: night trough, dips at 12:00 and 18:00 | Helios, SC '21 |
| F3 | Evals dominate counts, not load: ~93% of jobs, <1% of GPU-time; pretraining is 1–3% of jobs, 70–94% of GPU-time | Acme, NSDI '24 |
| F4 | Per-job resource-hours are Pareto, α ≈ 0.69–0.77: top 1% of jobs ≈ 97–99% of load | Borg TNG, EuroSys '20, Table 2 |
| F5 | Duration bodies are lognormal (Helios median 206 s, ~75% < 1000 s; Acme median 2 min), with a heavy tail carried by pretraining (weeks) | Helios; Acme |
| F6 | Sizes cluster at powers of two, small-heavy: >82% single-GPU (Philly); <10% of jobs ≥8 GPUs consume ~60% of GPU-time (Helios); TPU slices follow the ISCA '23 histogram | Philly, ATC '19; Helios; TPU v4, ISCA '23 |
| F7 | Tenant activity is Zipf: top 5% of users submit ~77% of instances (PAI) and consume 45–60% of GPU-time (Helios) | PAI, NSDI '22; Helios |
| F8 | Eval traffic is self-excited and cluster-structured: retries, iteration streaks, batches triggered by pretrain checkpoints | Acme; Feitelson, Workload Modeling ch. 8 |
| F9 | Bursty regimes exist above the diurnal cycle (crunch vs normal weeks); MMPPs match both IAT distribution and autocorrelation where renewal models fail | Li–Muskulus–Wolters, JSSPP '06 |
| F10 | Aggregate near-Poisson can hide per-source structure: Azure per-function rates span 8 decades yet each source has burstiness B ≈ −0.26 — the burst lives in the rate mixture | Shahrad et al., ATC '20 |
| F11 | Borg's four tiers: production ≈ 70% of CPU allocation / 60% of usage; batch job-count mass sits in best-effort-batch | Borg TNG |
| F12 | 30–40% of jobs end failed/killed, concentrated in large jobs (≥64-GPU <25% success), consuming ~55% of GPU-time | Philly; Acme; Helios |

---

## 2. The generative model

Each class runs its own arrival process on its own streams; classes are
superposed (heap-merged). Per-class calibration is mandatory: many sparse
bursty sources look Poisson in aggregate (Palm–Khintchine; F10) — never fit
or validate on the aggregate stream alone.

### 2.1 Arrival processes per class

| Class | Process | Why | Default parameters |
|---|---|---|---|
| PRETRAIN | homogeneous Poisson (optionally weekly-harmonic NHPP) | rare planned launches are human session starts — the one surviving Poisson case (F1) | `rate_per_week: 2`, no seasonality |
| FINETUNE | MMPP-2 × diurnal envelope | crunch/normal regimes on top of the daily cycle (F9) | `rate_ratio: 4`, `burst_frac: 0.25`, `switch_tau: 2d` (priors, fit yours) |
| EVAL | Hawkes, NHPP baseline μ(t) | self-excitation: failing evals re-run, users iterate (F8); NHPP alone undercounts queue variance | `branching: 0.4`, `kernel_tau: 15m` |
| INFER | Service replica curve (DESIGN §5), harmonic s(t) | fleet altitude schedules replicas, not requests | — |
| BEST_EFFORT (band 0; `free` is the legacy spelling) | closed-loop backlog (saturated closed source) | standing backlog makes utilization→1 experiments well-defined; BE wait is undefined under saturation | `concurrency: 4`, `think_time: fixed[0s]` |
| retry storms (optional class preset) | Hawkes, near-critical | Borel cluster sizes, mean 1/(1−n)=10 events/trigger | `branching: 0.9`, `kernel_tau: 2m` |

**NHPP with log-linear harmonic rate** (the shared diurnal machinery). The
rate of class *c* is

```
lambda_c(t) = exp( theta0 + sum_{k=1..Kd} [a_k cos(2πkt/24h) + b_k sin(2πkt/24h)]
                          + sum_{k=1..Kw} [A_k cos(2πkt/168h) + B_k sin(2πkt/168h)] )
```

The `exp` guarantees positivity; `theta0` is **derived**, not configured: it
normalizes the weekly mean of `lambda_c` to the configured `rate_*` (computed
once on a 1-minute grid, deterministic). The exact thinning bound needs no
grid: `lambda_max = exp(theta0 + Σ_k √(a_k²+b_k²) + Σ_k √(A_k²+B_k²))`.
Default daily coefficients (preset `helios_v01`) are the least-squares fit of
K=2 daily harmonics to the log of the v0.1 pinned step curve (behavioral
continuity): `daily: [[-0.205, -0.199], [-0.001, -0.221]]`. After the θ0
mean-normalization the multiplier actually applied to the configured mean
rate is: trough ≈ 0.59× at ~03:00, single peak ≈ 1.27× at ~10:15, with only
a mild local dip near 15:00 — a K=2 fit smooths away the step curve's
12:00/18:00 dips (its raw 0.50×/1.08× extrema are on the step curve's
peak-1.0 scale, before mean normalization). Weekly default is flat.

**Homogeneous Poisson (PRETRAIN).** Interarrivals iid Exp(1/λ). Valid here
and only here (F1); deterministic calendar entries remain the better model
for truly planned giant-job starts.

**MMPP-2 (FINETUNE).** Hidden 2-state CTMC, switch rates σ1 (quiet→burst),
σ2 (burst→quiet), Poisson rate λ_i in state i, all modulated by the diurnal
envelope. Config uses observables — mean rate λ̄, `burst_frac`
π_b = σ1/(σ1+σ2), correlation time `switch_tau` = 1/(σ1+σ2), `rate_ratio`
r = λ_burst/λ_quiet — from which (λ_quiet, λ_burst, σ1, σ2) solve in closed
form: λ_quiet = λ̄/(1−π_b+π_b·r). Overdispersion is then
IDC(∞) = 1 + 2·π_b(1−π_b)(λ_b−λ_q)²/(λ̄(σ1+σ2)) (Fischer &
Meier-Hellstern 1993).

**Hawkes (EVAL).** Conditional intensity with exponential kernel:

```
lambda(t) = mu(t) + sum_{t_i < t} alpha · exp(−beta·(t − t_i))
```

Config exposes the interpretable pair: `branching` n = α/β (mean children per
event, stationary iff n < 1) and `kernel_tau` = 1/β. The configured `rate_*`
is the **total** mean rate; the baseline is derived: μ(t) = (1−n)·λ_c(t) with
λ_c(t) the class NHPP curve. Cluster sizes are Borel(n), mean 1/(1−n): near-
critical n gives heavy-tailed cascades — the retry-storm preset (Laub; Bacry
et al.).

**Closed-loop backlog (BEST_EFFORT tier, band 0).** Not a point process: the source keeps
`concurrency` best-effort jobs in flight; each completion (or terminal
failure) draws a `think_time` and resubmits. Under preemptive tiers this is
the Borg-style saturated filler: prod waits are unaffected (preemptive
shielding), utilization → 1, and BE **goodput** = c(1−ρ_H)/E[S_BE]/(1+w_rework)
is the reportable metric — never BE mean wait, which is undefined under
saturation (Adan & Resing ch. 11; Grosof et al.).

### 2.2 Size distributions — weighted pow2 pmf

Sizes are discrete powers of two drawn from a per-class pmf over exponents
(`rng.choice`), then quantized per DESIGN §4.1 (sub-node pow2, whole-node
multiples above). Uniform-over-exponents (v0.1 `pow2[lo,hi]`) overweights
giants 5–10× and fabricates queueing; it remains available, not default:

| Class | pmf over chips |
|---|---|
| EVAL | `{1: .55, 2: .15, 4: .15, 8: .15}` (Philly >82% single-GPU; Helios 50–90%) |
| FINETUNE | `{8: .30, 16: .25, 32: .20, 64: .15, 128: .10}` |
| PRETRAIN | `{256: .28, 512: .22, 1024: .18, 2048: .14, 4096: .09, 8192: .06, 16384: .03}` |
| TPU clusters | ISCA '23 slice histogram preset (29% <64, 14% 64, 18% 128–192, …, 8% 2–3K) |

### 2.3 Duration distributions — lognormal body + Pareto tail splice

Bodies are lognormal, quantile-parameterized: μ = ln(median),
σ = (ln p_q − μ)/z_q with z_90 = 1.2816, z_99 = 2.3263 (`lognormal` accepts
exactly one of `p90` | `p99`). EVAL and FINETUNE get **no** Pareto tail: by
Breiman's lemma the product size×duration inherits tail index
min(α_size, α_dur), so class mix plus the pretrain tail controls the
Borg-like aggregate. PRETRAIN is a lognormal-body + Pareto-tail splice
(Cooray–Ananda 2005; Scollnik 2007):

```
f(x) = w · f_LN(x)/F_LN(theta)            x ≤ theta
     = (1−w) · alpha·theta^alpha/x^(alpha+1)   x > theta   (truncated at cap B)
```

Density continuity at θ forces `w = (α/θ) / (f_LN(θ)/F_LN(θ) + α/θ)` — w is
computed, never configured. With the splice point at the body's p90 (the
default), z = z_90 always, giving the closed form
`w = α / (α + φ(1.2816)/(0.9σ))` with φ(1.2816) = 0.17549.

Truncation caveat: that continuity holds for the UNTRUNCATED tail. The
implemented cap-truncated tail renormalizes by `q = 1 − (θ/cap)^α`, so the
density steps UP by `1/q` at θ — 1.71× at the pretrain default (θ = 30 d,
cap = 54 d). The CDF, moments and tail index are unaffected; only
density-based goodness-of-fit near θ sees the step. Users who want exact
density continuity under truncation can use the truncation-aware weight
`w_q = (α/(θq)) / (f_LN(θ)/F_LN(θ) + α/(θq))` instead.

| Class | body | tail | derived |
|---|---|---|---|
| EVAL | median 2 min, p99 1 h | none | σ = 1.462 |
| FINETUNE | median 4 h, p90 24 h | none | σ = 1.398 |
| PRETRAIN | median 12 d, p90 30 d | α = 1.5, splice at p90 (θ = 30 d), cap 54 d | σ = 0.715, w = 0.846 |

Every Pareto tail is truncated at its physical cap (`max_lifetime`, here
54 d): untruncated α ≤ 1.2 tails make sample means non-convergent
(10¹² samples for 2-digit accuracy — Crovella & Lipsky, WSC '97), while
truncation restores finite moments and matches real policy caps.

### 2.4 Tenant model — finite Zipf

Each arrival's tenant is rank i ∈ {1..U} with p_i ∝ i^(−s), sampled with one
uniform against a precomputed CDF. **Never `rng.zipf`** — that samples the
unbounded Zeta distribution, not a finite population. Calibrate by bisection
on the top-5% share: s = 1.19 reproduces PAI's top-5%-of-users → 77%-of-jobs
at U = 1300 (default `tenant_zipf_s: 1.2`); s = 0.9 matches Helios' top-5% → ~52% of
GPU-time. Per-class overrides allowed (pretrain tenants concentrate harder).
Replaces the v0.1 hardcoded s = 1.5 — a pinned behavior change.

### 2.5 Tier mix

Classes map to Borg bands (F11): PRETRAIN + INFER → PROD, FINETUNE + EVAL →
BATCH, filler → BEST_EFFORT (band 0). Arrival shares follow Acme: EVAL ~90%, FINETUNE ~8%,
PRETRAIN <2% of jobs. Validation target, not a knob: PROD holds ~60–70% of
allocated chip-hours (Borg TNG); the chip-hour tail satisfies F4 (§5.5).

---

## 3. Simulation algorithms (numpy-friendly pseudocode)

All algorithms draw only from the class's named streams, in a fixed order.

### 3.1 Lewis–Shedler thinning (NHPP; Lewis & Shedler 1979)

Vectorized over the whole horizon; no integration, no ordering tricks.

```python
def nhpp(rng, lam, lam_max, T):            # lam: vectorized rate fn, seconds
    n = rng.poisson(lam_max * T)           # candidate count at the bound
    t = np.sort(rng.uniform(0.0, T, n))    # conditional uniformity
    keep = rng.uniform(0.0, 1.0, n) < lam(t) / lam_max
    return t[keep]                         # caller converts to int µs
```

### 3.2 Ogata thinning (Hawkes; Ogata 1981, per Laub)

Sequential; state is the excitation sum A = Σ exp(−β(t−t_i)), O(1) per event
via decay. `mu_max` is the NHPP bound of §2.1.

```python
def hawkes(rng, mu, mu_max, alpha, beta, T):
    t, A, out = 0.0, 0.0, []
    while True:
        lam_bar = mu_max + alpha * A          # bound: mu ≤ mu_max, A decays
        dt = rng.exponential(1.0 / lam_bar)
        t += dt; A *= np.exp(-beta * dt)      # decay excitation to t
        if t >= T: return np.array(out)
        if rng.uniform() * lam_bar <= mu(t) + alpha * A:
            out.append(t); A += 1.0           # accepted event excites
```

The exact cluster/branching sampler (immigrants ~ NHPP(μ); each event spawns
Poisson(n) children at Exp(β) delays; iterate, sort) is the test cross-check;
Ogata is normative — streaming, single-pass, matching the pull `JobSource`.

### 3.3 MMPP-2 (regime sojourns + conditional uniformity)

Input scaling: `lam_q, lam_b` here are PEAK-scaled state rates — the §2.1
closed-form solution `(λ_q, λ_b)` multiplied by the envelope's peak factor
`λ_max/λ̄` (`DiurnalCurve.vmax`). Thinning by `envelope(t) ∈ (0, 1]` keeps
only `E[envelope] = λ̄/λ_max = exp(−Σ_k √(a_k²+b_k²) − Σ_k √(A_k²+B_k²))`
of candidates — 0.60 for `helios_v01` — so passing the §2.1 rates in
unscaled silently generates ~60% of the configured mean load, an error
large enough to flip queueing-regime conclusions. The shipped
`MMPP2Arrivals` applies this scaling internally.

```python
def mmpp2(rng, lam_q, lam_b, sig_qb, sig_bq, T, envelope):
    t, state, out = 0.0, 0, []                     # start in quiet (or draw pi)
    while t < T:
        rate, sw = (lam_q, sig_qb) if state == 0 else (lam_b, sig_bq)
        S = rng.exponential(1.0 / sw)              # sojourn in this regime
        end = min(t + S, T)
        n = rng.poisson(rate * (end - t))
        out.append(np.sort(rng.uniform(t, end, n)))
        t, state = end, 1 - state
    ts = np.concatenate(out)
    keep = rng.uniform(size=ts.size) < envelope(ts)   # diurnal thinning
    return ts[keep]           # envelope(t) = lambda_c(t)/lambda_max ∈ (0,1]
```

### 3.4 Splice sampling (fixed two-draw inverse transform)

One Bernoulli branch + one inversion per job — a **fixed** draw count,
preserving the per-stream accounting contract (no rejection loops). Φ⁻¹ is
Acklam's rational approximation (|err| < 1.2e-9), pure numpy.

```python
def splice(rng, mu, sigma, theta, alpha, w, cap):
    u1, u2 = rng.uniform(), rng.uniform()
    if u1 < w:                                   # body: truncated lognormal
        F_theta = Phi((np.log(theta) - mu) / sigma)
        return np.exp(mu + sigma * Phi_inv(u2 * F_theta))
    q = 1.0 - (theta / cap) ** alpha             # tail: truncated Pareto
    return theta * (1.0 - u2 * q) ** (-1.0 / alpha)
```

---

## 4. Fitting each component from your own trace

Per class (F10: never fit the aggregate), pure numpy:

- **NHPP harmonics**: bin submissions in 30-min windows; Poisson regression of
  counts on the harmonic design matrix (cos/sin columns) by IRLS — each step
  one weighted `np.linalg.lstsq`, <10 steps; the log-linear form makes any
  fitted coefficients positivity-safe.
- **MMPP-2**: moment matching (Heffes–Lucantoni style): estimate λ̄, the
  asymptotic Fano factor of windowed counts (window ≫ switch_tau), the ACF
  knee timescale (→ 1/(σ1+σ2)), and the burst-time fraction; invert the
  IDC(∞) formula of §2.1 for the four parameters. Rydén's EM is the heavier
  alternative if moments are noisy.
- **Hawkes**: exponential-kernel MLE in O(N) via the Ozaki recursion
  `A_i = exp(−β·(t_i − t_{i−1}))·(1 + A_{i−1})`;
  `logL = Σ log(μ + α·A_i) − μT − (α/β)·Σ (1 − exp(−β(T − t_i)))`.
  Grid-search (n, β) with μ profiled from the mean-rate identity
  μ = (1−n)·rate — no optimizer dependency, and n is directly reportable.
- **Sizes**: round observed chips to the nearest power of two; the normalized
  histogram is the pmf.
- **Durations**: body from quantiles (median + p90 or p99); tail α by the
  Hill estimator on the top k = 1–10% order statistics:
  `alpha_hat = 1 / mean(log(x_sorted[:k]) − log(x_sorted[k]))`; splice at the
  body p90; cap at your lifetime policy.
- **Tenant s**: bisect s so the finite-Zipf top-5% share matches your trace's
  (rank-frequency least squares on log-log ranks as a cross-check).
- **Priors, not fits**: the Lublin–Feitelson constants (batch interarrival
  Gamma(6.0415, 0.8531); 48 half-hour buckets from Gamma(6.1271, 5.2740),
  origin 11:00) are the canonical HPC diurnal prior — use as a cross-check
  target for the fitted NHPP, not as the generator.

---

## 5. Closed-form queueing validation targets (CI rungs)

Each rung runs on a degenerate scenario (single-chip nodes, no failures,
`checkpoint_interval: 0s`, `abort_prob: 0`) asserting the steady-state
measurement against the exact formula. Note `checkpoint_interval: 0s`
DISABLES checkpointing (no save overhead ⇒ service time == sampled
duration) — right for the FIFO rungs; the preemptive rung of §5.2 needs a
different recipe (see there). Shipped rungs (pytests in `validation/`):
M/M/c vs Erlang-C (`test_mmc.py`), §5.1 (`test_pk_mg1.py`), §5.2
(`test_priority_preemptive.py`), plus determinism and property tests.
§5.3, §5.4 and most of the §5.5 distributional self-checks are PLANNED
targets, not yet implemented (DESIGN §16.3 lists exactly what landed);
shipped §5.5 coverage today is an hourly-Fano check and sampler-level
Hill/tenant-skew spot checks in `tests/test_traffic_v02.py`.

### 5.1 Pollaczek–Khinchine M/G/1 (FIFO, lognormal service)

One 1-chip node, Poisson(λ), lognormal S, ρ = λE[S] < 1:

```
E[W] = λ·E[S²] / (2(1−ρ)),   E[S] = e^(μ+σ²/2),  E[S²] = e^(2μ+2σ²)
```

Catches duration-sampler and event-engine bias under variance. Keep σ ≤ 1 in
CI — mixing scales with c_s² = e^(σ²) − 1 (σ=1 → 1.7, σ=2 → 53.6); ~5%
tolerance at ≥10⁴ completions.

### 5.2 Preemptive-resume priority M/G/1 per-class waits (Adan & Resing ch. 11)

r tiers on one 1-chip node, Poisson λ_i, service B_i, ρ_i = λ_iE[B_i],
σ_i = ρ_1+…+ρ_i, residual E[R_j] = E[B_j²]/(2E[B_j]), class 1 highest:

```
E[W_i] = Σ_{j≤i} ρ_j·E[R_j] / [(1−σ_{i−1})(1−σ_i)]
E[T_i] = E[W_i] + E[B_i]/(1−σ_{i−1})
```

Two-class M/M/1, equal μ: `E[T_1] = (1/μ)/(1−ρ_1)`,
`E[T_2] = (1/μ)/[(1−ρ_1)(1−ρ)]`. fleetsim setup: `tiered_priority`, REQUEUE,
zero restart overhead, `checkpoint_interval: 1s`, `checkpoint_save: 0s`
(≤ 1 s work loss per preemption ≈ preemptive-RESUME; this is what
`validation/test_priority_preemptive.py` uses). `checkpoint_interval: 0s`
would DISABLE checkpointing — victims lose the whole stint
(preemptive-repeat-identical, not resume), and the Adan–Resing formula
fails for any non-exponential B (with exponential B repeat-identical is
distributionally identical to resume by memorylessness, which masks the
mistake). Also assert shielding: tier-1 statistics are independent of
lower-tier load. Tolerance ~2% (M/M/1), ~5% (M/G/1).

### 5.3 Conservation law (accounting invariant)

For non-preemptive non-anticipating disciplines on M/G/1,
`Σ_i ρ_i·E[W_i] = ρ·[Σ_j ρ_j E[R_j]]/(1−ρ)` is discipline-invariant: assert
equality (same seed) between FIFO and a non-preemptive tiered policy. Catches
wait-accounting bugs no absolute test sees.

### 5.4 M/M/c preemptive priority (Buzen & Bondi 1983)

c nodes, all classes Exp(μ): classes 1..k jointly form M/M/c at
Λ_k = Σ_{j≤k} λ_j, so per-class waits follow by Erlang-C differencing:

```
B_0 = 1;  B_k = a·B_{k−1}/(k + a·B_{k−1});  C = B_c/(1 − ρ(1−B_c))   # Erlang-B→C
W(Λ) = C(c, Λ/μ) / (c·μ − Λ)
E[W_k] = [Λ_k·W(Λ_k) − Λ_{k−1}·W(Λ_{k−1})] / λ_k
```

Exact for equal service rates; also assert the aggregate conditional-delay
tail `P(W > t) = C·e^{−(cμ−λ)t}`.

### 5.5 Generator self-checks (distributional CI)

- **Time-rescaling**: transform each class's arrivals by its fitted
  compensator Λ(t); gaps must be iid Exp(1) (numpy KS statistic).
- **IDC/Fano curve** over windows 1 min → 1 day: ≈1 flat for PRETRAIN,
  rising to 1/(1−n)² asymptotically for Hawkes (equivalently
  1 + (2n−n²)/(1−n)²; e.g. 2.78 at n = 0.4, 100 at the retry-storm
  n = 0.9 — the compound-Poisson/Borel derivation, cluster-size mean
  1/(1−n) and variance n/(1−n)³), to the §2.1 IDC(∞) for MMPP-2.
- **Heavy-tail envelope** on generated chip-hours (2M-job reference run,
  seed 42): Hill α̂ ∈ [0.6, 0.9] on the top 1%, top-1% share ∈ [92, 99.5]% —
  inside the Borg 2011–2019 envelope. The class mixture, not the within-class
  exponent, controls this (α 1.2 vs 2.0 moved the share <0.2 pt under the cap).
- **Fréchet max-in-window**: largest job's chip-hours over windows of n
  arrivals scales as n^(1/α) — assert slope 1/α on log(max) vs log(n).
- **Tenant skew**: top-5% tenant share of submissions ∈ [70, 85]% at s = 1.2.
- **Kingman telemetry** (not an assert): emit per-class, per-window c_a², c_s²
  next to occupancy for the sanity check E[W] ≈ ((c_a²+c_s²)/2)·(ρ/(1−ρ))·E[S];
  arrival upgrades move waits first-order only through c_a².

Do **not** CI-test mean waits with Pareto α ≤ 2 or lognormal σ ≥ 2 service
(E[W] infinite or astronomically slow-mixing); validate tails instead against
Pakes' asymptote `P(W>x) ~ (ρ/(1−ρ))·∫_x^∞ P(S>t)dt / E[S]` on a fixed grid.

---

## 6. Honest limitations

1. **No explicit long-range-dependence generator.** Slowly-decaying occupancy
   autocorrelation emerges from Poisson-ish arrivals × heavy-tailed durations
   (the Crovella–Bestavros/Taqqu ON/OFF mechanism, H = (3−α)/2 ≈ 0.9) —
   validated as emergent, not injected. Multifractal structure is out of scope.
2. **Tenants are marks, not agents.** Per-arrival Zipf marking reproduces
   rank skew but not per-tenant burstiness correlation or workload flurries
   (Tsafrir–Feitelson); the hierarchical mixed-Poisson / user-session models
   are deferred.
3. **No batch/gang arrival structure yet.** PAI-style recurring gang batches
   and Acme's checkpoint-triggered eval floods are approximated by Hawkes
   clustering; explicit compound-Poisson batches and engine-coupled
   Neyman–Scott floods are deferred (the latter couples arrivals to policy).
4. **Size ⊥ duration within a class.** The joint chip-hour tail is carried by
   the class mix (Breiman); within-class coupling (Lublin's size-dependent
   runtimes, Helios' size-dependent failure rates) is not modeled — outcome
   probability is size-independent in v0.2.
5. **Every run is a finite-time-scale approximation** (Crovella–Lipsky):
   single-run means are seed-variable under α ≈ 1.5 tails. Report chip-hour-
   weighted metrics over steady-state windows with multi-seed dispersion; flag
   pretrain-dominated metrics unless the window holds ≥100 pretrain arrivals
   (~1 year at 2/week).
6. **Inference is replica-granular.** Per-request burstiness (BurstGPT
   piecewise-Gamma, CV > 1) is below fleet altitude; the harmonic curve gives
   first-order diurnality; day-to-day random levels (log-Gaussian Cox) deferred.
7. **Defaults are priors.** MMPP-2 and Hawkes defaults are order-of-magnitude
   priors from grid/LLM-serving analogies, not fitted ML-fleet constants; the
   fitting recipes of §4 exist so a lab replaces them with its own trace.

## References

Paxson & Floyd, ToN 1995 · Lewis & Shedler, NRL 1979 · Ogata 1981 (per Laub,
*Hawkes Processes*, thesis) · Bacry, Mastromatteo & Muzy, arXiv 1502.04592 ·
Fischer & Meier-Hellstern, Perf. Eval. 1993 · Rydén 1996 · Li, Muskulus &
Wolters, JSSPP 2006 · Lublin & Feitelson, JPDC 2003 (m_lublin99.c) ·
Feitelson, *Workload Modeling* · Tsafrir & Feitelson, Flurries TR · Borg TNG,
EuroSys 2020 · PAI, NSDI 2022 · Helios, SC 2021 · Philly, ATC 2019 · Acme,
NSDI 2024 · TPU v4, ISCA 2023 · Shahrad et al., ATC 2020 · BurstGPT, arXiv
2401.17644 · ServeGen, arXiv 2505.09999 · Crovella & Bestavros, SIGMETRICS
1996 · Taqqu, Willinger & Sherman, CCR 1997 · Crovella & Lipsky, WSC 1997 ·
Cooray & Ananda 2005; Scollnik 2007 · Hill, Ann. Stat. 1975 · Adan & Resing,
*Queueing Theory* ch. 11 · Kingman 1961 · Buzen & Bondi, Oper. Res. 1983 ·
Grosof et al., arXiv 2010.00631 / 2109.12663 · Wang, Xie & Harchol-Balter,
SIGMETRICS 2021.
