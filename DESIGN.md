# FleetSim — Design Document

*A discrete-event simulator for ML accelerator fleets. Working name `fleetsim` (provisional; repo is currently `mini-compute-simulator`).*

*Status: v0 design, 2026-07-27. Synthesized from a survey of hyperscaler systems (Borg, TPU pods, MAST, HyperPod, Singularity, SuperPOD/KAI, Slurm, Kueue), public cluster traces (Philly, Helios, Alibaba PAI, Acme), and existing simulators (Blox, ASTRA-sim, SimAI, Vidur, Batsim). References at the end.*

---

## 1. What this is, and the gap it fills

**FleetSim simulates fleets, not jobs.** Given a fleet of heterogeneous GPU/TPU clusters described in YAML, a stochastic mix of pretraining gangs, fine-tunes, evals, and inference — plus node failures — it answers: *what occupancy, queue wait, preemption rate, and goodput does a given scheduling policy deliver?* A lab describes its fleet, plugs in a scheduler as a small Python class (or replays a public trace), and gets reproducible, seeded results.

The gap is real. Surveying the field:

| Tool | Altitude | What it lacks for fleet questions |
|---|---|---|
| Blox, Gavel, Pollux, Sia simulators | single cluster | flat GPU counts — no topology, no failures, no inference, no multi-DC |
| ASTRA-sim, SimAI | one job × network | simulates one job's iteration time; doesn't know what a queue is |
| Vidur, LLMServingSim | inference serving | inference-only, single service |
| Batsim, Slurm simulator | HPC batch | no ML gang/slice/preemption-cost semantics, no accelerator model |
| AIReSim | reliability | single job, no scheduling |
| kube-scheduler-simulator | k8s placement | wall-clock, no time model, no JCT |
| Meta MAST (OSDI '24) | **fleet** — the existence proof | real system; **no public simulator** |

No open-source tool combines *hierarchical multi-cluster fleets + heterogeneous chips + gang scheduling + preemption cost + failures + mixed training/inference traffic*. That combination is the product.

**Three phenomena the MVP must reproduce** (they dominate every trace we surveyed; missing any one makes answers wrong):

1. **Gang-scheduling fragmentation** — fragmentation, not fair-share, causes 59–98% of multi-GPU queueing delay (Philly, ATC '19).
2. **The mice/hogs mix** — evals are ~93% of jobs but <1% of GPU-time; pretraining is 1–3% of jobs but 70–94% of GPU-time (Acme, NSDI '24); top 1% of jobs ≈ 99% of resource-hours, Pareto α≈0.7 (Borg TNG, EuroSys '20).
3. **Goodput ≠ occupancy** — 16K-GPU Llama-3 training saw one unplanned interruption every ~3.1 hours; a healthy fleet targets ETTR > 0.9 (Meta). Clusters routinely show ~100% allocation with ~52% hardware utilization (Philly).

---

## 2. Design principles

1. **The tree is the map; feasibility lives in the tree, speed lives in a pluggable cost model.** Every production scheduler expresses placement as "fit within one domain at level L" (Slurm lowest-common-switch, Kueue topology levels, GKE slice pools, EC2 topology API). We model capacity as a tree of domains with config-defined level names (Kueue proves levels must come from config, not code). Per-job performance is an analytical multiplier (Gavel-style), never packets.
2. **Gang is the core primitive.** A training job's allocation is atomic (all-or-nothing), immutable once granted, and the unit of failure — OpenAI, MAST, TPU slices, and Volcano/KAI PodGroups all agree. Partial allocation of a training job is unrepresentable.
3. **Event-driven engine, round-driven policies.** Durations span six orders of magnitude (2-min evals to 54-day pretrains) — fixed ticks are either wasteful or inaccurate. But real DL schedulers act in rounds (Pollux: 60 s), so the scheduler is invoked via a coalesced wake event, giving Blox/Gavel-compatible semantics on an event core.
4. **Determinism is a contract.** Every run is a pure function of `(fleet config, workload config | trace, seed)`. Named independent RNG streams mean enabling failures never perturbs arrivals — A/B scheduler comparisons are paired experiments, not noise.
5. **Schema-now, semantics-later.** v1 simplifies aggressively (no torus shape-fitting, no elasticity, no quota) but the schema carries the fields (`shape`, `geometry`, `CapacityClass`, `gangs: list`) so later versions slot in without migration. `fleetsim validate` rejects configs using not-yet-implemented features — no silent no-ops.
6. **The engine mutates state; policies emit intents.** Schedulers are pure-ish functions from an immutable view to a list of actions, validated and applied by the engine (Batsim's separation, in-process for speed).

---

## 3. Fleet data model

### 3.1 Chip types

A flat registry. Jobs constrain on it; the (v0.4) performance model keys its throughput matrix on `(job_profile, chip_type)`.

```python
@dataclass(frozen=True)
class ChipType:
    name: str              # "h100", "gb200", "tpu_v5p", "trn2"
    vendor: str
    hbm_gib: float
    peak_tflops_bf16: float
    generation: int        # for affinity/compat rules
```

### 3.2 The domain tree

One class covers metro → node. Leaves carry chips and health state; interior domains carry incrementally-maintained capacity counters (which makes fragmentation queries O(1) per domain).

```python
class NodeState(Enum):
    HEALTHY = auto(); DRAINING = auto(); FAILED = auto(); MAINTENANCE = auto()

@dataclass
class Domain:
    id: str
    level: str                      # from the cluster's declared level list
    parent: str | None
    children: list[str]             # empty for leaves
    chip_type: str | None           # set on homogeneous subtrees
    chips: int = 0                  # leaves only
    state: NodeState = NodeState.HEALTHY   # leaves only
    lemon_factor: float = 1.0       # failure-rate multiplier (Meta "lemon nodes")
    total_chips: int = 0            # derived, maintained on alloc/free
    free_chips: int = 0             # derived
    attrs: dict[str, Any] = field(default_factory=dict)
    # attrs examples:
    #  {"pooled": true}                        -> scale-up domain (NVLink node/NVL72/UltraServer)
    #  {"geometry": [4,4,4], "ocs_pool": true} -> TPU cube under an OCS pod
    #  {"xover_bw_gbps": 400, "oversub": 7}    -> cost-model inputs (Llama-3 used 1:7)
```

**What's unified vs special-cased.** Unified: the tree, integer capacity, gang ownership, per-level constraints. Special-cased via `attrs`, interpreted by small registered predicates:

- **`pooled: true`** marks the *scale-up boundary* (8-GPU NVSwitch node, GB200 NVL72 rack, Trn2 UltraServer). The NVL72 bandwidth cliff (1.8 TB/s per GPU in-rack → ~50 GB/s per node across racks) makes this the hard placement boundary; jobs express it with an ordinary `within` constraint at that level.
- **`ocs_pool: true`** on a TPU pod (v0.3): a request of *c* ≥ 64 chips is feasible iff `free_healthy_cubes ≥ c/64` **anywhere under the pod** — the ISCA 2023 result that optical circuit switches remove contiguity constraints for cube-multiple slices. Sub-64-chip slices must fit inside one cube (contiguity *does* matter there). The requested shape `(a,b,c)` and `twisted` flag are stored on the allocation and consumed only by the cost model (twisting = 1.31–1.63× all-to-all, not a feasibility change).

The v1 core never branches on vendor: the scheduler sees domains, capacities, constraints. TPU-ness is two feasibility predicates registered per pod, implemented in v0.3 but present in the schema from day one.

### 3.3 Fleet YAML

Templates with `count` avoid enumerating thousands of nodes; each cluster declares its own level vocabulary.

```yaml
chip_types:
  h100:    {vendor: nvidia, hbm_gib: 80,  peak_tflops_bf16: 989}
  tpu_v5p: {vendor: google, hbm_gib: 95,  peak_tflops_bf16: 459}

templates:
  h100_node: {level: node, chips: 8, chip_type: h100, attrs: {pooled: true}}
  h100_su:   {level: su,  attrs: {rails: 8},
              children: {template: h100_node, count: 32}}    # 256 GPUs (SuperPOD RA)
  h100_pod:  {level: pod, attrs: {oversub: 7, xover_bw_gbps: 400},
              children: {template: h100_su, count: 16}}      # 4,096 GPUs
  v5p_host:  {level: host, chips: 4, chip_type: tpu_v5p}
  v5p_cube:  {level: cube, attrs: {geometry: [4,4,4]},
              children: {template: v5p_host, count: 16}}     # 64 chips
  v5p_pod:   {level: pod, attrs: {ocs_pool: true, dcn_gbps_per_host: 50},
              children: {template: v5p_cube, count: 140}}    # 8,960 chips (v5p)

fleet:
  - metro: us-east
    datacenters:
      - id: dc1
        clusters:
          - id: hopper-a
            levels: [cluster, pod, su, node]
            children: [{template: h100_pod, count: 4}]
          - id: tpu-a
            levels: [cluster, pod, cube, host]
            children: [{template: v5p_pod, count: 2}]

failure_model:
  node_mtbf_days: 42                 # Meta RSC measured
  repair:  {auto_min: [60, 180], manual_frac: 0.1, manual_days: [1, 3]}   # node downtime
  maintenance_rate_per_node_month: 1.0    # drains, not crashes (Borg / Llama-3)
```

Real-fleet quanta for reference configs: H100 SuperPOD SU = 32 nodes × 8 = 256 GPUs, pods 1K–16K; GB200 = NVL72 racks of 72 GPUs, SU = 8 racks = 576; Llama-3's cluster: 16-GPU racks, 3,072-GPU full-bisection pods, 8 pods at 1:7 oversubscription; TPU v4 pod = 4,096 chips (64 cubes of 64), v5p = 8,960, max schedulable slice 6,144; Trn2 UltraServer = 64 chips.

---

## 4. Allocation model

### 4.1 Granularity — the hardest call

**Chip-granular within a single leaf; whole-node multiples above.** ~90% of arriving jobs are 1–8-chip evals (Acme). If the node were the only atom, a 1-chip eval would strand 7 of 8 GPUs and corrupt occupancy, JCT, and the mice/hogs story; rounding evals up to full nodes falsifies the workload. So:

- A job smaller than one node requests `chips ≤ node_size` and shares a leaf: the leaf's `owners` is a small dict `{alloc_id: chips}`.
- A job of node size or larger requests whole nodes (multiples of `node_size`), matching OpenAI/GKE/SuperPOD practice.

This is the only scheme under which evals, occupancy, and Philly replay are simultaneously honest. (Fractional chips — PAI's 0.25/0.5 GPU shares — remain v2+.)

### 4.2 Constraints and allocations

Constraints are expressed against the tree, copied from Kueue/Slurm:

```python
@dataclass
class Constraint:
    level: str                    # "must fit within one <level>"
    required: bool = True         # preferred (relaxable) arrives in v0.3
    relax_after_s: float = 300.0  # Slurm max_switch_wait default; v0.3

@dataclass
class GangSpec:
    chips: int                        # generator emits log2/×node quantized
    chip_type: str | None             # v1: must be pinned (see §10)
    within: Constraint | None         # hard `within` IS in v1
    segments: tuple[int, str] | None  # (nodes_per_segment, level) — Slurm block, v0.3
    shape: tuple[int, int, int] | None  # TPU slice request; v1 validates chips only
    twisted: bool = False

@dataclass
class GangAlloc:
    nodes: list[str]              # leaf ids (or {leaf: chips} for sub-node)
    anchor: str                   # LCA domain that satisfied `within`
    relaxed: bool                 # v0.3: constraint relaxed -> cost-model penalty
    attrs: dict[str, Any]         # realized shape, twisted, cubes used

@dataclass
class Allocation:
    job_id: str
    gangs: list[GangAlloc]        # >1 = Multislice-style gang-of-gangs (v0.3)
```

**Semantics baked into the core** (not pluggable — every surveyed system agrees):

- **Atomic**: all gangs place or none (TPU queued resources, k8s Coscheduling).
- **Immutable** once granted (GKE slice node pools). Elasticity is a v2 `Resize` action.
- **Failure unit**: any member node failing kills the whole gang; for multi-gang jobs, one gang's death restarts *all* gangs (Multislice) — reproducing failure amplification for giant jobs.
- `anchor` gives the placement-quality score (LCA depth) and the explainability audit trail.

**v1 ships hard `within` constraints.** This is load-bearing: with whole-node atoms and no locality constraint, any k free nodes anywhere satisfy any gang and node-level fragmentation is *structurally impossible* — the simulator couldn't reproduce Philly's headline result. Relax-after-timeout and the placement-quality penalty are a matched pair deferred together to v0.3 (relaxing without a penalty is strictly dominant, so the timeout would only add fake queueing).

---

## 5. Job and workload model

One `Job` type for everything schedulable. Inference *services* sit above the scheduler as replica-job generators — Borg and MAST schedule serving as ordinary prod-tier jobs, and a single scheduler-facing type means every pluggable policy handles the full mix for free.

```python
class JobClass(Enum):     PRETRAIN, FINETUNE, EVAL, INFER_REPLICA
class Tier(IntEnum):      FREE = 0; BATCH = 1; PROD = 2; MONITORING = 3   # Borg bands
class CapacityClass(Enum): RESERVED, ON_DEMAND, SPOT, FLEX_START, CALENDAR  # v1: ON_DEMAND only
class JobStatus(Enum):    # Helios terminal states + queue states
    PENDING, ADMITTED, RUNNING, COMPLETED, FAILED, CANCELED, TIMEOUT, NODE_FAIL, PREEMPTED

@dataclass
class Job:
    id: str; tenant: str; job_class: JobClass
    submit_t: int                         # µs
    gangs: list[GangSpec]                 # v1 generator emits length-1
    tier: Tier
    capacity: CapacityClass = CapacityClass.ON_DEMAND
    preemptible: bool = True              # decoupled from tier (KAI does this explicitly)
    min_runtime_s: float = 0.0            # per-class default: pretrain 7200 (Meta 2h), eval 0
    max_lifetime_s: float | None = None   # Meta: 7 days; enforced via JOB_TIMEOUT event
    walltime_est_s: float | None = None   # scheduler-visible; unlocks backfill (v0.2)
    true_duration_s: float = 0.0          # hidden from scheduler
    checkpoint_interval_s: float = 3600.0
    checkpoint_save_s: float = 60.0       # save overhead; goodput subtracts it
    restart_overhead_s: float = 900.0     # job resume cost (5–20 min, Meta; ≠ node repair)
    valid_until: int | None = None        # queued-resource expiry -> FAILED (schema now)
    service_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    attained_service_chip_s: float = 0.0  # for LAS/Tiresias policies
    goodput_chip_s: float = 0.0           # forward progress only

@dataclass
class Service:                            # inference: emits/cancels INFER_REPLICA jobs
    id: str; tenant: str
    replica_spec: GangSpec                # typically one pooled node
    min_replicas: int; max_replicas: int  # v1: min == max (frozen); diurnal resize v0.2
    load: "DiurnalCurve"                  # target QPS -> desired replicas (v0.2)
    tier: Tier = Tier.PROD
```

**Tier semantics are band rules, not integers** (Borg's exact semantics): higher band preempts lower; **no preemption within PROD** (prevents cascades); preempted work requeues at original priority, never migrates. Preemption modes are `CANCEL` and `REQUEUE` only — `SUSPEND` is undefined on GPUs (held memory) and is cut. A job preempted or failed before its first checkpoint loses *all* progress.

**Quota/admission is a separate pipeline stage before the queue** (Borg's quota-vs-priority split). v1 ships it as a no-op pass-through — costs nothing, avoids an API break when real quota (MAST in-quota/over-quota, HyperPod lend/borrow) lands in v0.2.

**`CapacityClass` carries cloud semantics verbatim** (v0.2+): SPOT = zero-notice kill + checkpoint restart; FLEX_START = queued start, 7-day cap; RESERVED/CALENDAR = Capacity-Block-style windows whose `hard_end` creates capacity cliffs (eviction semantics to be specified with the feature, not left implicit).

### 5.1 Synthetic traffic defaults (per class, trace-derived)

| Class | Share of arrivals | Size | Duration | Priority |
|---|---|---|---|---|
| EVAL | ~90% | pow2 1–8 chips (sub-node) | lognormal, median ~2 min, p99 ~1 h | BATCH, `min_runtime=0` |
| FINETUNE | ~8% | pow2 8–128 chips | lognormal, median ~3–4 h, p99 ~48 h | BATCH |
| PRETRAIN | <2% | pow2/shape-quantized 256–16K | lognormal-body, Pareto tail: median ~10–14 d, tail to 54 d | PROD, `min_runtime=2h`, ckpt 1 h |
| INFER_REPLICA | via Services | 1 pooled node | open-ended | PROD |

- **Arrivals**: diurnally-modulated Poisson per class (thinning against a rate curve — night trough, 12pm/6pm dips per Helios), per-tenant skew (top 5% of users submit ~77% of jobs, PAI). Eval bursts optionally *triggered* by pretraining checkpoint events (Acme).
- **Sizes**: log2-quantized — every trace clusters at powers of two; TPU shapes are the analogous quantization. The sampler is validated so top 1% of jobs ≈ 99% of chip-hours (Pareto α≈0.7). Uniform or exponential sizing produces fictional queueing and is explicitly wrong.
- **Outcomes**: 30–40% of jobs end killed/failed (Philly 30.7%, Acme ~40%), skewed early in job life.
- TPU-cluster pretraining sizes default to the ISCA 2023 slice histogram (29% <64, 14% 64, 18% 128–192, …, 8% 2–3K chips).

---

## 6. Simulation engine

### 6.1 Events

Time is **int64 microseconds** since sim epoch — integer time makes tie-breaking and cross-platform determinism exact; 10¹³ µs (months) fits trivially.

```python
class EventType(IntEnum):        # value = same-timestamp ordering rank
    NODE_REPAIR = 0
    JOB_COMPLETION = 1
    NODE_FAILURE = 2
    PREEMPTION_DONE = 3          # grace/checkpoint window elapsed
    JOB_ARRIVAL = 4
    MAINTENANCE_DRAIN = 5        # planned; distinct from failure
    JOB_TIMEOUT = 6              # max_lifetime / valid_until enforcement
    SCHED_WAKE = 7               # runs after all state changes at t
    METRICS_FLUSH = 8

@dataclass(frozen=True, slots=True)
class Event:
    time: int; type: EventType; seq: int; payload: object
```

- Queue: `heapq` keyed `(time, type, seq)`. The type rank guarantees all completions/failures at time *t* land **before** `SCHED_WAKE`, so the scheduler always sees settled state; `seq` makes ordering total.
- **Lazy completions**: on start, schedule completion at `start + remaining_work / speed(placement)`; preemption/failure *tombstones* that event (a `cancelled` seq-set skipped on pop) and reschedules. `speed(placement) ≡ 1` in v1 — the hook exists, the penalty lands in v0.3.
- **Coalesced wakes**: arrivals/completions/failures set a dirty flag and ensure one pending `SCHED_WAKE` (next round boundary, default 60 s). The scheduler is never called once per event — essential at Borg-like arrival rates.

### 6.2 Determinism

NumPy `SeedSequence.spawn()` derives **named independent streams**: `arrivals`, `job_size`, `job_duration`, `failures`, `maintenance`, plus per-entity streams keyed by stable strings. Enabling failure injection cannot perturb the arrival sequence. Identical `(scenario, seed)` → byte-identical Parquet; CI asserts output hashes on golden scenarios.

### 6.3 Performance envelope

Target: **6 simulated months of a 100K-GPU fleet in ≤ minutes of pure Python.** Event math: Meta-RSC-scale traffic (~7K jobs/day) → ~1.3M jobs → ~5M events; failures at 42-day MTBF over 12.5K nodes → ~50K events. At ~1–2 µs/heap-op, 5–10M events ≈ 30–120 s. Requirements: no per-chip events (chips are counters); failure sampling by aggregate rate (`Exp(n_healthy/MTBF)`, then pick victim) not per-node timers; incrementally-maintained per-domain free counters; O(1) metric accumulators; `slots=True` dataclasses. Python is right: the ecosystem's policies are Python (Blox reproduced seven schedulers in 12–1157 LoC each) and the bottleneck is heap/dict ops, not numerics. If a lab needs 10×, the core loop is small enough to port to Rust behind the same API — not a v1 cost.

---

## 7. Scheduler API

The engine validates and applies actions; illegal intents raise in strict mode.

```python
class ClusterView(Protocol):
    now: int
    def pending(self) -> Sequence[JobView]        # queued, admission-passed
    def running(self) -> Sequence[JobView]        # incl. attained service, checkpoint age
    def free_capacity(self, domain: DomainId) -> int
    def domains(self, level: str) -> Sequence[DomainView]
    def find_placement(self, job: JobView, policy: PlacementPolicy) -> Placement | None
    def throughput(self, job: JobView, chip_type: str) -> float   # Gavel-matrix hook (v0.4)

@dataclass(frozen=True)
class Place:   job_id: JobId; placement: Placement   # complete gang node-set; atomic
@dataclass(frozen=True)
class Preempt: job_id: JobId; mode: PreemptMode      # CANCEL | REQUEUE
Action = Place | Preempt                              # Reserve joins in v0.2 (backfill)

class Scheduler(ABC):
    wake_interval: int | None = 60_000_000            # µs; None = event-triggered only
    @abstractmethod
    def schedule(self, view: ClusterView) -> list[Action]: ...
```

**Ordering and placement are separate axes** (Blox's decomposition): a `Scheduler` composes an *ordering* policy (who next: FIFO, LAS/Gittins, SJF-predicted, fair-share) with a `PlacementPolicy` (where: first-fit, bin-pack, spread, topology-aware). `view.find_placement()` runs the search inside the engine's indexes so ordering policies never touch node internals.

FIFO, complete:

```python
class FIFOScheduler(Scheduler):
    def __init__(self, placement=FirstFit(), strict=True):
        self.placement, self.strict = placement, strict

    def schedule(self, view):
        actions = []
        for job in sorted(view.pending(), key=lambda j: (j.submit_time, j.job_id)):
            p = view.find_placement(job, self.placement)
            if p is not None:
                actions.append(Place(job.job_id, p))
            elif self.strict:
                break        # StrictFIFO (Kueue): head-of-line blocks
            # else BestEffortFIFO: skip and continue
        return actions
```

**Growth path, without breaking the contract:**

- **Tiered priority + preemption** (v0.1's second policy): emit `Preempt(..., REQUEUE)` against lower-*band* victims; the engine applies grace time, checkpoint-loss accounting, the no-preemption-within-PROD guardrail, and per-class `min_runtime` protection — every policy inherits correct preemption cost.
- **Backfill (EASY, v0.2)**: adds `Reserve(job_id, placement, start_at)`; requires walltime estimates, which are in the job schema from day one.
- **Topology-aware (v0.3)**: purely a `PlacementPolicy` (score by LCA depth; pack small jobs to domain edges per NVIDIA's NVL72 block scheduler). Ordering code untouched.
- **Hierarchical/MAST-style (v0.4)**: `Scheduler` is instantiable per tier — a global policy scores regions and delegates to per-cluster schedulers on scoped views.
- **Elastic jobs (v2)**: `Resize(job_id, n)` joins the Action union — a pure extension.
- **Language-agnostic / RL (v0.5)**: Batsim-style JSON-over-socket mode of the same protocol; a Gym wrapper falls out (batsim-py precedent).

The scheduler-visible state deliberately includes the two inputs the canonical policy suite needs: per-job **attained service** (Tiresias/LAS) and the **throughput hook** (Gavel heterogeneity).

---

## 8. Failures, maintenance, recovery

Failures are events with a lifecycle, not noise:

- **Unplanned node failure**: per-node exponential MTBF, default **42 node-days** (Meta RSC). Sampled by aggregate rate. Job MTTF then scales inversely with size automatically — reproducing Meta's 7.9 h at 1024 GPUs and Llama-3's ~3.1 h at 16K.
- **Consequence**: the resident gang(s) die (slice-granularity, GKE/Multislice semantics); the job loses work back to its last checkpoint, pays `restart_overhead_s` (default 15 min), and re-enters the queue at original priority. Second-order preemptions (a restarted big job preempting small jobs) emerge naturally and are tagged in metrics — they cost Meta 16% of failure-induced goodput loss.
- **Node repair** (distinct from job restart): auto-repair 60–180 min for ~90% of failures, manual 1–3 days for the rest (AIReSim defaults). Node returns HEALTHY.
- **Planned maintenance**: a separate stream of *drains* (~1/node-month, Borg): node enters DRAINING — blocks new placements; resident gangs get a grace window (default 1 h) to checkpoint, then are preempted with REQUEUE semantics; node goes to MAINTENANCE for the repair duration, then HEALTHY.
- **Lemon nodes**: configurable fraction with an MTBF multiplier (detecting them changed Meta's large-job failure rate 14%→4% — a phenomenon worth simulating).
- Default failure-cause mix for reporting: ~60% GPU/HBM, ~10% network, ~13% software (Llama-3).

---

## 9. Metrics

Two mechanisms: **time-weighted integrals** for state metrics (on every state change, `acc += value × Δt` — O(1), exact) and **event-sourced records** for discrete outcomes.

The utilization split is non-negotiable — conflating these is the classic trace-paper error:

| Metric | Definition |
|---|---|
| **Allocation rate** | allocated / total chips (incl. failed/drained) |
| **Occupancy** | ∫ allocated dt / ∫ *healthy* dt — gap vs allocation rate = capacity lost to failures |
| **Goodput** | ∫ productive chip-time / ∫ allocated chip-time; productive excludes lost-since-checkpoint work, restart latency, and checkpoint-save overhead |
| **ETTR** | per-job productive/elapsed; fleet-healthy benchmark > 0.9 (Meta) |
| Queue wait | first_start − submit; **per class**, P50/P90/P99, bucketed by chip count |
| JCT | end − submit, completed jobs; count-weighted **and** chip-hour-weighted |
| Preemptions/min | split by trigger: priority, reclaim, failure-second-order |
| **Fragmentation** | `largest_placeable(level)` time series; frag index = 1 − largest_placeable/free; stranded capacity (free chips below the smallest gang quantum) |
| Replica availability | for inference services: replica-minutes running / desired — makes inference harm visible |
| Per-tenant share | chip-hours and queue-wait by tenant (Jain fairness index v0.2) |

Every distributional metric is reported **both job-weighted and chip-hour-weighted** (mice vs hogs). The metrics layer computes over a configurable **steady-state window** (exclude warmup/drain, Blox's jobs-3000–4000 convention) by default.

**Outputs**: `summary.json`, `jobs.parquet` (per-job records), `timeseries.parquet`, optional chrome://tracing timeline (one track per domain — Vidur's best debugging affordance), optional per-decision logs recording which policy/score chose each placement (kube-scheduler-simulator's explainability idea).

---

## 10. Trace replay

A `TraceSource` implements the same iterator interface as the generator, so schedulers can't tell replay from synthesis. Canonical job schema = union of the public traces: `job_id, user, tenant/vc, class, submit_time, num_chips, chip_type, num_nodes, duration, walltime_limit, final_status ∈ {COMPLETED, FAILED, CANCELED, TIMEOUT, NODE_FAIL}` (Helios's enum). Adapters: **Philly** in v0.1 (canonical replay workload; per-attempt records → retries), **Helios** in v0.2 (with the QSSF-beats-FIFO reproduction as a CI test), **PAI** when fractional GPUs exist (its `plan_gpu` fractions are unrepresentable before then). Trace `final_status` can be replayed verbatim or overridden by our failure model.

---

## 11. v1 scope

### IN (v0.1, demo-able)

- One metro, N clusters, config-defined level trees, heterogeneous chip types across clusters; **chips pinned per job** (unpinned chip_type + throughput matrix arrive together in v0.4 — without the matrix, an H100 and an A100 cluster would behave identically, which is fake heterogeneity).
- Gang allocation: chip-granular sub-node + whole-node multiples; atomic, immutable; **hard `within` constraints**.
- Synthetic per-class generator (defaults of §5.1) + **Philly replay**.
- Schedulers: **FIFO** (strict + best-effort) and **tiered_priority** (Borg bands, REQUEUE preemption, per-class min-runtime guard). Two policies is exactly enough to demo "the simulator changes a decision."
- Failures + repair + maintenance drains + checkpoint/restart accounting (§8). This is ~200 LoC and the single biggest differentiator vs Blox/Gavel/Pollux — cutting it would cut the moat.
- Full metrics table of §9, steady-state windowing, Parquet + summary + plots.
- No-op admission stage (seam for v0.2 quota).

### OUT (explicitly; schema hooks present)

TPU predicate math & multislice (v0.3) · relax-after-timeout + placement-speed penalty (v0.3, matched pair) · backfill/`Reserve` (v0.2) · quota & capacity classes beyond ON_DEMAND (v0.2) · autoscaling inference (v0.2; v1 freezes replica counts) · throughput matrix / unpinned chips (v0.4) · multi-metro two-stage scheduling (v0.4) · elasticity, fractional GPUs, network contention, RL interface (v0.5+).

### Roadmap

1. **v0.2 — Queue discipline & economics**: EASY backfill, tenant quota (in-quota/over-quota, MAST/HyperPod lend-borrow), capacity classes {reserved, on_demand, spot} with grace windows and specified hard-end eviction, inference autoscaling + unserved-demand metric, Helios adapter + QSSF CI test.
2. **v0.3 — Topology & TPU**: relaxable constraints + placement-quality penalty (together), Slurm-block segments, TPU `ocs_pool` predicates + shape validation, multi-gang Multislice with failure amplification. Validation target: NVIDIA's "within ~1% of theoretical occupancy" NVL72 block-packing result.
3. **v0.4 — Fleet altitude**: multi-metro, two-stage global/regional scheduler API (MAST), Gavel throughput matrix + unpinned chip types.
4. **v0.5 — Ecosystem surface**: JSON-over-socket scheduler protocol, Gym wrapper, chrome://tracing export.
5. **v1.0 — Trust release**: full validation ladder in CI, frozen plugin API, docs site, 3+ external scheduler examples.

---

## 12. Validation ladder (each rung a CI test)

1. **Analytical**: M/M/c scenarios match closed-form waits within 1%.
2. **Invariants** (property tests): no chip double-allocated; gangs atomic; causality; chip-hour conservation.
3. **Trace replay**: Philly under FIFO reproduces ~31% killed/failed consuming ~55% of GPU-time, and fragmentation dominating multi-GPU queueing delay.
4. **Cross-simulator** (v0.3, once placement penalties exist): Pollux's standard workload (160 Philly-sampled jobs, 20 jobs/hr, 64 GPUs) under FIFO matches Blox's published JCT percentiles within ~5%.
5. **Policy-effect reproduction** (v0.2): Helios's QSSF beats FIFO by 1.5–6.5× mean JCT.

A README table of *published result → our reproduction → delta* is the adoption artifact.

---

## 13. UX, packaging, repo

```
fleetsim run scenario.yaml -o out/ [--seed 42] [--override scheduler.name=fifo]
fleetsim validate scenario.yaml      # schema + feasibility; rejects unimplemented features
fleetsim plot out/                   # JCT CDF, queue-wait CDF, occupancy timeline, goodput
fleetsim compare out_fifo/ out_prio/
```

Minimal complete scenario:

```yaml
sim: {horizon: 14d, round: 60s, seed: 42}
fleet:
  metro: us-central
  clusters:
    - name: h100-main
      chip: {type: h100, per_node: 8}
      topology: {levels: [pod, rack, node], counts: [2, 16, 8]}   # 2,048 chips
      failures: {mtbf_node_days: 42, repair_auto_min: [60, 180]}
workload:
  kind: synthetic
  classes:
    pretrain: {rate_per_week: 2,  chips: pow2[256, 2048], duration: lognormal[median=10d, p90=30d],
               tier: prod, checkpoint_interval: 1h, min_runtime: 2h, within: pod}
    finetune: {rate_per_day: 30,  chips: pow2[8, 64],  duration: lognormal[median=4h, p90=24h], tier: batch}
    eval:     {rate_per_hour: 40, chips: pow2[1, 8],   duration: lognormal[median=2m, p90=30m],
               tier: batch, diurnal: true}
scheduler: {name: tiered_priority, params: {preempt: requeue}}
outputs: {events: parquet, plots: true}
```

Custom schedulers register via decorator or the `fleetsim.schedulers` entry-point group; `examples/03_custom_scheduler/` is a complete out-of-tree plugin package labs copy as a starting point.

- **License**: Apache-2.0 (patent grant matters to corporate labs; Blox/ASTRA-sim/Vidur/KAI all chose it).
- **Deps**: Python ≥3.11; `numpy`, `pandas`, `pyarrow`, `pyyaml` (+ optional `matplotlib`). Hand-rolled event loop, **no SimPy** — full control over determinism and tie-breaking is worth ~200 lines. No gRPC in core.

```
src/fleetsim/
├── engine/      # event loop, clock, RNG streams
├── fleet/       # domain tree, node state, failure injector
├── workload/    # synthetic generator, trace loaders (philly.py)
├── schedulers/  # base.py, fifo.py, tiered_priority.py, placement/
├── metrics/     # accumulators, steady-state window, summary, plots
└── cli.py
examples/  validation/  tests/  docs/
```

---

## 14. Key parameter defaults (with sources)

| Parameter | Default | Source |
|---|---|---|
| Node MTBF | 42 node-days | Meta RSC reliability (arXiv 2410.21680) |
| Job restart overhead | 15 min (5–20) | Meta RSC |
| Checkpoint interval / save cost | 1 h / 60 s | Meta RSC; Pollux measured 15–120 s restarts |
| Node auto-repair / manual repair | 60–180 min / 1–3 days (10%) | AIReSim |
| Maintenance drains | 1 per node-month | Borg |
| Preemption guard / max lifetime | 2 h (pretrain), 0 (eval) / 7 d | Meta RSC policy |
| Scheduler round | 60 s | Pollux (degrades past ~2 min) |
| Job-size tail | Pareto α≈0.7 (1% of jobs ≈ 99% chip-hours) | Borg TNG |
| Class mix | eval ~90% arrivals / <1% GPU-time; pretrain <2% / 70–94% | Acme |
| Failed/killed job share | 30–40% | Philly, Acme |
| Failure-cause mix | ~60% GPU/HBM, ~10% net, ~13% sw | Llama-3 paper |
| Healthy occupancy / ETTR benchmarks | 83–85% / >0.9 | Meta RSC |
| Interruption rate at 16K GPUs | ~1 per 3.1 h | Llama-3 paper |
| TPU slice sizes | ISCA 2023 Table 2 histogram | TPU v4 paper |

---

## 15. References

**Fleet managers & schedulers**: Borg (EuroSys '15), Borg TNG (EuroSys '20), MAST (OSDI '24), Singularity (arXiv 2202.07848), OpenAI 7.5K-node k8s blog, KAI Scheduler (github.com/NVIDIA/KAI-Scheduler), Slurm topology/preemption guides, Kueue TAS, Volcano.
**Hardware/topology**: TPU v4 (ISCA '23, arXiv 2304.01433), TPU v5p/Multislice/queued-resources docs, Pathways (MLSys '22), DGX SuperPOD H100/GB200 RAs, GB200 NVL72, NVIDIA Slurm-block blog, EC2 instance topology, Capacity Blocks, HyperPod task governance, Trainium2/Project Rainier.
**Traces & characterization**: Philly (ATC '19, msr-fiddle/philly-traces), Helios (SC '21, S-Lab-System-Group/HeliosData), PAI (NSDI '22, alibaba/clusterdata), Acme (NSDI '24, arXiv 2403.07648), Meta reliability (arXiv 2410.21680), Llama 3 (arXiv 2407.21783), MegaScale (NSDI '24).
**Scheduling literature**: Gandiva, Tiresias, Themis, Gavel, Pollux, Synergy, Lucid, Shockwave, Sia.
**Simulators**: Blox (EuroSys '24, msr-fiddle/blox), ASTRA-sim, SimAI (aliyun/SimAI), Batsim, UBCCR Slurm simulator, kube-scheduler-simulator, Vidur (MLSys '24), LLMServingSim, AIReSim.
