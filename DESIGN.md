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
    # NOTE: inside YAML flow mappings the bracket expressions MUST be
    # quoted (unquoted `[` is a YAML syntax error there); block-style
    # mappings (examples/01_minimal) need no quotes.
    pretrain: {rate_per_week: 2,  chips: "pow2[256, 2048]", duration: "lognormal[median=10d, p90=30d]",
               tier: prod, checkpoint_interval: 1h, min_runtime: 2h, within: pod}
    finetune: {rate_per_day: 30,  chips: "pow2[8, 64]",  duration: "lognormal[median=4h, p90=24h]", tier: batch}
    eval:     {rate_per_hour: 40, chips: "pow2[1, 8]",   duration: "lognormal[median=2m, p90=30m]",
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

---

## 16. v0.2 addendum

*Appended after v0.2 landed; §§1–15 are the v0 design as written. Where
they disagree, this section and the code win.*

### 16.1 The best-effort tier and the closed-loop backlog

Band 0's canonical name is now **`best_effort`** (`free` remains a legacy
spelling; `Tier.FREE` is an `IntEnum` alias and `Tier(0).name ==
"BEST_EFFORT"`). What makes the band real in v0.2 is its traffic model: a
workload class may declare a **closed-loop standing backlog** instead of
an arrival rate —

```yaml
best_effort:
  arrival: {process: closed_loop, closed_loop: {target_pending: 128}}
  # sugar: arrival: backlog[128]
```

The engine calls the source's `refill(now_us, pending_by_class)` hook at
every scheduler wake, after same-timestamp state changes settle and
before the scheduler view is built; the source tops the class back up to
`target_pending` pending jobs. Such classes default to `tier:
best_effort`, `min_runtime: 0`, preemptible, and (with
`checkpoint_interval: 0s`) zero-length preemption grace — instantly
reclaimable "cheap kills". This is the Borg-style saturated filler:
utilization → 1 while preemptive shielding leaves prod statistics
untouched (validated: `validation/test_priority_preemptive.py` measures
shielded prod sojourns at their solo M/M/1 value under a saturating
backlog). The metrics contract follows traffic-math.md §2.1: report
best-effort **goodput**, never best-effort mean wait — undefined under
saturation. The summary layer enforces it: BEST_EFFORT-tier jobs are
excluded from every queue-wait/JCT distribution, per-class stats are
additionally broken out by **source class** (`Job.source_class`, the
generating workload-class label, exported as a `jobs.parquet` column),
so a backlog reusing another JobClass never pollutes that class's
numbers. Rate keys plus `closed_loop` in one class are a validation
error, as is `diurnal: true` on a backlog class.

### 16.2 Segmented gangs and the frontier-scale rationale

Frontier training jobs do not fit inside one full-bisection domain and
have not for a while: Llama-3 405B ran as 8 data-parallel pods of 3,072
GPUs behind 1:7 oversubscription, TPU Multislice glues v5p slices over
DCN, and MAST schedules across clusters by moving data, not by finding a
mythical 100K-chip pod. The placement primitive this implies — and which
v0.2 implements (§4.2's `segments` field is now live) — is the
Slurm-block shape: `segment_nodes: N` + `segment_level: <level>` splits a
gang into equal whole-node blocks, each block contained in ONE domain at
the segment level, with `within` as the OUTER constraint (strictly above
the segment level; cluster roots when absent). Placement is atomic
(all segments or none), bin-packed onto the fewest segment domains by
descending free-node count on O(1) counters; `jobs.parquet` reports
`n_domains_spanned`. Failure semantics are unchanged: one node death
kills the whole segmented gang.

The matching scheduler capability is **segmented reclaim** in
`tiered_priority`: a pending segmented gang plans victims per segment
domain (greedy fewest-preemptions per segment, one aggregate victim set,
storm-capped at `max_preemptions_per_wake`, default 512), so one frontier
gang can empty multiple pods of best-effort mice in a single wake.
Two guardrails make reclaim exact rather than chip-count-optimistic:
every planned victim set is **dry-run-verified** by the engine's real
placement search with the victims hypothetically released (so node
shapes and leaf health are respected — a sub-node co-resident or a
DRAINING node cannot bait useless evictions), and an **in-flight claim**
suppresses re-planning for a preemptor whose victims are still in their
grace window (a grace longer than the round never triggers a redundant
second victim set).
`examples/04_frontier/` measures the whole story end to end at 524,288
chips: a 131,072-chip, 32-pod-segment job starts 233 s after submission
on a 99.5%-occupied fleet (345 preemptions in one wake), and the backlog
refills the reclaim dip within two rounds. Cross-segment bandwidth
penalties stay a v0.3 cost-model concern — `speed` remains 1.0 with the
hook in place.

### 16.3 Traffic v2

The v0.1 pinned diurnal step curve and uniform-over-exponents sizing are
superseded by the generative model specified in **docs/traffic-math.md**
(normative; §5.1 defaults here are retained only as the v0.1 sugar):
per-class arrival processes (Poisson, log-linear harmonic NHPP, MMPP-2,
Hawkes, closed-loop), weighted pow2 size pmfs with named presets,
lognormal-body durations with an optional truncated-Pareto splice tail,
finite-Zipf tenant marking (`tenant_zipf_s`, default 1.2 — a pinned
behavior change from v0.1's hardcoded 1.5), and the `google_fleet`
workload preset. Determinism contract unchanged and regression-pinned:
v0.1 scenarios produce byte-identical arrivals (`tests/test_traffic_v02.py`).
The closed-form validation ladder gained the traffic-math §5 rungs:
Pollaczek–Khinchine M/G/1 under lognormal service
(`validation/test_pk_mg1.py`) and preemptive-resume priority M/G/1
per-class sojourns plus backlog shielding
(`validation/test_priority_preemptive.py`), both asserted sharply at
realized moments and loosely against the analytic constants (the
round-alignment quantization and its 1/(1−ρ) amplification are modeled,
not hand-waved — see the test docstrings).

### 16.4 Roadmap update

v0.2 as shipped = traffic v2 + best-effort/closed-loop backlog +
segmented gangs with tiered reclaim + the frontier example + the new
validation rungs. **Moved out of v0.2 to v0.3**: EASY backfill/`Reserve`,
tenant quota (in/over-quota, lend-borrow), capacity classes beyond
on-demand, autoscaling inference + unserved-demand metric, and the
Helios adapter with the QSSF-beats-FIFO reproduction. v0.3 therefore
stacks those on top of its original scope (relaxable constraints +
placement-quality penalty, TPU `ocs_pool` predicates, multi-gang
Multislice, cross-segment speed penalty). v0.4+ unchanged (§11).

---

## 17. v0.4 addendum — the physics update

*Appended after v0.4 landed; where earlier sections disagree, this
section and the code win.  v0.4's theme: placements and capacity now
have PRICES — a crossing costs speed, an over-quota job costs its band,
a reservation costs everyone else the held nodes.  Every feature is
opt-in via a config key; a scenario using none of the new keys produces
byte-identical outputs to v0.3 for identical seeds (examples 01 and 04
are the CI regression anchors).*

### 17.1 The penalties model (with the relax pair)

`penalties: {xover: {<level>: <multiplier>}}` is the placement-quality
cost model that §4.2 and §16.2 deferred.  Semantics (pinned): at stint
start the engine computes `speed = Π multiplier(level)` over every
configured level at which the placement's leaves do NOT all sit under
one domain (a leaf with no ancestor at the level counts as its own
singleton domain); completion is scheduled at `start + overhead +
remaining/(speed·eff)` — the checkpoint math of §6.1 is otherwise
unchanged, and `speed` is constant per stint because allocations are
immutable.  The rule prices BOTH shapes of ugliness identically:
segmented multi-pod gangs (the v0.2 frontier shape) and relaxed
`within` placements.  Multipliers must be in (0, 1]; **there is no
default penalty** — the honest value is fleet-specific (example 05 uses
0.7 for a pod crossing behind 1:7 oversubscription; a full-bisection
fabric might be 0.95, a DCN hop 0.3).  Validation rung:
`validation/test_speed_penalty.py` asserts the completion identity
exactly (int-µs), including the segmented case.

Relaxable constraints ship in the same release, as §4.2 pinned:
`within: {level: pod, required: false, relax_after: 10m}` (also
`relax_after_s`; default 300 s).  The placement search (FirstFit) first
tries the constrained search; once `now − submit ≥ relax_after` it
retries unconstrained and marks the result `relaxed` — the engine
re-checks BOTH the relaxability and the elapsed timeout on every
`Place` (a scheduler cannot mark its way around a hard constraint), and
`GangAlloc.relaxed`/the `relaxed` jobs.parquet column record it.
Relaxing a segmented gang's OUTER constraint is rejected by validate.
Requeued jobs keep their original `submit_t`, so a preemption victim
does not restart its relax clock.

### 17.2 EASY backfill (estimate-error honesty)

`scheduler: {name: easy_backfill}`: FIFO ordering; the first
unplaceable job becomes the HEAD; the scheduler computes the head's
shadow time per candidate domain (chip-count accounting over
currently-free ELIGIBLE chips plus running jobs' estimated releases at
`stint_start + walltime_est/speed`, jobs without estimates never
release, same-wake placements included) and lets later jobs start now
only if `now + walltime_est/speed ≤ shadow` and they place — `speed`
is the §17.1 multiplier of the placement each job actually found
(`view.placement_speed`), so a penalized cross-domain backfill cannot
overstay its speed-1 promise by the penalty factor.  Free chips on
leaves held by a calendar reservation for another tenant (§17.4) are
SUBTRACTED from the shadow accounting — the head can never claim them,
so an active hold neither collapses the shadow to `now` (which would
disable backfill for the whole window) nor pads it with phantom
capacity.  Releases are still counted hold-blind and sufficiency is
chip-count, not node-shape — two documented approximations, and the
reason the non-delay rung runs on a fungible 1-chip-node fleet.  No
shadow ⇒ no backfill (never gamble against an unbounded reservation).
This is the CONSERVATIVE one-rule EASY: canonical EASY's second
disjunct (backfill any job that fits the "extra nodes" left over after
the head's reservation, regardless of estimate) is deliberately not
implemented, so measured backfill/utilization rates will undershoot
published EASY reproductions.  Pinned honesty contract: every decision
uses the scheduler-VISIBLE `walltime_est_s`, and the engine never
kills a job at its estimate.  The guarantee, stated exactly: with
EXACT estimates (est = true remaining) the head — pointwise, every job
on the rung's fungible fleet — starts no later than under strict FIFO
(`validation/test_backfill_property.py`, paired run).  With merely
honest OVER-estimates (est ≥ true) the pointwise property does NOT
hold — an inflated estimate widens the shadow window and a backfilled
job can outlive the head's true FIFO start (the rung carries the
counterexample); what holds then is the canonical EASY property
(Mu'alem & Feitelson, IEEE TPDS 2001): the head never starts later
than the shadow computed when backfill was granted.  Underestimates
delay the head past even the shadow, exactly as on a real cluster
without walltime enforcement (the same rung asserts the failure mode).
The v0.2 `Reserve` action was NOT added: the shadow reservation is
scheduler-internal state, and the engine's tentative-reservation
semantics (§7) already prevent double-booking within a wake.

### 17.3 Quota (in-quota / over-quota)

`quota: {tenants: {<name>: <chips>}, over_quota: best_effort|reject}` —
the v1 admission seam (§5) becomes `QuotaAdmission`.  Pinned semantics
(MAST-INSPIRED, deliberately simpler — see the honesty note):
commitment is taken at ADMISSION against the tenant's non-terminal
in-quota jobs (pending + running + graced) and released at the
terminal transition; a job that would exceed its tenant's cap is
OVER-QUOTA — demoted to the BEST_EFFORT band (`job.tier` overwritten,
`quota_demoted` marked in jobs.parquet, counted in
`counts.quota_demotions`) or rejected (FAILED) under `reject`.
Unlisted tenants are unlimited; `validate` rejects capped tenant names
that no configured workload class or service can ever produce (a
typo'd name would otherwise silently disable the cap).  Consequences
that follow from existing machinery, deliberately: demoted work is
preemptible scavenger load for tiered reclaim, and it is excluded from
queue-wait/JCT distributions like every BEST_EFFORT job (§16.1) —
per-source-class stats keep the open-loop classes honest.  Invariant
(CI rung, `validation/test_quota_conservation.py`): at every flush,
each tenant's running in-quota chips ≤ committed ≤ cap.

HONESTY NOTE (what this is NOT): the named references evaluate quota
differently.  MAST (OSDI '24) and HyperPod task governance judge
in-quota vs over-quota against a tenant's RUNNING usage at scheduling
time, treat over-quota work as a dynamic opportunistic state (borrowed
idle capacity, reclaimed on demand), and never permanently strip a
job's band because of queue depth.  fleetsim's model is admission-time
commitment over QUEUED demand with IRREVERSIBLE demotion: a tenant
bursting 100 short jobs against a 64-chip cap on an idle fleet keeps
only the first cap's worth in-quota and the rest stay demoted forever
(example 06's demotion counts are partly this artifact); demoted jobs
also keep their `min_runtime_s` shield, so the in-quota tenant cannot
instantly reclaim from long-guarded over-quota work.  Scheduling-time
evaluation and lend/borrow (HyperPod) remain future work.

### 17.4 Reservations as meta-jobs (calendar capacity blocks)

`reservations: [{id, tenant, chips, level, start, end, hard_end:
true}]` implements CALENDAR capacity semantics (§5) as an engine-driven
META-JOB above every band: at `start` the engine claims whole HEALTHY
nodes of the reservation's chip type inside ONE domain at `level`,
choosing the domain that needs the FEWEST EVICTIONS — scored by
distinct non-owner RUNNING jobs displaced, ties broken ascending
domain id, so a fully idle domain always beats a busy lower-id one
(Slurm advance reservations likewise select non-conflicting nodes);
within the chosen domain, leaves are taken free-first, then
owner-occupied, then foreign-occupied, ascending id within each group
(whole-node only, rounded up).  Non-owner residents are evicted
(REQUEUE, trigger `"reservation"`, bypassing preemptibility/
min-runtime — the capacity is contractually gone), and the leaves are
marked held.  GRACE LINGERING (pinned): evicted residents keep their
chips through the normal REQUEUE grace (`checkpoint_save_s`; zero for
SPOT) — the grace IS the eviction notice — so the first grace-window
of the hold can still be occupied by departing tenants, the owner
cannot place there yet, and that time DEBITS the owner's utilization
(capacity blocks bill from `start`; the exclusivity rung accounts for
exactly this lingering).  During `[start, end)` placement searches of
every other tenant skip held leaves (the engine also refuses non-owner
`Place` actions on them); the owner's jobs may use them but are not
steered.  Schedulers are NOT blind to holds: the engine view exposes
`reservations()` (every unfinished block — id, tenant, chips, level,
window, `active`, held leaves once claimed) and
`reserved_free_chips(domain, tenant)` (held-but-free capacity a tenant
cannot use — easy_backfill's shadow subtracts it), so a topology-aware
policy CAN refuse to place long jobs onto imminent holds, mirroring
Slurm's reservation-overlap check.  The built-in FirstFit does not
look ahead — on a busy fleet a claim still evicts.  At `end` the
marker lifts; with `hard_end` (default — capacity blocks end hard)
residents still on the hold are evicted first, INCLUDING the owner's:
the capacity cliff of §5 cuts through the owner's own run.  The
summary's `reservations` list reports, per block: claimed nodes,
evictions at claim and at hard end, and `utilization` = ∫ owner-used
chips on the hold dt / (chips_reserved × window) — an exact int-chip-µs
integral.  A fleet that cannot host the hold anywhere reports
`claim_failed` and holds nothing.  Failure semantics: a held node that
fails stays held (the lost time debits utilization — capacity blocks
bill for down nodes unless the operator releases them).  Exclusivity is
a CI rung via node-level stints cross-check
(`validation/test_reservation_exclusivity.py`).

SPOT (`capacity: spot`) completes the economics: preemption grace is
zero regardless of `checkpoint_save_s` (zero-notice kill) and recovery
is floor-of-checkpoint (§6.1) — a checkpoint-free spot job loses
everything on every kill.  This models the WORST CASE, deliberately
harsher than the cloud products it abstracts: EC2 Spot delivers a
2-minute interruption notice and GCP Spot ~30 s, inside which real
workloads checkpoint — fleetsim's spot banks nothing, so measured spot
losses are an upper bound (a configurable notice that banks an
out-of-band checkpoint, like drain grace, is future work).
`reserved`/`flex_start`/`calendar` as per-class capacity keys remain
rejected.

### 17.5 Supporting surface

`workload.classes.<c>.tenant: <name>` pins every job of a class to one
tenant (no Zipf draw; the `tenant/<class>` stream stays reserved) — the
idiom for reservation owners and spot fleets.  Feature-keyed output
schema: the new jobs.parquet columns (`relaxed`, `quota_demoted`) and
summary keys (`counts.relaxed_placements`, `counts.quota_demotions`,
`reservations`) appear iff the matching config section is present — so
they are stable per feature and absent (byte-identical) otherwise.
`fleetsim --version` prints the package version.  Reservation start/end
events ride the engine's maintenance event channel (rank 5) with tagged
payloads — `("res_start", i)` / `("res_end", i)` over the engine's
`(start, id)`-sorted reservation list — so claims and cliffs settle
after same-timestamp completions/failures/arrivals and BEFORE the wake:
the §6.1 ordering contract is preserved without renumbering
`EventType`.  The ENGINE view (not the `ClusterView` protocol) gained
four probed-with-`getattr` extras, in the `graced_job_ids`/
`reclaim_feasible` tradition: `placement_speed(placement)` (the §17.1
multiplier the engine will charge — exactly `Simulator.speed`),
`reserved_free_chips(domain_id, tenant)` and `reservations()` (§17.4),
and `release_tentative(job_id)` (same-wake rollback of an
examined-but-rejected `find_placement` reservation).

### 17.6 Roadmap update

v0.4 as shipped = penalties + relaxable constraints (the §4.2 pair,
including the v0.2-promised cross-segment penalty) + EASY backfill +
tenant quota + calendar reservations + SPOT + examples 05/06 + four new
validation rungs.  **Not shipped, moved out**: Gavel throughput matrix /
unpinned chip types and multi-metro two-stage scheduling (the original
§11 v0.4 scope) join the research-validation release.  Revised roadmap:
**v0.5 — web app** (a hosted UI over the visualizer's replay model:
scenario builder, run comparison, shareable reports; JSON-over-socket
scheduler protocol underneath); **v0.6 — research replay & validation**
(Helios adapter + QSSF reproduction, Philly/Pollux cross-simulator
rungs at scale, Gavel matrix + unpinned chips, the §12 ladder completed
in CI); v1.0 trust release unchanged (§11).

---

## 18. v0.6 addendum — the validation update

*Appended after v0.6 landed; where earlier sections disagree, this
section and the code win.  v0.6's theme: prove the numbers are REAL.
Every prior release validated fleetsim against itself (closed-form
queueing rungs, conservation invariants, determinism).  v0.6 validates it
against PUBLISHED cluster traces — it downloads the Helios (SC '21),
Philly (ATC '19), and Alibaba PAI (NSDI '22) traces, replays them, and
asserts the papers' reported ratios and distributions.  The full, honest,
numbers-populated writeup is [docs/validation.md](docs/validation.md);
this addendum is the design rationale.  No core engine file changed — the
suite is a new `fleetsim.validation` package plus one scheduler
(`schedulers/sjf.py`), so examples 01/04 stay byte-identical.*

### 18.1 Two kinds of claim (and the anti-goals)

A trace replay never reproduces a published headline to the last digit:
the released trace differs from the paper's analysis window, timestamps
carry no timezone, and every simulator makes placement choices the
original scheduler did not.  So the suite validates two kinds of claim and
weights them differently.  **Policy-effect** validations (strongest) assert
a *ratio* between two policies on the *same* trace — SJF's average JCT vs
FIFO's; ratios cancel absolute-scale error, so a 10%-light load leaves the
ratio intact.  **Distribution-match** validations reproduce a shape the
trace itself carries (Philly's killed/failed %); these test converter
fidelity, and their tolerance is dominated by the unpublished analysis
window.  Six published numbers are stated up front as **anti-goals**, never
asserted — Philly's 52.3% SM-cycle GPU *utilization* (a hardware counter, not
scheduler occupancy — a DES produces the different, higher *allocation*
occupancy), Helios's QSSF column (needs an absent `jobname` + a duration
predictor; SJF-oracle is the reproducible proxy), Alibaba's 50% GPU-sharing
saving (needs fractional GPUs, §18.6), Borg absolutes (normalized, BigQuery,
8 cells), Philly Table 2's delay-cause split, and co-location interference
(unmodeled).  The docs/validation.md §0 table carries each with its reason.

### 18.2 The `sjf` scheduler and SJF-oracle

V1 needed a scheduler v0.5 lacked.  `schedulers/sjf.py`
(`@register("sjf")`) orders pending jobs by `(walltime_est_s,
submit_time, id)` ascending — shortest *estimate* first; a `None` estimate
sorts LAST (`+inf`); ties break by `(submit_time, id)`.  `__init__(placement
=None -> FirstFit, strict=False)`; `strict=False` is a best-effort scan
(skip an unplaceable job, continue), `strict=True` blocks on the shortest
head-of-line job.  When the walltime estimate EQUALS the true duration —
which is exactly what the Helios converter writes — this is **SJF-oracle**:
a perfect service-time estimate, the exact analogue of the Helios reference
sim keying on `duration` and a strict upper bound on what the
duration-predicting QSSF policy can achieve.  A CI-always analytic rung
(`validation/test_sjf_ordering.py`) proves the underlying Smith-1956 SPT
optimality end to end through the real engine: on a fungible pool SJF starts
jobs shortest-first and its mean JCT ≤ FIFO's, at identical makespan.

### 18.3 The converter, the harness, and metric adapters

`convert_helios` (`validation/helios.py`) mirrors the existing
`convert_philly`: `cluster_log.csv` → canonical rows, dropping `gpu_num==0`
CPU jobs and pre-April-2020 rows, mapping the state enum (British
`CANCELLED` → `CANCELED`), capping `duration` at the 1,209,600 s (14-day)
Slurm max, and writing `duration` into BOTH `duration_s` and
`walltime_limit_s` (the SJF-oracle).  Per-VC node pools come from the
`cluster_gpu_number.csv` snapshot (GPUs ÷ 8).

The load-bearing engineering detail is that fleetsim schedules a **single
global pool** and never routes jobs to a cluster by tenant, but the Helios
reference sim schedules **each VC independently** (one worker per VC; jobs
never cross VC boundaries).  So V1/V2 are a **harness**
(`validation/harness.py::per_vc_replay`) that runs one simulation per VC on
a fleet sized to that VC and aggregates job-weighted to cluster totals — NOT
an engine change.  Replay fidelity is held fixed (plan §1): `failure_model`
off and checkpointing disabled so service time == trace `duration` exactly;
BATCH tier so jobs are included in the papers' distributions; an adaptive,
*verified* horizon so every windowed job reaches a terminal status (a
truncated long-waiter would bias the mean on exactly the jobs carrying the
FIFO-vs-SJF signal).  Two **metric adapters** (`validation/adapters.py`)
recompute the papers' summary definitions from `jobs.parquet`, where they
diverge from `summary.json`: average JCT over **all** terminal jobs (not
COMPLETED-only), and `#Queuing` = jobs whose wait exceeds one scheduler
round.

### 18.4 Results: what reproduces, and the one gap

On the real Helios September trace (deterministic; seed 0), fleetsim
reproduces the Table-3 FIFO-vs-SJF policy effect **strongly**: the direction
on all four clusters; the queuing-ratio band [3, 25]× on all four; the
JCT-ratio lower bound (SJF advantage, ≥ 1.3×) on all four; the cross-cluster
JCT-ratio rank (Saturn 8.75× strongest → Uranus 1.69× weakest, matching the
published Saturn 6.59 → Uranus 1.49 order); the queuing-share ordering
(Saturn > Venus > Earth > Uranus); and three of four JCT ratios inside
[1.3, 8]× (Venus 4.21, Earth 2.11, Uranus 1.69).  **The one documented gap**:
Saturn's JCT ratio is 8.75× (~9% past the 8× ceiling) — a MODELING gap, not
a bug.  fleetsim's FirstFit placement fragments Saturn's large gangs more
than the reference "consolidate" placer, so the most gang-heavy cluster's
FIFO blocking (absolute FIFO JCT 75,329 s vs a published 55,984 s, ~1.35×)
over-inflates.  The V1 opt-in test asserts [1.3, 8]× for the other three and
`xfail`s Saturn with exactly this diagnosis — the band is **not** widened to
pass; a consolidate placer (v0.7) is the remedy.

> **SUPERSEDED BY §19.** The gap is closed, but the diagnosis in the
> paragraph above is **wrong**: the replay fleet is single-level, so gang
> consolidation cannot change any outcome there, and the VC carrying 84% of
> the gap has no gang above 56 GPU. The real cause is *sub-node* stranding
> of whole-node capacity. Kept here as written because getting the
> mechanism wrong while getting the fix direction right is exactly the
> failure mode §19 exists to record.

Two plan assumptions were wrong and were corrected in the harness, not
fudged: (1) the reference scan is **strict/blocking**, not best-effort — a
best-effort scan collapses FIFO's head-of-line queuing (the whole point of
the published huge FIFO numbers) and yields ~1.4× ratios with the wrong
rank; (2) per-VC capacity must use the **September-max** quota, not a fixed
Sept-1 snapshot — several VCs' quotas drift within the month (on Uranus,
`vc7hD` spins up 0 → 416 GPU), and a Sept-1 pool over-congests Uranus and
breaks its rank.  Sept-max recovers published Uranus almost exactly.  V3
(Philly status split) is a converter-fidelity rung: `convert_philly` maps
Pass/Killed/Failed → COMPLETED/CANCELED/FAILED, and the by-count /
by-GPU-time split honours Table 6 (the killed+unsuccessful minority of jobs
is a majority-tilted share of GPU-time).  Its full-trace rung is written to
spec but **UNVERIFIED on real data** — the 1 GB Git-LFS artifact was not
fetched in this build (`fetch_trace` detects the LFS pointer and skips with
the `git lfs pull` remediation).

### 18.5 CI budget, downloads, attribution

CI runs only vendored slices — a REAL 2-VC Venus September slice and a
synthetic ~2k-row Philly slice under `tests/validation_traces/`, each with a
header comment recording source, license, and exact sampling command.  The
opt-in full replays are marked `@pytest.mark.trace_full` AND env-guarded
(`FLEETSIM_HELIOS_FULL` / `FLEETSIM_PHILLY_FULL`), so `pytest -m "not
trace_full"` stays fast and offline; the marker is registered in
`pyproject.toml`.  Downloads (`validation/fetch.py`) are **stdlib-only**
(`urllib` + `hashlib`, no new runtime dependency), cached under
`$FLEETSIM_TRACE_CACHE` or `~/.cache/fleetsim/traces/`, and **checksum- or
size-gated** — a truncated or wrong download can never silently pass a
validation, and a Git-LFS pointer is detected rather than cached as data.
`fleetsim validation cite [trace]` prints the license + citation each trace's
attribution requires (Helios CC-BY-4.0 / SC '21; Philly CC-BY-4.0 / ATC '19;
PAI free-use / NSDI '22), and `fleetsim validation run` runs the
vendored-slice checks with a human-readable PASS/FAIL report.

### 18.6 Roadmap update

v0.6 as shipped = the `sjf` scheduler + SJF-oracle + `convert_helios` +
per-VC replay harness + metric adapters + stdlib fetch/registry + the V1
(Helios FIFO-vs-SJF, CI smoke + opt-in four-cluster) and V3 (Philly status)
rungs + the SPT analytic rung + `fleetsim validation` CLI + docs/validation.md.
**Deferred to v0.7** (each blocked on one missing capability, now named
explicitly): **fractional / sub-chip GPU allocation** unlocks the Alibaba PAI
50% GPU-sharing saving and the 113 s V100 median queueing (V5); **per-job
delay-cause attribution** unlocks Philly Table 2's fair-share vs
fragmentation-delay split (V4); and a **consolidate placement policy** is
expected to bring Saturn's JCT ratio into band (§18.4).  The Gavel throughput
matrix / unpinned chip types and multi-metro two-stage scheduling (carried on
the roadmap since §17.6) remain future work.  v1.0 trust release unchanged
(§11).

---

## 19. v0.7 addendum — the placement update

v0.7's theme: **the *where* axis becomes a real, selectable policy family**,
and the last Helios deviation closes. Everything here is opt-in; `first_fit`
remains the default and every pre-v0.7 scenario's outputs are byte-identical
— asserted at the schema level for `examples/01_minimal` and
`examples/04_frontier`, and **verified by byte comparison** for this
release: both examples' `summary.json`, `jobs.parquet`,
`timeseries.parquet` and `stints.parquet` are `cmp`-identical to the same
scenarios run on the v0.6 tag.

Shipped surfaces: the policy family (§19.2), the config/metric/reclaim
plumbing (§19.3), the validation outcome (§19.4), plus
[docs/placement.md](docs/placement.md) (semantics, selection guide,
interactions and limits) and `examples/07_placement_study/` — a ~10-second
four-placer study on 256 chips whose README carries the measured
stranded-node / large-gang-wait / occupancy / goodput deltas and four
counter-results. Version 0.7.0.

### 19.1 Why: the v0.6 Saturn gap had the wrong explanation

§18.4 blamed Saturn's out-of-band JCT ratio (8.75× vs a published 6.59×) on
FirstFit fragmenting *large gangs* relative to a "consolidate" placer. That
cannot be right, and the refutation is structural, not statistical:

- the replay fleet is **single-level** (`topology: {levels: ["node"]}` → one
  cluster domain + one metro + N node leaves), replay jobs carry
  `within=None` / `segments=None`, and no `penalties.xover` is configured. So
  there is no higher domain to consolidate within and no crossing penalty to
  pay: **which** free nodes a multi-node gang takes provably cannot change
  the outcome;
- Saturn's largest gang is 200 GPU (25 nodes), and the single VC carrying
  **84%** of the FIFO−SJF gap tops out at 56 GPU with 0.7% of its jobs above
  one node. It is not a large-gang effect.

The real mechanism is **sub-node stranding of whole-node capacity**. §4.1's
allocation model says a leaf with ANY owner is ineligible for a whole-node
request — a faithful approximation of the trace rather than an identity:
`node_num == ceil(gpu_num/8)` holds for **99.66 %** of windowed multi-GPU
Helios jobs, and every one of the 225 exceptions used *more* nodes than the
ceiling (Helios sometimes spread a small job across hosts), a direction
fleetsim's one-leaf sub-node model does not represent either
(docs/validation.md §4.2). But FirstFit places *sub-node* gangs by
ascending leaf id with no preference for partially-used nodes, so on a
70.8%-single-GPU workload it opens fully-free nodes for 1-GPU jobs and
re-dirties nodes a gang just released. Free chips accumulate as 1–7-GPU
remainders no ≥ 8-GPU job can use; under the strict scan **100%** of
measured blocked-idle chip-seconds come from heads needing whole nodes, and
**87.7%** of free chips during blocks sit on partial nodes. At 94.4% offered
load that idleness sets the standing backlog, and FIFO's mean JCT *is* that
backlog. SJF barely notices — its head is almost always a 1-GPU job that
fits in a remainder.

**Design lesson, recorded deliberately:** the fix direction ("use a
consolidating placer") was right for the wrong reason, and the wrong reason
named the wrong axis (hierarchy instead of node packing). A modeling gap is
only closed when the *mechanism* is measured, not when a plausible label is
attached to it.

### 19.2 The policy family (`schedulers/placement.py`)

Ordering and placement were already separate axes (§7); v0.7 populates the
placement axis and makes it configurable:

| name | sub-node request | whole-node request |
|---|---|---|
| `first_fit` (default) | lowest leaf id with room | largest-first exact cover, ascending id |
| `best_fit` | **tightest sufficient** leaf (ties ascending id, early exit on exact fit) | tightest parent domain that fits alone; else parents ascending by capacity |
| `consolidate` | same as `best_fit` | tightest parent domain that fits alone; else parents **descending** by capacity (fewest *parent* domains — the grouping is flat, so on a 3-level fleet it can span more pods than `first_fit`; measured in docs/placement.md) |
| `spread` | leaf with the **most** free chips (worst fit) | round-robin across parent domains (the anti-policy / control arm) |

Search domains are likewise ordered tightest-first (emptiest-first for
`spread`) instead of by id. Segmented (`segments`) specs delegate unchanged
to §17's segment packer, and the v0.4 relax/penalty pair is
policy-independent: every policy runs the constrained search, then the
unconstrained retry after `relax_after_s`, marking the result `relaxed`.

**Named honestly:** `consolidate` and `best_fit` are *bit-for-bit identical
on a single-level fleet* (one parent domain), including the Helios replay.
`consolidate` is named for what the reference implementation calls its
placer; `best_fit` names the mechanism that actually matters. A test asserts
the equivalence, and `Consolidate`'s docstring states the degeneracy —
mistaking the name for the mechanism is what produced §18.4's misdiagnosis.

### 19.3 Surfaces

- **Tree primitives**: `FleetTree.search_best_fit` / `search_consolidate` /
  `search_spread`, plus a `search(spec, tenant, *, mode=...)` dispatcher.
  Additive and non-mutating, exactly like `search_first_fit` /
  `search_segmented` — **`search_first_fit` and `_scan_leaves` are
  untouched**. All modes honor tenant reservation holds (v0.4) and leaf
  health, and agree on *feasibility* on uniform-leaf fleets (they differ
  only in choice). On **mixed** leaf sizes — reachable via the template
  form, not the compact one — the greedy caveat of `_scan_leaves` applies
  and grouping can hide a cover, so the packed scan retries once in the
  ungrouped order; their whole-node feasibility is therefore a *superset* of
  first-fit's, never a subset.
- **View**: three `ClusterView` pass-throughs, so a policy never reaches
  into engine internals.
- **Config**: `scheduler: {name: <any>, params: {placement: <name>}}`.
  Resolved by `get_scheduler` for every scheduler that **opts in** by
  annotating `placement: PlacementPolicy` (all four built-ins, and any
  plugin following the convention); for those, unknown names are a
  **validation error** listing the available policies — the set is closed,
  unlike scheduler names. A plugin using `placement` for its own vocabulary
  gets the string passed through unvalidated, exactly as pre-v0.7: the name
  is a convention, not a reserved word.
- **Protocol**: `PlacementPolicy` declares `search_mode` as a member, not a
  convention — a policy omitting it would place one way and have its
  evictions planned another. `ClusterView` declares
  `reclaim_feasible(..., *, mode="first_fit")` for the same reason: a custom
  view stuck on the two-argument form works under the default policy only.
- **Reclaim consistency**: `search_after_release(..., mode=...)` closes a
  latent v0.2 inconsistency where reclaim planning always searched
  first-fit. `tiered_priority` forwards its policy's `search_mode` — and
  only when it is non-default, so existing two-argument `reclaim_feasible`
  callers and custom views are untouched **under the default policy**.
- **Validation harness**: `per_vc_replay` / `replay_canonical` take
  `placement=` and default it to the **engine** default (`first_fit`), so no
  pre-v0.7 caller's numbers move; the shipped rungs pass
  `VALIDATION_PLACEMENT = "consolidate"` explicitly, next to
  `pool_snapshot="max"`. `frag_prefix_s=` surfaces the mechanism metric out
  of the harness, which is what `scripts/helios_stranding_table.py`
  regenerates §19.4's table from.
- **Metrics**: `FleetTree.stranded_whole_nodes()` (count of HEALTHY leaves
  with `0 < free < chips`) and `stranded_whole_node_chips()` — the
  *mechanism* of §19.1 made directly observable. Sampled at flush into
  `timeseries.parquet` and aggregated into the summary's `fragmentation`
  map, with `counts.placement_policy` recording which placer ran. All three
  are gated on the scenario NAMING a policy, the same
  feature-enablement pattern as v0.4's trackers — so a scenario that names
  none keeps the exact pre-v0.7 output schema. Both surface in the human
  output too — a `stranded whole nodes (mean)` row in `fleetsim run`'s
  summary table and in `fleetsim compare` — under the same gate, because
  §19.4's first finding is that the *existing* headline rows (occupancy,
  goodput) can be bit-identical across placers, so a placement A/B read off
  them alone looks like a null result.

### 19.4 Result and its limits

Flipping the Helios harness to `placement="consolidate"` (a stated
validation-model choice, on the same footing as `pool_snapshot="max"`) puts
**all four** clusters inside the [1.3, 8]× JCT-ratio band with no `xfail`:
Saturn 6.87 (published 6.59), Venus 3.21 (3.07), Earth 2.95 (2.87), Uranus
1.51 (1.49). Saturn's absolute FIFO JCT lands at 55,978 s against a
published 55,984 s; all four absolute FIFO JCTs are within ±14% (was ±35%)
and all four `#Queuing` counts within ±12% (Earth's −27% deviation became
+7.3%). Six of those eight absolute quantities moved toward published and
two moved away (Uranus's FIFO JCT overshoots from +5% to −6.4%; Venus's
`#Queuing` drifts from −9% to −11.2%). Note the eight are **not independent** — they are all
functions of the same FIFO queue-wait distribution — so the evidence that
placement was the residual is the mechanism metric plus the cross-cluster
ordering, not a count of moved numbers (docs/validation.md §4.2, §5).

Re-measured end to end as its own step (all four policies × four clusters ×
FIFO and SJF on the real trace, 32 replays): every figure above reproduced to
the digit, `best_fit` came out **bit-identical** to `consolidate` on the real
trace as §19.2's degeneracy predicts, and the mean absolute error against the
published ratios is 3.4 % for `consolidate`/`best_fit` against 27.4 % for
`first_fit` and 29.5 % for `spread`.

Four limits are recorded rather than smoothed over.

1. **The ratio band does not select a placer.** `spread` — the anti-policy —
   also puts all four clusters inside [1.3, 8]×, also keeps Saturn-highest /
   Uranus-lowest, also keeps the q-share ordering, while being 45 % off
   Saturn's absolute FIFO JCT and 99 % off Earth's. What selects the placer
   is the **four-cluster mean absolute ratio error** (3.4 % vs 27.4 % /
   29.5 %) plus the mechanism metric; the absolute rung corroborates it and
   the validation asserts both, so "in band" cannot be read as "reproduces
   the paper" (docs/validation.md §4.2.3).
2. **`consolidate` does not dominate `first_fit`.** Of the five Saturn VCs
   carrying 97% of the gap, three get worse (+18.5% / +5.6% / +4.2% on FIFO
   mean JCT); the cluster win is carried by the one VC holding 41% of its
   jobs (−35.6%). The claim is "reproduces the reference placer's aggregate
   behavior", never "best-fit is a better scheduler".
3. **The bands are NOT tightened**, despite every point value now agreeing
   to within 5%, because the analysis window is unpublished and the capacity
   model is a choice. Separately, the suite is **order-sensitive**: 35.5% of
   Saturn's jobs share an exact submit second, and flipping FIFO's id
   tie-break to descending moves its FIFO JCT by 17%. That flip is a
   sensitivity probe against a knowingly wrong arrival order (ids are
   uniform-length and monotone in submit time on 3 of 4 clusters, so
   ascending id is the faithful one) — but it is large enough that Saturn's
   absolute number *alone* does not select a placer: under it `consolidate`
   is worse than `first_fit` on both the absolute deviation and the ratio
   (docs/validation.md §4.5).
4. **A fungible-pool counterfactual (node granularity removed entirely)
   overshoots published** (Saturn FIFO 33,181 s vs 55,984 s). The reference
   simulator fragments too; the target was never zero fragmentation.
   Node-granularity fragmentation is 56.0% of the `first_fit` FIFO JCT and
   best-fit recovers 45.9% of it — the rest is loss the real Helios
   scheduler also took. This row and the blocked-idle chip-second accounting
   are **ad-hoc one-off measurements** (throwaway instrumentation, not in the
   tree), unlike the placer rows and the mechanism table, which
   `scripts/helios_stranding_table.py` and the `trace_full` rungs regenerate.

The **mechanism metric** also separates the three placers on the real trace,
in the same order as their FIFO JCTs: time-average partially-occupied Saturn
nodes over the 26-day window (of 265) are 24.99 under `consolidate`, 27.43
under `first_fit`, 41.13 under `spread`. Honest leverage caveat: −8.9 % on
that time-average corresponds to −25.7 % on FIFO mean JCT, so it is a
directional indicator rather than a proportional explanatory variable — what
costs the FIFO head is stranding *at the moments it needs a whole node*
(docs/validation.md §4.2.3).

**A second, trace-free line of evidence** (`examples/07_placement_study/`,
4 racks × 8 nodes × 8 chips, mice + whole-node gangs at ρ ≈ 0.78 offered,
best-effort FIFO, seed 42, ~1 s per arm). It reproduces the mechanism away
from Helios and adds three findings the trace work could not show:

- `stranded_whole_nodes` 6.55 → 5.31 of 32 nodes from `first_fit` to
  `best_fit` (−19 %); 8-node-gang mean queue wait 19,057 s → 16,195 s
  (−15 %); `spread` strands 25 of 32 nodes, never places 80 of 109 8-node
  gangs, and delivers 28.4 % fewer chip-hours.
- **Full-run occupancy, goodput and allocated chip-hours are bit-identical
  across `first_fit`, `best_fit` and `consolidate`** (0.8140502977503281 /
  0.9831 / 70,021.35041129222): the same 3,436 jobs complete, so the entire
  placement effect is *when* work ran, never how much. Occupancy is
  therefore useless as a placement metric — which is the argument for
  §19.3's `stranded_whole_nodes` existing at all.
- **`best_fit` ≠ `consolidate`** on this 2-level fleet (16,195 s vs
  16,315 s on the 8-node mean), the complement of §19.2's single-level
  degeneracy.

And two limits the synthetic study surfaced, both recorded in the example's
README rather than smoothed away: across four seeds the *mechanism* metric
improves 19–20 % every time while the *outcome* regresses on one seed
(+15 % gang wait at seed 1234); and adding `within: rack` to the gang class
**inverts** the ranking (8-node-gang mean wait 39,677 s under `first_fit`
vs 121,486 s under `best_fit`) — node-level tightest-fit spreads gangs
across racks, so no rack ever drains. Placement policy interacts with
topology constraints; it does not dominate them.

### 19.5 Roadmap update

v0.7 as shipped = the `best_fit` / `consolidate` / `spread` policies + the
tree search primitives and view pass-throughs + the
`scheduler.params.placement` config surface with closed-set validation +
placement-mode-consistent reclaim planning + the stranded-whole-node metrics
and `counts.placement_policy` (with a console-summary row) + the Helios
harness `placement` knob and the V1p "placement model is load-bearing"
validation rung + `docs/placement.md` + `examples/07_placement_study/`.
Version 0.7.0.

**Still deferred, and two of these were pencilled in for v0.7 by §18.6 —
v0.7 spent its budget on placement instead, so they are restated as open
rather than quietly re-dated:**

1. **Fractional / sub-chip GPU allocation** — unlocks Alibaba PAI's V5
   (50 % GPU-sharing saving, median 0.042 GPU/instance). Chips are still
   shared whole inside a node, and no placement policy substitutes for it.
2. **Per-job delay-cause attribution** — unlocks Philly Table 2's V4
   (fair-share vs fragmentation-delay split). `stranded_whole_nodes` is a
   fleet-level down payment, not a substitute: it says how much capacity is
   stranded, not which job was delayed by it.
3. The Gavel throughput matrix / unpinned chip types, and multi-metro
   two-stage scheduling (unchanged from §11).
4. **A preempting scheduler under a non-default placement policy.** The
   plumbing is in place and unit-tested (`search_after_release(...,
   mode=...)`, forwarded by `tiered_priority`), but no shipped *validation*
   exercises the combination, so it is plumbing with a test rather than a
   measured result.
5. **Automatic placement-policy selection.** Nothing in v0.7 recommends a
   placer from fleet/workload shape, and the §19.4 counter-results argue
   against a naive heuristic: measure, don't guess.
