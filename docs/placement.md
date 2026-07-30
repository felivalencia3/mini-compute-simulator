# Placement policies (v0.7)

Scheduling has two axes, and fleetsim has always kept them separate:

- **Ordering** — *which* pending job goes next. That is the `Scheduler`
  (`fifo`, `sjf`, `easy_backfill`, `tiered_priority`, your plugin).
- **Placement** — *which chips* it gets. That is a `PlacementPolicy`.

Until v0.7 the second axis had exactly one implementation, `FirstFit`,
and it was invisible. v0.7 populates it with three opt-in siblings,
makes the choice configurable per scenario, and adds a fleet metric that
makes its effect observable. This document is the semantics and the
measured effects.

> **The default has not changed.** A scenario that does not name a
> placement policy runs `FirstFit` and produces byte-identical output to
> a pre-v0.7 build. Two levels of guard: CI asserts for
> `examples/01_minimal` and `examples/04_frontier` that they name no
> policy, resolve to `FirstFit`, and (for 01, which is fast enough to run
> in-test) emit none of the v0.7 keys or columns; and for this release both
> examples' `summary.json` / `jobs.parquet` / `timeseries.parquet` /
> `stints.parquet` were `cmp`-compared against the same scenarios run on
> the v0.6 tag and came out identical. Everything below is opt-in.

## Selecting one

```yaml
scheduler: {name: fifo, params: {placement: best_fit}}
```

Any scheduler that **opts in** by annotating the parameter
`placement: PlacementPolicy` — all four built-ins, and any plugin
following the convention — accepts the name; `get_scheduler` resolves the
string into a policy instance. For those, the name set is **closed**, so a
typo is a `fleetsim validate` error listing the four valid names rather
than a run-time traceback (unlike `scheduler.name`, which is an open
registry).

`placement` is a convention, **not a reserved word**: an out-of-tree
scheduler is free to take a `placement` param with its own vocabulary
(`placement: lowest_rack`), and such a string is passed through untouched
and not validated against the built-in set — exactly as it was before
v0.7. The gate is the annotation, checked by
`fleetsim.schedulers.placement.takes_placement_policy`.

Programmatically:

```python
from fleetsim.schedulers.fifo import FIFOScheduler
from fleetsim.schedulers.placement import BestFit, get_placement

sched = FIFOScheduler(placement=BestFit())
sched = FIFOScheduler(placement=get_placement("spread"))   # by name
```

From the CLI, for A/B studies:

```bash
fleetsim run scenario.yaml --override scheduler.params.placement=best_fit -o out_bf
```

## The four policies

Every policy answers the same feasibility question — *is there room for
this gang right now?* — and differs only in **which** of the fitting
leaves it picks.

One caveat, stated precisely because the first version of this sentence
was wrong: with **mixed leaf sizes** under one domain an exact whole-node
cover is a subset-sum problem and all of these use a greedy, so grouping
the leaves can hide a cover the ungrouped order finds. Mixed leaf sizes
*are* reachable in a v1 config — the **compact** form
(`chip.per_node` + `topology.counts`) is uniform by construction, but the
**template** form lets one cluster hold `children` templates with
different `chips`, and `fleetsim validate` accepts it. So the packed
modes retry once in first-fit's ungrouped order when the candidates are
not all the same size, which makes their whole-node feasibility a
**superset** of `first_fit`'s: opting into a policy can never strand a
gang the default would have placed. Uniform candidates take exactly one
pass, so every compact-form fleet is unaffected bit for bit.

Two request kinds matter, because fleetsim's allocation rule (DESIGN
§4.1) treats them differently:

- **sub-node** (`chips < node size`) — chips are shared inside one node.
- **whole-node** (`chips >= node size`) — needs an exact cover by
  **fully free** nodes. A node with *any* owner is invisible to it.

| policy | sub-node request | whole-node request | search domains |
|---|---|---|---|
| **`first_fit`** (default) | lowest-id eligible leaf with `free >= chips` | exact cover, largest leaves first, ties ascending id | ascending id |
| **`best_fit`** | **tightest sufficient** leaf (smallest `free >= chips`), ties ascending id, early exit on an exact fit | tightest parent domain that covers it **alone**; if none, parents **ascending** by capacity (fill tight holes, keep big empty domains intact) | ascending `(free_chips, id)` |
| **`consolidate`** | same as `best_fit` | tightest parent that covers it alone; if none, parents **descending** by capacity (**fewest _parent_ domains touched** — the `_pack_segments` rule; see the limit below) | ascending `(free_chips, id)` |
| **`spread`** | leaf with the **most** free chips (worst fit) | **round-robin** one leaf at a time across parent domains (parents descending by capacity) | descending `free_chips`, ties ascending id |

Inside any chosen parent domain, leaves are always taken largest-first
with ties by ascending id — unchanged from v0.1.

> **`consolidate` minimizes _parent_ domains, not crossings at every
> level.** The grouping is flat and happens at the leaves' parent level
> only, so "fewest domains" means fewest **racks** on a
> `[pod, rack, node]` fleet — never fewest pods. Measured on
> `levels: [pod, rack, node]`, `counts: [2, 2, 4]` with pod0/rack0 = 3
> free nodes, pod0/rack1 = 2, pod1/rack0 = 4, pod1/rack1 = 0, a 40-chip
> (5-node) gang: `first_fit` and `best_fit` take **1 pod / 2 racks**,
> `consolidate` takes **2 pods / 2 racks** — more pods and no fewer racks.
> With `penalties: {xover: {pod: 0.7}}` configured, `consolidate` is then
> the *only* one of the three that pays the crossing. Pinned by
> `tests/test_placement.py::test_consolidate_minimizes_parent_domains_not_pods`.

Three behaviours are **policy-independent** by construction, so switching
policies never silently changes them:

- **Segmented gangs** (`segment_nodes` + `segment_level`) delegate to
  `search_segmented` unchanged — segment packing already bin-packs by
  free-node capacity.
- **The v0.4 relax/penalty pair.** Every policy runs the constrained
  search, then (for `within: {required: false}` past `relax_after`) an
  unconstrained retry, marking the result `relaxed=True`.
- **Eligibility.** All modes skip non-`HEALTHY` leaves and leaves held by
  a calendar reservation for a different tenant.

### Determinism, and one gotcha

Policies are pure functions of `(job, view)` — no state, no randomness —
so a placement is a deterministic function of tree state, and every tie
break is a total order.

The gotcha: **leaf ids sort lexicographically.** `node0, node1, node10,
node11, ..., node2` — so "ties by ascending leaf id" is *not* numeric
order. This is the convention `first_fit` has used since v0.1 and the
new modes inherit it deliberately rather than diverging.

### The single-level degeneracy (read this before naming a mechanism)

On a fleet whose leaves are all direct children of the searched domain —
**every `levels: ["node"]` config**, including the Helios validation
replay — there is exactly **one** parent domain and one cluster root.
So:

- the search-domain ordering is a no-op (one candidate);
- the whole-node grouping has one group, and all three packed modes
  return precisely `first_fit`'s order;
- therefore **`best_fit` and `consolidate` are bit-for-bit identical**,
  and the only behavioural change versus `first_fit` is the *sub-node*
  tightest-fit.

This is asserted at unit scale in `tests/test_placement.py` and was
confirmed at trace scale: on the real Helios trace `best_fit` and
`consolidate` came out bit-identical on every one of the eight paired
cluster replays (4 clusters × FIFO and SJF), on every reported metric —
a measured fact, not an inference from the argument above
([validation.md](validation.md) §4.2.3). It is documented this
prominently because mistaking the *name* `consolidate` for the
*mechanism* is exactly what produced v0.6's wrong diagnosis of the
Saturn gap.

## Why it matters: the mechanism

A node with any owner is invisible to every whole-node request. So each
sub-node job placed on a *fully free* node converts a whole node's worth
of capacity into a 1..n−1-chip remainder that no gang can claim. On a
workload with many small jobs, first-fit-by-id opens a fresh node
whenever the lower-id nodes are momentarily full, and re-dirties any node
a whole-node gang just released (both allocators grow from the same
low-id end). Free chips then accumulate as remainders, and the jobs that
starve are the ones needing whole nodes.

v0.7 makes that directly observable rather than inferred from JCT — when
a scenario **names** a placement policy, these extra outputs appear:

| output | meaning |
|---|---|
| `timeseries.parquet` → `stranded_whole_nodes` | count of HEALTHY leaves with `0 < free < size`, sampled every flush |
| `timeseries.parquet` → `stranded_whole_node_chips` | free chips sitting on those leaves |
| `summary.json` → `fragmentation.stranded_whole_nodes` | `{mean, max, n_samples}` over flush samples, full run and window |
| `summary.json` → `counts.placement_policy` | which placer produced this run |
| console summary (`fleetsim run`) | a `stranded whole nodes (mean)` row + a `placement policy` row |
| `fleetsim compare` | a `stranded whole nodes (full mean)` row + a `placement policy` row, once any compared run names a policy |

All of them are gated on the scenario naming a policy — the same
feature-enablement pattern as v0.4's trackers — so a scenario that names
none keeps the exact pre-v0.7 output schema. A report that finds
`counts.placement_policy` absent should read it as `first_fit`.

Naming `first_fit` explicitly is therefore the way to get the
diagnostics without changing any scheduling decision.

## Measured effects

Nothing in this section is extrapolated; every number was produced by a
run in this repository.

### On the real Helios trace (SC '21) — the validation result

Four production clusters, per-VC replay of the September 2020 window,
strict scan, September-max sizing, FIFO and SJF under each placer — 32
cluster replays. Full detail and caveats:
[validation.md](validation.md) §4.2 and §4.2.3.

| placer | Saturn FIFO mean JCT | vs published 55,984 s | Saturn JCT ratio (published 6.59×) | mean abs error vs the four published ratios |
|---|---|---|---|---|
| `first_fit` | 75,329 s | +34.6 % | 8.75× (out of band) | 27.4 % |
| **`best_fit`** | **55,978 s** | **−0.01 %** | **6.87×** | **3.4 %** |
| **`consolidate`** | 55,978 s (bit-identical) | −0.01 % | 6.87× | 3.4 % |
| `spread` | 81,096 s | +44.9 % | 7.48× | 29.5 % |

Switching the harness from `first_fit` to `consolidate` moved six of the
eight published quantities in §5 of validation.md toward published
(including the suite's last out-of-band number) and two away from it. The
mechanism metric moves in the same order as the JCTs: time-average
partially-occupied Saturn nodes (of 265) over the 26-day window are
**24.99** under `consolidate`, **27.43** under `first_fit`, **41.13**
under `spread`.

Three honest limits from that work, restated here because they bound what
a placer choice buys you:

1. **The published ratio band does not select a placer.** `spread` — the
   anti-policy — also lands all four clusters inside the tolerance band,
   also keeps Saturn-highest / Uranus-lowest, and also keeps the
   queuing-share ordering, while being 45 % off Saturn's *absolute* FIFO
   JCT and 99 % off Earth's. What selects `consolidate` is the
   **four-cluster mean absolute ratio error** (3.4 % vs 27.4 % / 29.5 %)
   plus the mechanism metric; Saturn's absolute number corroborates it but
   is itself order-sensitive (validation.md §4.5).
2. **`consolidate` does not dominate `first_fit` per virtual cluster.**
   Of the five Saturn VCs carrying 97 % of the FIFO−SJF gap, three get
   *worse* (`vcQ4H` +18.5 %, `vcBLw` +5.6 %, `vcOIr` +4.2 % on FIFO mean
   JCT); the cluster win is carried by `vczIT` (−35.6 %, 41 % of Saturn's
   jobs).
3. **It does not improve every published quantity.** Two of the eight in
   validation.md §5 got worse: Uranus's absolute FIFO JCT overshoots
   (+5 % → −6.4 %) and Venus's `#Queuing` drifts further out (−9 % →
   −11.2 %).

### On a 10-second synthetic study — `examples/07_placement_study/`

4 racks × 8 nodes × 8 chips = 256 chips, mice (1/2/4-chip) + gangs
(16/32/64-chip) at ≈ 0.78 offered load, best-effort FIFO, seed 42, 14-day
horizon, ~1 s per arm. Measured (the example's README has the full table
and four counter-results):

| metric | `first_fit` | `best_fit` | `consolidate` | `spread` |
|---|---|---|---|---|
| stranded whole nodes, mean (of 32) | 6.55 | **5.31** | 5.37 | 25.31 |
| 64-chip (8-node) gang wait, mean | 19,057 s | **16,195 s** | 16,315 s | 739,842 s |
| 64-chip gangs ever started (of 109) | 107 | 107 | 107 | **29** |
| mice queue wait, p50 | 31.0 s | 31.4 s | 31.3 s | 30.1 s |
| occupancy / goodput, full run | 0.8141 / 0.9831 | 0.8141 / 0.9831 | 0.8141 / 0.9831 | 0.5829 / 0.9820 |
| allocated chip-hours | 70,021.35 | 70,021.35 | 70,021.35 | **50,139.90** |

Three things that example is in the repo to teach:

- **Occupancy and goodput cannot tell you which placer you ran.** The
  first three arms are bit-identical on both, to the last digit (the same
  3,436 jobs complete, each for its own duration); the whole difference is
  *when* jobs ran. The steady-state-window occupancy does move, by 0.2 pp —
  which is to say, by nothing a reader would act on. A placement study that
  reports only occupancy has measured almost nothing.
- **The node count is what matters, not the chip count.** Free chips on
  partial nodes barely move (16.95 → 16.49); the number of *nodes*
  holding them drops 19 %. 6.55 partial nodes deny 52.4 chips of
  whole-node capacity where 5.31 deny 42.5.
- **`best_fit` ≠ `consolidate` here** (4 racks, so the whole-node
  grouping has 4 groups) — unlike the single-level Helios fleet.

## Which one should you use?

| situation | policy | why |
|---|---|---|
| You are reproducing pre-v0.7 results, or you have no reason to choose | `first_fit` | the default; deterministic, cheapest to explain, and what every earlier fleetsim number was produced with |
| Many sub-node jobs **and** whole-node gangs competing for the same pool | `best_fit` | stops mice from converting whole nodes into unusable remainders; this is Slurm `cons_tres` best-fit / `CR_Pack_Nodes` |
| As above, **plus** you want a spanning gang packed into as few *racks* (leaf-parent domains) as possible | `consolidate` | same sub-node packing, but short parent domains are consumed biggest-first. **Not** a `penalties.xover` minimizer above that level — on a 3-level fleet it measurably spans *more* pods than `first_fit` (see the note in "The four policies"), so if you price pod crossings, measure all three rather than assuming this one |
| You want a control arm to prove a placement effect is real and signed | `spread` | manufactures exactly the remainders best-fit suppresses; if your metric does not separate `spread` from the rest, it is not measuring placement |
| Blast-radius or thermal spreading is a real operational requirement | `spread` | a policy real operators run — this is what it costs |

And the honest default advice: **measure it on your own fleet.**

## Where placement choice interacts with the rest of the engine

- **Preemption reclaim.** `tiered_priority`'s reclaim dry-run
  (`search_after_release`) searches with the configured policy's
  `search_mode`, so an eviction plan can never predict a placement the
  policy would not make. `search_mode` is a **member of the
  `PlacementPolicy` protocol**, not a convention: a policy that omits it
  would place one way and have its evictions planned another, so declare
  it. With the *default* policy the dry-run call keeps its exact pre-v0.7
  two-argument form, so a custom view implementing only that signature
  still works — **under the default policy only.** A non-default policy
  passes `mode=...`, so a custom view's `reclaim_feasible` must accept the
  keyword-only `mode` (it is declared on the `ClusterView` protocol with
  the documented `"first_fit"` default) or it raises `TypeError`. No
  shipped validation yet pairs a *preempting* scheduler with a non-default
  policy.
- **Calendar reservations (v0.4).** All modes honour tenant holds; a
  reserved leaf is simply not a candidate.
- **Node health.** All modes skip non-`HEALTHY` leaves, so a placement
  study and a failure study compose.
- **Crossing penalties (v0.4).** Placement decides the *span*;
  `penalties.xover` prices it — so the two are structurally coupled (the
  example-05 tradeoff, with the placer as a third knob). But **no policy
  here minimizes the span at every level**: `consolidate` minimizes the
  count of leaf-parent domains only, and on a 3-level fleet it spans more
  pods than `first_fit` (measured above). **Not measured end to end:** no
  shipped study pairs a non-default placer with a configured `xover`
  penalty, so treat the coupling as a design property, not a result — and
  A/B the placers on your own topology before assuming which one is
  cheapest to price.

## Limits worth knowing before you trust a result

1. **Tightest-fit at the node level is not tightest-fit at the rack
   level, and the ranking can invert.** In example 07, adding
   `within: rack` to the gang class (so a 64-chip gang needs a *fully
   empty rack*) makes `best_fit` far **worse** than `first_fit` on those
   gangs — measured 64-chip mean wait 39,677 s → 121,486 s at seed 42,
   with 99 rather than 107 of them ever starting. Node-level packing
   spreads *gangs* across racks, so no rack ever drains. Placement policy
   interacts with topology constraints; it does not dominate them.
2. **The mechanism metric is directional, not proportional.** On Saturn
   an 8.9 % drop in time-average stranded nodes accompanied a 25.7 % drop
   in FIFO mean JCT. What costs a blocked queue head is stranding *at the
   moments it needs a whole node*; a mean over all instants dilutes
   exactly those moments.
3. **Outcomes are seed-sensitive even when the mechanism is not.** Across
   four seeds of example 07, `stranded_whole_nodes` improves by 19–20 %
   every time, while the gang queue wait improves on three seeds and
   regresses ~15 % on one.
4. **Sub-chip sharing is still out of scope.** Chips are shared whole
   inside a node; fractional-GPU allocation (Alibaba PAI's headline) is
   deferred, and no placement policy substitutes for it.

## Writing your own

The protocol is one method plus one attribute — **both required**:

```python
from fleetsim.fleet.tree import Placement
from fleetsim.model import GangSpec


class PackTightly:
    #: REQUIRED protocol member: the tree search mode a preempting
    #: scheduler PLANS reclaims with.  Keep it consistent with what
    #: `place` actually calls — a policy that places with
    #: `search_best_fit` but reports (or omits, defaulting to)
    #: `first_fit` gets evictions planned for a placement it will never
    #: make.  `isinstance(policy, PlacementPolicy)` is False without it.
    search_mode = "best_fit"

    def place(self, job, view) -> Placement | None:
        spec = GangSpec(chips=job.chips, chip_type=job.chip_type)
        return view.search_best_fit(spec, job.tenant)
```

`ClusterView` exposes `search_first_fit`, `search_best_fit`,
`search_consolidate`, `search_spread` and `search_segmented` as raw,
side-effect-free searches, so a policy never reaches into engine
internals. Keep `place` a pure function of `(job, view)` — the
determinism contract depends on it.

The YAML `placement:` name registry is deliberately **closed** to the four
built-ins, so a custom policy reaches a scenario through a scheduler rather
than a name. `run_scenario` and the CLI construct the scheduler from
`scheduler.name`, so the wiring is a registered class that defaults to your
policy — exactly the `examples/03_custom_scheduler/` shape:

```python
from fleetsim import Place, Scheduler, register

@register("pack_tightly_fifo")
class PackTightlyFIFO(Scheduler):
    def __init__(self, placement=None, strict=False):
        self.placement = placement if placement is not None else PackTightly()
        self.strict = bool(strict)

    def schedule(self, view):
        ...    # your ordering; call view.find_placement(job, self.placement)
```

Keeping the `placement=None` keyword is what lets a scenario still say
`params: {placement: best_fit}` and override your default with a built-in.

One consequence to expect (verified, not guessed): the diagnostics gate is
on the *scenario naming a string*, so a run whose policy comes only from
your scheduler's default emits **no** `stranded_whole_nodes` columns and no
`counts.placement_policy` — the same as a plain pre-v0.7 run. Name a
built-in explicitly (even `first_fit`) when you want them, or record your
own metric.

The snippet above is the *raw* surface: it does **not** get the v0.4
relax/penalty retry or segmented-gang delegation for free. Those live in
`placement._SearchPolicy`, the (private) shared body the four built-ins
subclass — each of them is a two-line class over it, setting only
`search_mode` and `_view_method`. If you need those semantics, either
copy that ~30-line `place()` body into your policy or subclass it and
accept that it is internal API.
