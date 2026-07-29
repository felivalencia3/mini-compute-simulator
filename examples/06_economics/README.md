# Example 06 — capacity economics: quota, a calendar block, and a SPOT filler

**Question**: what does capacity *policy* — not scheduling — do to a
fleet? One 6-day run on 256 H100s (4 pods × 8 nodes × 8 chips) exercises
the three v0.4 economics mechanisms at once:

1. **Tenant quota** (`quota:`) — t0/t1/t2 share the on-demand pool under
   64-chip caps; Zipf marking makes t0 the over-subscriber, and its
   overflow is demoted to the `best_effort` band at admission.
2. **A calendar reservation** (`reservations:`) — tenant `acme` holds 64
   chips (8 whole nodes) inside one pod for days 2–4 with a *hard end*.
3. **SPOT** (`capacity: spot`) — a closed-loop best-effort backlog with
   checkpointing off: zero-notice kills, instantly reclaimable.

Failures are off, so every eviction in the run is a policy eviction.

## Run it

```bash
fleetsim run examples/06_economics/scenario.yaml -o out
fleetsim viz out -o report.html --open   # stints are at node level
```

## What it measures (seed 42, 6-day horizon)

All numbers below come from this exact run's `summary.json`,
`jobs.parquet`, and `stints.parquet`.

**Headline**: occupancy 0.987 (window 0.986), goodput 0.946 — capacity
policy keeps the fleet pinned near full while the band structure decides
*whose* work fills it.

### Quota: the over-subscriber pays, the others don't

1,144 of 3,719 shared-tenant jobs were demoted to best_effort
(`counts.quota_demotions`; 941 inside the steady-state window):

| tenant | jobs submitted | demoted | share |
|---|---|---|---|
| t0 (over-subscriber) | 2,157 | **785** | 36% |
| t1 | 971 | 229 | 24% |
| t2 | 591 | 130 | 22% |

Queue-wait split, median: **98 s** for in-quota jobs vs **8,272 s** for
demoted ones (per tenant, in-quota: t0 93 s / t1 102 s / t2 104 s;
demoted: t0 8,338 s / t1 4,510 s / t2 16,515 s). Demoted jobs are
preemptible band-0 residents: 133 of them were evicted at least once by
tiered-priority reclaim. (Demotion is charged at ADMISSION over queued
demand and is irreversible — a deliberately simplified, MAST-inspired
model; DESIGN §17.3 states the divergence from MAST/HyperPod exactly.)

The paired run without quota (`--override quota=null`, same seed) shows
what the caps buy: with no admission control the shared batch queue
melts down for *everyone* — median wait ≈ **53,000–55,000 s for all
three tenants**. Quota does not create capacity (occupancy is 0.987
capped vs 0.996 uncapped); it converts a fleet-wide meltdown into a
targeted one, priced to the tenant that over-subscribed.

### The calendar block: two policy spikes, one exclusive window

The summary's reservation report (`summary.json["reservations"]`):

| field | measured |
|---|---|
| claim | 8 nodes / 64 chips in `pod1`, at 2d + 90 s |
| evicted at claim | **3** running `train` jobs (trigger `reservation`) |
| utilization | **0.944** |
| evicted at hard end | **6** running `acme_train` jobs |
| status | completed |

- The claim picks the **fewest-evictions pod** (DESIGN §17.4): at the
  claim instant, 4 of `pod1`'s 8 nodes already ran `acme`'s own jobs
  (owner residents stay) and the other 4 carried 3 distinct `train`
  jobs — 3 evictions, vs 4 in `pod0`. The zero-notice spot filler had
  already been reclaimed off the contended nodes by tiered priority, so
  the claim never touches it directly.
- The claim still lands mid-round on a ~99%-occupied fleet, so it must
  evict: 3 in-quota `train` jobs are preempted (REQUEUE) at exactly
  172,890 s and requeue after their 60 s checkpoint-save grace (spike
  #1; `jobs.parquet` carries the per-job `n_preempt_reservation`
  counter). That grace-window lingering also debits `acme`'s
  utilization — capacity blocks bill from `start`.
- **Exclusivity** (cross-checked from `stints.parquet` against the
  report's node list): 48 stints started on the held nodes inside the
  window — **0** of them foreign. `acme` used 94.4% of what it reserved.
- At the 4-day hard end the capacity cliff cuts through the run: 6
  `acme_train` jobs are evicted mid-flight (spike #2), lose back to
  their last hourly checkpoint, and requeue for placement elsewhere.
  `validation/test_reservation_exclusivity.py` automates exactly these
  checks as a CI rung.

`acme` overall: 214 jobs, 204 completed, median wait 1,232 s on a
saturated fleet — the block is why its day-2-to-4 pipeline runs at all.

### SPOT: cheap to kill is expensive to finish

The spot filler is band 0 *alongside* the quota demotions — and the
demoted firehose out-competes it for scavenger capacity:

| class | chip-hours (of 36,864 fleet chip-hours) |
|---|---|
| acme_train | 19,836 |
| train (in-quota + demoted) | 13,803 |
| eval | 1,857 |
| spot_filler | **881** |

45 spot jobs suffered **207 zero-notice kills** (capacity `spot` ⇒ no
checkpoint-save grace; checkpointing off ⇒ every kill loses everything),
and only 7 ever completed. That is the honest price of the cheapest
tier on an over-subscribed fleet — and fleetsim's spot is deliberately
the WORST CASE (no usable interruption notice; DESIGN §17.4): spot
capacity is real only when the scavenger band has slack.

## How to read it

- **Quota moves pain, it doesn't add capacity.** Occupancy is ~0.99
  with caps and without (0.987 vs 0.996, measured); what changes is *who
  waits*: in-quota work at ~100 s vs the over-subscriber's overflow at
  8,338 s — instead of ~54,000 s for everybody.
- **A reservation is a meta-job above every band.** It claims the
  least-disruptive domain (evicting only what it must), excludes
  everyone else while it holds, and its hard end is a capacity cliff
  that evicts even its own tenant.
- **best_effort is a shared drain.** Demoted work, spot, and reclaim all
  meet in band 0 — the summary's per-source-class stats (and the
  BEST_EFFORT exclusion from wait/JCT distributions) keep the open-loop
  classes' numbers honest anyway.
