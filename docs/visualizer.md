# The visualizer (`fleetsim viz`, v0.3)

`fleetsim viz` turns a run output directory into **one self-contained
interactive HTML file** — a mission-control replay of the simulation:
scrub through time and watch jobs land on, get preempted off, and drain
from every pod of the fleet.

```bash
fleetsim run examples/01_minimal/scenario.yaml -o out_tiered
fleetsim viz out_tiered -o replay.html --open
```

The report is a single file with **zero external requests** — no CDN,
no fonts, no fetch, vanilla JS + Canvas + inline SVG only — so it
renders identically from `file://`, plain HTTP, an air-gapped laptop,
or an email attachment. This is test-enforced (`tests/test_viz_render.py`
greps both the template and rendered output for `http(s)://`, `url(`,
`@import`, `<link`, `fetch(`, `XMLHttpRequest`).

## CLI

```
fleetsim viz OUT_DIR [OUT_DIR_B] [-o REPORT] [--title T] [--map-level L] [--open]
```

| Flag | Meaning |
|---|---|
| `OUT_DIR` | a `fleetsim run` output directory (run A) |
| `OUT_DIR_B` | optional second run — compare mode (must share run A's horizon) |
| `-o REPORT` | report file to write (default: `OUT_DIR/report.html`) |
| `--title T` | report title (default: `fleetsim replay — <dir name(s)>`) |
| `--map-level L` | fleet-map level name, e.g. `pod` (default: inferred from the stint domain ids) |
| `--open` | open the written report in the default browser |

Errors: a missing/non-run directory exits 2 with a pointer to
`fleetsim run -o`; compare runs with different horizons exit 1 (their
timelines cannot share one axis — compare the same scenario at two
seeds or schedulers). A run **without** `stints.parquet` is not an
error: the report is still built in degraded mode (below) and the CLI
prints exactly what was dropped.

Python API, same pipeline:

```python
from fleetsim.viz import build_viz_model, render_html
model = build_viz_model("out_tiered")          # JSON-serializable dict
Path("replay.html").write_text(render_html(model))
```

## Recording stints (the replay input)

The fleet map replays **allocation stints**: who ran where, when. Opt
in per scenario:

```yaml
outputs: {stints: pod}    # one row per (stint × pod)
# or  stints: true        # the level directly below each cluster root
```

A named level must be declared by **every** cluster (`fleetsim
validate` catches this). The run then writes `stints.parquet`, one row
per (allocation stint × domain): `job_id`, `class_name` (workload
class label), `job_class`, `tier`, `domain`, `chips` (that domain's
share; shares sum to the gang's chips), `t0_us`, `t1_us`, and
`end_reason` ∈ {completed, preempted, failed, drained, canceled,
timeout, running_at_horizon}. `t1_us` is the **allocation release
time**: a requeue-preempted stint ends at grace expiry, a
failure/drain kill at the requeue instant; jobs still holding chips at
the horizon are truncated to it with `running_at_horizon`. Recording
cost is O(#leaves in the gang) per stint — measured deltas on the
frontier example are within run-to-run noise.

`examples/01_minimal` and `examples/04_frontier` both ship
`stints: pod`.

## The honesty contract — every pixel traceable

The report shows **only** what the run wrote. No smoothing, no
interpolation, no cherry-picking; where the model had to reconstruct or
truncate anything it says so in the **reconstruction notes** — printed
by the CLI at build time *and* expandable in the report header. Where
`summary.json` and a recomputation could both answer, `summary.json`
wins. Panel by panel:

| Panel | Source → pixels |
|---|---|
| **Summary cards** | `summary.json` verbatim: occupancy + goodput (steady-state window), jobs finished (full run), preemptions/min (window, all triggers), and wait p50/p99 for the top-3 workload classes by started count. |
| **Fleet map** (Canvas) | `stints.parquet`: a domain block at time *t* is filled by class with the chips of stints where `t0_us ≤ t < t1_us` (`running_at_horizon` stints count as still allocated at `t = horizon` — their `t1_us` is the truncation instant, not a release). Border pulses replay `end_reason` = failed (red — the *job's* failure, which includes workload aborts) / drained (orange) for `clamp(horizon/100, 2×round, 6×round)` after `t1_us`. Domain capacity: exact configured chips when a scenario copy is in the output directory, else the max concurrent chips ever observed (a stated lower bound). |
| **Timelines** (SVG) | `timeseries.parquet` downsampled to ≤ 1200 frames of contiguous buckets — occupancy (= allocated/healthy per raw sample) and goodput as bucket **means**, preemption/failure deltas as bucket **sums of raw per-sample diffs** (totals exactly preserved), frame time = last raw sample in the bucket. Allocation = allocated/total, with total chips derived exactly from `summary.json`. Pending-by-class is reconstructed from `jobs.parquet` (first wait only — requeued time after a preemption is not re-counted; noted). |
| **Gantt** ("the hogs") | `jobs.parquet`: top 300 jobs by chips × (end − start), plus **all** jobs ≥ 4096 chips. Preemption notches and failure end-caps come from the job's stints. |
| **Distributions** | `jobs.parquet`: empirical queue-wait and JCT CDFs per class — real ranks, ≤ 200 evenly-spaced points always keeping first and last. JCT over completed jobs only; tier `best_effort` excluded (matching `summary.json`'s own rule). |
| **Event ticks** | preemption waves (raw per-sample delta ≥ max(10, p99 of deltas)), node failures, and frontier (≥ 32768-chip) submits/starts. |
| **Footer** | `fleetsim <version> — deterministic replay of <out_dir>, seed <seed|n/a>`. |

Colors are consistent everywhere (map, gantt, pending, CDFs, legend):
pretrain `#4c6ef5`, finetune `#12b886`, eval `#fab005`, best_effort
`#64748b`, inference `#9775fa`; states failed `#e03131`, draining
`#f76707`, maintenance `#845ef7`, idle near-transparent. Custom class
labels (e.g. `frontier`) are mapped to their bucket's color by tier and
job class; when several labels share one bucket, the extra labels take
pinned lighter/darker shades of the bucket color (validated for
colorblind-safe separation) so no two classes ever render identically.
Timeline series that are *not* classes (occupancy,
allocation) use two reserved accent colors so class colors always mean
class identity.

Copying the scenario YAML into the output directory (any of
`scenario.yaml|yml`, `config.yaml|yml`) upgrades the model: exact
domain capacities including domains that never hosted a stint, exact
`round`, and the seed in the footer. Without it the model reconstructs
(observed domains, modal sample gap as the round) and says so in the
notes.

## Controls

Space play/pause · ←/→ step one round · speed ×1/×4/×16/×64
sim-hours per wall-second · drag any timeline to zoom the playback
window, double-click to reset · `z`/`x` zoom in/out around the cursor,
Home/End jump · on the map: arrow keys + Enter navigate and pin a
domain (pinning filters the gantt); hover for per-domain detail.

## Compare mode

```bash
fleetsim run scenario.yaml -o out_fifo --override scheduler.name=fifo
fleetsim viz out_tiered out_fifo -o ab.html
```

Run B appears as **dashed overlays** on all three timelines and as a
second column on the summary cards. The fleet map, gantt,
distributions, and event ticks always describe **run A** (a banner
says so). Column labels are the directory basenames (full paths on
collision). Both runs must share the same horizon.

## Degraded mode

Without `stints.parquet` the fleet map is hidden entirely
(`capabilities.map = false`) and the report degrades to fleet-level
replay: timelines, gantt, distributions, events, and cards all still
work. The CLI prints the note and the report header shows it.

## Performance

The model is bounded by construction: ≤ 1200 timeline frames and ≤ 300
gantt rows regardless of run size (stints are **not** downsampled —
the map replays every row). Measured at ~100K stint rows: ~7.5 MB
page, clean boot, 15 ms full forward scrub, 3 ms backward seek (the
playback cursor is two sorted pointers over typed arrays; seeking back
resets and replays). The 524,288-chip frontier example (~56K stint
rows) builds in seconds and lands well under 20 MB. More than 512
domains at the map level auto-aggregates the map to clusters, with a
banner.

Determinism: the model is a pure function of the output directory —
no wall clock, no randomness — and `render_html` is a pure function of
the model, so identical runs produce byte-identical reports.
