# The web app (`fleetsim serve`, v0.5; parallel + sweeps + live replay + compare + analysis + validation in v0.8)

`fleetsim serve` starts a **local** web app around the run/viz pipeline:
browse every run in a workspace, open any finished run as the
interactive 2D report, submit new scenario YAML from an editor and watch
live progress, and replay the fleet in a **three.js 3D view** — pods as
stacks of node slabs, colored by workload class, failure and drain
pulses included.  In v0.8 the 3D view also renders a run *while it is
executing*, drills into a pod's individual nodes, and hands out deep
links and PNG snapshots.

```bash
fleetsim serve --open        # http://127.0.0.1:8500/, workspace ./fleetsim-runs
```

Everything is served from your machine: the only dependency beyond the
Python package is a browser. The three.js build is vendored inside the
package (`fleetsim/serve/static/vendor/`, MIT, sha256-pinned by tests) —
the app makes **zero external requests**, and its Content-Security-Policy
would block any that slipped in.

## CLI

```
fleetsim serve [-p PORT] [--workspace DIR] [--host H] [--workers N] [--open]
```

| Flag | Default | Meaning |
|---|---|---|
| `-p`, `--port` | `8500` | port to bind |
| `--workspace` | `./fleetsim-runs` | run workspace directory, created if missing |
| `--host` | `127.0.0.1` | bind address — **non-loopback values expose the app to your network** and print a loud warning; prefer the default plus an SSH tunnel |
| `--workers` | `min(4, cpu_count-1)` | simultaneous simulation **worker processes**; submissions beyond `N` queue FIFO and report their position |
| `--open` | off | open the app in the default browser |

Ctrl-C shuts down cleanly and **within about a second**: queued runs are
cancelled (marked `failed` with a clear error), each in-flight run is
asked to stop at its next metrics flush, and after a short grace any
worker still going is terminated and its run marked `failed`. A long
simulation can never hold the terminal hostage.

Python API, same server — **guard your `__main__`**:

```python
from fleetsim.serve import serve

if __name__ == "__main__":                 # required, see below
    serve(port=8500, workspace="./fleetsim-runs", workers=4)  # blocks until Ctrl-C
```

Simulations run in worker processes, and worker processes re-execute the
main module (the documented requirement of Python's `spawn`/`forkserver`
start methods). Without the guard, an unguarded script would start a
second server — second workspace lock, second port bind, second set of
runs — inside *every* worker; fleetsim refuses with a
`UnguardedMainError` naming the fix instead of letting you debug a
phantom "workspace already owned" error. The `fleetsim serve` command
line is already guarded, so this only affects your own scripts.

## Parallel runs (v0.8)

Runs execute in a `ProcessPoolExecutor` — separate **processes**, not
threads. The simulator is CPU-bound pure Python holding the GIL, so
threads would timeslice and finish *later* than running one at a time;
processes actually use the cores. Determinism is untouched: a run is a
pure function of `(scenario, seed)`, so *where* it executes cannot change
its bytes (test-enforced — the same sweep run twice is byte-identical
cell for cell).

Admission is FIFO with at most `--workers` runs in flight. Everything
still waiting reports `status: queued` plus a 1-based `queue_position`
(`1` = next to start) on `/api/runs` and `/api/runs/{id}/progress`. The
pool is created lazily on the first run, uses the `forkserver` start
method where POSIX offers it (each worker forks from a single-threaded,
pre-imported helper — cheap, and it never forks the threaded HTTP
server), and its workers ignore SIGINT so the parent is the only Ctrl-C
handler.

## Sweeps (v0.8)

A sweep is one base scenario plus a grid of dotted-path overrides,
expanded into one ordinary run per cell:

```bash
curl -sX POST http://127.0.0.1:8500/api/sweeps \
  -H 'Content-Type: application/json' \
  -d '{"yaml": "...", "title": "packing study",
       "grid": {"scheduler.params.placement": ["first_fit", "consolidate"]},
       "seeds": [1, 2, 3]}'
# -> {"sweep_id": "sweep-...", "run_ids": [...], "n_runs": 6}
```

- Values are JSON and are applied through the **same override machinery**
  as `fleetsim run -o path=value`, so a sweep cell means exactly what the
  equivalent CLI flag means.
- `seeds` is sugar for one more axis (`sim.seed`) appended **last**. The
  axis order is the request's key order and the last axis varies fastest,
  so cells (and therefore `run_ids`) read like nested loops.
- The product is capped at **64 cells**; a larger request is `413` with
  the computed size, and the cap is checked on the computed product — a
  20-axis request is never expanded in order to be rejected.
- **All-or-nothing validation.** Every cell is parsed, schema-checked and
  feasibility-checked *before* a single run directory is created. One bad
  cell means `400` with per-cell messages and nothing on disk.
- Each cell's `scenario.yaml` is the base document with that cell's
  overrides applied, **re-serialized** — a complete, self-describing
  scenario (which is what `fleetsim viz` reads back for seed/round
  metadata), at the cost of dropping the base file's comments. The exact
  cell values also live in `meta.json` under `sweep_cell`.
- Cells are ordinary runs: `model`, `report`, `live`, `cancel` and
  `DELETE` all work on them, and their rail rows carry `sweep_id` /
  `sweep_cell`.
- `DELETE /api/sweeps/{id}` dequeues only cells still `queued`. A sweep
  whose every cell was still queued disappears entirely; a partially
  executed one keeps its record and its remaining cells.

Sweep records live in `workspace/.sweeps/<sweep_id>.json` — a
dot-directory, so they never surface as runs.

## Live replay (v0.8)

Before v0.8 stints reached disk only when a run *finished*, so the fleet
map could not fill in while a run was going. Now each metrics flush also
spools the stint rows that settled since the previous one, and
`GET /api/runs/{id}/live?cursor=N` streams them:

```json
{"status": "running", "cursor": 812, "more": false,
 "progress": {"t_us": 1800000000, "...": "..."},
 "stints": [{"job_id": "j7", "domain": "m/c/node0", "chips": 4,
             "t0_us": 60000000, "t1_us": 240000000,
             "end_reason": "completed", "...": "..."}],
 "open_stints": [{"...": "...", "end_reason": "open"}],
 "open_truncated": false, "open_pending": false, "stalled_at": null,
 "fleet": {"map_level": "node", "clusters": [], "domains": []}}
```

**The cursor contract**, precisely:

- The cursor is a **count of settled rows already consumed**. Start at
  `0`, then always send back the `cursor` the last response returned.
- `stints` holds settled rows *after* that cursor — immutable, never
  revised, returned **exactly once ever**, in settlement order (not the
  sorted order `stints.parquet` uses).
- At most **5000 rows** per response. `more: true` means rows remain on
  disk: request again immediately with the new cursor — **unless the
  cursor did not move**, which means the stream cannot advance and
  re-polling at 0 ms is an unthrottled request loop (measured 1,240
  req/s from one tab against the single stdlib server). A response that
  advances nothing must fall back to the normal poll spacing and surface
  the stall; `stalled_at` names the spool line the server could not read
  past when that is the reason (a truncated or corrupt row — the
  finished run's `stints.parquet` is unaffected).
- `open_stints` is a **replace-wholesale overlay** of the stints still
  open at that flush — `end_reason: "open"`, `t1_us` = the flush time. The
  same stint reappears in successive overlays with a growing `t1_us` until
  it settles, at which point it leaves the overlay and appears once in the
  cursor stream with its real `end_reason`. It is `null` while `more` is
  true, because a lagging client's settled prefix does not line up with
  it, and while `open_pending` is true (nothing was polling at the last
  flush, so the child did not spool it — see *What the spool actually
  costs*). `open_truncated` reports the overlay hitting the 5000-row cap.
  Both flags are **rendered**: the live 2D map and the 3D HUD each show a
  persistent badge saying the fleet on screen is partial, because a
  half-empty fleet with a confident caption is indistinguishable from an
  idle one.
- `progress` always reflects the **latest** flush, even when `stints` lags.
- `fleet` is the stint level's exact domain geometry — same shape *and
  same flat `domains` order* as the finished run's model, so one
  `domain_idx` indexes both. It is sent only on a `cursor=0` request
  (it never changes during a run) and is `null` when the scenario records
  no stints, or before the first flush.
- **The union is exact**: settled rows + the final overlay (with
  `open` read as `running_at_horizon`) equals `stints.parquet` row for
  row. That is asserted end to end in `tests/test_serve_live.py`.
- When `status` becomes `done`, switch to `/api/runs/{id}/model` for the
  full report model. The spool stays on disk, so a client that arrives
  late — or after a server restart — reads the identical stream from
  cursor 0.

## The app

- **Runs rail** (left): every run in the workspace, newest first, with a
  status dot and a three-number headline (window occupancy, window
  goodput, jobs finished) once done. Output directories dropped into the
  workspace by CLI runs (`fleetsim run -o workspace/name`) show up too.
  The list refreshes every 3 s.
- **New run** (`#new`): a YAML editor with line numbers, a starter
  template dropdown (the bundled `examples/*/scenario.yaml` plus a
  seconds-fast default), an optional title, **Validate** (schema +
  feasibility, exact CLI-parity error messages) and **Run scenario**.
  Up to `--workers` runs execute at once in worker processes; the rest
  queue FIFO and report their position. Under the editor, a **fleet-shape
  preview** redraws as you type — see
  [Fleet-shape preview](#fleet-shape-preview-v08).
- **Run view** (`#run/<id>/report`): while the run is queued/running, a
  live progress panel (sim time, occupancy to date, jobs
  finished/running/pending, chips allocated — one update per scheduler
  round, polled every 1 s; queued runs also show their FIFO position)
  with a **Cancel run** button while it
  executes; when it finishes, the full **2D report**
  (the same self-contained HTML `fleetsim viz` writes) in an iframe,
  plus a **Download report.html** button — the downloaded file works
  from `file://`, email, anywhere.
- **3D fleet** (`#run/<id>/fleet3d`): the showcase view, below.
- **Analysis** (`#run/<id>/insight`): attribution for one run — see
  [Analysis](#analysis-v08) below.
- **Compare** (`#compare/<id>,<id>,…`): two or more runs side by side —
  see below.
- **Explore** (`#sweep`) and the **sweep board** (`#sweep/<id>`): define a
  parameter grid, watch the cells run, chart the result — see below.
- **Validation** (`#validation`): every published number fleetsim is
  measured against, and every quantity it deliberately does not
  reproduce — see [The validation tab](#the-validation-tab-v08).
- **Live fleet map**: while a run executes, the run view's default tab
  grows a 2D block diagram that fills in as stints settle. The 3D tab
  shows the same stream as a fleet you can fly through.

Tip: give scenarios `outputs: {stints: true}` — the fleet map in the 2D
report and the entire 3D view replay `stints.parquet`. Without it both
degrade honestly and tell you what to add. The starter template sets it.
(`true` records at the level below the cluster root, which is what the
fleet views want; a level name like `stints: pod` also works, but only
when the scenario's tree actually has a level with that name.)

## Comparing runs (v0.8)

fleetsim exists to answer comparative questions ("does a disruption spike
cost occupancy?", "does tighter gang packing reduce fragmentation?"), so
comparison is a first-class view rather than two browser tabs.

**Selecting runs.** The rail's **compare** toggle shows a checkbox on
every row (a checkbox also appears when you hover a row, and is always
reachable by Tab). Shift-click extends a range. Two or more selected runs
enable **Compare N runs**, which opens `#compare/<id>,<id>,…` — a plain
URL you can bookmark or paste. The cap is **8 runs**: the compare view
has eight line colors and a ninth would have to reuse one, so the
selection refuses politely instead of drawing an ambiguous chart.

**The compare view** has four panels:

1. **Runs** — one chip per run: its letter (A, B, C…), its line color, and
   links to that run's own report and 3D view. Identity is carried by the
   letter as well as the color, never by color alone.
2. **Metric matrix** — rows are runs; columns are occupancy, allocation,
   goodput, ETTR p50, preemptions per minute, jobs finished, and
   queue-wait p50/p99 **per workload class**. Every cell is read straight
   from that run's `summary.json` and its tooltip names the exact key path
   (`summary.json · window.queue_wait_s_by_source_class.eval.job_weighted.p50 = 12.34`),
   so a number here and the same number in the run's report cannot
   disagree. The best value in a column is marked, but only where "best"
   is defined *and* visible: `jobs finished` is never marked (throughput
   follows the offered load), and a column whose winner ties at the
   displayed precision is left unmarked. The **scope** control switches
   every cell between the steady-state window and the full run.
3. **Overlaid timelines** — occupancy, pending jobs (all classes) and
   preemption rate, one 2px line per run, labelled with the run letter at
   the line's right end. Hovering shows a crosshair and a readout of every
   run's value at that time; the chart is focusable, and `←`/`→`/`Home`/
   `End` walk the same readout for keyboard and screen-reader use. Runs
   that have not finished are listed in the matrix but excluded from the
   charts, and the panel says so. If the runs cover different spans, the x
   axis stays absolute simulated time and a note says why the short ones
   end early.
4. **Config diff** — only the scenario keys that actually **differ**,
   from `GET /api/runs/{id}/scenario`, with a toggle to show the identical
   ones too. Runs without a readable scenario file are named and left out
   rather than making every key look changed.

Colors: the compare view uses a **run palette** deliberately disjoint from
the workload-class palette the report and 3D view use (test-enforced), so
a run line is never read as a class. Class names in the matrix are plain
text for the same reason.

## Sweeps in the app (v0.8)

**Explore** (`#sweep`) is the scenario editor plus a parameter-axis panel:
each axis is a dotted path and a comma-separated value list, with
one-click presets for arrival rate, scheduler, placement policy and seed.
Values are parsed as JSON when they parse (`24`, `true`, `[2, 8]`) and
kept as plain strings otherwise (`2h`, `pow2[8, 32]`); splitting respects
brackets and quotes, so `[2, 8], [4, 16]` is two values. The expansion
count is shown **before** anything is created (`2 × 3 = 6 runs · cap 64`)
and Launch is refused past the server's 64-cell cap. Launching posts to
`/api/sweeps` — all-or-nothing, so an invalid cell means nothing on disk
and the per-cell messages appear under the editor.

The **sweep board** (`#sweep/<id>`) polls while cells run:

- a cell table with status (queued cells show their FIFO position), the
  chosen metric per cell, and a link into each cell's own run view;
- **Dequeue queued** for the cells that have not started yet
  (`DELETE /api/sweeps/{id}` — running cells are left alone; cancel those
  from their run view);
- a chart of the chosen metric (occupancy, goodput or jobs finished):
  **bars** for a one-axis sweep, a **heatmap** for two axes — one hue,
  light-to-dark by magnitude, and every cell prints its value so color is
  never the only channel. Cells that have not finished read `…`;
- cell selection that hands straight off to the compare view.

Sweep records are keyed by the record's own axis order (JSON keys are
stored sorted), so a two-axis heatmap may show your axes transposed
relative to the order you typed; the row and column headers always name
which axis is which.

## Analysis (v0.8)

The compare view answers "which run is better". **Analysis**
(`#run/<id>/insight`) answers "why did *this* run do that". It reads
nothing but the run's own viz model, and where the recorded output cannot
answer a question it says so rather than guessing.

**Event drill-down.** Pick a preemption wave or a node-failure round from
the run's `events[]` and see the jobs that actually stopped in it — the
stints whose `t1_us` lands in the window with `end_reason` `preempted`,
`failed` or `drained` — plus chips freed, chips claimed in the same
window, and how many of the freed chips a *higher-tier* gang picked up on
the *same domain*. That last number is the "did the wave pay for itself"
answer, and the gang it names is labelled **inferred**: fleetsim records
no victim→beneficiary link, so co-location plus tier order is the whole
of the evidence. A victim with no higher-tier starter on its domain reads
"not determinable", never a guess.

> **Settlement window.** A preemption is *counted* when the scheduler
> decides it, but the job keeps its chips until its `checkpoint_save_s`
> grace expires — so its stint settles a round or more later, and at
> "the event's round only" a preemption wave legitimately shows an empty
> table. The window control extends the search by 0/1/3/10 rounds
> (1 by default) and the panel prints the interval it used, plus how many
> victims settled inside the event's own round.

**Correlation.** A scatter over the model's frames with an ordinary
least-squares fit, Pearson *r*, *r²* and *n* printed: fragmentation index
vs pending jobs (the queue-pressure proxy — the model carries no
per-round wait), occupancy vs goodput, node failures vs occupancy, and
fragmentation vs occupancy. One dot per round; ← → walk the dots with the
readout. It is a correlation over the rounds of **one** run under **one**
scenario — the panel says so, and points at the sweep + compare views for
anything stronger.

**Occupancy-dip attribution.** Rounds whose occupancy sits more than *k*
robust sigma below its local median (median over a 21-round centered
window; sigma is the MAD-based scale of the residual, so the dips do not
inflate their own threshold). *k* is a slider. Each dip's drop in
**allocated chips** is decomposed over the stints that ended inside its
window — node failures, preemptions, drains, normal endings — against the
chips re-claimed by stints that started in the same window, and the
identity

```
chips freed − chips re-claimed + residual  ==  observed drop
```

holds exactly, with the **residual drawn as a hatched segment** rather
than absorbed. It is chips-at-fault bookkeeping, not a causal model: a
stint is charged whole to one window, a frame is a bucket mean, and
grace-period jobs keep their chips — which is precisely what the residual
carries. `Δ healthy` is a column of its own, because a dip whose
denominator moved is a smaller fleet, not idle capacity. Without
`outputs: {stints: …}` the dips are still *sized*; their causes read
"no stint data — not decomposable".

The three panels need the v0.8 additions to the model's `frames`
(`allocated_chips`, `healthy_chips`, `pending_jobs`, `running_jobs`,
`failure_kills_delta`, `frag_index`) — all straight reads of
`timeseries.parquet` columns under the same bucket rule as the older
series. See the schema block in `src/fleetsim/viz/data.py`.

## Live replay in the app (v0.8)

The stream above is post-mortem only if nothing reads it while the run is
going. Two surfaces do.

**The 2D live fleet map** sits under the progress panel on a running
run's default tab: one block per domain at the stint level, filled
bottom-up by the classes on it right now, with per-cluster "chips busy"
and a PNG button. It appears at the first metrics flush and disappears
when the run finishes (the full 2D report replaces it).

**The 3D fleet tab renders a running run.** There is no model yet — the
view builds the equivalent from the live stream, re-indexes on every
flush, and grows the fleet in front of you. Concretely:

- an open stint is stored with `end_reason: "running_at_horizon"`, the
  same token the finished model uses for a stint that never released its
  chips, so the interval sweep that draws the fleet needs **no live-only
  branch** — 2D, live 3D and finished 3D are one code path;
- the live stream delivers rows in *settlement* order while the finished
  model ships them sorted by `t0_us`, so the sweep's start-order index is
  built explicitly rather than assumed;
- a **LIVE** badge means the transport is following the leading edge.
  Scrub, step or press play and it stops following (that is how you look
  back at a wave while the run keeps going); **⟲ live** re-attaches;
- the HUD reports **allocated chips** rather than occupancy: a running
  run has no timeline frames yet, and the stint index knows allocation
  exactly. It is labelled as what it is;
- the scheduler round is inferred from the flush spacing until the run
  finishes, so `←`/`→` step by a real round;
- when the run reaches `done`, the view re-mounts on the finished model
  (with events, frames and occupancy) by itself.

Without `outputs: {stints: …}` the live view says so — the same honest
"no map" signal the finished report gives.

## Node drill-down (v0.8)

**Click a pod** in the 3D view and the camera flies in and explodes it
into its individual **nodes**, floating above the pod's own footprint. A
breadcrumb names the trail — `cluster › pod › N nodes` — and `Esc` backs
out one level (node view first, then the pin).

- Node height is **chips used / capacity**; color is the dominant class
  on that node; hovering one lists the gangs on it.
- Chips per node is an **exact read** of the run's own scenario file
  (`GET /api/runs/{id}/scenario`), not a guess. Without a readable
  scenario the pod is drawn as a single node and the breadcrumb says so.
- **What is measured and what is not.** fleetsim records placement at the
  *stint level* — the pod. Which of a pod's nodes a gang actually sat on
  is therefore **not recorded**, and the split shown is a deterministic
  largest-first packing of that pod's residents: a *layout*, labelled as
  one in the breadcrumb every time. The exception is a pod that **is** a
  node (`outputs: {stints: node}`), where the fill is exactly what the
  stints record and the panel says that instead.
- Health is likewise a **pod-level** state ("node failure in the last N
  rounds"), because that is the granularity the run recorded.

Performance is unchanged: the expanded pod gets **one more instanced
mesh** (capacity 512, allocated once and reused), never one object per
node. The whole fleet is still a single draw call.

## Deep links, PNG export, and recording (v0.8)

**Deep links.** A run route may carry a query string after the mode:

```
#run/<id>/fleet3d?t=1192080000000&cam=-9.475,4.6,-0.775,9.6,-0.78,1&pin=demo%2Fh100-east%2Frack6&x=demo%2Fh100-east%2Frack6&hide=eval
```

| Field | Meaning |
|---|---|
| `t` | simulated time, **integer microseconds** |
| `cam` | camera pose: `targetX,targetY,targetZ,radius,theta,phi` |
| `pin` | pinned domain id |
| `x` | expanded (drilled-into) domain id |
| `hide` | comma-separated workload classes filtered out |

**Copy link** in the transport bar mints that URL from the current view;
opening it restores the exact moment, angle, pin, drill-down and filters.
Every field is optional and independently validated — a hand-edited link
never throws, it just opens at the default. The hash is **not** rewritten
while you scrub: one history entry per frame would bury the back button.

**Class filters.** The 3D legend keys are buttons. Clicking one drops
that class's chips back to idle capacity, which is how you isolate one
workload in a busy fleet. Filter state travels in the deep link.

**PNG export.** `PNG` in the 3D transport bar downloads the current
frame straight off the WebGL canvas (the renderer keeps its drawing
buffer for exactly this). Every 2D chart panel — compare timelines,
analysis scatter and dip bars, the live fleet map, the editor's fleet
preview — has its own `PNG` button: the live `<svg>` is serialized,
rasterized through a `data:` URI and saved. Both paths are local; nothing
is uploaded, and the app still makes zero external requests.

**No GIF, deliberately.** Animated capture would mean either a new
runtime dependency or a hand-rolled encoder, and the dependency surface
of this app is zero on purpose. Record the 3D view the way you would
record anything else on your screen — macOS `Cmd-Shift-5`, Windows Game
Bar, `ffmpeg -f x11grab`, or your meeting tool — and use a deep link to
set up the shot: open the link, press play, record. The link makes the
take reproducible; the recorder is your OS's job.

## Fleet-shape preview (v0.8)

The scenario editor draws the fleet a document describes **before it
runs**. Typing is debounced (700 ms) into `POST /api/preview`, whose
response carries a `fleet` object for a parseable scenario; the editor
renders it with the same block-diagram code the live fleet map uses —
one block per domain at the stint level, grouped by cluster, with total
chips, nodes, chips-per-node, chip type and the level vocabulary.

**Why its own route.** `/api/validate` runs the full gate, whose
feasibility half *builds* the declared fleet: measured 1.9 s at the
262,144-node ceiling and 89 s for a 100,000-rack tree, synchronously in
an HTTP handler thread. Firing that on every typing pause dragged the
rail's own 3-second poll from 0.1 s to 4.9 s. `/api/preview` stops at
parse + schema, and the editor holds **one preview request in flight at
a time** (a pause while one is outstanding coalesces into a single
re-run). Nothing that can reach execution is validated any less:
`/api/validate` and `/api/runs` are unchanged, and this route creates
nothing. The preview reports "not a valid scenario document yet" rather
than "invalid" — Validate is what makes the stronger claim.

**The blocks are the level the run will actually record**, read from
`outputs.stints`: a level *name* records domains at that level, `true`
records the level directly below each cluster root, and absent/`false`
records nothing at all — in which case the preview draws the fleet but
says the report's fleet map and the live map will both be empty. (Drawing
the root-children level unconditionally was wrong for every scenario that
names a level: example 07 ships `stints: node` and previewed 4 rack
blocks for a run that records 32 node domains.)

The shape is **arithmetic**: the declarative count tree is summed, never
materialized, and the block list is capped at every level of the walk —
so a 262,144-node fleet costs 256 blocks, not 262,144 iterations. A
scenario may legally declare that many nodes (the validation ceiling),
and this runs on every keystroke. It is exact all the same, including the
domain ids (`rack0`, `rack1`, …), which are numbered the way
`fleetsim.fleet.build` numbers them, so a preview block and a later stint
row name the same domain. Past 24 clusters or 256 domains the preview
truncates and prints the real counts.

While the scenario is invalid the panel says so and shows the first few
messages; the full error list still belongs to the explicit **Validate**
button, because dumping every error on every keystroke is noise.

## The validation tab (v0.8)

`#validation` renders `GET /api/validation`, which **is**
`fleetsim.validation.results.payload()` — the same module the validation
tests import. Nothing on the page is typed in by hand, so a number on
screen and a number asserted in CI cannot disagree.

The page leads with **what fleetsim does not reproduce**: the documented
anti-goals, each with the reason it is out of reach (a hardware SM-cycle
counter is not scheduler occupancy; a per-job delay-cause split is not
computed; …) and its disposition. A validation page that lists only
successes is marketing, so the anti-goals sit above the results, not in a
footnote.

Then one panel per published table — subject, published value, fleetsim's
measurement, relative error, the band or tolerance, and a verdict. Three
rendering rules carry the honesty:

- `fleetsim: null` is **honestly unmeasured** (an opt-in rung that was
  not run) and renders as "not measured" — never as agreement, and never
  with a relative error;
- a row carrying neither a band nor a tolerance is **reported, never
  asserted**, and says so instead of showing a verdict;
- the previous release's measurement appears as movement
  (`6.870 ← 8.750`), never as a second claim.

Below that: the validation ladder (what each rung asserts and where it
ships), the placement-policy sweep, and the trace citations and licences.
The long form is `docs/validation.md`; the tab links to it by name and
does not paraphrase it.

## API contract

All endpoints are same-origin JSON unless noted; every response carries
`Cache-Control: no-store` and `X-Content-Type-Options: nosniff`; every
error is JSON `{"error": str}` — never an HTML error page (framework
errors like an unsupported method or a bad request line included),
never a traceback (a 500 carries the exception *class* only; the detail
goes to the server terminal). Request bodies are capped at 5 MB, and
`POST` bodies must be `Content-Type: application/json`. The `Host`
header must be a loopback authority (`127.0.0.1[:port]`,
`localhost[:port]`, `[::1][:port]`, plus an explicit `--host` value) —
anything else is `421`; state-changing routes additionally reject
requests bearing a foreign `Origin` or a cross-site `Sec-Fetch-Site`.

| Route | Returns |
|---|---|
| `GET /api/runs` | `[{id, title, status, created, headline, queue_position, error?, sweep_id?, sweep_cell?}]` newest first; `status` ∈ `queued\|running\|done\|failed`; `headline` = `{occupancy, goodput, jobs_finished}` for done runs, else `null`; `queue_position` is 1-based, non-null only while `queued`, and **contiguous across the listing** (the positions are renumbered against the rows actually observed as queued, so a run admitted mid-listing cannot leave a gap) |
| `GET /api/runs/{id}` | the row's meta plus `summary: <full summary.json>` when done (`null` otherwise); 404 unknown id |
| `GET /api/runs/{id}/progress` | `{status, queue_position, progress}`; `progress` = `{t_us, horizon_us, jobs_finished, jobs_running, pending, occupancy_to_date, allocated_chips, healthy_chips}`, one snapshot per metrics flush, final snapshot at `t_us == horizon_us`; `null` before the first flush and for external runs. Read from the run's on-disk spool, so it survives a server restart |
| `GET /api/runs/{id}/live?cursor=N` | the live replay stream — `{status, cursor, more, progress, stints, open_stints, open_truncated, open_pending, stalled_at, fleet}`; see [Live replay](#live-replay-v08) for the cursor contract. Requesting this also registers "a client is watching", which is what makes the child spool the open-stint overlay |
| `GET /api/runs/{id}/scenario` | `{id, name, yaml, flat, truncated, parse_error?}` — the run's own scenario file **verbatim** plus `flat`, the document as `{dotted.path: value}` (nested mappings recurse; a list is one leaf). Available at any status; `flat` is `null` with a `parse_error` when the file is not a YAML mapping; 404 when the run has no scenario file. This is what the compare view's config diff reads — a browser has no YAML parser and the app ships no new dependency |
| `GET /api/runs/{id}/model` | the viz JSON model (`fleetsim.viz.build_viz_model`; schema in `src/fleetsim/viz/data.py`), disk-cached as `viz_model.json`; 409 until the run is done. A cache written by an older fleetsim whose `frames` predate the v0.8 analysis series is **rebuilt** rather than served (`runs.MODEL_CACHE_MARKERS`) — the recorded run bytes never change, but the model schema grows |
| `GET /api/runs/{id}/report` | the self-contained 2D report HTML, disk-cached as `report.html`; 409 until done |
| `POST /api/validate` | body `{yaml: str}` → always 200 `{ok, errors: [str], fleet?}` for a well-formed request; 400 for a bad envelope. **The full gate** — parse, schema and feasibility, the same one `POST /api/runs` applies. `fleet` is present **only when the scenario is valid**: the fleet it describes as counts (`{total_chips, total_nodes, n_clusters, chip_types, stints_mode, stints_level, clusters_truncated, clusters: [{id, metro, name, levels, map_level, chips, nodes, chip_type, chips_per_node, n_domains, domains_truncated, domains: [{short, path, chips, nodes}]}]}`), summed from the declarative count tree — never materialized |
| `POST /api/preview` | the **same** request and response shape, **parse + schema only** — no feasibility pass, so no fleet is built. This is what the editor's fleet-shape preview calls on every typing pause; it creates nothing and validates nothing that reaches execution |
| `POST /api/runs` | body `{yaml: str, title?: str}` → 200 `{id}`; 400 `{ok: false, errors}` for an invalid scenario |
| `DELETE /api/runs/{id}` | 200 `{ok: true}` for **queued** runs only (dequeue); 409 for running/done/failed, 404 unknown |
| `POST /api/runs/{id}/cancel` | 200 `{ok: true}` for **running** runs: cooperative cancel — the run stops at its next metrics flush and is marked `failed` with `cancelled by request`; 409 for any other status, 404 unknown |
| `POST /api/sweeps` | body `{yaml, title?, grid: {dotted.path: [values]}, seeds?: [int]}` → 200 `{sweep_id, run_ids, n_runs}`; 400 `{ok: false, errors}` for a malformed grid or **any** invalid expansion (nothing is created); 413 beyond the 64-run cap |
| `GET /api/sweeps` | `[{sweep_id, title, created, n_runs, n_done, n_failed, grid, seeds}]` newest first |
| `GET /api/sweeps/{id}` | the sweep plus one row per cell: `runs: [{id, title, status, created, queue_position, cell, headline, error?}]`; 404 unknown |
| `DELETE /api/sweeps/{id}` | 200 `{ok, dequeued, kept, removed_record}` — dequeues only cells still `queued` |
| `GET /api/validation` | the validation suite's measured published-vs-fleetsim table as data (`fleetsim.validation.results.payload()`): `{version, headline, results, groups, counts, ladder, anti_goals, placer_sweep, citations, doc}`. Same module the validation tests import, so a number on screen and a number asserted in CI cannot disagree |
| `GET /api/examples` | `[{name, yaml, runnable, note?}]` — the bundled starter scenarios, read-only, sorted; `yaml` is verbatim, `runnable: false` (+ human `note`) marks a starter that cannot run as web-submitted (`[]` on an installed wheel without the repo checkout) |
| `GET /`, `GET /static/*` | the app shell (packaged static files, no build step) |

## Workspace layout

Each managed run is one directory, `workspace/<slug>/`:

| File | Written | Contents |
|---|---|---|
| `scenario.yaml` | at submit | the submitted scenario text, verbatim (for a sweep cell: the base document with that cell's overrides applied, re-serialized) |
| `meta.json` | at submit, updated on transitions | `{title, created_unix, status, error?}` (+ `sweep_id`, `sweep_cell` for a sweep cell) |
| `live.json` | every metrics flush, replaced atomically | `{cursor, progress, open_stints, open_truncated, open_omitted}` — the live stream's state |
| `live.jsonl` | every metrics flush, appended | settled stint rows, one JSON object per line; **line index == cursor** |
| `live_fleet.json` | first metrics flush | the stint level's domain geometry (absent when the scenario records no stints) |
| `live.watch` | touched on every `/live` request | "a client is watching" — see below |
| `cancel.flag` | on `POST /cancel` or shutdown | the parent's cooperative-stop signal to the worker process |
| `summary.json`, `jobs.parquet`, `timeseries.parquet` (+ `stints.parquet`) | at completion | the standard `fleetsim run` outputs |
| `viz_model.json`, `report.html` | lazily, on first request | disk caches of the model/report endpoints. Runs are immutable once done, so a cache never invalidates *on content* — but both are rebuilt when they were written by an **older fleetsim** (the model schema and the report renderer grow between releases), so the report iframe and the analysis tab can never be built by two different versions |

### What the spool actually costs

The live spool is kept after the run finishes (superseded by
`stints.parquet`, but it is what makes a late or reconnecting client see
the identical stream). Delete it with the run directory if the space
matters — and it is not small:

- **`live.jsonl` runs about 195–210 bytes per stint row**, roughly **9×**
  the `stints.parquet` holding the same data (example 04: 11.7 MB of
  JSONL against 744 KB of parquet). One line per settled (stint ×
  domain), so it scales with gangs × domains spanned, not with jobs.
- **`live.json` is rewritten whole at every flush**, and its open-stint
  overlay grows with *concurrent* jobs — example 04 holds 3,206 open
  stints, a 680 KB rewrite. At 1,440 flushes that would be ~1 GB of file
  writes for an 86-second run, so **the overlay is spooled only while a
  client is actually polling `/live`**: `live_payload` touches
  `live.watch`, the child includes the overlay only when that file is
  fresh (15 s), and `_LiveSpool.finish` always writes it once at the end
  so a client arriving *after* the run still replays the identical open
  set. A state file written without it carries `open_omitted: true`,
  which the payload reports as `open_stints: null` plus
  `open_pending: true` — never as an empty overlay, which would claim
  nothing was running.

With nobody watching, example 04's whole spool is the JSONL plus a
~300-byte state file per flush.

Slugs are **server-generated** —
`run-<UTCdate>-<UTCtime>-<seq>-<rand>` matching
`^run-\d{8}-\d{6}-\d{3}-[a-z0-9]{4}$`; clients never pick ids. A
directory containing a `summary.json` but no `meta.json` (a CLI run
dropped into the workspace) is listed as a done "external" run with the
directory name as its id and title.

The scenario's own `outputs.dir` is **ignored** for web-submitted runs:
the server forces all output into the run directory, so a scenario
(typo'd or hostile) can never choose where the server writes.

**One live server per workspace.** The server takes a
`workspace/.serve.lock` file (owner pid inside) at startup and releases
it on shutdown. A second `fleetsim serve` pointed at the same workspace
— even on a different port — refuses to start (exit 2) instead of
"repairing" the first server's queued/running runs to `failed` while
they are still executing. A lock left behind by a crashed process (dead
pid) is reclaimed automatically.

## Security posture

Local-first, defense in depth — the HTTP surface is treated as untrusted
even though it binds loopback:

- **Loopback bind by default.** `--host` can widen it, prints a loud
  warning, and is on you; the app exposes your runs and accepts scenario
  submissions from anyone who can reach the address.
- **Host-header pin (anti DNS-rebinding).** A rebound attacker domain
  resolves to 127.0.0.1 but sends its own name in `Host`; the server
  answers `421` for any authority that is not the pinned loopback set
  (or the explicit `--host` value), so a hostile page can never become
  same-origin with the app. Wildcard binds (`0.0.0.0`/`::`) disable the
  pin — client names are unknowable there.
- **CSRF rejection on state-changing routes.** `POST`/`DELETE` reject
  any request carrying a foreign `Origin` or a cross-site
  `Sec-Fetch-Site`, and JSON bodies must be
  `Content-Type: application/json` (a non-simple type, so cross-origin
  browsers must preflight — and `OPTIONS` grants no CORS headers, so
  the preflight fails).
- **Bounded validation.** The declared fleet size is checked
  arithmetically from the topology counts *before* any node is
  materialized — a 200-byte YAML declaring billions of nodes is a
  validation error, not an OOM (ceiling: 262,144 nodes / 4,194,304
  chips, shared with `fleetsim validate`).
- **Path containment.** Run ids are server-generated slugs; every id or
  static path from a request is re-validated (no separators, no
  dot-names) and then resolved with
  `Path.resolve().is_relative_to(root)` against the workspace / static
  root. Traversal — plain, percent-encoded, or symlinked — never leaves
  either root (test-enforced with raw-socket request paths).
- **`yaml.safe_load` only.** Scenario text is parsed by the same config
  layer as the CLI; no object construction from YAML tags.
- **Worker-process runs (v0.8).** Simulations execute in
  `ProcessPoolExecutor` workers started by `multiprocessing` — an
  already-running interpreter handed a module-level function and a path
  string. Still no shell, no subprocess of submitted content, no
  user-controlled string reaching an interpreter; the scenario travels as
  a file the worker reads, and `out_dir` is forced to the run directory.
- **Bounded sweeps.** A sweep's cell count is checked on the *computed*
  product (cap 64) before anything is expanded, so a few hundred bytes of
  grid JSON cannot become a combinatorial explosion; sweep ids are
  server-generated and re-validated with the same slug-shape +
  containment gates as run ids.
- **JSON errors only.** Tracebacks never leave the process; internal
  errors surface as `{"error": "..."}` with the exception type, not the
  stack.
- **Strict CSP.** The app shell is pinned to `default-src 'self';
  img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src
  'self'; frame-ancestors 'self'` — no inline scripts, no external
  anything, no framing by foreign pages. The 2D report (a
  self-contained single file whose one inline script *is* the app)
  additionally allows `'unsafe-inline'` while keeping
  `default-src 'self'` and `frame-ancestors 'self'`; its data payload
  is script-escaped at render time.
- **Server-side hygiene.** Bodies capped at 5 MB, no `Server:` version
  fingerprint, quiet request logging.

## The 3D fleet view

Halls are clusters; each pod is a stack of 8–20 node slabs (a
power-of-two chips-per-slab budget keeps the whole fleet inside one
instanced draw call, capped at 40,000 instances). Slabs fill bottom-up
with the classes running on that pod at time T, in the same palette as
the 2D report; idle capacity is near-black. A translucent shell pulses
red/orange around a pod for a short window after a node failure or
drain. Red pulses are scoped to genuine *node-failure* kills (stints
whose end coincides with a node-failure event) — routine job aborts
(`abort_prob`) end stints with the same `failed` reason but do not
pulse. Playback state (time, speed, pin, camera) persists per run
while the tab is open, so switching 2D ↔ 3D never loses your place.

Requires stint output (`outputs: {stints: true}`) in the scenario —
without stints the view shows a notice saying exactly that.

A **running** run renders too, from the live stint stream rather than the
model — see [Live replay in the app](#live-replay-in-the-app-v08). Pods
expand into their nodes on click — see
[Node drill-down](#node-drill-down-v08).

### Controls

| Input | Action |
|---|---|
| drag | orbit |
| wheel / trackpad scroll | zoom |
| shift-drag or right-drag | pan |
| hover a pod | tooltip: pod id, per-class chips, share of chips busy, top resident jobs |
| click a pod | **drill into its nodes** — camera flies in, the pod expands into node slabs, breadcrumb appears; clicking the expanded pod again toggles its side card |
| click a legend key | filter that workload class out (it renders as idle capacity) |
| `Esc` | back out one level: node view first, then the pin |
| `PNG` | download the current frame as a PNG |
| `Copy link` | copy a deep link to this exact moment, angle, pin, drill-down and filters |
| `⟲ live` | (running runs) re-attach the transport to the leading edge |
| `Space` | play / pause |
| `←` / `→` | step one scheduler round |
| `Home` / `End` | jump to start / horizon |
| `1` / `2` / `3` | camera poses: overview / hall / floor (also buttons in the transport bar) |
| speed select | ×1 / ×4 / ×16 / ×64 sim-hours per wall-second |
| scrubber | seek; event ticks underneath jump to preemption waves, failures, frontier-gang launches |

Keyboard shortcuts are ignored while a form control has focus, and only
apply while the 3D view is visible. With `prefers-reduced-motion:
reduce`, pulses become a steady glow and camera poses snap instead of
tweening; there is no ambient camera drift in any mode.

## Troubleshooting

- **`error: cannot bind 127.0.0.1:8500: [Errno 48] Address already in
  use`** — another process (often a previous `fleetsim serve`) owns the
  port. Pick another with `-p`, or stop the other server. The process
  exits 2 without touching the workspace.
- **`error: workspace … is already owned by a running fleetsim serve
  (pid N)`** — one live server per workspace: a second server on the
  same workspace (any port) would corrupt the first one's run state, so
  it refuses to start (exit 2). Use a different `--workspace`, or stop
  the other server. If that pid is really gone, delete
  `workspace/.serve.lock` (a lock whose pid is dead is normally
  reclaimed automatically).
- **`error: fleetsim serve cannot start inside a worker process`** — the
  script calling `serve()` (or building a `RunManager`) is missing an
  `if __name__ == "__main__":` guard. Simulation workers re-execute the
  main module, so an unguarded script would start a second server in each
  one. Add the guard; the `fleetsim serve` command line already has it.
- **"no stints" notice / no fleet map** — the run was executed without
  `outputs: {stints: true}`. The 2D report degrades to fleet-level
  replay and the 3D view shows a notice; re-run the scenario with stints
  on. The starter template and examples 01/04 already set it.
- **Report/model return 409** — the run isn't finished; the UI shows
  live progress until it is. For a `failed` run, `GET /api/runs/{id}`
  carries the error string.
- **Trace scenarios: "trace file not found"** — web-submitted runs
  execute from a fresh server-named directory, so a *relative*
  `workload.source` can never resolve (the bundled `02_trace_replay`
  example hits this by design — it is written for CLI use, where the
  path is relative to the scenario file). Use an **absolute path** to
  the trace CSV when submitting through the web app; validation reports
  this up front.
- **Big runs** — the 2D model is bounded by construction (≤ 1200
  timeline frames, ≤ 300 gantt rows; stints are *not* downsampled), and
  a frontier-scale run's report lands well under 20 MB. The first
  `model`/`report` request after a run does the build and can take
  seconds — it is disk-cached after that. More than 512 map domains
  auto-aggregates the 2D map to clusters; the 3D view compresses
  chips-per-slab instead. Up to `--workers` runs execute at once (default
  `min(4, cpu_count-1)`); further submissions queue FIFO and report their
  position.
- **Cancelling a runaway run** — the progress panel's **Cancel run**
  button (or `POST /api/runs/{id}/cancel`) stops the *running* run at
  its next metrics flush and marks it `failed (cancelled by request)`;
  queued submissions behind it then start. Use it when a mistyped
  horizon or an oversized fleet would otherwise occupy a worker for
  hours.
- **Deleting runs** — only *queued* runs can be deleted through the API
  (the × on the rail row). Anything that ever ran is immutable history
  to the server: delete its directory on disk while the server is down.
- **Empty template dropdown** — the examples endpoint serves from the
  repo checkout; an installed wheel without the checkout returns `[]`
  (the built-in starter template still works).
- **The live 3D view says "waiting for the first metrics flush"** — it
  is: nothing is written until the simulator completes a scheduler round.
  A long `sim.round` means a long wait. If it never appears and the
  progress panel is updating, the scenario has no `outputs.stints` and
  there is no fleet to draw.
- **"node fill is a LAYOUT"** in the drill-down breadcrumb — fleetsim
  records placement per domain at the stint level, so the per-node split
  is a deterministic packing, not a measurement. Record at node level
  (`outputs: {stints: node}`, when your topology has a `node` level) and
  the drill-down becomes exact — the breadcrumb then says so.
- **PNG export produces a blank or missing file** — a chart PNG is
  rasterized through a `data:` URI; a browser that refuses shows
  "no PNG" on the button instead of failing silently. The 3D export
  needs a live WebGL context: if the tab was backgrounded long enough for
  the browser to drop it, reopen the view and export again.
- **A deep link opens at the default view** — every field is validated
  independently, so a truncated or hand-edited link silently drops the
  parts it cannot parse rather than failing the navigation. Re-copy it
  from the view.
