# The web app (`fleetsim serve`, v0.5)

`fleetsim serve` starts a **local** web app around the run/viz pipeline:
browse every run in a workspace, open any finished run as the
interactive 2D report, submit new scenario YAML from an editor and watch
live progress, and replay the fleet in a **three.js 3D view** — pods as
stacks of node slabs, colored by workload class, failure and drain
pulses included.

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
fleetsim serve [-p PORT] [--workspace DIR] [--host H] [--open]
```

| Flag | Default | Meaning |
|---|---|---|
| `-p`, `--port` | `8500` | port to bind |
| `--workspace` | `./fleetsim-runs` | run workspace directory, created if missing |
| `--host` | `127.0.0.1` | bind address — **non-loopback values expose the app to your network** and print a loud warning; prefer the default plus an SSH tunnel |
| `--open` | off | open the app in the default browser |

Ctrl-C shuts down cleanly: queued runs are cancelled (marked `failed`
with a clear error), the active run is aborted cooperatively at its next
metrics flush, and the process exits 0 — a long simulation can never
hold the terminal hostage.

Python API, same server:

```python
from fleetsim.serve import serve
serve(port=8500, workspace="./fleetsim-runs")   # blocks until Ctrl-C
```

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
  Runs execute one at a time, FIFO, in-process — submitting several
  queues them.
- **Run view** (`#run/<id>/report`): while the run is queued/running, a
  live progress panel (sim time, occupancy to date, jobs
  finished/running/pending, chips allocated — one update per scheduler
  round, polled every 1 s) with a **Cancel run** button while it
  executes; when it finishes, the full **2D report**
  (the same self-contained HTML `fleetsim viz` writes) in an iframe,
  plus a **Download report.html** button — the downloaded file works
  from `file://`, email, anywhere.
- **3D fleet** (`#run/<id>/fleet3d`): the showcase view, below.

Tip: give scenarios `outputs: {stints: true}` — the fleet map in the 2D
report and the entire 3D view replay `stints.parquet`. Without it both
degrade honestly and tell you what to add. The starter template sets it.
(`true` records at the level below the cluster root, which is what the
fleet views want; a level name like `stints: pod` also works, but only
when the scenario's tree actually has a level with that name.)

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
| `GET /api/runs` | `[{id, title, status, created, headline, error?}]` newest first; `status` ∈ `queued\|running\|done\|failed`; `headline` = `{occupancy, goodput, jobs_finished}` for done runs, else `null` |
| `GET /api/runs/{id}` | the row's meta plus `summary: <full summary.json>` when done (`null` otherwise); 404 unknown id |
| `GET /api/runs/{id}/progress` | `{status, progress}`; `progress` = `{t_us, horizon_us, jobs_finished, jobs_running, pending, occupancy_to_date, allocated_chips, healthy_chips}`, one snapshot per metrics flush, final snapshot at `t_us == horizon_us`; `null` before the first flush, for external runs, and for runs finished in a previous server session |
| `GET /api/runs/{id}/model` | the viz JSON model (`fleetsim.viz.build_viz_model`; schema in `src/fleetsim/viz/data.py`), disk-cached as `viz_model.json`; 409 until the run is done |
| `GET /api/runs/{id}/report` | the self-contained 2D report HTML, disk-cached as `report.html`; 409 until done |
| `POST /api/validate` | body `{yaml: str}` → always 200 `{ok, errors: [str]}` for a well-formed request; 400 for a bad envelope |
| `POST /api/runs` | body `{yaml: str, title?: str}` → 200 `{id}`; 400 `{ok: false, errors}` for an invalid scenario |
| `DELETE /api/runs/{id}` | 200 `{ok: true}` for **queued** runs only (dequeue); 409 for running/done/failed, 404 unknown |
| `POST /api/runs/{id}/cancel` | 200 `{ok: true}` for **running** runs: cooperative cancel — the run stops at its next metrics flush and is marked `failed` with `cancelled by request`; 409 for any other status, 404 unknown |
| `GET /api/examples` | `[{name, yaml, runnable, note?}]` — the bundled starter scenarios, read-only, sorted; `yaml` is verbatim, `runnable: false` (+ human `note`) marks a starter that cannot run as web-submitted (`[]` on an installed wheel without the repo checkout) |
| `GET /`, `GET /static/*` | the app shell (packaged static files, no build step) |

## Workspace layout

Each managed run is one directory, `workspace/<slug>/`:

| File | Written | Contents |
|---|---|---|
| `scenario.yaml` | at submit | the submitted scenario text, verbatim |
| `meta.json` | at submit, updated on transitions | `{title, created_unix, status, error?}` |
| `summary.json`, `jobs.parquet`, `timeseries.parquet` (+ `stints.parquet`) | at completion | the standard `fleetsim run` outputs |
| `viz_model.json`, `report.html` | lazily, on first request | disk caches of the model/report endpoints (runs are immutable once done, so they never invalidate) |

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
- **In-process runs.** Simulations execute in a worker thread via the
  Python API — no subprocess, no shell, no user-controlled string ever
  reaches an interpreter.
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

### Controls

| Input | Action |
|---|---|
| drag | orbit |
| wheel / trackpad scroll | zoom |
| shift-drag or right-drag | pan |
| hover a pod | tooltip: pod id, per-class chips, share of chips busy, top resident jobs |
| click a pod | pin it — opens a side card with the same detail (Esc or ✕ unpins) |
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
  chips-per-slab instead. Runs execute **one at a time** (the simulator
  is CPU-bound pure Python; parallel runs would timeslice the GIL and
  finish later) — queued submissions wait their turn and say so.
- **Cancelling a runaway run** — the progress panel's **Cancel run**
  button (or `POST /api/runs/{id}/cancel`) stops the *running* run at
  its next metrics flush and marks it `failed (cancelled by request)`;
  queued submissions behind it then start. Use it when a mistyped
  horizon or an oversized fleet would otherwise hold the single FIFO
  worker for hours.
- **Deleting runs** — only *queued* runs can be deleted through the API
  (the × on the rail row). Anything that ever ran is immutable history
  to the server: delete its directory on disk while the server is down.
- **Empty template dropdown** — the examples endpoint serves from the
  repo checkout; an installed wheel without the checkout returns `[]`
  (the built-in starter template still works).
