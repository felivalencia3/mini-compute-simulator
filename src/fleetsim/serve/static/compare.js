/* fleetsim v0.8 — the compare view (#compare/<id>,<id>,…).

   Lazily imported by app.js.  Owns everything under #compareBody:

     1. a RUN LEGEND: one chip per run — letter, line color, title link,
        3D link, status;
     2. a METRIC MATRIX: rows = runs, columns = occupancy / allocation /
        goodput / ETTR p50 / preemptions per minute / jobs finished /
        queue-wait p50+p99 per workload class.  Every cell carries the
        exact summary.json key path it came from in its title, and the
        best value in a column is subtly marked (columns where "best" is
        not defined — jobs finished — are deliberately left unmarked);
     3. OVERLAID TIMELINES: occupancy, pending jobs (all classes), and
        preemption rate, one line per run, direct-labelled at the line
        end, with a shared crosshair readout (pointer AND keyboard);
     4. a CONFIG DIFF: only the scenario keys that actually differ across
        the selected runs, from GET /api/runs/{id}/scenario.

   DATA SOURCES.  Metrics come from each run's summary.json
   (GET /api/runs/{id}) — never recomputed here, so a number on this
   screen and the same number in the run's own report cannot disagree.
   Timelines come from the run's viz model frames
   (GET /api/runs/{id}/model), which is the same downsampled series the
   2D report draws.  Scenario keys come from the flattened scenario doc.

   COLOR.  Runs use their OWN categorical palette (see RUN_COLORS), kept
   away from the workload-class palette so a run line is never read as a
   class; identity is additionally carried by a letter (A, B, C…) and by
   direct labels, never by color alone.  Class names in the matrix are
   deliberately plain text — no class color chips — so the two palettes
   never share a surface.

   All dynamic text goes through textContent; SVG is built with
   createElementNS.  No markup interpolation anywhere. */

import { pngButton, safeName } from "./export.js";

"use strict";

const US = 1e6;

/* Categorical run palette: the eight-hue dark-mode scale from the
   dataviz reference (validated on this app's panel surface #11151d —
   lightness band, chroma floor, adjacent CVD ΔE >= 8, normal-vision
   ΔE >= 15, contrast >= 3:1).  Assigned in fixed slot order, never
   cycled: MAX_COMPARE in app.js is exactly this length. */
const RUN_COLORS = [
  "#3987e5", // blue
  "#d95926", // orange
  "#199e70", // aqua
  "#c98500", // yellow
  "#d55181", // magenta
  "#008300", // green
  "#9085e9", // violet
  "#e66767", // red
];
const LETTERS = "ABCDEFGH";

/** One line per color, never a cycled hue: a hand-typed #compare URL with
    more ids than this shows the first MAX_RUNS and says so. */
const MAX_RUNS = RUN_COLORS.length;

const MUTED = "#8b93a1";
const GRID = "rgba(255,255,255,.07)";
const SVG_NS = "http://www.w3.org/2000/svg";

/* ------------------------------------------------------------------ *
 * formatting (kept local: this module is self-contained, like fleet3d)
 * ------------------------------------------------------------------ */
const isNum = (v) => typeof v === "number" && isFinite(v);

function fmtClock(us) {
  const s = Math.floor(us / US);
  const d = Math.floor(s / 86400);
  const hh = Math.floor((s % 86400) / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const p = (n) => String(n).padStart(2, "0");
  return (d ? d + "d " : "") + p(hh) + ":" + p(mm);
}
const fmtPct = (v) => (isNum(v) ? (v * 100).toFixed(1) + "%" : "–");
const fmtNum = (v, n) => (isNum(v) ? v.toFixed(n) : "–");
const fmtInt = (v) => (isNum(v) ? String(Math.round(v)) : "–");

function fmtSecs(v) {
  if (!isNum(v)) return "–";
  if (v < 90) return v.toFixed(1) + "s";
  if (v < 5400) return (v / 60).toFixed(1) + "m";
  return (v / 3600).toFixed(2) + "h";
}

function eln(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function svgEl(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) n.setAttribute(k, String(attrs[k]));
  return n;
}

/** WCAG relative luminance of a #rrggbb color. */
function luminance(hex) {
  const ch = [1, 3, 5]
    .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
  return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
}

/** The run's letter badge, inked with whichever of near-black / white has
    MORE contrast on that slot (a fixed threshold leaves the mid-lightness
    hues at ~3.2:1).  The letter carries the run's identity, so it has to
    stay legible in all eight slots; measured worst case 3.9:1. */
function runKey(entry) {
  const key = eln("span", "runkey", entry.letter);
  key.style.background = entry.color;
  const y = luminance(entry.color);
  const onDark = (y + 0.05) / (0.00435 + 0.05); // ink #0b0e14
  const onWhite = 1.05 / (y + 0.05); // ink #ffffff
  key.style.color = onDark >= onWhite ? "#0b0e14" : "#ffffff";
  return key;
}

/** Nested lookup by key path; null for any missing/non-object hop. */
function dig(obj, path) {
  let cur = obj;
  for (const key of path) {
    if (cur == null || typeof cur !== "object") return null;
    cur = cur[key];
  }
  return cur === undefined ? null : cur;
}

async function getJSON(path) {
  let resp;
  try {
    resp = await fetch(path);
  } catch (err) {
    return { ok: false, status: 0, doc: null };
  }
  let doc = null;
  try {
    doc = await resp.json();
  } catch (err) {
    /* non-JSON body: treated as no document */
  }
  return { ok: resp.ok, status: resp.status, doc };
}

/* ------------------------------------------------------------------ *
 * module state
 * ------------------------------------------------------------------ */
let token = 0; // stale-async guard: every load bumps it
let currentIds = [];
let entries = []; // [{id, letter, color, info, scen, frames}]
let pollTimer = null;
let wired = false;
let showAllKeys = false;
let dropped = 0; // ids beyond MAX_RUNS that the route asked for
/** Run ids hidden from the overlaid timelines.  At realistic frame
    counts an overlay is a hairball in which the last-drawn colour paints
    over the rest, so isolating one or two runs is the difference between
    a comparison and a probe. */
const hiddenSeries = new Set();

/* Immutable once a run is done, so cached for the whole session and
   never re-fetched by the poll loop. */
const scenCache = new Map(); // id -> {flat, name, parse_error} | {missing}
const framesCache = new Map(); // id -> frames | null

export function unmountCompare() {
  token++;
  stopPoll();
}

function stopPoll() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function wire() {
  if (wired) return;
  wired = true;
  const scope = document.querySelector("#scopeSelect");
  if (scope) scope.addEventListener("change", () => render());
  const refresh = document.querySelector("#compareRefresh");
  if (refresh) {
    refresh.addEventListener("click", () => {
      framesCache.clear();
      scenCache.clear();
      load(currentIds);
    });
  }
}

export function mountCompare(requested) {
  wire();
  const ids = requested.slice(0, MAX_RUNS);
  const same =
    ids.length === currentIds.length && ids.every((v, i) => v === currentIds[i]);
  if (same && entries.length) {
    render(); // returning to a view we already hold: no refetch
    schedulePoll();
    return;
  }
  load(requested);
}

async function load(requested) {
  const mine = ++token;
  stopPoll();
  dropped = Math.max(0, requested.length - MAX_RUNS);
  const ids = requested.slice(0, MAX_RUNS);
  currentIds = ids.slice();
  const body = document.querySelector("#compareBody");
  if (!ids.length) {
    entries = [];
    showEmpty("No runs selected. Pick two or more in the rail, then Compare.");
    return;
  }
  if (!entries.length) {
    body.textContent = "";
    body.appendChild(panelNote("Loading " + ids.length + " runs…"));
  }
  const loaded = await Promise.all(ids.map((id) => loadOne(id)));
  if (mine !== token) return; // route moved on
  entries = loaded.map((e, i) => ({
    ...e,
    letter: LETTERS[i] || "?",
    color: RUN_COLORS[i % RUN_COLORS.length],
  }));
  render();
  schedulePoll();
}

async function loadOne(id) {
  const enc = encodeURIComponent(id);
  const info = await getJSON("/api/runs/" + enc);
  const out = { id, info: info.ok ? info.doc : null, status: info.status };
  if (!scenCache.has(id)) {
    const scen = await getJSON("/api/runs/" + enc + "/scenario");
    scenCache.set(id, scen.ok ? scen.doc : { missing: true });
  }
  out.scen = scenCache.get(id);
  const done = out.info && out.info.status === "done";
  if (done && !framesCache.has(id)) {
    const model = await getJSON("/api/runs/" + enc + "/model");
    // keep ONLY the frames: a full model carries every stint, and eight
    // of them would sit in memory for the whole session
    framesCache.set(id, model.ok && model.doc ? model.doc.frames || null : null);
  }
  out.frames = done ? framesCache.get(id) || null : null;
  return out;
}

function schedulePoll() {
  stopPoll();
  const pending = entries.some(
    (e) => e.info && (e.info.status === "queued" || e.info.status === "running")
  );
  if (!pending) return;
  const mine = token;
  pollTimer = setTimeout(() => {
    pollTimer = null;
    if (mine === token) load(currentIds);
  }, 4000);
}

function panelNote(text) {
  const p = eln("div", "panel");
  p.appendChild(eln("p", "sub", text));
  return p;
}

function showEmpty(text) {
  const body = document.querySelector("#compareBody");
  body.textContent = "";
  body.appendChild(panelNote(text));
  document.querySelector("#compareTitle").textContent = "Compare";
  document.querySelector("#compareMeta").textContent = "";
}

/* ------------------------------------------------------------------ *
 * render
 * ------------------------------------------------------------------ */
function scopeName() {
  const sel = document.querySelector("#scopeSelect");
  return sel && sel.value === "full" ? "full" : "window";
}

function render() {
  const body = document.querySelector("#compareBody");
  if (!entries.length) return;
  const scope = scopeName();
  document.querySelector("#compareTitle").textContent =
    "Compare " + entries.length + " runs";
  const nDone = entries.filter((e) => e.info && e.info.status === "done").length;
  document.querySelector("#compareMeta").textContent =
    nDone + " of " + entries.length + " finished · " +
    (scope === "window" ? "steady-state window" : "full run");

  body.textContent = "";
  body.appendChild(runLegend());
  body.appendChild(matrixPanel(scope));
  body.appendChild(timelinePanel());
  body.appendChild(diffPanel());
}

/** A sweep cell's grid values as a short label. */
function cellLabel(cell) {
  if (!cell || typeof cell !== "object") return "";
  return Object.keys(cell)
    .map((k) => k.split(".").pop() + "=" + JSON.stringify(cell[k]))
    .join(", ");
}

/** The SHORT label for a run in this view: its sweep cell when it has
    one (that is what distinguishes cells), else its title. */
function runLabel(e) {
  const cell = cellLabel(e.info && e.info.sweep_cell);
  return cell || (e.info && e.info.title) || e.id;
}

/** Everything about the run, for a title attribute on a truncated label. */
function runFullLabel(e) {
  const parts = [];
  if (e.info && e.info.title) parts.push(e.info.title);
  const cell = cellLabel(e.info && e.info.sweep_cell);
  if (cell) parts.push(cell);
  parts.push(e.id);
  return parts.join("\n");
}

function statusText(e) {
  if (!e.info) return e.status === 404 ? "not in this workspace" : "unreachable";
  if (e.info.status === "failed") return "failed: " + (e.info.error || "no error recorded");
  if (e.info.status === "queued") {
    return e.info.queue_position != null
      ? "queued (#" + e.info.queue_position + ")"
      : "queued";
  }
  return e.info.status;
}

function runLegend() {
  const panel = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, "Runs"));
  head.appendChild(
    eln("span", "sub", "Line color and letter identify the run in every panel below.")
  );
  panel.appendChild(head);

  const list = eln("div", "runchips");
  for (const e of entries) {
    const chip = eln("div", "runchip");
    chip.appendChild(runKey(e));

    const mid = eln("span", "runchipmain");
    const title = eln("a", "runchiptitle");
    title.href = "#run/" + encodeURIComponent(e.id) + "/report";
    /* PREFER THE SWEEP CELL.  A sweep's generated titles are 600+ px of
       identical prefix that differ only in the tail this 320 px box
       ellipsises, so four cells rendered as four copies of one string.
       The cell label leads with what actually varies. */
    title.textContent = runLabel(e);
    title.title = runFullLabel(e);
    mid.appendChild(title);
    const meta = eln("span", "sub mono", statusText(e));
    mid.appendChild(meta);
    chip.appendChild(mid);

    // no links for an id that does not resolve: they would only lead to
    // the "No such run" panel
    if (e.info) {
      const links = eln("span", "runchiplinks");
      const rep = eln("a", "minilink", "report");
      rep.href = "#run/" + encodeURIComponent(e.id) + "/report";
      rep.title = "Open this run's own 2D report";
      links.appendChild(rep);
      const three = eln("a", "minilink", "3D");
      three.href = "#run/" + encodeURIComponent(e.id) + "/fleet3d";
      three.title = "Open this run's 3D fleet replay";
      links.appendChild(three);
      chip.appendChild(links);
    }
    list.appendChild(chip);
  }
  panel.appendChild(list);
  if (dropped) {
    panel.appendChild(
      eln(
        "p",
        "sub",
        "Showing the first " + MAX_RUNS + " of " + (MAX_RUNS + dropped) +
          " ids in the link: the run palette holds " + MAX_RUNS +
          " lines, and a ninth would have to reuse a color."
      )
    );
  }
  return panel;
}

/* ---- metric matrix --------------------------------------------------- */

/** Fixed columns; `dir` null means "best is not defined here". */
const BASE_COLS = [
  {
    label: "occupancy",
    path: ["occupancy"],
    dir: "high",
    fmt: fmtPct,
    help: "allocated chip-time / healthy chip-time",
  },
  {
    label: "allocation",
    path: ["allocation_rate"],
    dir: "high",
    fmt: fmtPct,
    help: "allocated chip-time / total fleet chip-time",
  },
  {
    label: "goodput",
    path: ["goodput"],
    dir: "high",
    fmt: fmtPct,
    help: "productive chip-time / allocated chip-time",
  },
  {
    label: "ETTR p50",
    path: ["ettr", "job_weighted", "p50"],
    dir: "high",
    fmt: (v) => fmtNum(v, 3),
    help: "median job productive/elapsed ratio (job-weighted)",
  },
  {
    label: "preempt/min",
    path: ["preemptions_per_min", "total"],
    dir: "low",
    fmt: (v) => fmtNum(v, 2),
    help: "preemptions + node-failure kills per minute",
  },
  {
    label: "jobs finished",
    path: ["counts", "jobs_finished"],
    dir: null,
    fmt: fmtInt,
    help: "terminal jobs in scope — throughput follows the offered load, so no best is marked",
  },
];

/** Fragmentation columns, one set per fleet LEVEL the runs recorded.
 *
 * "Does tighter gang packing reduce fragmentation?" is one of the two
 * questions this whole view exists to answer, and it was unanswerable
 * here: a placement sweep is FLAT on occupancy (69.4 % / 69.4 %) while
 * the same runs' summary.json carries fragmentation.node.mean 0.936 and
 * stranded_whole_nodes.mean 13.63 — the v0.7 diagnostic added for
 * exactly this. Reading occupancy alone reports "placement changes
 * nothing", which is the wrong conclusion from the right data.
 *
 * Levels are discovered from the runs themselves: `fragmentation` is
 * keyed by level name, plus the reserved `stranded_whole_nodes` key that
 * only appears when the scenario named a placement policy.
 */
function fragColumns(scope) {
  const levels = new Set();
  let stranded = false;
  for (const e of entries) {
    const table = dig(e.info, ["summary", scope, "fragmentation"]);
    if (!table || typeof table !== "object") continue;
    for (const key of Object.keys(table)) {
      if (key === "stranded_whole_nodes") stranded = true;
      else levels.add(key);
    }
  }
  const cols = [];
  for (const level of [...levels].sort()) {
    cols.push({
      label: "frag " + level,
      group: "fragmentation",
      path: ["fragmentation", level, "mean"],
      dir: "low",
      fmt: (v) => fmtNum(v, 3),
      help:
        "mean " + level + "-level fragmentation index over flush samples" +
        " (0 = every free chip is in one place, 1 = maximally scattered)",
    });
  }
  if (stranded) {
    cols.push({
      label: "stranded nodes",
      group: "fragmentation",
      path: ["fragmentation", "stranded_whole_nodes", "mean"],
      dir: "low",
      fmt: (v) => fmtNum(v, 2),
      help:
        "mean count of partially-occupied healthy leaves — whole nodes no" +
        " gang can use because something small is sitting on them" +
        " (recorded only when the scenario named a placement policy)",
    });
  }
  return cols;
}

function classColumns(scope) {
  const names = new Set();
  for (const e of entries) {
    const table = dig(e.info, ["summary", scope, "queue_wait_s_by_source_class"]);
    if (table && typeof table === "object") {
      for (const k of Object.keys(table)) names.add(k);
    }
  }
  const cols = [];
  for (const cls of [...names].sort()) {
    for (const q of ["p50", "p99"]) {
      cols.push({
        label: cls + " " + q,
        group: cls,
        path: ["queue_wait_s_by_source_class", cls, "job_weighted", q],
        dir: "low",
        fmt: fmtSecs,
        help: "queue wait " + q + " for workload class " + cls + " (job-weighted)",
      });
    }
  }
  return cols;
}


/** A horizontally scrollable table region.
 *
 * Containment alone hid columns: with macOS overlay scrollbars nothing is
 * drawn until you happen to scroll over the region, so a clipped column
 * ("ENDED" rendering as "ENDEI") reads as the end of the table. The CSS
 * keeps a thin scrollbar visible always; this makes the region focusable
 * and named, so it is reachable and announced rather than only mousable.
 */
function tableWrap(label) {
  const wrap = eln("div", "tablewrap");
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "region");
  wrap.setAttribute("aria-label", label + " (scrolls sideways)");
  return wrap;
}

function matrixPanel(scope) {
  const panel = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, "Metric matrix"));
  head.appendChild(
    eln(
      "span",
      "sub",
      "Every cell reads straight from that run's summary.json — hover a cell for its exact key path."
    )
  );
  panel.appendChild(head);

  const cols = BASE_COLS.concat(fragColumns(scope), classColumns(scope));
  const rows = entries.map((e) => ({
    entry: e,
    summary: dig(e.info, ["summary", scope]),
  }));

  // Best per column, but only where the mark can be READ OFF THE SCREEN:
  // a direction has to be defined, at least two runs have to answer, and
  // the winner has to be unique AT DISPLAY PRECISION.  Otherwise the
  // column is left unmarked — a green cell among three identical-looking
  // numbers (a 1.0000 vs 0.9999 tie) reads as a bug, not as a finding.
  const best = cols.map((col) => {
    if (!col.dir) return null;
    const vals = rows.map((r) => dig(r.summary, col.path)).filter((v) => isNum(v));
    if (vals.length < 2) return null;
    const win = col.dir === "high" ? Math.max(...vals) : Math.min(...vals);
    const shown = vals.map((v) => col.fmt(v));
    if (shown.filter((s) => s === col.fmt(win)).length !== 1) return null;
    return win;
  });

  const wrap = tableWrap("metric matrix");
  const table = eln("table", "matrix");
  const thead = eln("thead");

  // two header rows: group labels over the per-class blocks, then the
  // column names.  The first cell of the group row is a spacer above the
  // run column (the "run" header itself lives in the second row).
  const groupRow = eln("tr", "grouprow");
  const spacer = eln("th", "runcol", "");
  spacer.setAttribute("aria-hidden", "true");
  groupRow.appendChild(spacer);
  let i = 0;
  while (i < cols.length) {
    const g = cols[i].group || null;
    let span = 1;
    while (i + span < cols.length && (cols[i + span].group || null) === g) span++;
    const label = !g
      ? ""
      : g === "fragmentation"
        ? "fragmentation"
        : "queue wait · " + g;
    const th = eln("th", "groupth", label);
    th.colSpan = span;
    if (g === "fragmentation") {
      th.title =
        "how scattered the free capacity is — the placement question," +
        " which occupancy alone cannot answer";
    } else if (g) {
      th.title = "job-weighted queue wait for class " + g;
    }
    groupRow.appendChild(th);
    i += span;
  }
  thead.appendChild(groupRow);

  const headRow = eln("tr");
  const runTh = eln("th", "runcol", "run");
  runTh.scope = "col";
  headRow.appendChild(runTh);
  for (const col of cols) {
    const th = eln("th", null, col.label);
    th.setAttribute("scope", "col");
    th.title = col.help + (col.dir ? " · " + col.dir + "er is better" : "");
    if (col.dir) {
      const arrow = eln("span", "dirmark", col.dir === "high" ? "▲" : "▼");
      arrow.setAttribute("aria-hidden", "true");
      th.appendChild(arrow);
    }
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = eln("tbody");
  for (const r of rows) {
    const tr = eln("tr");
    const th = eln("th", "runcol");
    th.setAttribute("scope", "row");
    const cell = eln("span", "runcell");
    cell.appendChild(runKey(r.entry));
    const link = eln("a", "runchiptitle");
    link.href = "#run/" + encodeURIComponent(r.entry.id) + "/report";
    link.textContent = runLabel(r.entry);
    link.title = runFullLabel(r.entry);
    cell.appendChild(link);
    th.appendChild(cell);
    tr.appendChild(th);

    if (!r.summary) {
      const td = eln("td", "nosummary", statusText(r.entry) + " — no summary yet");
      td.colSpan = cols.length;
      tr.appendChild(td);
      tbody.appendChild(tr);
      continue;
    }
    cols.forEach((col, ci) => {
      const raw = dig(r.summary, col.path);
      const td = eln("td", "num", col.fmt(raw));
      td.title =
        "summary.json · " + scope + "." + col.path.join(".") +
        " = " + (raw == null ? "null" : String(raw));
      if (best[ci] != null && isNum(raw) && raw === best[ci]) {
        td.classList.add("best");
        td.appendChild(eln("span", "sronly", " (best of these runs)"));
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  panel.appendChild(wrap);

  const legend = eln("p", "sub");
  legend.textContent =
    "▲ higher is better · ▼ lower is better · a marked cell is the best of these runs" +
    " · “–” means the run reported no value for that key (e.g. a class it never ran).";
  panel.appendChild(legend);
  return panel;
}

/* ---- overlaid timelines ---------------------------------------------- */

/** Per-frame preemption rate (per minute) from the cumulative deltas. */
function preemptRate(frames) {
  const t = frames.t_us || [];
  const d = frames.preemptions_delta || [];
  const out = [];
  for (let i = 0; i < t.length; i++) {
    const prev = i === 0 ? 0 : t[i - 1];
    const dtMin = (t[i] - prev) / 6e7;
    out.push(dtMin > 0 && isNum(d[i]) ? d[i] / dtMin : null);
  }
  return out;
}

/** Total pending jobs per frame (sum of the per-class series). */
function pendingTotal(frames) {
  const byClass = frames.pending_by_class || {};
  const n = (frames.t_us || []).length;
  const out = new Array(n).fill(0);
  for (const key of Object.keys(byClass)) {
    const arr = byClass[key] || [];
    for (let i = 0; i < n; i++) if (isNum(arr[i])) out[i] += arr[i];
  }
  return out;
}

/** Series toggles for the timelines: click a letter to hide/show it,
    with an "only" affordance for isolating one run. */
function seriesToggles(plotted) {
  const row = eln("div", "serieskeys");
  row.setAttribute("role", "group");
  row.setAttribute("aria-label", "show or hide a run in the timelines below");
  for (const e of plotted) {
    const on = !hiddenSeries.has(e.id);
    const btn = eln("button", "serieskey" + (on ? "" : " off"));
    btn.type = "button";
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    const swatch = eln("i");
    swatch.style.background = e.color;
    btn.appendChild(swatch);
    btn.appendChild(document.createTextNode(e.letter + " " + runLabel(e)));
    btn.title =
      (on ? "Hide " : "Show ") + runFullLabel(e) +
      "\n(shift-click to show only this run)";
    btn.addEventListener("click", (ev) => {
      if (ev.shiftKey) {
        hiddenSeries.clear();
        for (const other of plotted) {
          if (other.id !== e.id) hiddenSeries.add(other.id);
        }
      } else if (on) {
        hiddenSeries.add(e.id);
      } else {
        hiddenSeries.delete(e.id);
      }
      render();
    });
    row.appendChild(btn);
  }
  if (hiddenSeries.size) {
    const all = eln("button", "serieskey showall");
    all.type = "button";
    all.textContent = "show all";
    all.addEventListener("click", () => {
      hiddenSeries.clear();
      render();
    });
    row.appendChild(all);
  }
  return row;
}

function timelinePanel() {
  const panel = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, "Overlaid timelines"));
  head.appendChild(
    eln("span", "sub", "One line per run, labelled at its right end. Hover or focus a chart and use ← → for the readout.")
  );
  panel.appendChild(head);

  const finished = entries.filter((e) => e.frames && (e.frames.t_us || []).length);
  const plotted = finished.filter((e) => !hiddenSeries.has(e.id));
  if (finished.length > 1) panel.appendChild(seriesToggles(finished));
  if (finished.length && !plotted.length) {
    panel.appendChild(
      eln("p", "sub", "Every run is hidden — use “show all” above.")
    );
    return panel;
  }
  if (!plotted.length) {
    panel.appendChild(
      eln(
        "p",
        "sub",
        "No finished run among these yet — timelines appear as runs complete."
      )
    );
    return panel;
  }
  if (finished.length < entries.length) {
    panel.appendChild(
      eln(
        "p",
        "sub",
        finished.length + " of " + entries.length +
          " runs are finished; the charts show those."
      )
    );
  }

  const ends = plotted.map((e) => Math.max(...e.frames.t_us));
  const xMax = Math.max(...ends);
  const xMin = Math.min(...ends);
  // comparing a 2 h run with a 14 d one on one absolute axis is honest but
  // surprising — say so rather than letting the short run look empty
  if (xMin > 0 && xMax / xMin >= 2) {
    panel.appendChild(
      eln(
        "p",
        "sub",
        "These runs cover different spans (" + fmtClock(xMin) + " … " +
          fmtClock(xMax) + "). The x axis is absolute simulated time, so the" +
          " shorter runs end early rather than being stretched."
      )
    );
  }

  panel.appendChild(
    chart({
      title: "Occupancy",
      unit: "allocated / healthy chips",
      xMax,
      yMax: 1,
      yFmt: fmtPct,
      series: plotted.map((e) => ({
        letter: e.letter,
        color: e.color,
        name: runLabel(e),
        t: e.frames.t_us,
        v: e.frames.occupancy || [],
      })),
    })
  );
  panel.appendChild(
    chart({
      title: "Pending jobs",
      unit: "all workload classes",
      xMax,
      yMax: null,
      intAxis: true,
      yFmt: fmtInt,
      series: plotted.map((e) => ({
        letter: e.letter,
        color: e.color,
        name: runLabel(e),
        t: e.frames.t_us,
        v: pendingTotal(e.frames),
      })),
    })
  );
  panel.appendChild(
    chart({
      title: "Preemption rate",
      unit: "preemptions per minute",
      xMax,
      yMax: null,
      yFmt: (v) => fmtNum(v, 2),
      series: plotted.map((e) => ({
        letter: e.letter,
        color: e.color,
        name: runLabel(e),
        t: e.frames.t_us,
        v: preemptRate(e.frames),
      })),
    })
  );
  /* FRAGMENTATION over time — the placement question's own series.  A
     placement sweep is flat on occupancy and moves entirely here, so
     leaving it out made "does tighter packing reduce fragmentation?"
     unanswerable in the one view built for comparing runs. */
  const fragLevel = commonFragLevel(plotted);
  if (fragLevel) {
    panel.appendChild(
      chart({
        title: "Fragmentation index · " + fragLevel,
        unit:
          "0 = free chips in one place, 1 = maximally scattered · the level" +
          " these runs differ on most",
        xMax,
        yMax: 1,
        yFmt: (v) => fmtNum(v, 2),
        series: plotted.map((e) => ({
          letter: e.letter,
          color: e.color,
          name: runLabel(e),
          t: e.frames.t_us,
          v: (e.frames.frag_index || {})[fragLevel] || [],
        })),
      })
    );
  }
  return panel;
}

/** Which fleet level to plot the fragmentation index at.
 *
 * Only levels EVERY plotted run recorded (a mixed selection must compare
 * like with like or show nothing), and among those the one with the
 * widest observed range across the runs — i.e. the level these runs
 * actually differ on, which is the whole reason the chart is here. A
 * top-level `metro` index is 0.000 in every run and would answer
 * "placement changes nothing" just as misleadingly as occupancy did.
 * Ties break by name, so the choice is deterministic; the chart names
 * the level it picked.
 */
function commonFragLevel(plotted) {
  let common = null;
  for (const e of plotted) {
    const keys = Object.keys((e.frames && e.frames.frag_index) || {}).sort();
    if (!keys.length) return null;
    common = common === null ? keys : common.filter((k) => keys.includes(k));
    if (!common.length) return null;
  }
  let best = null;
  let bestRange = -1;
  for (const level of common) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const e of plotted) {
      for (const v of e.frames.frag_index[level] || []) {
        if (!isNum(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const range = hi >= lo ? hi - lo : -1;
    if (range > bestRange) {
      bestRange = range;
      best = level;
    }
  }
  return best;
}

const CW = 860; // viewBox width (the SVG scales to its container)
const CH = 168;
const PADL = 52;
const PADR = 42; // room for the direct end labels
const PADT = 12;
const PADB = 22;

function niceMax(v) {
  if (!isNum(v) || v <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (v <= step * pow) return step * pow;
  }
  return 10 * pow;
}

/**
 * Reduce a series to at most `cols` plot columns.
 *
 * Returns `{t, lo, hi, mid, bucketed}`.  When the series already has
 * `<= cols` points nothing is bucketed and the arrays are the originals
 * (`bucketed: false`), so a short run is drawn exactly as before.
 * Otherwise the points are split into `cols` contiguous equal-count
 * buckets — the same rule the viz model uses to build frames — and each
 * bucket contributes its min, max and MEDIAN.  Median, not mean: a
 * single deep dip should not drag the readable line down when the band
 * already shows it.  A bucket with no finite value yields a gap.
 */
function reduceSeries(t, v, cols) {
  const n = Math.min(t.length, v.length);
  if (n <= cols) return { t, lo: v, hi: v, mid: v, bucketed: false };
  const outT = [];
  const lo = [];
  const hi = [];
  const mid = [];
  for (let b = 0; b < cols; b++) {
    const a = Math.floor((n * b) / cols);
    const z = Math.floor((n * (b + 1)) / cols);
    if (z <= a) continue;
    const vals = [];
    for (let i = a; i < z; i++) if (isNum(v[i])) vals.push(v[i]);
    outT.push(t[z - 1]);
    if (!vals.length) {
      lo.push(null);
      hi.push(null);
      mid.push(null);
      continue;
    }
    vals.sort((x, y) => x - y);
    lo.push(vals[0]);
    hi.push(vals[vals.length - 1]);
    mid.push(
      vals.length % 2
        ? vals[(vals.length - 1) / 2]
        : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2
    );
  }
  /* The band path needs a value at every index; a gap column borrows its
     neighbours' so the polygon stays closed, and the LINE (drawn from
     `mid`) still shows the gap. */
  for (let i = 0; i < lo.length; i++) {
    if (lo[i] == null) {
      lo[i] = i > 0 && lo[i - 1] != null ? lo[i - 1] : 0;
      hi[i] = i > 0 && hi[i - 1] != null ? hi[i - 1] : 0;
    }
  }
  return { t: outT, lo, hi, mid, bucketed: true };
}

function chart(spec) {
  const wrap = eln("div", "chart");
  const cap = eln("div", "chartcap");
  cap.appendChild(eln("span", "charttitle", spec.title));
  cap.appendChild(eln("span", "sub", spec.unit));
  const readout = eln("span", "chartreadout mono");
  readout.setAttribute("role", "status");
  cap.appendChild(readout);
  wrap.appendChild(cap);

  let yMax = spec.yMax;
  if (yMax == null) {
    let m = 0;
    for (const s of spec.series) for (const v of s.v) if (isNum(v) && v > m) m = v;
    yMax = niceMax(m);
    // a count axis keeps its midpoint whole (25 -> 26, so the middle
    // gridline reads 13 rather than 12.5 rounded to 13)
    if (spec.intAxis) yMax = 2 * Math.ceil(yMax / 2);
  }
  const xMax = spec.xMax > 0 ? spec.xMax : 1;
  const X = (t) => PADL + (t / xMax) * (CW - PADL - PADR);
  const Y = (v) => CH - PADB - (Math.min(v, yMax) / yMax) * (CH - PADT - PADB);

  const svg = svgEl("svg", {
    viewBox: "0 0 " + CW + " " + CH,
    class: "chartsvg",
    tabindex: "0",
    role: "img",
    "aria-label":
      spec.title + " over simulated time, " + spec.series.length +
      " runs: " + spec.series.map((s) => s.letter + " " + s.name).join(", "),
  });

  // gridlines + y labels (recessive).  A label that formats identically
  // to one already drawn is dropped rather than repeated — an integer
  // series with yMax 1 would otherwise print "1" twice (1 and 0.5).
  const seenLabels = new Set();
  for (const frac of [1, 0.5, 0]) {
    const y = Y(yMax * frac);
    svg.appendChild(
      svgEl("line", { x1: PADL, x2: CW - PADR, y1: y, y2: y, stroke: GRID })
    );
    const text = spec.yFmt(yMax * frac);
    if (seenLabels.has(text)) continue;
    seenLabels.add(text);
    const lab = svgEl("text", {
      x: PADL - 6,
      y: y + 3.5,
      "text-anchor": "end",
      fill: MUTED,
      "font-size": "9.5",
    });
    lab.textContent = text;
    svg.appendChild(lab);
  }
  // x ticks
  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const x = X(xMax * frac);
    const lab = svgEl("text", {
      x,
      y: CH - 7,
      "text-anchor": frac === 0 ? "start" : frac === 1 ? "end" : "middle",
      fill: MUTED,
      "font-size": "9.5",
    });
    lab.textContent = fmtClock(xMax * frac);
    svg.appendChild(lab);
  }

  const cross = svgEl("line", {
    x1: 0,
    x2: 0,
    y1: PADT,
    y2: CH - PADB,
    stroke: "rgba(230,232,235,.45)",
    "stroke-width": "1",
    visibility: "hidden",
  });

  /* DENSITY.  1,200 frames per run in an 860-unit viewBox is ~1.4 units
     per point: every series becomes a solid noise band and the
     last-drawn colour simply paints over the earlier ones. So each
     series is reduced to at most one bucket PER RENDERED UNIT and drawn
     as two marks — a recessive min/max band (the real spread, nothing
     hidden) and the bucket MEDIAN at full opacity (the readable line).
     Below that density nothing is bucketed and the raw line is drawn, so
     short runs are untouched. The crosshair readout still reads the RAW
     series, so a probe never reports a smoothed number. */
  /* One bucket per ~4 rendered units, not per unit: 1,200 frames into 766
     columns is a 1.6:1 reduction and still draws a solid noise band.  At
     ~190 columns the median line is a LINE and the min/max band reads as
     the spread it is. */
  const plotW = Math.max(1, Math.round(CW - PADL - PADR));
  const plotCols = Math.max(48, Math.round(plotW / 4));
  let bucketed = false;
  const endLabels = [];
  for (const s of spec.series) {
    const red = reduceSeries(s.t, s.v, plotCols);
    bucketed = bucketed || red.bucketed;
    if (red.bucketed) {
      let up = "";
      let down = "";
      for (let i = 0; i < red.t.length; i++) {
        up += (up ? "L" : "M") + X(red.t[i]).toFixed(1) + " " + Y(red.hi[i]).toFixed(1);
      }
      for (let i = red.t.length - 1; i >= 0; i--) {
        down += "L" + X(red.t[i]).toFixed(1) + " " + Y(red.lo[i]).toFixed(1);
      }
      if (up) {
        svg.appendChild(
          svgEl("path", {
            d: up + down + "Z",
            fill: s.color,
            "fill-opacity": "0.22",
            stroke: "none",
          })
        );
      }
    }
    // null gaps break the line instead of drawing through them
    let d = "";
    let open = false;
    for (let i = 0; i < red.t.length; i++) {
      const v = red.mid[i];
      if (!isNum(v)) {
        open = false;
        continue;
      }
      d += (open ? "L" : "M") + X(red.t[i]).toFixed(1) + " " + Y(v).toFixed(1);
      open = true;
    }
    if (d) {
      svg.appendChild(
        svgEl("path", {
          d,
          fill: "none",
          stroke: s.color,
          "stroke-width": "2",
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );
      // direct label: the run letter at the line's last point (placed
      // after the loop, so labels that land on one pixel can be nudged
      // apart instead of overprinting each other)
      let last = -1;
      for (let i = s.t.length - 1; i >= 0; i--) {
        if (isNum(s.v[i])) {
          last = i;
          break;
        }
      }
      if (last >= 0) {
        endLabels.push({
          x: Math.min(X(s.t[last]) + 6, CW - 8),
          y: Y(s.v[last]) + 3.5,
          color: s.color,
          letter: s.letter,
        });
      }
    }
  }
  /* De-collide the end labels: sort by y and push each at least 10 units
     below the previous one.  Two runs ending at the same occupancy used
     to print their letters on top of each other, which is exactly the
     case where you most need to tell them apart. */
  endLabels.sort((a, b) => a.y - b.y || a.letter.localeCompare(b.letter));
  let lastY = -Infinity;
  for (const lab of endLabels) {
    const y = Math.max(lab.y, lastY + 10);
    lastY = y;
    const node = svgEl("text", {
      x: lab.x,
      y: Math.min(y, CH - 4),
      fill: lab.color,
      "font-size": "10.5",
      "font-weight": "700",
    });
    node.textContent = lab.letter;
    svg.appendChild(node);
  }
  svg.appendChild(cross);
  if (bucketed) {
    const n = spec.series[0] ? spec.series[0].t.length : 0;
    const cue = eln("span", "sub chartdens");
    cue.textContent =
      n.toLocaleString() + " frames in " + plotCols + " plot columns (~" +
      Math.round(n / plotCols) + " frames each): the BAND is each column's" +
      " min…max — the full spread, nothing hidden — and the line is its" +
      " median. The crosshair still reads the raw frames.";
    cap.appendChild(cue);
  }

  const dots = spec.series.map((s) =>
    svgEl("circle", { r: 3.2, fill: s.color, stroke: "#11151d", "stroke-width": "1.5", visibility: "hidden" })
  );
  for (const dot of dots) svg.appendChild(dot);

  const nearest = (arr, t) => {
    let lo = 0;
    let hi = arr.length - 1;
    if (hi < 0) return -1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(arr[lo - 1] - t) <= Math.abs(arr[lo] - t)) return lo - 1;
    return lo;
  };

  function showAt(t) {
    const x = X(t);
    cross.setAttribute("x1", x);
    cross.setAttribute("x2", x);
    cross.setAttribute("visibility", "visible");
    const parts = [fmtClock(t)];
    spec.series.forEach((s, i) => {
      const idx = nearest(s.t, t);
      const v = idx >= 0 ? s.v[idx] : null;
      const dot = dots[i];
      if (idx >= 0 && isNum(v)) {
        dot.setAttribute("cx", X(s.t[idx]));
        dot.setAttribute("cy", Y(v));
        dot.setAttribute("visibility", "visible");
      } else {
        dot.setAttribute("visibility", "hidden");
      }
      parts.push(s.letter + " " + spec.yFmt(v));
    });
    readout.textContent = parts.join("  ·  ");
  }

  function hide() {
    cross.setAttribute("visibility", "hidden");
    for (const dot of dots) dot.setAttribute("visibility", "hidden");
    readout.textContent = "";
  }

  const tFromEvent = (ev) => {
    const r = svg.getBoundingClientRect();
    const frac = (ev.clientX - r.left) / (r.width || 1);
    const px = frac * CW;
    const inner = (px - PADL) / (CW - PADL - PADR);
    return Math.max(0, Math.min(1, inner)) * xMax;
  };

  svg.addEventListener("pointermove", (ev) => showAt(tFromEvent(ev)));
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);

  // keyboard readout: the same crosshair, walked over the first run's frames
  let kbIdx = -1;
  const grid = spec.series[0] ? spec.series[0].t : [];
  svg.addEventListener("keydown", (ev) => {
    if (!grid.length) return;
    let next = kbIdx;
    if (ev.key === "ArrowRight") next = kbIdx < 0 ? 0 : Math.min(grid.length - 1, kbIdx + 1);
    else if (ev.key === "ArrowLeft") next = kbIdx < 0 ? grid.length - 1 : Math.max(0, kbIdx - 1);
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = grid.length - 1;
    else if (ev.key === "Escape") {
      kbIdx = -1;
      hide();
      return;
    } else return;
    ev.preventDefault();
    kbIdx = next;
    showAt(grid[kbIdx]);
  });

  wrap.appendChild(svg);
  /* export: the same panel as a PNG (export.js serializes the live node,
     which is why every mark above paints through attributes, not CSS) */
  cap.appendChild(pngButton(svg, "compare-" + safeName(spec.title, "chart") + ".png"));
  return wrap;
}

/* ---- config diff ----------------------------------------------------- */

/** Longest config value shown inline; the rest lives in the cell title. */
const VALUE_CHARS = 48;

function diffPanel() {
  const panel = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, "Config diff"));

  const withScen = entries.filter((e) => e.scen && e.scen.flat);
  const without = entries.filter((e) => !e.scen || !e.scen.flat);

  if (withScen.length < 2) {
    head.appendChild(eln("span", "sub", "needs two readable scenarios"));
    panel.appendChild(head);
    panel.appendChild(
      eln(
        "p",
        "sub",
        "Fewer than two of these runs have a readable scenario file, so there is nothing to diff."
      )
    );
    return panel;
  }

  const keys = new Set();
  for (const e of withScen) for (const k of Object.keys(e.scen.flat)) keys.add(k);
  const allKeys = [...keys].sort();
  const differing = allKeys.filter((k) => {
    const first = JSON.stringify(withScen[0].scen.flat[k]);
    return withScen.some((e) => JSON.stringify(e.scen.flat[k]) !== first);
  });
  const shown = showAllKeys ? allKeys : differing;

  head.appendChild(
    eln(
      "span",
      "sub",
      differing.length + " of " + allKeys.length + " scenario keys differ" +
        (differing.length ? "" : " — these runs share one scenario")
    )
  );
  const toggle = eln("label", "toggle");
  const box = eln("input");
  box.type = "checkbox";
  box.checked = showAllKeys;
  box.addEventListener("change", () => {
    showAllKeys = box.checked;
    render();
  });
  toggle.appendChild(box);
  toggle.appendChild(eln("span", "ctl", "show identical keys too"));
  head.appendChild(toggle);
  panel.appendChild(head);

  if (without.length) {
    panel.appendChild(
      eln(
        "p",
        "sub",
        "No scenario file for: " +
          without.map((e) => e.letter + " " + ((e.info && e.info.title) || e.id)).join(", ") +
          " — those runs are left out of the diff."
      )
    );
  }

  if (!shown.length) {
    panel.appendChild(
      eln("p", "sub", "Identical scenarios: nothing to show. Any difference is in the seed or in the code.")
    );
    return panel;
  }

  const wrap = tableWrap("config diff");
  const table = eln("table", "matrix difftable");
  const thead = eln("thead");
  const hr = eln("tr");
  const kh = eln("th", "keycol", "scenario key");
  kh.setAttribute("scope", "col");
  hr.appendChild(kh);
  for (const e of withScen) {
    const th = eln("th");
    th.setAttribute("scope", "col");
    th.appendChild(runKey(e));
    th.title = (e.info && e.info.title) || e.id;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = eln("tbody");
  for (const k of shown) {
    const tr = eln("tr");
    const th = eln("th", "keycol", k);
    th.setAttribute("scope", "row");
    tr.appendChild(th);
    const first = JSON.stringify(withScen[0].scen.flat[k]);
    for (const e of withScen) {
      const has = Object.prototype.hasOwnProperty.call(e.scen.flat, k);
      const raw = e.scen.flat[k];
      const text = JSON.stringify(raw);
      // a whole cluster list on one nowrap line would push the later
      // run columns off the panel; the full value stays in the title
      const shown = has
        ? text.length > VALUE_CHARS
          ? text.slice(0, VALUE_CHARS - 1) + "…"
          : text
        : "—";
      const td = eln("td", "mono valcell", shown);
      td.title = has ? text : "key absent from this run's scenario";
      if (text !== first) td.classList.add("changed");
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  panel.appendChild(wrap);
  panel.appendChild(
    eln(
      "p",
      "sub",
      "Values are the run's own scenario.yaml, flattened to dotted paths" +
        " (GET /api/runs/{id}/scenario) — a marked cell differs from run " +
        withScen[0].letter + "."
    )
  );
  return panel;
}
