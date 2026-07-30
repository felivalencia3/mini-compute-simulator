/* fleetsim v0.8 — experiments: the sweep launcher and the sweep board.

   Lazily imported by app.js.  Two surfaces, one module because they are
   two halves of one loop (define a grid -> watch it run -> compare it):

   1. mountExplore() wires the EXPLORE panel inside the scenario editor
      (#sweep).  Each axis is a dotted path plus a comma-separated value
      list — exactly what `POST /api/sweeps` takes and exactly what
      `fleetsim run -o path=value` means — and the expansion count is
      shown live, BEFORE anything is created, against the server's 64-cell
      cap.  The base scenario is whatever is in the editor above.

   2. mountSweep(id) renders the sweep board (#sweep/<id>): a live grid of
      cells with status, a metric chart that fills in as cells finish
      (bar for one axis, heatmap for two), and cell selection that hands
      off to the compare view.

   Cell values are parsed as JSON when they parse (numbers, booleans,
   lists) and kept as plain strings otherwise, so `2h`, `pow2[8, 32]` and
   `first_fit` all mean what they look like.  Splitting respects
   bracket/quote nesting, so `[2, 8], [4, 16]` is two values.

   All dynamic text goes through textContent; SVG is built with
   createElementNS. */

"use strict";

/* Mirrors sweeps.MAX_SWEEP_RUNS / _PATH_RE — client-side so the count and
   the refusal are immediate; the server re-checks both regardless. */
const MAX_CELLS = 64;
const PATH_RE = /^[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z_][A-Za-z0-9_-]*)*$/;
const MAX_COMPARE = 8;

/* Sequential ramp for the heatmap (one hue, near-surface -> bright: on a
   dark surface the low end is the step that recedes).  Magnitude gets one
   hue, never the categorical run colors. */
const RAMP = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"];
const BAR = "#3987e5";
const MUTED = "#8b93a1";
const GRID = "rgba(255,255,255,.07)";
const SVG_NS = "http://www.w3.org/2000/svg";

const PRESETS = {
  scheduler: {
    path: "scheduler.name",
    values: "fifo, sjf, easy_backfill, tiered_priority",
  },
  placement: {
    path: "scheduler.params.placement",
    values: "first_fit, best_fit, consolidate, spread",
  },
  seed: { path: "sim.seed", values: "1, 2, 3" },
  rate: { path: null, values: "3, 6, 12" }, // path derived from the scenario
};

const isNum = (v) => typeof v === "number" && isFinite(v);
const fmtPct = (v) => (isNum(v) ? (v * 100).toFixed(1) + "%" : "–");
const fmtInt = (v) => (isNum(v) ? String(Math.round(v)) : "–");

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

async function apiJSON(path, options) {
  let resp;
  try {
    resp = await fetch(path, options);
  } catch (err) {
    return { ok: false, status: 0, doc: null };
  }
  let doc = null;
  try {
    doc = await resp.json();
  } catch (err) {
    /* non-JSON body */
  }
  return { ok: resp.ok, status: resp.status, doc };
}

const postJSON = (path, body) =>
  apiJSON(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/* ------------------------------------------------------------------ *
 * value parsing
 * ------------------------------------------------------------------ */

/** Split on TOP-LEVEL commas only (brackets, braces and quotes nest). */
function splitValues(text) {
  const out = [];
  let depth = 0;
  let quote = null;
  let cur = "";
  for (const ch of text) {
    if (quote) {
      cur += ch;
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
      continue;
    }
    if (ch === "[" || ch === "{") depth++;
    if (ch === "]" || ch === "}") depth = Math.max(0, depth - 1);
    if (ch === "," && depth === 0) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim()).filter((s) => s !== "");
}

/** One token as JSON when it parses, else the literal string. */
function parseValue(token) {
  try {
    return JSON.parse(token);
  } catch (err) {
    return token;
  }
}

const parseValues = (text) => splitValues(text).map(parseValue);

/** The first workload-class name in the editor's YAML (for the rate
    preset), or null.  A deliberately shallow indentation scan — the
    browser has no YAML parser and this only seeds an editable field. */
function firstWorkloadClass(text) {
  const lines = text.split("\n");
  let classesIndent = -1;
  for (const line of lines) {
    if (/^\s*#/.test(line) || line.trim() === "") continue;
    const indent = line.length - line.trimStart().length;
    if (classesIndent < 0) {
      if (/^\s*classes:\s*$/.test(line)) classesIndent = indent;
      continue;
    }
    if (indent <= classesIndent) return null; // block ended, nothing found
    const m = /^\s*([A-Za-z_][A-Za-z0-9_-]*):\s*$/.exec(line);
    if (m) return m[1];
  }
  return null;
}

/* ================================================================== *
 * 1. the explore panel (inside the scenario editor)
 * ================================================================== */
let exploreWired = false;
let launching = false;

export function mountExplore() {
  if (!exploreWired) {
    exploreWired = true;
    document.querySelector("#addAxisBtn").addEventListener("click", () => {
      addAxis("", "");
      focusLastPath();
    });
    document.querySelector("#launchSweepBtn").addEventListener("click", launchSweep);
    for (const btn of document.querySelectorAll("#explorePanel .preset")) {
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
    }
  }
  if (!document.querySelectorAll("#axisList .axisrow").length) {
    // open with one empty axis so the shape of the thing is obvious
    addAxis("", "");
  }
  updateSize();
}

function axisRows() {
  return [...document.querySelectorAll("#axisList .axisrow")];
}

function readAxes() {
  const axes = [];
  for (const row of axisRows()) {
    const path = row.querySelector(".axpath").value.trim();
    const raw = row.querySelector(".axvals").value;
    axes.push({ row, path, values: parseValues(raw), raw });
  }
  return axes;
}

function addAxis(path, values) {
  const row = eln("div", "axisrow");

  const pathIn = eln("input", "axpath");
  pathIn.type = "text";
  pathIn.placeholder = "scheduler.params.placement";
  pathIn.value = path || "";
  pathIn.setAttribute("aria-label", "parameter path");
  pathIn.addEventListener("input", updateSize);
  row.appendChild(pathIn);

  const valIn = eln("input", "axvals");
  valIn.type = "text";
  valIn.placeholder = "value, value, …";
  valIn.value = values || "";
  valIn.setAttribute("aria-label", "values, comma separated");
  valIn.addEventListener("input", updateSize);
  row.appendChild(valIn);

  const note = eln("span", "axn sub mono");
  row.appendChild(note);

  const del = eln("button", "axdel", "×");
  del.type = "button";
  del.title = "Remove this axis";
  del.setAttribute("aria-label", "Remove this axis");
  del.addEventListener("click", () => {
    row.remove();
    if (!axisRows().length) addAxis("", "");
    updateSize();
  });
  row.appendChild(del);

  document.querySelector("#axisList").appendChild(row);
  updateSize();
  return row;
}

function focusLastPath() {
  const rows = axisRows();
  if (rows.length) rows[rows.length - 1].querySelector(".axpath").focus();
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;
  let path = preset.path;
  if (name === "rate") {
    const cls = firstWorkloadClass(document.querySelector("#yamlBox").value);
    path = "workload.classes." + (cls || "CLASS") + ".rate_per_hour";
  }
  // an empty first row is filled in place instead of leaving a blank
  const empty = axisRows().find(
    (r) =>
      !r.querySelector(".axpath").value.trim() &&
      !r.querySelector(".axvals").value.trim()
  );
  const existing = axisRows().find(
    (r) => r.querySelector(".axpath").value.trim() === path
  );
  const row = existing || empty || addAxis("", "");
  row.querySelector(".axpath").value = path;
  if (!existing) row.querySelector(".axvals").value = preset.values;
  row.querySelector(".axvals").focus();
  updateSize();
}

function updateSize() {
  const axes = readAxes();
  let cells = 1;
  let usable = 0;
  for (const ax of axes) {
    const bad = ax.path !== "" && !PATH_RE.test(ax.path);
    const note = ax.row.querySelector(".axn");
    ax.row.querySelector(".axpath").setAttribute("aria-invalid", bad ? "true" : "false");
    ax.row.classList.toggle("bad", bad);
    if (bad) {
      note.textContent = "not a dotted path";
    } else if (!ax.path) {
      note.textContent = ax.values.length ? "needs a path" : "";
    } else if (!ax.values.length) {
      note.textContent = "needs values";
    } else {
      note.textContent = ax.values.length + (ax.values.length === 1 ? " value" : " values");
    }
    if (!bad && ax.path && ax.values.length) {
      usable++;
      cells *= ax.values.length;
    }
  }
  const size = document.querySelector("#sweepSize");
  const btn = document.querySelector("#launchSweepBtn");
  if (!usable) {
    size.textContent = "add an axis with a path and at least one value";
    size.classList.remove("over");
    btn.disabled = true;
    btn.textContent = "Launch sweep";
    return;
  }
  const factors = readAxes()
    .filter((ax) => ax.path && ax.values.length && PATH_RE.test(ax.path))
    .map((ax) => ax.values.length);
  const over = cells > MAX_CELLS;
  // one axis reads "4 runs"; several read "2 × 3 = 6 runs"
  const product = factors.length > 1 ? factors.join(" × ") + " = " : "";
  size.textContent =
    product + cells + (cells === 1 ? " run" : " runs") +
    (over ? "  ·  over the " + MAX_CELLS + "-cell cap" : "  ·  cap " + MAX_CELLS);
  size.classList.toggle("over", over);
  btn.disabled = over || launching;
  btn.textContent = "Launch sweep (" + cells + ")";
}

function showSweepErrors(list) {
  document.querySelector("#valOk").classList.add("hidden");
  const ul = document.querySelector("#valErrors");
  ul.textContent = "";
  for (const err of list) ul.appendChild(eln("li", null, String(err)));
  ul.classList.toggle("hidden", list.length === 0);
}

async function launchSweep() {
  if (launching) return;
  const axes = readAxes().filter((ax) => ax.path || ax.values.length);
  const errors = [];
  const grid = {};
  for (const ax of axes) {
    if (!ax.path) errors.push("an axis has values but no path");
    else if (!PATH_RE.test(ax.path)) errors.push(ax.path + ": not a dotted path");
    else if (!ax.values.length) errors.push(ax.path + ": no values");
    else if (grid[ax.path]) errors.push(ax.path + ": listed twice");
    else grid[ax.path] = ax.values;
  }
  if (!Object.keys(grid).length) errors.push("add at least one axis");
  if (errors.length) {
    showSweepErrors(errors);
    return;
  }
  showSweepErrors([]);
  launching = true;
  updateSize();
  const status = document.querySelector("#editorStatus");
  status.textContent = "Launching…";
  const body = { yaml: document.querySelector("#yamlBox").value, grid };
  const title = document.querySelector("#titleInput").value.trim();
  if (title) body.title = title;
  const { ok, doc } = await postJSON("/api/sweeps", body);
  launching = false;
  status.textContent = "";
  updateSize();
  if (ok && doc && doc.sweep_id) {
    location.hash = "#sweep/" + encodeURIComponent(doc.sweep_id);
    return;
  }
  // 400/413 answer {ok: false, errors: [...]}; 404/405/415/421/500
  // answer {error: str} — handle both envelopes
  if (doc && Array.isArray(doc.errors)) showSweepErrors(doc.errors);
  else {
    showSweepErrors([
      (doc && doc.error) || "the server did not answer — is it still running?",
    ]);
  }
}

/* ================================================================== *
 * 2. the sweep board (#sweep/<id>)
 * ================================================================== */
let sweepToken = 0;
let sweepId = null;
let sweepDoc = null;
let sweepTimer = null;
let sweepWired = false;
let cellSel = new Set();
let cellNote = "";

export function unmountSweep() {
  sweepToken++;
  if (sweepTimer !== null) {
    clearTimeout(sweepTimer);
    sweepTimer = null;
  }
}

export function mountSweep(id) {
  if (!sweepWired) {
    sweepWired = true;
    const metric = document.querySelector("#sweepMetric");
    if (metric) metric.addEventListener("change", () => renderSweep());
  }
  if (id !== sweepId) {
    sweepId = id;
    sweepDoc = null;
    cellSel = new Set();
    cellNote = "";
  }
  loadSweep();
}

async function loadSweep() {
  const mine = ++sweepToken;
  const id = sweepId;
  const { ok, status, doc } = await apiJSON("/api/sweeps/" + encodeURIComponent(id));
  if (mine !== sweepToken || id !== sweepId) return;
  if (!ok) {
    sweepDoc = null;
    const body = document.querySelector("#sweepBody");
    body.textContent = "";
    const panel = eln("div", "panel");
    const head = eln("div", "phead");
    head.appendChild(eln("h2", null, "No such sweep"));
    panel.appendChild(head);
    panel.appendChild(
      eln(
        "p",
        "sub",
        status === 404
          ? "This sweep is not in the workspace. Every cell may have been dequeued."
          : "The server did not answer. Is fleetsim serve still running?"
      )
    );
    body.appendChild(panel);
    document.querySelector("#sweepTitle").textContent = "Sweep";
    document.querySelector("#sweepMeta").textContent = "";
    return;
  }
  sweepDoc = doc;
  renderSweep();
  const busy = (doc.runs || []).some(
    (r) => r.status === "queued" || r.status === "running"
  );
  if (busy) {
    sweepTimer = setTimeout(() => {
      sweepTimer = null;
      if (mine === sweepToken) loadSweep();
    }, 2500);
  }
}

/** Axis order: the grid's key order, with the seeds sugar appended. */
function sweepAxes(doc) {
  const axes = [];
  for (const path of Object.keys(doc.grid || {})) {
    axes.push({ path, values: (doc.grid[path] || []).map((v) => v) });
  }
  // the `seeds` sugar IS a sim.seed axis: a record carrying both (only
  // reachable through the API by hand) must not list the axis twice
  if (
    Array.isArray(doc.seeds) &&
    doc.seeds.length &&
    !axes.some((a) => a.path === "sim.seed")
  ) {
    axes.push({ path: "sim.seed", values: doc.seeds.slice() });
  }
  return axes;
}

const cellKey = (value) => JSON.stringify(value);

function cellLabel(cell) {
  if (!cell || typeof cell !== "object") return "base";
  return Object.keys(cell)
    .map((k) => k.split(".").pop() + "=" + JSON.stringify(cell[k]))
    .join(", ");
}

/** Fill the metric selector from what the CELLS actually carry.
 *
 * The board shipped three fixed options — occupancy, goodput, jobs
 * finished — and a placement sweep is DEAD FLAT on all three while
 * moving entirely in fragmentation, so the heatmap reported "placement
 * changes nothing" from runs whose own summary.json said otherwise.
 * Every `frag.*` key a finished cell reports is offered here.
 */
function syncMetricOptions(doc) {
  const sel = document.querySelector("#sweepMetric");
  if (!sel) return;
  const keys = new Set();
  for (const row of doc.runs || []) {
    for (const key of Object.keys(row.headline || {})) {
      if (key.startsWith("frag.") && isNum(row.headline[key])) keys.add(key);
    }
  }
  const want = ["occupancy", "goodput", "jobs_finished"].concat([...keys].sort());
  const have = [...sel.options].map((o) => o.value);
  if (have.join("|") === want.join("|")) return;
  const current = sel.value;
  sel.textContent = "";
  for (const key of want) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = metricLabel(key);
    sel.appendChild(opt);
  }
  sel.value = want.includes(current) ? current : "occupancy";
}

function metricLabel(key) {
  if (key === "jobs_finished") return "jobs finished";
  if (key === "frag.stranded_whole_nodes") return "stranded whole nodes";
  if (key.startsWith("frag.")) return "fragmentation · " + key.slice(5);
  return key;
}

function metricSpec() {
  const sel = document.querySelector("#sweepMetric");
  const key = sel ? sel.value : "occupancy";
  if (key === "jobs_finished") {
    return { key, label: "jobs finished", fmt: fmtInt, pct: false };
  }
  if (key === "frag.stranded_whole_nodes") {
    // a COUNT of leaves, not a ratio: never rendered as a percentage
    return {
      key,
      label: metricLabel(key),
      fmt: (v) => (isNum(v) ? v.toFixed(2) : "–"),
      pct: false,
    };
  }
  if (key.startsWith("frag.")) {
    // an index in [0, 1] but NOT a percentage of anything — 0.936 is
    // "very scattered", not "93.6% of something"
    return {
      key,
      label: metricLabel(key),
      fmt: (v) => (isNum(v) ? v.toFixed(3) : "–"),
      pct: false,
      lowIsBetter: true,
    };
  }
  return { key, label: key, fmt: fmtPct, pct: true };
}

const metricOf = (row, key) =>
  row && row.headline && isNum(row.headline[key]) ? row.headline[key] : null;

function renderSweep() {
  const doc = sweepDoc;
  if (!doc) return;
  syncMetricOptions(doc);
  const body = document.querySelector("#sweepBody");
  document.querySelector("#sweepTitle").textContent = doc.title || doc.sweep_id;
  const runs = doc.runs || [];
  const nFailed = runs.filter((r) => r.status === "failed").length;
  const nQueued = runs.filter((r) => r.status === "queued").length;
  const nRunning = runs.filter((r) => r.status === "running").length;
  const bits = [doc.n_done + "/" + doc.n_runs + " done"];
  if (nRunning) bits.push(nRunning + " running");
  if (nQueued) bits.push(nQueued + " queued");
  if (nFailed) bits.push(nFailed + " failed");
  bits.push(sweepAxes(doc).map((a) => a.path).join(" × ") || "no axes");
  document.querySelector("#sweepMeta").textContent = bits.join(" · ");

  body.textContent = "";
  body.appendChild(cellsPanel(doc));
  body.appendChild(chartPanel(doc));
}

function selectionBar(doc) {
  const bar = eln("div", "ctlrow selbar");
  const n = cellSel.size;
  bar.appendChild(eln("span", "sub mono", n ? n + " selected" : "none selected"));
  const cmp = eln("a", "btn primary", n >= 2 ? "Compare " + n + " cells" : "Compare");
  const ids = [...cellSel];
  if (n >= 2) {
    cmp.href = "#compare/" + ids.map(encodeURIComponent).join(",");
  } else {
    cmp.href = "#compare/";
    cmp.setAttribute("aria-disabled", "true");
    cmp.addEventListener("click", (ev) => ev.preventDefault());
    cmp.title = "Select at least two cells";
  }
  bar.appendChild(cmp);

  const done = (doc.runs || []).filter((r) => r.status === "done");
  const pick = eln(
    "button",
    "btn",
    done.length < 2
      ? "Select done cells"
      : "Select first " + Math.min(MAX_COMPARE, done.length) + " done"
  );
  pick.type = "button";
  pick.disabled = done.length < 2;
  if (pick.disabled) pick.title = "at least two cells have to finish first";
  pick.addEventListener("click", () => {
    cellSel = new Set(done.slice(0, MAX_COMPARE).map((r) => r.id));
    cellNote = done.length > MAX_COMPARE
      ? "Compare holds " + MAX_COMPARE + " runs — selected the first " + MAX_COMPARE + "."
      : "";
    renderSweep();
  });
  bar.appendChild(pick);

  const clear = eln("button", "btn", "Clear");
  clear.type = "button";
  clear.disabled = cellSel.size === 0;
  clear.addEventListener("click", () => {
    cellSel = new Set();
    cellNote = "";
    renderSweep();
  });
  bar.appendChild(clear);

  const queued = (doc.runs || []).filter((r) => r.status === "queued");
  if (queued.length) {
    const deq = eln("button", "btn", "Dequeue " + queued.length + " queued");
    deq.type = "button";
    deq.title = "Remove the cells that have not started yet";
    deq.addEventListener("click", () => dequeueSweep(doc));
    bar.appendChild(deq);
  }
  return bar;
}

async function dequeueSweep(doc) {
  const n = (doc.runs || []).filter((r) => r.status === "queued").length;
  if (!window.confirm("Remove " + n + " queued cell(s) from this sweep? Cells already running are left alone.")) {
    return;
  }
  await apiJSON("/api/sweeps/" + encodeURIComponent(doc.sweep_id), {
    method: "DELETE",
  });
  loadSweep();
}

function toggleCell(id, want) {
  if (want) {
    if (cellSel.size >= MAX_COMPARE) {
      cellNote = "Compare holds " + MAX_COMPARE + " runs (one per line color) — deselect one first.";
      renderSweep();
      return;
    }
    cellSel.add(id);
  } else {
    cellSel.delete(id);
  }
  cellNote = "";
  renderSweep();
}

function cellCheckbox(row) {
  const box = eln("input", "cellsel");
  box.type = "checkbox";
  box.checked = cellSel.has(row.id);
  box.disabled = row.status !== "done";
  box.setAttribute(
    "aria-label",
    "Select cell " + cellLabel(row.cell) + " for comparison"
  );
  if (row.status !== "done") box.title = "only finished cells can be compared";
  box.addEventListener("change", () => toggleCell(row.id, box.checked));
  return box;
}

function statusDot(status) {
  const dot = eln("span", "dot " + status);
  dot.setAttribute("aria-hidden", "true");
  return dot;
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

function cellsPanel(doc) {
  const panel = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, "Cells"));
  head.appendChild(
    eln("span", "sub", "One ordinary run per cell — open one, or select two or more to compare.")
  );
  panel.appendChild(head);
  panel.appendChild(selectionBar(doc));
  if (cellNote) panel.appendChild(eln("p", "railnote", cellNote));

  const metric = metricSpec();
  const wrap = tableWrap("sweep cells");
  const table = eln("table", "matrix celltable");
  const thead = eln("thead");
  const hr = eln("tr");
  for (const label of ["", "cell", "status", metric.label]) {
    const th = eln("th", label === "" ? "tinycol" : null, label);
    th.setAttribute("scope", "col");
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = eln("tbody");
  for (const row of doc.runs || []) {
    const tr = eln("tr");
    const sel = eln("td", "tinycol");
    sel.appendChild(cellCheckbox(row));
    tr.appendChild(sel);

    const cellTd = eln("td");
    const link = eln("a", null, cellLabel(row.cell));
    link.href = "#run/" + encodeURIComponent(row.id) + "/report";
    link.title = row.title || row.id;
    cellTd.appendChild(link);
    tr.appendChild(cellTd);

    const st = eln("td");
    const stWrap = eln("span", "statuswrap");
    stWrap.appendChild(statusDot(row.status));
    stWrap.appendChild(
      eln(
        "span",
        "sub",
        row.status === "queued" && row.queue_position != null
          ? "queued #" + row.queue_position
          : row.status
      )
    );
    st.appendChild(stWrap);
    if (row.status === "failed" && row.error) st.title = row.error;
    tr.appendChild(st);

    const val = metricOf(row, metric.key);
    const vt = eln("td", "num", metric.fmt(val));
    if (val == null) vt.title = "no value yet — the cell has not finished";
    tr.appendChild(vt);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  panel.appendChild(wrap);
  return panel;
}

/* ---- the metric chart ------------------------------------------------ */

function chartPanel(doc) {
  const panel = eln("div", "panel");
  const metric = metricSpec();
  const axes = sweepAxes(doc);
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, metric.label + " across the grid"));
  head.appendChild(
    eln(
      "span",
      "sub",
      axes.length === 2
        ? "rows: " + axes[0].path + " · columns: " + axes[1].path
        : axes.length === 1
          ? "one axis: " + axes[0].path
          : axes.length + " axes — one bar per cell"
    )
  );
  panel.appendChild(head);

  const rows = doc.runs || [];
  const withValue = rows.filter((r) => metricOf(r, metric.key) != null);
  if (!withValue.length) {
    panel.appendChild(
      eln("p", "sub", "No cell has finished yet — the chart fills in as they do.")
    );
    return panel;
  }
  if (withValue.length < rows.length) {
    panel.appendChild(
      eln(
        "p",
        "sub",
        withValue.length + " of " + rows.length + " cells have a value so far."
      )
    );
  }
  if (axes.length === 2) panel.appendChild(heatmap(doc, axes, metric));
  else panel.appendChild(barChart(rows, metric, axes));
  return panel;
}

function barChart(rows, metric, axes) {
  const W = 860;
  const H = 190;
  const padL = 52;
  const padR = 12;
  const padT = 14;
  const padB = 46;
  const wrap = eln("div", "chart");
  const values = rows.map((r) => metricOf(r, metric.key));
  const max = metric.pct ? 1 : Math.max(0.0001, ...values.filter(isNum));
  const svg = svgEl("svg", {
    viewBox: "0 0 " + W + " " + H,
    class: "chartsvg",
    role: "img",
    "aria-label":
      metric.label + " per sweep cell: " +
      rows
        .map((r) => cellLabel(r.cell) + " " + metric.fmt(metricOf(r, metric.key)))
        .join(", "),
  });
  for (const frac of [0, 0.5, 1]) {
    const y = padT + (1 - frac) * (H - padT - padB);
    svg.appendChild(svgEl("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: GRID }));
    const lab = svgEl("text", {
      x: padL - 6,
      y: y + 3.5,
      "text-anchor": "end",
      fill: MUTED,
      "font-size": "9.5",
    });
    lab.textContent = metric.fmt(max * frac);
    svg.appendChild(lab);
  }
  const n = rows.length;
  const span = W - padL - padR;
  // a two-cell sweep should not stretch two bars across 800px: cap the
  // slot and center the group instead
  const slot = Math.min(120, span / Math.max(1, n));
  const x0 = padL + (span - slot * n) / 2;
  const bw = Math.max(6, Math.min(56, slot - 6)); // 2px+ surface gap either side
  rows.forEach((row, i) => {
    const v = metricOf(row, metric.key);
    const cx = x0 + slot * i + slot / 2;
    if (isNum(v)) {
      const h = Math.max(1, (v / max) * (H - padT - padB));
      svg.appendChild(
        svgEl("rect", {
          x: (cx - bw / 2).toFixed(1),
          y: (H - padB - h).toFixed(1),
          width: bw.toFixed(1),
          height: h.toFixed(1),
          rx: 4, // rounded data-end, anchored to the baseline
          fill: BAR,
        })
      );
      const val = svgEl("text", {
        x: cx,
        y: H - padB - h - 4,
        "text-anchor": "middle",
        fill: "#e6e8eb",
        "font-size": "9.5",
      });
      val.textContent = metric.fmt(v);
      svg.appendChild(val);
    } else {
      const pend = svgEl("text", {
        x: cx,
        y: H - padB - 6,
        "text-anchor": "middle",
        fill: MUTED,
        "font-size": "9.5",
      });
      pend.textContent = row.status === "failed" ? "failed" : "…";
      svg.appendChild(pend);
    }
    const label =
      axes.length === 1
        ? String(row.cell ? row.cell[axes[0].path] : "base")
        : cellLabel(row.cell);
    const text = label.length > 28 ? label.slice(0, 27) + "…" : label;
    // short values sit level under their bar; only long cell labels are
    // angled (a rotated "7" reads as a typo)
    const tilt = text.length > 6;
    const ly = H - padB + (tilt ? 14 : 13);
    const lab = svgEl("text", {
      x: cx,
      y: ly,
      "text-anchor": tilt ? "end" : "middle",
      fill: MUTED,
      "font-size": "9.5",
    });
    if (tilt) lab.setAttribute("transform", "rotate(-35 " + cx.toFixed(1) + " " + ly + ")");
    lab.textContent = text;
    svg.appendChild(lab);
  });
  svg.appendChild(
    svgEl("line", { x1: padL, x2: W - padR, y1: H - padB, y2: H - padB, stroke: "rgba(255,255,255,.18)" })
  );
  wrap.appendChild(svg);
  return wrap;
}

function heatmap(doc, axes, metric) {
  const wrap = eln("div", "heatwrap");
  const [ay, ax] = axes;
  const byCell = new Map();
  for (const row of doc.runs || []) {
    const cell = row.cell || {};
    byCell.set(cellKey(cell[ay.path]) + "|" + cellKey(cell[ax.path]), row);
  }
  const values = (doc.runs || [])
    .map((r) => metricOf(r, metric.key))
    .filter(isNum);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const step = (v) => {
    if (hi === lo) return RAMP.length - 1;
    const f = (v - lo) / (hi - lo);
    return Math.max(0, Math.min(RAMP.length - 1, Math.round(f * (RAMP.length - 1))));
  };

  const table = eln("table", "matrix heat");
  const thead = eln("thead");
  const hr = eln("tr");
  // short leaf names in the corner (the panel subhead already spells the
  // full dotted paths out); the title carries them here too
  const leaf = (p) => p.split(".").pop();
  const corner = eln("th", "keycol", leaf(ay.path) + " ╲ " + leaf(ax.path));
  corner.setAttribute("scope", "col");
  corner.title = "rows: " + ay.path + " · columns: " + ax.path;
  hr.appendChild(corner);
  for (const xv of ax.values) {
    const th = eln("th", null, String(xv));
    th.setAttribute("scope", "col");
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = eln("tbody");
  for (const yv of ay.values) {
    const tr = eln("tr");
    const th = eln("th", "keycol", String(yv));
    th.setAttribute("scope", "row");
    tr.appendChild(th);
    for (const xv of ax.values) {
      const row = byCell.get(cellKey(yv) + "|" + cellKey(xv));
      const v = row ? metricOf(row, metric.key) : null;
      const td = eln("td", "heatcell");
      if (isNum(v)) {
        const s = step(v);
        td.style.background = RAMP[s];
        td.style.color = s >= 3 ? "#0b0e14" : "#e6e8eb";
        const link = eln("a", "heatlink", metric.fmt(v));
        link.href = "#run/" + encodeURIComponent(row.id) + "/report";
        link.style.color = "inherit";
        link.title =
          ay.path + "=" + JSON.stringify(yv) + ", " + ax.path + "=" +
          JSON.stringify(xv) + " · " + metric.label + " " + metric.fmt(v);
        td.appendChild(link);
      } else {
        td.classList.add("empty");
        td.textContent = row ? (row.status === "failed" ? "failed" : "…") : "—";
        if (!row) td.title = "no cell for this combination";
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);

  const legend = eln("div", "heatlegend");
  legend.appendChild(eln("span", "sub", metric.fmt(lo)));
  for (const hex of RAMP) {
    const sw = eln("span", "heatswatch");
    sw.style.background = hex;
    sw.setAttribute("aria-hidden", "true");
    legend.appendChild(sw);
  }
  legend.appendChild(eln("span", "sub", metric.fmt(hi)));
  legend.appendChild(
    eln("span", "sub", "· every cell also prints its value, so color is never the only channel")
  );
  wrap.appendChild(legend);
  return wrap;
}
