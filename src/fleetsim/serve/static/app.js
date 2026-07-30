/* fleetsim app shell (v0.5, extended v0.8) — vanilla ES module, no build
   step, no external requests.  Talks only to the same-origin /api routes
   (contract pinned in src/fleetsim/serve/server.py).

   Views (hash-routed):
     #                     home (+ the sweep list)
     #new                  scenario editor (+ the live fleet-shape preview)
     #sweep                the same editor in EXPLORE mode (axis grid)
     #sweep/<id>           sweep dashboard (live cells + metric chart)
     #run/<id>/report      2D report (iframe) or live progress + live map
     #run/<id>/fleet3d     3D fleet replay — LIVE while the run executes
     #run/<id>/insight     analysis: attribution panels over the model
     #compare/<id>,<id>…   compare view (matrix, timelines, config diff)
     #validation           published-vs-fleetsim results + anti-goals

   DEEP LINKS (v0.8): a run route may carry a query string after the
   mode — `#run/<id>/fleet3d?t=<us>&cam=<6 numbers>&pin=<domain>&
   x=<domain>&hide=<class,class>`.  It is decoded into `route.deep` and
   handed to the 3D view, which restores the exact moment, camera pose,
   pin, drill-down and class filters.  The hash is NOT rewritten while
   you scrub (that would bury the back button under one entry per
   frame); the view's "Copy link" button mints the URL on demand.

   This module owns the rail (including the multi-select that feeds
   #compare), routing, the run view and the editor.  The heavier
   surfaces are lazily imported and own their own DOM, polling and
   controls: ./compare.js (#compare), ./experiment.js (#sweep and the
   editor's explore panel), ./insight.js (the analysis tab) and
   ./validation.js (#validation), the same shape as ./fleet3d.js.
   ./live.js + ./fleetmap.js (the live 2D map and the editor preview)
   are imported the same way.

   All dynamic text goes through textContent — titles and error strings
   are data, never markup. */

"use strict";

const $ = (s) => document.querySelector(s);

/* ------------------------------------------------------------------ *
 * formatting
 * ------------------------------------------------------------------ */
const US = 1e6;

function fmtClock(us) {
  const s = Math.floor(us / US);
  const d = Math.floor(s / 86400);
  const hh = Math.floor((s % 86400) / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const p = (n) => String(n).padStart(2, "0");
  return (d ? d + "d " : "") + p(hh) + ":" + p(mm) + ":" + p(ss);
}

const fmtPct = (v) =>
  v == null || !isFinite(v) ? "–" : (v * 100).toFixed(1) + "%";
const fmtInt = (v) => (v == null ? "–" : String(v));

function fmtWhen(unix) {
  if (unix == null) return "";
  const then = new Date(unix * 1000);
  const ageS = (Date.now() - then.getTime()) / 1000;
  if (ageS < 60) return "just now";
  if (ageS < 3600) return Math.floor(ageS / 60) + "m ago";
  if (ageS < 86400) return Math.floor(ageS / 3600) + "h ago";
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " + then.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/* ------------------------------------------------------------------ *
 * tiny fetch wrapper: JSON in / JSON out, never throws on HTTP errors
 * ------------------------------------------------------------------ */
async function apiFetch(path, options) {
  let resp;
  try {
    resp = await fetch(path, options);
  } catch (err) {
    return { ok: false, status: 0, doc: null, netError: String(err) };
  }
  let doc = null;
  try {
    doc = await resp.json();
  } catch (err) {
    /* non-JSON body (should not happen on /api routes) */
  }
  return { ok: resp.ok, status: resp.status, doc };
}

const apiGet = (path) => apiFetch(path);
const apiPost = (path, body) =>
  apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
const apiDelete = (path) => apiFetch(path, { method: "DELETE" });

/* ------------------------------------------------------------------ *
 * routing
 * ------------------------------------------------------------------ */

/* A hand-edited or truncated URL can carry a malformed percent-escape;
   decodeURIComponent would throw and kill the navigation.  Falling back
   to the raw text turns it into a literal id the server 404s into the
   "No such run" panel instead. */
function safeDecode(text) {
  try {
    return decodeURIComponent(text);
  } catch (err) {
    return text;
  }
}

/* The deep-link query: every field optional, every field validated.  A
   hand-edited link never throws and never half-applies — an unparseable
   value is simply absent, and the view opens at its default. */
function parseDeep(query) {
  if (!query) return null;
  let params;
  try {
    params = new URLSearchParams(query);
  } catch (err) {
    return null;
  }
  const deep = {};
  const t = Number(params.get("t"));
  if (params.has("t") && isFinite(t) && t >= 0) deep.t = t;
  /* Number("") is 0, so a plain `.map(Number)` accepts "cam=,,,,," as six
     valid zeros — a radius-0 camera INSIDE the geometry, with no way out
     but reloading. Each field must be a real number, and the radius must
     be positive; the 3D view clamps it to the orbit controller's range. */
  const camRaw = params.get("cam");
  if (camRaw != null) {
    const parts = camRaw.split(",");
    const cam = parts.map((s) => (s.trim() === "" ? NaN : Number(s)));
    if (cam.length === 6 && cam.every((v) => isFinite(v)) && cam[3] > 0) {
      deep.cam = cam;
    }
  }
  const pin = params.get("pin");
  if (pin) deep.pin = pin;
  const expand = params.get("x");
  if (expand) deep.expand = expand;
  const hide = params.get("hide");
  if (hide) deep.hide = hide.split(",").filter(Boolean);
  return Object.keys(deep).length ? deep : null;
}

function parseRoute() {
  const raw = location.hash.replace(/^#/, "");
  const qi = raw.indexOf("?");
  const h = qi >= 0 ? raw.slice(0, qi) : raw;
  const query = qi >= 0 ? raw.slice(qi + 1) : "";
  if (h === "new") return { view: "new", explore: false };
  if (h === "sweep") return { view: "new", explore: true };
  if (h === "validation") return { view: "validation" };
  let m = /^sweep\/([^/]+)$/.exec(h);
  if (m) return { view: "sweep", id: safeDecode(m[1]) };
  m = /^compare\/(.+)$/.exec(h);
  if (m) {
    const ids = [];
    for (const raw2 of m[1].split(",")) {
      const id = safeDecode(raw2).trim();
      if (id && !ids.includes(id)) ids.push(id); // dedupe, keep order
    }
    return { view: "compare", ids };
  }
  m = /^run\/([^/]+)\/(report|fleet3d|insight)$/.exec(h);
  if (m) {
    return {
      view: "run", id: safeDecode(m[1]), mode: m[2], deep: parseDeep(query),
    };
  }
  return { view: "home" };
}

let route = { view: "home" };

function showView(name) {
  for (const v of ["home", "run", "new", "compare", "sweep", "validation"]) {
    $("#view-" + v).classList.toggle("hidden", v !== name);
  }
  if (name !== "run" && fleet3dMod) fleet3dMod.hideFleet3d();
  if (name !== "run" && insightMod) insightMod.unmountInsight();
  if (name !== "run") stopLiveMap();
}

/* ------------------------------------------------------------------ *
 * 3D fleet replay — lazily imported (three.js is ~750 KB; only the 3D
 * tab pays for it).  The module keeps per-run timeline state, so
 * switching 2D <-> 3D keeps T.
 * ------------------------------------------------------------------ */
let fleet3dMod = null;
let fleet3dLoad = null;

function loadFleet3d() {
  if (!fleet3dLoad) {
    fleet3dLoad = import("./fleet3d.js").then((m) => {
      fleet3dMod = m;
      return m;
    });
  }
  return fleet3dLoad;
}

async function openFleet3d(id, opts) {
  try {
    await loadFleet3d();
  } catch (err) {
    const mount = $("#fleet3dMount");
    if (mount) mount.textContent = "3D module failed to load: " + String(err);
    return;
  }
  if (route.view === "run" && route.id === id && route.mode === "fleet3d") {
    fleet3dMod.mountFleet3d(id, opts || {});
  }
}

/* ------------------------------------------------------------------ *
 * compare / experiment modules — same lazy pattern as fleet3d: each one
 * owns its view's DOM, fetches and polling, and is only paid for when
 * the route asks for it.
 * ------------------------------------------------------------------ */
let compareMod = null;
let compareLoad = null;
let expMod = null;
let expLoad = null;
let insightMod = null;
let insightLoad = null;
let validationMod = null;
let validationLoad = null;

/* live.js + fleetmap.js + export.js travel together: the live 2D map and
   the editor's fleet-shape preview are the same drawing code. */
let mapMod = null;
let mapLoad = null;

function loadMapModules() {
  if (!mapLoad) {
    mapLoad = Promise.all([
      import("./live.js"),
      import("./fleetmap.js"),
      import("./export.js"),
    ]).then(([live, fleetmap, exp]) => (mapMod = { live, fleetmap, exp }));
  }
  return mapLoad;
}

function loadCompare() {
  if (!compareLoad) {
    compareLoad = import("./compare.js").then((m) => (compareMod = m));
  }
  return compareLoad;
}

function loadExperiment() {
  if (!expLoad) {
    expLoad = import("./experiment.js").then((m) => (expMod = m));
  }
  return expLoad;
}

function loadInsight() {
  if (!insightLoad) {
    insightLoad = import("./insight.js").then((m) => (insightMod = m));
  }
  return insightLoad;
}

function loadValidation() {
  if (!validationLoad) {
    validationLoad = import("./validation.js").then((m) => (validationMod = m));
  }
  return validationLoad;
}

function moduleFailed(mountSel, err) {
  const mount = $(mountSel);
  if (mount) {
    mount.textContent = "";
    const p = document.createElement("p");
    p.className = "railnote err";
    p.textContent = "this view's module failed to load: " + String(err);
    mount.appendChild(p);
  }
}

async function openCompare(ids) {
  try {
    await loadCompare();
  } catch (err) {
    moduleFailed("#compareBody", err);
    return;
  }
  // identity, not equality: `ids` IS route.ids, so a navigation that
  // happened while the module loaded cannot mount the stale selection
  if (route.view === "compare" && route.ids === ids) compareMod.mountCompare(ids);
}

async function openInsight(id) {
  try {
    await loadInsight();
  } catch (err) {
    moduleFailed("#insightMount", err);
    return;
  }
  if (route.view === "run" && route.id === id && route.mode === "insight") {
    insightMod.mountInsight(id);
  }
}

async function openValidation() {
  try {
    await loadValidation();
  } catch (err) {
    moduleFailed("#validationBody", err);
    return;
  }
  if (route.view === "validation") validationMod.mountValidation();
}

async function openSweep(id) {
  try {
    await loadExperiment();
  } catch (err) {
    moduleFailed("#sweepBody", err);
    return;
  }
  if (route.view === "sweep" && route.id === id) expMod.mountSweep(id);
}

async function openExplore() {
  try {
    await loadExperiment();
  } catch (err) {
    moduleFailed("#axisList", err);
    return;
  }
  if (route.view === "new" && route.explore) expMod.mountExplore();
}

/* ------------------------------------------------------------------ *
 * runs list (left rail) — polled every 3 s
 * ------------------------------------------------------------------ */
const RUNS_POLL_MS = 3000;
let runsCache = [];
let runsJSON = "";
let runsFetchInFlight = false;

function statusStats(run) {
  if (run.status === "done" && run.headline) {
    const h = run.headline;
    const parts = [];
    if (h.occupancy != null) parts.push("occ " + fmtPct(h.occupancy));
    if (h.goodput != null) parts.push("gp " + fmtPct(h.goodput));
    if (h.jobs_finished != null) parts.push(h.jobs_finished + " jobs");
    return parts.join(" · ");
  }
  return run.status;
}

function rowMetaText(run) {
  return [fmtWhen(run.created), statusStats(run)].filter(Boolean).join(" · ");
}

/** A sweep cell's grid values as a short label — the part of a generated
    title that actually distinguishes one cell from another. */
export function cellLabel(cell) {
  if (!cell || typeof cell !== "object") return "";
  return Object.keys(cell)
    .map((k) => k.split(".").pop() + "=" + JSON.stringify(cell[k]))
    .join(", ");
}

/** Full text for a truncated run label: the title, plus the sweep cell
    when there is one (the title's tail is what a sweep varies, and that
    tail is exactly what gets ellipsised). */
function runTooltip(run) {
  const parts = [run.title || run.id];
  const cell = cellLabel(run.sweep_cell);
  if (cell) parts.push(cell);
  parts.push(run.id);
  return parts.join("\n");
}

/* Relative timestamps ("3m ago") age even when /api/runs is unchanged;
   rewrite just the .rmeta text in place so DOM and focus stay intact. */
function refreshRunTimes() {
  const byId = new Map(runsCache.map((r) => [r.id, r]));
  for (const row of $("#runsList").querySelectorAll(".runrow")) {
    const run = byId.get(row.dataset.runId);
    const meta = row.querySelector(".rmeta");
    if (run && meta) meta.textContent = rowMetaText(run);
  }
  if (route.view === "run") {
    const run = byId.get(route.id);
    if (run && run.created != null) {
      $("#runMeta").textContent = fmtWhen(run.created);
    }
  }
}

/* ---- multi-select for #compare -------------------------------------- *
 * Selection lives in a Set of run ids, NOT in the DOM: the rail
 * re-renders whenever /api/runs changes, so anything kept in the markup
 * would be lost every 3 s.  MAX_COMPARE is the run palette's size —
 * a 9th line would have to reuse a color, so the cap is honest and the
 * refusal says why instead of silently dropping a run.
 * ------------------------------------------------------------------ */
const MAX_COMPARE = 8;
const selected = new Set();
let selecting = false; // the "compare" toggle: checkboxes always visible
let selAnchorId = null; // shift-click range anchor

function selectedIds() {
  // rail order (newest first), not click order: the compare view's run
  // letters then match the rail top-to-bottom.
  const ids = runsCache.map((r) => r.id).filter((id) => selected.has(id));
  for (const id of selected) if (!ids.includes(id)) ids.push(id);
  return ids;
}

function setSelNote(text) {
  const el = $("#selNote");
  el.textContent = text || "";
  el.classList.toggle("hidden", !text);
}

function applySelection(ids, want) {
  let capped = false;
  for (const id of ids) {
    if (want) {
      if (selected.has(id)) continue;
      if (selected.size >= MAX_COMPARE) {
        capped = true;
        continue;
      }
      selected.add(id);
    } else {
      selected.delete(id);
    }
  }
  setSelNote(
    capped
      ? "Compare holds " + MAX_COMPARE + " runs (one per line color) —" +
        " deselect one first."
      : ""
  );
}

function onSelClick(ev) {
  const box = ev.currentTarget;
  const id = box.dataset.runId;
  const want = box.checked; // state the native toggle just produced
  let ids = [id];
  if (ev.shiftKey && selAnchorId && selAnchorId !== id) {
    const order = runsCache.map((r) => r.id);
    const from = order.indexOf(selAnchorId);
    const to = order.indexOf(id);
    if (from >= 0 && to >= 0) {
      ids = order.slice(Math.min(from, to), Math.max(from, to) + 1);
    }
  }
  applySelection(ids, want);
  selAnchorId = id;
  syncSelectionUI();
}

function syncSelectionUI() {
  for (const box of $("#runsList").querySelectorAll(".rsel")) {
    box.checked = selected.has(box.dataset.runId);
  }
  const n = selected.size;
  const ids = selectedIds();
  $("#selCount").textContent = n ? n + " selected" : "none selected";
  $("#railSelect").classList.toggle("hidden", !selecting && n === 0);
  const btn = $("#compareBtn");
  btn.textContent = n >= 2 ? "Compare " + n + " runs" : "Compare";
  if (n >= 2) {
    btn.href = "#compare/" + ids.map(encodeURIComponent).join(",");
    btn.removeAttribute("aria-disabled");
    btn.title = "Open these " + n + " runs side by side";
  } else {
    btn.href = "#compare/";
    btn.setAttribute("aria-disabled", "true");
    btn.title = "Select at least two runs";
  }
  $("#clearSelBtn").disabled = n === 0;
}

function toggleSelecting(on) {
  selecting = on;
  $("#selectToggle").setAttribute("aria-pressed", on ? "true" : "false");
  $("#rail").classList.toggle("selecting", on);
  if (!on) setSelNote("");
  syncSelectionUI();
}

function buildRunRow(run) {
  const li = document.createElement("li");
  li.className = "runrow";
  li.dataset.runId = run.id;

  const box = document.createElement("input");
  box.type = "checkbox";
  box.className = "rsel";
  box.dataset.runId = run.id;
  box.checked = selected.has(run.id);
  box.setAttribute(
    "aria-label",
    "Select " + (run.title || run.id) + " for comparison"
  );
  box.addEventListener("click", onSelClick);
  li.appendChild(box);

  const a = document.createElement("a");
  a.href = "#run/" + encodeURIComponent(run.id) + "/report";

  const dot = document.createElement("span");
  dot.className = "dot " + run.status;
  a.appendChild(dot);

  const main = document.createElement("span");
  main.className = "rmain";
  const title = document.createElement("span");
  title.className = "rtitle";
  title.textContent = run.title || run.id;
  /* The rail truncates at ~187 px and a sweep's generated titles are
     600+ px that differ only in their TAIL, so four cells read as four
     copies of one string. The tooltip carries the whole thing, and the
     short cell label leads with the part that distinguishes them. */
  title.title = runTooltip(run);
  const meta = document.createElement("span");
  meta.className = "rmeta";
  meta.textContent = rowMetaText(run);
  main.appendChild(title);
  main.appendChild(meta);
  a.appendChild(main);
  li.appendChild(a);

  if (run.sweep_id) {
    // sweep cells are ordinary runs; this is the way back to the grid
    const sw = document.createElement("a");
    sw.className = "sweepbadge";
    sw.href = "#sweep/" + encodeURIComponent(run.sweep_id);
    sw.textContent = "S";
    sw.title = "Cell of sweep " + run.sweep_id;
    sw.setAttribute("aria-label", "Open the sweep this cell belongs to");
    li.appendChild(sw);
  }

  if (run.status === "queued") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dequeue";
    btn.textContent = "×";
    btn.title = "Remove this queued run";
    btn.setAttribute("aria-label", "Remove queued run " + (run.title || run.id));
    btn.addEventListener("click", () => dequeueRun(run.id));
    li.appendChild(btn);
  }
  return li;
}

function renderRunsList() {
  const list = $("#runsList");
  const active = document.activeElement;
  let refocus = null;
  if (active && list.contains(active)) {
    const row = active.closest(".runrow");
    if (row) {
      refocus = {
        id: row.dataset.runId,
        part: active.classList.contains("dequeue")
          ? ".dequeue"
          : active.classList.contains("rsel")
            ? ".rsel"
            : active.classList.contains("sweepbadge")
              ? ".sweepbadge"
              : "a",
      };
    }
  }

  // a selected run that left the workspace (dequeued, deleted) leaves
  // the selection too — a compare link to a gone id is a dead link
  const present = new Set(runsCache.map((r) => r.id));
  for (const id of [...selected]) if (!present.has(id)) selected.delete(id);
  if (selAnchorId && !present.has(selAnchorId)) selAnchorId = null;

  list.textContent = "";
  for (const run of runsCache) list.appendChild(buildRunRow(run));

  $("#runsCount").textContent = runsCache.length ? String(runsCache.length) : "";
  $("#railEmpty").classList.toggle("hidden", runsCache.length > 0);
  updateSelection();
  syncSelectionUI();

  if (refocus) {
    const row = list.querySelector('[data-run-id="' + CSS.escape(refocus.id) + '"]');
    if (row) {
      const el = row.querySelector(refocus.part);
      if (el) el.focus();
    }
  }
}

function updateSelection() {
  const current = route.view === "run" ? route.id : null;
  for (const a of $("#runsList").querySelectorAll(".runrow > a")) {
    const id = a.closest(".runrow").dataset.runId;
    if (id === current) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
}

async function refreshRuns() {
  if (runsFetchInFlight) return;
  runsFetchInFlight = true;
  const { ok, doc } = await apiGet("/api/runs");
  runsFetchInFlight = false;
  const errEl = $("#railError");
  if (!ok || !Array.isArray(doc)) {
    errEl.textContent = "Cannot reach the server. Is fleetsim serve still running?";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");
  const json = JSON.stringify(doc);
  if (json === runsJSON) return; // nothing changed: keep DOM (and focus) intact
  runsJSON = json;
  runsCache = doc;
  renderRunsList();
}

async function dequeueRun(id) {
  const run = runsCache.find((r) => r.id === id);
  const label = run ? run.title || run.id : id;
  if (!window.confirm('Remove queued run "' + label + '"? It has not started yet.')) return;
  await apiDelete("/api/runs/" + encodeURIComponent(id));
  runsJSON = ""; // force re-render on next poll
  await refreshRuns();
  if (route.view === "run" && route.id === id) location.hash = "";
}

/* ------------------------------------------------------------------ *
 * live fleet map — the 2D counterpart of the live 3D view.  It rides
 * the same GET /api/runs/{id}/live stream and sits under the progress
 * panel, so the DEFAULT tab of a running run already shows the fleet
 * filling up.  Feed and DOM are torn down the moment the route leaves.
 * ------------------------------------------------------------------ */
let liveMapFeed = null;
let liveMapRunId = null;
let liveMapSvg = null;
/* Mount token, the same guard fleet3d uses.  `liveMapRunId === id` alone
   is NOT idempotence: startLiveMap is called from the 1 Hz progress tick
   and awaits a module import, so a tick arriving during a COLD import
   sees the id already claimed but no feed yet, starts a second feed, and
   the first invocation then overwrites `liveMapFeed` — leaving feed B
   polling /live at 1 Hz for the rest of the session with nothing holding
   a reference to stop it.  A token invalidates every earlier mount. */
let liveMapToken = 0;

function stopLiveMap() {
  liveMapToken++;
  if (liveMapFeed) liveMapFeed.stop();
  liveMapFeed = null;
  liveMapRunId = null;
  liveMapSvg = null;
  const panel = $("#liveMap");
  if (panel) panel.classList.add("hidden");
}

async function startLiveMap(id) {
  if (liveMapRunId === id) return; // already mounted OR mounting
  stopLiveMap();
  liveMapRunId = id;
  const token = ++liveMapToken;
  try {
    await loadMapModules();
  } catch (err) {
    if (token === liveMapToken) liveMapRunId = null;
    return;
  }
  if (token !== liveMapToken || route.view !== "run" || route.id !== id) return;
  const { LiveFeed, LIVE_MAX_ROWS, LIVE_ROW_LIMIT, livePalette } = mapMod.live;
  const { renderFleetMap, liveSpec } = mapMod.fleetmap;
  const { pngButton, safeName } = mapMod.exp;

  const feed = new LiveFeed(id, {
    onUpdate: (up) => {
      if (token !== liveMapToken) return;
      const t = up.progress ? up.progress.t_us : 0;
      const spec = liveSpec(up.stints, t, livePalette(up.stints.buckets()));
      const panel = $("#liveMap");
      if (!spec) {
        // no stint output in this scenario: the honest "no map" signal,
        // the same one the finished report gives
        panel.classList.add("hidden");
        return;
      }
      panel.classList.remove("hidden");
      liveMapSvg = renderFleetMap($("#liveMapMount"), spec);
      $("#liveMapSub").textContent =
        "T " + fmtClock(t) + " · " + up.stints.nSettled + " settled · " +
        up.stints.nOpen + " running" + (up.more ? " · catching up…" : "");
      /* EVERY way this map can be less than the whole fleet, said out
         loud.  A partly-empty fleet drawn with a confident caption is
         indistinguishable from an idle one. */
      let warn = "";
      if (up.stalled) warn = up.stalled;
      else if (up.truncated) {
        warn =
          "Truncated: the live feed retains " + LIVE_MAX_ROWS.toLocaleString() +
          " stint rows and this run has settled more, so work after that" +
          " point is not drawn. The finished run's report shows all of it.";
      } else if (up.openTruncated) {
        warn =
          "Incomplete: more than " + LIVE_ROW_LIMIT.toLocaleString() +
          " stints are running at once and the server caps the overlay it" +
          " streams, so some running work is not drawn.";
      } else if (up.openPending) {
        warn =
          "Running stints appear from the next metrics flush — the server" +
          " only spools the running-stint overlay while a client is" +
          " watching, and this view just started watching.";
      }
      const warnEl = $("#liveMapWarn");
      warnEl.textContent = warn;
      warnEl.classList.toggle("hidden", !warn);
      // the SVG is rebuilt every flush, so the export button is too — a
      // button holding the previous frame's node would export a stale map
      const tools = $("#liveMapTools");
      tools.textContent = "";
      if (liveMapSvg) {
        tools.appendChild(
          pngButton(liveMapSvg, safeName(id, "run") + "-live-fleet.png")
        );
      }
    },
  });
  if (token !== liveMapToken) return;
  liveMapFeed = feed;
  feed.start();
}

/* ------------------------------------------------------------------ *
 * run view: toolbar + report iframe + live progress + 3D placeholder
 * ------------------------------------------------------------------ */
const PROGRESS_POLL_MS = 1000;
let progressTimer = null;
let progressRunId = null;
let loadedReportId = null; // which run's report the iframe currently holds

function stopProgressPoll() {
  if (progressTimer !== null) {
    clearTimeout(progressTimer);
    progressTimer = null;
  }
  progressRunId = null;
}

function setRunBodyPanel(name) {
  // progress | failed | missing | unreachable | report | fleet3d | insight | none
  $("#runProgress").classList.toggle("hidden", name !== "progress");
  $("#runFailed").classList.toggle("hidden", name !== "failed");
  $("#runMissing").classList.toggle("hidden", name !== "missing");
  $("#runUnreachable").classList.toggle("hidden", name !== "unreachable");
  $("#reportFrame").classList.toggle("hidden", name !== "report");
  $("#fleet3d").classList.toggle("hidden", name !== "fleet3d");
  $("#insight").classList.toggle("hidden", name !== "insight");
  // the live map lives INSIDE the progress panel's stack; it hides with
  // it and re-shows on the next flush
  if (name !== "progress") $("#liveMap").classList.add("hidden");
  if (name !== "fleet3d" && fleet3dMod) fleet3dMod.hideFleet3d();
  if (name !== "insight" && insightMod) insightMod.unmountInsight();
}

function renderToolbar(id, info) {
  const titleEl = $("#runTitle");
  titleEl.textContent = info ? info.title || id : id;
  // 563 px of box for 832 px of text: without this you cannot tell which
  // sweep cell the page you are on belongs to
  titleEl.title = info ? runTooltip({ ...info, id }) : id;
  $("#runMeta").textContent = info && info.created != null ? fmtWhen(info.created) : "";
  const dot = $("#runDot");
  dot.className = "dot" + (info ? " " + info.status : "");

  const base = "#run/" + encodeURIComponent(id) + "/";
  const tabs = {
    report: $("#tabReport"),
    fleet3d: $("#tab3d"),
    insight: $("#tabInsight"),
  };
  for (const mode of Object.keys(tabs)) {
    tabs[mode].href = base + mode;
    if (route.mode === mode) tabs[mode].setAttribute("aria-current", "true");
    else tabs[mode].removeAttribute("aria-current");
  }

  const dl = $("#downloadReport");
  if (info && info.status === "done") {
    dl.href = "/api/runs/" + encodeURIComponent(id) + "/report";
    dl.setAttribute("download", id + "-report.html");
    dl.classList.remove("hidden");
  } else {
    dl.classList.add("hidden");
    dl.removeAttribute("href");
  }
}

function showReport(id) {
  const frame = $("#reportFrame");
  if (loadedReportId !== id) {
    frame.src = "/api/runs/" + encodeURIComponent(id) + "/report";
    loadedReportId = id;
  }
  setRunBodyPanel("report");
}

/** Copy for a queued run, INCLUDING its place in the queue.
    v0.8 runs several simulations at once, so "runs execute one at a
    time" is now false — and it was on screen next to a rail showing
    three of them running.  The position is already in the /progress
    payload; the run's own page was the last surface not showing it. */
function queuedSubtitle(prog) {
  const pos = prog && prog.queue_position;
  if (typeof pos === "number" && pos > 0) {
    return (
      "queued — position " + pos + " in the run queue (runs execute in" +
      " parallel worker processes; this one starts when a worker frees up)"
    );
  }
  return "queued — waiting for a free worker process";
}

function renderProgressSnapshot(status, prog, doc) {
  $("#progressHead").textContent = status === "queued" ? "Queued" : "Running";
  $("#progressSub").textContent =
    status === "queued"
      ? queuedSubtitle(doc)
      : prog
        ? "live — one update per metrics flush"
        : "starting — waiting for the first metrics flush";
  $("#cancelRunBtn").classList.toggle("hidden", status !== "running");

  const track = $("#progressTrack");
  const fill = $("#progressFill");
  let fraction = 0;
  if (prog && prog.horizon_us > 0) fraction = Math.min(1, prog.t_us / prog.horizon_us);
  fill.style.width = (fraction * 100).toFixed(2) + "%";
  track.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));

  $("#statTime").textContent = prog
    ? fmtClock(prog.t_us) + " / " + fmtClock(prog.horizon_us)
    : "–";
  $("#statOcc").textContent = prog ? fmtPct(prog.occupancy_to_date) : "–";
  $("#statFinished").textContent = prog ? fmtInt(prog.jobs_finished) : "–";
  $("#statRunning").textContent = prog ? fmtInt(prog.jobs_running) : "–";
  $("#statPending").textContent = prog ? fmtInt(prog.pending) : "–";
  const chipsEl = $("#statChips");
  chipsEl.textContent =
    prog && prog.allocated_chips != null
      ? prog.allocated_chips + (prog.healthy_chips != null ? " / " + prog.healthy_chips : "")
      : "–";
  // allocated > healthy is real, not corruption: jobs on draining/failed
  // nodes stay allocated through their grace window — say so.
  const over =
    prog && prog.healthy_chips != null && prog.allocated_chips > prog.healthy_chips;
  chipsEl.title = over
    ? "allocated exceeds healthy while grace-period jobs finish on draining/failed nodes"
    : "";
  $("#progressNote").textContent =
    "The report opens automatically when the run finishes." +
    (over
      ? " Chips allocated can exceed healthy chips while grace-period jobs finish on draining nodes."
      : "");
}

async function showFailed(id) {
  const { doc } = await apiGet("/api/runs/" + encodeURIComponent(id));
  if (route.view !== "run" || route.id !== id) return;
  $("#failError").textContent = (doc && doc.error) || "no error recorded";
  setRunBodyPanel("failed");
}

function startProgressPoll(id) {
  stopProgressPoll();
  progressRunId = id;

  const tick = async () => {
    if (progressRunId !== id) return;
    const { ok, status, doc } = await apiGet(
      "/api/runs/" + encodeURIComponent(id) + "/progress"
    );
    if (progressRunId !== id) return; // route moved on while awaiting
    if (!ok) {
      if (status === 404) {
        stopProgressPoll();
        renderToolbar(id, null);
        setRunBodyPanel("missing");
        return;
      }
      progressTimer = setTimeout(tick, PROGRESS_POLL_MS); // transient; retry
      return;
    }
    const st = doc.status;
    $("#runDot").className = "dot " + st;
    if (st === "done") {
      stopProgressPoll();
      stopLiveMap();
      runsJSON = "";
      refreshRuns();
      const info = await fetchRunInfo(id);
      if (route.view === "run" && route.id === id) {
        renderToolbar(id, info || { title: id, status: "done", created: null });
        if (route.mode === "fleet3d") {
          setRunBodyPanel("fleet3d");
          // no `live`: this remounts the view on the finished model
          openFleet3d(id, { deep: route.deep });
        } else if (route.mode === "insight") {
          setRunBodyPanel("insight");
          openInsight(id);
        } else if (route.mode === "report") {
          showReport(id);
        }
      }
      return;
    }
    if (st === "failed") {
      stopProgressPoll();
      stopLiveMap();
      runsJSON = "";
      refreshRuns();
      if (route.view === "run" && route.id === id) {
        $("#runDot").className = "dot failed";
        await showFailed(id);
      }
      return;
    }
    renderProgressSnapshot(st, doc.progress, doc);
    // queued -> running is when the live surfaces can start: the 3D tab
    // switches itself to live mode, the report tab grows a live map
    if (st === "running" && route.view === "run" && route.id === id) {
      if (route.mode === "fleet3d") {
        setRunBodyPanel("fleet3d");
        openFleet3d(id, { live: true, deep: route.deep });
      } else if (route.mode === "report") {
        startLiveMap(id);
      }
    }
    progressTimer = setTimeout(tick, PROGRESS_POLL_MS);
  };
  tick();
}

async function fetchRunInfo(id) {
  const { ok, doc } = await apiGet("/api/runs/" + encodeURIComponent(id));
  return ok ? doc : null;
}

/* A transient server outage must NOT render as "No such run" (review
   fix): only a real 404 means the run is gone.  Anything else (network
   error, 5xx) shows the unreachable panel and retries. */
let runRetryTimer = null;

function clearRunRetry() {
  if (runRetryTimer !== null) {
    clearTimeout(runRetryTimer);
    runRetryTimer = null;
  }
}

function scheduleRunRetry() {
  clearRunRetry();
  const { id, mode } = route;
  runRetryTimer = setTimeout(() => {
    runRetryTimer = null;
    if (route.view === "run" && route.id === id && route.mode === mode) {
      renderRunView();
    }
  }, RUNS_POLL_MS);
}

async function renderRunView() {
  const { id, mode } = route;
  showView("run");
  clearRunRetry();
  const { ok, status, doc } = await apiGet("/api/runs/" + encodeURIComponent(id));
  if (route.view !== "run" || route.id !== id || route.mode !== mode) return;
  const info = ok ? doc : null;
  renderToolbar(id, info);
  updateSelection();

  if (!ok) {
    stopProgressPoll();
    if (status === 404) {
      setRunBodyPanel("missing");
    } else {
      setRunBodyPanel("unreachable");
      scheduleRunRetry();
    }
    return;
  }
  if (mode === "fleet3d" && info.status === "done") {
    stopProgressPoll();
    stopLiveMap();
    setRunBodyPanel("fleet3d");
    openFleet3d(id, { deep: route.deep });
    return;
  }
  // LIVE 3D: a running run has no model, so the 3D view reads the stint
  // stream instead.  The progress poll stays up — it is what notices the
  // run finishing and re-mounts the view on the finished model.
  if (mode === "fleet3d" && info.status === "running") {
    stopLiveMap();
    setRunBodyPanel("fleet3d");
    openFleet3d(id, { live: true, deep: route.deep });
    if (progressRunId !== id) startProgressPoll(id);
    return;
  }
  if (mode === "insight" && info.status === "done") {
    stopProgressPoll();
    stopLiveMap();
    setRunBodyPanel("insight");
    openInsight(id);
    return;
  }
  // report mode — or a 3D / analysis tab on a run that has not finished
  // yet, which gets the same progress / failure panels as the report tab
  if (info.status === "done") {
    stopProgressPoll();
    stopLiveMap();
    showReport(id);
  } else if (info.status === "failed") {
    stopProgressPoll();
    stopLiveMap();
    $("#failError").textContent = info.error || "no error recorded";
    setRunBodyPanel("failed");
  } else {
    renderProgressSnapshot(info.status, null, info);
    setRunBodyPanel("progress");
    if (progressRunId !== id) startProgressPoll(id);
    if (info.status === "running") startLiveMap(id);
    else stopLiveMap();
  }
}

/* ------------------------------------------------------------------ *
 * scenario editor (#new)
 * ------------------------------------------------------------------ */
const DEFAULT_YAML = `# fleetsim starter — small enough to finish in seconds.
# Validate first; Run executes it in the server workspace.

sim: {horizon: 2h, round: 60s, seed: 42}

fleet:
  metro: demo
  clusters:
    - name: h100-demo
      chip: {type: h100, per_node: 8}
      topology: {levels: [rack, node], counts: [2, 8]}  # 128 chips

failure_model: {node_mtbf_days: 0, maintenance_rate_per_node_month: 0}

workload:
  kind: synthetic
  classes:
    finetune:
      rate_per_hour: 6
      chips: pow2[8, 32]
      duration: lognormal[median=20m, p90=1h]
      tier: batch
    eval:
      rate_per_hour: 30
      chips: pow2[1, 8]
      duration: lognormal[median=2m, p90=10m]
      tier: batch

scheduler: {name: tiered_priority, params: {preempt: requeue}}

# stints: true feeds the report's fleet map (who-ran-where-when).
outputs: {stints: true}
`;

let editorReady = false;
let editorDirty = false;
let examplesByName = new Map();
let requestInFlight = false;

function gutterSync() {
  const box = $("#yamlBox");
  const gutter = $("#gutter");
  const lines = box.value.split("\n").length;
  let n = gutter.childElementCount;
  while (n < lines) {
    gutter.appendChild(document.createElement("span"));
    n++;
  }
  while (n > lines) {
    gutter.removeChild(gutter.lastElementChild);
    n--;
  }
  gutter.scrollTop = box.scrollTop;
}

function setEditorText(text) {
  const box = $("#yamlBox");
  box.value = text;
  box.setSelectionRange(0, 0); // open at the top, not the end
  box.scrollTop = 0;
  editorDirty = false;
  clearValidation();
  gutterSync();
}

function clearValidation() {
  $("#valOk").classList.add("hidden");
  const ul = $("#valErrors");
  ul.textContent = "";
  ul.classList.add("hidden");
  $("#editorStatus").textContent = "";
}

function showErrors(errors) {
  $("#valOk").classList.add("hidden");
  const ul = $("#valErrors");
  ul.textContent = "";
  for (const e of errors) {
    const li = document.createElement("li");
    li.textContent = String(e); // validator messages, verbatim
    ul.appendChild(li);
  }
  ul.classList.remove("hidden");
}

function setBusy(busy, label) {
  requestInFlight = busy;
  $("#validateBtn").disabled = busy;
  $("#runBtn").disabled = busy;
  $("#editorStatus").textContent = busy ? label : "";
}

async function doValidate() {
  if (requestInFlight) return;
  clearValidation();
  setBusy(true, "Validating…");
  const { ok, doc } = await apiPost("/api/validate", { yaml: $("#yamlBox").value });
  setBusy(false);
  if (!ok || !doc) {
    showErrors([(doc && doc.error) || "the server did not answer — is it still running?"]);
    return;
  }
  if (doc.ok) $("#valOk").classList.remove("hidden");
  else showErrors(doc.errors || ["invalid scenario"]);
}

async function doRun() {
  if (requestInFlight) return;
  clearValidation();
  setBusy(true, "Submitting…");
  const body = { yaml: $("#yamlBox").value };
  const title = $("#titleInput").value.trim();
  if (title) body.title = title;
  const { ok, status, doc } = await apiPost("/api/runs", body);
  setBusy(false);
  if (ok && doc && doc.id) {
    runsJSON = "";
    refreshRuns();
    location.hash = "#run/" + encodeURIComponent(doc.id) + "/report";
    return;
  }
  if (status === 400 && doc && Array.isArray(doc.errors)) showErrors(doc.errors);
  else showErrors([(doc && doc.error) || "the server did not answer — is it still running?"]);
}

const exampleNotes = new Map();

async function loadExamples() {
  const { ok, doc } = await apiGet("/api/examples");
  if (!ok || !Array.isArray(doc)) return;
  const select = $("#tplSelect");
  for (const ex of doc) {
    if (!ex || typeof ex.name !== "string" || typeof ex.yaml !== "string") continue;
    examplesByName.set(ex.name, ex.yaml);
    // A starter that cannot run as web-submitted (e.g. a relative trace
    // path) says so — but BELOW the control, not in the option label: an
    // option's text sets the <select>'s intrinsic width, and this one
    // caveat made the control 583 px wide and pushed it off the panel at
    // every viewport under ~940 px, clipped with no scrollbar.
    if (ex.runnable === false && typeof ex.note === "string") {
      exampleNotes.set(ex.name, ex.note);
    }
    const opt = document.createElement("option");
    opt.value = ex.name;
    opt.textContent =
      exampleNotes.has(ex.name) ? ex.name + " (see note)" : ex.name;
    select.appendChild(opt);
  }
}

function showTemplateNote(name) {
  const el = $("#tplNote");
  const note = exampleNotes.get(name);
  el.textContent = note ? name + ": " + note : "";
  el.classList.toggle("hidden", !note);
}

/* ---- fleet-shape preview ------------------------------------------- *
 * Debounced POST /api/preview.  The browser has no YAML parser and the
 * app ships no new dependency, so the shape is computed server-side
 * (arithmetically, from the count tree — nothing is materialized) and
 * drawn with the same block-diagram code the live fleet map uses.
 *
 * NOT /api/validate: that route runs the feasibility pass, which BUILDS
 * the declared fleet (measured 1.9 s at the 262,144-node ceiling and 89 s
 * for a 100,000-rack tree) inside the HTTP handler thread.  Firing that
 * on every typing pause, with no in-flight guard, piled concurrent
 * CPU-bound requests onto a ThreadingHTTPServer and dragged the rail's
 * own poll from 0.1 s to 4.9 s.  /api/preview stops at parse + schema.
 *
 * ONE REQUEST AT A TIME.  A second pause while one is outstanding just
 * marks the preview dirty; the in-flight response re-runs at the end.
 *
 * It deliberately does NOT drive the big error list: that belongs to the
 * explicit Validate button, and a full error dump on every keystroke is
 * noise.  The preview reports "not valid yet" plus the first problems.
 * ------------------------------------------------------------------ */
const PREVIEW_DEBOUNCE_MS = 700;
let previewTimer = null;
let previewSeq = 0;
let previewText = null;
let previewInFlight = false;
let previewAgain = false;

function schedulePreview() {
  if (previewTimer !== null) clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, PREVIEW_DEBOUNCE_MS);
}

async function runPreview() {
  previewTimer = null;
  if (previewInFlight) {
    previewAgain = true; // coalesce: re-run once, with the latest text
    return;
  }
  const text = $("#yamlBox").value;
  if (text === previewText) return; // nothing changed since the last draw
  const seq = ++previewSeq;
  let mod;
  previewInFlight = true;
  try {
    mod = await loadMapModules();
  } catch (err) {
    previewInFlight = false;
    return;
  }
  const { ok, doc } = await apiPost("/api/preview", { yaml: text });
  previewInFlight = false;
  if (previewAgain) {
    previewAgain = false;
    schedulePreview();
  }
  if (seq !== previewSeq || route.view !== "new") return;
  previewText = text;
  const mount = $("#previewMount");
  const note = $("#previewNote");
  const tools = $("#previewTools");
  tools.textContent = "";
  if (!ok || !doc) {
    mount.textContent = "";
    $("#previewSub").textContent = "";
    note.textContent = "the server did not answer — is it still running?";
    return;
  }
  if (!doc.ok || !doc.fleet) {
    mount.textContent = "";
    /* "not a scenario yet", not "invalid": this route checks parse and
       schema only, so a document that gets here may still fail the full
       gate (and Validate is what says so). */
    $("#previewSub").textContent = "not a valid scenario document yet";
    const errs = (doc.errors || []).slice(0, 3);
    note.textContent = errs.length
      ? errs.join(" · ") +
        ((doc.errors || []).length > 3
          ? " · +" + (doc.errors.length - 3) + " more — press Validate"
          : "")
      : "this scenario declares no fleet to draw";
    return;
  }
  const shape = doc.fleet;
  $("#previewSub").textContent =
    shape.n_clusters + (shape.n_clusters === 1 ? " cluster · " : " clusters · ") +
    shape.total_chips.toLocaleString() + " chips · " +
    shape.total_nodes.toLocaleString() + " nodes" +
    (shape.chip_types.length ? " · " + shape.chip_types.join(", ") : "");
  const spec = mapMod.fleetmap.shapeSpec(shape);
  const svg = mapMod.fleetmap.renderFleetMap(mount, spec);
  note.textContent = "";
  if (svg) {
    tools.appendChild(mapMod.exp.pngButton(svg, "fleet-shape.png"));
  }
}

function initEditor() {
  if (editorReady) return;
  editorReady = true;
  const box = $("#yamlBox");
  setEditorText(DEFAULT_YAML);

  box.addEventListener("input", () => {
    editorDirty = true;
    /* An explicit Validate result belongs to the TEXT IT WAS COMPUTED
       FROM.  Without this the editor shows a red error list for a
       document that no longer exists beside a green preview of the one
       that does — and, worse, a green "Scenario is valid." survives an
       edit that breaks it. */
    clearValidation();
    gutterSync();
    schedulePreview();
  });
  box.addEventListener("scroll", () => {
    $("#gutter").scrollTop = box.scrollTop;
  });

  $("#tplSelect").addEventListener("change", (ev) => {
    const name = ev.target.value;
    if (!name) {
      showTemplateNote("");
      return;
    }
    const yaml = examplesByName.get(name);
    if (yaml == null) return;
    if (
      editorDirty &&
      box.value.trim() !== "" &&
      !window.confirm("Replace the editor contents with the “" + name + "” starter?")
    ) {
      ev.target.value = "";
      return;
    }
    setEditorText(yaml);
    showTemplateNote(name);
    schedulePreview();
  });

  $("#validateBtn").addEventListener("click", doValidate);
  $("#runBtn").addEventListener("click", doRun);
  schedulePreview(); // draw the starter template's fleet straight away
  $("#view-new").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      doValidate();
    }
  });

  loadExamples();
}

/* ------------------------------------------------------------------ *
 * home: the sweep list (runs are the rail's job; sweeps need a home)
 * ------------------------------------------------------------------ */
async function refreshHomeSweeps() {
  const { ok, doc } = await apiGet("/api/sweeps");
  if (route.view !== "home") return;
  const wrap = $("#homeSweeps");
  const list = $("#homeSweepList");
  if (!ok || !Array.isArray(doc) || doc.length === 0) {
    wrap.classList.add("hidden");
    list.textContent = "";
    return;
  }
  list.textContent = "";
  for (const sw of doc.slice(0, 8)) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#sweep/" + encodeURIComponent(sw.sweep_id);
    a.textContent = sw.title || sw.sweep_id;
    li.appendChild(a);
    const meta = document.createElement("span");
    meta.className = "sub mono";
    const bits = [sw.n_done + "/" + sw.n_runs + " done"];
    if (sw.n_failed) bits.push(sw.n_failed + " failed");
    bits.push(fmtWhen(sw.created));
    meta.textContent = " " + bits.join(" · ");
    li.appendChild(meta);
    list.appendChild(li);
  }
  $("#homeSweepsSub").textContent =
    doc.length > 8 ? "8 most recent of " + doc.length : "";
  wrap.classList.remove("hidden");
}

/* ------------------------------------------------------------------ *
 * editor mode: #new is a single run, #sweep is the same editor with the
 * parameter-axis panel open (experiment.js owns that panel)
 * ------------------------------------------------------------------ */
function setEditorMode(explore) {
  $("#explorePanel").classList.toggle("hidden", !explore);
  $("#runBtn").classList.toggle("hidden", explore);
  $("#launchSweepBtn").classList.toggle("hidden", !explore);
  $("#editorHead").textContent = explore ? "Explore" : "New run";
  $("#editorSub").textContent = explore
    ? "One base scenario plus parameter axes — every cell is validated before any run is created."
    : "Scenario YAML — validated server-side before anything executes.";
  const single = $("#modeSingle");
  const exp = $("#modeExplore");
  if (explore) {
    exp.setAttribute("aria-current", "true");
    single.removeAttribute("aria-current");
  } else {
    single.setAttribute("aria-current", "true");
    exp.removeAttribute("aria-current");
  }
}

/* ------------------------------------------------------------------ *
 * boot
 * ------------------------------------------------------------------ */
function leaveHeavyViews(keep) {
  if (keep !== "compare" && compareMod) compareMod.unmountCompare();
  if (keep !== "sweep" && expMod) expMod.unmountSweep();
}

function onRoute() {
  route = parseRoute();
  if (route.view === "new") {
    stopProgressPoll();
    clearRunRetry();
    leaveHeavyViews("new");
    initEditor();
    setEditorMode(route.explore);
    showView("new");
    updateSelection();
    if (route.explore) openExplore();
    else $("#yamlBox").focus();
  } else if (route.view === "run") {
    leaveHeavyViews("run");
    renderRunView();
  } else if (route.view === "compare") {
    stopProgressPoll();
    clearRunRetry();
    leaveHeavyViews("compare");
    showView("compare");
    updateSelection();
    openCompare(route.ids);
  } else if (route.view === "sweep") {
    stopProgressPoll();
    clearRunRetry();
    leaveHeavyViews("sweep");
    showView("sweep");
    updateSelection();
    openSweep(route.id);
  } else if (route.view === "validation") {
    stopProgressPoll();
    clearRunRetry();
    leaveHeavyViews("validation");
    showView("validation");
    updateSelection();
    openValidation();
  } else {
    stopProgressPoll();
    clearRunRetry();
    leaveHeavyViews("home");
    showView("home");
    updateSelection();
    refreshHomeSweeps();
  }
}

async function cancelActiveRun() {
  if (route.view !== "run") return;
  const id = route.id;
  const run = runsCache.find((r) => r.id === id);
  const label = run ? run.title || run.id : id;
  if (
    !window.confirm(
      'Cancel running run "' + label + '"? It stops at the next metrics flush and is marked failed.'
    )
  ) {
    return;
  }
  const { ok, doc } = await apiPost("/api/runs/" + encodeURIComponent(id) + "/cancel", {});
  if (!ok) {
    // 409 (already finished) or the server went away: the poll shows
    // the real state momentarily; surface the reason meanwhile.
    $("#progressSub").textContent =
      (doc && doc.error) || "cancel failed — is the server still running?";
    return;
  }
  $("#progressSub").textContent = "cancelling — waiting for the next metrics flush";
}

$("#cancelRunBtn").addEventListener("click", cancelActiveRun);

$("#selectToggle").addEventListener("click", () => toggleSelecting(!selecting));
$("#clearSelBtn").addEventListener("click", () => {
  selected.clear();
  selAnchorId = null;
  setSelNote("");
  syncSelectionUI();
});
$("#compareBtn").addEventListener("click", (ev) => {
  // aria-disabled is advisory; the click must not navigate to a compare
  // route that cannot render
  if ($("#compareBtn").getAttribute("aria-disabled") === "true") {
    ev.preventDefault();
    setSelNote("Select at least two runs to compare.");
  }
});

window.addEventListener("hashchange", onRoute);
syncSelectionUI();
refreshRuns();
setInterval(refreshRuns, RUNS_POLL_MS);
setInterval(refreshRunTimes, 60000); // keep "Xm ago" honest between polls
onRoute();
