/* fleetsim v0.5 app shell — vanilla ES module, no build step, no
   external requests.  Talks only to the same-origin /api routes
   (contract pinned in src/fleetsim/serve/server.py).

   Views (hash-routed):
     #                     home
     #new                  scenario editor
     #run/<id>/report      2D report (iframe) or live progress
     #run/<id>/fleet3d     3D placeholder (filled by a later phase)

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
function parseRoute() {
  const h = location.hash.replace(/^#/, "");
  if (h === "new") return { view: "new" };
  const m = /^run\/([^/]+)\/(report|fleet3d)$/.exec(h);
  if (m) {
    let id = m[1];
    try {
      id = decodeURIComponent(id);
    } catch (err) {
      /* malformed percent-escape (hand-edited or truncated URL): fall
         back to the raw match as a literal id — the server 404s it into
         the "No such run" panel instead of an uncaught URIError killing
         the navigation */
    }
    return { view: "run", id, mode: m[2] };
  }
  return { view: "home" };
}

let route = { view: "home" };

function showView(name) {
  for (const v of ["home", "run", "new"]) {
    $("#view-" + v).classList.toggle("hidden", v !== name);
  }
  if (name !== "run" && fleet3dMod) fleet3dMod.hideFleet3d();
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

async function openFleet3d(id) {
  try {
    await loadFleet3d();
  } catch (err) {
    const mount = $("#fleet3dMount");
    if (mount) mount.textContent = "3D module failed to load: " + String(err);
    return;
  }
  if (route.view === "run" && route.id === id && route.mode === "fleet3d") {
    fleet3dMod.mountFleet3d(id);
  }
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

function buildRunRow(run) {
  const li = document.createElement("li");
  li.className = "runrow";
  li.dataset.runId = run.id;

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
  const meta = document.createElement("span");
  meta.className = "rmeta";
  meta.textContent = rowMetaText(run);
  main.appendChild(title);
  main.appendChild(meta);
  a.appendChild(main);
  li.appendChild(a);

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
    if (row) refocus = { id: row.dataset.runId, dequeue: active.classList.contains("dequeue") };
  }

  list.textContent = "";
  for (const run of runsCache) list.appendChild(buildRunRow(run));

  $("#runsCount").textContent = runsCache.length ? String(runsCache.length) : "";
  $("#railEmpty").classList.toggle("hidden", runsCache.length > 0);
  updateSelection();

  if (refocus) {
    const row = list.querySelector('[data-run-id="' + CSS.escape(refocus.id) + '"]');
    if (row) {
      const el = refocus.dequeue ? row.querySelector(".dequeue") : row.querySelector("a");
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
  // one of: progress | failed | missing | unreachable | report | fleet3d | none
  $("#runProgress").classList.toggle("hidden", name !== "progress");
  $("#runFailed").classList.toggle("hidden", name !== "failed");
  $("#runMissing").classList.toggle("hidden", name !== "missing");
  $("#runUnreachable").classList.toggle("hidden", name !== "unreachable");
  $("#reportFrame").classList.toggle("hidden", name !== "report");
  $("#fleet3d").classList.toggle("hidden", name !== "fleet3d");
  if (name !== "fleet3d" && fleet3dMod) fleet3dMod.hideFleet3d();
}

function renderToolbar(id, info) {
  $("#runTitle").textContent = info ? info.title || id : id;
  $("#runMeta").textContent = info && info.created != null ? fmtWhen(info.created) : "";
  const dot = $("#runDot");
  dot.className = "dot" + (info ? " " + info.status : "");

  const base = "#run/" + encodeURIComponent(id) + "/";
  const tabReport = $("#tabReport");
  const tab3d = $("#tab3d");
  tabReport.href = base + "report";
  tab3d.href = base + "fleet3d";
  if (route.mode === "report") tabReport.setAttribute("aria-current", "true");
  else tabReport.removeAttribute("aria-current");
  if (route.mode === "fleet3d") tab3d.setAttribute("aria-current", "true");
  else tab3d.removeAttribute("aria-current");

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

function renderProgressSnapshot(status, prog) {
  $("#progressHead").textContent = status === "queued" ? "Queued" : "Running";
  $("#progressSub").textContent =
    status === "queued"
      ? "waiting for the worker — runs execute one at a time"
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
      runsJSON = "";
      refreshRuns();
      const info = await fetchRunInfo(id);
      if (route.view === "run" && route.id === id) {
        renderToolbar(id, info || { title: id, status: "done", created: null });
        if (route.mode === "fleet3d") {
          setRunBodyPanel("fleet3d");
          openFleet3d(id);
        } else if (route.mode === "report") {
          showReport(id);
        }
      }
      return;
    }
    if (st === "failed") {
      stopProgressPoll();
      runsJSON = "";
      refreshRuns();
      if (route.view === "run" && route.id === id) {
        $("#runDot").className = "dot failed";
        await showFailed(id);
      }
      return;
    }
    renderProgressSnapshot(st, doc.progress);
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
    setRunBodyPanel("fleet3d");
    openFleet3d(id);
    return;
  }
  // report mode — or a 3D tab on a run that has not finished yet, which
  // gets the same progress / failure panels as the report tab
  if (info.status === "done") {
    stopProgressPoll();
    showReport(id);
  } else if (info.status === "failed") {
    stopProgressPoll();
    $("#failError").textContent = info.error || "no error recorded";
    setRunBodyPanel("failed");
  } else {
    renderProgressSnapshot(info.status, null);
    setRunBodyPanel("progress");
    if (progressRunId !== id) startProgressPoll(id);
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

async function loadExamples() {
  const { ok, doc } = await apiGet("/api/examples");
  if (!ok || !Array.isArray(doc)) return;
  const select = $("#tplSelect");
  for (const ex of doc) {
    if (!ex || typeof ex.name !== "string" || typeof ex.yaml !== "string") continue;
    examplesByName.set(ex.name, ex.yaml);
    const opt = document.createElement("option");
    opt.value = ex.name;
    // A starter that cannot run as web-submitted (e.g. a relative trace
    // path) says so in the dropdown instead of surprising at Validate.
    opt.textContent =
      ex.runnable === false && typeof ex.note === "string"
        ? ex.name + " — " + ex.note
        : ex.name;
    select.appendChild(opt);
  }
}

function initEditor() {
  if (editorReady) return;
  editorReady = true;
  const box = $("#yamlBox");
  setEditorText(DEFAULT_YAML);

  box.addEventListener("input", () => {
    editorDirty = true;
    gutterSync();
  });
  box.addEventListener("scroll", () => {
    $("#gutter").scrollTop = box.scrollTop;
  });

  $("#tplSelect").addEventListener("change", (ev) => {
    const name = ev.target.value;
    if (!name) return;
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
  });

  $("#validateBtn").addEventListener("click", doValidate);
  $("#runBtn").addEventListener("click", doRun);
  $("#view-new").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      doValidate();
    }
  });

  loadExamples();
}

/* ------------------------------------------------------------------ *
 * boot
 * ------------------------------------------------------------------ */
function onRoute() {
  route = parseRoute();
  if (route.view === "new") {
    stopProgressPoll();
    clearRunRetry();
    initEditor();
    showView("new");
    updateSelection();
    $("#yamlBox").focus();
  } else if (route.view === "run") {
    renderRunView();
  } else {
    stopProgressPoll();
    clearRunRetry();
    showView("home");
    updateSelection();
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

window.addEventListener("hashchange", onRoute);
refreshRuns();
setInterval(refreshRuns, RUNS_POLL_MS);
setInterval(refreshRunTimes, 60000); // keep "Xm ago" honest between polls
onRoute();
