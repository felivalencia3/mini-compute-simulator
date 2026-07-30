/* fleetsim v0.5 — 3D fleet replay (the showcase view).

   Lazily imported by app.js when the "3D fleet" tab opens.  Renders the
   run's viz model (GET /api/runs/{id}/model) as a fleet of halls: one
   hall per cluster, pods in a rack-row grid, each pod a vertical stack
   of node-slabs.  All slabs live in ONE InstancedMesh (single draw call)
   and are recolored per frame from the same two-pointer stint sweep the
   2D fleet map uses, so 2D and 3D always agree about who runs where.

   Self-contained: the only import is the vendored three.js module build
   (same-origin, CSP script-src 'self').  No ambient camera drift — the
   camera moves only on user input; playback moves only the colors.
   All dynamic text goes through textContent (labels are data, never
   markup). */

import * as THREE from "./vendor/three.module.min.js";

/* Pinned vendor build: three 0.185.1 (REVISION "185"), MIT — see
   vendor/THREE_LICENSE.  tests/test_serve_static.py pins the files. */
export const THREE_REVISION = "185";

const US = 1e6;
const CANON = ["pretrain", "finetune", "eval", "best_effort", "inference"];
const IDLE_HEX = "#1a2030"; /* idle slabs: a faint blue-gray, per design */
const TEXT_HEX = "#e6e8eb";
const MUTED_HEX = "#8b93a1";
const SPEEDS = [1, 4, 16, 64]; /* sim-hours per wall-second, as in 2D */

/* Live media query (not a boot-time snapshot): toggling the OS
   Reduce Motion setting mid-session takes effect immediately because
   every use site reads .matches at call time. */
const REDUCED_MQ = typeof matchMedia === "function"
  ? matchMedia("(prefers-reduced-motion: reduce)") : null;
const reducedMotion = () => !!(REDUCED_MQ && REDUCED_MQ.matches);

/* ------------------------------------------------------------------ *
 * small utils (ported from the 2D report where noted)
 * ------------------------------------------------------------------ */
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const bisect = (a, x) => {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m; }
  return lo;
};

function fmtClock(us) {
  const s = Math.floor(us / US);
  const d = Math.floor(s / 86400), hh = Math.floor((s % 86400) / 3600);
  const mm = Math.floor((s % 3600) / 60), ss = s % 60;
  const p = (n) => String(n).padStart(2, "0");
  return (d ? d + "d " : "") + p(hh) + ":" + p(mm) + ":" + p(ss);
}
const fmtPct = (v) => (v == null || !isFinite(v) ? "–" : (v * 100).toFixed(1) + "%");

function eln(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

/* ------------------------------------------------------------------ *
 * module state: one live view, plus per-run persistence so switching
 * 2D <-> 3D (or away and back) keeps T, speed, camera and pin
 * ------------------------------------------------------------------ */
let view = null;                 /* the active _View, or null */
const saved = new Map();         /* runId -> {t, speed, pin, cam} */
let renderer = null;             /* WebGLRenderer singleton (one context) */
let keysWired = false;
let mountToken = 0;              /* stale-async guard across mounts */

export function hideFleet3d() {
  if (view) view.setVisible(false);
}

export async function mountFleet3d(runId) {
  const token = ++mountToken;
  const mount = document.querySelector("#fleet3dMount");
  if (!mount) return;
  if (view && view.runId === runId) { view.setVisible(true); return; }
  if (view) { view.dispose(); view = null; }

  mount.classList.remove("threedph");
  mount.classList.add("f3droot");
  const panel = mount.closest("#fleet3d");
  if (panel) panel.classList.add("threed-live");
  renderNotice(mount, "Loading 3D fleet replay…", ["fetching the run's replay model"]);

  let model = null, errText = null;
  try {
    const resp = await fetch("/api/runs/" + encodeURIComponent(runId) + "/model");
    if (resp.ok) model = await resp.json();
    else {
      const doc = await resp.json().catch(() => null);
      errText = (doc && doc.error) || ("model request failed (" + resp.status + ")");
    }
  } catch (err) {
    errText = "cannot reach the server — is fleetsim serve still running?";
  }
  if (token !== mountToken) return; /* another mount superseded this one */
  if (!model) {
    renderNotice(mount, "3D fleet replay unavailable", [errText || "no model"]);
    return;
  }
  if (!model.capabilities || !model.capabilities.map ||
      !model.fleet || !(model.fleet.clusters || []).length) {
    /* same degradation as the 2D fleet map */
    renderNotice(mount, "This run has no stint data", [
      "the 3D fleet replay (like the 2D fleet map) needs who-ran-where data:",
      "set  outputs: {stints: true}  in the scenario and re-run",
    ]);
    return;
  }
  if (THREE.REVISION !== THREE_REVISION) {
    console.warn("fleet3d: vendored three.js REVISION " + THREE.REVISION +
      " != pinned " + THREE_REVISION);
  }
  try {
    view = new _View(runId, model, mount);
  } catch (err) {
    console.error("fleet3d:", err);
    renderNotice(mount, "3D view failed to start", [String(err && err.message || err),
      "(WebGL may be unavailable in this browser)"]);
    return;
  }
  wireKeysOnce();
  view.setVisible(true);
}

function renderNotice(mount, head, lines) {
  mount.textContent = "";
  const box = eln("div", "f3dnotice");
  box.appendChild(eln("h2", null, head));
  for (const ln of lines) box.appendChild(eln("p", "sub mono", ln));
  mount.appendChild(box);
}

function wireKeysOnce() {
  if (keysWired) return;
  keysWired = true;
  document.addEventListener("keydown", (ev) => {
    if (!view || !view.visible) return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const tgt = ev.target;
    if (tgt && tgt.closest && tgt.closest("input,select,textarea,button,[contenteditable]")) return;
    switch (ev.key) {
      case " ": ev.preventDefault(); view.togglePlay(); break;
      case "ArrowLeft": ev.preventDefault(); view.setT(view.S.t - view.ROUND); break;
      case "ArrowRight": ev.preventDefault(); view.setT(view.S.t + view.ROUND); break;
      case "Home": ev.preventDefault(); view.setT(0); break;
      case "End": ev.preventDefault(); view.setT(view.HOR); break;
      case "1": view.goPose(0); break;
      case "2": view.goPose(1); break;
      case "3": view.goPose(2); break;
      case "Escape": view.setPin(null); break;
    }
  });
}

function getRenderer() {
  if (renderer) return renderer;
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x0b0e14, 1);
  return renderer;
}

/* ------------------------------------------------------------------ *
 * slab budget: pods read as stacks of 8-20 node-slabs; the chips-per-
 * slab heuristic keeps the WHOLE fleet within an instance budget
 * ------------------------------------------------------------------ */
const MAX_INSTANCES = 40000;
function slabCounts(units) {
  let perPodMax = 20;
  if (units.length * perPodMax > MAX_INSTANCES) {
    perPodMax = Math.max(1, Math.floor(MAX_INSTANCES / units.length));
  }
  return units.map((u) => {
    const chips = Math.max(1, u.chips | 0);
    let cps = 1; /* chips per slab: powers of two, like real node boards */
    while (chips / cps > perPodMax) cps *= 2;
    return Math.max(1, Math.ceil(chips / cps));
  });
}

/* ------------------------------------------------------------------ *
 * the view
 * ------------------------------------------------------------------ */
class _View {
  constructor(runId, model, mount) {
    this.runId = runId;
    this.M = model;
    this.mount = mount;
    this.visible = false;
    this.raf = null;
    this.disposed = false;

    this.HOR = Math.max(1, model.meta.horizon_us || 1);
    this.ROUND = model.meta.round_us || Math.max(1, Math.round(this.HOR / 1440));
    this.PULSE = clamp(Math.round(this.HOR / 100), 2 * this.ROUND, 6 * this.ROUND);

    this._buildIndexes();

    const was = saved.get(runId);
    this.S = {
      t: was ? clamp(was.t, 0, this.HOR) : 0,
      playing: false,
      speed: was ? was.speed : 4,
      pin: null,
      hover: null,
    };

    this._buildDom();
    this._buildScene();
    if (was && was.cam) this.orbit.restore(was.cam);
    else this.goPose(0, true);
    if (was && was.pin != null) {
      const u = this.unitIdxById.get(was.pin);
      if (u != null) this.setPin(u);
    }

    this.seek(this.S.t);
    this._recolor();
    this._syncTransport();
    this._updateHud();
  }

  /* ---------------- indexes: port of the 2D map's data model -------- */
  _buildIndexes() {
    const M = this.M;
    const observed = new Set();
    for (const k of Object.keys((M.frames || {}).pending_by_class || {})) observed.add(k);
    for (const c of M.stints.class_name) observed.add(c);
    for (const g of M.gantt || []) observed.add(g.class_name);
    this.LABELS = CANON.filter((c) => observed.has(c))
      .concat([...observed].filter((c) => !CANON.includes(c)).sort());
    if (!this.LABELS.length) this.LABELS = CANON.slice();
    this.NC = this.LABELS.length;
    const labIdx = new Map(this.LABELS.map((l, i) => [l, i]));

    /* units = domains (pods), flattened in model order */
    this.units = [];
    this.unitIdxById = new Map();
    this.clusters = [];
    (M.fleet.clusters || []).forEach((c, ci) => {
      const idxs = [];
      for (const d of c.domains) {
        const u = this.units.length;
        this.units.push({ id: d.id, short: d.short, chips: d.chips, cluster: ci });
        this.unitIdxById.set(d.id, u);
        idxs.push(u);
      }
      this.clusters.push({ id: c.id, chips: c.chips, unitIdxs: idxs });
    });
    this.NU = this.units.length;

    /* stints -> typed arrays; identical semantics to the 2D report,
       including running_at_horizon staying allocated at T = horizon */
    const st = M.stints, NSt = st.t0_us.length;
    this.NSt = NSt;
    this.sT0 = Float64Array.from(st.t0_us);
    this.sT1 = Float64Array.from(st.t1_us);
    this.sChips = Int32Array.from(st.chips);
    this.sUnit = Int32Array.from(st.domain_idx);
    this.sCls = Int16Array.from(st.class_name, (c) => labIdx.get(c) ?? 0);
    this.sReason = Uint8Array.from(st.end_reason, (r) => (
      r === "preempted" ? 1 : r === "failed" ? 2 : r === "drained" ? 3 : 0));
    this.sRel = Float64Array.from(st.end_reason, (r, i) => (
      r === "running_at_horizon" ? Infinity : this.sT1[i]));
    /* end_reason "failed" covers BOTH node-failure kills and routine
       job aborts (abort_prob) — pulsing red on every abort painted the
       fleet on fire even with node failures disabled (review fix).
       Scope the failure pulse to stints whose end sits within one
       scheduler round of an actual node-failure event: failure counts
       flush on round boundaries, so a kill at t lands in an event at
       the flush right after it. */
    const failTimes = (M.events || [])
      .filter((e) => e.kind === "failure")
      .map((e) => e.t_us)
      .sort((a, b) => a - b);
    this.sNodeKill = new Uint8Array(NSt);
    if (failTimes.length) {
      for (let i = 0; i < NSt; i++) {
        if (this.sReason[i] !== 2) continue;
        const t1 = this.sT1[i];
        const j = bisect(failTimes, t1 - this.ROUND);
        if (j < failTimes.length && failTimes[j] <= t1 + this.ROUND) {
          this.sNodeKill[i] = 1;
        }
      }
    }
    const order = Array.from({ length: NSt }, (_, i) => i);
    order.sort((a, b) => (this.sRel[a] < this.sRel[b] ? -1 : this.sRel[a] > this.sRel[b] ? 1 : a - b));
    this.byRel = Uint32Array.from(order);

    this.cur = {
      t: -1, a: 0, r: 0,
      occ: new Int32Array(this.NU * this.NC),
      active: new Set(),
      lastFail: new Float64Array(this.NU).fill(-Infinity),
      lastDrain: new Float64Array(this.NU).fill(-Infinity),
    };

    /* palette -> linear-space THREE colors (one per class + idle) */
    const pal = M.palette || {};
    this.clsColor = this.LABELS.map((l) => new THREE.Color().setStyle(pal[l] || MUTED_HEX));
    this.idleColor = new THREE.Color().setStyle(IDLE_HEX);
    this.failColor = new THREE.Color().setStyle(pal.failed || "#e03131");
    this.drainColor = new THREE.Color().setStyle(pal.draining || "#f76707");
    this.pinColor = new THREE.Color().setStyle(TEXT_HEX).multiplyScalar(0.5);
    this.hoverColor = new THREE.Color().setStyle(MUTED_HEX).multiplyScalar(0.35);
  }

  /* the two-pointer interval sweep, ported verbatim from the 2D map */
  seek(T) {
    const cur = this.cur;
    if (T < cur.t) {
      cur.a = 0; cur.r = 0; cur.occ.fill(0); cur.active.clear();
      cur.lastFail.fill(-Infinity); cur.lastDrain.fill(-Infinity);
    }
    while (cur.a < this.NSt && this.sT0[cur.a] <= T) {
      const i = cur.a++;
      cur.occ[this.sUnit[i] * this.NC + this.sCls[i]] += this.sChips[i];
      cur.active.add(i);
    }
    while (cur.r < this.NSt && this.sRel[this.byRel[cur.r]] <= T) {
      const i = this.byRel[cur.r++];
      cur.occ[this.sUnit[i] * this.NC + this.sCls[i]] -= this.sChips[i];
      cur.active.delete(i);
      const u = this.sUnit[i];
      /* only genuine node-failure kills pulse — not job aborts */
      if (this.sNodeKill[i]) cur.lastFail[u] = Math.max(cur.lastFail[u], this.sT1[i]);
      else if (this.sReason[i] === 3) cur.lastDrain[u] = Math.max(cur.lastDrain[u], this.sT1[i]);
    }
    cur.t = T;
  }

  nearestFrame(t) {
    const FT = (this.M.frames || {}).t_us || [];
    if (!FT.length) return -1;
    const i = bisect(FT, t);
    if (i <= 0) return 0;
    if (i >= FT.length) return FT.length - 1;
    return (t - FT[i - 1] <= FT[i] - t) ? i - 1 : i;
  }

  /* ---------------- DOM: canvas + HUD + transport + tooltip --------- */
  _buildDom() {
    const mount = this.mount;
    mount.textContent = "";

    this.canvasBox = eln("div", "f3dcanvas");
    mount.appendChild(this.canvasBox);

    /* HUD (top-left): T + occupancy + legend + hint */
    const hud = eln("div", "f3dhud");
    this.hudT = eln("div", "f3dT mono");
    this.hudOcc = eln("div", "f3docc mono");
    hud.appendChild(this.hudT);
    hud.appendChild(this.hudOcc);
    const legend = eln("div", "f3dlegend");
    for (let c = 0; c < this.NC; c++) {
      const k = eln("span", "f3dkey");
      const i = eln("i");
      i.style.background = (this.M.palette || {})[this.LABELS[c]] || MUTED_HEX;
      k.appendChild(i);
      k.appendChild(document.createTextNode(this.LABELS[c]));
      legend.appendChild(k);
    }
    hud.appendChild(legend);
    hud.appendChild(eln("div", "f3dhint sub",
      "drag orbit · wheel zoom · shift-drag pan · hover a pod · click pins · 1/2/3 camera"));
    mount.appendChild(hud);

    /* pinned-pod side card (hidden until a pin) */
    this.card = eln("div", "f3dcard hidden");
    mount.appendChild(this.card);

    /* transport bar (bottom): consistent with the 2D report's */
    const bar = eln("div", "f3dbar");
    const row = eln("div", "ctlrow");
    this.btnBack = eln("button", "btn", "◀◀ round");
    this.btnBack.type = "button";
    this.btnBack.title = "step back one round (left arrow)";
    this.btnPlay = eln("button", "btn primary", "▶ play");
    this.btnPlay.type = "button";
    this.btnPlay.title = "play / pause (Space)";
    this.btnPlay.setAttribute("aria-pressed", "false");
    this.btnFwd = eln("button", "btn", "round ▶▶");
    this.btnFwd.type = "button";
    this.btnFwd.title = "step forward one round (right arrow)";
    row.appendChild(this.btnBack); row.appendChild(this.btnPlay); row.appendChild(this.btnFwd);

    const spd = eln("label", "ctl", "speed ");
    this.speedSel = document.createElement("select");
    this.speedSel.setAttribute("aria-label", "playback speed, sim-hours per second");
    for (const v of SPEEDS) {
      const o = document.createElement("option");
      o.value = String(v); o.textContent = "×" + v;
      if (v === this.S.speed) o.selected = true;
      this.speedSel.appendChild(o);
    }
    spd.appendChild(this.speedSel);
    spd.appendChild(document.createTextNode(" sim-h/s"));
    row.appendChild(spd);

    const cams = eln("span", "f3dcams");
    this.camBtns = [];
    [["1", "overview"], ["2", "hall"], ["3", "floor"]].forEach(([k, name], i) => {
      const b = eln("button", "btn", k);
      b.type = "button";
      b.title = "camera: " + name + " (key " + k + ")";
      b.addEventListener("click", () => this.goPose(i));
      cams.appendChild(b);
      this.camBtns.push(b);
    });
    row.appendChild(cams);

    this.readout = eln("span", "f3dreadout mono");
    row.appendChild(this.readout);
    bar.appendChild(row);

    const scrubwrap = eln("div", "f3dscrubwrap");
    this.scrub = document.createElement("input");
    this.scrub.type = "range";
    this.scrub.id = "f3dScrub";
    this.scrub.min = "0";
    this.scrub.max = String(this.HOR);
    this.scrub.step = String(this.ROUND);
    this.scrub.setAttribute("aria-label", "simulation time scrubber");
    scrubwrap.appendChild(this.scrub);
    this.ticksEl = eln("div", "f3dticks");
    scrubwrap.appendChild(this.ticksEl);
    bar.appendChild(scrubwrap);
    mount.appendChild(bar);

    /* tooltip */
    this.tip = eln("div", "f3dtip hidden");
    this.tip.setAttribute("role", "status");
    mount.appendChild(this.tip);

    /* wiring */
    this.btnPlay.addEventListener("click", () => this.togglePlay());
    this.btnBack.addEventListener("click", () => this.setT(this.S.t - this.ROUND));
    this.btnFwd.addEventListener("click", () => this.setT(this.S.t + this.ROUND));
    this.speedSel.addEventListener("change", () => { this.S.speed = +this.speedSel.value; });
    this.scrub.addEventListener("input", () => this.setT(+this.scrub.value));
    this._buildTicks();
  }

  _buildTicks() {
    /* event ticks under the scrubber, grouped when < 6 px apart —
       same grouping rule as the 2D report */
    const W = Math.max(120, this.ticksEl.clientWidth || this.mount.clientWidth || 800);
    this.tickGroups = [];
    let g = null;
    for (const ev of this.M.events || []) {
      const px = (ev.t_us / this.HOR) * W;
      if (g && px - g.px0 < 6) { g.evs.push(ev); g.px1 = px; }
      else { g = { px0: px, px1: px, evs: [ev] }; this.tickGroups.push(g); }
    }
    this.ticksEl.textContent = "";
    this.tickGroups.forEach((grp, i) => {
      const first = grp.evs[0], last = grp.evs[grp.evs.length - 1];
      const oneKind = grp.evs.every((e) => e.kind === first.kind);
      const b = eln("button", "f3dtick k-" + (oneKind ? first.kind : "mixed"));
      b.type = "button";
      b.style.left = ((grp.px0 / W) * 100).toFixed(3) + "%";
      b.style.width = Math.max(4, Math.round(grp.px1 - grp.px0) + 4) + "px";
      const label = grp.evs.length === 1
        ? first.label + " at " + fmtClock(first.t_us)
        : grp.evs.length + " events from " + fmtClock(first.t_us) + " to " + fmtClock(last.t_us);
      b.setAttribute("aria-label", label);
      b.dataset.g = String(i);
      this.ticksEl.appendChild(b);
    });
    this.ticksEl.addEventListener("click", (e) => {
      const b = e.target.closest(".f3dtick"); if (!b) return;
      this.setT(this.tickGroups[+b.dataset.g].evs[0].t_us);
    });
    this.ticksEl.addEventListener("pointerover", (e) => {
      const b = e.target.closest(".f3dtick"); if (!b) return;
      const grp = this.tickGroups[+b.dataset.g];
      const first = grp.evs[0], last = grp.evs[grp.evs.length - 1];
      if (grp.evs.length === 1) {
        this.showTip(e.clientX, e.clientY, fmtClock(first.t_us), [{ v: first.label, n: first.kind }]);
      } else {
        const byKind = new Map();
        for (const ev of grp.evs) byKind.set(ev.kind, (byKind.get(ev.kind) || 0) + 1);
        const rows = [...byKind].map(([k, n]) => ({ v: n + "×", n: k }));
        this.showTip(e.clientX, e.clientY,
          fmtClock(first.t_us) + " – " + fmtClock(last.t_us) + " · " + grp.evs.length + " events", rows);
      }
    });
    this.ticksEl.addEventListener("pointerout", () => this.hideTip());
  }

  /* ---------------- scene ------------------------------------------ */
  _buildScene() {
    const r = getRenderer();
    this.canvasBox.appendChild(r.domElement);
    r.domElement.setAttribute("aria-label",
      "3D fleet replay: drag to orbit, wheel to zoom, shift-drag to pan");

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e14);
    this.camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 4000);
    this.disposables = [];

    /* layout: halls (clusters) on a ground plane; pods in ceil(sqrt)
       rack-row grids inside each hall */
    const PITCH = 1.55, PAD = 1.0, HALLGAP = 2.6, F = 1.0;
    const slabs = slabCounts(this.units);
    this.slabsOf = slabs;
    const halls = this.clusters.map((cl) => {
      const k = cl.unitIdxs.length;
      const cols = Math.max(1, Math.ceil(Math.sqrt(k)));
      const rows = Math.ceil(k / cols);
      return { cl, cols, rows, w: (cols - 1) * PITCH + F + 2 * PAD, d: (rows - 1) * PITCH + F + 2 * PAD };
    });
    const nH = halls.length;
    const hallCols = nH <= 4 ? nH : Math.ceil(Math.sqrt(nH));
    const cellW = Math.max(...halls.map((h) => h.w)) + HALLGAP;
    const cellD = Math.max(...halls.map((h) => h.d)) + HALLGAP;
    const hallRows = Math.ceil(nH / hallCols);
    const originX = -((hallCols - 1) * cellW) / 2;
    const originZ = -((hallRows - 1) * cellD) / 2;

    this.podPos = new Float32Array(this.NU * 2); /* x, z per pod */
    this.hallLabels = [];
    const floorGeo = [], edgeMat = new THREE.LineBasicMaterial({ color: 0x232a36 });
    this.disposables.push(edgeMat);
    let maxSlabs = 1;
    for (const s of slabs) maxSlabs = Math.max(maxSlabs, s);

    halls.forEach((h, hi) => {
      const cx = originX + (hi % hallCols) * cellW;
      const cz = originZ + Math.floor(hi / hallCols) * cellD;
      /* hall floor slab + border */
      const fg = new THREE.BoxGeometry(h.w, 0.07, h.d);
      const fm = new THREE.MeshLambertMaterial({ color: 0x11151d });
      const floor = new THREE.Mesh(fg, fm);
      floor.position.set(cx, 0.035, cz);
      this.scene.add(floor);
      const eg = new THREE.EdgesGeometry(fg);
      const edges = new THREE.LineSegments(eg, edgeMat);
      edges.position.copy(floor.position);
      this.scene.add(edges);
      this.disposables.push(fg, fm, eg);
      floorGeo.push(floor);
      /* hall label sprite: signage floating above the hall's front edge;
         faded out when the camera flies in close (it would fill the
         screen otherwise) */
      const spr = makeLabelSprite(h.cl.id + " · " + h.cl.chips + " chips", h.w * 1.15);
      const maxHallSlabs = Math.max(1, ...h.cl.unitIdxs.map((u) => slabs[u]));
      spr.position.set(cx, maxHallSlabs * 0.2 + 1.0, cz + h.d / 2);
      this.scene.add(spr);
      this.hallLabels.push({ spr, r0: Math.max(h.w, h.d) });
      this.disposables.push(spr.material.map, spr.material);
      /* pod positions */
      h.cl.unitIdxs.forEach((u, j) => {
        const col = j % h.cols, row = Math.floor(j / h.cols);
        this.podPos[u * 2] = cx - h.w / 2 + PAD + F / 2 + col * PITCH;
        this.podPos[u * 2 + 1] = cz - h.d / 2 + PAD + F / 2 + row * PITCH;
      });
    });

    const extentX = hallCols * cellW, extentZ = hallRows * cellD;
    this.extent = Math.max(extentX, extentZ, 6);

    /* ground plane + grid */
    const groundSize = this.extent * 3.2;
    const gg = new THREE.PlaneGeometry(groundSize, groundSize);
    const gm = new THREE.MeshLambertMaterial({ color: 0x0d1118 });
    const ground = new THREE.Mesh(gg, gm);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.02;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(groundSize, Math.round(groundSize / 2), 0x1c2330, 0x141a24);
    grid.position.y = 0.0;
    this.scene.add(grid);
    this.disposables.push(gg, gm, grid.geometry, grid.material);

    /* lights (no shadows: the fleet is the story, not the render) */
    const hemi = new THREE.HemisphereLight(0xdde3ee, 0x0b0e14, 1.15);
    const dir = new THREE.DirectionalLight(0xffffff, 2.0);
    dir.position.set(this.extent, this.extent * 1.4, this.extent * 0.6);
    this.scene.add(hemi, dir);

    /* THE fleet: one InstancedMesh of node-slabs (single draw call) */
    const SLAB_W = 0.92, SLAB_H = 0.15, SLAB_STEP = 0.2;
    this.slabStep = SLAB_STEP;
    let total = 0;
    this.instBase = new Int32Array(this.NU);
    for (let u = 0; u < this.NU; u++) { this.instBase[u] = total; total += slabs[u]; }
    this.totalSlabs = total;
    this.podOfInstance = new Int32Array(total);
    for (let u = 0; u < this.NU; u++) {
      for (let s = 0; s < slabs[u]; s++) this.podOfInstance[this.instBase[u] + s] = u;
    }

    const slabGeo = new THREE.BoxGeometry(SLAB_W, SLAB_H, SLAB_W);
    const slabMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.slabMesh = new THREE.InstancedMesh(slabGeo, slabMat, total);
    this.slabMesh.frustumCulled = false;
    const m4 = new THREE.Matrix4();
    for (let u = 0; u < this.NU; u++) {
      const x = this.podPos[u * 2], z = this.podPos[u * 2 + 1];
      for (let s = 0; s < slabs[u]; s++) {
        m4.makeTranslation(x, 0.07 + SLAB_H / 2 + s * SLAB_STEP, z);
        this.slabMesh.setMatrixAt(this.instBase[u] + s, m4);
      }
    }
    this.slabMesh.instanceMatrix.needsUpdate = true;
    for (let i = 0; i < total; i++) this.slabMesh.setColorAt(i, this.idleColor);
    this.slabMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.slabMesh);
    this.disposables.push(slabGeo, slabMat);

    /* pod glow shells: one slightly-scaled additive box per pod; black
       (= invisible under additive blending) unless failing / draining /
       pinned / hovered */
    const shellMat = new THREE.MeshBasicMaterial({
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
      opacity: 0.55, color: 0xffffff,
    });
    const shellGeo = new THREE.BoxGeometry(1, 1, 1);
    this.shellMesh = new THREE.InstancedMesh(shellGeo, shellMat, this.NU);
    this.shellMesh.frustumCulled = false;
    const BLACK = new THREE.Color(0x000000);
    for (let u = 0; u < this.NU; u++) {
      const h = slabs[u] * SLAB_STEP + 0.16;
      m4.makeScale(SLAB_W * 1.14, h, SLAB_W * 1.14);
      m4.setPosition(this.podPos[u * 2], 0.07 + h / 2 - 0.06, this.podPos[u * 2 + 1]);
      this.shellMesh.setMatrixAt(u, m4);
      this.shellMesh.setColorAt(u, BLACK);
    }
    this.shellMesh.instanceMatrix.needsUpdate = true;
    this.shellMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.shellMesh);
    this.disposables.push(shellGeo, shellMat);

    this.maxStackH = maxSlabs * SLAB_STEP + 0.3;

    /* camera poses (1 overview / 2 hall / 3 low floor view) */
    const c0 = halls.length ? {
      x: originX, z: originZ, w: halls[0].w, d: halls[0].d,
    } : { x: 0, z: 0, w: 6, d: 6 };
    const fitR = (this.extent / 2) / Math.tan((this.camera.fov * Math.PI / 180) / 2) * 1.25;
    const hallR = (Math.max(c0.w, c0.d) / 2) / Math.tan((this.camera.fov * Math.PI / 180) / 2) * 1.3;
    this.poses = [
      { tx: 0, ty: 0.4, tz: 0, radius: fitR, theta: -0.78, phi: 0.86 },
      { tx: c0.x, ty: 0.6, tz: c0.z, radius: hallR, theta: -0.52, phi: 1.08 },
      { tx: c0.x, ty: this.maxStackH * 0.4, tz: c0.z,
        radius: Math.max(7, Math.max(c0.w, c0.d) * 1.05), theta: 0.9, phi: 1.32 },
    ];

    this.orbit = new _Orbit(this.camera, r.domElement, {
      minR: 1.2, maxR: fitR * 3, bound: this.extent * 1.6, maxY: this.maxStackH + this.extent,
    });

    /* raycast hover / click-pin */
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this._tipLast = 0;
    r.domElement.addEventListener("pointermove", (e) => this._onHover(e));
    r.domElement.addEventListener("pointerleave", () => {
      this.S.hover = null; this.hideTip();
    });
    r.domElement.addEventListener("click", (e) => {
      if (this.orbit.dragged) return; /* an orbit drag is not a pick */
      const u = this._podAt(e);
      this.setPin(u == null ? null : (this.S.pin === u ? null : u));
    });

    /* size tracking */
    this.resizeObs = new ResizeObserver(() => this._resize());
    this.resizeObs.observe(this.canvasBox);
    this._resize();
  }

  _resize() {
    const w = Math.max(2, this.canvasBox.clientWidth);
    const h = Math.max(2, this.canvasBox.clientHeight);
    getRenderer().setSize(w, h, false);
    getRenderer().domElement.style.width = "100%";
    getRenderer().domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /* ---------------- recolor: pod class mix -> slab stack ------------- */
  _recolor() {
    const NC = this.NC, occ = this.cur.occ;
    const counts = new Int32Array(NC), give = new Int32Array(NC);
    const rem = new Float64Array(NC);
    for (let u = 0; u < this.NU; u++) {
      const S = this.slabsOf[u], base = this.instBase[u];
      const cap = Math.max(1, this.units[u].chips);
      let used = 0;
      for (let c = 0; c < NC; c++) { counts[c] = occ[u * NC + c]; if (counts[c] > 0) used += counts[c]; }
      let usedSlabs = 0;
      if (used > 0) {
        usedSlabs = Math.min(S, Math.max(1, Math.round((used / cap) * S)));
        /* proportional slabs per class, each present class >= 1 slab,
           largest remainder for the leftovers */
        let assigned = 0, present = 0;
        for (let c = 0; c < NC; c++) {
          if (counts[c] <= 0) { give[c] = 0; rem[c] = -1; continue; }
          present++;
          const raw = (counts[c] / used) * usedSlabs;
          give[c] = Math.max(1, Math.floor(raw));
          rem[c] = raw - Math.floor(raw);
          assigned += give[c];
        }
        while (assigned > usedSlabs && present > 0) {
          /* too many forced 1s: shrink the biggest holders */
          let bi = -1;
          for (let c = 0; c < NC; c++) if (give[c] > 1 && (bi < 0 || give[c] > give[bi])) bi = c;
          if (bi < 0) break;
          give[bi]--; assigned--;
        }
        while (assigned < usedSlabs) {
          let bi = -1;
          for (let c = 0; c < NC; c++) if (give[c] > 0 && (bi < 0 || rem[c] > rem[bi])) bi = c;
          if (bi < 0) break;
          give[bi]++; rem[bi] = -1; assigned++;
        }
      }
      /* write colors bottom-up in class order, idle above */
      let s = 0;
      if (used > 0) {
        for (let c = 0; c < NC && s < S; c++) {
          for (let k = 0; k < give[c] && s < S; k++) {
            this.slabMesh.setColorAt(base + s, this.clsColor[c]);
            s++;
          }
        }
      }
      for (; s < S; s++) this.slabMesh.setColorAt(base + s, this.idleColor);
    }
    this.slabMesh.instanceColor.needsUpdate = true;
  }

  /* shells: failure / drain pulse (subtle), pin, hover.  The steady
     (paused / reduced-motion) intensity is deliberately low: at 0.9 the
     additive shell tinted the whole pod so strongly that slab class
     colors stopped matching the legend on exactly the pods with recent
     failures. */
  _updateShells(now) {
    const t = this.S.t;
    const pulseA = (this.S.playing && !reducedMotion() && now != null)
      ? 0.5 + 0.45 * Math.sin(now / 170) : 0.35;
    const col = new THREE.Color();
    let any = false;
    for (let u = 0; u < this.NU; u++) {
      const pf = t - this.cur.lastFail[u] < this.PULSE;
      const pd = t - this.cur.lastDrain[u] < this.PULSE;
      if (pf) { col.copy(this.failColor).multiplyScalar(pulseA); any = true; }
      else if (pd) { col.copy(this.drainColor).multiplyScalar(pulseA); any = true; }
      else col.setRGB(0, 0, 0);
      /* the pin cue ADDS to any pulse instead of losing to it — a
         pinned pod stays findable even inside a failure window */
      if (this.S.pin === u) col.add(this.pinColor);
      else if (this.S.hover === u && !pf && !pd) col.copy(this.hoverColor);
      this.shellMesh.setColorAt(u, col);
    }
    this.shellMesh.instanceColor.needsUpdate = true;
    this.pulsing = any;
  }

  /* ---------------- hover / pin ------------------------------------- */
  _podAt(e) {
    const rect = getRenderer().domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.slabMesh, false);
    if (!hits.length || hits[0].instanceId == null) return null;
    return this.podOfInstance[hits[0].instanceId];
  }

  _onHover(e) {
    if (this.orbit.buttons) { this.hideTip(); return; } /* mid-drag */
    const now = performance.now();
    if (now - this._tipLast < 28) return;
    this._tipLast = now;
    const u = this._podAt(e);
    if (u == null) {
      if (this.S.hover != null) { this.S.hover = null; this._updateShells(); }
      this.hideTip();
      return;
    }
    if (this.S.hover !== u) { this.S.hover = u; this._updateShells(); }
    this.showTip(e.clientX, e.clientY, this.units[u].id + " @ " + fmtClock(this.S.t),
      this._podRows(u, 4));
  }

  /* tooltip / card rows for one pod: per-class chips + top jobs, same
     content as the 2D map's tooltip */
  _podRows(u, maxJobs) {
    const rows = [];
    let used = 0;
    for (let c = 0; c < this.NC; c++) {
      const n = this.cur.occ[u * this.NC + c];
      if (n > 0) {
        rows.push({ c: (this.M.palette || {})[this.LABELS[c]] || MUTED_HEX, v: String(n), n: this.LABELS[c] + " chips" });
        used += n;
      }
    }
    rows.push({ v: fmtPct(used / Math.max(1, this.units[u].chips)), n: "of " + this.units[u].chips + " chips busy" });
    if (this.cur.active.size && this.cur.active.size <= 20000) {
      const per = new Map();
      for (const i of this.cur.active) {
        if (this.sUnit[i] !== u) continue;
        const id = this.M.stints.job_id[i];
        per.set(id, (per.get(id) || 0) + this.sChips[i]);
      }
      const top = [...per.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, maxJobs);
      for (const [id, n] of top) rows.push({ v: String(n), n: id });
      if (per.size > maxJobs) rows.push({ v: "+" + (per.size - maxJobs), n: "more jobs" });
    }
    return rows;
  }

  setPin(u) {
    this.S.pin = u;
    this._updateShells();
    this._renderCard();
  }

  _renderCard() {
    const u = this.S.pin;
    if (u == null) { this.card.classList.add("hidden"); this.card.textContent = ""; return; }
    this.card.classList.remove("hidden");
    this.card.textContent = "";
    const head = eln("div", "f3dcardhead");
    head.appendChild(eln("span", "mono", this.units[u].id));
    const x = eln("button", "f3dclose", "✕");
    x.type = "button";
    x.title = "unpin (Esc)";
    x.addEventListener("click", () => this.setPin(null));
    head.appendChild(x);
    this.card.appendChild(head);
    this.card.appendChild(eln("div", "sub", "residents @ " + fmtClock(this.S.t)));
    const list = eln("div", "f3dcardrows");
    for (const r of this._podRows(u, 8)) {
      const d = eln("div", "trw");
      if (r.c) { const i = eln("i"); i.style.background = r.c; d.appendChild(i); }
      d.appendChild(eln("span", "tv mono", r.v));
      d.appendChild(eln("span", "tn", r.n));
      list.appendChild(d);
    }
    this.card.appendChild(list);
  }

  showTip(x, y, head, rows) {
    const tip = this.tip;
    tip.textContent = "";
    if (head != null) tip.appendChild(eln("div", "th mono", head));
    for (const r of rows || []) {
      const d = eln("div", "trw");
      if (r.c) { const i = eln("i"); i.style.background = r.c; d.appendChild(i); }
      d.appendChild(eln("span", "tv mono", r.v));
      d.appendChild(eln("span", "tn", r.n));
      tip.appendChild(d);
    }
    tip.classList.remove("hidden");
    const rect = this.mount.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let px = x - rect.left + 14, py = y - rect.top + 14;
    if (px + tw > rect.width - 8) px = x - rect.left - tw - 10;
    if (py + th > rect.height - 8) py = y - rect.top - th - 10;
    tip.style.left = Math.max(4, px) + "px";
    tip.style.top = Math.max(4, py) + "px";
  }
  hideTip() { this.tip.classList.add("hidden"); }

  /* ---------------- transport / HUD ---------------------------------- */
  togglePlay() { this.S.playing ? this.pause() : this.play(); }
  play() {
    if (this.S.playing) return;
    if (this.S.t >= this.HOR) this.S.t = 0;
    this.S.playing = true;
    this.btnPlay.textContent = "⏸ pause";
    this.btnPlay.setAttribute("aria-pressed", "true");
    this._lastNow = performance.now();
  }
  pause() {
    this.S.playing = false;
    this.btnPlay.textContent = "▶ play";
    this.btnPlay.setAttribute("aria-pressed", "false");
  }
  setT(t) {
    this.S.t = clamp(t, 0, this.HOR);
    this.seek(this.S.t);
    this._recolor();
    this._updateShells();
    this._syncTransport();
    this._updateHud();
    if (this.S.pin != null) this._renderCard();
  }
  _syncTransport() { this.scrub.value = String(this.S.t); }
  _updateHud() {
    this.hudT.textContent = "T " + fmtClock(this.S.t);
    const i = this.nearestFrame(this.S.t);
    const occ = i >= 0 ? (this.M.frames.occupancy || [])[i] : null;
    /* occupancy can legitimately top 100% while grace-period jobs
       finish on draining/failed nodes — annotate instead of looking
       like corrupted data */
    this.hudOcc.textContent = "occupancy " + fmtPct(occ) +
      (occ != null && occ > 1 ? " (incl. draining-node grace)" : "");
    this.readout.textContent = fmtClock(this.S.t) + " / " + fmtClock(this.HOR);
  }

  goPose(i, snap) {
    const p = this.poses[i];
    if (!p) return;
    this.orbit.flyTo(p, snap || reducedMotion());
  }

  /* ---------------- lifecycle --------------------------------------- */
  setVisible(vis) {
    if (this.disposed) return;
    if (vis === this.visible) { if (vis && this.raf == null) this._startLoop(); return; }
    this.visible = vis;
    if (vis) {
      /* the renderer canvas is shared; make sure it is OURS and sized */
      if (getRenderer().domElement.parentNode !== this.canvasBox) {
        this.canvasBox.appendChild(getRenderer().domElement);
      }
      this._resize();
      this._startLoop();
    } else {
      this.pause();
      this._save();
      if (this.raf != null) { cancelAnimationFrame(this.raf); this.raf = null; }
    }
  }

  _save() {
    saved.set(this.runId, {
      t: this.S.t,
      speed: this.S.speed,
      pin: this.S.pin != null ? this.units[this.S.pin].id : null,
      cam: this.orbit.save(),
    });
  }

  _startLoop() {
    if (this.raf != null) return;
    const step = (now) => {
      this.raf = null;
      if (!this.visible || this.disposed) return;
      if (this.S.playing) {
        const dt = (now - this._lastNow) / 1000;
        this._lastNow = now;
        this.S.t = clamp(this.S.t + dt * this.S.speed * 3600 * US, 0, this.HOR);
        this.seek(this.S.t);
        this._recolor();
        this._syncTransport();
        this._updateHud();
        if (this.S.pin != null && now - (this._cardAt || 0) > 250) {
          this._cardAt = now; this._renderCard();
        }
        if (this.S.t >= this.HOR) this.pause();
      }
      this.orbit.update(now);
      this._updateShells(now);
      for (const { spr, r0 } of this.hallLabels) {
        const d = this.camera.position.distanceTo(spr.position);
        spr.material.opacity = clamp((d - r0 * 0.6) / (r0 * 0.5), 0, 1) * 0.95;
      }
      getRenderer().render(this.scene, this.camera);
      this.raf = requestAnimationFrame(step);
    };
    this._lastNow = performance.now();
    this.raf = requestAnimationFrame(step);
  }

  dispose() {
    this._save();
    this.disposed = true;
    this.visible = false;
    if (this.raf != null) { cancelAnimationFrame(this.raf); this.raf = null; }
    this.resizeObs.disconnect();
    this.orbit.dispose();
    this.slabMesh.dispose();
    this.shellMesh.dispose();
    for (const d of this.disposables) { if (d && d.dispose) d.dispose(); }
    this.mount.textContent = "";
  }
}

/* ------------------------------------------------------------------ *
 * tiny orbit controller (drag orbit, wheel dolly, right/shift-drag
 * pan) — implemented inline so the vendor surface stays the two
 * three.js build files
 * ------------------------------------------------------------------ */
class _Orbit {
  constructor(camera, dom, opts) {
    this.camera = camera;
    this.dom = dom;
    this.opts = opts;
    this.target = new THREE.Vector3(0, 0.4, 0);
    this.radius = 20; this.theta = -0.78; this.phi = 0.9;
    this.buttons = 0;
    this.dragged = false;
    this.anim = null;

    this._down = (e) => {
      if (e.button !== 0 && e.button !== 2) return;
      this.buttons = e.buttons;
      this.dragged = false;
      this._px = e.clientX; this._py = e.clientY;
      this._pan = e.button === 2 || e.shiftKey;
      dom.setPointerCapture(e.pointerId);
      this.anim = null; /* user input cancels any pose tween */
    };
    this._move = (e) => {
      if (!this.buttons) return;
      const dx = e.clientX - this._px, dy = e.clientY - this._py;
      this._px = e.clientX; this._py = e.clientY;
      if (Math.abs(dx) + Math.abs(dy) > 1) this.dragged = true;
      if (this._pan) {
        const per = this.radius * 0.0016;
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
        this.target.addScaledVector(right, -dx * per).addScaledVector(up, dy * per);
        const b = this.opts.bound;
        this.target.x = clamp(this.target.x, -b, b);
        this.target.z = clamp(this.target.z, -b, b);
        this.target.y = clamp(this.target.y, 0, this.opts.maxY);
      } else {
        this.theta -= dx * 0.0052;
        this.phi = clamp(this.phi - dy * 0.0052, 0.12, 1.5);
      }
      this.apply();
    };
    this._up = (e) => { this.buttons = 0; };
    this._wheel = (e) => {
      e.preventDefault();
      this.anim = null;
      this.radius = clamp(this.radius * Math.exp(e.deltaY * 0.0011), this.opts.minR, this.opts.maxR);
      this.apply();
    };
    this._ctx = (e) => e.preventDefault();
    dom.addEventListener("pointerdown", this._down);
    dom.addEventListener("pointermove", this._move);
    dom.addEventListener("pointerup", this._up);
    dom.addEventListener("pointercancel", this._up);
    dom.addEventListener("wheel", this._wheel, { passive: false });
    dom.addEventListener("contextmenu", this._ctx);
    this.apply();
  }

  apply() {
    const t = this.target, r = this.radius, sp = Math.sin(this.phi);
    this.camera.position.set(
      t.x + r * sp * Math.sin(this.theta),
      t.y + r * Math.cos(this.phi),
      t.z + r * sp * Math.cos(this.theta),
    );
    this.camera.lookAt(t);
  }

  flyTo(pose, snap) {
    const to = {
      tx: pose.tx, ty: pose.ty, tz: pose.tz,
      radius: pose.radius, theta: pose.theta, phi: pose.phi,
    };
    if (snap) {
      this.target.set(to.tx, to.ty, to.tz);
      this.radius = to.radius; this.theta = to.theta; this.phi = to.phi;
      this.anim = null;
      this.apply();
      return;
    }
    this.anim = {
      from: this.save(), to, t0: performance.now(), dur: 420,
    };
  }

  update(now) {
    const a = this.anim;
    if (!a) return;
    const k = clamp((now - a.t0) / a.dur, 0, 1);
    const e = 1 - Math.pow(1 - k, 3); /* ease-out cubic */
    const lerp = (x, y) => x + (y - x) * e;
    this.target.set(lerp(a.from.tx, a.to.tx), lerp(a.from.ty, a.to.ty), lerp(a.from.tz, a.to.tz));
    this.radius = lerp(a.from.radius, a.to.radius);
    this.theta = lerp(a.from.theta, a.to.theta);
    this.phi = lerp(a.from.phi, a.to.phi);
    this.apply();
    if (k >= 1) this.anim = null;
  }

  save() {
    return {
      tx: this.target.x, ty: this.target.y, tz: this.target.z,
      radius: this.radius, theta: this.theta, phi: this.phi,
    };
  }
  restore(s) {
    this.target.set(s.tx, s.ty, s.tz);
    this.radius = s.radius; this.theta = s.theta; this.phi = s.phi;
    this.apply();
  }

  dispose() {
    const d = this.dom;
    d.removeEventListener("pointerdown", this._down);
    d.removeEventListener("pointermove", this._move);
    d.removeEventListener("pointerup", this._up);
    d.removeEventListener("pointercancel", this._up);
    d.removeEventListener("wheel", this._wheel);
    d.removeEventListener("contextmenu", this._ctx);
  }
}

/* hall label: a small canvas-texture sprite (local drawing only);
   maxW caps the world-space width so a label never dwarfs a small hall */
function makeLabelSprite(text, maxW) {
  const fs = 44, pad = 14;
  const cv = document.createElement("canvas");
  let g = cv.getContext("2d");
  g.font = "600 " + fs + "px ui-monospace, Menlo, monospace";
  const w = Math.ceil(g.measureText(text).width) + pad * 2;
  cv.width = Math.max(2, w);
  cv.height = fs + pad * 2;
  g = cv.getContext("2d");
  g.font = "600 " + fs + "px ui-monospace, Menlo, monospace";
  g.textBaseline = "middle";
  g.fillStyle = "rgba(139,147,161,.95)";
  g.fillText(text, pad, cv.height / 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  /* depthTest off: labels are annotations and must not be occluded by
     the stacks behind them */
  const mat = new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false, depthTest: false,
  });
  const spr = new THREE.Sprite(mat);
  const k = Math.min(0.010, (maxW || Infinity) / cv.width);
  spr.scale.set(cv.width * k, cv.height * k, 1);
  return spr;
}
