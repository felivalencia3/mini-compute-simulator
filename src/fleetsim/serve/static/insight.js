/* fleetsim v0.8 — the analysis tab (#run/<id>/insight).

   Lazily imported by app.js.  Owns everything under #insightMount and
   turns the run's viz model (GET /api/runs/{id}/model) from something
   you eyeball into something you can attribute:

     1. EVENT DRILL-DOWN — pick a preemption wave or a node-failure tick
        from `events[]` and see the jobs that actually stopped (stints
        whose t1_us lands in the window with end_reason preempted /
        failed / drained), which higher-tier gang most likely took their
        place (a stint that STARTED in the same window on the SAME domain
        at a higher tier — labelled INFERRED, because the model records
        no causal link), and chips freed vs chips claimed, so "did the
        wave pay for itself?" has a number.  The window is the event's
        own round plus a VISIBLE settlement lookahead: a preempted job
        keeps its chips until its grace expires, so its stint settles a
        round or more after the counter moved.
     2. CORRELATION — a scatter over the model's frames with a
        least-squares fit, Pearson r and n printed.  One run, one
        scenario: correlation, never causation, and the panel says so.
     3. OCCUPANCY-DIP ATTRIBUTION — rounds whose occupancy sits more than
        k robust sigma below its local median, each decomposed into chips
        freed by node failures / drains / preemptions / normal endings,
        chips re-claimed inside the same window, and an UNEXPLAINED
        RESIDUAL that is always drawn, never hidden.

   EVERY NUMBER ON THIS SCREEN COMES OUT OF THE RUN'S OWN RECORDED
   OUTPUT.  Nothing is modelled or filled in: where the model cannot
   answer (no stints recorded, no round length, a victim with no
   higher-tier starter on its domain) the panel says so in words instead
   of guessing.  The one inference in the whole tab — "who displaced
   this job" — is marked as an inference everywhere it appears.

   The kernels below (buildStintIndex / drillDown / pearsonFit /
   detectDips / attributeDip) are PURE: no DOM, no fetch, no module
   state.  tests/test_viz_insight.py imports this file into node and
   runs them against a synthetic model with a planted wave and a planted
   dip, so the arithmetic on this screen is under test, not just the
   markup.

   All dynamic text goes through textContent; SVG is built with
   createElementNS.  No markup interpolation anywhere. */

"use strict";

const US = 1e6;
const SVG_NS = "http://www.w3.org/2000/svg";

const MUTED = "#8b93a1";
const TEXT = "#e6e8eb";
const GRID = "rgba(255,255,255,.07)";
const ACCENT = "#4dabf7";
const PANEL = "#11151d";

/* Disruption palette for the dip decomposition.  Three slots are pinned
   by the app's design system (--failed, --maintenance, --done); the
   fourth (preemptions) is the one free choice and was stepped to an
   amber that clears CVD separation against the red.  Validated with the
   dataviz palette validator against this app's panel surface #11151d:
   all-pairs CVD ΔE 9.7 worst (protan), normal-vision ΔE 23.1 worst,
   chroma floor and 3:1 contrast PASS.  The validator's lightness-band
   check FAILS by design — three of the four hexes are fixed by the
   design system, and separability beats cohesion here.  Identity is
   additionally carried by the legend text, the direct value labels and
   the table, never by hue alone; the residual is hatched as well as
   colored so "not attributed" reads differently from a cause. */
const DIP_PARTS = [
  { key: "failed", label: "node failures", color: "#e03131" },
  { key: "preempted", label: "preemptions", color: "#f59f00" },
  { key: "drained", label: "drains", color: "#845ef7" },
  { key: "other", label: "normal endings", color: "#12b886" },
];
const RESIDUAL_COLOR = MUTED;

/** stint end_reason -> the DIP_PARTS bucket it is attributed to. */
const END_REASON_PART = {
  failed: "failed",
  preempted: "preempted",
  drained: "drained",
  completed: "other",
  canceled: "other",
  timeout: "other",
};

/** Not endings at all: the run hit its horizon (or a live flush) while
    the stint was still holding its chips.  Attributing these as "freed"
    would fabricate a release in the last window, so they are excluded
    from the decomposition and counted separately. */
const NON_ENDINGS = new Set(["running_at_horizon", "open"]);

/** Stints that ended because something took the chips away. */
export const DISRUPTIVE_END_REASONS = new Set(["preempted", "failed", "drained"]);

/** Tier bands, low to high (fleetsim.model.Tier, lowercased). */
export const TIER_RANK = { best_effort: 0, batch: 1, prod: 2, monitoring: 3 };

/** Event kinds the drill-down can explain (frontier_* are markers, not
    disruptions — they name a job, they do not stop one). */
const DRILL_KINDS = new Set(["preemption_wave", "failure"]);

/** Rows/rounds beyond these are summarized rather than listed: the point
    of the panel is attribution, not an unbounded dump. */
const MAX_EVENT_ROWS = 400;
const MAX_VICTIM_ROWS = 200;
const MAX_DIP_BARS = 12;
const MAX_DIP_ROWS = 60;

/** Centered window (in frames) the dip detector takes its local median
    over.  Odd so the median has a true center; wide enough that a dip a
    few rounds long does not drag its own baseline down. */
const DIP_MEDIAN_WINDOW = 21;

/* ------------------------------------------------------------------ *
 * pure kernels (no DOM, no state — tested directly under node)
 * ------------------------------------------------------------------ */

const isNum = (v) => typeof v === "number" && isFinite(v);

/** First index i with vals[i] > x, over an ASCENDING array. */
function upperBound(vals, x) {
  let lo = 0;
  let hi = vals.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (vals[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Search structures over model.stints.
 *
 * The schema pins `t0_us` ascending, but this index sorts BOTH endpoints
 * itself: a silently mis-ordered array would make the binary search
 * return a plausible-looking subset of the round rather than fail, and a
 * wrong attribution is worse than a slow one.  Both queries are half-open
 * on the left — (lo, hi] — which is exactly the interval a metrics flush
 * at `hi` covers.
 */
export function buildStintIndex(model) {
  const st = (model && model.stints) || {};
  const t0 = st.t0_us || [];
  const t1 = st.t1_us || [];
  const n = t0.length;

  const sortedBy = (times) => {
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => times[a] - times[b] || a - b);
    const keys = new Array(n);
    for (let i = 0; i < n; i++) keys[i] = times[order[i]];
    return { order, keys };
  };
  const starts = sortedBy(t0);
  const ends = sortedBy(t1);

  const between = (sorted, lo, hi) => {
    const out = [];
    for (let i = upperBound(sorted.keys, lo); i < n && sorted.keys[i] <= hi; i++) {
      out.push(sorted.order[i]);
    }
    out.sort((a, b) => a - b); // stable, model order — deterministic output
    return out;
  };

  const domains = [];
  for (const cluster of (model && model.fleet && model.fleet.clusters) || []) {
    for (const dom of cluster.domains || []) domains.push(dom.id);
  }
  return {
    n,
    st,
    domains,
    /** Indices of stints that STARTED in (lo, hi]. */
    startsIn(lo, hi) {
      return between(starts, lo, hi);
    },
    /** Indices of stints that ENDED in (lo, hi]. */
    endsIn(lo, hi) {
      return between(ends, lo, hi);
    },
    domainName(idx) {
      return this.domains[idx] != null ? this.domains[idx] : "domain " + idx;
    },
  };
}

/**
 * The window a drill-down covers: the round the flush at `tEnd` samples,
 * (tEnd - round_us, tEnd], optionally extended by `lookahead` further
 * rounds.  null when the model could not determine a round length.
 *
 * WHY THE LOOKAHEAD EXISTS.  A preemption is COUNTED when the scheduler
 * decides it, but the job keeps its chips through its grace window
 * (`checkpoint_save_s`) and its stint only settles at the requeue — one
 * or more rounds later.  Node failures kill instantly and settle inside
 * their own round; preemption waves almost never do.  Rather than
 * silently widening the window (or silently showing an empty panel), the
 * app makes the extension a visible control and prints the interval it
 * used.
 */
export function roundWindowFor(model, tEnd, lookahead) {
  const r = model && model.meta ? model.meta.round_us : null;
  if (!isNum(r) || r <= 0 || !isNum(tEnd)) return null;
  const extra = isNum(lookahead) && lookahead > 0 ? Math.floor(lookahead) : 0;
  return [tEnd - r, tEnd + extra * r];
}

const tierRank = (tier) => {
  const r = TIER_RANK[String(tier)];
  return r === undefined ? null : r;
};

/**
 * Everything the drill-down panel knows about one disruption event.
 *
 * Victims are stints that ENDED in the window for a disruptive reason,
 * grouped back into jobs (one job can hold a stint per domain), each
 * tagged `in_round` when it settled inside the event's OWN round rather
 * than during the settlement lookahead.  A displacer CANDIDATE is a
 * stint of ANOTHER job that started in the same window, on a domain the
 * victim was holding, at a strictly higher tier — the only evidence the
 * recorded model carries.  It is an inference and is labelled as one;
 * `displacers` is empty when nothing qualifies.
 */
export function drillDown(model, index, tEnd, lookahead) {
  const win = roundWindowFor(model, tEnd, lookahead);
  if (!win) return { ok: false, reason: "no-round", t_us: tEnd };
  if (!index.n) return { ok: false, reason: "no-stints", t_us: tEnd, window: win };
  const [lo, hi] = win;
  const roundEnd = tEnd; // the event's OWN round ends here
  const st = index.st;
  const ended = index.endsIn(lo, hi);
  const started = index.startsIn(lo, hi);

  /* domain -> starters on it, so a victim's candidates are one lookup */
  const startsByDomain = new Map();
  let chipsClaimed = 0;
  for (const i of started) {
    chipsClaimed += st.chips[i];
    const d = st.domain_idx[i];
    if (!startsByDomain.has(d)) startsByDomain.set(d, []);
    startsByDomain.get(d).push(i);
  }

  const jobs = new Map();
  let chipsFreed = 0;
  const displacerStints = new Set();
  for (const i of ended) {
    const reason = String(st.end_reason[i]);
    if (!DISRUPTIVE_END_REASONS.has(reason)) continue;
    chipsFreed += st.chips[i];
    const jid = st.job_id[i];
    let job = jobs.get(jid);
    if (!job) {
      job = {
        job_id: jid,
        class_name: st.class_name[i],
        tier: st.tier[i],
        chips: 0,
        domains: [],
        end_reasons: [],
        displacers: [],
        rank_known: tierRank(st.tier[i]) !== null,
        in_round: false,
      };
      jobs.set(jid, job);
    }
    if (st.t1_us[i] <= roundEnd) job.in_round = true;
    job.chips += st.chips[i];
    const dname = index.domainName(st.domain_idx[i]);
    if (!job.domains.includes(dname)) job.domains.push(dname);
    if (!job.end_reasons.includes(reason)) job.end_reasons.push(reason);

    const mine = tierRank(st.tier[i]);
    for (const j of startsByDomain.get(st.domain_idx[i]) || []) {
      // a job cannot displace itself: a preempted job that is re-placed
      // inside the same window is a restart, not a beneficiary
      if (st.job_id[j] === jid) continue;
      const theirs = tierRank(st.tier[j]);
      if (mine === null || theirs === null || theirs <= mine) continue;
      displacerStints.add(j);
      const existing = job.displacers.find((d) => d.job_id === st.job_id[j]);
      if (existing) {
        existing.chips += st.chips[j];
        if (!existing.domains.includes(dname)) existing.domains.push(dname);
      } else {
        job.displacers.push({
          job_id: st.job_id[j],
          class_name: st.class_name[j],
          tier: st.tier[j],
          chips: st.chips[j],
          domains: [dname],
        });
      }
    }
  }

  let chipsToDisplacers = 0;
  for (const j of displacerStints) chipsToDisplacers += st.chips[j];

  const list = [...jobs.values()];
  for (const job of list) {
    // deterministic: biggest claim first, then job id
    job.displacers.sort((a, b) => b.chips - a.chips || (a.job_id < b.job_id ? -1 : 1));
    job.end_reasons.sort();
  }
  list.sort((a, b) => b.chips - a.chips || (a.job_id < b.job_id ? -1 : 1));

  return {
    ok: true,
    t_us: tEnd,
    window: win,
    round_end_us: roundEnd,
    round_us: model.meta.round_us,
    jobs: list,
    n_jobs: list.length,
    n_in_round: list.filter((j) => j.in_round).length,
    n_ended: ended.length,
    n_started: started.length,
    chips_freed: chipsFreed,
    chips_claimed: chipsClaimed,
    chips_to_displacers: chipsToDisplacers,
    n_with_displacer: list.filter((j) => j.displacers.length).length,
  };
}

/**
 * Pearson r plus the least-squares line, over the pairs where BOTH
 * values are finite.  null when fewer than three pairs survive or one
 * axis has no variance (r is undefined there — reporting 0 would be a
 * lie, reporting 1 a bigger one).
 */
export function pearsonFit(xs, ys) {
  const px = [];
  const py = [];
  const n0 = Math.min(xs.length, ys.length);
  for (let i = 0; i < n0; i++) {
    if (isNum(xs[i]) && isNum(ys[i])) {
      px.push(xs[i]);
      py.push(ys[i]);
    }
  }
  const n = px.length;
  const base = { n, r: null, slope: null, intercept: null, xs: px, ys: py };
  if (n < 3) return { ...base, reason: "fewer than 3 rounds have both values" };
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += px[i];
    sy += py[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = px[i] - mx;
    const dy = py[i] - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  if (sxx <= 0 || syy <= 0) {
    return { ...base, reason: "one axis never varies" };
  }
  const r = sxy / Math.sqrt(sxx * syy);
  const slope = sxy / sxx;
  return { ...base, r, slope, intercept: my - slope * mx, reason: null };
}

function median(sorted) {
  const n = sorted.length;
  if (!n) return null;
  const mid = n >> 1;
  return n % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Rounds whose occupancy sits more than `k` robust sigma below its LOCAL
 * median (a centered DIP_MEDIAN_WINDOW of frames), merged into
 * contiguous dips.
 *
 * Local median, not a global mean: a run that ramps up would otherwise
 * flag its whole warm-up.  Sigma is the MAD-based robust scale of the
 * residual (1.4826 * median|resid|), so the dips themselves do not
 * inflate the threshold that is meant to catch them; it falls back to
 * the plain standard deviation when the MAD is exactly zero (a flat
 * series with a handful of excursions).
 */
export function detectDips(frames, k, windowFrames) {
  const t = (frames && frames.t_us) || [];
  const occ = (frames && frames.occupancy) || [];
  const alloc = (frames && frames.allocated_chips) || [];
  const healthy = (frames && frames.healthy_chips) || [];
  const n = t.length;
  const W = windowFrames || DIP_MEDIAN_WINDOW;
  const half = Math.max(1, (W - 1) >> 1);

  const valid = [];
  for (let i = 0; i < n; i++) if (isNum(occ[i])) valid.push(i);
  if (valid.length < 5) {
    return { dips: [], sigma: null, window: W, reason: "fewer than 5 rounds with an occupancy value" };
  }

  const resid = new Array(n).fill(null);
  const local = new Array(n).fill(null);
  for (const i of valid) {
    const seg = [];
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) {
      if (isNum(occ[j])) seg.push(occ[j]);
    }
    seg.sort((a, b) => a - b);
    const m = median(seg);
    local[i] = m;
    resid[i] = occ[i] - m;
  }
  const absr = valid.map((i) => Math.abs(resid[i])).sort((a, b) => a - b);
  let sigma = 1.4826 * median(absr);
  if (!(sigma > 0)) {
    let s = 0;
    let mean = 0;
    for (const i of valid) mean += resid[i];
    mean /= valid.length;
    for (const i of valid) s += (resid[i] - mean) * (resid[i] - mean);
    sigma = Math.sqrt(s / valid.length);
  }
  if (!(sigma > 0)) {
    return { dips: [], sigma: 0, window: W, reason: "occupancy never moves off its local median" };
  }

  const threshold = -k * sigma;
  const flagged = new Array(n).fill(false);
  for (const i of valid) flagged[i] = resid[i] < threshold;

  const dips = [];
  let i = 0;
  while (i < n) {
    if (!flagged[i]) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < n && flagged[j + 1]) j++;
    let trough = i;
    for (let q = i; q <= j; q++) {
      if (isNum(occ[q]) && (!isNum(occ[trough]) || occ[q] < occ[trough])) trough = q;
    }
    const pre = i > 0 ? i - 1 : null;
    const pick = (arr, idx) => (idx != null && isNum(arr[idx]) ? arr[idx] : null);
    const occBefore = pre != null ? pick(occ, pre) : local[trough];
    const allocBefore = pick(alloc, pre);
    const allocTrough = pick(alloc, trough);
    dips.push({
      i0: i,
      i1: j,
      pre,
      trough,
      // the window a flush at t[pre] .. t[trough] covers, half-open left
      t_start_us: pre != null ? t[pre] : 0,
      t_end_us: t[trough],
      t_last_us: t[j],
      occ_before: occBefore,
      occ_trough: pick(occ, trough),
      occ_local_median: local[trough],
      occ_drop: isNum(occBefore) && isNum(occ[trough]) ? occBefore - occ[trough] : null,
      sigma_below: isNum(resid[trough]) ? -resid[trough] / sigma : null,
      alloc_before: allocBefore,
      alloc_trough: allocTrough,
      alloc_drop:
        isNum(allocBefore) && isNum(allocTrough) ? allocBefore - allocTrough : null,
      healthy_before: pick(healthy, pre),
      healthy_trough: pick(healthy, trough),
      healthy_delta:
        pre != null && isNum(healthy[pre]) && isNum(healthy[trough])
          ? healthy[trough] - healthy[pre]
          : null,
      baseline_known: pre != null,
    });
    i = j + 1;
  }
  return { dips, sigma, window: W, threshold, reason: null };
}

/**
 * Attribute one dip's drop in ALLOCATED chips to the stints that ended
 * inside its window, and expose what is left over.
 *
 * Identity, by construction and asserted in the tests:
 *
 *     freed_total - claimed + residual === alloc_drop
 *
 * `freed_*` are chips released by stints ending in the window bucketed
 * by end_reason; `claimed` is chips taken by stints starting in the same
 * window (they push allocation back up); `residual` is whatever the
 * recorded stints do NOT account for.  This is chips-at-fault
 * bookkeeping, not a causal model: a frame is a bucket mean, a stint is
 * charged whole to the window its end lands in, and jobs on draining or
 * failed nodes keep their chips through the grace period — all of which
 * live in the residual, which is why it is always shown.
 */
export function attributeDip(model, index, dip) {
  const out = {
    ...dip,
    freed: { failed: 0, preempted: 0, drained: 0, other: 0 },
    freed_total: 0,
    claimed: 0,
    residual: null,
    n_ended: 0,
    n_started: 0,
    n_open_at_horizon: 0,
    attributable: false,
  };
  if (!index.n || !isNum(dip.alloc_drop)) return out;
  const st = index.st;
  const ended = index.endsIn(dip.t_start_us, dip.t_end_us);
  const started = index.startsIn(dip.t_start_us, dip.t_end_us);
  out.n_started = started.length;
  for (const i of ended) {
    const reason = String(st.end_reason[i]);
    if (NON_ENDINGS.has(reason)) {
      // still holding at the horizon: recorded with t1 = horizon, but
      // nothing was released, so this is not a freed chip
      out.n_open_at_horizon++;
      continue;
    }
    out.n_ended++;
    const part = END_REASON_PART[reason] || "other";
    out.freed[part] += st.chips[i];
    out.freed_total += st.chips[i];
  }
  for (const i of started) out.claimed += st.chips[i];
  out.residual = dip.alloc_drop - (out.freed_total - out.claimed);
  out.attributable = true;
  return out;
}

/* ------------------------------------------------------------------ *
 * formatting helpers (kept local — this module imports nothing)
 * ------------------------------------------------------------------ */
function fmtClock(us) {
  const s = Math.floor(us / US);
  const d = Math.floor(s / 86400);
  const hh = Math.floor((s % 86400) / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const p = (v) => String(v).padStart(2, "0");
  return (d ? d + "d " : "") + p(hh) + ":" + p(mm) + ":" + p(ss);
}

/** Thin-space grouped integer — locale-independent, so two people
    reading the same run see the same string. */
function fmtCount(v) {
  if (!isNum(v)) return "–";
  const neg = v < 0;
  const digits = String(Math.round(Math.abs(v)));
  let out = "";
  for (let i = 0; i < digits.length; i++) {
    if (i && (digits.length - i) % 3 === 0) out += " ";
    out += digits[i];
  }
  return (neg ? "−" : "") + out;
}
const fmtSigned = (v) => (isNum(v) && v > 0 ? "+" + fmtCount(v) : fmtCount(v));
const fmtPct = (v) => (isNum(v) ? (v * 100).toFixed(1) + "%" : "–");
const fmtNum = (v, n) => (isNum(v) ? v.toFixed(n) : "–");

function eln(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) node.setAttribute(k, String(attrs[k]));
  return node;
}

function panel(title, subtitle) {
  const p = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, title));
  if (subtitle) head.appendChild(eln("span", "sub", subtitle));
  p.appendChild(head);
  return p;
}

function note(parent, text, cls) {
  parent.appendChild(eln("p", cls || "sub insnote", text));
  return parent;
}

/* ------------------------------------------------------------------ *
 * module state
 * ------------------------------------------------------------------ */
let token = 0;
let runId = null;
let model = null;
let index = null;
let mounted = false;

/* UI state, kept across re-renders of the same run */
let eventTime = null; // t_us of the selected event
let pairKey = "frag_pending";
let fragLevel = null;
let kSigma = 2;
/* Rounds of grace-window lookahead the drill-down includes past the
   event's own round.  1 by default: a preempted job keeps its chips
   until its checkpoint-save grace expires, so at 0 every preemption
   wave shows an empty (and technically correct, but useless) table. */
let lookahead = 1;

export function unmountInsight() {
  token++;
  mounted = false;
}

export async function mountInsight(id) {
  const mine = ++token;
  const mount = document.querySelector("#insightMount");
  if (!mount) return;
  if (mounted && runId === id && model) {
    render();
    return;
  }
  if (runId !== id) {
    model = null;
    index = null;
    eventTime = null;
    fragLevel = null;
  }
  runId = id;
  mounted = true;
  if (!model) {
    mount.textContent = "";
    const p = panel("Analysis", "reading the run's replay model");
    note(p, "Loading…");
    mount.appendChild(p);
    let doc = null;
    let err = null;
    try {
      const resp = await fetch("/api/runs/" + encodeURIComponent(id) + "/model");
      if (resp.ok) doc = await resp.json();
      else {
        const body = await resp.json().catch(() => null);
        err = (body && body.error) || "model request failed (" + resp.status + ")";
      }
    } catch (e) {
      err = "cannot reach the server — is fleetsim serve still running?";
    }
    if (mine !== token) return;
    if (!doc) {
      mount.textContent = "";
      const fail = panel("Analysis unavailable", null);
      note(fail, err || "no model");
      mount.appendChild(fail);
      return;
    }
    model = doc;
    index = buildStintIndex(doc);
  }
  if (mine !== token) return;
  render();
}

/* ------------------------------------------------------------------ *
 * render
 * ------------------------------------------------------------------ */
function render() {
  const mount = document.querySelector("#insightMount");
  if (!mount || !model) return;
  // a control that triggers a re-render must not lose the keyboard: the
  // panels are rebuilt wholesale, so the focused control's id is carried
  // across and re-focused on the new node
  const active = document.activeElement;
  const keepId = active && mount.contains(active) && active.id ? active.id : null;
  mount.textContent = "";
  mount.appendChild(headerPanel());
  mount.appendChild(drillPanel());
  mount.appendChild(correlationPanel());
  mount.appendChild(dipPanel());
  if (keepId) {
    const again = mount.querySelector("#" + CSS.escape(keepId));
    if (again) again.focus();
  }
}

function frames() {
  return model.frames || {};
}

/** The model's frame SPACING in microseconds, or null.

    A model frame is NOT a round: `build_viz_model` downsamples the
    timeseries into at most `max_frames` buckets, so on a 150-day run at
    a 60 s round one frame covers 180 minutes — 180 rounds.  Calling
    frames "rounds" while printing the real round length beside them
    understates every window this tab reports by that factor. */
function frameSpanUs() {
  const t = frames().t_us || [];
  if (t.length < 2) return null;
  const span = t[1] - t[0];
  return isNum(span) && span > 0 ? span : null;
}

/** "3h 00m" — a duration for prose, not a clock reading. */
function fmtSpan(us) {
  if (!isNum(us) || us <= 0) return "–";
  const s = Math.round(us / US);
  if (s < 60) return s + " s";
  const m = Math.round(s / 60);
  if (m < 60) return m + " min";
  const h = Math.floor(m / 60);
  return h + "h " + String(m % 60).padStart(2, "0") + "m";
}

/** "21 frames (63h 00m)" — a frame count and what it means in time. */
function framesText(n) {
  const span = frameSpanUs();
  const unit = n === 1 ? " frame" : " frames";
  return n + unit + (span ? " (" + fmtSpan(n * span) + ")" : "");
}

function headerPanel() {
  const p = panel(
    "Analysis",
    "every number below is read out of this run's own recorded output"
  );
  const f = frames();
  const nFrames = (f.t_us || []).length;
  const span = frameSpanUs();
  const bits = [];
  bits.push(nFrames + " frames in the model");
  if (span) bits.push(fmtSpan(span) + " per frame");
  const r = model.meta ? model.meta.round_us : null;
  bits.push(isNum(r) ? "round " + (r / US).toFixed(0) + " s" : "round length unknown");
  bits.push(
    index && index.n
      ? index.n + " stint rows over " + index.domains.length + " domains"
      : "no stint data recorded"
  );
  const row = eln("p", "sub mono", bits.join("  ·  "));
  p.appendChild(row);
  if (span && isNum(r) && r > 0 && span > r) {
    note(
      p,
      "A FRAME IS NOT A ROUND. The model downsamples this run's " +
        (r / US).toFixed(0) + " s rounds into " + nFrames + " frames of " +
        fmtSpan(span) + " each (about " + Math.round(span / r) +
        " rounds per frame), so every window below is stated in frames" +
        " with its real duration beside it."
    );
  }
  note(
    p,
    "Attribution here is bookkeeping over recorded stints and rounds, not" +
      " a causal model. Anything the run did not record is called out as" +
      " unknown rather than filled in."
  );
  /* The model's own reconstruction notes — truncations, degraded
     columns, inferred levels.  The 2D report has always shown these;
     this tab omitting them meant a capped event list read as the whole
     population, with the note that said otherwise on another page. */
  const notes = (model.meta && model.meta.notes) || [];
  if (notes.length) {
    const det = eln("details", "insnotes");
    det.appendChild(
      eln("summary", null, "reconstruction notes (" + notes.length + ")")
    );
    const ul = eln("ul", "insnotelist");
    for (const n of notes) ul.appendChild(eln("li", null, String(n)));
    det.appendChild(ul);
    p.appendChild(det);
  }
  return p;
}

/* ---- 1. event drill-down --------------------------------------------- */

function drillEvents() {
  return (model.events || []).filter((e) => DRILL_KINDS.has(e.kind));
}

function drillPanel() {
  const p = panel(
    "Event drill-down",
    "who stopped, and who took the chips"
  );
  const evs = drillEvents();
  if (!evs.length) {
    note(
      p,
      "This run recorded no preemption waves and no node-failure rounds," +
        " so there is nothing to drill into. (Waves are rounds whose" +
        " preemption count reaches the run's own p99, floor 10.)"
    );
    return p;
  }
  if (!index.n) {
    note(
      p,
      "This run has " + evs.length + " disruption events, but no stint data:" +
        " which jobs stopped cannot be recovered from the fleet-level" +
        " series alone. Re-run with  outputs: {stints: true}  to attribute" +
        " them."
    );
    return p;
  }

  if (eventTime == null) {
    // open on the biggest event: the one most worth explaining
    let best = evs[0];
    for (const e of evs) if (e.magnitude > best.magnitude) best = e;
    eventTime = best.t_us;
  }

  const ctl = eln("div", "ctlrow");
  ctl.appendChild(eln("label", "ctl", "settlement window"));
  const sel = eln("select");
  sel.id = "insLookahead";
  sel.setAttribute("aria-label", "how many rounds past the event to include");
  for (const [value, label] of [
    [0, "the event's round only"],
    [1, "+ 1 round (grace)"],
    [3, "+ 3 rounds"],
    [10, "+ 10 rounds"],
  ]) {
    const opt = eln("option", null, label);
    opt.value = String(value);
    if (value === lookahead) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => {
    lookahead = Number(sel.value);
    render();
  });
  ctl.appendChild(sel);
  ctl.appendChild(
    eln(
      "span",
      "sub",
      "A preemption is counted when the scheduler decides it, but the job" +
        " keeps its chips until its checkpoint-save grace expires — so its" +
        " stint settles a round or more later. Node failures settle inside" +
        " their own round."
    )
  );
  p.appendChild(ctl);

  const layout = eln("div", "insplit");
  const listWrap = eln("div", "evlistwrap");
  const listHead = eln("div", "ctlrow");
  /* The model CAPS the failure event list (300 rounds, sampled across
     the horizon) and records the true count in meta.event_totals.
     Printing the capped length alone read as the whole population — a
     15x undercount on a 4,544-round run, with nothing on screen to say
     so. */
  const shownByKind = {};
  for (const e of evs) shownByKind[e.kind] = (shownByKind[e.kind] || 0) + 1;
  const totals = (model.meta && model.meta.event_totals) || {};
  let trueTotal = 0;
  let capped = false;
  for (const kind of Object.keys(shownByKind)) {
    const real = totals[kind];
    const n = isNum(real) ? real : shownByKind[kind];
    if (n > shownByKind[kind]) capped = true;
    trueTotal += n;
  }
  listHead.appendChild(
    eln(
      "span",
      "ctl",
      capped
        ? evs.length + " of " + trueTotal.toLocaleString() + " disruption rounds"
        : evs.length + " disruption rounds"
    )
  );
  listWrap.appendChild(listHead);
  if (capped) {
    note(
      listWrap,
      "The model caps its event list, so this is a SAMPLE: the run was" +
        " split into equal time windows and the largest round in each was" +
        " kept, which is why the ticks span the whole run instead of" +
        " clustering at the start. The full count above comes from the" +
        " run's own timeseries."
    );
  }
  const list = eln("ul", "evlist");
  list.setAttribute("role", "list");
  list.setAttribute("aria-label", "disruption rounds — pick one to explain it");
  const shown = evs.slice(0, MAX_EVENT_ROWS);
  shown.forEach((e, i) => {
    const li = eln("li");
    const btn = eln("button", "evbtn");
    btn.type = "button";
    // stable id so render()'s focus carry-over survives the rebuild
    btn.id = "insEv" + i;
    // the two spans below carry no separating whitespace, so an implicit
    // name would read "00:29:00node failures: 1"
    btn.setAttribute("aria-label", fmtClock(e.t_us) + " — " + e.label);
    if (e.t_us === eventTime) btn.setAttribute("aria-current", "true");
    const dot = eln("span", "evdot k-" + e.kind);
    dot.setAttribute("aria-hidden", "true");
    btn.appendChild(dot);
    const main = eln("span", "evmain");
    main.appendChild(eln("span", "evt mono", fmtClock(e.t_us)));
    main.appendChild(eln("span", "evlabel", e.label));
    btn.appendChild(main);
    btn.addEventListener("click", () => {
      eventTime = e.t_us;
      render();
    });
    li.appendChild(btn);
    list.appendChild(li);
  });
  listWrap.appendChild(list);
  if (evs.length > shown.length) {
    note(
      listWrap,
      "Showing the first " + shown.length + " of " + evs.length + " rounds."
    );
  }
  layout.appendChild(listWrap);

  const detail = eln("div", "evdetail");
  renderDrillDetail(detail);
  layout.appendChild(detail);
  p.appendChild(layout);
  return p;
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

function renderDrillDetail(host) {
  const d = drillDown(model, index, eventTime, lookahead);
  const ev = drillEvents().find((e) => e.t_us === eventTime);
  const head = eln("div", "phead");
  head.appendChild(
    eln("h2", null, ev ? ev.label : "round ending " + fmtClock(eventTime))
  );
  head.appendChild(eln("span", "sub mono", "at " + fmtClock(eventTime)));
  host.appendChild(head);

  if (!d.ok && d.reason === "no-round") {
    note(
      host,
      "The model could not determine this run's round length (fewer than" +
        " two timeseries samples), so the round a stint ended in is not" +
        " defined. Nothing can be attributed."
    );
    return;
  }
  if (!d.ok) {
    note(host, "No stint data for this run.");
    return;
  }

  const stats = eln("div", "statrow");
  const stat = (label, value, title) => {
    const box = eln("div", "stat");
    box.appendChild(eln("span", "clabel", label));
    box.appendChild(eln("span", "cval mono", value));
    if (title) box.title = title;
    stats.appendChild(box);
  };
  stat(
    "jobs stopped",
    fmtCount(d.n_jobs) + (d.n_jobs && d.n_in_round < d.n_jobs
      ? " (" + d.n_in_round + " in-round)" : ""),
    "distinct jobs whose stints ended in the window for a disruptive reason;" +
      " “in-round” counts those that settled inside the event's own round" +
      " rather than during the grace lookahead"
  );
  stat("chips freed", fmtCount(d.chips_freed), "sum of the chips those stints were holding");
  stat("chips claimed", fmtCount(d.chips_claimed), "chips taken by every stint that started in the same window");
  stat(
    "→ to higher tiers",
    fmtCount(d.chips_to_displacers),
    "of the chips claimed, those taken on a domain that just freed chips by a gang at a strictly higher tier"
  );
  host.appendChild(stats);

  const paid = d.chips_freed > 0 ? d.chips_to_displacers / d.chips_freed : null;
  const split =
    fmtCount(d.chips_to_displacers) + " of " + fmtCount(d.chips_freed) + " chips";
  let verdict;
  if (paid == null) {
    verdict = "No chips were freed in this window.";
  } else if (paid > 1) {
    verdict =
      "Higher-tier gangs on the same domains took MORE than this event" +
      " freed (" + split + ", " + fmtPct(paid) + ") — the surplus came" +
      " from capacity that was already free.";
  } else {
    verdict =
      fmtPct(paid) + " of the freed chips were picked up inside the same" +
      " window by higher-tier gangs on the same domains (" + split +
      "). The rest went back to the free pool, to same-or-lower-tier work," +
      " or stayed idle until a later round.";
  }
  note(host, "Did this event pay for itself? " + verdict);

  if (!d.jobs.length) {
    note(
      host,
      "No stint ended in (" + fmtClock(d.window[0]) + ", " +
        fmtClock(d.window[1]) + "] with end_reason preempted, failed or" +
        " drained. The event counter moved, but the jobs it counted still" +
        " held their chips at the end of this window — widen the settlement" +
        " window above to follow them."
    );
    return;
  }

  const rows = d.jobs.slice(0, MAX_VICTIM_ROWS);
  const wrap = tableWrap("disruption victims");
  const table = eln("table", "matrix");
  const thead = eln("thead");
  const hr = eln("tr");
  for (const [label, cls, title] of [
    ["job", "keycol", "the job whose stint ended in this round"],
    ["class", null, "workload class label"],
    ["tier", null, "priority band the job held"],
    ["chips", "num", "chips this job released in this round"],
    ["domain", null, "map domains the job was holding"],
    ["ended", null, "stint end_reason as recorded"],
    ["displaced by (inferred)", null, "a gang that started in the SAME round on the SAME domain at a strictly higher tier — the model records no causal link, so this is an inference"],
  ]) {
    const th = eln("th", cls, label);
    th.setAttribute("scope", "col");
    if (title) th.title = title;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = eln("tbody");
  for (const job of rows) {
    const tr = eln("tr");
    const th = eln("th", "keycol", job.job_id);
    th.setAttribute("scope", "row");
    tr.appendChild(th);
    tr.appendChild(eln("td", null, job.class_name));
    tr.appendChild(eln("td", null, job.tier));
    tr.appendChild(eln("td", "num", fmtCount(job.chips)));
    const dom = eln("td", "mono", job.domains.slice(0, 2).join(", ") +
      (job.domains.length > 2 ? " +" + (job.domains.length - 2) : ""));
    dom.title = job.domains.join("\n");
    tr.appendChild(dom);
    tr.appendChild(eln("td", null, job.end_reasons.join(", ")));

    const td = eln("td");
    if (job.displacers.length) {
      const top = job.displacers[0];
      const span = eln("span", "inferred");
      span.appendChild(eln("span", "inftag", "inferred"));
      span.appendChild(
        eln(
          "span",
          "mono",
          " " + top.job_id + " (" + top.tier + ", " + fmtCount(top.chips) + " chips)"
        )
      );
      if (job.displacers.length > 1) {
        span.appendChild(
          eln("span", "sub", " +" + (job.displacers.length - 1) + " other candidate" +
            (job.displacers.length > 2 ? "s" : ""))
        );
      }
      td.title =
        "candidates (started in the same round on " + job.domains.join(", ") +
        " at a higher tier):\n" +
        job.displacers
          .map((x) => x.job_id + " · " + x.tier + " · " + x.chips + " chips")
          .join("\n");
      td.appendChild(span);
    } else {
      td.className = "nosummary";
      td.textContent = job.rank_known
        ? "not determinable"
        : "tier not rankable";
      td.title = job.rank_known
        ? "no stint started in this round on this job's domain at a higher tier — the chips went somewhere the model does not record"
        : "this job's tier is not one of the known bands, so no ordering can be applied";
      td.appendChild(eln("span", "sronly", " — no higher-tier starter on this domain in this round"));
    }
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  host.appendChild(wrap);
  if (d.jobs.length > rows.length) {
    note(host, "Showing the " + rows.length + " largest of " + d.jobs.length + " jobs.");
  }
  note(
    host,
    "Window = (" + fmtClock(d.window[0]) + ", " + fmtClock(d.window[1]) + "]" +
      " — the round this metrics flush covers" +
      (d.window[1] > d.round_end_us
        ? " plus " + Math.round((d.window[1] - d.round_end_us) / d.round_us) +
          " round(s) of settlement lookahead."
        : ".") +
      " “displaced by” is an INFERENCE from co-location and tier order;" +
      " fleetsim records no victim→beneficiary link, so a blank cell means" +
      " the evidence is absent, not that nothing happened."
  );
}

/* ---- 2. correlation --------------------------------------------------- */

function fragLevels() {
  return Object.keys((frames().frag_index) || {}).sort();
}

function currentFragLevel() {
  const levels = fragLevels();
  if (!levels.length) return null;
  if (fragLevel && levels.includes(fragLevel)) return fragLevel;
  const hint = model.fleet ? model.fleet.map_level : null;
  return hint && levels.includes(hint) ? hint : levels[levels.length - 1];
}

function pairSpecs() {
  const f = frames();
  const lvl = currentFragLevel();
  const frag = lvl ? (f.frag_index || {})[lvl] || [] : [];
  const pending = f.pending_jobs || [];
  const occ = f.occupancy || [];
  const good = f.goodput_to_date || [];
  const fails = f.failures_delta || [];
  return [
    {
      key: "frag_pending",
      label: "fragmentation vs queue pressure",
      x: { v: frag, label: "fragmentation index" + (lvl ? " (" + lvl + ")" : ""), fmt: (v) => fmtNum(v, 3) },
      y: { v: pending, label: "pending jobs", fmt: fmtCount },
      caveat:
        "Fragmentation index and pending-job count are both per-round samples" +
        " from timeseries.parquet. Pending jobs is the queue-wait PROXY: the" +
        " model carries no per-round wait, and a job's wait is only known once" +
        " it starts.",
    },
    {
      key: "occ_goodput",
      label: "occupancy vs goodput",
      x: { v: occ, label: "occupancy (allocated / healthy)", fmt: fmtPct },
      y: { v: good, label: "goodput to date", fmt: fmtPct },
      caveat:
        "Goodput is the TO-DATE ratio, the only goodput a round carries:" +
        " fleetsim credits productive chip-time in a lump when a stint" +
        " settles, so a per-round goodput would be a settlement artefact" +
        " rather than a rate. Early rounds move a cumulative series a lot" +
        " more than late ones — read the trend, not the slope.",
    },
    {
      key: "fail_occ",
      label: "node failures vs occupancy",
      x: { v: fails, label: "node failures in the round", fmt: fmtCount },
      y: { v: occ, label: "occupancy (allocated / healthy)", fmt: fmtPct },
      caveat:
        "Both are same-round values: a failure's cost usually lands in the" +
        " NEXT round or two, so a flat r here does not mean failures are" +
        " free. The dip panel below is the lagged view.",
    },
    {
      key: "frag_occ",
      label: "fragmentation vs occupancy",
      x: { v: frag, label: "fragmentation index" + (lvl ? " (" + lvl + ")" : ""), fmt: (v) => fmtNum(v, 3) },
      y: { v: occ, label: "occupancy (allocated / healthy)", fmt: fmtPct },
      caveat:
        "The fragmentation index is defined from free capacity, so it moves" +
        " with occupancy by construction — this pair shows how tightly, not" +
        " whether one causes the other.",
    },
  ];
}

function correlationPanel() {
  const p = panel(
    "Correlation",
    "one run's rounds — a relationship, never a cause"
  );
  const specs = pairSpecs();
  const spec = specs.find((s) => s.key === pairKey) || specs[0];

  const ctl = eln("div", "ctlrow");
  ctl.appendChild(eln("label", "ctl", "pair"));
  const sel = eln("select");
  sel.id = "insPair";
  sel.setAttribute("aria-label", "correlation pair");
  for (const s of specs) {
    const opt = eln("option", null, s.label);
    opt.value = s.key;
    if (s.key === spec.key) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => {
    pairKey = sel.value;
    render();
  });
  ctl.appendChild(sel);

  const levels = fragLevels();
  if (levels.length > 1) {
    ctl.appendChild(eln("label", "ctl", "fleet level"));
    const lv = eln("select");
    lv.id = "insFragLevel";
    lv.setAttribute("aria-label", "fragmentation fleet level");
    for (const l of levels) {
      const opt = eln("option", null, l);
      opt.value = l;
      if (l === currentFragLevel()) opt.selected = true;
      lv.appendChild(opt);
    }
    lv.addEventListener("change", () => {
      fragLevel = lv.value;
      render();
    });
    ctl.appendChild(lv);
  }
  p.appendChild(ctl);

  if (!spec.x.v.length || !spec.y.v.length) {
    note(
      p,
      "This run's model carries no series for one side of this pair" +
        (levels.length ? "" : " (no fragmentation index was recorded)") +
        " — nothing to plot."
    );
    return p;
  }

  const fit = pearsonFit(spec.x.v, spec.y.v);
  p.appendChild(scatter(spec, fit));

  const stats = eln("p", "insstats mono");
  if (fit.r == null) {
    stats.textContent =
      "r = not computable (" + fit.reason + ") · n = " + framesText(fit.n);
  } else {
    const c = fit.intercept;
    stats.textContent =
      "Pearson r = " + fit.r.toFixed(3) +
      "   ·   r² = " + (fit.r * fit.r).toFixed(3) +
      "   ·   fit: y = " + fit.slope.toPrecision(4) + " x " +
      (c < 0 ? "− " : "+ ") + Math.abs(c).toPrecision(4) +
      "   ·   n = " + framesText(fit.n);
  }
  p.appendChild(stats);
  note(
    p,
    "This is a CORRELATION over the rounds of ONE run under ONE scenario." +
      " It is not causation, and it does not generalise: r moves with the" +
      " seed, the horizon and the round length. To make a claim, sweep the" +
      " parameter and compare runs."
  );
  note(p, spec.caveat);
  return p;
}

const SW = 860;
const SH = 300;
const SPAD = { l: 62, r: 18, t: 14, b: 40 };

/* 1.25/1.5 are in the ladder because occupancy legitimately runs a few
   percent past 1 (grace-period jobs stay allocated on draining or failed
   nodes), and a 200% axis for a 105% maximum is unreadable. */
function niceMax(v) {
  if (!isNum(v) || v <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  for (const step of [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 10]) {
    if (v <= step * pow) return step * pow;
  }
  return 10 * pow;
}

/* PNG export.  export.js is imported LAZILY and never at module load:
   tests/test_viz_insight.py executes THIS FILE bare under node (copying
   only insight.js into a temp directory), so a static import would leave
   the harness with an unresolvable specifier.  A dynamic import inside a
   function body is only resolved if the function runs, and the kernels
   the harness drives never build a chart. */
function attachPng(cap, svg, name) {
  import("./export.js")
    .then((m) => cap.appendChild(m.pngButton(svg, name)))
    .catch(() => {}); /* no export button is better than a broken panel */
}

function scatter(spec, fit) {
  const wrap = eln("div", "chart");
  const cap = eln("div", "chartcap");
  cap.appendChild(eln("span", "charttitle", spec.y.label + " vs " + spec.x.label));
  const readout = eln("span", "chartreadout mono");
  readout.setAttribute("role", "status");
  cap.appendChild(readout);
  wrap.appendChild(cap);

  const xs = fit.xs || [];
  const ys = fit.ys || [];
  if (!xs.length) {
    note(wrap, "No model frame has a value on both axes.");
    return wrap;
  }
  const xMax = niceMax(Math.max(...xs, 0));
  const yMax = niceMax(Math.max(...ys, 0));
  const xMin = Math.min(0, ...xs);
  const yMin = Math.min(0, ...ys);
  const X = (v) => SPAD.l + ((v - xMin) / (xMax - xMin || 1)) * (SW - SPAD.l - SPAD.r);
  const Y = (v) => SH - SPAD.b - ((v - yMin) / (yMax - yMin || 1)) * (SH - SPAD.t - SPAD.b);

  const svg = svgEl("svg", {
    viewBox: "0 0 " + SW + " " + SH,
    class: "chartsvg",
    tabindex: "0",
    role: "img",
    "aria-label":
      spec.y.label + " against " + spec.x.label + " for " + xs.length +
      " rounds of this run" +
      (fit.r == null ? ", correlation not computable" : ", Pearson r " + fit.r.toFixed(3)),
  });

  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const yv = yMin + (yMax - yMin) * frac;
    const y = Y(yv);
    svg.appendChild(svgEl("line", { x1: SPAD.l, x2: SW - SPAD.r, y1: y, y2: y, stroke: GRID }));
    const lab = svgEl("text", {
      x: SPAD.l - 6, y: y + 3.5, "text-anchor": "end", fill: MUTED, "font-size": "9.5",
    });
    lab.textContent = spec.y.fmt(yv);
    svg.appendChild(lab);
  }
  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const xv = xMin + (xMax - xMin) * frac;
    const x = X(xv);
    svg.appendChild(
      svgEl("line", { x1: x, x2: x, y1: SPAD.t, y2: SH - SPAD.b, stroke: GRID })
    );
    const lab = svgEl("text", {
      x, y: SH - SPAD.b + 14,
      "text-anchor": frac === 0 ? "start" : frac === 1 ? "end" : "middle",
      fill: MUTED, "font-size": "9.5",
    });
    lab.textContent = spec.x.fmt(xv);
    svg.appendChild(lab);
  }
  const xTitle = svgEl("text", {
    x: (SPAD.l + SW - SPAD.r) / 2, y: SH - 6, "text-anchor": "middle",
    fill: MUTED, "font-size": "10",
  });
  xTitle.textContent = spec.x.label;
  svg.appendChild(xTitle);

  for (let i = 0; i < xs.length; i++) {
    const dot = svgEl("circle", {
      cx: X(xs[i]).toFixed(1), cy: Y(ys[i]).toFixed(1), r: 3,
      fill: ACCENT, "fill-opacity": "0.55", stroke: PANEL, "stroke-width": "1",
    });
    svg.appendChild(dot);
  }

  if (fit.r != null) {
    /* CLIP THE FIT TO THE OBSERVED x RANGE.  Drawn across the whole axis
       it asserts a relationship over x values the run never reached — on
       a run whose fragmentation never fell below 0.60, two thirds of the
       visible line covered fragmentation that never happened, and a line
       is the first thing read. */
    const xLo = Math.min(...xs);
    const xHi = Math.max(...xs);
    const y0 = fit.intercept + fit.slope * xLo;
    const y1 = fit.intercept + fit.slope * xHi;
    const clampY = (v) => Math.max(yMin, Math.min(yMax, v));
    svg.appendChild(
      svgEl("line", {
        x1: X(xLo), x2: X(xHi), y1: Y(clampY(y0)), y2: Y(clampY(y1)),
        stroke: TEXT, "stroke-width": "2", "stroke-dasharray": "6 4",
        "stroke-linecap": "round", opacity: "0.75",
      })
    );
    fit._drawnRange = [xLo, xHi];
  }

  const marker = svgEl("circle", {
    r: 5.5, fill: "none", stroke: TEXT, "stroke-width": "2", visibility: "hidden",
  });
  svg.appendChild(marker);

  const showIdx = (i) => {
    if (i < 0 || i >= xs.length) return;
    marker.setAttribute("cx", X(xs[i]));
    marker.setAttribute("cy", Y(ys[i]));
    marker.setAttribute("visibility", "visible");
    readout.textContent =
      "round " + (i + 1) + " of " + xs.length + "  ·  " +
      spec.x.label + " " + spec.x.fmt(xs[i]) + "  ·  " +
      spec.y.label + " " + spec.y.fmt(ys[i]);
  };
  const hide = () => {
    marker.setAttribute("visibility", "hidden");
    readout.textContent = "";
  };

  svg.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / (r.width || 1)) * SW;
    const py = ((ev.clientY - r.top) / (r.height || 1)) * SH;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const dx = X(xs[i]) - px;
      const dy = Y(ys[i]) - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0 && bestD <= 40 * 40) showIdx(best);
    else hide();
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);

  let kb = -1;
  svg.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight") kb = kb < 0 ? 0 : Math.min(xs.length - 1, kb + 1);
    else if (ev.key === "ArrowLeft") kb = kb < 0 ? xs.length - 1 : Math.max(0, kb - 1);
    else if (ev.key === "Home") kb = 0;
    else if (ev.key === "End") kb = xs.length - 1;
    else if (ev.key === "Escape") {
      kb = -1;
      hide();
      return;
    } else return;
    ev.preventDefault();
    showIdx(kb);
  });

  wrap.appendChild(svg);
  attachPng(cap, svg, "insight-correlation.png");
  note(
    wrap,
    "One dot per model frame (in time order — use ← → to walk them). The" +
      " dashed line is the ordinary least-squares fit of y on x, drawn" +
      " ONLY over the x range this run actually visited" +
      (fit._drawnRange
        ? " (" + fmtNum(fit._drawnRange[0], 3) + " … " +
          fmtNum(fit._drawnRange[1], 3) + ")"
        : "") +
      " — it is not extrapolated to values the run never reached."
  );
  return wrap;
}

/* ---- 3. occupancy-dip attribution ------------------------------------ */

function dipPanel() {
  const p = panel(
    "Occupancy dips",
    "where the chips went, and what is left unexplained"
  );
  const f = frames();
  const ctl = eln("div", "ctlrow");
  ctl.appendChild(eln("label", "ctl", "sensitivity k (sigma below local median)"));
  const range = eln("input");
  range.type = "range";
  range.id = "insK";
  range.min = "0.5";
  range.max = "6";
  range.step = "0.5";
  range.value = String(kSigma);
  range.className = "insrange";
  range.setAttribute("aria-label", "dip sensitivity in sigma below the local median");
  const kOut = eln("span", "sizenote mono", "k = " + kSigma.toFixed(1));
  kOut.setAttribute("role", "status");
  range.addEventListener("input", () => {
    kSigma = Number(range.value);
    kOut.textContent = "k = " + kSigma.toFixed(1);
  });
  range.addEventListener("change", () => {
    kSigma = Number(range.value);
    render();
  });
  ctl.appendChild(range);
  ctl.appendChild(kOut);
  p.appendChild(ctl);

  const found = detectDips(f, kSigma);
  if (found.reason) {
    note(p, "No dips detectable: " + found.reason + ".");
    return p;
  }
  if (!found.dips.length) {
    note(
      p,
      "No round sits more than " + kSigma.toFixed(1) + " sigma below its local" +
        " median (robust sigma = " + found.sigma.toPrecision(3) + " occupancy)." +
        " Lower k to widen the net."
    );
    return p;
  }

  const attributed = found.dips.map((d) => attributeDip(model, index, d));
  attributed.sort(
    (a, b) => (b.alloc_drop || 0) - (a.alloc_drop || 0) || a.t_end_us - b.t_end_us
  );

  note(
    p,
    found.dips.length + " dip" + (found.dips.length === 1 ? "" : "s") +
      " at k = " + kSigma.toFixed(1) + " (robust sigma = " +
      found.sigma.toPrecision(3) + " occupancy, local median over " +
      framesText(found.window) + "). Deepest first. A dip's window is at" +
      " least one model frame" +
      (frameSpanUs() ? " — " + fmtSpan(frameSpanUs()) + " here" : "") +
      ", not one scheduler round."
  );

  if (!index.n) {
    note(
      p,
      "This run recorded no stints, so a dip cannot be decomposed at all —" +
        " only its size is known. Re-run with  outputs: {stints: true}."
    );
  } else {
    p.appendChild(dipBars(attributed.slice(0, MAX_DIP_BARS)));
  }
  p.appendChild(dipTable(attributed.slice(0, MAX_DIP_ROWS)));
  if (attributed.length > MAX_DIP_ROWS) {
    note(p, "Showing the " + MAX_DIP_ROWS + " deepest of " + attributed.length + " dips.");
  }
  note(
    p,
    "APPROXIMATE, by construction. This is chips-at-fault bookkeeping over" +
      " the stints whose end lands inside the dip window — not a causal" +
      " model. A stint is charged whole to one window, a frame is a bucket" +
      " mean of its samples, and jobs on draining or failed nodes keep their" +
      " chips through the grace period; all of that lands in the residual," +
      " which is why the residual is drawn rather than absorbed."
  );
  return p;
}

const BAR_H = 26;
const BAR_GAP = 12;
const BAR_LABEL_W = 118;
const BAR_VALUE_W = 92;
const BAR_W = 860;

function dipBars(dips) {
  const wrap = eln("div", "chart");
  const cap = eln("div", "chartcap");
  cap.appendChild(eln("span", "charttitle", "Chips freed inside each dip window, by cause"));
  cap.appendChild(eln("span", "sub", "plus the unexplained residual"));
  wrap.appendChild(cap);

  const rows = dips.filter((d) => d.attributable);
  if (!rows.length) {
    note(wrap, "None of these dips has a measurable drop in allocated chips.");
    return wrap;
  }
  let maxTotal = 1;
  for (const d of rows) {
    maxTotal = Math.max(maxTotal, d.freed_total + Math.max(0, d.residual || 0));
  }
  const plotW = BAR_W - BAR_LABEL_W - BAR_VALUE_W;
  const H = rows.length * (BAR_H + BAR_GAP) + 8;
  const svg = svgEl("svg", {
    viewBox: "0 0 " + BAR_W + " " + H,
    class: "chartsvg",
    role: "img",
    "aria-label":
      "Stacked chips freed by cause for " + rows.length +
      " occupancy dips, with the unexplained residual as the last segment",
  });

  const defs = svgEl("defs");
  const pat = svgEl("pattern", {
    id: "insResidualHatch",
    width: "6",
    height: "6",
    patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  pat.appendChild(svgEl("rect", { width: "6", height: "6", fill: "rgba(139,147,161,.22)" }));
  pat.appendChild(
    svgEl("rect", { width: "2.5", height: "6", fill: RESIDUAL_COLOR, "fill-opacity": "0.85" })
  );
  defs.appendChild(pat);
  svg.appendChild(defs);

  rows.forEach((d, ri) => {
    const y = ri * (BAR_H + BAR_GAP) + 4;
    const lab = svgEl("text", {
      x: 0, y: y + BAR_H / 2 + 4, fill: MUTED, "font-size": "10.5",
    });
    lab.textContent = fmtClock(d.t_end_us);
    svg.appendChild(lab);

    let x = BAR_LABEL_W;
    const seg = (value, fill, title) => {
      if (!(value > 0)) return;
      const w = (value / maxTotal) * plotW;
      const rect = svgEl("rect", {
        x: x.toFixed(1), y, width: Math.max(0.5, w - 2).toFixed(1), height: BAR_H,
        rx: 3, fill,
      });
      const t = svgEl("title");
      t.textContent = title + ": " + fmtCount(value) + " chips";
      rect.appendChild(t);
      svg.appendChild(rect);
      x += w;
    };
    for (const part of DIP_PARTS) seg(d.freed[part.key], part.color, part.label);
    if (d.residual > 0) seg(d.residual, "url(#insResidualHatch)", "unexplained residual");

    const val = svgEl("text", {
      x: BAR_W - 2, y: y + BAR_H / 2 + 4, "text-anchor": "end",
      fill: TEXT, "font-size": "10.5", "font-weight": "600",
    });
    val.textContent = fmtCount(d.alloc_drop) + " chips";
    svg.appendChild(val);
  });
  wrap.appendChild(svg);
  attachPng(cap, svg, "insight-dip-attribution.png");

  const legend = eln("div", "insleg");
  for (const part of DIP_PARTS) {
    const item = eln("span", "inskey");
    const sw = eln("i");
    sw.style.background = part.color;
    item.appendChild(sw);
    item.appendChild(eln("span", null, part.label));
    legend.appendChild(item);
  }
  const res = eln("span", "inskey");
  const resSw = eln("i", "hatch");
  res.appendChild(resSw);
  res.appendChild(eln("span", null, "unexplained residual"));
  legend.appendChild(res);
  wrap.appendChild(legend);
  note(
    wrap,
    "Bar length is chips FREED in the window plus a positive residual; the" +
      " number at the right is the drop in allocated chips the frames" +
      " actually show. They differ by the chips re-claimed inside the same" +
      " window (see the table) — freed − re-claimed + residual = observed" +
      " drop, exactly." +
      (frameSpanUs()
        ? " Each window is at least one model frame (" +
          fmtSpan(frameSpanUs()) + "), so these are the endings over that" +
          " interval, not over one scheduler round — the table's 'window'" +
          " column gives each one exactly."
        : "")
  );
  return wrap;
}

function dipTable(dips) {
  const wrap = tableWrap("occupancy dips");
  const table = eln("table", "matrix");
  const thead = eln("thead");
  const hr = eln("tr");
  const cols = [
    ["window ends", "keycol", "the frame the dip bottoms out in"],
    ["window", "num", "how long the attribution window is — chips freed and re-claimed below are counted over THIS interval, which is one or more model frames, not one scheduler round"],
    ["σ below", "num", "how far below its local median the trough sits"],
    ["occupancy", "num", "occupancy before the dip → at the trough"],
    ["Δ allocated", "num", "drop in allocated chips from the frame before the dip to the trough"],
    ["Δ healthy", "num", "change in HEALTHY chips over the same window: a negative number means the fleet itself shrank"],
    ["failures", "num", "chips freed by stints ending with end_reason failed"],
    ["preempt", "num", "chips freed by stints ending with end_reason preempted"],
    ["drains", "num", "chips freed by stints ending with end_reason drained"],
    ["normal", "num", "chips freed by stints ending completed / canceled / timeout"],
    ["re-claimed", "num", "chips taken by stints that STARTED inside the same window"],
    ["residual", "num", "observed drop − (freed − re-claimed): what the recorded stints do not explain"],
  ];
  for (const [label, cls, title] of cols) {
    const th = eln("th", cls, label);
    th.setAttribute("scope", "col");
    th.title = title;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = eln("tbody");
  for (const d of dips) {
    const tr = eln("tr");
    const th = eln("th", "keycol mono", fmtClock(d.t_end_us));
    th.setAttribute("scope", "row");
    th.title =
      "window (" + fmtClock(d.t_start_us) + ", " + fmtClock(d.t_end_us) + "]" +
      (d.baseline_known ? "" : " — the dip starts at the first frame, so there is no earlier baseline");
    tr.appendChild(th);
    /* the attribution INTERVAL, spelled out: reading "430 chips freed"
       as one 60 s round when the window is 3 hours is what makes
       disruption look like it explains almost nothing */
    const span = eln("td", "num mono", fmtSpan(d.t_end_us - d.t_start_us));
    span.title =
      "window (" + fmtClock(d.t_start_us) + ", " + fmtClock(d.t_end_us) + "]";
    tr.appendChild(span);
    tr.appendChild(eln("td", "num", fmtNum(d.sigma_below, 1)));
    const occCell = eln(
      "td", "num",
      fmtPct(d.occ_before) + " → " + fmtPct(d.occ_trough)
    );
    occCell.title = "local median at the trough: " + fmtPct(d.occ_local_median);
    tr.appendChild(occCell);
    tr.appendChild(eln("td", "num", fmtCount(d.alloc_drop)));
    const hc = eln("td", "num", fmtSigned(d.healthy_delta));
    if (isNum(d.healthy_delta) && d.healthy_delta < 0) {
      hc.title = "healthy chips fell too: part of this dip is the DENOMINATOR, not idle capacity";
      hc.className = "num warncell";
    }
    tr.appendChild(hc);
    if (!d.attributable) {
      const td = eln("td", "nosummary", "no stint data — not decomposable");
      td.colSpan = 6;
      tr.appendChild(td);
      tbody.appendChild(tr);
      continue;
    }
    for (const part of DIP_PARTS) {
      tr.appendChild(eln("td", "num", fmtCount(d.freed[part.key])));
    }
    tr.appendChild(eln("td", "num", fmtCount(d.claimed)));
    const rc = eln("td", "num residualcell", fmtSigned(d.residual));
    rc.title =
      "observed drop " + fmtCount(d.alloc_drop) + " − (freed " +
      fmtCount(d.freed_total) + " − re-claimed " + fmtCount(d.claimed) + ") = " +
      fmtCount(d.residual) + " chips the recorded stints do not explain" +
      "\n" + d.n_ended + " stints ended and " + d.n_started +
      " started inside this window" +
      (d.n_open_at_horizon
        ? "; " + d.n_open_at_horizon +
          " more were still holding chips at the horizon and are NOT counted as freed"
        : "");
    tr.appendChild(rc);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}
