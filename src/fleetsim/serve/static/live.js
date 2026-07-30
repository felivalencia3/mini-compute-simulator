/* fleetsim v0.8 — the live replay client.

   One module, two consumers: the run view's live fleet map (app.js) and
   the 3D fleet replay (fleet3d.js).  It owns the cursor contract for
   `GET /api/runs/{id}/live` and nothing else — no DOM, no drawing.

   THE CONTRACT, restated because the code depends on every clause
   (docs/webapp.md "Live replay" is the long form):

     - `cursor` counts SETTLED rows already consumed.  Start at 0 and
       always send back the cursor the previous response returned.
     - `stints` are settled rows after that cursor, immutable, delivered
       exactly once ever, in settlement order.
     - `more: true` means rows remain on disk: poll again IMMEDIATELY —
       UNLESS the cursor did not move, which means the stream cannot
       advance (a corrupt spool line; the server names it in
       `stalled_at`).  Re-polling THAT at 0 ms is an unthrottled request
       loop against a single-threaded stdlib server: measured 1,240
       req/s from one tab, forever.  So a response that advances nothing
       falls back to the normal poll spacing and reports the stall.
     - `open_stints` REPLACES the open overlay wholesale, and is null
       while `more` is true (a lagging client's settled prefix does not
       line up with it) or while `open_pending` says the child is not
       spooling it yet (nobody was polling at the last flush).
     - `fleet` arrives only on a cursor<=0 request and never changes.
     - `progress` always reflects the latest flush, even when rows lag.

   An open stint is stored with end_reason "running_at_horizon" — the
   same token the finished model uses for a stint that never released its
   chips — so the replay's interval sweep keeps it allocated at the
   leading edge without a single live-only branch. */

"use strict";

/** Poll spacing once caught up.  A metrics flush is one scheduler round,
    typically far slower than this; polling faster only burns CPU. */
export const LIVE_POLL_MS = 1000;

/** Backoff after a network error (the server may be restarting). */
const LIVE_RETRY_MS = 2500;

/** Hard cap on retained live rows.  A frontier run can settle millions
    of stints; past this the feed stops accumulating and says so rather
    than growing the tab's heap without bound.  "Says so" means BOTH
    consumers: the 3D HUD note and the 2D live map's sub-line each read
    `stints.truncated`, and both read the server's own overlay cap
    (`openTruncated`) as well — a half-drawn fleet with no warning is
    worse than no fleet. */
export const LIVE_MAX_ROWS = 300000;

/** The server's own cap on the open-stint overlay in ONE response
    (`runs.LIVE_ROW_LIMIT`).  Mirrored here only so a consumer can NAME
    the number when it reports the overlay as truncated. */
export const LIVE_ROW_LIMIT = 5000;

const TERMINAL = new Set(["done", "failed"]);

/* ------------------------------------------------------------------ *
 * LiveStints — the accumulating columnar index
 * ------------------------------------------------------------------ */
export class LiveStints {
  constructor() {
    this.fleet = null;        /* {map_level, clusters, domains} */
    this.domains = [];        /* flat ordered id list; index == domain_idx */
    this.domainIdx = new Map();
    this.classes = [];        /* observed class labels, first-seen order */
    this.truncated = false;
    this.version = 0;         /* bumped on every mutation */
    this._settled = emptyCols();
    this._open = emptyCols();
    /* label -> {bucket: votes}: the palette BUCKET evidence, counted the
       way fleetsim.viz.data._label_buckets counts it — once per JOB, not
       once per stint row (a gang spans many domains and would otherwise
       out-vote a single-domain class). `_voted` is the job-id set that
       makes each job count once, including across overlay replacements. */
    this._votes = new Map();
    this._voted = new Set();
  }

  get nSettled() { return this._settled.chips.length; }
  get nOpen() { return this._open.chips.length; }

  /** label -> palette bucket, by majority vote with ties broken in
      canonical palette order — the rule fleetsim.viz.data._label_buckets
      applies to the finished run.  (One knowable difference: a job whose
      TIER changes mid-run votes with the tier it was first seen at, the
      only tier a live view has; the model uses the final one.) */
  buckets() {
    const out = {};
    for (const [label, votes] of this._votes) {
      let best = null;
      let bestKey = null;
      for (const [bucket, n] of votes) {
        const key = [n, -BUCKET_ORDER.indexOf(bucket)];
        if (bestKey === null || key[0] > bestKey[0] ||
            (key[0] === bestKey[0] && key[1] > bestKey[1])) {
          best = bucket;
          bestKey = key;
        }
      }
      out[label] = best || "best_effort";
    }
    return out;
  }

  setFleet(fleet) {
    if (!fleet || this.fleet) return;
    this.fleet = fleet;
    this.domains = Array.isArray(fleet.domains) ? fleet.domains.slice() : [];
    this.domainIdx = new Map(this.domains.map((id, i) => [id, i]));
    this.version++;
  }

  /* A row's domain is a NAME in the stream and an INDEX in the model;
     an unknown name (a fleet payload that has not arrived yet) is
     dropped rather than guessed at index 0. */
  _push(cols, row) {
    const di = this.domainIdx.get(row.domain);
    if (di == null) return false;
    /* lowercased, exactly as viz.data._label lowercases the model's
       labels — otherwise a trace job's "EVAL" and the model's "eval"
       are two different palette entries for one class */
    const cls = String(row.class_name || "unknown").toLowerCase();
    if (!this.classes.includes(cls)) this.classes.push(cls);
    const jid = String(row.job_id);
    if (!this._voted.has(jid)) {
      this._voted.add(jid);
      const bucket = bucketOf(row.tier, row.job_class);
      let votes = this._votes.get(cls);
      if (!votes) this._votes.set(cls, (votes = new Map()));
      votes.set(bucket, (votes.get(bucket) || 0) + 1);
    }
    cols.job_id.push(jid);
    cols.class_name.push(cls);
    cols.tier.push(String(row.tier || ""));
    cols.domain_idx.push(di);
    cols.chips.push(+row.chips || 0);
    cols.t0_us.push(+row.t0_us || 0);
    cols.t1_us.push(+row.t1_us || 0);
    cols.end_reason.push(
      row.end_reason === "open" ? "running_at_horizon" : String(row.end_reason)
    );
    return true;
  }

  addSettled(rows) {
    if (!Array.isArray(rows) || !rows.length) return 0;
    let n = 0;
    for (const row of rows) {
      if (this.nSettled + this.nOpen >= LIVE_MAX_ROWS) {
        this.truncated = true;
        break;
      }
      if (this._push(this._settled, row)) n++;
    }
    if (n) this.version++;
    return n;
  }

  /** Replace the open overlay wholesale (the contract's word). */
  setOpen(rows) {
    if (!Array.isArray(rows)) return;
    this._open = emptyCols();
    for (const row of rows) {
      if (this.nSettled + this.nOpen >= LIVE_MAX_ROWS) {
        this.truncated = true;
        break;
      }
      this._push(this._open, row);
    }
    this.version++;
  }

  /** Settled rows plus the open overlay as ONE columnar block, in the
      shape `model.stints` has — which is what makes the live 3D view the
      same code path as the finished one. */
  columns() {
    const out = emptyCols();
    for (const key of Object.keys(out)) {
      out[key] = this._settled[key].concat(this._open[key]);
    }
    return out;
  }

  /** chips[domain_idx][class] busy at time `t`, as a flat Int32Array of
      width `classes.length`, plus the per-domain total.  One linear scan:
      the live view repaints once per flush, not once per frame. */
  occupancyAt(t) {
    const NC = Math.max(1, this.classes.length);
    const ND = this.domains.length;
    const occ = new Int32Array(ND * NC);
    const total = new Int32Array(ND);
    const clsIdx = new Map(this.classes.map((c, i) => [c, i]));
    for (const cols of [this._settled, this._open]) {
      const n = cols.chips.length;
      for (let i = 0; i < n; i++) {
        if (cols.t0_us[i] > t) continue;
        const open = cols.end_reason[i] === "running_at_horizon";
        if (!open && cols.t1_us[i] <= t) continue;
        const d = cols.domain_idx[i];
        const c = clsIdx.get(cols.class_name[i]) || 0;
        occ[d * NC + c] += cols.chips[i];
        total[d] += cols.chips[i];
      }
    }
    return { occ, total, NC };
  }
}

function emptyCols() {
  return {
    job_id: [], class_name: [], tier: [], domain_idx: [],
    chips: [], t0_us: [], t1_us: [], end_reason: [],
  };
}

/* ------------------------------------------------------------------ *
 * LiveFeed — the polling loop
 * ------------------------------------------------------------------ */
export class LiveFeed {
  /** `handlers`: {onUpdate({stints, progress, status, settled, more}),
      onEnd(status), onError(message)}.  Every handler is optional.
      `onError` fires on a transport failure the feed will retry, so a
      consumer that has not rendered anything yet can say WHY instead of
      showing "waiting for the first flush" forever. */
  constructor(runId, handlers) {
    this.runId = runId;
    this.h = handlers || {};
    this.stints = new LiveStints();
    this.progress = null;
    this.status = null;
    this.cursor = 0;
    this.error = null;
    /** Non-null when the stream reports rows remaining but cannot hand
        any over — see `_tick`.  Consumers show it instead of drawing a
        partial fleet as if it were the whole one. */
    this.stalled = null;
    this._stopped = true;
    this._timer = null;
    this._token = 0;
  }

  start() {
    if (!this._stopped) return;
    this._stopped = false;
    this._token++;
    this._tick(this._token);
  }

  stop() {
    this._stopped = true;
    this._token++;
    if (this._timer !== null) {
      clearTimeout(this._timer);
      this._timer = null;
    }
  }

  _later(token, ms) {
    if (this._stopped || token !== this._token) return;
    this._timer = setTimeout(() => this._tick(token), ms);
  }

  async _tick(token) {
    if (this._stopped || token !== this._token) return;
    this._timer = null;
    let resp = null;
    let doc = null;
    try {
      resp = await fetch(
        "/api/runs/" + encodeURIComponent(this.runId) +
        "/live?cursor=" + this.cursor
      );
      doc = await resp.json();
    } catch (err) {
      if (this._stopped || token !== this._token) return;
      this.error = "cannot reach the server — is fleetsim serve still running?";
      if (this.h.onError) this.h.onError(this.error);
      this._later(token, LIVE_RETRY_MS);
      return;
    }
    if (this._stopped || token !== this._token) return;
    if (!resp.ok || !doc || typeof doc !== "object") {
      /* 404 is terminal (the run is gone); anything else is transient. */
      this.error = (doc && doc.error) || "live stream unavailable (" + resp.status + ")";
      if (resp.status === 404) {
        this.stop();
        if (this.h.onEnd) this.h.onEnd("missing");
        return;
      }
      if (this.h.onError) this.h.onError(this.error);
      this._later(token, LIVE_RETRY_MS);
      return;
    }
    this.error = null;
    if (doc.fleet) this.stints.setFleet(doc.fleet);
    const added = this.stints.addSettled(doc.stints);
    const before = this.cursor;
    this.cursor = typeof doc.cursor === "number" ? doc.cursor : this.cursor;
    if (!doc.more && doc.open_stints) this.stints.setOpen(doc.open_stints);
    if (doc.progress) this.progress = doc.progress;
    this.status = doc.status;
    /* A `more: true` response that moved the cursor NOWHERE cannot be
       caught up with, ever: the server hit a spool row it could not
       parse (`stalled_at`) or the retained-row cap stopped us
       accumulating.  Re-polling at 0 ms here is an unthrottled loop, so
       treat it as an ordinary poll and say what happened. */
    this.stalled =
      doc.more && this.cursor <= before
        ? (typeof doc.stalled_at === "number"
            ? "the live stream stopped at row " + doc.stalled_at +
              " — the spool file is truncated or corrupt, so no further" +
              " stints can be read (the finished run's stints.parquet is" +
              " unaffected)"
            : "the live stream is not advancing — showing " +
              this.stints.nSettled + " settled rows of " + doc.cursor)
        : null;
    if (this.h.onUpdate) {
      this.h.onUpdate({
        stints: this.stints,
        progress: this.progress,
        status: doc.status,
        settled: added,
        more: !!doc.more,
        openTruncated: !!doc.open_truncated,
        openPending: !!doc.open_pending,
        truncated: !!this.stints.truncated,
        stalled: this.stalled,
      });
    }
    if (this.stalled) {
      if (this.h.onError) this.h.onError(this.stalled);
      if (TERMINAL.has(doc.status)) {
        this.stop();
        if (this.h.onEnd) this.h.onEnd(doc.status);
        return;
      }
      this._later(token, LIVE_POLL_MS);
      return;
    }
    if (doc.more) {
      /* rows remain on disk: catch up with no delay, as the contract asks */
      this._later(token, 0);
      return;
    }
    if (TERMINAL.has(doc.status)) {
      this.stop();
      if (this.h.onEnd) this.h.onEnd(doc.status);
      return;
    }
    this._later(token, LIVE_POLL_MS);
  }
}

/* ------------------------------------------------------------------ *
 * palette — the live stream carries class NAMES, not colors (the
 * palette is a property of the finished model).  These are the same
 * hexes fleetsim.viz.data pins, so a live pod and the finished report's
 * pod are the same color.
 * ------------------------------------------------------------------ */
export const CLASS_COLORS = {
  pretrain: "#4c6ef5",
  finetune: "#12b886",
  eval: "#fab005",
  best_effort: "#64748b",
  inference: "#9775fa",
};

export const STATE_COLORS = {
  failed: "#e03131",
  draining: "#f76707",
  maintenance: "#845ef7",
};

/* viz.data._BUCKET_VARIANTS, per bucket and in the pinned order.  Used
   for labels that are not one of the five canonical names (a custom
   `frontier` class, a trace's own labels). */
export const BUCKET_VARIANTS = {
  pretrain: ["#8ba1f9", "#c6d1fc"],
  finetune: ["#89dcc3", "#0c7857"],
  eval: ["#af7b04", "#fee7b4"],
  best_effort: ["#9aa5b4", "#cdd3da"],
  inference: ["#c1acfc", "#6a49b5"],
};

/** Canonical palette order — also the tie-break order for a label seen
    in several buckets (viz.data._label_buckets). */
export const BUCKET_ORDER = Object.keys(CLASS_COLORS);

/** viz.data._JOBCLASS_BUCKET: JobClass enum name -> palette bucket. */
const JOBCLASS_BUCKET = {
  PRETRAIN: "pretrain",
  FINETUNE: "finetune",
  EVAL: "eval",
  INFER_REPLICA: "inference",
};

/** One stint row's palette bucket: tier BEST_EFFORT wins, else the job
    class, else best_effort — viz.data._label_buckets' rule verbatim. */
export function bucketOf(tier, jobClass) {
  if (String(tier || "") === "BEST_EFFORT") return "best_effort";
  return JOBCLASS_BUCKET[String(jobClass || "")] || "best_effort";
}

/** A stable color per observed class label — THE SAME ASSIGNMENT
    fleetsim.viz.data._build_palette makes for the finished run, so a run
    does not change color the moment it completes.

    The rule (both sides): canonical labels keep their pinned hex; a
    non-canonical label takes its BUCKET's canonical color while that
    color is unclaimed (the bucket's own canonical label unobserved),
    else the bucket's pinned shade variants; assignment walks labels in
    SORTED order so it never depends on arrival order.

    Assigning from a flat variant list in first-SEEN order, as v0.8 first
    shipped, made a live `gangs: {class: finetune}` periwinkle (a PRETRAIN
    shade) that turned finetune green on completion — the same work in
    two different palette families in one sitting.

    `buckets` maps label -> bucket (LiveStints.buckets()).  A bare label
    list is still accepted; without bucket evidence every non-canonical
    label falls in best_effort, which is what the model does for a job
    whose class it cannot place either. */
export function livePalette(buckets) {
  const map = Array.isArray(buckets)
    ? Object.fromEntries(buckets.map((l) => [l, "best_effort"]))
    : (buckets || {});
  const out = Object.assign({}, CLASS_COLORS);
  const taken = {};
  for (const b of BUCKET_ORDER) taken[b] = Object.hasOwn(map, b);
  const extra = {};
  for (const label of Object.keys(map).sort()) {
    if (Object.hasOwn(CLASS_COLORS, label) || Object.hasOwn(STATE_COLORS, label)) continue;
    const bucket = Object.hasOwn(BUCKET_VARIANTS, map[label])
      ? map[label]
      : "best_effort";
    if (!taken[bucket]) {
      out[label] = CLASS_COLORS[bucket];
      taken[bucket] = true;
      continue;
    }
    const i = extra[bucket] || 0;
    extra[bucket] = i + 1;
    const variants = BUCKET_VARIANTS[bucket];
    out[label] = variants[i % variants.length];
  }
  for (const [k, hex] of Object.entries(STATE_COLORS)) out[k] = hex;
  return out;
}
