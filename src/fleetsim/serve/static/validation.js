/* fleetsim v0.8 — the validation tab (#validation).

   Renders `GET /api/validation` — which IS
   `fleetsim.validation.results.payload()`, the same module the
   validation tests import.  Nothing on this page is typed in by hand:
   a number shown here and a number asserted in CI cannot disagree,
   because there is only one copy.

   HONESTY RULES the rendering obeys (they are the point of the page):

     - `fleetsim: null` means HONESTLY UNMEASURED (an opt-in rung that
       was not run) and renders as "not measured" — never as agreement.
     - `in_band: null` means unmeasured OR a reported-only row: rows
       that carry neither a band nor a tolerance are REPORTED, never
       asserted, and are labelled "reported".
     - the documented ANTI-GOALS — the quantities fleetsim deliberately
       does not reproduce — sit at the TOP of the page, not in a
       footnote, because a validation page that only lists successes is
       marketing.
     - `previous` is last release's measurement and is shown as movement
       (6.87 ← 8.75), never as a second claim.

   Semantic colors only (good / bad / muted).  No workload-class color
   appears here — a validation row is not a class. */

"use strict";

const $ = (s) => document.querySelector(s);

let wired = false;
let cache = null;
let rungFilter = "all";

function eln(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

const fmtNum = (v, unit) => {
  if (v == null || !isFinite(v)) return "–";
  if (unit === "share") return (v * 100).toFixed(1) + "%";
  if (unit === "count") return String(Math.round(v));
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(3);
};

/** A tolerance in the ROW'S OWN unit.

    A `share` row's tolerance is an ABSOLUTE band in percentage POINTS
    (PHILLY_BY_COUNT_TOL = 0.05 is "+/- 5 pp", results.py §6), and
    printing it as "±5%" beside a value rendered "69.3%" reads as ±3.5 pp
    — a band 30 % tighter than the one CI asserts. On the one page whose
    job is not overstating what is asserted, that unit has to be right. */
function fmtTolerance(tolerance, unit) {
  if (unit === "share") return "±" + (tolerance * 100).toFixed(0) + " pp";
  if (unit === "count") return "±" + String(Math.round(tolerance));
  if (unit === "s") return "±" + fmtNum(tolerance, unit) + " s";
  return "±" + fmtNum(tolerance, unit);
}

/* No unmount counterpart, deliberately: unlike compare.js and
   experiment.js this view polls nothing, holds no timers and wires no
   listener outside its own mount, so leaving the route costs nothing.
   The payload is cached because it cannot change while the server runs
   — it is a module-level constant table, not a measurement taken now. */
export async function mountValidation() {
  const mount = $("#validationBody");
  if (!mount) return;
  wireOnce();
  if (cache) {
    render(cache);
    return;
  }
  mount.textContent = "";
  mount.appendChild(eln("p", "sub", "loading the validation suite's results…"));
  let doc = null;
  try {
    const resp = await fetch("/api/validation");
    doc = resp.ok ? await resp.json() : null;
    if (!doc) {
      mount.textContent = "";
      mount.appendChild(
        eln("p", "railnote err", "the validation endpoint returned " + resp.status)
      );
      return;
    }
  } catch (err) {
    mount.textContent = "";
    mount.appendChild(
      eln("p", "railnote err", "cannot reach the server — is fleetsim serve still running?")
    );
    return;
  }
  cache = doc;
  render(doc);
}

function wireOnce() {
  if (wired) return;
  wired = true;
  const sel = $("#valRung");
  if (sel) {
    sel.addEventListener("change", () => {
      rungFilter = sel.value;
      if (cache) render(cache);
    });
  }
}

/** Fill the rung filter from the payload.  The options are DATA, not
    markup: hand-written labels shipped with V2 and V3 swapped (V2 is the
    absolute Table-3 rung, V3 the job-status distribution) and offered a
    V1p rung that no measurement row carries, so selecting it emptied the
    page. Both failures are unrepresentable when the list is built from
    the rungs that actually have rows. */
function fillRungs(doc) {
  const sel = $("#valRung");
  if (!sel || !Array.isArray(doc.rungs)) return;
  const want = ["all"].concat(doc.rungs.map((r) => r.rung));
  const have = [...sel.options].map((o) => o.value);
  if (have.join("|") === want.join("|")) return;
  const current = sel.value;
  sel.textContent = "";
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = "all rungs";
  sel.appendChild(all);
  for (const r of doc.rungs) {
    const opt = document.createElement("option");
    opt.value = r.rung;
    opt.textContent = r.label;
    sel.appendChild(opt);
  }
  sel.value = want.includes(current) ? current : "all";
  rungFilter = sel.value;
}

function render(doc) {
  const mount = $("#validationBody");
  fillRungs(doc);
  mount.textContent = "";
  $("#validationMeta").textContent =
    doc.version + " · " + doc.counts.measured + " of " + doc.counts.total +
    " measured · " + doc.counts.in_band + " of " + doc.counts.asserted +
    " asserted rows in band";

  mount.appendChild(headlinePanel(doc));
  mount.appendChild(antiGoalPanel(doc));
  for (const group of doc.groups) {
    if (rungFilter !== "all" && group.rung !== rungFilter) continue;
    mount.appendChild(groupPanel(doc, group));
  }
  mount.appendChild(ladderPanel(doc));
  mount.appendChild(placerPanel(doc));
  mount.appendChild(citationPanel(doc));
}

/* ---- panels -------------------------------------------------------- */

function panel(title, sub) {
  const p = eln("div", "panel");
  const head = eln("div", "phead");
  head.appendChild(eln("h2", null, title));
  if (sub) head.appendChild(eln("span", "sub", sub));
  p.appendChild(head);
  return p;
}

function headlinePanel(doc) {
  const p = panel("What is validated", doc.version);
  p.appendChild(eln("p", "vheadline", doc.headline));
  const stats = eln("div", "statrow");
  const cells = [
    ["results", doc.counts.total, "published quantities tracked"],
    ["measured", doc.counts.measured, "fleetsim has a number for"],
    ["asserted", doc.counts.asserted, "CI fails if these move"],
    ["in band", doc.counts.in_band, "of the asserted rows"],
  ];
  for (const [label, value, sub] of cells) {
    const s = eln("div", "stat");
    s.appendChild(eln("span", "clabel", label));
    s.appendChild(eln("span", "cval mono", String(value)));
    s.appendChild(eln("span", "sub", sub));
    stats.appendChild(s);
  }
  p.appendChild(stats);
  const note = eln("p", "sub vsrc");
  note.appendChild(
    document.createTextNode(
      "Every number on this page is read from GET /api/validation, which is " +
      "fleetsim.validation.results.payload() — the same module the validation " +
      "tests import. The long form is "
    )
  );
  const code = eln("span", "kbd", doc.doc);
  note.appendChild(code);
  note.appendChild(document.createTextNode(" in the repository."));
  p.appendChild(note);
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

function antiGoalPanel(doc) {
  const p = panel(
    "What fleetsim does NOT reproduce",
    "documented anti-goals — read these before quoting anything above"
  );
  p.classList.add("antipanel");
  const wrap = tableWrap("anti-goals table");
  const table = eln("table", "matrix");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["published quantity", "why fleetsim cannot answer it", "disposition"]) {
    hr.appendChild(eln("th", null, h));
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const row of doc.anti_goals) {
    const tr = document.createElement("tr");
    tr.appendChild(eln("td", "antiq", row.quantity));
    tr.appendChild(eln("td", null, row.why));
    tr.appendChild(eln("td", "mono antidisp", row.disposition));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  p.appendChild(wrap);
  return p;
}

function verdict(row) {
  if (row.fleetsim == null) {
    return { cls: "vmuted", mark: "—", label: "not measured" };
  }
  if (row.in_band === true) return { cls: "vgood", mark: "✓", label: "in band" };
  if (row.in_band === false) return { cls: "vbad", mark: "✕", label: "OUT OF BAND" };
  return { cls: "vmuted", mark: "·", label: "reported, not asserted" };
}

function groupPanel(doc, group) {
  const rows = doc.results.filter((r) => group.ids.includes(r.id));
  const p = panel(
    group.group,
    group.rung + " · " + group.trace + " · " + group.unit + " · " + group.doc_ref
  );
  const wrap = tableWrap("measurement table");
  const table = eln("table", "matrix vtable");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  const heads = [
    ["subject", ""],
    ["published", "num"],
    ["fleetsim", "num"],
    ["rel. error", "num"],
    ["tolerance / band", ""],
    ["verdict", ""],
  ];
  for (const [h, cls] of heads) hr.appendChild(eln("th", cls || null, h));
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const v = verdict(row);
    const tr = document.createElement("tr");
    tr.appendChild(eln("td", null, row.subject));
    tr.appendChild(eln("td", "num mono", fmtNum(row.published, row.unit)));

    const meas = eln("td", "num mono");
    if (row.fleetsim == null) {
      meas.appendChild(eln("span", "vmuted", "not measured"));
      meas.title = "fleetsim has no measurement for this row (opt-in rung)";
    } else {
      meas.appendChild(document.createTextNode(fmtNum(row.fleetsim, row.unit)));
      if (row.previous != null && row.previous !== row.fleetsim) {
        const prev = eln("span", "vprev", " ← " + fmtNum(row.previous, row.unit));
        prev.title = "previous release's measurement (movement, not a second claim)";
        meas.appendChild(prev);
      }
    }
    tr.appendChild(meas);

    tr.appendChild(
      eln(
        "td",
        "num mono",
        row.rel_error == null ? "–" : (row.rel_error * 100).toFixed(1) + "%"
      )
    );

    let boundText = "reported only";
    if (row.band) boundText = "band [" + row.band[0] + ", " + row.band[1] + "]";
    else if (row.tolerance != null) boundText = fmtTolerance(row.tolerance, row.unit);
    const bt = eln("td", "mono vbound", boundText);
    if (!row.band && row.tolerance == null) {
      bt.title = "this row carries neither a band nor a tolerance: it is reported, never asserted";
    }
    tr.appendChild(bt);

    const vd = eln("td", "vverdict " + v.cls);
    vd.appendChild(eln("span", "vmark", v.mark));
    vd.appendChild(eln("span", null, v.label));
    vd.title = row.id + " · " + row.doc_ref;
    tr.appendChild(vd);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  p.appendChild(wrap);
  /* The group's own caveat, next to the numbers it is about.  A caveat
     that lives only in the headline is one a reader who scrolled to the
     table never sees — and these tables are exactly where a -0.0 % cell
     invites a reading the caveat refuses. */
  if (group.caption) p.appendChild(eln("p", "sub vcaption", group.caption));
  return p;
}

function ladderPanel(doc) {
  const p = panel("The validation ladder", "what each rung asserts, and where");
  const wrap = tableWrap("validation ladder");
  const table = eln("table", "matrix");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["rung", "validation", "kind", "shipped as", "in CI", "full run"]) {
    hr.appendChild(eln("th", null, h));
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const row of doc.ladder) {
    const tr = document.createElement("tr");
    tr.appendChild(eln("td", "mono", row.rung));
    tr.appendChild(eln("td", null, row.validation));
    tr.appendChild(eln("td", "sub", row.kind));
    tr.appendChild(eln("td", "mono vpath", row.shipped_as));
    tr.appendChild(eln("td", "sub", row.ci));
    tr.appendChild(eln("td", "mono sub", row.full));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  p.appendChild(wrap);
  return p;
}

function placerPanel(doc) {
  const p = panel(
    "Placement policy sweep",
    "mean absolute ratio error against the published table, per placer"
  );
  const wrap = tableWrap("placement sweep");
  const table = eln("table", "matrix");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["placer", "shipped default", "mean |ratio error|"]) {
    hr.appendChild(eln("th", null, h));
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const row of doc.placer_sweep) {
    const tr = document.createElement("tr");
    tr.appendChild(eln("td", "mono", row.placer));
    tr.appendChild(eln("td", null, row.shipped ? "yes" : "no"));
    tr.appendChild(
      eln("td", "num mono", (row.mean_abs_ratio_error * 100).toFixed(1) + "%")
    );
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  p.appendChild(wrap);
  return p;
}

function citationPanel(doc) {
  const p = panel("Traces and citations", "the published sources these rows compare against");
  for (const [name, cit] of Object.entries(doc.citations)) {
    const box = eln("div", "vcite");
    box.appendChild(eln("div", "vcitename mono", name));
    box.appendChild(eln("div", null, cit.citation));
    const meta = eln("div", "sub mono");
    meta.textContent = "license " + cit.license + " · " + cit.source;
    box.appendChild(meta);
    p.appendChild(box);
  }
  return p;
}
