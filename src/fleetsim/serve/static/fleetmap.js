/* fleetsim v0.8 — the compact 2D fleet block diagram.

   One drawing routine, two callers:

     - the scenario editor's SHAPE PREVIEW: the fleet a YAML document
       describes (`POST /api/validate` -> `fleet`), drawn before anything
       runs — capacity only, no occupancy;
     - the run view's LIVE FLEET MAP: the same blocks, filled bottom-up
       with the classes running on each domain right now, fed by
       `GET /api/runs/{id}/live`.

   One block per domain at the map level (the level `outputs: {stints:
   true}` records), grouped into a row of blocks per cluster.  Blocks are
   uniform size and read as capacity; the FILL is the only thing that
   means occupancy, so an empty preview and a fully idle fleet look the
   same on purpose — both are "nothing running".

   SVG is built with createElementNS only (no markup interpolation), and
   every color is a presentation attribute rather than a CSS class, which
   is what lets export.js turn any of these into a PNG that still looks
   like the panel. */

"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

const VIEW_W = 920;
const PAD = 10;
const GAP = 3;
const HEAD_H = 17;      /* per-cluster header line */
const BLOCK_GAP = 12;   /* between clusters */

const OUTLINE = "rgba(255,255,255,.14)";
const CELL_BG = "#161b26";     /* idle capacity — the 3D view's idle slab */
const TEXT = "#e6e8eb";
const MUTED = "#8b93a1";

function el(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, String(attrs[k]));
  return n;
}

function text(x, y, str, attrs) {
  const t = el("text", Object.assign({ x, y, fill: MUTED, "font-size": "10.5" }, attrs || {}));
  t.textContent = str;
  return t;
}

/** Cell edge for a given block count: big enough to label while small
    enough that a 256-domain fleet still fits one screen. */
function cellSize(n) {
  if (n <= 48) return 34;
  if (n <= 120) return 24;
  if (n <= 320) return 15;
  return 9;
}

/**
 * Draw the diagram into `mount` (its contents are replaced) and return
 * the <svg> element, or null when there is nothing to draw.
 *
 * spec = {
 *   clusters: [{label, sub, cells: [{short, cap, segs: [{color, chips,
 *              label}], title}]}],
 *   legend:   [{color, label}],
 *   note:     string | null,
 * }
 */
export function renderFleetMap(mount, spec) {
  mount.textContent = "";
  const clusters = (spec && spec.clusters) || [];
  const nCells = clusters.reduce((a, c) => a + c.cells.length, 0);
  if (!clusters.length) return null;

  const S = cellSize(nCells);
  const pitch = S + GAP;
  const cols = Math.max(1, Math.floor((VIEW_W - 2 * PAD) / pitch));

  /* pass 1: height */
  let y = PAD;
  const rows = [];
  for (const cl of clusters) {
    const r = Math.max(1, Math.ceil(cl.cells.length / cols));
    rows.push({ y0: y + HEAD_H, r });
    y += HEAD_H + r * pitch + BLOCK_GAP;
  }
  const H = Math.max(40, y - BLOCK_GAP + PAD);

  const svg = el("svg", {
    viewBox: "0 0 " + VIEW_W + " " + H,
    class: "fmapsvg",
    role: "img",
    preserveAspectRatio: "xMinYMin meet",
    "aria-label": spec.ariaLabel || "fleet block diagram",
  });

  clusters.forEach((cl, ci) => {
    const band = rows[ci];
    const head = text(PAD, band.y0 - 5, cl.label, { fill: TEXT, "font-size": "11.5" });
    head.setAttribute("font-weight", "650");
    svg.appendChild(head);
    if (cl.sub) {
      const w = cl.label.length * 6.6 + 14;
      svg.appendChild(text(PAD + w, band.y0 - 5, cl.sub));
    }
    cl.cells.forEach((cell, i) => {
      const cx = PAD + (i % cols) * pitch;
      const cy = band.y0 + Math.floor(i / cols) * pitch;
      const g = el("g", {});
      g.appendChild(
        el("rect", {
          x: cx, y: cy, width: S, height: S, rx: Math.min(4, S / 5),
          fill: CELL_BG, stroke: OUTLINE, "stroke-width": "1",
        })
      );
      /* fill bottom-up, one segment per class; a 2px surface gap between
         segments keeps two adjacent classes from reading as one band */
      const cap = Math.max(1, cell.cap || 1);
      let acc = 0;
      for (const seg of cell.segs || []) {
        const chips = Math.min(seg.chips, cap - acc);
        const h = (chips / cap) * (S - 2);
        if (h <= 0) continue;
        g.appendChild(
          el("rect", {
            x: cx + 1,
            y: cy + S - 1 - acc / cap * (S - 2) - h,
            width: S - 2,
            height: Math.max(1, h - (acc > 0 ? 1 : 0)),
            fill: seg.color,
          })
        );
        acc += chips;
      }
      if (S >= 24 && cell.short) {
        const lab = text(cx + S / 2, cy + S / 2 + 3.5, cell.short, {
          "text-anchor": "middle",
          fill: acc > 0 ? "rgba(11,14,20,.85)" : MUTED,
          "font-size": S >= 30 ? "9.5" : "8",
        });
        g.appendChild(lab);
      }
      if (cell.title) {
        const t = el("title", {});
        t.textContent = cell.title;
        g.appendChild(t);
      }
      svg.appendChild(g);
    });
  });

  mount.appendChild(svg);

  if (spec.legend && spec.legend.length) {
    const leg = document.createElement("div");
    leg.className = "fmapleg";
    for (const k of spec.legend) {
      const chip = document.createElement("span");
      chip.className = "f3dkey";
      const i = document.createElement("i");
      i.style.background = k.color;
      chip.appendChild(i);
      chip.appendChild(document.createTextNode(k.label));
      leg.appendChild(chip);
    }
    mount.appendChild(leg);
  }
  if (spec.note) {
    const p = document.createElement("p");
    p.className = "sub fmapnote";
    p.textContent = spec.note;
    mount.appendChild(p);
  }
  return svg;
}

/* ------------------------------------------------------------------ *
 * spec builders
 * ------------------------------------------------------------------ */

const fmtInt = (n) => (n == null ? "–" : n.toLocaleString());

/** Editor preview: `POST /api/preview` -> `fleet` (capacity only).
 *
 * The blocks are the level the RUN WILL RECORD (`shape.stints_mode`),
 * not always the level below the cluster root: a scenario naming a level
 * — example 07 ships `outputs: {stints: node}` — previewed 4 rack blocks
 * for a run that records 32 node domains, under a caption asserting the
 * opposite.  With stints off there is no recording level at all, and the
 * caption says the fleet map will be empty rather than naming one.
 */
export function shapeSpec(shape) {
  const clusters = (shape.clusters || []).map((c) => ({
    label: c.id,
    sub: [
      fmtInt(c.chips) + " chips",
      fmtInt(c.nodes) + " nodes",
      c.chips_per_node ? c.chips_per_node + "/node" : null,
      c.chip_type,
      c.n_domains + " × " + (c.map_level || "domain"),
      c.levels && c.levels.length ? c.levels.join(" › ") : null,
    ].filter(Boolean).join(" · "),
    cells: c.domains.map((d) => ({
      short: d.short,
      cap: d.chips,
      segs: [],
      title:
        c.id + "/" + (d.path || d.short) + " — " + fmtInt(d.chips) +
        " chips, " + fmtInt(d.nodes) + " nodes",
    })),
  }));
  const notes = [];
  if (shape.clusters_truncated) {
    notes.push(
      "showing the first " + clusters.length + " of " + shape.n_clusters +
      " clusters"
    );
  }
  for (const c of shape.clusters || []) {
    if (c.domains_truncated) {
      notes.push(
        c.id + ": showing " + c.domains.length + " of " +
        c.n_domains.toLocaleString() + " " +
        (c.map_level || "domain") + " blocks"
      );
    }
  }
  const level = (shape.clusters[0] && shape.clusters[0].map_level) || "domain";
  if (shape.stints_mode === "off") {
    notes.push(
      "one block per " + level + " — but this scenario sets no" +
      " outputs.stints, so it records no stints: the live fleet map and" +
      " the report's fleet map will both be empty. Add  outputs:" +
      " {stints: true}  (or a level name) to record who ran where"
    );
  } else if (shape.stints_mode === "level") {
    notes.push(
      "one block per " + level + " — the level  outputs: {stints: " +
      shape.stints_level + "}  records"
    );
  } else {
    notes.push(
      "one block per " + level + " — the level directly below each" +
      " cluster root, which is what  outputs: {stints: true}  records"
    );
  }
  return {
    clusters,
    legend: null,
    ariaLabel:
      "fleet shape: " + shape.n_clusters + " clusters, " +
      fmtInt(shape.total_chips) + " chips, " + fmtInt(shape.total_nodes) + " nodes",
    note: notes.join(" · "),
  };
}

/** Live map: the live stream's fleet geometry + an occupancy snapshot. */
export function liveSpec(stints, t, palette) {
  const fleet = stints.fleet;
  if (!fleet || !fleet.clusters || !fleet.clusters.length) return null;
  const { occ, total, NC } = stints.occupancyAt(t);
  const labels = stints.classes;
  let flat = 0;
  const clusters = fleet.clusters.map((c) => {
    let used = 0;
    let capacity = 0;
    const cells = c.domains.map((d) => {
      const idx = flat++;
      const cap = Math.max(1, d.chips || 1);
      capacity += cap;
      used += total[idx] || 0;
      const segs = [];
      const parts = [];
      for (let k = 0; k < NC; k++) {
        const n = occ[idx * NC + k];
        if (n > 0) {
          segs.push({ color: palette[labels[k]] || MUTED, chips: n });
          parts.push(labels[k] + " " + n);
        }
      }
      return {
        short: d.short,
        cap,
        segs,
        title:
          d.id + " — " + (total[idx] || 0) + " / " + cap + " chips busy" +
          (parts.length ? " · " + parts.join(", ") : " · idle"),
      };
    });
    return {
      label: c.id,
      sub:
        fmtInt(used) + " / " + fmtInt(capacity) + " chips busy · " +
        ((used / Math.max(1, capacity)) * 100).toFixed(1) + "%",
      cells,
    };
  });
  return {
    clusters,
    legend: labels.map((l) => ({ color: palette[l] || MUTED, label: l })),
    ariaLabel:
      "live fleet map: " + clusters.length + " clusters, " +
      fleet.domains.length + " " + (fleet.map_level || "domain") + " blocks",
    note: null,
  };
}
