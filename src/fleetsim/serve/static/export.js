/* fleetsim v0.8 — export helpers: PNG snapshots and copy-to-clipboard.

   Everything here is local and dependency-free.  An SVG panel becomes a
   PNG by serializing the live node, drawing the serialized markup into a
   canvas through a data: URI (allowed by the shell CSP's `img-src 'self'
   data:`), and handing the canvas to toBlob.  A WebGL frame skips the
   serialize step and goes straight to toBlob.

   No network, no library, no build step: an <a download> pointed at an
   object URL is the whole download path, and the URL is revoked on the
   next tick so nothing leaks.

   Why the SVG clone carries explicit styling: a serialized SVG is
   rendered by the browser in ISOLATION — the document's stylesheet does
   not travel with it.  Our charts already paint through presentation
   attributes (fill/stroke set per element), so the only things that must
   be re-stated are the font stack and the page background, which is what
   `decorate` does.  A chart that relied on a CSS class for color would
   export unstyled, so don't add one. */

"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

/* The app's surfaces, restated here because the export leaves the DOM
   (and therefore app.css) behind. */
export const EXPORT_BG = "#0b0e14";
export const EXPORT_PANEL = "#11151d";
const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

/* One shared <a> would be simpler but a click on a detached element is
   ignored in some engines; a fresh one, appended and removed, always
   works. */
function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.position = "fixed";
  a.style.opacity = "0";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** A filesystem-safe stem: the app builds names from run ids and panel
    titles, both of which are user data. */
export function safeName(text, fallback) {
  const stem = String(text == null ? "" : text)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return stem || fallback || "fleetsim";
}

/** PNG from a canvas (the 3D frame).  The renderer must keep its drawing
    buffer (`preserveDrawingBuffer`) or the read comes back blank —
    fleet3d.js sets it for exactly this reason.  Resolves false when the
    browser refuses (no toBlob, or a tainted canvas). */
export function canvasToPng(canvas, filename) {
  return new Promise((resolve) => {
    if (!canvas || typeof canvas.toBlob !== "function") {
      resolve(false);
      return;
    }
    try {
      canvas.toBlob((blob) => {
        if (!blob) {
          resolve(false);
          return;
        }
        saveBlob(blob, filename);
        resolve(true);
      }, "image/png");
    } catch (err) {
      resolve(false);
    }
  });
}

/* btoa() throws on any code point above U+00FF, and panel titles carry
   real text (·, ×, —).  Encode UTF-8 by hand rather than hoping. */
function utf8Base64(text) {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function decorate(clone, width, height, background) {
  clone.setAttribute("xmlns", SVG_NS);
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.setAttribute("font-family", MONO);
  clone.removeAttribute("tabindex");
  clone.removeAttribute("class");
  if (background) {
    const bg = document.createElementNS(SVG_NS, "rect");
    bg.setAttribute("x", "0");
    bg.setAttribute("y", "0");
    bg.setAttribute("width", "100%");
    bg.setAttribute("height", "100%");
    bg.setAttribute("fill", background);
    clone.insertBefore(bg, clone.firstChild);
  }
  /* Interaction affordances are state, not content: a crosshair frozen
     wherever the pointer happened to be would be baked into the file. */
  for (const el of clone.querySelectorAll('[visibility="hidden"]')) {
    el.parentNode.removeChild(el);
  }
}

/** PNG from a live <svg> node (any 2D panel).  `scale` renders at that
    multiple of the viewBox for a crisp file; the source node is never
    touched.  Resolves false if the browser could not rasterize. */
export function svgToPng(svg, filename, opts) {
  const o = opts || {};
  const scale = o.scale || 2;
  const box = (svg.getAttribute("viewBox") || "").trim().split(/\s+/);
  const vbW = box.length === 4 ? parseFloat(box[2]) : 0;
  const vbH = box.length === 4 ? parseFloat(box[3]) : 0;
  const rect = svg.getBoundingClientRect();
  const width = vbW > 0 ? vbW : Math.max(1, Math.round(rect.width));
  const height = vbH > 0 ? vbH : Math.max(1, Math.round(rect.height));

  const clone = svg.cloneNode(true);
  decorate(clone, width, height, o.background === undefined ? EXPORT_BG : o.background);
  const markup = new XMLSerializer().serializeToString(clone);

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const g = canvas.getContext("2d");
      g.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvasToPng(canvas, filename).then(resolve);
    };
    img.onerror = () => resolve(false);
    try {
      img.src = "data:image/svg+xml;base64," + utf8Base64(markup);
    } catch (err) {
      resolve(false);
    }
  });
}

/** A "PNG" button wired to one chart's <svg>.  Returns the button so the
    caller decides where it sits; the label reports failure in place
    instead of failing silently. */
export function pngButton(svg, filename, opts) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "minibtn pngbtn";
  btn.textContent = "PNG";
  btn.title = "Download this panel as a PNG image";
  btn.addEventListener("click", () => {
    const was = btn.textContent;
    btn.disabled = true;
    svgToPng(svg, filename, opts).then((ok) => {
      btn.disabled = false;
      btn.textContent = ok ? was : "no PNG";
      if (!ok) setTimeout(() => (btn.textContent = was), 2500);
    });
  });
  return btn;
}

/** Copy text to the clipboard.  navigator.clipboard needs a secure
    context, and http://127.0.0.1 IS one — but a non-loopback --host over
    plain HTTP is not, so the execCommand path stays as the fallback. */
export function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text).then(
      () => true,
      () => legacyCopy(text)
    );
  }
  return Promise.resolve(legacyCopy(text));
}

function legacyCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "-1000px";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (err) {
    ok = false;
  }
  document.body.removeChild(ta);
  return ok;
}

/** A "Copy link" button whose href is computed at click time (the deep
    link encodes live view state, so it must not be captured early). */
export function copyLinkButton(getUrl, label) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn";
  btn.textContent = label || "Copy link";
  btn.title = "Copy a link that reopens this exact moment and camera angle";
  btn.addEventListener("click", () => {
    const url = getUrl();
    copyText(url).then((ok) => {
      btn.textContent = ok ? "Copied ✓" : "Copy failed";
      setTimeout(() => (btn.textContent = label || "Copy link"), 1800);
    });
  });
  return btn;
}
