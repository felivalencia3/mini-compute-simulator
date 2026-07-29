"""Render a viz model into the single-file replay HTML (v0.3).

:func:`render_html` injects :func:`fleetsim.viz.to_json`'s output into
the packaged ``template.html`` at the ``/*__FLEETSIM_DATA__*/`` token
(as ``const DATA = {...};``) and sets the document ``<title>``.  The
result is ONE self-contained page: inline CSS/JS only, no external
requests of any kind, so it renders identically from ``file:`` and
plain HTTP.

The injected JSON is made ``<script>``-safe: every ``<`` becomes the
``\\u003c`` string escape — a no-op under JS evaluation (in valid JSON
``<`` can only occur inside string values) that leaves the payload with
no ``</script``, ``<script`` or ``<!--`` sequence, so hostile job ids
or labels can never end the script block early.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .data import to_json

__all__ = ["render_html"]

_TEMPLATE_PATH = Path(__file__).with_name("template.html")
_DATA_TOKEN = "/*__FLEETSIM_DATA__*/"
_TITLE_TOKEN = "__FLEETSIM_TITLE__"


def _script_safe(payload: str) -> str:
    """Escape every ``<`` as ``\\u003c``.  In valid JSON a ``<`` can
    only appear inside a string value, where the escape is equivalent —
    and with no literal ``<`` left, none of the sequences the HTML spec
    treats specially inside a ``<script>`` block can occur."""
    return payload.replace("<", "\\u003c")


def render_html(model: dict[str, Any]) -> str:
    """The self-contained replay page for a ``build_viz_model`` model.

    Pure function of the model: identical models render byte-identical
    HTML (the template is static and the injection deterministic).
    Raises ``ValueError`` (via :func:`to_json`) if the model carries a
    non-finite float, and ``RuntimeError`` if the packaged template has
    lost its injection token.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if _DATA_TOKEN not in template:
        raise RuntimeError(
            f"template.html is missing the {_DATA_TOKEN!r} injection token"
        )
    meta = model.get("meta") or {}
    title = str(meta.get("title") or "fleetsim replay")
    payload = _script_safe(to_json(model))
    out = template.replace(_DATA_TOKEN, f"const DATA = {payload};", 1)
    return out.replace(_TITLE_TOKEN, html.escape(title, quote=True))
