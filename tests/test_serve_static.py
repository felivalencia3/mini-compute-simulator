"""Structural tests for the `fleetsim serve` app shell (v0.5).

The shell is plain static files (no build step), so these tests pin the
properties the strict CSP and the local-first posture depend on:

- every static asset is self-contained — no external URL anywhere (the
  CSP would block the fetch; a stray CDN reference is a bug either way);
- ``index.html`` parses, its tags balance, and it loads JS only as
  same-origin module files (inline ``<script>`` would be CSP-blocked);
- every ``/static/...`` asset the shell references actually exists;
- the ``GET /api/examples`` endpoint serves the bundled starter
  scenarios read-only, sorted, with the standard security headers.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pytest

from fleetsim.serve.runs import RunManager
from fleetsim.serve.server import (
    CSP_APP,
    FleetsimHTTPServer,
    list_examples,
)

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "fleetsim" / "serve" / "static"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

TEXT_SUFFIXES = {".html", ".css", ".js", ".mjs", ".json", ".svg", ".txt", ".md"}

#: XML namespace identifiers are opaque strings, not network requests;
#: they are the only sanctioned `http://` occurrences in static assets.
_ALLOWED_URL_PREFIXES = ("http://www.w3.org/",)

_VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta source track wbr".split()
)


def static_text_files() -> list[Path]:
    files = [
        p
        for p in sorted(STATIC_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    ]
    assert files, f"no static assets found under {STATIC_DIR}"
    return files


# ---------------------------------------------------------------------------
# self-containment: no external URLs in any static asset
# ---------------------------------------------------------------------------


def test_static_assets_have_no_external_urls():
    url_re = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s\"'<>)]*")
    for path in static_text_files():
        if path.parent.name == "vendor":
            # The vendored three.js build legitimately contains inert URL
            # strings (shader doc comments, regex literals).  Those files
            # are pinned BYTE-EXACTLY by sha256 below — a stronger
            # self-containment guarantee than this string scan.
            continue
        text = path.read_text(encoding="utf-8")
        for match in url_re.finditer(text):
            url = match.group(0)
            assert url.startswith(_ALLOWED_URL_PREFIXES), (
                f"{path.name}: external URL {url!r} — static assets must be"
                f" self-contained (the CSP blocks external fetches anyway)"
            )
        assert "integrity=" not in text, path.name
        assert "@import" not in text, path.name  # css: no external sheets


def test_no_inline_event_handlers_or_javascript_urls():
    # belt for the strict CSP: onclick= etc. would silently do nothing
    handler_re = re.compile(r"\son[a-z]+\s*=", re.IGNORECASE)
    for path in static_text_files():
        if path.suffix.lower() != ".html":
            continue
        text = path.read_text(encoding="utf-8")
        assert not handler_re.search(text), f"{path.name}: inline event handler"
        assert "javascript:" not in text.lower(), path.name


# ---------------------------------------------------------------------------
# index.html structure
# ---------------------------------------------------------------------------


class ShellParser(HTMLParser):
    """Collects tag balance, script/link references, and element ids."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.scripts: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []
        self.ids: set[str] = set()
        self._in_script = False
        self._script_data: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs and attrs["id"]:
            if attrs["id"] in self.ids:
                self.errors.append(f"duplicate id: {attrs['id']}")
            self.ids.add(attrs["id"])
        if tag == "script":
            self.scripts.append(attrs)
            self._in_script = True
            self._script_data = []
        if tag == "link":
            self.links.append(attrs)
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.stack.pop()
        if tag == "script":
            self._in_script = False

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
            if "".join(self._script_data).strip():
                self.errors.append("inline script body (CSP script-src 'self' blocks it)")
        if tag in _VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"stray closing tag: </{tag}>")
            return
        if self.stack[-1] != tag:
            self.errors.append(
                f"mis-nested: </{tag}> while <{self.stack[-1]}> is open"
            )
        else:
            self.stack.pop()

    def handle_data(self, data):
        if self._in_script:
            self._script_data.append(data)


def parse_shell(path: Path) -> ShellParser:
    parser = ShellParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    assert not parser.errors, f"{path.name}: {parser.errors}"
    assert not parser.stack, f"{path.name}: unclosed tags {parser.stack}"
    return parser


def test_all_static_html_pages_parse():
    pages = [p for p in static_text_files() if p.suffix.lower() == ".html"]
    assert pages
    for page in pages:
        parse_shell(page)


def test_index_loads_js_as_same_origin_modules_only():
    parser = parse_shell(STATIC_DIR / "index.html")
    assert parser.scripts, "index.html loads no script"
    for attrs in parser.scripts:
        src = attrs.get("src")
        assert src, "inline <script> would be blocked by script-src 'self'"
        assert src.startswith("/static/"), src
        assert attrs.get("type") == "module", src


def test_index_references_only_existing_static_assets():
    parser = parse_shell(STATIC_DIR / "index.html")
    refs = [a.get("src") for a in parser.scripts] + [
        a.get("href") for a in parser.links
    ]
    static_refs = [r for r in refs if r and r.startswith("/static/")]
    assert static_refs, "no static references found"
    for ref in static_refs:
        target = STATIC_DIR / ref.removeprefix("/static/")
        assert target.is_file(), f"index.html references missing asset {ref}"
    # the favicon must be inline (data:) — a /favicon.ico request would 404
    icons = [a for a in parser.links if "icon" in (a.get("rel") or "")]
    assert icons and all((a.get("href") or "").startswith("data:") for a in icons)


def test_index_has_the_app_shell_anatomy():
    parser = parse_shell(STATIC_DIR / "index.html")
    required = {
        # rail
        "rail", "runsList", "newRunBtn", "railEmpty",
        # views
        "view-home", "view-run", "view-new",
        # run view
        "runToolbar", "runTitle", "tabReport", "tab3d", "downloadReport",
        "reportFrame", "fleet3d", "runProgress", "progressTrack", "progressFill",
        # editor
        "yamlBox", "gutter", "tplSelect", "titleInput", "validateBtn",
        "runBtn", "valErrors", "valOk",
    }
    missing = required - parser.ids
    assert not missing, f"index.html is missing ids: {sorted(missing)}"


def test_app_js_routes_match_the_shell():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for needle in ("#run/", "fleet3d", "#new", "/api/runs", "/api/validate", "/api/examples"):
        assert needle in js, needle
    # dynamic text goes through textContent, never markup injection
    assert ".innerHTML" not in js
    assert "document.write" not in js
    assert "eval(" not in js


# ---------------------------------------------------------------------------
# GET /api/examples
# ---------------------------------------------------------------------------


@pytest.fixture()
def served(tmp_path):
    manager = RunManager(tmp_path / "ws", start_worker=False)
    httpd = FleetsimHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        manager.shutdown(timeout=10.0)


def get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as resp:
        return resp.status, dict(resp.headers), resp.read()


def test_api_examples_serves_bundled_scenarios(served):
    status, headers, body = get(served, "/api/examples")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    examples = json.loads(body)
    names = [e["name"] for e in examples]
    assert names == sorted(names)
    assert "01_minimal" in names
    assert len(examples) >= 6
    for ex in examples:
        assert set(ex) == {"name", "yaml"}
        on_disk = (EXAMPLES_DIR / ex["name"] / "scenario.yaml").read_text(
            encoding="utf-8"
        )
        assert ex["yaml"] == on_disk  # read-only, verbatim


def test_list_examples_is_defensive(tmp_path):
    root = tmp_path / "examples"
    (root / "b_ok").mkdir(parents=True)
    (root / "b_ok" / "scenario.yaml").write_text("sim: {}\n", encoding="utf-8")
    (root / "a_ok").mkdir()
    (root / "a_ok" / "scenario.yaml").write_text("# a\n", encoding="utf-8")
    (root / "no_scenario").mkdir()
    (root / "big").mkdir()
    (root / "big" / "scenario.yaml").write_text("x" * (300 * 1024), encoding="utf-8")
    (root / "loose-file.yaml").write_text("ignored", encoding="utf-8")

    out = list_examples(root)
    assert [e["name"] for e in out] == ["a_ok", "b_ok"]  # sorted; others dropped
    assert out[0]["yaml"] == "# a\n"

    assert list_examples(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# the shell over HTTP: content types + CSP
# ---------------------------------------------------------------------------


def test_shell_assets_served_with_expected_types(served):
    status, headers, body = get(served, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Content-Security-Policy"] == CSP_APP
    assert b"/static/app.js" in body

    status, headers, _ = get(served, "/static/app.css")
    assert status == 200
    assert headers["Content-Type"].startswith("text/css")

    status, headers, _ = get(served, "/static/app.js")
    assert status == 200
    assert headers["Content-Type"].startswith("text/javascript")


# ---------------------------------------------------------------------------
# 3D fleet replay: vendored three.js + fleet3d.js module
# ---------------------------------------------------------------------------

VENDOR_DIR = STATIC_DIR / "vendor"

#: Pinned vendor build: three 0.185.1 (module + core, npm `three` package,
#: build/ directory).  Verified byte-identical from two mirrors (unpkg and
#: jsdelivr) at vendoring time; any local edit or swap fails this test.
THREE_VERSION = "0.185.1"
THREE_REVISION = "185"
THREE_SHA256 = {
    "three.module.min.js": (
        "86bcee248b64f44bcfc23c331ae74619061957d59cab040171dcb6fb5900beb6"
    ),
    "three.core.min.js": (
        "05b2609338c76cd65daf74f3ac515bc9a5045e1b3b33edc07d8c9bd55250fa90"
    ),
}


def test_vendored_three_is_present_and_byte_pinned():
    for name, expected in THREE_SHA256.items():
        path = VENDOR_DIR / name
        assert path.is_file(), f"missing vendored file {name}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, (
            f"{name}: sha256 {digest} != pinned {expected} — the vendored"
            f" three.js {THREE_VERSION} build must not be modified locally"
        )
    module = (VENDOR_DIR / "three.module.min.js").read_text(encoding="utf-8")
    core = (VENDOR_DIR / "three.core.min.js").read_text(encoding="utf-8")
    # the split build's only import is the sibling core file, relative
    assert 'from"./three.core.min.js"' in module
    # the pinned version string is embedded in the build (REVISION const)
    assert f'="{THREE_REVISION}"' in core
    for text in (module, core):
        assert "SPDX-License-Identifier: MIT" in text


def test_vendored_three_license_ships_alongside():
    lic = (VENDOR_DIR / "THREE_LICENSE").read_text(encoding="utf-8")
    assert "The MIT License" in lic
    assert "three.js authors" in lic
    assert "Permission is hereby granted" in lic


def test_fleet3d_module_imports_three_relatively_and_pins_revision():
    js = (STATIC_DIR / "fleet3d.js").read_text(encoding="utf-8")
    # relative same-origin module import only (CSP script-src 'self')
    assert 'from "./vendor/three.module.min.js"' in js
    m = re.search(r'THREE_REVISION\s*=\s*"(\d+)"', js)
    assert m, "fleet3d.js must pin the vendored three REVISION"
    assert m.group(1) == THREE_REVISION
    # same hygiene rules as app.js: dynamic text is data, never markup
    for banned in (".innerHTML", "document.write", "eval("):
        assert banned not in js, banned


def test_app_js_lazy_loads_fleet3d():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'import("./fleet3d.js")' in js


def test_vendored_three_served_from_static_subdirectory(served):
    status, headers, body = get(served, "/static/vendor/three.module.min.js")
    assert status == 200
    assert headers["Content-Type"].startswith("text/javascript")
    assert hashlib.sha256(body).hexdigest() == THREE_SHA256["three.module.min.js"]

    status, headers, _ = get(served, "/static/fleet3d.js")
    assert status == 200
    assert headers["Content-Type"].startswith("text/javascript")
