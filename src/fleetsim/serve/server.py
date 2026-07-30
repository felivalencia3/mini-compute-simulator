"""The ``fleetsim serve`` HTTP server (v0.5) — pure stdlib, local-first.

ROUTES (pinned contract; the app-shell phases code against THIS):

- ``GET  /api/runs``                -> ``[{id, title, status, created,
  headline: {occupancy, goodput, jobs_finished} | null, error?}]``
  (newest first; ``status`` one of ``queued|running|done|failed``)
- ``GET  /api/runs/{id}``           -> the row's meta plus
  ``summary: <full summary.json> | null`` (filled when done)
- ``GET  /api/runs/{id}/progress``  -> ``{status, progress: {t_us,
  horizon_us, jobs_finished, jobs_running, pending, occupancy_to_date,
  allocated_chips, healthy_chips} | null}``
- ``GET  /api/runs/{id}/model``     -> the viz JSON model
  (``build_viz_model``; disk-cached; 409 until the run is done)
- ``GET  /api/runs/{id}/report``    -> the self-contained 2D report HTML
  (``render_html``; disk-cached; 409 until done)
- ``POST /api/validate``  body ``{yaml: str}`` -> ``{ok, errors: [str]}``
  (always 200 for well-formed requests; bad request envelope -> 400)
- ``POST /api/runs``      body ``{yaml: str, title?: str}`` ->
  ``{id}`` (200), or 400 ``{ok: false, errors: [str]}`` when invalid
- ``DELETE /api/runs/{id}``         -> 200 ``{ok: true}`` for queued runs
  only (dequeue); 409 otherwise, 404 for unknown ids.  Running/done runs
  are never deleted by the API — deleting history is a filesystem
  concern, done while the server is down.
- ``GET /api/examples``             -> ``[{name, yaml}]`` — the bundled
  ``examples/*/scenario.yaml`` starter scenarios, read-only, sorted by
  name (``[]`` when the examples directory is not present, e.g. an
  installed wheel without the repo checkout).
- ``GET /`` and ``GET /static/*``   -> the app shell from the packaged
  ``static/`` directory.

SECURITY MODEL (local-first, DESIGN v0.5):

- The server binds 127.0.0.1 unless the operator explicitly widens it
  (the CLI prints a warning); it serves the operator's own filesystem
  workspace, so the HTTP surface is treated as UNTRUSTED anyway:
- run ids are server-generated slugs; every id from a request is
  re-validated (no separators / dot-names) and then path-resolved with
  ``Path.resolve().is_relative_to(workspace)`` — same belt for
  ``/static/*`` against the static root.  Traversal never leaves either
  root, regardless of encoding tricks.
- request bodies are JSON with a hard size cap; scenario text is parsed
  with ``yaml.safe_load`` only (inside the config layer), and runs
  execute IN-PROCESS in a worker thread — no subprocess, no shell, no
  string ever reaches an interpreter.
- errors are always JSON (``{"error": ...}``); tracebacks never leave
  the process (they would leak paths and internals to any local page
  that can make a request).
- every HTML response carries a Content-Security-Policy.  The app shell
  gets the strict pin ``default-src 'self'; img-src 'self' data:;
  style-src 'self' 'unsafe-inline'; script-src 'self'`` — belt on top of
  the local bind: even if a malicious job label or scenario string ever
  ended up interpolated into shell HTML, no external script could load
  and no exfil request could leave.  The 2D report is a SELF-CONTAINED
  single file whose one inline ``<script>`` is the whole app, so its CSP
  swaps ``'self'`` for ``'unsafe-inline'`` while keeping
  ``default-src 'self'`` (no external fetches possible); the report's
  own injection layer (`viz.render`) already script-escapes the data
  payload.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .runs import RunManager

__all__ = ["FleetsimHTTPServer", "list_examples", "serve"]

_STATIC_DIR = Path(__file__).with_name("static")

#: Repo-checkout location of the bundled example scenarios (absent on an
#: installed wheel; ``/api/examples`` then serves ``[]``).
_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"

#: Hard cap on request bodies (scenario YAML is text; 5 MB is generous).
_MAX_BODY = 5 * 1024 * 1024

#: Per-file cap for served example scenarios (they are all < 10 KB;
#: anything bigger is not one of ours).
_MAX_EXAMPLE_BYTES = 256 * 1024


def list_examples(examples_dir: Path | None = None) -> list[dict[str, str]]:
    """The bundled starter scenarios as ``[{name, yaml}]``, sorted by
    name.  Read-only and defensive: a missing directory, unreadable file,
    or oversized file simply drops out — never an error."""
    root = examples_dir if examples_dir is not None else _EXAMPLES_DIR
    out: list[dict[str, str]] = []
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    for child in children:
        path = child / "scenario.yaml"
        try:
            if not path.is_file() or path.stat().st_size > _MAX_EXAMPLE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append({"name": child.name, "yaml": text})
    return out

#: Pinned CSP for app-shell HTML (see module docstring).
CSP_APP = (
    "default-src 'self'; img-src 'self' data:;"
    " style-src 'self' 'unsafe-inline'; script-src 'self'"
)
#: CSP for the self-contained report (inline script/style by design).
CSP_REPORT = (
    "default-src 'self'; img-src 'self' data:;"
    " style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
)

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


class FleetsimHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the :class:`RunManager` and static
    root for its handlers.  ``daemon_threads`` so in-flight requests
    never block interpreter exit."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        runs: RunManager,
        static_dir: Path | None = None,
        examples_dir: Path | None = None,
    ):
        self.runs = runs
        self.static_dir = (static_dir or _STATIC_DIR).resolve()
        self.examples_dir = examples_dir  # None -> repo default
        super().__init__(address, _Handler)

    def handle_error(self, request, client_address):  # noqa: D102
        # A browser closing a keep-alive connection mid-read (tab close,
        # refresh, poll abort) raises ConnectionResetError inside the
        # socketserver framework, BEFORE any do_* handler runs.  That is
        # normal churn, not an error — the stdlib default would dump a
        # full traceback to the operator's terminal for every one.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    server_version = "fleetsim"
    sys_version = ""  # no Python version fingerprint in headers
    protocol_version = "HTTP/1.1"

    server: FleetsimHTTPServer  # narrowed for type checkers

    # -- response helpers -------------------------------------------------

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_html(self, html: str, csp: str, status: int = 200) -> None:
        self._send_bytes(
            status,
            html.encode("utf-8"),
            "text/html; charset=utf-8",
            extra={"Content-Security-Policy": csp},
        )

    def _read_json_body(self) -> dict[str, Any] | None:
        """The request body as a JSON object, or ``None`` after having
        already sent a 4xx JSON error."""
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True  # unread body would desync keep-alive
            self._send_error_json(411, "Content-Length required")
            return None
        if length < 0 or length > _MAX_BODY:
            self.close_connection = True
            self._send_error_json(413, f"body too large (max {_MAX_BODY} bytes)")
            return None
        raw = self.rfile.read(length)
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json(400, "body must be a JSON object")
            return None
        if not isinstance(doc, dict):
            self._send_error_json(400, "body must be a JSON object")
            return None
        return doc

    # -- routing -----------------------------------------------------------

    def _segments(self) -> list[str]:
        """Percent-decoded path segments, empties dropped.  Decoding
        happens BEFORE the run-id / static containment checks, so
        ``..%2F`` tricks are seen as the separators they are."""
        path = urlsplit(self.path).path
        return [unquote(seg) for seg in path.split("/") if seg]

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        try:
            self._route_get()
        except BrokenPipeError:  # client went away mid-response
            pass
        except Exception as exc:  # noqa: BLE001 - JSON errors, never tracebacks
            self._safe_500(exc)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._route_post()
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._safe_500(exc)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            self._route_delete()
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._safe_500(exc)

    def _safe_500(self, exc: Exception) -> None:
        try:
            self._send_error_json(
                500, f"internal error: {type(exc).__name__}: {exc}"
            )
        except Exception:  # headers already sent / socket dead
            pass

    # -- GET ----------------------------------------------------------------

    def _route_get(self) -> None:
        seg = self._segments()
        if not seg:  # GET /
            self._serve_static(["index.html"])
            return
        if seg[0] == "static":
            self._serve_static(seg[1:])
            return
        if seg == ["api", "examples"]:
            self._send_json(list_examples(self.server.examples_dir))
            return
        if seg[0] == "api" and len(seg) >= 2 and seg[1] == "runs":
            runs = self.server.runs
            if len(seg) == 2:
                self._send_json(runs.list_runs())
                return
            run_id = seg[2]
            if len(seg) == 3:
                info = runs.get_run(run_id)
                if info is None:
                    self._send_error_json(404, "no such run")
                else:
                    self._send_json(info)
                return
            if len(seg) == 4:
                sub = seg[3]
                if sub == "progress":
                    prog = runs.get_progress(run_id)
                    if prog is None:
                        self._send_error_json(404, "no such run")
                    else:
                        self._send_json(prog)
                    return
                if sub == "model":
                    code, payload = runs.model_json(run_id)
                    if code == 200:
                        self._send_bytes(
                            200,
                            payload.encode("utf-8"),
                            "application/json; charset=utf-8",
                        )
                    else:
                        self._send_error_json(code, payload)
                    return
                if sub == "report":
                    code, payload = runs.report_html(run_id)
                    if code == 200:
                        self._send_html(payload, CSP_REPORT)
                    else:
                        self._send_error_json(code, payload)
                    return
        self._send_error_json(404, "not found")

    def _serve_static(self, parts: list[str]) -> None:
        """Serve one packaged static file; the resolve+is_relative_to
        belt keeps every request inside the static root."""
        root = self.server.static_dir
        if not parts or any(p in (".", "..") for p in parts):
            self._send_error_json(404, "not found")
            return
        target = root.joinpath(*parts)
        try:
            resolved = target.resolve()
        except OSError:
            self._send_error_json(404, "not found")
            return
        if not resolved.is_relative_to(root) or not resolved.is_file():
            self._send_error_json(404, "not found")
            return
        ctype = _CONTENT_TYPES.get(
            resolved.suffix.lower(), "application/octet-stream"
        )
        body = resolved.read_bytes()
        extra = (
            {"Content-Security-Policy": CSP_APP}
            if ctype.startswith("text/html")
            else None
        )
        self._send_bytes(200, body, ctype, extra=extra)

    # -- POST ----------------------------------------------------------------

    def _route_post(self) -> None:
        seg = self._segments()
        if seg == ["api", "validate"]:
            doc = self._read_json_body()
            if doc is None:
                return
            text = doc.get("yaml")
            if not isinstance(text, str):
                self._send_error_json(400, "'yaml' must be a string")
                return
            errors = self.server.runs.validate_text(text)
            self._send_json({"ok": not errors, "errors": errors})
            return
        if seg == ["api", "runs"]:
            doc = self._read_json_body()
            if doc is None:
                return
            text = doc.get("yaml")
            if not isinstance(text, str):
                self._send_error_json(400, "'yaml' must be a string")
                return
            title = doc.get("title")
            if title is not None and not isinstance(title, str):
                self._send_error_json(400, "'title' must be a string")
                return
            errors = self.server.runs.validate_text(text)
            if errors:
                self._send_json({"ok": False, "errors": errors}, status=400)
                return
            run_id = self.server.runs.submit(text, title)
            self._send_json({"id": run_id})
            return
        self._send_error_json(404, "not found")

    # -- DELETE ----------------------------------------------------------------

    def _route_delete(self) -> None:
        seg = self._segments()
        if len(seg) == 3 and seg[0] == "api" and seg[1] == "runs":
            code, msg = self.server.runs.delete_queued(seg[2])
            if code == 200:
                self._send_json({"ok": True})
            else:
                self._send_error_json(code, msg)
            return
        self._send_error_json(404, "not found")

    # -- logging ----------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Quiet by default: progress polling would flood the terminal.
        # (BaseHTTPRequestHandler logs every request to stderr otherwise.)
        pass


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def serve(
    port: int = 8500,
    workspace: str | Path = "./fleetsim-runs",
    host: str = "127.0.0.1",
    open_browser: bool = False,
) -> int:
    """Run the fleetsim web app until Ctrl-C; returns an exit code.

    Binds ``host`` (loopback by default; anything else prints a loud
    warning — the app exposes the operator's runs and accepts scenario
    submissions).  Ctrl-C stops accepting requests, cancels queued runs,
    and cooperatively aborts the active run at its next metrics flush
    (marked ``failed`` with a clear error) — the process never hangs on
    a long simulation.
    """
    manager = RunManager(workspace)
    try:
        httpd = FleetsimHTTPServer((host, port), manager)
    except OSError as exc:
        print(f"error: cannot bind {host}:{port}: {exc}")
        manager.shutdown(timeout=5.0)
        return 2
    bound_host, bound_port = httpd.server_address[:2]
    if host not in _LOOPBACK_HOSTS:
        print(
            f"WARNING: binding non-loopback host {host!r} — the web app"
            f" exposes your runs and accepts scenario submissions from"
            f" anyone who can reach this address. Prefer the default"
            f" 127.0.0.1 plus an SSH tunnel."
        )
    display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
    url = f"http://{display_host}:{bound_port}/"
    print(f"fleetsim serve on {url}  (workspace: {manager.workspace})")
    print("Ctrl-C to stop.")
    if open_browser:
        import webbrowser

        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nshutting down: cancelling queued runs, aborting any active run…")
    finally:
        httpd.server_close()
        manager.shutdown()
    print("bye")
    return 0
