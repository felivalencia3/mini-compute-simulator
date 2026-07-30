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
- ``POST /api/runs/{id}/cancel``    -> 200 ``{ok: true}`` for RUNNING
  runs: cooperative cancel at the next metrics flush (the run is marked
  ``failed`` with ``cancelled by request``); 409 for any other status,
  404 unknown.
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
- errors are always JSON (``{"error": ...}``) — including framework
  errors (bad request line, unsupported method), which override the
  stdlib's HTML pages; tracebacks and exception MESSAGES never leave
  the process (they would leak paths and internals to any local page
  that can make a request) — a 500 carries the exception class only,
  the detail goes to the operator's terminal.
- the Host header is pinned to loopback authorities (anti
  DNS-rebinding: a rebound attacker domain resolves here but sends its
  own name in Host -> 421), and state-changing routes reject requests
  bearing a foreign ``Origin`` / cross-site ``Sec-Fetch-Site`` and
  require ``Content-Type: application/json`` (no CORS-"simple" CSRF
  POSTs; OPTIONS grants nothing, so preflights fail).
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
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from .runs import RunManager, WorkspaceLockError

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


def _example_web_note(text: str) -> str | None:
    """A short caveat for a starter scenario that cannot run as
    web-submitted (today: a trace workload with a RELATIVE source path —
    web runs execute from a fresh run directory, so it can never
    resolve).  ``None`` when the example runs as shipped."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None  # validation reports the real problem
    if not isinstance(doc, Mapping):
        return None
    workload = doc.get("workload")
    if not isinstance(workload, Mapping):
        return None
    if str(workload.get("kind", "")) != "trace":
        return None
    source = str(workload.get("source") or "")
    if source and not Path(source).is_absolute():
        return "CLI-only as shipped — web runs need an absolute trace path"
    return None


def list_examples(examples_dir: Path | None = None) -> list[dict[str, Any]]:
    """The bundled starter scenarios as ``[{name, yaml, runnable,
    note?}]``, sorted by name.  ``yaml`` is served VERBATIM; ``runnable``
    is false (with a human ``note``) when the scenario cannot run as
    web-submitted, so the editor can say so BEFORE the user hits
    Validate.  Read-only and defensive: a missing directory, unreadable
    file, or oversized file simply drops out — never an error."""
    root = examples_dir if examples_dir is not None else _EXAMPLES_DIR
    out: list[dict[str, Any]] = []
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
        note = _example_web_note(text)
        entry: dict[str, Any] = {
            "name": child.name,
            "yaml": text,
            "runnable": note is None,
        }
        if note is not None:
            entry["note"] = note
        out.append(entry)
    return out

#: Pinned CSP for app-shell HTML (see module docstring).
#: ``frame-ancestors`` does NOT fall back to default-src, so it is pinned
#: explicitly: only same-origin pages may frame the shell or the report.
CSP_APP = (
    "default-src 'self'; img-src 'self' data:;"
    " style-src 'self' 'unsafe-inline'; script-src 'self';"
    " frame-ancestors 'self'"
)
#: CSP for the self-contained report (inline script/style by design; the
#: app shell frames it same-origin, which 'self' permits).
CSP_REPORT = (
    "default-src 'self'; img-src 'self' data:;"
    " style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline';"
    " frame-ancestors 'self'"
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
        # HOST-HEADER PIN (anti DNS-rebinding): the loopback bind is only
        # a defense if the browser reached us via a loopback NAME — a
        # rebound attacker domain resolves to 127.0.0.1 but sends its own
        # name in Host, becoming same-origin with this server otherwise.
        # ``allowed_hosts`` is the exact authority set the handler
        # accepts; ``None`` disables the check (wildcard binds — the
        # operator explicitly exposed the app and client names are
        # unknowable).
        bind_host = str(address[0])
        port = int(self.server_address[1])
        if bind_host in ("", "0.0.0.0", "::"):
            self.allowed_hosts: frozenset[str] | None = None
        else:
            names = {"127.0.0.1", "localhost", "[::1]"}
            h = bind_host.lower()
            names.add(f"[{h}]" if ":" in h and not h.startswith("[") else h)
            allowed = set()
            for name in names:
                allowed.add(name)
                allowed.add(f"{name}:{port}")
            self.allowed_hosts = frozenset(allowed)

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
    #: Socket timeout: a connection that stalls mid-request (or declares
    #: a large Content-Length and never sends it) releases its thread
    #: instead of pinning it forever.
    timeout = 30

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

    def send_error(self, code, message=None, explain=None):  # noqa: D102
        # Framework-level failures (unsupported method, bad request line,
        # oversized request line, unknown HTTP version) route through
        # here BEFORE any do_* handler exists.  The stdlib default is a
        # text/html page with none of the hardening headers — the API
        # contract pins "every error is JSON, never an HTML error page",
        # so emit the same envelope _send_error_json uses.
        self.close_connection = True
        short = "error"
        if code in self.responses:
            short = self.responses[code][0]
        body = json.dumps({"error": message or short}).encode("utf-8")
        try:
            # A malformed request line leaves request_version at the
            # HTTP/0.9 default, which would suppress the status line and
            # headers entirely; real 0.9 clients do not exist, so always
            # emit a full response.
            if self.request_version == "HTTP/0.9":
                self.request_version = "HTTP/1.0"
            self.send_response(code, short)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD" and code >= 200 and code not in (204, 304):
                self.wfile.write(body)
        except OSError:
            pass  # socket already gone

    def _read_json_body(self) -> dict[str, Any] | None:
        """The request body as a JSON object, or ``None`` after having
        already sent a 4xx JSON error."""
        ctype = (
            (self.headers.get("Content-Type") or "")
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        if ctype != "application/json":
            # CSRF belt: text/plain (and form) POSTs are CORS-"simple" —
            # a hostile page can fire them cross-origin with no
            # preflight.  Requiring application/json forces a preflight,
            # which fails because OPTIONS carries no CORS grants.
            self._send_error_json(415, "Content-Type must be application/json")
            return None
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True  # unread body would desync keep-alive
            self._send_error_json(411, "Content-Length required")
            return None
        if length < 0:
            self.close_connection = True
            self._send_error_json(400, "invalid Content-Length")
            return None
        if length > _MAX_BODY:
            self.close_connection = True
            self._send_error_json(413, f"body too large (max {_MAX_BODY} bytes)")
            return None
        raw = self.rfile.read(length)
        self._body_consumed = True
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json(400, "body must be a JSON object")
            return None
        if not isinstance(doc, dict):
            self._send_error_json(400, "body must be a JSON object")
            return None
        return doc

    def _drain_unread_body(self) -> None:
        """Consume a request body no route read, so HTTP/1.1 keep-alive
        never parses leftover body bytes as the next request line.
        Anything undrainable closes the connection instead."""
        if self._body_consumed:
            return
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True  # chunked: not supported here
            return
        raw = self.headers.get("Content-Length")
        if raw is None:
            return
        try:
            length = int(raw)
        except ValueError:
            self.close_connection = True
            return
        if length < 0 or length > _MAX_BODY:
            self.close_connection = True
            return
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                self.close_connection = True
                return
            remaining -= len(chunk)
        self._body_consumed = True

    # -- routing -----------------------------------------------------------

    def _segments(self) -> list[str]:
        """Percent-decoded path segments, empties dropped.  Decoding
        happens BEFORE the run-id / static containment checks, so
        ``..%2F`` tricks are seen as the separators they are."""
        path = urlsplit(self.path).path
        return [unquote(seg) for seg in path.split("/") if seg]

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        self._dispatch(self._route_get)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(self._route_get)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch(self._route_post)

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch(self._route_delete)

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch(self._method_not_allowed)

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch(self._method_not_allowed)

    def do_OPTIONS(self) -> None:  # noqa: N802
        # Deliberately NO Access-Control-* headers: a cross-origin
        # preflight must fail, keeping non-simple requests unsendable
        # from other origins.
        self._dispatch(self._method_not_allowed)

    def _dispatch(self, route) -> None:
        self._body_consumed = False
        try:
            if not self._host_ok():
                self.close_connection = True
                self._send_error_json(
                    421,
                    "misdirected request: unexpected Host header"
                    " (use the address fleetsim serve printed)",
                )
            else:
                route()
        except BrokenPipeError:  # client went away mid-response
            pass
        except Exception as exc:  # noqa: BLE001 - JSON errors, never tracebacks
            self._safe_500(exc)
        finally:
            try:
                self._drain_unread_body()
            except Exception:  # noqa: BLE001 - never let cleanup raise
                self.close_connection = True

    def _method_not_allowed(self) -> None:
        body = json.dumps(
            {"error": f"method {self.command} not allowed"}
        ).encode("utf-8")
        self._send_bytes(
            405,
            body,
            "application/json; charset=utf-8",
            extra={"Allow": "GET, HEAD, POST, DELETE"},
        )

    def _host_ok(self) -> bool:
        """Anti DNS-rebinding: the client must have addressed us by a
        pinned loopback authority (see ``allowed_hosts``)."""
        allowed = self.server.allowed_hosts
        if allowed is None:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        return host in allowed

    def _cross_origin_rejected(self) -> bool:
        """CSRF gate for state-changing routes: reject any request that
        a browser marks as coming from ANOTHER origin.  Non-browser
        clients (curl, scripts) send neither header and pass.  Returns
        True after sending the 403."""
        origin = (self.headers.get("Origin") or "").strip().lower()
        if origin:
            ok = False
            if origin != "null":
                parts = urlsplit(origin)
                allowed = self.server.allowed_hosts
                if allowed is not None:
                    ok = parts.scheme == "http" and parts.netloc in allowed
                else:  # widened bind: same-authority check via Host
                    host = (self.headers.get("Host") or "").strip().lower()
                    ok = bool(host) and parts.netloc == host
            if not ok:
                self._send_error_json(403, "cross-origin request rejected")
                return True
        fetch_site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if fetch_site and fetch_site not in ("same-origin", "none"):
            self._send_error_json(403, "cross-site request rejected")
            return True
        return False

    def _safe_500(self, exc: Exception) -> None:
        # Full detail (whose message can embed filesystem paths) goes to
        # the operator's terminal ONLY; the client sees the exception
        # class alone — same privacy rule as the report path scrub.
        try:
            sys.stderr.write(
                f"fleetsim serve: internal error on"
                f" {self.command} {self.path}:"
                f" {type(exc).__name__}: {exc}\n"
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            self._send_error_json(500, f"internal error: {type(exc).__name__}")
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
        try:
            # ValueError: embedded NUL bytes (%00) make resolve()/stat()
            # raise — malformed path, not a 500.
            resolved = root.joinpath(*parts).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                self._send_error_json(404, "not found")
                return
        except (OSError, ValueError):
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
        if self._cross_origin_rejected():
            return
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
        if (
            len(seg) == 4
            and seg[0] == "api"
            and seg[1] == "runs"
            and seg[3] == "cancel"
        ):
            code, msg = self.server.runs.cancel_run(seg[2])
            if code == 200:
                self._send_json({"ok": True})
            else:
                self._send_error_json(code, msg)
            return
        self._send_error_json(404, "not found")

    # -- DELETE ----------------------------------------------------------------

    def _route_delete(self) -> None:
        if self._cross_origin_rejected():
            return
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
    try:
        manager = RunManager(workspace)
    except WorkspaceLockError as exc:
        print(f"error: {exc}")
        return 2
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
