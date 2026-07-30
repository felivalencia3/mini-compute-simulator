"""Run workspace management for ``fleetsim serve`` (v0.5).

:class:`RunManager` owns one *workspace* directory.  Every run managed by
the server lives in ``workspace/<slug>/`` containing:

- ``scenario.yaml`` — the submitted scenario text, verbatim;
- ``meta.json``     — ``{title, created_unix, status, error?}`` with
  ``status`` one of ``queued | running | done | failed``;
- the standard ``fleetsim run`` outputs (``summary.json``,
  ``jobs.parquet``, ``timeseries.parquet``, optional ``stints.parquet``)
  written directly into the run directory once the run executes;
- lazily-built caches: ``viz_model.json`` (the 2D report's data model)
  and ``report.html`` (the rendered self-contained report).

SLUGS ARE SERVER-GENERATED.  Clients never choose ids: a slug is a UTC
timestamp + boot-monotonic counter + short random suffix (RNG seeded once
at boot), matching ``^run-[0-9]{8}-[0-9]{6}-[0-9]{3}-[a-z0-9]{4}$``.
Wall-clock use is deliberate and allowed HERE — this is app plumbing, not
simulation code; determinism of the *runs themselves* is untouched.
Every id received from a client is re-validated by :meth:`resolve_dir`:
separator/dot-dot rejection first, then the ``Path.resolve()`` +
``is_relative_to(workspace)`` containment belt.

ONE WORKER THREAD executes runs FIFO off a queue.  The simulator is
CPU-bound pure Python, so running one scenario at a time is the correct
throughput choice (parallel runs would timeslice the GIL and finish
later, while making progress reporting misleading); queued runs report
``status: queued`` until their turn.  Live progress arrives through the
engine's ``progress_cb`` hook (one snapshot per metrics flush) and is
stored per run under the manager lock.

EXTERNAL RUNS: :meth:`list_runs` also surfaces any directory dropped
into the workspace that contains a ``summary.json`` but no ``meta.json``
(e.g. the output of a CLI ``fleetsim run -o workspace/name``); they show
as ``status: done`` with the directory name as id/title and the
``summary.json`` mtime as their creation time.

DELETION: only ``queued`` runs can be removed (dequeue — the directory
holds nothing but the server-written scenario/meta yet).  Running, done,
and failed runs are never deleted through the API; removing history is a
filesystem concern (delete the run directory while the server is down).

SHUTDOWN: :meth:`shutdown` cancels queued runs (marked ``failed`` with a
clear error) and cooperatively aborts the active one — the abort flag is
checked inside the progress callback, so the run stops at its next
metrics flush and is marked ``failed`` before the worker exits.  On boot
any leftover ``queued``/``running`` meta (a crashed previous server) is
repaired to ``failed``.

ONE SERVER PER WORKSPACE: a ``workspace/.serve.lock`` file (created
``O_EXCL``, containing the owner pid) is taken at init and released at
shutdown.  A second live server pointed at the same workspace would
otherwise "repair" the first server's queued/running meta to ``failed``
while those runs are still executing; instead it refuses to start with
:class:`WorkspaceLockError`.  A lock left behind by a crashed process
(dead pid, or unparseable content) is reclaimed automatically.
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .. import api
from ..config import ScenarioError, load_scenario, validate

__all__ = ["RunManager", "WorkspaceLockError", "validate_scenario_text"]

_SLUG_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_STOP = object()  # worker-queue sentinel

#: Managed statuses (pinned; the API contract exposes exactly these).
_STATUSES = ("queued", "running", "done", "failed")


class _RunAborted(Exception):
    """Raised inside the engine's progress callback to stop the active
    run cooperatively at the next metrics flush (server shutdown)."""


class WorkspaceLockError(RuntimeError):
    """Another live ``fleetsim serve`` process owns this workspace."""


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe for a lockfile pid.

    POSIX: signal 0 probes without side effects.  Elsewhere (Windows —
    where ``os.kill`` with an arbitrary signal would TERMINATE the
    process) be conservative and treat any recorded pid as live; the
    lock error message tells the operator how to clear a stale lock.
    """
    if pid <= 0:
        return False
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # EPERM etc.: some process with that pid exists
        return True
    return True


def validate_scenario_text(text: str, base_dir: Path) -> list[str]:
    """Validate scenario YAML *text*: parse, schema, feasibility.

    Returns the (possibly empty) error list — never raises for bad input.
    ``base_dir`` anchors relative ``workload.source`` paths for the
    trace-file existence check; the server passes a not-yet-existing
    directory under the workspace, which matches execution reality:
    web-submitted runs execute from a fresh run directory, so a relative
    trace path can never resolve (use absolute paths for traces).
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"invalid YAML: {exc}"]
    if not isinstance(doc, Mapping):
        return [f"scenario must be a mapping, got {type(doc).__name__}"]
    try:
        scenario = load_scenario(doc, strict=False)
    except ScenarioError as exc:
        return list(exc.errors)
    errors = validate(scenario)
    if not errors:
        from ..cli import _feasibility_errors

        errors = _feasibility_errors(scenario, base_dir)
    return [_hide_base_dir(e, base_dir) for e in errors]


def _hide_base_dir(error: str, base_dir: Path) -> str:
    """Web-friendly feasibility messages: the internal anchor directory
    (``workspace/_pending-run``) means nothing to a web user and leaks
    the operator's directory layout, so strip it back to the scenario's
    own relative path and state the actual remedy."""
    base = str(base_dir)
    if base not in error:
        return error
    error = error.replace(base + os.sep, "").replace(base, "")
    return (
        error
        + " (web-submitted runs execute from a fresh run directory, so a"
        " relative path cannot resolve — use an absolute path)"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write-then-rename so readers never observe a torn file.

    The temp name is UNIQUE PER WRITER (pid + random) — a shared
    ``<name>.tmp`` would let concurrent builders of the same target
    rename each other's temp file away, turning the loser's ``replace``
    into ``FileNotFoundError`` (review: concurrent cold-cache /model
    requests 500'd).  With unique names every writer renames its own
    file; the last rename wins and losers replaced an identical,
    already-published payload."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _summary_headline(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The three-number headline for run listings, straight from
    ``summary.json``: window occupancy/goodput, full jobs_finished."""
    window = summary.get("window") or {}
    full = summary.get("full") or {}
    counts = full.get("counts") or {}
    return {
        "occupancy": window.get("occupancy"),
        "goodput": window.get("goodput"),
        "jobs_finished": counts.get("jobs_finished"),
    }


class RunManager:
    """Owns the workspace: run directories, the FIFO worker, progress.

    All mutable state (meta files, progress dicts, the cancelled set) is
    touched only under ``self._lock`` — handler threads (HTTP) and the
    single worker thread share it.  ``start_worker=False`` is a testing
    seam: runs stay ``queued`` until :meth:`shutdown`.
    """

    def __init__(self, workspace: str | Path, *, start_worker: bool = True):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._progress: dict[str, dict[str, Any]] = {}
        self._cancelled: set[str] = set()
        #: Runs whose RUNNING execution should stop at the next metrics
        #: flush (cooperative cancel; POST /api/runs/{id}/cancel).
        self._cancel_requested: set[str] = set()
        #: Per-run build serialization for the model/report caches (an
        #: RLock because report_html builds the model inside its own
        #: hold).  Guarded by ``self._lock``.
        self._build_locks: dict[str, threading.RLock] = {}
        self._abort = threading.Event()
        self._shutdown = False
        self._active: str | None = None
        self._rng = random.Random()  # seeded once at boot (OS entropy)
        self._seq = 0
        self._owns_lock = False
        # The workspace lock MUST come before boot repair: repairing
        # another live server's queued/running meta is exactly the
        # corruption the lock exists to prevent.
        self._acquire_workspace_lock()
        self._repair_stale_meta()
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(
                target=self._work_loop, name="fleetsim-run-worker", daemon=True
            )
            self._worker.start()

    # -- workspace lock (one live server per workspace) --------------------

    @property
    def _lock_path(self) -> Path:
        return self.workspace / ".serve.lock"

    def _acquire_workspace_lock(self) -> None:
        """Take ``workspace/.serve.lock`` (O_EXCL, owner pid inside), or
        raise :class:`WorkspaceLockError` when a LIVE process holds it.
        A stale lock (dead pid / unreadable content) is reclaimed."""
        for _ in range(2):
            try:
                fd = os.open(
                    self._lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                )
            except FileExistsError:
                pid = self._read_lock_pid()
                if pid is not None and _pid_alive(pid):
                    raise WorkspaceLockError(
                        f"workspace {self.workspace} is already owned by a"
                        f" running fleetsim serve (pid {pid}) — two servers"
                        f" on one workspace would corrupt each other's run"
                        f" state. Use a different --workspace, or if that"
                        f" process is really gone delete {self._lock_path}"
                    )
                try:  # stale lock from a crash: reclaim and retry once
                    self._lock_path.unlink()
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n")
            self._owns_lock = True
            return
        raise WorkspaceLockError(
            f"cannot acquire {self._lock_path} — another server keeps"
            f" re-locking the workspace"
        )

    def _read_lock_pid(self) -> int | None:
        try:
            return int(self._lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _release_workspace_lock(self) -> None:
        if not self._owns_lock:
            return
        self._owns_lock = False
        try:
            self._lock_path.unlink()
        except OSError:
            pass

    # -- boot repair -----------------------------------------------------

    def _repair_stale_meta(self) -> None:
        """A previous server that crashed leaves ``queued``/``running``
        meta behind; nothing will ever execute those, so mark them
        failed with an explanatory error."""
        for meta_path in self.workspace.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if meta.get("status") in ("queued", "running"):
                meta["status"] = "failed"
                meta["error"] = "interrupted by a server restart"
                _atomic_write_text(
                    meta_path, json.dumps(meta, sort_keys=True) + "\n"
                )

    # -- slugs and path containment ---------------------------------------

    def _new_slug(self) -> str:
        """UTC-ordered, unique, URL-safe: ``run-<UTC>-<seq>-<rand>``."""
        while True:
            with self._lock:
                self._seq += 1
                seq = self._seq
                suffix = "".join(
                    self._rng.choice(_SLUG_ALPHABET) for _ in range(4)
                )
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            slug = f"run-{stamp}-{seq % 1000:03d}-{suffix}"
            if not (self.workspace / slug).exists():
                return slug

    def resolve_dir(self, run_id: str) -> Path | None:
        """Map a client-supplied id to its run directory, or ``None``.

        Two independent gates (belt and suspenders): reject ids that are
        empty, contain path separators, or are dot-names; then resolve
        the joined path and require it to stay under the workspace AND be
        a direct child.  No filesystem access outside the workspace can
        result from any input string.
        """
        if (
            not run_id
            or "/" in run_id
            or "\\" in run_id
            or run_id in (".", "..")
            or run_id.startswith(".")
        ):
            return None
        try:
            # ValueError: an embedded NUL byte (e.g. percent-encoded %00)
            # makes resolve()/stat() raise — a malformed id, not a 500.
            candidate = (self.workspace / run_id).resolve()
            if not candidate.is_relative_to(self.workspace):
                return None
            if candidate.parent != self.workspace:
                return None
            if not candidate.is_dir():
                return None
        except (OSError, ValueError):
            return None
        return candidate

    # -- meta ------------------------------------------------------------

    def _read_meta(self, run_dir: Path) -> dict[str, Any] | None:
        try:
            meta = json.loads(
                (run_dir / "meta.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return meta if isinstance(meta, dict) else None

    def _write_meta(self, run_dir: Path, meta: dict[str, Any]) -> None:
        _atomic_write_text(
            run_dir / "meta.json", json.dumps(meta, sort_keys=True) + "\n"
        )

    def _set_status(
        self, slug: str, status: str, error: str | None = None
    ) -> None:
        assert status in _STATUSES, status
        with self._lock:
            run_dir = self.workspace / slug
            if not run_dir.is_dir():
                return  # dequeued/removed concurrently; nothing to update
            meta = self._read_meta(run_dir) or {}
            meta["status"] = status
            if error is not None:
                meta["error"] = error
            self._write_meta(run_dir, meta)

    def _read_summary(self, run_dir: Path) -> dict[str, Any] | None:
        try:
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return summary if isinstance(summary, dict) else None

    # -- public API used by the HTTP layer ---------------------------------

    def validate_text(self, text: str) -> list[str]:
        """Errors for scenario YAML text ([] = valid); see
        :func:`validate_scenario_text` for the base-dir convention."""
        return validate_scenario_text(text, self.workspace / "_pending-run")

    def submit(self, text: str, title: str | None = None) -> str:
        """Create a run directory for pre-validated scenario text and
        enqueue it; returns the new server-generated id.  Raises
        ``RuntimeError`` after :meth:`shutdown`."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("server is shutting down")
        slug = self._new_slug()
        run_dir = self.workspace / slug
        run_dir.mkdir(parents=True)
        (run_dir / "scenario.yaml").write_text(text, encoding="utf-8")
        self._write_meta(
            run_dir,
            {
                "title": (title or "").strip() or slug,
                "created_unix": int(time.time()),
                "status": "queued",
            },
        )
        self._queue.put(slug)
        return slug

    def list_runs(self) -> list[dict[str, Any]]:
        """All runs in the workspace, newest first.

        Managed runs come from their ``meta.json``; external directories
        (``summary.json`` present, no ``meta.json``) surface as done runs
        so CLI outputs dropped into the workspace are browsable.
        ``headline`` is filled from ``summary.json`` when the run is done
        (``None`` otherwise or when the summary is unreadable).
        """
        rows: list[dict[str, Any]] = []
        for child in self.workspace.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            meta = self._read_meta(child)
            if meta is not None:
                status = meta.get("status", "failed")
                row: dict[str, Any] = {
                    "id": child.name,
                    "title": meta.get("title") or child.name,
                    "status": status,
                    "created": meta.get("created_unix"),
                    "headline": None,
                }
                if meta.get("error") is not None:
                    row["error"] = meta["error"]
            elif (child / "summary.json").is_file():
                try:
                    created = int((child / "summary.json").stat().st_mtime)
                except OSError:
                    created = None
                status = "done"
                row = {
                    "id": child.name,
                    "title": child.name,
                    "status": "done",
                    "created": created,
                    "headline": None,
                }
            else:
                continue
            if status == "done":
                summary = self._read_summary(child)
                if summary is not None:
                    row["headline"] = _summary_headline(summary)
            rows.append(row)
        rows.sort(key=lambda r: (-(r["created"] or 0), r["id"]))
        return rows

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Meta for one run, plus the full ``summary.json`` under
        ``"summary"`` when the run is done (``None`` otherwise).
        ``None`` when the id does not resolve to a run."""
        run_dir = self.resolve_dir(run_id)
        if run_dir is None:
            return None
        meta = self._read_meta(run_dir)
        if meta is not None:
            out: dict[str, Any] = {
                "id": run_id,
                "title": meta.get("title") or run_id,
                "status": meta.get("status", "failed"),
                "created": meta.get("created_unix"),
                "summary": None,
            }
            if meta.get("error") is not None:
                out["error"] = meta["error"]
        elif (run_dir / "summary.json").is_file():
            try:
                created = int((run_dir / "summary.json").stat().st_mtime)
            except OSError:
                created = None
            out = {
                "id": run_id,
                "title": run_id,
                "status": "done",
                "created": created,
                "summary": None,
            }
        else:
            return None
        if out["status"] == "done":
            out["summary"] = self._read_summary(run_dir)
        return out

    def get_progress(self, run_id: str) -> dict[str, Any] | None:
        """``{status, progress}`` for one run (``None`` = unknown id).
        ``progress`` is the engine's last flush snapshot — ``None`` until
        the first flush, for external runs, and after boot for runs that
        finished in an earlier server session."""
        info = self.get_run(run_id)
        if info is None:
            return None
        with self._lock:
            progress = self._progress.get(run_id)
        return {
            "status": info["status"],
            "progress": dict(progress) if progress is not None else None,
        }

    def delete_queued(self, run_id: str) -> tuple[int, str]:
        """Dequeue a queued run: ``(200, "ok")`` on success, else
        ``(404 | 409, reason)``.  Only ``queued`` runs are deletable —
        their directory holds nothing but the server-written scenario
        and meta; anything that ever ran is immutable history here."""
        with self._lock:
            run_dir = self.resolve_dir(run_id)
            meta = self._read_meta(run_dir) if run_dir is not None else None
            if run_dir is None or meta is None:
                return 404, "no such run"
            if meta.get("status") != "queued":
                return (
                    409,
                    f"only queued runs can be deleted"
                    f" (status: {meta.get('status')})",
                )
            self._cancelled.add(run_id)
            shutil.rmtree(run_dir, ignore_errors=True)
        return 200, "ok"

    def cancel_run(self, run_id: str) -> tuple[int, str]:
        """Request cooperative cancellation of a RUNNING run:
        ``(200, "ok")`` on acceptance, else ``(404 | 409, reason)``.

        The flag is polled inside the engine's progress callback (the
        same hook the shutdown abort uses), so the run stops at its next
        metrics flush and is marked ``failed`` with ``cancelled by
        request``.  Queued runs are dequeued with DELETE; done/failed
        history stays immutable."""
        run_dir = self.resolve_dir(run_id)
        meta = self._read_meta(run_dir) if run_dir is not None else None
        if run_dir is None or meta is None:
            return 404, "no such run"
        status = meta.get("status")
        if status != "running":
            return (
                409,
                f"only running runs can be cancelled (status: {status})"
                f" — queued runs are removed with DELETE",
            )
        with self._lock:
            self._cancel_requested.add(run_id)
        return 200, "ok"

    # -- viz model / report caches ----------------------------------------

    def _build_lock(self, run_id: str) -> threading.RLock:
        """The per-run cache-build lock (created on first use)."""
        with self._lock:
            lock = self._build_locks.get(run_id)
            if lock is None:
                lock = self._build_locks[run_id] = threading.RLock()
            return lock

    @staticmethod
    def _read_cache(cache: Path) -> str | None:
        if cache.is_file():
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                pass  # rebuild
        return None

    def _scrub_paths(self, message: str) -> str:
        """Strip the operator's absolute workspace path from a message
        bound for an HTTP response body (same privacy rule as
        ``_hide_base_dir`` for validation errors)."""
        base = str(self.workspace)
        return message.replace(base + os.sep, "").replace(base, "workspace")

    def model_json(self, run_id: str) -> tuple[int, str]:
        """``(200, model-json)`` for a done run; else ``(404|409|500,
        error message)``.  Built once via ``build_viz_model`` and cached
        as ``viz_model.json`` in the run directory (runs are immutable
        once done, so the cache never invalidates).  Builds are
        serialized per run so concurrent cold-cache requests wait for
        one build instead of racing it."""
        status, run_dir, msg = self._require_done(run_id)
        if run_dir is None:
            return status, msg
        cache = run_dir / "viz_model.json"
        payload = self._read_cache(cache)
        if payload is not None:
            return 200, payload
        with self._build_lock(run_id):
            payload = self._read_cache(cache)  # built while we waited?
            if payload is not None:
                return 200, payload
            from ..viz import build_viz_model, to_json

            try:
                model = build_viz_model(run_dir)
                # Display path only: the report footer/meta shows
                # out_dir, and "Download report.html" invites sharing
                # the file — the operator's absolute workspace path
                # (username, machine layout) must not travel with it.
                # The run id is the directory name, so nothing else
                # changes.
                model["meta"]["out_dir"] = run_id
                # The one human-chosen name (the submit-time title)
                # heads the report instead of the server slug; the slug
                # stays available as out_dir.
                meta = self._read_meta(run_dir)
                title = (meta or {}).get("title")
                if isinstance(title, str) and title.strip():
                    model["meta"]["title"] = (
                        f"fleetsim replay — {title.strip()}"
                    )
                payload = to_json(model)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                return 500, self._scrub_paths(
                    f"cannot build viz model: {exc}"
                )
            _atomic_write_text(cache, payload)
            return 200, payload

    def report_html(self, run_id: str) -> tuple[int, str]:
        """``(200, html)`` for a done run's self-contained 2D report,
        cached as ``report.html``; else ``(404|409|500, error)``."""
        status, run_dir, msg = self._require_done(run_id)
        if run_dir is None:
            return status, msg
        cache = run_dir / "report.html"
        html = self._read_cache(cache)
        if html is not None:
            return 200, html
        with self._build_lock(run_id):  # RLock: model_json re-enters
            html = self._read_cache(cache)
            if html is not None:
                return 200, html
            code, payload = self.model_json(run_id)
            if code != 200:
                return code, payload
            from ..viz import render_html

            html = render_html(json.loads(payload))
            _atomic_write_text(cache, html)
            return 200, html

    def _require_done(
        self, run_id: str
    ) -> tuple[int, Path | None, str]:
        """Shared gate for model/report: (status_code, run_dir | None,
        error message)."""
        run_dir = self.resolve_dir(run_id)
        if run_dir is None:
            return 404, None, "no such run"
        meta = self._read_meta(run_dir)
        if meta is not None:
            status = meta.get("status")
            if status != "done":
                return 409, None, f"run is not finished (status: {status})"
        elif not (run_dir / "summary.json").is_file():
            return 404, None, "no such run"
        return 200, run_dir, ""

    # -- worker ------------------------------------------------------------

    def _work_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            slug: str = item
            with self._lock:
                if slug in self._cancelled:
                    self._cancelled.discard(slug)
                    continue
                aborting = self._abort.is_set()
                if not aborting:
                    self._active = slug
            if aborting:  # raced a shutdown: never start, mark it failed
                self._set_status(
                    slug, "failed", error="server shut down before the run started"
                )
                continue
            self._set_status(slug, "running")
            self._execute(slug)
            with self._lock:
                self._active = None
                self._cancel_requested.discard(slug)

    def _execute(self, slug: str) -> None:
        run_dir = self.workspace / slug

        def on_progress(snapshot: dict[str, Any]) -> None:
            if self._abort.is_set():
                raise _RunAborted()
            with self._lock:
                if slug in self._cancel_requested:
                    raise _RunAborted()
                self._progress[slug] = snapshot

        try:
            # out_dir is FORCED to the run directory: a scenario's own
            # outputs.dir (attacker- or typo-controlled) never chooses
            # where the server writes.
            api.run_scenario(
                run_dir / "scenario.yaml",
                out_dir=run_dir,
                progress_cb=on_progress,
            )
        except _RunAborted:
            with self._lock:
                cancelled = slug in self._cancel_requested
            if cancelled:
                self._set_status(slug, "failed", error="cancelled by request")
            else:
                self._set_status(
                    slug, "failed", error="aborted at server shutdown"
                )
        except ScenarioError as exc:
            self._set_status(slug, "failed", error="; ".join(exc.errors))
        except Exception as exc:  # noqa: BLE001 - one run never kills the worker
            self._set_status(slug, "failed", error=f"{type(exc).__name__}: {exc}")
        else:
            self._set_status(slug, "done")

    # -- shutdown -----------------------------------------------------------

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop accepting, cancel queued runs, abort the active run at
        its next metrics flush, and join the worker (bounded by
        ``timeout``; the worker is a daemon thread, so a stuck run cannot
        hold the process hostage)."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._abort.set()
        # Drain the queue: everything still waiting is marked failed.
        drained: list[str] = []
        try:
            while True:
                item = self._queue.get_nowait()
                if item is not _STOP:
                    drained.append(item)
        except queue.Empty:
            pass
        for slug in drained:
            with self._lock:
                cancelled = slug in self._cancelled
                self._cancelled.discard(slug)
            if not cancelled:
                self._set_status(
                    slug, "failed", error="server shut down before the run started"
                )
        self._queue.put(_STOP)
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        self._release_workspace_lock()
