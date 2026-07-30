"""Run workspace management for ``fleetsim serve`` (v0.5, parallel in v0.8).

:class:`RunManager` owns one *workspace* directory.  Every run managed by
the server lives in ``workspace/<slug>/`` containing:

- ``scenario.yaml`` — the submitted scenario text, verbatim;
- ``meta.json``     — ``{title, created_unix, status, error?}`` (plus
  ``sweep_id`` / ``sweep_cell`` for a sweep cell) with ``status`` one of
  ``queued | running | done | failed``;
- the standard ``fleetsim run`` outputs (``summary.json``,
  ``jobs.parquet``, ``timeseries.parquet``, optional ``stints.parquet``)
  written directly into the run directory once the run executes;
- the LIVE SPOOL written while the run executes (v0.8): ``live.json``
  (latest progress snapshot + open-stint overlay + cursor, replaced
  atomically), ``live.jsonl`` (settled stint rows, one JSON object per
  line, append-only — line index == cursor), ``live_fleet.json`` (the
  stint level's domain geometry, written once) and ``cancel.flag`` (the
  parent's cooperative-stop signal to the child);
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

PARALLEL EXECUTION (v0.8).  Runs execute in a
:class:`~concurrent.futures.ProcessPoolExecutor` — SEPARATE PROCESSES, not
threads: the simulator is CPU-bound pure Python holding the GIL, so
threads would timeslice and finish *later* than serial execution, while
processes actually use the cores.  Determinism is untouched because a run
is a pure function of ``(scenario, seed)`` (fleetsim.api) — where it
executes cannot change its bytes.

Admission is FIFO with at most ``max_workers`` runs in flight: one
dispatcher thread pops the submit queue, waits for a free slot
(a semaphore), and hands the run to the pool.  Everything still waiting
reports ``status: queued`` with its 1-based ``queue_position``
(``1`` = next to start).  ``max_workers`` defaults to
``min(4, cpu_count - 1)`` (at least 1) and is set by ``serve --workers``.

The pool is created LAZILY on the first dispatch (a server that never
runs anything spawns nothing) with the ``forkserver`` start method where
POSIX offers it, else ``spawn``: forkserver forks each worker from a
single-threaded, module-preloaded helper — cheap like ``fork`` but
without forking this threaded HTTP server, which is the classic deadlock.
Workers ignore SIGINT (the parent alone handles Ctrl-C).

A pool whose worker died ABRUPTLY (SIGKILL — the OOM killer's signal, and
a legal scenario may declare 262,144 nodes) is permanently unusable: every
later ``submit`` raises ``BrokenProcessPool``.  So the pool is treated as
a CACHE, not a singleton: :meth:`RunManager._ensure_pool` checks it for
brokenness and rebuilds, and :meth:`RunManager._submit_run` rebuilds once
more if it breaks between the check and the submit.  Runs that were in
flight on the dead pool are already finalized as ``failed``, so a rebuild
loses nothing, and the failure message says the pool was rebuilt and the
run can be resubmitted instead of leaking a CPython exception name.

LIVE PROGRESS + STINTS THROUGH A PER-RUN SPOOL FILE, not a
``multiprocessing.Queue``.  Four reasons, in weight order: (1) the stream
now carries STINT ROWS, whose aggregate size is unbounded — on disk the
parent's memory stays flat and nothing is lost if no client is polling,
whereas an unread queue grows in RAM forever; (2) the read path becomes a
stateless file read, so any HTTP handler thread serves
``/api/runs/{id}/live`` without a drain thread or manager state — and a
run's stream is still readable after the run (or the server) ends;
(3) the cursor contract falls out for free (cursor == line index in an
append-only file), including for a client that reconnects; (4) a
``multiprocessing.Queue`` cannot be pickled into
``ProcessPoolExecutor.submit`` at all — it would have to ride an
``initializer`` global, which is CPython-internals-dependent.

CURSOR CONTRACT (``GET /api/runs/{id}/live?cursor=N``).  The child writes
each flush as: append the newly SETTLED stint rows to ``live.jsonl``,
then atomically replace ``live.json`` with ``{cursor, progress,
open_stints, open_truncated}`` where ``cursor`` counts every settled row
on disk.  A reader therefore reads ``live.json`` FIRST and then at most
``cursor`` rows from ``live.jsonl``, which pins it to one consistent
flush: the settled prefix and the open overlay never double-count a
stint.  Rows are immutable and returned exactly once; ``cursor`` is
monotonically non-decreasing.  ``open_stints`` is a REPLACE-WHOLESALE
overlay (``end_reason "open"``, ``t1_us`` = that flush's time) and is
``null`` while ``more`` is true, because a lagging client's settled
prefix does not line up with it.

THE OVERLAY IS SPOOLED ONLY WHILE SOMEONE IS WATCHING.  It is the one
part of a flush whose size grows with CONCURRENT jobs (example 04 holds
3,206 open stints — a 680 KB rewrite) and rewriting it at every flush is
write amplification proportional to concurrency times horizon/round: ~1 GB
of file writes for an 86-second run, to support a page nobody may have
open.  So ``live_payload`` touches ``live.watch`` on every poll and the
child includes the overlay only when that file is fresh (within
:data:`LIVE_WATCH_S`), or on the FINAL state write (:meth:`_LiveSpool.finish`,
called from the child's ``finally``) so a client that arrives after the run
still replays the identical open set.  A state file written with the overlay
omitted carries ``open_omitted: true``, which the payload reports as
``open_stints: null`` plus ``open_pending: true`` — never as an empty
overlay, which would be a claim that nothing was running.

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
clear error) and asks every in-flight run to stop by dropping its
``cancel.flag``, which the child checks inside its progress callback — so
a run stops at its next metrics flush.  Because a flush can be far away
in a big scenario, the wait is BOUNDED: after a short grace the worker
processes are terminated outright and anything still unfinished is marked
``failed``.  Ctrl-C therefore always returns promptly.  On boot any
leftover ``queued``/``running`` meta (a crashed previous server) is
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
import multiprocessing
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import yaml

from .. import api
from ..config import ScenarioError, load_scenario, validate

__all__ = [
    "LIVE_OFFSET_MEMO",
    "LIVE_ROW_LIMIT",
    "LIVE_WATCH_S",
    "MAX_PREVIEW_CLUSTERS",
    "MAX_PREVIEW_DOMAINS",
    "MAX_SCENARIO_BYTES",
    "MODEL_CACHE_MARKERS",
    "RunManager",
    "UnguardedMainError",
    "WorkspaceLockError",
    "default_max_workers",
    "flatten_scenario",
    "scenario_fleet_shape",
    "scenario_shape_errors",
    "validate_scenario_text",
]

_SLUG_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_STOP = object()  # dispatcher-queue sentinel

#: Managed statuses (pinned; the API contract exposes exactly these).
_STATUSES = ("queued", "running", "done", "failed")

#: Live-spool file names inside a run directory (see module docstring).
LIVE_STATE = "live.json"
LIVE_ROWS = "live.jsonl"
LIVE_FLEET = "live_fleet.json"
LIVE_WATCH = "live.watch"
CANCEL_FLAG = "cancel.flag"

#: Hard cap on stint rows in ONE ``/live`` response — the client loops on
#: ``more: true`` until it catches up.  Also caps the open-stint overlay
#: the child writes per flush (a bounded worst case per state rewrite).
LIVE_ROW_LIMIT = 5000

#: How long a ``live.watch`` touch keeps the open-stint overlay spooled.
#: Comfortably above the client's 1 s poll and below a human's patience:
#: a tab that closes stops paying for the overlay within this window.
LIVE_WATCH_S = 15.0

#: Entries kept in the per-run live-read memo (run id -> (lines, byte
#: offset)).  Bounded because a long-lived server over a big workspace
#: would otherwise hold one entry per run ever scrubbed, forever.
LIVE_OFFSET_MEMO = 256

#: The pinned v0.5 progress-snapshot keys (the ``/progress`` contract).
#: The v0.8 stint keys are stripped back out before spooling: progress
#: stays exactly the shape the app has always polled.
_PROGRESS_KEYS = (
    "t_us",
    "horizon_us",
    "jobs_finished",
    "jobs_running",
    "pending",
    "occupancy_to_date",
    "allocated_chips",
    "healthy_chips",
)

#: What a run says when its worker process was killed from outside.  The
#: operator can act on this; ``BrokenProcessPool: A child process
#: terminated abruptly, the process pool is not usable anymore`` (the
#: CPython text this replaces) reads like the server is finished, and the
#: server is NOT finished — the pool is rebuilt on the next dispatch.
_POOL_DIED_MSG = (
    "the worker process running this run was killed from outside — the"
    " OOM killer is the usual cause (a scenario may legally declare"
    " 262,144 nodes). The worker pool has been rebuilt, so submitting"
    " the run again will execute it; a smaller fleet or fewer --workers"
    " avoids the kill."
)

#: Bounded grace (seconds) between "please stop at your next flush" and
#: terminating worker processes at shutdown.  Ctrl-C must not wait on a
#: scenario whose next metrics flush is minutes away.
_ABORT_GRACE_S = 2.0

#: Cap on a run TITLE.  A title is echoed by every ``GET /api/runs``,
#: which every open tab polls on a 3-second interval forever, so an
#: unbounded one is permanent bandwidth: a generated sweep label once
#: reached 1.55 MB and made the rail poll ~500 KB/s per tab.  240 is
#: past any readable title and the rail truncates at ~40 anyway.
MAX_TITLE_CHARS = 240

#: Cap on the scenario text ``GET /api/runs/{id}/scenario`` serves.  Runs
#: we wrote hold at most the 5 MB submitted body, and a CLI-dropped
#: directory could hold anything — a compare view must not pull a
#: gigabyte through the browser.  1 MB is ~50x the largest example.
MAX_SCENARIO_BYTES = 1024 * 1024

#: Keys a CURRENT ``viz_model.json`` must contain.  A cache written by an
#: older fleetsim is rebuilt instead of served: the run's recorded bytes
#: never change, but the model schema grows, and the v0.8 analysis tab
#: reads frames series that older caches do not carry.  Keep this list in
#: step with the newest additive block in ``fleetsim.viz.data``.
MODEL_CACHE_MARKERS = ('"failure_kills_delta":', '"frag_index":')

#: Scenario file names looked for inside a run directory, in order (the
#: same set and precedence :func:`fleetsim.viz.data.build_viz_model` uses
#: to recover seed/round metadata, so both read the same file).
_SCENARIO_NAMES = ("scenario.yaml", "scenario.yml", "config.yaml", "config.yml")


def default_max_workers() -> int:
    """``min(4, cpu_count - 1)``, at least 1 — the default parallelism.

    One core is left for the HTTP server and the operator's machine, and
    the cap keeps a default-configured server from filling a big host
    with simulations nobody asked for (``serve --workers`` overrides)."""
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus - 1))


class UnguardedMainError(RuntimeError):
    """A :class:`RunManager` was constructed inside a worker process.

    Which means one thing in practice: the program that started the server
    is a SCRIPT whose top level is not guarded by
    ``if __name__ == "__main__":``.  Worker processes re-execute the
    parent's ``__main__`` module (the documented requirement of the
    ``spawn`` / ``forkserver`` start methods), so an unguarded script
    starts a second server — new workspace lock, new HTTP bind, new runs —
    inside every worker.  Raised eagerly, with the fix, because the
    downstream symptom (a workspace-lock error attributed to a phantom
    second server) points nowhere near the cause.
    """


def _in_worker_process() -> bool:
    """True when this code is running inside a multiprocessing worker.

    During the child's ``__main__`` re-execution ``parent_process()`` is
    still ``None`` (it is set later, in ``_bootstrap``) but the process
    name has already been overwritten with e.g. ``ForkServerProcess-1`` —
    verified on CPython 3.11/3.12 — so both signals are checked.
    """
    try:
        if multiprocessing.parent_process() is not None:
            return True
        return multiprocessing.current_process().name != "MainProcess"
    except Exception:  # noqa: BLE001 - a probe must never break startup
        return False


class _RunAborted(Exception):
    """Raised inside the child's progress callback to stop the run
    cooperatively at the next metrics flush (cancel request / shutdown).
    Never crosses the process boundary — the child converts it to an
    ``{"outcome": "aborted"}`` result."""


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


def _feasibility_key(scenario: Any, base_dir: Path) -> str:
    """A cache key covering EXACTLY what ``_feasibility_errors`` reads.

    That function builds the fleet (``scenario.fleet`` only — see
    ``fleet.build.build_fleet``), resolves the scheduler
    (``name`` + ``params``) and stats the trace source
    (``workload.kind`` / ``workload.source``, anchored at ``base_dir``).
    Nothing else in the scenario can change its answer, so two documents
    agreeing on these agree on the errors.  Config objects are plain
    dataclasses with no suppressed fields, so ``repr`` is total: a key
    collision would need two DIFFERENT configs with equal reprs, which
    cannot happen; an accidental key DIFFERENCE only costs a cache miss.
    """
    return repr(
        (
            scenario.fleet,
            scenario.scheduler.name,
            scenario.scheduler.params,
            scenario.workload.kind,
            scenario.workload.source,
            str(base_dir),
        )
    )


def validate_scenario_text(
    text: str,
    base_dir: Path,
    feasibility_cache: "_FeasibilityCache | None" = None,
) -> list[str]:
    """Validate scenario YAML *text*: parse, schema, feasibility.

    Returns the (possibly empty) error list — never raises for bad input.
    ``base_dir`` anchors relative ``workload.source`` paths for the
    trace-file existence check; the server passes a not-yet-existing
    directory under the workspace, which matches execution reality:
    web-submitted runs execute from a fresh run directory, so a relative
    trace path can never resolve (use absolute paths for traces).

    ``feasibility_cache`` memoizes the EXPENSIVE half (building the fleet
    is O(nodes) and a legal scenario may declare 262,144 of them).  It is
    keyed on :func:`_feasibility_key`, so a sweep whose only axis is
    ``sim.seed`` builds its fleet once instead of once per cell, and two
    editors previewing the same fleet share one build.
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

        if feasibility_cache is None:
            errors = _feasibility_errors(scenario, base_dir)
        else:
            errors = feasibility_cache.get(
                _feasibility_key(scenario, base_dir),
                lambda: _feasibility_errors(scenario, base_dir),
            )
    return [_hide_base_dir(e, base_dir) for e in errors]


def scenario_shape_errors(text: str) -> list[str]:
    """PARSE + SCHEMA errors only — the editor preview's gate.

    Deliberately NOT the run gate: it skips the feasibility pass, which
    builds the fleet and is the one part of validation whose cost scales
    with the declared node count (measured: 1.9 s at the 262,144-node
    ceiling, 89 s for a wide 100,000-rack tree).  The preview is
    arithmetic over the count tree and runs on every debounced keystroke,
    so it must not carry a gate that expensive.  ``POST /api/validate``
    and ``POST /api/runs`` still run the full pass — nothing that can
    reach execution is validated any less than before.
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
    return list(validate(scenario))


class _FeasibilityCache:
    """A tiny bounded LRU of feasibility results, shared by every caller.

    Entries are pure functions of their key (:func:`_feasibility_key`),
    so this is a cache in the strict sense — dropping one changes only
    the cost of the next call, never its answer.  ``_gate`` bounds how
    many misses compute at once: three editors typing on the same
    262,144-node scenario should cost ONE fleet build, not three
    concurrent ones on a ``ThreadingHTTPServer`` with no thread cap.
    """

    def __init__(self, maxsize: int = 64, concurrency: int = 2):
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, list[str]] = OrderedDict()
        self._gate = threading.BoundedSemaphore(max(1, int(concurrency)))

    def get(self, key: str, compute: Any) -> list[str]:
        hit = self._peek(key)
        if hit is not None:
            return hit
        with self._gate:
            # Recheck: a request we queued behind may have filled it.
            hit = self._peek(key)
            if hit is not None:
                return hit
            errors = list(compute())
        with self._lock:
            self._entries[key] = errors
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        return list(errors)

    def _peek(self, key: str) -> list[str] | None:
        with self._lock:
            errors = self._entries.get(key)
            if errors is None:
                return None
            self._entries.move_to_end(key)
            return list(errors)


#: Caps for the editor's fleet-shape preview.  It is a block diagram, not
#: a fleet dump: past these the preview shows the counts and says it
#: truncated, rather than shipping thousands of blocks to the browser.
MAX_PREVIEW_CLUSTERS = 24
MAX_PREVIEW_DOMAINS = 256


def _leaf_specs(group: Any) -> list[tuple[int, str | None]]:
    """``[(chips_per_node, chip_type), ...]`` for a count-tree subtree —
    one entry per DISTINCT leaf shape, not per instance."""
    if not group.children:
        return [(int(group.chips), group.chip_type)]
    out: list[tuple[int, str | None]] = []
    for child in group.children:
        for spec in _leaf_specs(child):
            if spec not in out:
                out.append(spec)
    return out


def _has_level(groups: Any, level: str) -> bool:
    """Does any group in this subtree sit at ``level``?"""
    return any(
        g.level == level or _has_level(g.children, level) for g in groups
    )


def _count_recording(groups: Any, level: str | None) -> int:
    """How many RECORDING domains a subtree declares — arithmetic only.

    ``level is None`` is the ``stints: true`` rule (the domains directly
    below the cluster root, so just the sum of the top counts).  A level
    NAME counts a group's instances when it sits at that level and no
    descendant does, mirroring ``_build_stint_leaf_map``'s "the leaf's
    NEAREST ancestor at that level" (nearest = deepest).
    """
    total = 0
    for group in groups:
        count = int(group.count)
        if level is None:
            total += count
        elif group.level == level and not _has_level(group.children, level):
            total += count
        elif group.children:
            total += count * _count_recording(group.children, level)
    return total


def _preview_domains(
    groups: Any,
    prefix: str,
    counters: dict[tuple[str, str], int],
    depth: int,
    level: str | None,
    out: list[dict[str, Any]],
) -> None:
    """Append at most :data:`MAX_PREVIEW_DOMAINS` recording-domain blocks.

    Numbered exactly as ``fleet.build`` numbers them (per parent + level,
    document order) so a preview block and a later stint row carry the
    same id.  Bounded by the output cap at EVERY level of the recursion:
    the true total comes from :func:`_count_recording`, which is
    arithmetic, so a 262,144-node fleet costs 256 blocks here, not
    262,144 iterations.
    """
    for group in groups:
        if len(out) >= MAX_PREVIEW_DOMAINS:
            return
        count = int(group.count)
        key = (prefix, group.level)
        base = counters.get(key, 0)
        counters[key] = base + count
        recording = (
            depth == 0
            if level is None
            else group.level == level and not _has_level(group.children, level)
        )
        if recording:
            each_chips = group.total_chips() // max(1, count)
            each_nodes = group.total_nodes() // max(1, count)
            for i in range(count):
                if len(out) >= MAX_PREVIEW_DOMAINS:
                    return
                out.append(
                    {
                        "short": f"{group.level}{base + i}",
                        "path": f"{prefix}/{group.level}{base + i}".lstrip("/"),
                        "chips": each_chips,
                        "nodes": each_nodes,
                    }
                )
        elif group.children:
            for i in range(count):
                if len(out) >= MAX_PREVIEW_DOMAINS:
                    return
                _preview_domains(
                    group.children,
                    f"{prefix}/{group.level}{base + i}",
                    counters,
                    depth + 1,
                    level,
                    out,
                )


def scenario_fleet_shape(text: str) -> dict[str, Any] | None:
    """The fleet a scenario DESCRIBES, as counts — the data the editor's
    block-diagram preview draws before anything runs.

    Arithmetic only: the declarative count-tree is *summed*, never
    materialized.  A scenario may legally declare 262,144 nodes (the
    validation ceiling) and this runs on every debounced keystroke, so
    building a real :class:`~fleetsim.fleet.tree.FleetTree` here would be
    a self-inflicted denial of service.

    THE BLOCKS ARE THE LEVEL THE RUN WILL ACTUALLY RECORD, read from
    ``outputs.stints`` — a level NAME records domains at that level
    (``stints_mode: "level"``), ``true`` records the level directly below
    each cluster root (``"root_children"``), and absent/``false`` records
    nothing at all (``"off"``, and the shape is drawn at the root-children
    level purely to show the fleet).  Drawing the root-children level
    unconditionally, as v0.8 first shipped, was wrong for every scenario
    that names a level — example 07 ships ``stints: node`` and previewed
    4 rack blocks for a run that records 32 node domains.

    Returns ``None`` when the text does not parse into a scenario at all
    (the caller has already reported those errors).  Shape::

        {total_chips, total_nodes, n_clusters, chip_types: [str],
         stints_mode: "level"|"root_children"|"off",
         stints_level: str|null,
         clusters_truncated: bool,
         clusters: [{id, metro, name, levels: [str], map_level: str|null,
                     chips, nodes, chip_type: str|null,
                     chips_per_node: int|null, n_domains,
                     domains_truncated,
                     domains: [{short, path, chips, nodes}]}]}
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(doc, Mapping):
        return None
    try:
        scenario = load_scenario(doc, strict=False)
    except ScenarioError:
        return None
    fleet = scenario.fleet
    stints = scenario.outputs.stints
    stints_level = stints if isinstance(stints, str) else None
    stints_mode = (
        "level" if stints_level else ("root_children" if stints else "off")
    )
    clusters: list[dict[str, Any]] = []
    total_chips = 0
    total_nodes = 0
    chip_types: list[str] = []
    n_clusters = 0
    for metro in fleet.metros:
        for dc in metro.datacenters:
            for cluster in dc.clusters:
                n_clusters += 1
                c_chips = cluster.total_chips()
                c_nodes = cluster.total_nodes()
                total_chips += c_chips
                total_nodes += c_nodes
                specs = [s for g in cluster.children for s in _leaf_specs(g)]
                for _chips, ctype in specs:
                    if ctype and ctype not in chip_types:
                        chip_types.append(ctype)
                if len(clusters) >= MAX_PREVIEW_CLUSTERS:
                    continue
                per_node = {c for c, _t in specs}
                types = {t for _c, t in specs if t}
                # The domains the RUN will record (outputs.stints), so a
                # preview block and a later stint row are the same thing.
                target = stints_level if stints_mode == "level" else None
                counters: dict[tuple[str, str], int] = {}
                domains: list[dict[str, Any]] = []
                _preview_domains(
                    cluster.children, "", counters, 0, target, domains
                )
                n_domains = _count_recording(cluster.children, target)
                map_level: str | None = (
                    stints_level
                    if stints_mode == "level"
                    else (cluster.children[0].level if cluster.children else None)
                )
                clusters.append(
                    {
                        "id": f"{metro.name}/{cluster.id}",
                        "metro": metro.name,
                        "name": cluster.id,
                        "levels": list(cluster.levels),
                        "map_level": map_level,
                        "chips": c_chips,
                        "nodes": c_nodes,
                        "chip_type": next(iter(types)) if len(types) == 1 else None,
                        "chips_per_node": (
                            next(iter(per_node)) if len(per_node) == 1 else None
                        ),
                        "n_domains": n_domains,
                        "domains_truncated": n_domains > len(domains),
                        "domains": domains,
                    }
                )
    return {
        "total_chips": total_chips,
        "total_nodes": total_nodes,
        "n_clusters": n_clusters,
        "chip_types": chip_types,
        "stints_mode": stints_mode,
        "stints_level": stints_level,
        "clusters_truncated": n_clusters > len(clusters),
        "clusters": clusters,
    }


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
    """The headline numbers for run listings, straight from
    ``summary.json``: window occupancy/goodput, full jobs_finished, and
    the window FRAGMENTATION means.

    Fragmentation rides along (v0.8) because "does tighter gang packing
    reduce fragmentation?" is a question the sweep board could not answer
    at all: a placement axis is flat on occupancy and moves entirely in
    ``fragmentation``.  Keys are ``frag.<level>`` plus
    ``frag.stranded_whole_nodes`` when the run recorded it, so a sweep
    metric selector can offer exactly what the cells actually carry.
    """
    window = summary.get("window") or {}
    full = summary.get("full") or {}
    counts = full.get("counts") or {}
    out: dict[str, Any] = {
        "occupancy": window.get("occupancy"),
        "goodput": window.get("goodput"),
        "jobs_finished": counts.get("jobs_finished"),
    }
    frag = window.get("fragmentation")
    if isinstance(frag, Mapping):
        for key, stats in frag.items():
            if isinstance(stats, Mapping) and "mean" in stats:
                out[f"frag.{key}"] = stats["mean"]
    return out


def _json_safe(value: Any) -> Any:
    """A ``yaml.safe_load`` value narrowed to something ``json.dumps``
    can serialize.  ``safe_load`` yields dates, times and byte strings
    for ordinary-looking YAML (``2026-07-30``, ``!!binary``), which would
    turn a scenario response into a 500; those become their ``str()``.
    Numbers, strings, booleans and null pass through untouched."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # NaN / +-inf are valid YAML floats and invalid JSON.
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def flatten_scenario(doc: Any, prefix: str = "") -> dict[str, Any]:
    """A scenario document as ``{dotted.path: value}``.

    Nested MAPPINGS recurse; everything else (scalars and lists alike) is
    a leaf, so ``topology.counts`` is one comparable value rather than
    ``counts.0`` / ``counts.1`` rows that read as unrelated changes.  An
    empty mapping is its own leaf (``{}``) — otherwise dropping a whole
    subtree would silently vanish from a config diff.  Keys are
    ``str()``-ed and values narrowed by :func:`_json_safe`.
    """
    out: dict[str, Any] = {}
    if not isinstance(doc, Mapping):
        return {prefix: _json_safe(doc)} if prefix else {}
    if not doc and prefix:
        return {prefix: {}}
    for key, value in doc.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_scenario(value, path))
        else:
            out[path] = _json_safe(value)
    return out


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """One JSON object from ``path``, or ``None`` (missing / unreadable /
    torn / not an object).  Never raises — every caller is on a request
    path where a partially-written spool file is normal churn."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ---------------------------------------------------------------------------
# Child process: execute one run, spool live progress + stints to disk
# ---------------------------------------------------------------------------


def _pool_initializer() -> None:
    """Worker-process setup: IGNORE SIGINT.

    Ctrl-C in the operator's terminal is delivered to every process in the
    group.  A worker that dies of KeyboardInterrupt takes the whole pool
    down as ``BrokenProcessPool`` before the parent can record *why* its
    runs stopped.  Ignoring it here makes the parent the single Ctrl-C
    handler: it drops each in-flight run's ``cancel.flag`` (cooperative
    stop at the next metrics flush) and terminates whatever is left.
    """
    import signal

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass


class _LiveSpool:
    """The child's progress callback: spool one flush to the run directory.

    Write order is load-bearing (see the module docstring's cursor
    contract): stint rows are APPENDED first, then ``live.json`` is
    atomically replaced with the cursor that counts them.  A reader that
    takes ``live.json`` first can therefore trust that every row it names
    is already on disk, and that the open-stint overlay belongs to exactly
    that prefix.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._rows_path = run_dir / LIVE_ROWS
        self._state_path = run_dir / LIVE_STATE
        self._fleet_path = run_dir / LIVE_FLEET
        self._watch_path = run_dir / LIVE_WATCH
        self._cancel_path = run_dir / CANCEL_FLAG
        self._fh: Any = None
        self._cursor = 0
        self._fleet_written = False
        #: Last flush's progress + overlay, kept so :meth:`finish` can
        #: publish the final state with the overlay even when nobody was
        #: polling — a client that arrives after the run must replay the
        #: identical open set.
        self._last_progress: dict[str, Any] = {}
        self._last_open: list[Any] = []
        self._last_truncated = False
        self._omitted = False

    def _watched(self) -> bool:
        """Is a client polling ``/live`` right now?

        ``live_payload`` touches ``live.watch`` on every request; a stat
        per flush is the whole cost of asking, against a 680 KB overlay
        rewrite per flush for the answer "nobody is looking".
        """
        try:
            age = time.time() - self._watch_path.stat().st_mtime
        except OSError:
            return False
        return -LIVE_WATCH_S <= age <= LIVE_WATCH_S

    def __call__(self, snapshot: dict[str, Any]) -> None:
        if self._cancel_path.exists():
            raise _RunAborted()
        fleet = snapshot.get("stint_fleet")
        if fleet is not None and not self._fleet_written:
            _atomic_write_text(self._fleet_path, json.dumps(fleet, sort_keys=True))
            self._fleet_written = True
        rows = snapshot.get("stints") or []
        if rows:
            if self._fh is None:
                self._fh = self._rows_path.open("a", encoding="utf-8")
            self._fh.write(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
            )
            self._fh.flush()
        self._cursor = int(snapshot.get("stint_cursor", self._cursor + len(rows)))
        open_rows = snapshot.get("open_stints") or []
        self._last_progress = {k: snapshot.get(k) for k in _PROGRESS_KEYS}
        self._last_open = open_rows[:LIVE_ROW_LIMIT]
        self._last_truncated = len(open_rows) > LIVE_ROW_LIMIT
        self._write_state(with_overlay=self._watched())

    def _write_state(self, *, with_overlay: bool) -> None:
        self._omitted = not with_overlay
        _atomic_write_text(
            self._state_path,
            json.dumps(
                {
                    "cursor": self._cursor,
                    "progress": self._last_progress,
                    "open_stints": self._last_open if with_overlay else None,
                    "open_truncated": self._last_truncated and with_overlay,
                    "open_omitted": not with_overlay,
                },
                sort_keys=True,
            ),
        )

    def finish(self) -> None:
        """Publish the FINAL state with the open overlay, always.

        The overlay is skipped mid-run when nothing is polling, but the
        last one is the one a late client replays (``cursor=0`` after the
        run) — withholding it there would make the same run read
        differently depending on whether a tab happened to be open.
        """
        if self._omitted and self._last_progress:
            try:
                self._write_state(with_overlay=True)
            except OSError:  # pragma: no cover - a full disk at the very end
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def _child_run(run_dir_str: str) -> dict[str, Any]:
    """Execute ``<run_dir>/scenario.yaml`` IN A POOL WORKER PROCESS.

    Returns ``{"outcome": "done" | "aborted" | "failed", "error"?: str}``
    rather than raising: an outcome dict pickles cleanly and keeps the
    parent's status bookkeeping in one place (the parent knows whether an
    abort was a user cancel or a shutdown; the child does not).

    ``out_dir`` is FORCED to the run directory, exactly as the in-process
    v0.5 path did: a scenario's own ``outputs.dir`` never chooses where
    the server writes.
    """
    run_dir = Path(run_dir_str)
    spool = _LiveSpool(run_dir)
    try:
        api.run_scenario(
            run_dir / "scenario.yaml",
            out_dir=run_dir,
            progress_cb=spool,
            progress_stints=True,
        )
    except _RunAborted:
        return {"outcome": "aborted"}
    except KeyboardInterrupt:  # pragma: no cover - SIGINT is ignored above
        return {"outcome": "aborted"}
    except ScenarioError as exc:
        return {"outcome": "failed", "error": "; ".join(exc.errors)}
    except Exception as exc:  # noqa: BLE001 - one run never kills the worker
        return {"outcome": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        spool.finish()
        spool.close()
    return {"outcome": "done"}


class RunManager:
    """Owns the workspace: run directories, the FIFO dispatcher, progress.

    All mutable state (meta files, the queued order, the cancelled set) is
    touched only under ``self._lock`` — handler threads (HTTP), the
    dispatcher thread, and future-completion callbacks share it.
    ``start_worker=False`` is a testing seam: runs stay ``queued`` (and no
    process pool is ever created) until :meth:`shutdown`.

    ``max_workers`` caps runs in flight; ``None`` takes
    :func:`default_max_workers`.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        start_worker: bool = True,
        max_workers: int | None = None,
    ):
        if _in_worker_process():
            raise UnguardedMainError(
                "fleetsim serve cannot start inside a worker process. This"
                " almost always means the script that called it is missing"
                ' an `if __name__ == "__main__":` guard: simulation workers'
                " re-execute the main module, so an unguarded script starts"
                " a second server in every worker. Wrap the call:\n"
                '    if __name__ == "__main__":\n'
                "        serve(port=8500, workspace='./fleetsim-runs')\n"
                " (the `fleetsim serve` command line is already guarded.)"
            )
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_workers = (
            default_max_workers() if max_workers is None else max(1, int(max_workers))
        )
        self._lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue()
        #: Queued slugs in FIFO admission order — the source of the
        #: ``queue_position`` a client sees.  A slug leaves the list at the
        #: moment it is handed to the pool.
        self._queued: list[str] = []
        #: Slugs currently executing in a worker process -> their future.
        self._inflight: dict[str, Future] = {}
        self._cancelled: set[str] = set()
        #: Runs whose RUNNING execution should stop at the next metrics
        #: flush (cooperative cancel; POST /api/runs/{id}/cancel).
        self._cancel_requested: set[str] = set()
        #: Per-run build serialization for the model/report caches (an
        #: RLock because report_html builds the model inside its own
        #: hold).  Guarded by ``self._lock``.
        self._build_locks: dict[str, threading.RLock] = {}
        #: Live-spool read memo: run id -> (lines consumed, byte offset
        #: after them), so a client walking the cursor forward does not
        #: re-scan ``live.jsonl`` from byte 0 on every poll.  BOUNDED
        #: (LRU, :data:`LIVE_OFFSET_MEMO` entries): it is a pure cache —
        #: an evicted run's next read simply rescans — and an unbounded
        #: one grows with every run any client ever scrubbed.
        self._live_offsets: OrderedDict[str, tuple[int, int]] = OrderedDict()
        #: Feasibility-pass memo shared by every validation caller (see
        #: :class:`_FeasibilityCache`).
        self._feasibility = _FeasibilityCache()
        self._abort = threading.Event()
        self._shutdown = False
        self._rng = random.Random()  # seeded once at boot (OS entropy)
        self._seq = 0
        self._owns_lock = False
        self._slots = threading.Semaphore(self.max_workers)
        self._pool: ProcessPoolExecutor | None = None
        # The workspace lock MUST come before boot repair: repairing
        # another live server's queued/running meta is exactly the
        # corruption the lock exists to prevent.
        self._acquire_workspace_lock()
        self._repair_stale_meta()
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(
                target=self._work_loop, name="fleetsim-run-dispatch", daemon=True
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
        with self._lock:
            self._set_status_locked(slug, status, error)

    def _set_status_locked(
        self, slug: str, status: str, error: str | None = None
    ) -> None:
        """:meth:`_set_status` for a caller already holding ``self._lock``.

        Exists so the dispatcher can leave the queued list and write
        ``running`` in ONE lock hold: otherwise a listing taken in between
        would report a run as ``queued`` with no ``queue_position``, which
        is a state the API contract does not have.
        """
        assert status in _STATUSES, status
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
        :func:`validate_scenario_text` for the base-dir convention.

        The expensive feasibility pass is memoized across calls (same
        fleet + scheduler + trace source = same answer), which is what
        keeps a 64-cell seed sweep from building one 262,144-node fleet
        per cell."""
        return validate_scenario_text(
            text, self.workspace / "_pending-run", self._feasibility
        )

    def submit(
        self,
        text: str,
        title: str | None = None,
        *,
        meta_extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Create a run directory for pre-validated scenario text and
        enqueue it; returns the new server-generated id.  Raises
        ``RuntimeError`` after :meth:`shutdown`.

        ``meta_extra`` adds keys to ``meta.json`` (the sweep layer passes
        ``sweep_id`` / ``sweep_cell``); it can never overwrite the four
        keys the manager owns.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("server is shutting down")
        slug = self._new_slug()
        run_dir = self.workspace / slug
        run_dir.mkdir(parents=True)
        (run_dir / "scenario.yaml").write_text(text, encoding="utf-8")
        meta: dict[str, Any] = dict(meta_extra or {})
        meta.pop("error", None)
        clean = (title or "").strip()
        if len(clean) > MAX_TITLE_CHARS:
            clean = clean[: MAX_TITLE_CHARS - 1].rstrip() + "…"
        meta.update(
            {
                "title": clean or slug,
                "created_unix": int(time.time()),
                "status": "queued",
            }
        )
        # ONE lock hold for "meta says queued" and "the queue knows it":
        # a reader that caught the gap would see a queued run with no
        # queue_position, a state the API contract does not have.
        with self._lock:
            self._write_meta(run_dir, meta)
            self._queued.append(slug)
        self._queue.put(slug)
        return slug

    def queue_positions(self) -> dict[str, int]:
        """Queued slug -> 1-based position (``1`` = next to start)."""
        with self._lock:
            return {slug: i + 1 for i, slug in enumerate(self._queued)}

    def list_runs(self) -> list[dict[str, Any]]:
        """All runs in the workspace, newest first.

        Managed runs come from their ``meta.json``; external directories
        (``summary.json`` present, no ``meta.json``) surface as done runs
        so CLI outputs dropped into the workspace are browsable.
        ``headline`` is filled from ``summary.json`` when the run is done
        (``None`` otherwise or when the summary is unreadable).
        ``queue_position`` is the 1-based FIFO position of a ``queued`` run
        and ``None`` for every other status; ``sweep_id`` / ``sweep_cell``
        appear only on sweep cells.
        """
        positions = self.queue_positions()
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
                    "queue_position": (
                        positions.get(child.name) if status == "queued" else None
                    ),
                }
                if meta.get("error") is not None:
                    row["error"] = meta["error"]
                if meta.get("sweep_id") is not None:
                    row["sweep_id"] = meta["sweep_id"]
                    row["sweep_cell"] = meta.get("sweep_cell")
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
                    "queue_position": None,
                }
            else:
                continue
            if status == "done":
                summary = self._read_summary(child)
                if summary is not None:
                    row["headline"] = _summary_headline(summary)
            rows.append(row)
        # RENUMBER against the rows actually observed as queued.  The
        # positions snapshot is taken before the directory walk, so a run
        # admitted mid-walk reads `running` here while the runs behind it
        # still carry their old (now off-by-one) positions — a listing
        # with positions [2, 3] and no 1.  Renumbering in the snapshot's
        # order keeps FIFO meaning and makes "1 = next to start" true of
        # every listing, not just of an unraced one.
        queued_rows = [r for r in rows if r["status"] == "queued"]
        queued_rows.sort(
            key=lambda r: (
                r["queue_position"] is None,  # not in the queue yet: last
                r["queue_position"] or 0,
                r["id"],
            )
        )
        for pos, row in enumerate(queued_rows, start=1):
            row["queue_position"] = pos
        rows.sort(key=lambda r: (-(r["created"] or 0), r["id"]))
        return rows

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Meta for one run, plus the full ``summary.json`` under
        ``"summary"`` when the run is done (``None`` otherwise).
        ``None`` when the id does not resolve to a run."""
        run_dir = self.resolve_dir(run_id)
        if run_dir is None:
            return None
        # Meta AND position under ONE lock hold.  Read separately, the
        # dispatcher could admit the run in between and the caller would
        # get status "queued" with queue_position null — the state
        # _set_status_locked's comment says the contract does not have.
        with self._lock:
            meta = self._read_meta(run_dir)
            position = (
                (self._queued.index(run_id) + 1)
                if (
                    meta is not None
                    and meta.get("status") == "queued"
                    and run_id in self._queued
                )
                else None
            )
        if meta is not None:
            status = meta.get("status", "failed")
            out: dict[str, Any] = {
                "id": run_id,
                "title": meta.get("title") or run_id,
                "status": status,
                "created": meta.get("created_unix"),
                "summary": None,
                "queue_position": position if status == "queued" else None,
            }
            if meta.get("error") is not None:
                out["error"] = meta["error"]
            if meta.get("sweep_id") is not None:
                out["sweep_id"] = meta["sweep_id"]
                out["sweep_cell"] = meta.get("sweep_cell")
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
                "queue_position": None,
            }
        else:
            return None
        if out["status"] == "done":
            out["summary"] = self._read_summary(run_dir)
        return out

    def get_progress(self, run_id: str) -> dict[str, Any] | None:
        """``{status, progress}`` for one run (``None`` = unknown id).

        ``progress`` is the last flush snapshot the run spooled, with
        exactly the pinned v0.5 keys — ``None`` until the first flush and
        for external (CLI-dropped) runs.  Read straight off the run's
        ``live.json``, so it survives a server restart (v0.5 kept it in
        memory and lost it); ``queue_position`` rides along for queued
        runs so a poller needs one request, not two.
        """
        info = self.get_run(run_id)
        if info is None:
            return None
        state = _read_json_file(self.workspace / run_id / LIVE_STATE)
        progress = (state or {}).get("progress")
        if not isinstance(progress, dict):
            progress = None
        return {
            "status": info["status"],
            "progress": progress,
            "queue_position": info.get("queue_position"),
        }

    # -- live replay stream (v0.8) ----------------------------------------

    def live_payload(
        self, run_id: str, cursor: int = 0, limit: int = LIVE_ROW_LIMIT
    ) -> tuple[int, Any]:
        """``(200, payload)`` for ``GET /api/runs/{id}/live``, else
        ``(404, message)``.

        Payload::

            {status, cursor, more, progress, stints, open_stints,
             open_truncated, open_pending, stalled_at, fleet}

        - ``stints``: settled rows after the requested ``cursor``, capped
          at :data:`LIVE_ROW_LIMIT`, each returned exactly once ever;
        - ``cursor``: the caller's next cursor (``request cursor +
          len(stints)``, with a cursor past the end of the spool clamped
          back to the true row count so a stale value self-heals);
        - ``more``: rows remain on disk beyond this response — poll again
          immediately with the returned cursor;
        - ``open_stints``: the flush's open-stint overlay (``end_reason
          "open"``), or ``null`` while ``more`` is true (a lagging
          client's settled prefix does not line up with it) or while the
          child is not spooling it (``open_pending``);
        - ``open_truncated``: the overlay hit the row cap;
        - ``open_pending``: the overlay is not on disk yet because nothing
          was polling when the last flush was written (module docstring).
          This request registered interest; the next flush carries it.
          NOT the same as an empty overlay, which claims nothing is
          running;
        - ``stalled_at``: normally ``null``.  A line number means the spool
          could not be read past it (a truncated or corrupt row), so the
          stream CANNOT advance — ``more`` may be true with no rows to
          give, forever.  A client must stop hot-looping and say so;
        - ``progress``: the LATEST flush snapshot (never lags, even when
          ``more`` is true);
        - ``fleet``: the stint level's exact domain geometry
          (``{map_level, clusters, domains}``), sent only when the request
          asked for ``cursor <= 0`` — it never changes during a run.
          ``null`` when the scenario records no stints, or before the
          first flush.

        Reading ``live.json`` BEFORE ``live.jsonl`` and clamping to its
        cursor is what makes the settled prefix and the overlay one
        consistent snapshot (module docstring).
        """
        run_dir = self.resolve_dir(run_id)
        info = self.get_run(run_id) if run_dir is not None else None
        if run_dir is None or info is None:
            return 404, "no such run"
        self._touch_watch(run_dir)
        start = max(0, int(cursor))
        want = max(1, min(int(limit), LIVE_ROW_LIMIT))
        state = _read_json_file(run_dir / LIVE_STATE) or {}
        try:
            total = max(0, int(state.get("cursor") or 0))
        except (TypeError, ValueError):
            total = 0
        rows, stalled_at = self._read_live_rows(
            run_id, run_dir, start, min(total - start, want)
        )
        # A cursor BEYOND the spool (a stale or hand-typed value) is clamped
        # to the real row count instead of being echoed back forever, so the
        # stream self-heals.  For a client that only ever echoes the cursor
        # it was given, ``start <= total`` always holds and this is a no-op.
        new_cursor = min(start, total) + len(rows)
        more = new_cursor < total
        progress = state.get("progress")
        overlay = state.get("open_stints")
        pending = bool(state.get("open_omitted")) and not isinstance(overlay, list)
        return 200, {
            "status": info["status"],
            "cursor": new_cursor,
            "more": more,
            "progress": progress if isinstance(progress, dict) else None,
            "stints": rows,
            "open_stints": (
                None
                if (more or pending)
                else (overlay if isinstance(overlay, list) else [])
            ),
            "open_truncated": bool(state.get("open_truncated")) and not more,
            "open_pending": pending,
            "stalled_at": stalled_at,
            "fleet": (
                _read_json_file(run_dir / LIVE_FLEET) if start == 0 else None
            ),
        }

    @staticmethod
    def _touch_watch(run_dir: Path) -> None:
        """Register "a client is watching this run's live stream".

        The child reads the mtime once per flush and spools the open-stint
        overlay only while it is fresh — see the module docstring.  Best
        effort: a run directory removed under us is not an error here.
        """
        try:
            (run_dir / LIVE_WATCH).touch()
        except OSError:
            pass

    def _read_live_rows(
        self, run_id: str, run_dir: Path, start: int, count: int
    ) -> tuple[list[dict[str, Any]], int | None]:
        """``(rows, stalled_at)``: ``count`` spooled stint rows from line
        ``start``, and the line the read could not get past (``None`` when
        it simply ran out of complete rows).

        Seeks with a per-run ``(lines, byte offset)`` memo so sequential
        polling is O(new rows) instead of O(file); a cursor BEHIND the memo
        (a reconnecting client) rescans from byte 0.  A trailing line
        without its newline is a write in flight — dropped, never parsed,
        and NOT a stall.  A COMPLETE line that does not parse is a stall:
        the cursor is the line index, so the stream can never move past it
        and the caller must be told rather than left polling for rows that
        will never come.
        """
        if count <= 0:
            return [], None
        with self._lock:
            seen, offset = self._live_offsets.get(run_id, (0, 0))
        if seen > start:
            seen, offset = 0, 0
        rows: list[dict[str, Any]] = []
        stalled_at: int | None = None
        idx = seen
        try:
            with (run_dir / LIVE_ROWS).open("rb") as fh:
                fh.seek(offset)
                while idx < start:
                    line = fh.readline()
                    if not line.endswith(b"\n"):
                        return [], None
                    idx += 1
                    offset += len(line)
                while len(rows) < count:
                    line = fh.readline()
                    if not line.endswith(b"\n"):
                        break
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        stalled_at = idx
                        break
                    if not isinstance(row, dict):
                        stalled_at = idx  # never skip a line: cursor == index
                        break
                    idx += 1
                    offset += len(line)
                    rows.append(row)
        except OSError:
            return [], None
        with self._lock:
            prev = self._live_offsets.get(run_id)
            if prev is None or idx > prev[0]:
                self._live_offsets[run_id] = (idx, offset)
            self._live_offsets.move_to_end(run_id, last=True)
            while len(self._live_offsets) > LIVE_OFFSET_MEMO:
                self._live_offsets.popitem(last=False)
        return rows, stalled_at

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
            if run_id in self._queued:
                self._queued.remove(run_id)
            self._live_offsets.pop(run_id, None)
            shutil.rmtree(run_dir, ignore_errors=True)
        return 200, "ok"

    def cancel_run(self, run_id: str) -> tuple[int, str]:
        """Request cooperative cancellation of a RUNNING run:
        ``(200, "ok")`` on acceptance, else ``(404 | 409, reason)``.

        The request is delivered to the worker PROCESS as a ``cancel.flag``
        file in the run directory, which the child checks inside its
        progress callback — so the run stops at its next metrics flush and
        is marked ``failed`` with ``cancelled by request``.  Queued runs
        are dequeued with DELETE; done/failed history stays immutable."""
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
        self._request_child_stop(run_id)
        return 200, "ok"

    def _request_child_stop(self, slug: str) -> None:
        """Drop the run's ``cancel.flag`` (best effort — a run that already
        finished has nothing to stop, and its directory may be gone)."""
        try:
            (self.workspace / slug / CANCEL_FLAG).write_text("1", encoding="utf-8")
        except OSError:
            pass

    # -- scenario text (config diff / re-edit) -----------------------------

    def scenario_doc(self, run_id: str) -> tuple[int, Any]:
        """``(200, {id, name, yaml, flat, parse_error?})`` for a run whose
        directory holds a scenario file; else ``(404, message)``.

        Available at EVERY status (queued included) — a compare view is
        most useful before the runs finish.  ``flat`` is the document as
        ``{dotted.path: value}`` (:func:`flatten_scenario`), which is what
        a config diff needs and what a browser cannot compute for itself
        without a YAML parser; ``null`` plus ``parse_error`` when the file
        is not a YAML mapping.  ``yaml`` is the file VERBATIM, truncated
        at :data:`MAX_SCENARIO_BYTES` (``truncated: true`` says so) — the
        server never re-serializes a run's scenario, so what the compare
        view shows is what the run executed.
        """
        run_dir = self.resolve_dir(run_id)
        if run_dir is None:
            return 404, "no such run"
        if self._read_meta(run_dir) is None and not (
            run_dir / "summary.json"
        ).is_file():
            return 404, "no such run"
        for name in _SCENARIO_NAMES:
            path = run_dir / name
            try:
                if not path.is_file():
                    continue
                raw = path.read_bytes()[: MAX_SCENARIO_BYTES + 1]
            except OSError:
                continue
            truncated = len(raw) > MAX_SCENARIO_BYTES
            text = raw[:MAX_SCENARIO_BYTES].decode("utf-8", errors="replace")
            out: dict[str, Any] = {
                "id": run_id,
                "name": name,
                "yaml": text,
                "flat": None,
                "truncated": truncated,
            }
            if truncated:
                out["parse_error"] = (
                    f"scenario file is larger than {MAX_SCENARIO_BYTES} bytes"
                    f" — shown truncated, not parsed"
                )
                return 200, out
            try:
                doc = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                out["parse_error"] = f"invalid YAML: {exc}"
                return 200, out
            if not isinstance(doc, Mapping):
                out["parse_error"] = "scenario file is not a YAML mapping"
                return 200, out
            out["flat"] = flatten_scenario(doc)
            return 200, out
        return 404, "no scenario file recorded for this run"

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

    @staticmethod
    def _model_cache_is_current(payload: str) -> bool:
        """Is a cached ``viz_model.json`` from the CURRENT schema?

        Runs are immutable, so the cache never invalidates on content —
        but the model SCHEMA grows between versions, and a run finished
        under an older fleetsim would otherwise keep serving a model the
        app can no longer read (the v0.8 analysis tab needs the frames
        series added in that release).  The marker is a substring check
        on the compact JSON rather than a parse: ``to_json`` pins the
        separators, so these keys appear exactly like this, and a 15 MB
        frontier model is not worth deserializing to answer one bool.
        """
        return all(marker in payload for marker in MODEL_CACHE_MARKERS)

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
        if payload is not None and self._model_cache_is_current(payload):
            return 200, payload
        with self._build_lock(run_id):
            payload = self._read_cache(cache)  # built while we waited?
            if payload is not None and self._model_cache_is_current(payload):
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

    @staticmethod
    def _report_cache_is_current(html: str) -> bool:
        """Is a cached ``report.html`` from the CURRENT fleetsim?

        The report EMBEDS the model as JSON, so the same schema markers
        that gate ``viz_model.json`` gate the report — plus the version
        the model records, which catches a change in ``render_html``
        itself.  Without this the report iframe and the analysis tab on
        one run could be built by two different releases and disagree
        about which one produced them.
        """
        from .. import __version__

        return RunManager._model_cache_is_current(html) and (
            f'"fleetsim_version":"{__version__}"' in html
        )

    def report_html(self, run_id: str) -> tuple[int, str]:
        """``(200, html)`` for a done run's self-contained 2D report,
        cached as ``report.html``; else ``(404|409|500, error)``.  A
        cache written by an older fleetsim is rebuilt, exactly as
        :meth:`model_json` rebuilds a stale model cache."""
        status, run_dir, msg = self._require_done(run_id)
        if run_dir is None:
            return status, msg
        cache = run_dir / "report.html"
        html = self._read_cache(cache)
        if html is not None and self._report_cache_is_current(html):
            return 200, html
        with self._build_lock(run_id):  # RLock: model_json re-enters
            html = self._read_cache(cache)
            if html is not None and self._report_cache_is_current(html):
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

    # -- dispatcher + process pool -----------------------------------------

    @staticmethod
    def _pool_is_broken(pool: ProcessPoolExecutor) -> bool:
        """Has a worker died abruptly, making this pool unusable forever?

        ``_broken`` is a CPython internal (a message string once set, else
        False), so this is a probe, not a contract: when it is missing we
        answer "not broken" and :meth:`_submit_run` still catches the
        ``BrokenProcessPool`` the submit itself raises.  Belt and braces,
        because the alternative is a server that accepts submissions
        forever and can never execute one.
        """
        return bool(getattr(pool, "_broken", False))

    def _discard_pool(self, pool: ProcessPoolExecutor) -> None:
        """Let a dead pool go without waiting on it."""
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001 - a corpse must not block the rebuild
            pass

    def _drop_broken_pool(self) -> None:
        """Forget the current pool so the next :meth:`_ensure_pool` builds
        a fresh one.  Safe at any time: runs that were in flight on it are
        finalized by their own done-callbacks."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            self._discard_pool(pool)

    def _submit_run(self, slug: str) -> Future:
        """Hand one run to the pool, rebuilding it if it is dead.

        A pool can break BETWEEN the health check and the submit (the
        worker dies in that window), so the retry is on the submit itself
        and happens exactly once — a second failure is a real error and
        surfaces as one.
        """
        arg = str(self.workspace / slug)
        try:
            return self._ensure_pool().submit(_child_run, arg)
        except BrokenProcessPool:
            self._drop_broken_pool()
            return self._ensure_pool().submit(_child_run, arg)

    def _ensure_pool(self) -> ProcessPoolExecutor:
        """The lazily created worker pool (see the module docstring for the
        start-method choice).

        A BROKEN pool counts as no pool: it is dropped and rebuilt, because
        a ProcessPoolExecutor whose worker was SIGKILLed (the OOM killer's
        signal) refuses every future submit, and caching it forever means
        one dead worker bricks the server until an operator restarts it.

        Created under the lock and refused after :meth:`shutdown`, so a
        dispatcher racing shutdown can never bring a fresh pool (and fresh
        worker processes) into existence after ``_kill_pool`` ran.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("server is shutting down")
            pool, stale = self._pool, None
            if pool is not None:
                if not self._pool_is_broken(pool):
                    return pool
                stale, self._pool = pool, None
        if stale is not None:
            self._discard_pool(stale)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("server is shutting down")
            if self._pool is not None:  # rebuilt while we let the corpse go
                return self._pool
            try:
                ctx = multiprocessing.get_context("forkserver")
                # Preload this module in the forkserver helper so every
                # forked worker starts with fleetsim (and pandas) imported.
                try:
                    multiprocessing.set_forkserver_preload([__name__])
                except (AttributeError, ValueError):  # pragma: no cover
                    pass
            except ValueError:  # pragma: no cover - non-POSIX
                ctx = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=ctx,
                initializer=_pool_initializer,
            )
            return self._pool

    def _work_loop(self) -> None:
        """FIFO admission: one run leaves the queue only when a worker slot
        is free, so ``queue_position`` means what it says."""
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            slug: str = item
            if not self._acquire_slot():  # shutting down
                self._drop_queued(slug, "server shut down before the run started")
                continue
            with self._lock:
                if slug in self._cancelled:
                    self._cancelled.discard(slug)
                    self._slots.release()
                    continue
                aborting = self._abort.is_set()
                if not aborting:
                    # ONE lock hold for the whole admission step: a reader
                    # must never catch a run between "no longer queued" and
                    # "running" (that would be a queued row with no
                    # queue_position, a state the contract does not have).
                    if slug in self._queued:
                        self._queued.remove(slug)
                    self._set_status_locked(slug, "running")
            if aborting:  # raced a shutdown: never start, mark it failed
                self._slots.release()
                self._drop_queued(slug, "server shut down before the run started")
                continue
            try:
                future = self._submit_run(slug)
            except BrokenProcessPool:
                self._slots.release()
                self._set_status(slug, "failed", error=_POOL_DIED_MSG)
                continue
            except Exception as exc:  # noqa: BLE001 - pool creation/submit
                self._slots.release()
                self._set_status(
                    slug,
                    "failed",
                    error=self._scrub_paths(
                        f"cannot start the run: {type(exc).__name__}: {exc}"
                    ),
                )
                continue
            with self._lock:
                self._inflight[slug] = future
            future.add_done_callback(
                lambda fut, slug=slug: self._finalize(slug, fut)
            )

    def _acquire_slot(self) -> bool:
        """Wait for a free worker slot; False when shutdown intervened.
        Polled rather than blocking so Ctrl-C is never stuck behind a slot
        that will not free until a long run finishes."""
        while not self._abort.is_set():
            if self._slots.acquire(timeout=0.05):
                return True
        return False

    def _drop_queued(self, slug: str, error: str) -> None:
        """Take a run off the queue without running it (shutdown / abort).
        One lock hold, for the same reason admission is one hold: never a
        queued row without a ``queue_position``."""
        with self._lock:
            cancelled = slug in self._cancelled
            self._cancelled.discard(slug)
            if slug in self._queued:
                self._queued.remove(slug)
            if not cancelled:
                self._set_status_locked(slug, "failed", error=error)

    def _finalize(self, slug: str, future: Future) -> None:
        """Record one finished run's terminal status (runs in the pool's
        completion thread — must never raise)."""
        outcome = "failed"
        error: str | None = None
        try:
            try:
                result = future.result()
            except CancelledError:
                outcome, error = "aborted", None
            except BrokenProcessPool:
                # The worker was killed from outside (the OOM killer is the
                # usual cause).  Say that, and say the server recovered —
                # leaking "BrokenProcessPool: A child process terminated
                # abruptly" told the operator nothing they could act on.
                outcome, error = "failed", _POOL_DIED_MSG
                self._drop_broken_pool()
            except BaseException as exc:  # noqa: BLE001 - pool/worker death
                outcome = "failed"
                error = self._scrub_paths(f"{type(exc).__name__}: {exc}")
            else:
                if isinstance(result, Mapping):
                    outcome = str(result.get("outcome") or "failed")
                    raw = result.get("error")
                    error = self._scrub_paths(str(raw)) if raw else None
                else:  # pragma: no cover - _child_run always returns a dict
                    outcome, error = "failed", "worker returned no result"
            with self._lock:
                cancelled = slug in self._cancel_requested
                self._cancel_requested.discard(slug)
                self._inflight.pop(slug, None)
                aborting = self._abort.is_set()
            if outcome == "done":
                self._set_status(slug, "done")
            elif cancelled:
                self._set_status(slug, "failed", error="cancelled by request")
            elif outcome == "aborted" or aborting:
                self._set_status(
                    slug, "failed", error="aborted at server shutdown"
                )
            else:
                self._set_status(slug, "failed", error=error or "run failed")
        except BaseException as exc:  # noqa: BLE001 - must not escape
            # A raising callback would leave the run stuck in ``running``
            # with nothing recorded, so say so on the operator's terminal
            # (the one place detail is allowed) instead of vanishing.
            try:
                sys.stderr.write(
                    f"fleetsim serve: failed to record the outcome of"
                    f" {slug}: {type(exc).__name__}: {exc}\n"
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._slots.release()

    # -- shutdown -----------------------------------------------------------

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop accepting, cancel queued runs, ask in-flight runs to stop,
        and release the workspace.

        Bounded end to end: queued runs are marked ``failed`` immediately;
        each in-flight run gets its ``cancel.flag`` and a short grace to
        stop at its next metrics flush; then the worker processes are
        terminated and anything still unfinished is marked ``failed``.  So
        Ctrl-C returns in about a second even mid-simulation."""
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
            self._drop_queued(slug, "server shut down before the run started")
        with self._lock:
            inflight = list(self._inflight)
        for slug in inflight:
            self._request_child_stop(slug)
        self._queue.put(_STOP)
        if self._worker is not None:
            self._worker.join(timeout=min(timeout, 5.0))
        deadline = time.monotonic() + min(timeout, _ABORT_GRACE_S)
        while time.monotonic() < deadline:
            with self._lock:
                if not self._inflight:
                    break
            time.sleep(0.02)
        self._kill_pool()
        with self._lock:
            stragglers = list(self._inflight)
            self._inflight.clear()
        for slug in stragglers:
            self._set_status(slug, "failed", error="aborted at server shutdown")
        self._release_workspace_lock()

    def _kill_pool(self) -> None:
        """Shut the pool down without waiting, then terminate any worker
        still executing.  ``_processes`` is a CPython internal, so every
        step is defensive: the contract this keeps is "shutdown returns",
        and a leaked worker would break it."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is None:
            return
        procs = list(getattr(pool, "_processes", {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
            except (OSError, ValueError, AttributeError):  # pragma: no cover
                pass
        for proc in procs:
            try:
                proc.join(timeout=1.0)
            except (OSError, ValueError, AttributeError):  # pragma: no cover
                pass
