"""Parameter sweeps for ``fleetsim serve`` (v0.8).

A *sweep* is one base scenario plus a grid of dotted-path overrides: the
cartesian product is expanded into one ordinary run per cell, all of them
submitted to the same FIFO queue the parallel worker pool drains.  Nothing
about a cell's execution is special — the comparison value is entirely in
the METADATA that ties the cells together, so every existing run endpoint
(model, report, live, cancel, delete) works on a sweep cell unchanged.

REQUEST
-------
``POST /api/sweeps`` body::

    {yaml: str, title?: str,
     grid: {"<dotted.path>": [v1, v2, ...], ...},
     seeds?: [int, ...]}

``grid`` values are JSON (int / float / str / bool / list / mapping) and
are applied through the SAME override machinery the CLI uses
(:func:`fleetsim.api.apply_overrides`), so ``sim.horizon: "2h"`` in a
sweep means exactly what ``-o sim.horizon=2h`` means on the command line.
``seeds`` is sugar for one more axis, ``sim.seed``, appended LAST — the
axis order is the request's key order, which fixes the cell order and so
the run ids' order.

EXPANSION RULES (all pinned, all tested)
----------------------------------------
- The product is capped at :data:`MAX_SWEEP_RUNS` cells, each axis at
  :data:`MAX_SWEEP_AXIS` values, and the NUMBER of axes at
  :data:`MAX_SWEEP_AXES`; a bigger request is rejected before any work.
  The axis-count cap is not redundant with the product cap: 50,000 axes
  of one value each is a product of 1, and its generated cell label was
  a 1.55 MB run title that then rode every 3-second ``GET /api/runs``
  poll forever.  Labels are additionally truncated
  (:data:`MAX_LABEL_CHARS`) — the exact cell always lives in
  ``meta.json`` under ``sweep_cell``, so the title never has to carry it.
- EVERY AXIS MUST MOVE THE SCENARIO.  ``sim.horizonn`` (one typo) is
  accepted by the document — the schema checks unknown keys only at the
  top level — and expands into N BYTE-IDENTICAL runs that the sweep board
  charts and the compare view diffs as if they were an experiment.  So
  after rendering, each axis is probed: if every one of its values loads
  into the identical :class:`~fleetsim.config.Scenario`, the sweep is
  refused and the message names the nearest real path.
- EVERY cell is fully validated (parse + schema + feasibility, the same
  gate ``POST /api/runs`` applies) BEFORE a single run directory is
  created: a sweep is all-or-nothing, because a half-created sweep is
  worse than no sweep — the user would have to reason about which cells
  are missing while the rest burn CPU.  The expensive half of that gate
  (building the fleet) is memoized by :class:`RunManager` across cells,
  so a 64-cell seed sweep over a 262,144-node fleet builds it once
  rather than 64 times — which was 89 s of one HTTP thread.
- Each cell's ``scenario.yaml`` is the base document with that cell's
  overrides applied, RE-SERIALIZED (``yaml.safe_dump``, author key order
  preserved).  It is therefore a complete, self-describing scenario —
  ``fleetsim viz`` reads it back for seed/round metadata — at the cost of
  dropping the base file's comments.  The exact cell values also live in
  ``meta.json`` under ``sweep_cell``, so nothing depends on re-parsing.

STORAGE
-------
One JSON file per sweep under ``workspace/.sweeps/<sweep_id>.json``
(a dot-directory, so :meth:`RunManager.list_runs` never mistakes it for a
run).  Sweep ids are server-generated on the same pattern as run slugs
(``sweep-<UTCdate>-<UTCtime>-<seq>-<rand>``) and re-validated on every
request with the same separator/dot-name rejection plus a
``resolve()``/``is_relative_to`` containment check.

``DELETE /api/sweeps/{id}`` dequeues only the cells that are still
``queued`` — running and finished cells are history, exactly as for a
single run — and rewrites the record without them.  A sweep whose every
cell was still queued disappears entirely (nothing of it ever existed on
disk); a partially-executed one keeps its record and its remaining cells.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
import re
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..config import ScenarioError, load_scenario
from ..api import apply_overrides
from .runs import RunManager, _atomic_write_text, _summary_headline

__all__ = [
    "MAX_LABEL_CHARS",
    "MAX_SWEEP_AXES",
    "MAX_SWEEP_AXIS",
    "MAX_SWEEP_RUNS",
    "SweepManager",
    "expand_grid",
    "grid_size",
]

#: Hard cap on cells in ONE sweep.  64 keeps the queue, the workspace, and
#: the compare UI legible, and bounds the all-or-nothing validation pass.
MAX_SWEEP_RUNS = 64

#: Hard cap on values per axis (a 64-value axis is already the whole cap;
#: this only makes the error message specific).
MAX_SWEEP_AXIS = 64

#: Hard cap on the NUMBER of axes.  A 16-axis sweep is already past what
#: any chart can show; the cap exists because axis count is the one grid
#: dimension the cell-product cap does not bound (50,000 one-value axes
#: have a product of 1) and it lands in a title on every run listing.
MAX_SWEEP_AXES = 16

#: Longest generated cell label carried in a run title / error message.
MAX_LABEL_CHARS = 160

#: A dotted override path: YAML-ish identifier segments, dot separated.
#: Deliberately narrow — the paths are document keys, not expressions.
_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z_][A-Za-z0-9_-]*)*$")

_SWEEP_ID_RE = re.compile(r"^sweep-\d{8}-\d{6}-\d{3}-[a-z0-9]{4}$")

#: Most cell-validation errors reported in one 400 (a 64-cell sweep with a
#: systematic mistake would otherwise return the same message 64 times).
_MAX_REPORTED_ERRORS = 20


def _yaml_scalar(value: Any) -> str:
    """A JSON value as the YAML text :func:`apply_overrides` parses.

    Round-tripping through YAML rather than assigning the value straight
    into the document is deliberate: a sweep cell must be
    indistinguishable from the same edit made by hand or by ``-o`` on the
    command line, and that is defined by the override machinery.
    """
    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True)


def grid_size(
    grid: Mapping[str, Sequence[Any]], seeds: Sequence[int] | None = None
) -> int:
    """Cell count of the product WITHOUT materializing it.

    The cap has to be enforced on this number, not on an expanded list: a
    request with 20 axes of 64 values is a few hundred bytes of JSON and
    ``64 ** 20`` cells, so expanding first would be the denial of service.
    """
    sizes = [len(values) for values in grid.values()]
    if seeds:
        sizes.append(len(seeds))
    return math.prod(sizes) if sizes else 1


def expand_grid(
    grid: Mapping[str, Sequence[Any]], seeds: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """The grid's cartesian product as a list of ``{path: value}`` cells.

    Axis order is the mapping's iteration order (JSON object order as sent)
    with ``sim.seed`` from ``seeds`` appended last; the LAST axis varies
    fastest, so cells read like nested loops in the order they were
    written.  Returns ``[{}]`` for an empty grid (one cell = the base
    scenario) — callers reject that earlier.
    """
    axes: list[tuple[str, list[Any]]] = [
        (path, list(values)) for path, values in grid.items()
    ]
    if seeds:
        axes.append(("sim.seed", [int(s) for s in seeds]))
    cells: list[dict[str, Any]] = []
    for combo in itertools.product(*[values for _, values in axes]):
        cells.append({path: value for (path, _), value in zip(axes, combo)})
    return cells


def _describe(cell: Mapping[str, Any]) -> str:
    """A cell as a short, stable label for error messages and the UI.

    BOUNDED at :data:`MAX_LABEL_CHARS`: this string becomes a run title,
    and a run title is re-sent on every 3-second ``GET /api/runs`` poll of
    every open tab, forever.  Truncation is lossless in the sense that
    matters — the exact cell is in ``meta.json`` under ``sweep_cell``, and
    the sweep record holds the whole grid.
    """
    text = ", ".join(
        f"{k}={json.dumps(v, sort_keys=True)}" for k, v in cell.items()
    )
    if len(text) <= MAX_LABEL_CHARS:
        return text
    return text[: MAX_LABEL_CHARS - 1].rstrip(", ") + "…"


class SweepManager:
    """Sweep records beside a :class:`RunManager`'s workspace.

    Holds no execution state of its own — cells are ordinary runs — so the
    only lock is around the per-sweep record files.
    """

    def __init__(self, runs: RunManager):
        self.runs = runs
        self.dir = runs.workspace / ".sweeps"
        self._lock = threading.Lock()
        self._seq = 0
        self._rng = random.Random()  # seeded once at boot (OS entropy)

    # -- ids and paths ------------------------------------------------------

    def _new_id(self) -> str:
        while True:
            with self._lock:
                self._seq += 1
                seq = self._seq
                suffix = "".join(
                    self._rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                    for _ in range(4)
                )
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            sweep_id = f"sweep-{stamp}-{seq % 1000:03d}-{suffix}"
            if not self._path(sweep_id, checked=True).exists():
                return sweep_id

    def _path(self, sweep_id: str, *, checked: bool = False) -> Path:
        """The record path for ``sweep_id``.

        ``checked=True`` skips the client-input gate (server-generated ids
        only).  Otherwise the id must match the pinned slug shape AND the
        joined path must resolve inside the sweeps directory — the same two
        independent gates :meth:`RunManager.resolve_dir` applies.
        """
        if not checked:
            if not _SWEEP_ID_RE.match(sweep_id):
                raise ValueError("malformed sweep id")
        path = self.dir / f"{sweep_id}.json"
        resolved_root = self.dir.resolve()
        try:
            if not path.resolve().is_relative_to(resolved_root):
                raise ValueError("sweep id escapes the workspace")
        except (OSError, ValueError) as exc:
            raise ValueError("malformed sweep id") from exc
        return path

    def _read(self, sweep_id: str) -> dict[str, Any] | None:
        try:
            path = self._path(sweep_id)
        except ValueError:
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return doc if isinstance(doc, dict) else None

    # -- create -------------------------------------------------------------

    def create(
        self,
        text: str,
        grid: Mapping[str, Any],
        seeds: Sequence[Any] | None = None,
        title: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Expand, validate every cell, then create one run per cell.

        Returns ``(200, {sweep_id, run_ids, n_runs})`` or an error tuple:
        ``400 {ok: false, errors: [...]}`` for a malformed grid or any
        invalid expansion (nothing is created), ``413`` when the product
        exceeds :data:`MAX_SWEEP_RUNS`.  Raises ``RuntimeError`` after the
        manager has shut down (the HTTP layer answers 503).
        """
        errors = _grid_errors(grid, seeds)
        if errors:
            return 400, {"ok": False, "errors": errors}
        seed_axis = [int(s) for s in (seeds or [])]
        size = grid_size(grid, seed_axis)
        if size > MAX_SWEEP_RUNS:
            return 413, {
                "ok": False,
                "errors": [
                    f"sweep expands to {size} runs, more than the"
                    f" {MAX_SWEEP_RUNS}-run cap — shrink a grid axis (or"
                    f" split the sweep) and submit again"
                ],
            }
        cells = expand_grid(grid, seed_axis)

        try:
            base = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return 400, {"ok": False, "errors": [f"invalid YAML: {exc}"]}
        if not isinstance(base, Mapping):
            return 400, {
                "ok": False,
                "errors": [
                    f"scenario must be a mapping, got {type(base).__name__}"
                ],
            }

        inert = _inert_axis_errors(base, grid, seed_axis)
        if inert:
            return 400, {"ok": False, "errors": inert}

        # ALL-OR-NOTHING: render + validate every cell first.
        rendered: list[tuple[dict[str, Any], str]] = []
        cell_errors: list[str] = []
        for cell in cells:
            doc = copy.deepcopy(dict(base))
            try:
                apply_overrides(doc, {k: _yaml_scalar(v) for k, v in cell.items()})
            except Exception as exc:  # noqa: BLE001 - ScenarioError and friends
                cell_errors.append(f"cell [{_describe(cell)}]: {exc}")
                continue
            cell_yaml = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
            for err in self.runs.validate_text(cell_yaml):
                cell_errors.append(f"cell [{_describe(cell)}]: {err}")
            rendered.append((cell, cell_yaml))
        if cell_errors:
            shown = cell_errors[:_MAX_REPORTED_ERRORS]
            if len(cell_errors) > len(shown):
                shown.append(
                    f"... and {len(cell_errors) - len(shown)} more cell"
                    f" error(s) — no runs were created"
                )
            return 400, {"ok": False, "errors": shown}

        sweep_id = self._new_id()
        base_title = (title or "").strip() or sweep_id
        run_ids: list[str] = []
        cell_rows: list[dict[str, Any]] = []
        try:
            for cell, cell_yaml in rendered:
                label = _describe(cell) or "base"
                run_id = self.runs.submit(
                    cell_yaml,
                    f"{base_title} — {label}",
                    meta_extra={"sweep_id": sweep_id, "sweep_cell": cell},
                )
                run_ids.append(run_id)
                cell_rows.append({"run_id": run_id, "cell": cell})
        except Exception:  # noqa: BLE001 - roll back a partial sweep
            for run_id in run_ids:
                self.runs.delete_queued(run_id)
            raise
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self._path(sweep_id, checked=True),
            json.dumps(
                {
                    "sweep_id": sweep_id,
                    "title": base_title,
                    "created_unix": int(time.time()),
                    "grid": {k: list(v) for k, v in grid.items()},
                    "seeds": [int(s) for s in (seeds or [])],
                    "cells": cell_rows,
                },
                sort_keys=True,
            )
            + "\n",
        )
        return 200, {
            "sweep_id": sweep_id,
            "run_ids": run_ids,
            "n_runs": len(run_ids),
        }

    # -- read ---------------------------------------------------------------

    def list_sweeps(self) -> list[dict[str, Any]]:
        """``[{sweep_id, title, created, n_runs, n_done, grid, seeds}]``,
        newest first.  Unreadable records drop out silently."""
        rows: list[dict[str, Any]] = []
        try:
            paths = sorted(self.dir.glob("sweep-*.json"))
        except OSError:
            return []
        for path in paths:
            doc = self._read(path.stem)
            if doc is None:
                continue
            cells = doc.get("cells") or []
            statuses = [self._cell_status(c) for c in cells]
            rows.append(
                {
                    "sweep_id": doc.get("sweep_id") or path.stem,
                    "title": doc.get("title") or path.stem,
                    "created": doc.get("created_unix"),
                    "n_runs": len(cells),
                    "n_done": sum(1 for s in statuses if s == "done"),
                    "n_failed": sum(1 for s in statuses if s == "failed"),
                    "grid": doc.get("grid") or {},
                    "seeds": doc.get("seeds") or [],
                }
            )
        rows.sort(key=lambda r: (-(r["created"] or 0), r["sweep_id"]))
        return rows

    def _cell_status(self, cell: Mapping[str, Any]) -> str:
        info = self.runs.get_run(str(cell.get("run_id") or ""))
        return "missing" if info is None else str(info["status"])

    def get_sweep(self, sweep_id: str) -> dict[str, Any] | None:
        """The sweep plus one row per cell::

            {sweep_id, title, created, grid, seeds, n_runs, n_done,
             runs: [{id, title, status, created, queue_position, cell,
                     headline, error?}]}

        ``headline`` is the run listing's three-number headline
        (``{occupancy, goodput, jobs_finished}``) for finished cells and
        ``null`` otherwise, so a compare table needs exactly one request.
        ``None`` when the id is unknown or malformed.
        """
        doc = self._read(sweep_id)
        if doc is None:
            return None
        rows: list[dict[str, Any]] = []
        for cell in doc.get("cells") or []:
            run_id = str(cell.get("run_id") or "")
            info = self.runs.get_run(run_id)
            if info is None:
                rows.append(
                    {
                        "id": run_id,
                        "title": None,
                        "status": "missing",
                        "created": None,
                        "queue_position": None,
                        "cell": cell.get("cell") or {},
                        "headline": None,
                    }
                )
                continue
            summary = info.get("summary")
            rows.append(
                {
                    "id": run_id,
                    "title": info["title"],
                    "status": info["status"],
                    "created": info["created"],
                    "queue_position": info.get("queue_position"),
                    "cell": cell.get("cell") or {},
                    "headline": (
                        _summary_headline(summary)
                        if isinstance(summary, Mapping)
                        else None
                    ),
                    **({"error": info["error"]} if info.get("error") else {}),
                }
            )
        _renumber_queued(rows)
        return {
            "sweep_id": doc.get("sweep_id") or sweep_id,
            "title": doc.get("title") or sweep_id,
            "created": doc.get("created_unix"),
            "grid": doc.get("grid") or {},
            "seeds": doc.get("seeds") or [],
            "n_runs": len(rows),
            "n_done": sum(1 for r in rows if r["status"] == "done"),
            "runs": rows,
        }

    # -- delete -------------------------------------------------------------

    def delete_sweep(self, sweep_id: str) -> tuple[int, dict[str, Any] | str]:
        """Dequeue every still-``queued`` cell; keep everything else.

        ``(200, {ok, dequeued, kept, removed_record})`` — ``removed_record``
        is true when no cell survived and the record itself was deleted.
        ``(404, msg)`` for an unknown id.
        """
        doc = self._read(sweep_id)
        if doc is None:
            return 404, "no such sweep"
        dequeued: list[str] = []
        kept: list[dict[str, Any]] = []
        for cell in doc.get("cells") or []:
            run_id = str(cell.get("run_id") or "")
            code, _ = self.runs.delete_queued(run_id)
            if code == 200:
                dequeued.append(run_id)
            else:
                kept.append(cell)
        path = self._path(sweep_id)
        removed = False
        if kept:
            doc["cells"] = kept
            _atomic_write_text(path, json.dumps(doc, sort_keys=True) + "\n")
        else:
            try:
                path.unlink()
                removed = True
            except OSError:
                pass
        return 200, {
            "ok": True,
            "dequeued": dequeued,
            "kept": [str(c.get("run_id") or "") for c in kept],
            "removed_record": removed,
        }


def _renumber_queued(rows: list[dict[str, Any]]) -> None:
    """Make ``queue_position`` contiguous ``1..n`` over the rows observed
    as queued, in place.

    The same fix ``RunManager.list_runs`` applies, for the same reason and
    now on the surface where a user actually WATCHES a queue drain: each
    row's status and position come from a separate ``get_run`` call, so
    the dispatcher admitting a cell mid-scan leaves the cells behind it
    carrying positions that no longer start at 1.  "#2, #3, #4 and no #1"
    is not a state the board should ever render.
    """
    queued = [r for r in rows if r.get("status") == "queued"]
    queued.sort(
        key=lambda r: (
            r.get("queue_position") is None,
            r.get("queue_position") or 0,
            str(r.get("id") or ""),
        )
    )
    for pos, row in enumerate(queued, start=1):
        row["queue_position"] = pos


def _load_or_none(doc: Mapping[str, Any]) -> str | None:
    """A loaded scenario's ``repr`` (its total, field-by-field identity),
    or ``None`` when the document does not load at all.

    ``repr`` rather than equality because the config dataclasses are not
    all hashable and some are not ``eq``-comparable across nesting; every
    field is in the repr, so two documents with the same repr produced
    the same scenario — which is the question here.
    """
    try:
        return repr(load_scenario(dict(doc), strict=False))
    except (ScenarioError, Exception):  # noqa: BLE001 - a probe never raises
        return None


def _nearest_path(path: str, known: Sequence[str]) -> str | None:
    """The closest real dotted path to a typo, or ``None``.

    Deliberately cheap and conservative: same prefix, and a last segment
    within one edit (a dropped, doubled or transposed character) — enough
    for ``sim.horizonn`` -> ``sim.horizon`` without inventing suggestions
    for paths the user never meant.
    """
    head, _, tail = path.rpartition(".")
    best: str | None = None
    for cand in known:
        chead, _, ctail = cand.rpartition(".")
        if chead != head or ctail == tail:
            continue
        if abs(len(ctail) - len(tail)) > 1:
            continue
        if ctail in tail or tail in ctail or sorted(ctail) == sorted(tail):
            if best is None or len(cand) < len(best):
                best = cand
    return best


def _doc_paths(doc: Any, prefix: str = "") -> list[str]:
    """Every dotted path the BASE document already spells (used only to
    suggest a correction for a grid key that changes nothing)."""
    out: list[str] = []
    if not isinstance(doc, Mapping):
        return out
    for key, value in doc.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        out.append(path)
        out.extend(_doc_paths(value, path))
    return out


def _inert_axis_errors(
    base: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    seeds: Sequence[int],
) -> list[str]:
    """Axes that move NOTHING the scenario schema reads.

    A dotted path below the top level is never checked against the schema
    (``_build_scenario`` rejects unknown keys only at depth 0), so one
    typo — ``sim.horizonn``, ``scheduler.paramz``, ``workload.clases`` —
    silently produces N identical runs that the sweep board then presents
    as an experiment.  Verified per AXIS, not per cell: an axis is inert
    when EVERY one of its values loads into the identical scenario, so an
    axis that happens to repeat the base value among others is fine.
    """
    axes: list[tuple[str, list[Any]]] = [
        (path, list(values)) for path, values in grid.items()
    ]
    if seeds:
        axes.append(("sim.seed", [int(s) for s in seeds]))
    baseline = _load_or_none(base)
    if baseline is None:
        return []  # the base itself is broken; cell validation reports that
    known = _doc_paths(base)
    errors: list[str] = []
    for path, values in axes:
        moved = False
        for value in values:
            doc = copy.deepcopy(dict(base))
            try:
                apply_overrides(doc, {path: _yaml_scalar(value)})
            except Exception:  # noqa: BLE001 - cell validation reports it
                moved = True
                break
            loaded = _load_or_none(doc)
            if loaded is None or loaded != baseline:
                moved = True
                break
        if moved:
            continue
        hint = _nearest_path(path, known)
        errors.append(
            f"grid key {path!r} changes nothing the scenario schema reads:"
            f" every value on this axis loads into an identical scenario,"
            f" so the cells would be byte-identical runs presented as a"
            f" comparison"
            + (f" (did you mean {hint!r}?)" if hint else "")
            + ". Fix the path, or drop the axis."
        )
    return errors


def _grid_errors(
    grid: Mapping[str, Any], seeds: Sequence[Any] | None
) -> list[str]:
    """Shape errors for a sweep request's ``grid`` / ``seeds`` ([] = ok)."""
    errors: list[str] = []
    if not isinstance(grid, Mapping):
        return ["'grid' must be an object of {dotted.path: [values]}"]
    if not grid and not seeds:
        return [
            "'grid' is empty — give at least one axis (or a 'seeds' list)"
        ]
    if len(grid) > MAX_SWEEP_AXES:
        return [
            f"'grid' has {len(grid)} axes, more than the"
            f" {MAX_SWEEP_AXES}-axis cap — a sweep with that many axes"
            f" cannot be read on any chart, and its generated cell labels"
            f" would ride every run listing"
        ]
    for path, values in grid.items():
        if not isinstance(path, str) or not _PATH_RE.match(path):
            errors.append(
                f"grid key {path!r} is not a dotted scenario path"
                f" (e.g. 'sim.seed' or 'workload.classes.eval.rate_per_hour')"
            )
            continue
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            errors.append(f"grid[{path!r}] must be a non-empty list of values")
            continue
        if not values:
            errors.append(f"grid[{path!r}] must be a non-empty list of values")
            continue
        if len(values) > MAX_SWEEP_AXIS:
            errors.append(
                f"grid[{path!r}] has {len(values)} values, more than the"
                f" {MAX_SWEEP_AXIS}-value axis cap"
            )
    if seeds is not None:
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
            errors.append("'seeds' must be a list of integers")
        elif not seeds:
            errors.append("'seeds' must be a non-empty list of integers")
        else:
            for seed in seeds:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    errors.append(f"'seeds' entry {seed!r} is not an integer")
                    break
    return errors
