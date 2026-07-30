"""Viz data pipeline: run outputs in, one JSON-serializable model out.

:func:`build_viz_model` reads a ``fleetsim run`` output directory —
``summary.json``, ``jobs.parquet``, ``timeseries.parquet`` and the
optional ``stints.parquet`` — and produces the replay model the v0.3
HTML app renders.  :func:`to_json` serializes it compactly.

PINNED MODEL SCHEMA (the app phase codes against THIS)
------------------------------------------------------
::

    { meta: {title, out_dir, horizon_us, round_us, seed, scenario_name,
             fleetsim_version, generated_unix_ms: null, notes: [str]},
      capabilities: {map: bool, compare: bool},
      palette: {class_name -> color, state -> color},
      fleet: {map_level: str|null,
              clusters: [{id, chips, domains: [{id, short, chips}]}]},
      frames: {t_us: int[], occupancy: f[], allocation: f[],
               goodput_to_date: f[], pending_by_class: {cls: int[]},
               preemptions_delta: int[], failures_delta: int[]},
      stints: {job_id: str[], class_name: str[], tier: str[],
               domain_idx: int[], chips: int[], t0_us: int[],
               t1_us: int[], end_reason: str[]},
      gantt: [{id, class_name, chips, submit_us, start_us, end_us,
               status, n_preemptions, n_restarts, domains_spanned}],
      cdfs: {queue_wait_s: {cls: [[x, p], ...]}, jct_s: {...}},
      events: [{t_us, kind, label, magnitude}],
      summary_cards: [{label, value, sub}],
      compare: null | {label_a, label_b, frames_b: {...same as frames},
                       summary_cards_b: [...]} }

Field-by-field rules (every value traceable to the run outputs; where
summary.json and a recomputation could both answer, summary.json wins):

meta
    ``horizon_us`` from ``summary.json``.  ``round_us``/``seed``/
    ``scenario_name`` come from a scenario file found INSIDE the output
    directory (``scenario.yaml|yml``/``config.yaml|yml``, first that
    parses); ``scenario_name`` is the file's stem, nulled when the stem
    is the generic ``scenario``/``config`` (the standard copy names —
    a "scenario scenario" metaline conveys nothing).  Without a
    scenario file, ``seed``/``scenario_name`` are null and ``round_us``
    falls back to the most common gap between consecutive
    ``timeseries`` samples (ties -> smallest gap; null if < 2 samples).
    ``generated_unix_ms`` is ALWAYS null (a model is a pure function of
    its inputs — no wall clock).  ``notes`` lists every degradation /
    reconstruction decision taken (which chips source, truncations, …).

capabilities
    ``map`` is true iff ``stints.parquet`` exists AND at least one map
    domain could be reconstructed; ``compare`` iff ``compare_dir`` was
    given.  Without stints the model still carries the fleet-level
    replay (``frames``), with ``fleet.clusters == []`` and empty
    ``stints`` arrays — the app degrades to the no-map layout.

palette
    The five canonical class colors, then every observed class label
    (see "class labels" below) mapped to its class bucket's color —
    labels sharing an already-claimed bucket take that bucket's pinned
    shade variants (``_BUCKET_VARIANTS``, validated for CVD and
    normal-vision separation) so two classes never render in one color
    — then the state colors.  One flat dict so every panel colors by
    direct ``palette[label]`` lookup.

class labels (used by ``pending_by_class``, ``stints.class_name``,
``gantt.class_name`` and the ``cdfs`` keys)
    A job's label is its workload source class (``source_class`` /
    stints ``class_name``), falling back to the JobClass enum name for
    hand-built/trace jobs; always lowercased.  A label's palette bucket:
    tier ``BEST_EFFORT`` -> ``best_effort``, else by JobClass
    (``INFER_REPLICA`` -> ``inference``); a label seen with several
    buckets takes the majority (ties -> canonical palette order).

fleet
    Domains observed in ``stints.parquet`` grouped into clusters by
    their id prefix (ids are ``metro/cluster/...`` paths; the cluster is
    the first two segments).  ``map_level``: the ``map_level_hint``
    argument if given, else the modal level name inferred by stripping
    the trailing index digits off each domain id's last segment (depth 1
    -> ``metro``, depth 2 -> ``cluster``); a hint that disagrees with
    the inferred level, or a level not among the ``fragmentation``
    level names in summary.json, adds a note (the blocks are always
    the recorded domains — a hint only labels them).
    Domain ``chips``: when a scenario file was found and its built fleet
    covers every observed domain at ``map_level``, the exact configured
    chips (and the full domain list, including domains that never hosted
    a stint); otherwise the MAX CONCURRENT chips observed in stints per
    domain (a lower bound on capacity) over observed domains only.
    ``meta.notes`` states which source was used.  Clusters sort by id;
    domains sort naturally (``pod2`` before ``pod10``); ``short`` is the
    last path segment.

frames
    Downsampled from ``timeseries.parquet`` to <= ``max_frames`` frames.
    Rule: samples are split into <= max_frames contiguous buckets of
    near-equal size (bucket i covers samples [floor(i*n/F),
    floor((i+1)*n/F))); per bucket, ``t_us`` is the LAST sample's t_us,
    value series are the bucket MEAN, delta series the bucket SUM.
    - ``occupancy``: per-sample allocated_chips / healthy_chips (null
      where healthy is 0), bucket-mean.
    - ``allocation``: per-sample allocated_chips / total fleet chips,
      where total chips = allocated_chip_hours.total * 3600 /
      (allocation_rate * duration_s) from summary.json's full scope
      (exact; falls back to max healthy_chips — noted — when the run
      never allocated).
    - ``goodput_to_date``: the timeseries column, bucket-mean.
    - ``pending_by_class``: reconstructed from jobs.parquet — a job is
      pending at sample time t iff submit_t <= t and it has neither
      started (first_start > t or never) nor ended (end > t or still
      running).  First-wait only: re-queued time after a preemption is
      NOT re-counted (the collector's ``pending_jobs`` total counts
      re-queues, so the per-class sum may undercount it; noted).
      Bucket-mean rounded to int.
    - ``preemptions_delta`` / ``failures_delta``: per-sample diffs of
      ``cum_preemptions`` / ``cum_node_failures`` (first sample diffs
      against 0), bucket-sum.  Failure kills of jobs are part of
      neither (they are in ``cum_failure_kills``).

stints
    Columnar copy of stints.parquet rows sorted by (t0_us, job_id,
    domain, t1_us) — primary order t0_us as pinned.  ``class_name`` and
    ``tier`` lowercased; ``domain_idx`` indexes the flattened
    ``fleet.clusters[*].domains`` list in model order.  Empty arrays
    when there is no stints.parquet (or no reconstructable domain).

gantt
    From jobs.parquet: the top ``max_gantt_jobs`` jobs by chips *
    (end - start) — end falling back to the horizon for still-running
    jobs, score 0 for never-started jobs — PLUS every job with >= 4096
    chips (even never-started ones).  Entries sorted by (submit_us, id).
    ``start_us`` null = never started; ``end_us`` null = not terminal at
    the horizon; ``status`` is the jobs.parquet status lowercased
    verbatim — terminal ``completed|failed|canceled|timeout|node_fail``,
    non-terminal ``pending|admitted|running`` (the last observed
    state).  ``domains_spanned`` null = never placed.

cdfs
    Empirical CDFs from jobs.parquet, per class label: queue_wait_s
    over jobs that started, jct_s over COMPLETED jobs — both excluding
    tier BEST_EFFORT (same exclusion summary.json applies; a saturated
    closed loop has no meaningful wait).  Points are [value, p] with
    p = (i+1)/n over ascending values; when n > 200 the curve keeps 200
    evenly spaced ranks (always including the first and last).

events
    Sorted by (t_us, kind, label); ``magnitude`` is an int.
    - ``preemption_wave``: raw (undownsampled) timeseries samples whose
      preemption delta >= max(10, p99 of all raw deltas) (p99 by
      numpy's linear interpolation).
    - ``failure``: raw samples with a node-failure delta > 0; if more
      than 300, the 300 largest (magnitude desc, then time asc) are
      kept and a note is added.
    - ``frontier_submit`` / ``frontier_start``: submit / first-start
      times of jobs with >= 32768 chips.

summary_cards
    Pulled from summary.json ONLY: occupancy (window), goodput
    (window), jobs finished (full), preemptions/min (window, total),
    then p50/p99 queue wait for the top-3 source classes by started-job
    count in the window scope (falling back to the full scope when the
    window saw none).  ``value``/``sub`` are display strings.

compare
    ``label_a``/``label_b`` are the directory basenames (full given
    paths when the basenames collide).  ``frames_b`` follows the exact
    ``frames`` rules on run B's outputs; ``summary_cards_b`` likewise.
    The map/stints/gantt/cdfs/events always describe run A.

INVARIANTS
----------
- Pure function of the on-disk inputs: identical inputs -> identical
  ``to_json`` bytes (no wall clock, no randomness, deterministic
  iteration order everywhere).
- JSON-safe: only dict/list/str/int/float/bool/None; every float is
  finite (NaN/inf are mapped to null); :func:`to_json` enforces this
  with ``allow_nan=False``.
- UNITS: ``*_us`` int microseconds, ``*_s`` float seconds.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__ as _FLEETSIM_VERSION

__all__ = ["build_viz_model", "to_json"]

# ---------------------------------------------------------------------------
# Shared palette (v0.3 pinned theme — class colors consistent everywhere)
# ---------------------------------------------------------------------------

#: Canonical class-bucket colors, in canonical order (also the tie-break
#: order when one label maps to several buckets).
_CLASS_COLORS: dict[str, str] = {
    "pretrain": "#4c6ef5",
    "finetune": "#12b886",
    "eval": "#fab005",
    "best_effort": "#64748b",
    "inference": "#9775fa",
}

_STATE_COLORS: dict[str, str] = {
    "failed": "#e03131",
    "draining": "#f76707",
    "maintenance": "#845ef7",
    "idle": "rgba(255,255,255,.05)",
}

#: Shade variants used when several observed labels share one bucket, so
#: e.g. a custom ``frontier`` class is distinguishable from ``pretrain``
#: in every panel.  Pinned (not derived at runtime) and validated against
#: the dark panel surface #11151d with the dataviz palette validator:
#: each first variant vs its base passes CVD separation (OKLab dE >= 8
#: under protan/deutan simulation; worst pair 13.1), the normal-vision
#: floor (dE >= 15) and 3:1 mark contrast.  Variants keep the bucket's
#: hue so bucket identity still reads at a glance.
_BUCKET_VARIANTS: dict[str, tuple[str, ...]] = {
    "pretrain": ("#8ba1f9", "#c6d1fc"),
    "finetune": ("#89dcc3", "#0c7857"),
    "eval": ("#af7b04", "#fee7b4"),
    "best_effort": ("#9aa5b4", "#cdd3da"),
    "inference": ("#c1acfc", "#6a49b5"),
}

#: JobClass enum name -> palette bucket (tier BEST_EFFORT wins over these).
_JOBCLASS_BUCKET = {
    "PRETRAIN": "pretrain",
    "FINETUNE": "finetune",
    "EVAL": "eval",
    "INFER_REPLICA": "inference",
}

_FRONTIER_CHIPS = 32768  # frontier event threshold
_BIG_JOB_CHIPS = 4096  # always-in-gantt threshold
_CDF_MAX_POINTS = 200
_MAX_FAILURE_EVENTS = 300
_PREEMPT_WAVE_FLOOR = 10

#: Scenario files probed inside the output directory, in order.
_SCENARIO_NAMES = ("scenario.yaml", "scenario.yml", "config.yaml", "config.yml")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(x: Any) -> float | None:
    """A finite Python float, else None (the model's only float rule)."""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _label(source_class: Any, job_class: Any) -> str:
    """The class label: source class, falling back to the JobClass enum
    name; always lowercased."""
    if source_class is not None and not pd.isna(source_class):
        return str(source_class).lower()
    return str(job_class).lower()


def _natural_key(domain_id: str) -> tuple[str, int, str]:
    """Sort key putting ``pod2`` before ``pod10`` (ids are unpadded)."""
    last = domain_id.rsplit("/", 1)[-1]
    m = re.match(r"^(.*?)(\d+)$", last)
    if m:
        return (m.group(1), int(m.group(2)), domain_id)
    return (last, -1, domain_id)


def _level_of(domain_id: str) -> str:
    """The level name a domain id implies: depth 1 = metro, depth 2 =
    cluster (build_fleet's id grammar), deeper = the last segment with
    its trailing index digits stripped."""
    parts = domain_id.split("/")
    if len(parts) == 1:
        return "metro"
    if len(parts) == 2:
        return "cluster"
    stripped = re.sub(r"\d+$", "", parts[-1])
    return stripped if stripped else parts[-1]


def _fmt_dur(seconds: float | None) -> str:
    """Deterministic humanized duration for summary cards."""
    if seconds is None:
        return "n/a"
    s = float(seconds)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


class _Run:
    """One output directory's parsed artifacts (+ optional scenario)."""

    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir)
        for name in ("summary.json", "jobs.parquet", "timeseries.parquet"):
            if not (self.path / name).is_file():
                raise FileNotFoundError(
                    f"{self.path} is not a fleetsim output directory"
                    f" (missing {name}); pass the -o directory of a"
                    f" previous `fleetsim run`"
                )
        self.summary: dict[str, Any] = json.loads(
            (self.path / "summary.json").read_text(encoding="utf-8")
        )
        self.jobs = pd.read_parquet(self.path / "jobs.parquet")
        self.ts = pd.read_parquet(self.path / "timeseries.parquet")
        stints_path = self.path / "stints.parquet"
        self.stints: pd.DataFrame | None = (
            pd.read_parquet(stints_path) if stints_path.is_file() else None
        )
        self.scenario = None
        self.scenario_file: str | None = None
        self.fleet = None  # built FleetTree, when the scenario allows it
        self._load_scenario()

    def _load_scenario(self) -> None:
        from ..config import load_scenario
        from ..fleet.build import build_fleet

        for name in _SCENARIO_NAMES:
            p = self.path / name
            if not p.is_file():
                continue
            try:
                scenario = load_scenario(p, strict=True)
                fleet = build_fleet(scenario)
            except Exception:
                continue  # a bad copy never degrades the model
            self.scenario = scenario
            self.scenario_file = name
            self.fleet = fleet
            return

    @property
    def horizon_us(self) -> int:
        return int(self.summary["horizon_us"])

    def total_chips(self) -> tuple[int | None, str | None]:
        """(total fleet chips, degradation note or None).

        Derived exactly from summary.json's full scope:
        allocation_rate = allocated_chip_s / (total_chips * duration_s).
        Falls back to max(healthy_chips) when the run never allocated.
        """
        full = self.summary.get("full", {})
        rate = _finite(full.get("allocation_rate"))
        dur = _finite(full.get("duration_s"))
        ch = _finite(_get(full, "allocated_chip_hours", "total"))
        if rate and dur and ch is not None and rate > 0 and dur > 0:
            return int(round(ch * 3600.0 / (rate * dur))), None
        if len(self.ts):
            return (
                int(self.ts["healthy_chips"].max()),
                "total chips: max healthy_chips observed (run never"
                " allocated; summary.json allocation_rate is null)",
            )
        return None, "total chips unknown (empty timeseries)"


# ---------------------------------------------------------------------------
# Class labels and palette
# ---------------------------------------------------------------------------


def _label_buckets(jobs: pd.DataFrame) -> dict[str, str]:
    """Observed class label -> palette bucket (majority vote; ties break
    in canonical palette order)."""
    votes: dict[str, Counter] = {}
    src = jobs["source_class"].tolist()
    jcls = jobs["job_class"].tolist()
    tier = jobs["tier"].tolist()
    for s, jc, t in zip(src, jcls, tier):
        lab = _label(s, jc)
        bucket = (
            "best_effort"
            if str(t) == "BEST_EFFORT"
            else _JOBCLASS_BUCKET.get(str(jc), "best_effort")
        )
        votes.setdefault(lab, Counter())[bucket] += 1
    order = list(_CLASS_COLORS)
    return {
        lab: max(c, key=lambda b: (c[b], -order.index(b)))
        for lab, c in sorted(votes.items())
    }


def _build_palette(
    buckets: dict[str, str], notes: list[str] | None = None
) -> dict[str, str]:
    """One flat label -> color dict.

    Canonical labels keep the pinned bucket colors.  A non-canonical
    label takes its bucket's exact color only while that color is
    unclaimed (bucket's canonical label unobserved); further labels in
    the same bucket take the pinned :data:`_BUCKET_VARIANTS` shades so
    two classes never render identically.  Deterministic: labels are
    assigned in sorted order.
    """
    palette = dict(_CLASS_COLORS)
    taken = {b: (b in buckets) for b in _CLASS_COLORS}
    extra: dict[str, int] = {}
    crowded: list[str] = []
    for lab in sorted(buckets):
        if lab in palette or lab in _STATE_COLORS:
            continue
        bucket = buckets[lab]
        if not taken.get(bucket, True):
            palette[lab] = _CLASS_COLORS[bucket]
            taken[bucket] = True
            continue
        i = extra.get(bucket, 0)
        extra[bucket] = i + 1
        variants = _BUCKET_VARIANTS[bucket]
        if i >= len(variants):
            crowded.append(lab)
        palette[lab] = variants[i % len(variants)]
    if crowded and notes is not None:
        notes.append(
            "palette: bucket shade variants exhausted; "
            + ", ".join(crowded)
            + " repeat colors already used within their bucket"
        )
    palette.update(_STATE_COLORS)
    return palette


# ---------------------------------------------------------------------------
# Fleet reconstruction (map panel)
# ---------------------------------------------------------------------------


def _observed_domain_chips(stints: pd.DataFrame) -> dict[str, int]:
    """Per domain, the max concurrent chips observed (sweep line; at
    equal times releases apply before acquires)."""
    out: dict[str, int] = {}
    for dom, grp in stints.groupby("domain", sort=True):
        events: list[tuple[int, int]] = []
        for t0, t1, chips in zip(
            grp["t0_us"].tolist(), grp["t1_us"].tolist(), grp["chips"].tolist()
        ):
            events.append((int(t0), int(chips)))
            events.append((int(t1), -int(chips)))
        events.sort()  # (t, delta): negative deltas first at equal t
        cur = peak = 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        # A zero-length stint (t0 == t1: killed in its start round) is
        # invisible to the sweep; a domain still held its chips for an
        # instant, so never report less than the largest single stint.
        out[str(dom)] = max(peak, int(grp["chips"].max()))
    return out


def _cluster_of(domain_id: str) -> str:
    parts = domain_id.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _build_fleet_model(
    run: _Run, map_level_hint: str | None, notes: list[str]
) -> tuple[dict[str, Any], dict[str, int]]:
    """The ``fleet`` model block + the domain id -> flat index map."""
    if run.stints is None:
        notes.append(
            "no stints.parquet in the run outputs: fleet map disabled,"
            " fleet-level replay only (set `outputs: {stints: <level>}`"
            " and re-run to enable the map)"
        )
        return {"map_level": None, "clusters": []}, {}

    observed = sorted({str(d) for d in run.stints["domain"].tolist()})
    if not observed:
        notes.append("stints.parquet is empty: fleet map disabled")
        return {"map_level": None, "clusters": []}, {}

    # Level: hint wins; else modal inferred name (ties -> lexicographic).
    counts = Counter(_level_of(d) for d in observed)
    inferred = min(counts, key=lambda lv: (-counts[lv], lv))
    if map_level_hint is not None:
        map_level = str(map_level_hint)
        if map_level != inferred:
            notes.append(
                f"map level {map_level!r} (hint) does not match the"
                f" level inferred from the stint domain ids"
                f" ({inferred!r}); the map blocks are the recorded"
                f" domains regardless"
            )
    else:
        map_level = inferred
    # The `fragmentation` map is keyed by LEVEL name, plus (v0.7) the
    # placement diagnostics of FRAG_NON_LEVEL_KEYS — subtract those before
    # treating the keys as the fleet's level vocabulary.
    from ..metrics.collector import FRAG_NON_LEVEL_KEYS

    frag_levels = sorted(
        set(_get(run.summary, "full", "fragmentation") or {}) - FRAG_NON_LEVEL_KEYS
    )
    if frag_levels and map_level not in frag_levels:
        notes.append(
            f"map level {map_level!r} is not among summary.json"
            f" fragmentation levels {frag_levels}"
        )

    # Chips per domain: exact from a scenario copy when it covers every
    # observed domain, else max-concurrent observed (lower bound).
    chips_by_domain: dict[str, int] | None = None
    if run.fleet is not None and map_level in run.fleet.levels():
        config_domains = list(run.fleet.domains_at(map_level))
        if set(observed) <= set(config_domains):
            chips_by_domain = {
                d: int(run.fleet.total_chips(d)) for d in config_domains
            }
            notes.append(
                f"domain chips: exact, from {run.scenario_file} found in"
                f" the output directory (all configured {map_level}"
                f" domains shown, {len(config_domains)} total)"
            )
    if chips_by_domain is None:
        chips_by_domain = _observed_domain_chips(run.stints)
        notes.append(
            "domain chips: max concurrent chips observed in stints (a"
            " lower bound on capacity; copy the scenario YAML into the"
            " output directory for exact capacities)"
        )

    by_cluster: dict[str, list[str]] = {}
    for d in chips_by_domain:
        by_cluster.setdefault(_cluster_of(d), []).append(d)
    clusters: list[dict[str, Any]] = []
    domain_idx: dict[str, int] = {}
    for cid in sorted(by_cluster):
        doms = sorted(by_cluster[cid], key=_natural_key)
        entries = []
        for d in doms:
            domain_idx[d] = len(domain_idx)
            entries.append(
                {
                    "id": d,
                    "short": d.rsplit("/", 1)[-1],
                    "chips": int(chips_by_domain[d]),
                }
            )
        clusters.append(
            {
                "id": cid,
                "chips": int(sum(e["chips"] for e in entries)),
                "domains": entries,
            }
        )
    return {"map_level": map_level, "clusters": clusters}, domain_idx


# ---------------------------------------------------------------------------
# Frames (downsampled fleet-level replay)
# ---------------------------------------------------------------------------


def _bucket_edges(n: int, max_frames: int) -> list[tuple[int, int]]:
    if n <= max_frames:
        return [(i, i + 1) for i in range(n)]
    return [
        (n * i // max_frames, n * (i + 1) // max_frames)
        for i in range(max_frames)
    ]


def _bucket_mean(arr: np.ndarray, edges: list[tuple[int, int]]) -> list[float | None]:
    out: list[float | None] = []
    for a, b in edges:
        seg = arr[a:b]
        seg = seg[~np.isnan(seg)]
        out.append(_finite(seg.mean()) if seg.size else None)
    return out


def _bucket_mean_int(arr: np.ndarray, edges: list[tuple[int, int]]) -> list[int]:
    return [int(round(float(arr[a:b].mean()))) if b > a else 0 for a, b in edges]


def _bucket_sum_int(arr: np.ndarray, edges: list[tuple[int, int]]) -> list[int]:
    return [int(arr[a:b].sum()) for a, b in edges]


def _pending_counts(
    jobs: pd.DataFrame, t_us: np.ndarray
) -> dict[str, np.ndarray]:
    """Per class label, jobs pending at each sample time (first-wait
    only — see module docstring)."""
    labels = [
        _label(s, jc)
        for s, jc in zip(jobs["source_class"].tolist(), jobs["job_class"].tolist())
    ]
    submit = jobs["submit_t_us"].to_numpy(dtype="float64", na_value=np.nan)
    start = jobs["first_start_t_us"].to_numpy(dtype="float64", na_value=np.nan)
    end = jobs["end_t_us"].to_numpy(dtype="float64", na_value=np.nan)
    # A job stops being pending at its first start or (if it never
    # starts) its terminal event; never -> +inf.
    stop = np.fmin(start, end)
    stop = np.where(np.isnan(stop), np.inf, stop)
    out: dict[str, np.ndarray] = {}
    for lab in sorted(set(labels)):
        mask = np.array([l == lab for l in labels])
        subs = np.sort(submit[mask])
        stops = np.sort(stop[mask])
        out[lab] = np.searchsorted(subs, t_us, side="right") - np.searchsorted(
            stops, t_us, side="right"
        )
    return out


def _build_frames(run: _Run, max_frames: int, notes: list[str]) -> dict[str, Any]:
    ts = run.ts
    n = len(ts)
    t_raw = ts["t_us"].to_numpy(dtype="int64") if n else np.array([], dtype="int64")
    edges = _bucket_edges(n, max_frames)

    total, note = run.total_chips()
    if note:
        notes.append(note)

    alloc = ts["allocated_chips"].to_numpy(dtype="float64") if n else np.array([])
    healthy = ts["healthy_chips"].to_numpy(dtype="float64") if n else np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        occ_raw = np.where(healthy > 0, alloc / np.where(healthy > 0, healthy, 1), np.nan)
        alloc_raw = (
            alloc / total if (total or 0) > 0 else np.full(n, np.nan)
        )
    good_raw = (
        ts["goodput_to_date"].to_numpy(dtype="float64") if n else np.array([])
    )
    preempt_delta = (
        np.diff(ts["cum_preemptions"].to_numpy(dtype="int64"), prepend=0)
        if n
        else np.array([], dtype="int64")
    )
    fail_delta = (
        np.diff(ts["cum_node_failures"].to_numpy(dtype="int64"), prepend=0)
        if n
        else np.array([], dtype="int64")
    )

    pending = _pending_counts(run.jobs, t_raw.astype("float64"))
    notes.append(
        "pending_by_class: reconstructed from jobs.parquet (first wait"
        " only; time re-queued after a preemption is not re-counted, so"
        " the per-class sum can undercount the collector's pending_jobs"
        " total)"
    )

    return {
        "t_us": [int(t_raw[b - 1]) for _, b in edges],
        "occupancy": _bucket_mean(occ_raw, edges),
        "allocation": _bucket_mean(alloc_raw, edges),
        "goodput_to_date": _bucket_mean(good_raw, edges),
        "pending_by_class": {
            lab: _bucket_mean_int(arr.astype("float64"), edges)
            for lab, arr in pending.items()
        },
        "preemptions_delta": _bucket_sum_int(preempt_delta, edges),
        "failures_delta": _bucket_sum_int(fail_delta, edges),
    }


# ---------------------------------------------------------------------------
# Stints (columnar), gantt, CDFs
# ---------------------------------------------------------------------------

_EMPTY_STINTS: dict[str, list[Any]] = {
    "job_id": [],
    "class_name": [],
    "tier": [],
    "domain_idx": [],
    "chips": [],
    "t0_us": [],
    "t1_us": [],
    "end_reason": [],
}


def _build_stints(
    run: _Run, domain_idx: dict[str, int], notes: list[str]
) -> dict[str, list[Any]]:
    if run.stints is None or not len(run.stints) or not domain_idx:
        return {k: list(v) for k, v in _EMPTY_STINTS.items()}
    df = run.stints.sort_values(
        ["t0_us", "job_id", "domain", "t1_us"], kind="mergesort"
    )
    missing = sorted(set(map(str, df["domain"])) - set(domain_idx))
    if missing:  # pragma: no cover - fleet is built from these domains
        notes.append(f"dropped stints in unmapped domains: {missing[:5]}")
        df = df[~df["domain"].isin(missing)]
    return {
        "job_id": [str(x) for x in df["job_id"].tolist()],
        "class_name": [str(x).lower() for x in df["class_name"].tolist()],
        "tier": [str(x).lower() for x in df["tier"].tolist()],
        "domain_idx": [domain_idx[str(d)] for d in df["domain"].tolist()],
        "chips": [int(x) for x in df["chips"].tolist()],
        "t0_us": [int(x) for x in df["t0_us"].tolist()],
        "t1_us": [int(x) for x in df["t1_us"].tolist()],
        "end_reason": [str(x) for x in df["end_reason"].tolist()],
    }


def _build_gantt(run: _Run, max_gantt_jobs: int) -> list[dict[str, Any]]:
    jobs = run.jobs
    horizon = run.horizon_us
    rows: list[tuple[int, str, dict[str, Any]]] = []
    for r in jobs.to_dict("records"):
        start = None if pd.isna(r["first_start_t_us"]) else int(r["first_start_t_us"])
        end = None if pd.isna(r["end_t_us"]) else int(r["end_t_us"])
        chips = int(r["chips"])
        score = (
            chips * ((end if end is not None else horizon) - start)
            if start is not None
            else 0
        )
        status = str(r["status"]).lower()
        n_spanned = (
            None if pd.isna(r["n_domains_spanned"]) else int(r["n_domains_spanned"])
        )
        rows.append(
            (
                score,
                str(r["job_id"]),
                {
                    "id": str(r["job_id"]),
                    "class_name": _label(r["source_class"], r["job_class"]),
                    "chips": chips,
                    "submit_us": int(r["submit_t_us"]),
                    "start_us": start,
                    "end_us": end,
                    "status": status,
                    "n_preemptions": int(r["n_preemptions"]),
                    "n_restarts": int(r["n_restarts"]),
                    "domains_spanned": n_spanned,
                },
            )
        )
    top = sorted(rows, key=lambda x: (-x[0], x[1]))[:max_gantt_jobs]
    chosen = {jid for _, jid, _ in top}
    chosen |= {jid for _, jid, e in rows if e["chips"] >= _BIG_JOB_CHIPS}
    picked = [e for _, jid, e in rows if jid in chosen]
    picked.sort(key=lambda e: (e["submit_us"], e["id"]))
    return picked


def _cdf_points(values: list[float]) -> list[list[float]]:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return []
    if n <= _CDF_MAX_POINTS:
        idxs = range(n)
    else:
        idxs = sorted(
            {
                round(k * (n - 1) / (_CDF_MAX_POINTS - 1))
                for k in range(_CDF_MAX_POINTS)
            }
        )
    return [[_finite(vals[i]) or 0.0, (i + 1) / n] for i in idxs]


def _build_cdfs(run: _Run) -> dict[str, dict[str, list[list[float]]]]:
    jobs = run.jobs
    open_tier = jobs[jobs["tier"] != "BEST_EFFORT"]
    out: dict[str, dict[str, list[list[float]]]] = {}
    specs = (
        ("queue_wait_s", open_tier[open_tier["queue_wait_s"].notna()], "queue_wait_s"),
        (
            "jct_s",
            open_tier[(open_tier["status"] == "COMPLETED") & open_tier["jct_s"].notna()],
            "jct_s",
        ),
    )
    for key, pool, col in specs:
        by_class: dict[str, list[float]] = {}
        for s, jc, v in zip(
            pool["source_class"].tolist(),
            pool["job_class"].tolist(),
            pool[col].tolist(),
        ):
            f = _finite(v)
            if f is not None:
                by_class.setdefault(_label(s, jc), []).append(f)
        out[key] = {lab: _cdf_points(vs) for lab, vs in sorted(by_class.items())}
    return out


# ---------------------------------------------------------------------------
# Events and summary cards
# ---------------------------------------------------------------------------


def _build_events(run: _Run, notes: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ts = run.ts
    n = len(ts)
    if n:
        t_raw = ts["t_us"].to_numpy(dtype="int64")
        preempt = np.diff(ts["cum_preemptions"].to_numpy(dtype="int64"), prepend=0)
        fails = np.diff(ts["cum_node_failures"].to_numpy(dtype="int64"), prepend=0)
        p99 = float(np.percentile(preempt, 99)) if n else 0.0
        threshold = max(_PREEMPT_WAVE_FLOOR, p99)
        for i in np.nonzero(preempt >= threshold)[0]:
            events.append(
                {
                    "t_us": int(t_raw[i]),
                    "kind": "preemption_wave",
                    "label": f"preemption wave: {int(preempt[i])} jobs",
                    "magnitude": int(preempt[i]),
                }
            )
        fail_events = [
            {
                "t_us": int(t_raw[i]),
                "kind": "failure",
                "label": f"node failures: {int(fails[i])}",
                "magnitude": int(fails[i]),
            }
            for i in np.nonzero(fails > 0)[0]
        ]
        if len(fail_events) > _MAX_FAILURE_EVENTS:
            fail_events.sort(key=lambda e: (-e["magnitude"], e["t_us"]))
            fail_events = fail_events[:_MAX_FAILURE_EVENTS]
            notes.append(
                f"failure events truncated to the {_MAX_FAILURE_EVENTS}"
                f" largest rounds"
            )
        events.extend(fail_events)

    big = run.jobs[run.jobs["chips"] >= _FRONTIER_CHIPS]
    for r in big.to_dict("records"):
        jid, chips = str(r["job_id"]), int(r["chips"])
        events.append(
            {
                "t_us": int(r["submit_t_us"]),
                "kind": "frontier_submit",
                "label": f"{jid} submitted ({chips} chips)",
                "magnitude": chips,
            }
        )
        if not pd.isna(r["first_start_t_us"]):
            events.append(
                {
                    "t_us": int(r["first_start_t_us"]),
                    "kind": "frontier_start",
                    "label": f"{jid} started ({chips} chips)",
                    "magnitude": chips,
                }
            )
    events.sort(key=lambda e: (e["t_us"], e["kind"], e["label"]))
    return events


def _pct(x: Any) -> str:
    f = _finite(x)
    return f"{f * 100:.1f}%" if f is not None else "n/a"


def _build_summary_cards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    win = summary.get("window", {})
    full = summary.get("full", {})
    cards = [
        {
            "label": "occupancy",
            "value": _pct(win.get("occupancy")),
            "sub": "steady-state window",
        },
        {
            "label": "goodput",
            "value": _pct(win.get("goodput")),
            "sub": "steady-state window",
        },
        {
            "label": "jobs finished",
            "value": str(int(_get(full, "counts", "jobs_finished") or 0)),
            "sub": "full run",
        },
        {
            "label": "preemptions/min",
            "value": (
                f"{_finite(_get(win, 'preemptions_per_min', 'total')) or 0.0:.2f}"
            ),
            "sub": "window, all triggers",
        },
    ]
    by_src = win.get("queue_wait_s_by_source_class") or {}
    scope = "window"
    if not by_src:
        by_src = full.get("queue_wait_s_by_source_class") or {}
        scope = "full run"

    def count_of(cls: str) -> int:
        return int(_get(by_src, cls, "job_weighted", "n") or 0)

    top = sorted(by_src, key=lambda c: (-count_of(c), c))[:3]
    for cls in top:
        jw = _get(by_src, cls, "job_weighted") or {}
        cards.append(
            {
                "label": f"{cls} wait p50/p99",
                "value": f"{_fmt_dur(_finite(jw.get('p50')))} /"
                f" {_fmt_dur(_finite(jw.get('p99')))}",
                "sub": f"n={count_of(cls)}, {scope}",
            }
        )
    return cards


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def _modal_round_us(ts: pd.DataFrame) -> int | None:
    if len(ts) < 2:
        return None
    diffs = np.diff(ts["t_us"].to_numpy(dtype="int64"))
    counts = Counter(int(d) for d in diffs)
    return min(counts, key=lambda d: (-counts[d], d))


def _build_meta(
    run: _Run, out_dir: str | Path, title: str, notes: list[str]
) -> dict[str, Any]:
    if run.scenario is not None:
        round_us: int | None = int(run.scenario.sim.round_us)
        seed: int | None = int(run.scenario.sim.seed)
        # The standard upgrade path copies the scenario in under a
        # generic name; "scenario scenario" in the header conveys
        # nothing, so generic stems are suppressed.
        stem = Path(run.scenario_file or "").stem
        scenario_name: str | None = (
            stem if stem not in ("", "scenario", "config") else None
        )
    else:
        round_us = _modal_round_us(run.ts)
        seed = None
        scenario_name = None
        if round_us is not None:
            notes.append(
                "round_us: modal timeseries sample gap (no scenario copy"
                " in the output directory)"
            )
    return {
        "title": title,
        "out_dir": str(out_dir),
        "horizon_us": run.horizon_us,
        "round_us": round_us,
        "seed": seed,
        "scenario_name": scenario_name,
        "fleetsim_version": _FLEETSIM_VERSION,
        "generated_unix_ms": None,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_viz_model(
    out_dir: str | Path,
    compare_dir: str | Path | None = None,
    map_level_hint: str | None = None,
    max_frames: int = 1200,
    max_gantt_jobs: int = 300,
) -> dict[str, Any]:
    """Build the v0.3 replay model from one (or two) run directories.

    Parameters
    ----------
    out_dir:
        A ``fleetsim run`` output directory (``summary.json`` +
        ``jobs.parquet`` + ``timeseries.parquet``; ``stints.parquet``
        enables the fleet map).
    compare_dir:
        Optional second run: adds ``compare`` (B's frames and summary
        cards) and sets ``capabilities.compare``.
    map_level_hint:
        Level name for the fleet map (default: inferred from the stint
        domain ids).
    max_frames:
        Downsampling budget for ``frames`` (and ``frames_b``).
    max_gantt_jobs:
        Ranked-selection budget for ``gantt`` (jobs >= 4096 chips are
        always included on top).

    Raises ``FileNotFoundError`` when a directory is not a fleetsim
    output directory.  See the module docstring for the full pinned
    model schema and per-field rules.
    """
    notes: list[str] = []
    run = _Run(out_dir)
    run_b = _Run(compare_dir) if compare_dir is not None else None

    label_a = Path(out_dir).name or str(out_dir)
    if run_b is not None:
        label_b = Path(compare_dir).name or str(compare_dir)
        if label_a == label_b:
            label_a, label_b = str(out_dir), str(compare_dir)
        title = f"fleetsim replay — {label_a} vs {label_b}"
    else:
        title = f"fleetsim replay — {label_a}"

    fleet_model, domain_idx = _build_fleet_model(run, map_level_hint, notes)
    frames = _build_frames(run, max_frames, notes)
    stints = _build_stints(run, domain_idx, notes)
    buckets = _label_buckets(run.jobs)
    if run_b is not None:
        buckets.update(
            {
                lab: b
                for lab, b in _label_buckets(run_b.jobs).items()
                if lab not in buckets
            }
        )

    compare: dict[str, Any] | None = None
    if run_b is not None:
        notes_b: list[str] = []
        compare = {
            "label_a": label_a,
            "label_b": label_b,
            "frames_b": _build_frames(run_b, max_frames, notes_b),
            "summary_cards_b": _build_summary_cards(run_b.summary),
        }
        notes.extend(f"compare run: {n}" for n in notes_b)

    model: dict[str, Any] = {
        "meta": None,  # placed first; filled after notes are complete
        "capabilities": {
            "map": bool(fleet_model["clusters"]),
            "compare": run_b is not None,
        },
        "palette": _build_palette(buckets, notes),
        "fleet": fleet_model,
        "frames": frames,
        "stints": stints,
        "gantt": _build_gantt(run, max_gantt_jobs),
        "cdfs": _build_cdfs(run),
        "events": _build_events(run, notes),
        "summary_cards": _build_summary_cards(run.summary),
        "compare": compare,
    }
    model["meta"] = _build_meta(run, out_dir, title, notes)
    return model


def to_json(model: dict[str, Any]) -> str:
    """Serialize the model compactly and safely: separators ``(",",
    ":")``, key order preserved (the pinned schema order), and
    ``allow_nan=False`` — a non-finite float anywhere is a bug and
    raises ``ValueError`` instead of emitting invalid JSON."""
    return json.dumps(model, separators=(",", ":"), allow_nan=False)
