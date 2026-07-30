"""The validation suite's MEASURED published-vs-fleetsim numbers, as data.

ONE SOURCE OF TRUTH.  Before v0.8 these numbers lived only inside the
validation tests (``validation/test_helios_ratio.py::_PUB`` and friends) and
were re-typed by hand in ``docs/validation.md``.  A web app that wanted to
show "how well does fleetsim match the papers?" would have had to parse the
prose — a third copy, free to drift.  They now live HERE: the tests import
them (so a number that changes fails the suite it belongs to) and
``GET /api/validation`` serves :func:`payload` (so the app renders exactly
what the tests assert).  ``docs/validation.md`` keeps the narrative — the
*why* behind each figure — and cites this module for the figures.

WHAT IS AND IS NOT IN HERE.  Only *quantities*: published value, the value
fleetsim measured, the previous release's value where one was recorded, the
assertion band or tolerance, and the citation.  Every judgement — why a band
is wide, why a placer was chosen, what a rung does not prove — stays in
``docs/validation.md``, referenced per row by ``doc_ref``.  Rows whose
``fleetsim`` is ``None`` are HONESTLY UNMEASURED (the opt-in rung has not
been run against the real artifact); the app must render them as such rather
than as agreement.

PROVENANCE OF EACH FIGURE
- ``published``: read off the cited paper's table.
- ``fleetsim``: measured by the opt-in full-trace rung named in ``rung``,
  under the validation model's stated choices (strict scan, September-max
  per-VC sizing, ``consolidate`` placement — docs/validation.md §4.2–§4.4).
- ``previous``: the same quantity measured by v0.6 (``first_fit``
  placement), kept so the record shows movement rather than only the
  current state.

Pure data: importing this module pulls in nothing but the trace registry
(for citations, so licence/attribution strings also have one home) — no
pandas, no network.  That is what lets the stdlib-only web server import it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .registry import TRACE_REGISTRY

__all__ = [
    "ANTI_GOALS",
    "HELIOS_JCT_RATIO_BAND",
    "HELIOS_PUBLISHED_RATIOS",
    "HELIOS_Q_RATIO_BAND",
    "HELIOS_SATURN_FIFO_JCT_PUBLISHED",
    "LADDER",
    "PHILLY_BY_COUNT_TOL",
    "PHILLY_BY_GPU_TOL",
    "PHILLY_PUBLISHED_BY_COUNT",
    "PHILLY_PUBLISHED_BY_GPU",
    "PLACER_SWEEP",
    "RESULTS",
    "Measurement",
    "citations",
    "payload",
    "results_for",
]


@dataclass(frozen=True, slots=True)
class Measurement:
    """One published quantity and what fleetsim measured against it.

    Fields
    ------
    id:
        Stable dotted slug (``helios.jct_ratio.Saturn``) — the app's key.
    rung:
        Validation-ladder rung (``V1`` | ``V2`` | ``V3``).
    trace:
        Key into :data:`fleetsim.validation.registry.TRACE_REGISTRY`, which
        carries the citation, licence, and attribution URL.
    group / subject:
        Human grouping (a table in the paper) and the row within it.
    unit:
        ``ratio`` | ``share`` | ``s`` | ``count`` — the app formats on this.
    published / fleetsim / previous:
        The paper's number, fleetsim's measurement (``None`` = not measured
        on the real artifact yet), and the previous release's measurement
        (``None`` = not recorded).
    band:
        The asserted ``(lo, hi)`` interval, when the rung asserts a band.
    tolerance:
        The asserted absolute tolerance around ``published``, when the rung
        asserts one instead of a band (Philly's percentage-point bars).
    doc_ref:
        Section of ``docs/validation.md`` that explains the figure.
    """

    id: str
    rung: str
    trace: str
    group: str
    subject: str
    unit: str
    published: float
    fleetsim: float | None = None
    previous: float | None = None
    band: tuple[float, float] | None = None
    tolerance: float | None = None
    doc_ref: str = ""

    @property
    def rel_error(self) -> float | None:
        """Signed relative error of ``fleetsim`` against ``published``
        (``None`` when unmeasured or the published value is zero)."""
        if self.fleetsim is None or not self.published:
            return None
        return (self.fleetsim - self.published) / self.published

    @property
    def in_band(self) -> bool | None:
        """Does the measurement satisfy this row's assertion?  ``None`` when
        unmeasured, or when the row carries neither band nor tolerance
        (reported-only rows — docs/validation.md §5)."""
        if self.fleetsim is None:
            return None
        if self.band is not None:
            return self.band[0] <= self.fleetsim <= self.band[1]
        if self.tolerance is not None:
            return abs(self.fleetsim - self.published) <= self.tolerance
        return None


# ---------------------------------------------------------------------------
# Assertion bands and tolerances (imported by the validation tests)
# ---------------------------------------------------------------------------

#: V1 (plan §2 V1(f)) FIFO/SJF average-JCT ratio band.  Deliberately much
#: wider than the point values and NOT tightened in v0.7 even though all
#: four clusters land inside: the paper's analysis window is unpublished
#: (docs/validation.md §1) and the per-VC capacity model is a choice with
#: its own bias (§4.4).
HELIOS_JCT_RATIO_BAND: tuple[float, float] = (1.3, 8.0)

#: V1 FIFO/SJF queuing-time ratio band (same reasoning).
HELIOS_Q_RATIO_BAND: tuple[float, float] = (3.0, 25.0)

#: V3 (plan §2 V3(f)) tolerances in absolute share: the released
#: 117,325-job / 137-day Philly trace is not the paper's 96,260-job window,
#: so the bars absorb the window residual.
PHILLY_BY_COUNT_TOL: float = 0.05  # +/- 5 percentage points
PHILLY_BY_GPU_TOL: float = 0.08  # +/- 8 percentage points


# ---------------------------------------------------------------------------
# The measurement table
# ---------------------------------------------------------------------------

_HELIOS_JCT_RATIO = (
    # cluster, published, fleetsim v0.7 (consolidate), v0.6 (first_fit)
    ("Saturn", 6.59, 6.87, 8.75),
    ("Venus", 3.07, 3.21, 4.21),
    ("Earth", 2.87, 2.95, 2.11),
    ("Uranus", 1.49, 1.51, 1.69),
)

_HELIOS_Q_RATIO = (
    ("Saturn", 18.5, 19.55, 22.91),
    ("Earth", 16.4, 10.67, 5.73),
    ("Venus", 5.68, 7.78, 9.97),
    ("Uranus", 4.51, 5.08, 6.10),
)

_HELIOS_Q_SHARE = (
    ("Saturn", 0.897, 0.901, 0.926),
    ("Venus", 0.818, 0.791, 0.847),
    ("Earth", 0.693, 0.730, 0.638),
    ("Uranus", 0.425, 0.422, 0.487),
)

_HELIOS_FIFO_JCT_S = (
    ("Venus", 64_702.0, 55_714.0),
    ("Earth", 19_754.0, 22_463.0),
    ("Saturn", 55_984.0, 55_978.0),
    ("Uranus", 19_758.0, 18_492.0),
)

_HELIOS_N_QUEUING = (
    ("Venus", 15_336.0, 13_624.0),
    ("Earth", 30_030.0, 32_232.0),
    ("Saturn", 65_991.0, 72_936.0),
    ("Uranus", 16_917.0, 18_449.0),
)

_PHILLY_BY_COUNT = (
    ("Passed", 0.693),
    ("Killed", 0.135),
    ("Unsuccessful", 0.172),
)

_PHILLY_BY_GPU = (
    ("Passed", 0.4453),
    ("Killed", 0.3769),
    ("Unsuccessful", 0.1776),
)


def _build_results() -> tuple[Measurement, ...]:
    out: list[Measurement] = []
    for cluster, pub, fs, prev in _HELIOS_JCT_RATIO:
        out.append(
            Measurement(
                id=f"helios.jct_ratio.{cluster}",
                rung="V1",
                trace="helios",
                group="FIFO/SJF average-JCT ratio (Table 3)",
                subject=cluster,
                unit="ratio",
                published=pub,
                fleetsim=fs,
                previous=prev,
                band=HELIOS_JCT_RATIO_BAND,
                doc_ref="docs/validation.md §4.1",
            )
        )
    for cluster, pub, fs, prev in _HELIOS_Q_RATIO:
        out.append(
            Measurement(
                id=f"helios.q_ratio.{cluster}",
                rung="V1",
                trace="helios",
                group="FIFO/SJF queuing-time ratio (Table 3)",
                subject=cluster,
                unit="ratio",
                published=pub,
                fleetsim=fs,
                previous=prev,
                band=HELIOS_Q_RATIO_BAND,
                doc_ref="docs/validation.md §4.1",
            )
        )
    for cluster, pub, fs, prev in _HELIOS_Q_SHARE:
        out.append(
            Measurement(
                id=f"helios.q_share.{cluster}",
                rung="V1",
                trace="helios",
                group="Queuing share of FIFO JCT (Table 3)",
                subject=cluster,
                unit="share",
                published=pub,
                fleetsim=fs,
                previous=prev,
                doc_ref="docs/validation.md §4.1",
            )
        )
    for cluster, pub, fs in _HELIOS_FIFO_JCT_S:
        out.append(
            Measurement(
                id=f"helios.fifo_jct_s.{cluster}",
                rung="V2",
                trace="helios",
                group="Absolute FIFO average JCT (Table 3)",
                subject=cluster,
                unit="s",
                published=pub,
                fleetsim=fs,
                doc_ref="docs/validation.md §5",
            )
        )
    for cluster, pub, fs in _HELIOS_N_QUEUING:
        out.append(
            Measurement(
                id=f"helios.n_queuing.{cluster}",
                rung="V2",
                trace="helios",
                group="Absolute #Queuing jobs (Table 3)",
                subject=cluster,
                unit="count",
                published=pub,
                fleetsim=fs,
                doc_ref="docs/validation.md §5",
            )
        )
    for status, pub in _PHILLY_BY_COUNT:
        out.append(
            Measurement(
                id=f"philly.status_by_count.{status}",
                rung="V3",
                trace="philly",
                group="Job-status split by count (Table 6)",
                subject=status,
                unit="share",
                published=pub,
                fleetsim=None,  # opt-in rung not run on the 1 GB LFS artifact
                tolerance=PHILLY_BY_COUNT_TOL,
                doc_ref="docs/validation.md §6",
            )
        )
    for status, pub in _PHILLY_BY_GPU:
        out.append(
            Measurement(
                id=f"philly.status_by_gpu_time.{status}",
                rung="V3",
                trace="philly",
                group="Job-status split by GPU-time (Table 6)",
                subject=status,
                unit="share",
                published=pub,
                fleetsim=None,
                tolerance=PHILLY_BY_GPU_TOL,
                doc_ref="docs/validation.md §6",
            )
        )
    return tuple(out)


#: Every published-vs-fleetsim quantity the suite records.
RESULTS: tuple[Measurement, ...] = _build_results()


# ---------------------------------------------------------------------------
# Derived views the validation tests import (so nothing is typed twice)
# ---------------------------------------------------------------------------


def results_for(*, rung: str | None = None, trace: str | None = None) -> tuple[
    Measurement, ...
]:
    """The subset of :data:`RESULTS` matching ``rung`` / ``trace``."""
    return tuple(
        m
        for m in RESULTS
        if (rung is None or m.rung == rung) and (trace is None or m.trace == trace)
    )


def _helios_published_ratios() -> dict[str, dict[str, float]]:
    """``{cluster: {jct, q, share}}`` folded out of :data:`RESULTS` — the
    shape ``validation/test_helios_ratio.py`` uses, derived rather than
    re-typed so the table stays the only place these numbers exist."""
    key = {"jct_ratio": "jct", "q_ratio": "q", "q_share": "share"}
    out: dict[str, dict[str, float]] = {}
    for m in results_for(rung="V1", trace="helios"):
        kind = m.id.split(".")[1]
        out.setdefault(m.subject, {})[key[kind]] = m.published
    return out


#: Published Helios Table 3 point values, per cluster (V1's reference).
HELIOS_PUBLISHED_RATIOS: dict[str, dict[str, float]] = _helios_published_ratios()

#: Published Saturn FIFO average JCT (s) — the V2 absolute rung's anchor.
HELIOS_SATURN_FIFO_JCT_PUBLISHED: float = next(
    m.published for m in RESULTS if m.id == "helios.fifo_jct_s.Saturn"
)

#: Published Philly Table 6 status shares (V3's reference).
PHILLY_PUBLISHED_BY_COUNT: dict[str, float] = {
    m.subject: m.published for m in RESULTS if m.id.startswith("philly.status_by_count.")
}
PHILLY_PUBLISHED_BY_GPU: dict[str, float] = {
    m.subject: m.published
    for m in RESULTS
    if m.id.startswith("philly.status_by_gpu_time.")
}


# ---------------------------------------------------------------------------
# Context tables (reported, never asserted)
# ---------------------------------------------------------------------------

#: v0.7a placer sweep: mean absolute relative error of the four Helios JCT
#: ratios against published, per placement policy.  This AGGREGATE is what
#: selected the validation model's placer (docs/validation.md §4.2.3);
#: ``best_fit`` came out bit-identical to ``consolidate`` on the real trace.
PLACER_SWEEP: tuple[dict[str, Any], ...] = (
    {"placer": "consolidate", "shipped": True, "mean_abs_ratio_error": 0.034},
    {"placer": "best_fit", "shipped": False, "mean_abs_ratio_error": 0.034},
    {"placer": "first_fit", "shipped": False, "mean_abs_ratio_error": 0.274},
    {"placer": "spread", "shipped": False, "mean_abs_ratio_error": 0.295},
)

#: docs/validation.md §0 — published numbers fleetsim deliberately does not
#: reproduce.  Shipped as data so the app can state the anti-goals next to
#: the results instead of letting a reader assume total coverage.
ANTI_GOALS: tuple[dict[str, str], ...] = (
    {
        "quantity": "Philly ~52.3 % GPU utilization (Tables 3-5)",
        "why": "a hardware SM-cycle counter, not scheduler occupancy;"
        " a DES produces allocation occupancy, a different quantity",
        "disposition": "out of scope (documented anti-goal)",
    },
    {
        "quantity": "Philly fair-share vs fragmentation-delay split (Table 2)",
        "why": "needs per-job delay-cause attribution fleetsim does not compute",
        "disposition": "deferred; v0.7's stranded_whole_nodes is a fleet-level"
        " down payment, not a substitute",
    },
    {
        "quantity": "Helios QSSF column",
        "why": "needs a jobname column (absent from the public CSV) plus a"
        " GBDT duration predictor",
        "disposition": "out of scope; SJF-oracle is the reproducible"
        " upper-bound proxy (V1)",
    },
    {
        "quantity": "Alibaba 50 % GPU-sharing saving (median 0.042 GPU/inst)",
        "why": "needs fractional sub-chip GPU allocation; fleetsim shares only"
        " whole chips within a node",
        "disposition": "deferred (fractional-GPU packing)",
    },
    {
        "quantity": "Borg absolute occupancy / JCT",
        "why": "resources are normalized [0,1], BigQuery-only, 8 independent"
        " 12k-machine cells",
        "disposition": "never a replay target; generator-distribution check only",
    },
    {
        "quantity": "CPU contention, co-location interference",
        "why": "not modeled (analytical speed model, no memory-bandwidth"
        " contention)",
        "disposition": "out of scope",
    },
)

#: docs/validation.md §3 — the validation ladder, as data.
LADDER: tuple[dict[str, Any], ...] = (
    {
        "rung": "V1",
        "validation": "Helios FIFO-vs-SJF JCT & queuing ratio",
        "kind": "policy-effect",
        "shipped_as": "validation/test_helios_ratio.py",
        "ci": "2-VC Venus slice, direction only",
        "full": "opt-in FLEETSIM_HELIOS_FULL",
    },
    {
        "rung": "V2",
        # ABSOLUTE, not "distribution": this rung compares fleetsim's FIFO
        # average JCT and #Queuing counts against Table 3's published
        # VALUES (units s / count).  V3 is the distribution rung.
        "validation": "Helios FIFO absolute Table 3",
        "kind": "absolute",
        "shipped_as": "harness rung (reported in docs/validation.md §5)",
        "ci": None,
        "full": "opt-in (same trace)",
    },
    {
        "rung": "V3",
        "validation": "Philly job-status split",
        "kind": "job-status distribution",
        "shipped_as": "validation/test_philly_status.py",
        "ci": "~2k-row slice, ordering invariants",
        "full": "opt-in FLEETSIM_PHILLY_FULL",
    },
    {
        "rung": "SPT",
        "validation": "SJF is SPT-optimal on a fungible pool",
        "kind": "analytic",
        "shipped_as": "validation/test_sjf_ordering.py",
        "ci": "always",
        "full": None,
    },
    {
        "rung": "V1p",
        "validation": "Helios Saturn: the placement model is load-bearing",
        "kind": "policy-effect",
        "shipped_as": "validation/test_helios_ratio.py",
        "ci": "3-placer slice inequality",
        "full": "opt-in FLEETSIM_HELIOS_FULL",
    },
)

#: The headline claim, stated exactly as strongly as the evidence supports
#: — the SAME sentence docs/validation.md opens with, including BOTH of
#: its qualifiers.  The web page renders this string verbatim, so a reader
#: who never opens the repository must not get a stronger claim than a
#: reader who does: dropping the modeling conditions turns a conditional
#: reproduction into an unconditional one, and dropping the
#: order-sensitivity paragraph invites reading Saturn's -0.01 % as
#: precision, which the doc pre-emptively refuses.
HEADLINE: str = (
    "fleetsim reproduces the Helios (SC '21) FIFO-vs-SJF average-JCT policy"
    " effect across all four clusters — under per-VC September-max sizing"
    " (§4.4), a strict blocking scan (§4.3) and consolidate placement"
    " (§4.2): all four JCT ratios inside the [1.3-8]x tolerance band, all"
    " four queuing ratios inside [3-25]x, the Saturn-strongest to"
    " Uranus-weakest rank, and the queuing-share ordering. Absolute FIFO"
    " JCT lands within +/-14 % on every cluster and #Queuing within"
    " +/-12 %."
    " The bands are NOT tightened to match: the paper's analysis window is"
    " unpublished and the capacity model is a choice."
    " Separately, the result is ORDER-SENSITIVE: 35.5 % of Saturn's jobs"
    " share an exact submit second, and reordering within those seconds"
    " moves its FIFO JCT by 17 % (§4.5). So agreement closer than ~20 % on"
    " any single point value here is not evidence of accuracy, and the"
    " bands stay bands."
)

#: Per-GROUP captions rendered directly under that group's table.  A
#: caveat that lives only in the headline is a caveat a reader scrolling
#: to a table never sees — and the V2 tables are exactly where a -0.01 %
#: cell invites the reading the headline refuses.
GROUP_CAPTIONS: dict[str, str] = {
    "Absolute FIFO average JCT (Table 3)": (
        "Reported, not asserted — these rows carry no band, so CI does not"
        " fail if they move. What the table supports is the +/-14 % band,"
        " NOT any single cell: reordering within the trace's tied submit"
        " seconds (35.5 % of Saturn's jobs share one) moves Saturn's FIFO"
        " JCT by 17 %, so Saturn's -0.01 % is luck, not precision"
        " (docs/validation.md §4.5)."
    ),
    "Absolute #Queuing jobs (Table 3)": (
        "Reported, not asserted — the supported claim is the +/-12 % band"
        " across the four clusters, not any single count."
    ),
}


#: The percentage figures :data:`HEADLINE` quotes for the V2 groups,
#: keyed by group.  Stated in the sentence AND derived from the rows by
#: :func:`headline_bounds`, so a measurement that drifts past its quoted
#: bound fails a test instead of quietly making the page overclaim.
HEADLINE_BOUNDS_PCT: dict[str, float] = {
    "Absolute FIFO average JCT (Table 3)": 14.0,
    "Absolute #Queuing jobs (Table 3)": 12.0,
}


def headline_bounds() -> dict[str, float]:
    """Observed ``max(|rel_error|)`` in percent, per headline group.

    Groups with no measured row are absent (an unmeasured rung bounds
    nothing).  Compare against :data:`HEADLINE_BOUNDS_PCT`: the quoted
    figure must still cover the data.
    """
    out: dict[str, float] = {}
    for m in RESULTS:
        if m.group not in HEADLINE_BOUNDS_PCT:
            continue
        rel = m.rel_error
        if rel is None:
            continue
        out[m.group] = max(out.get(m.group, 0.0), abs(rel) * 100.0)
    return out


def citations() -> dict[str, dict[str, str]]:
    """Per-trace citation / licence / attribution, read from the trace
    registry so the strings the licences require have exactly one home."""
    out: dict[str, dict[str, str]] = {}
    for trace in sorted({m.trace for m in RESULTS}):
        spec = TRACE_REGISTRY.get(trace)
        if spec is None:  # pragma: no cover - a row must name a real trace
            continue
        out[trace] = {
            "citation": spec.citation,
            "license": spec.license,
            "source": spec.attribution_url,
        }
    return out


def payload() -> dict[str, Any]:
    """The ``GET /api/validation`` document.

    ``{version, headline, results: [...], groups: [...], ladder, anti_goals,
    placer_sweep, citations, doc}`` — every ``results`` row is a
    :class:`Measurement` plus its derived ``rel_error`` / ``in_band``, so
    the app never recomputes an assertion the tests own.
    """
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for m in RESULTS:
        row = asdict(m)
        row["band"] = list(m.band) if m.band is not None else None
        row["rel_error"] = m.rel_error
        row["in_band"] = m.in_band
        rows.append(row)
        key = f"{m.rung}:{m.group}"
        group = seen.get(key)
        if group is None:
            group = seen[key] = {
                "rung": m.rung,
                "trace": m.trace,
                "group": m.group,
                "unit": m.unit,
                "doc_ref": m.doc_ref,
                "caption": GROUP_CAPTIONS.get(m.group),
                "ids": [],
            }
            groups.append(group)
        group["ids"].append(m.id)
    measured = [m for m in RESULTS if m.fleetsim is not None]
    asserted = [m for m in measured if m.in_band is not None]
    # The rung filter's options, BUILT FROM THE DATA.  Hand-written labels
    # drifted: the tab shipped "V2 - distribution" / "V3 - absolute" while
    # V2 is the absolute Table-3 rung and V3 the job-status distribution,
    # and it offered a V1p rung no results row uses, so it could never be
    # selected.  Deriving both the label and the option set from the rows
    # that exist makes either failure unrepresentable.
    ladder_kind = {row["rung"]: row["kind"] for row in LADDER}
    rungs: list[dict[str, str]] = []
    for group in groups:
        if any(r["rung"] == group["rung"] for r in rungs):
            continue
        rungs.append(
            {
                "rung": group["rung"],
                "label": (
                    f"{group['rung']} — "
                    f"{ladder_kind.get(group['rung'], 'measurements')}"
                ),
            }
        )
    return {
        "version": "v0.7",
        "headline": HEADLINE,
        "results": rows,
        "groups": groups,
        "rungs": rungs,
        "counts": {
            "total": len(RESULTS),
            "measured": len(measured),
            "asserted": len(asserted),
            "in_band": sum(1 for m in asserted if m.in_band),
        },
        "ladder": [dict(row) for row in LADDER],
        "anti_goals": [dict(row) for row in ANTI_GOALS],
        "placer_sweep": [dict(row) for row in PLACER_SWEEP],
        "citations": citations(),
        "doc": "docs/validation.md",
    }
