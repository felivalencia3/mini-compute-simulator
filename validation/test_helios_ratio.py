"""V1 — the flagship "our numbers are real" validation (plan §2 V1).

Replays the Helios trace per VC (:func:`fleetsim.validation.harness`) under
FIFO and SJF and checks fleetsim reproduces the published Table 3
FIFO-vs-SJF **policy effect** (Hu et al., SC '21, arXiv:2109.01313;
September 2020-09-01..2020-09-26 window, GPU jobs only).

Two rungs:

- **CI smoke** (always runs, < 10 s): the vendored 2-VC Venus September
  slice (``tests/validation_traces/helios_venus_2vc_sept.csv``, REAL
  trace).  Asserts only the *direction* — FIFO's mean JCT is worse than
  SJF's by a clear margin, and both ratios are finite — so CI proves the
  machinery end to end without a network fetch.
- **Full trace** (opt-in, ``@pytest.mark.trace_full`` + ``FLEETSIM_HELIOS_FULL``):
  downloads the real ``data.zip`` and replays all four clusters, asserting
  the plan §2 V1(f) bands and cross-cluster ranks.

SCAN MODE.  Both rungs run **strict** (head-of-line blocking) — the load-
bearing fidelity choice.  The published FIFO queuing is head-of-line
blocking of large gangs; a best-effort scan collapses it and the ratio
(see :mod:`fleetsim.validation.harness`).

PLACEMENT MODEL (v0.7).  Both rungs run ``placement="consolidate"`` — a
stated validation-model choice on the same footing as
``pool_snapshot="max"``, and passed EXPLICITLY at each call site for the
same reason.  fleetsim's engine default is ``first_fit``, which strands
whole-node capacity behind sub-node remainders; that is what put Saturn's
JCT ratio out of band in v0.6.  The ``first_fit`` baseline stays reachable
(``placement="first_fit"``) and one rung below asserts the placers really
differ, so the choice is testable rather than assumed.

WHY ``consolidate`` AND NOT ANOTHER PLACER — measured, all four policies x
all four clusters x FIFO and SJF on the real trace (v0.7a).  Mean absolute
relative error against the published JCT ratios: ``consolidate`` /
``best_fit`` **3.4 %**, ``first_fit`` 27.4 %, ``spread`` 29.5 %.
``best_fit`` and ``consolidate`` came out BIT-IDENTICAL on every cluster
metric — the single-level degeneracy documented on
:class:`fleetsim.schedulers.placement.Consolidate`, confirmed on the trace
and not only in a unit test.

HONEST LIMIT OF THE BAND RUNG.  The ratio bands and both rank invariants
below are satisfied by ``spread`` too (Saturn 7.48x, Venus 4.36x, Earth
3.06x, Uranus 2.32x; Saturn highest, Uranus lowest, q-share order intact),
even though ``spread`` is 45 % off Saturn's absolute FIFO JCT and 99 % off
Earth's.  So V1's bands alone do NOT single out a placer.  What does, in
descending weight: the FOUR-CLUSTER mean absolute ratio error above, then
the ``stranded_whole_nodes`` mechanism metric, then Saturn's ABSOLUTE FIFO
JCT.  The third is corroboration only — under the descending-id tie-break of
docs/validation.md §4.5 it inverts (``first_fit`` +11.7 % vs ``consolidate``
+13.2 %), so a placer choice resting on Saturn's absolute number alone would
flip on a knob unrelated to placement.  Stated here rather than left for a
reader to discover.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import pytest

from fleetsim.validation.harness import per_vc_replay, replay_canonical
from fleetsim.validation.results import (
    HELIOS_JCT_RATIO_BAND,
    HELIOS_PUBLISHED_RATIOS,
    HELIOS_Q_RATIO_BAND,
    HELIOS_SATURN_FIFO_JCT_PUBLISHED,
)

# Repo layout: this file is <root>/validation/test_helios_ratio.py.
_ROOT = Path(__file__).resolve().parents[1]
_SLICE = _ROOT / "tests" / "validation_traces" / "helios_venus_2vc_sept.csv"

#: Per-VC pool sizes for the vendored slice (nodes, 8 GPU/node), recorded
#: in the slice's own header comment ("per-VC pool sizes ... vcvGl=20,
#: vcvlY=2").  The per-VC replay MUST size each pool to these.
_SLICE_POOLS = {"vcvGl": 20, "vcvlY": 2}

#: Direction bar for the CI smoke: FIFO's mean JCT is at least this many
#: times SJF's on the slice (measured ~2.75x strict; ~1.47x best-effort).
_SMOKE_MIN_RATIO = 1.15

#: The validation model's placement policy, passed explicitly at every call
#: site in this file exactly like ``pool_snapshot="max"`` rather than
#: inherited from the harness default (which is the ENGINE default,
#: ``first_fit``).  Selected by measurement — see the module docstring.
_PLACEMENT = "consolidate"


# ---------------------------------------------------------------------------
# CI smoke — vendored slice, direction only (always runs, no network)
# ---------------------------------------------------------------------------


def _load_slice() -> pd.DataFrame:
    """The vendored 2-VC Venus slice as a canonical DataFrame (``#`` header
    comments skipped; ``submit_time`` already int microseconds)."""
    return pd.read_csv(_SLICE, comment="#")


def test_helios_slice_smoke_fifo_worse_than_sjf() -> None:
    """CI smoke: on the vendored Venus slice, strict-FIFO mean JCT is
    clearly worse than strict-SJF (the SJF policy advantage), and both
    ratios are finite.  Direction only — no point-value assertion (plan §2
    V1(g))."""
    df = _load_slice()
    assert set(df["tenant"].unique()) == set(_SLICE_POOLS)

    fifo = replay_canonical(
        df, _SLICE_POOLS, "fifo", cluster="Venus-slice", placement=_PLACEMENT
    )
    sjf = replay_canonical(
        df, _SLICE_POOLS, "sjf", cluster="Venus-slice", placement=_PLACEMENT
    )
    assert fifo["placement"] == sjf["placement"] == _PLACEMENT

    # Every windowed job was replayed (no VC dropped) and reached a
    # terminal status (no horizon truncation) — the long-waiting FIFO jobs
    # are exactly the signal, so neither loss may bias the means.
    assert fifo["n_dropped"] == sjf["n_dropped"] == 0
    assert fifo["n_terminal"] == fifo["n_jobs"] == len(df)
    assert sjf["n_terminal"] == sjf["n_jobs"] == len(df)

    jct_ratio = fifo["avg_jct"] / sjf["avg_jct"]
    q_ratio = fifo["avg_queuing"] / sjf["avg_queuing"]
    assert math.isfinite(jct_ratio) and math.isfinite(q_ratio)
    assert math.isfinite(fifo["avg_jct"]) and math.isfinite(sjf["avg_jct"])
    # FIFO is worse (longer JCT / longer queue) than SJF, by a clear margin.
    assert jct_ratio > _SMOKE_MIN_RATIO, (jct_ratio, fifo["avg_jct"], sjf["avg_jct"])
    assert q_ratio > 1.0, (q_ratio, fifo["avg_queuing"], sjf["avg_queuing"])
    # #Queuing is nonzero and FIFO stalls at least as many jobs as SJF.
    assert fifo["n_queuing"] >= sjf["n_queuing"] > 0


def test_helios_slice_smoke_placement_selection_reaches_the_engine() -> None:
    """CI smoke for the v0.7 placement model, no network — **WIRING ONLY**.

    Asserts the harness's own default is the ENGINE default, that the
    validation-model placer is the one the shipped rungs pass, that the run
    echoes which placer ran, and that the selection genuinely reaches the
    engine.

    The last assertion is specifically ``first_fit != consolidate``, not
    "the placers are not all equal".  ``spread`` alone satisfies the weaker
    form, so a refactor that silently degraded ``consolidate`` (the policy
    carrying the entire v0.7 result) back to first-fit would have left it
    green — a hole confirmed by sabotage, and the reason this rung and the
    ``fleetsim validation run`` rung both assert the tighter form.  On this
    slice the two differ by only ~0.4 s on ~28,557 s, which is enough for
    an exact-inequality wiring check and not enough for a claim about size.

    It deliberately asserts **no direction and no magnitude**: this 2-VC
    Venus slice does not exhibit the Saturn stranding pathology at any
    meaningful size, so a directional claim here would be noise dressed as
    evidence.  The magnitude claim lives in
    ``test_helios_saturn_placement_model_is_load_bearing`` (full trace,
    ~26 % on Saturn's FIFO JCT); the *mechanism* is proved at unit scale in
    ``tests/test_placement.py::TestStrandingMechanism``.
    """
    from fleetsim.validation.harness import DEFAULT_PLACEMENT, VALIDATION_PLACEMENT

    # The harness default is the ENGINE default: an unqualified call must
    # not silently carry the validation model's placer (v0.7 API note).
    assert DEFAULT_PLACEMENT == "first_fit"
    assert VALIDATION_PLACEMENT == _PLACEMENT == "consolidate"
    df = _load_slice()
    jct = {}
    for placement in ("first_fit", "consolidate", "spread"):
        out = replay_canonical(
            df, _SLICE_POOLS, "fifo", cluster="Venus-slice", placement=placement
        )
        assert out["placement"] == placement
        assert out["n_dropped"] == 0 and out["n_terminal"] == out["n_jobs"]
        assert math.isfinite(out["avg_jct"])
        jct[placement] = out["avg_jct"]
    # The load-bearing pair really differs — not merely "some placer does".
    assert jct["first_fit"] != jct["consolidate"], jct
    assert jct["first_fit"] != jct["spread"], jct
    # And the harness default really is what an unspecified call produces.
    default = replay_canonical(df, _SLICE_POOLS, "fifo", cluster="Venus-slice")
    assert default["placement"] == DEFAULT_PLACEMENT
    assert default["avg_jct"] == jct["first_fit"]


def test_helios_slice_smoke_best_effort_direction() -> None:
    """CI smoke: the direction survives even a best-effort scan (the
    weaker mode) — FIFO still beats-worse than SJF on the slice, so the
    smoke bar is not an artifact of the blocking scan alone."""
    df = _load_slice()
    fifo = replay_canonical(
        df,
        _SLICE_POOLS,
        "fifo",
        cluster="Venus-slice",
        strict=False,
        placement=_PLACEMENT,
    )
    sjf = replay_canonical(
        df,
        _SLICE_POOLS,
        "sjf",
        cluster="Venus-slice",
        strict=False,
        placement=_PLACEMENT,
    )
    ratio = fifo["avg_jct"] / sjf["avg_jct"]
    assert math.isfinite(ratio)
    assert ratio > _SMOKE_MIN_RATIO, ratio


# ---------------------------------------------------------------------------
# Full trace — opt-in (downloads real data.zip; asserts §2 V1(f))
# ---------------------------------------------------------------------------

#: plan §2 V1(f) assertion bands and the published Table 3 point values.
#: BOTH come from :mod:`fleetsim.validation.results`, the single source of
#: truth these numbers have since v0.8 — the same table
#: ``GET /api/validation`` serves to the web app, so an assertion here and
#: a number on screen can never disagree.  The bands stay BANDS,
#: deliberately much wider than the point values, and v0.7 does NOT tighten
#: them despite all four clusters now landing inside; the width comes from
#: two things this suite cannot pin down (the paper's unpublished ANALYSIS
#: WINDOW, §1, and the per-VC CAPACITY MODEL choice, §4.4).  A third,
#: separate fact bounds how much any single point value can be leaned on:
#: 35.5 % of Saturn's jobs share an exact submit second, and reordering
#: within those seconds moves Saturn's FIFO JCT by 17 % (docs/validation.md
#: §4.5) — that is a scheduler-order SENSITIVITY, not an uncertainty band
#: (ascending id is the faithful order), but it is why the placer choice
#: rests on the four-cluster aggregate rather than on Saturn alone.
_JCT_RATIO_BAND = HELIOS_JCT_RATIO_BAND
_Q_RATIO_BAND = HELIOS_Q_RATIO_BAND

#: Published Table 3 point ratios, for the failure message (not asserted).
_PUB = HELIOS_PUBLISHED_RATIOS

#: Published Table 3 Saturn FIFO average JCT (s) — the V2 absolute rung's
#: reference point.  It CORROBORATES the placer choice; it does not decide it
#: (the four-cluster mean ratio error does, and the band does not either —
#: see the module docstring).
_PUB_SATURN_FIFO_JCT = HELIOS_SATURN_FIFO_JCT_PUBLISHED


@pytest.mark.trace_full
@pytest.mark.skipif(
    not os.environ.get("FLEETSIM_HELIOS_FULL"),
    reason="set FLEETSIM_HELIOS_FULL=1 to download and replay the full Helios trace",
)
def test_helios_four_cluster_ratio_bands_and_ranks() -> None:
    """Opt-in full replay: all four clusters, FIFO and SJF, Sept window,
    strict scan, **September-max** per-VC capacity sizing
    (``pool_snapshot="max"``), **``consolidate`` placement** — the placer
    that best reproduces the paper, selected by measuring all four policies
    end to end on the real trace (module docstring: mean absolute relative
    error 3.4 % vs 27.4 % for ``first_fit`` and 29.5 % for ``spread``).

    Asserts (plan §2 V1(f)): every cluster's FIFO/SJF queuing ratio in
    [3, 25]x; every JCT ratio in [1.3, 8]x — **all four**, no ``xfail``;
    Saturn has the highest JCT ratio and Uranus the lowest; and the
    queuing-share-of-FIFO-JCT ordering is Saturn > Venus > Earth > Uranus.
    These all hold on the real trace.

    Measured (v0.7a, re-run on the real trace for this rung): JCT ratios
    Saturn 6.87 / Venus 3.21 / Earth 2.95 / Uranus 1.51 against published
    6.59 / 3.07 / 2.87 / 1.49; q ratios 19.55 / 7.78 / 10.67 / 5.08 against
    18.5 / 5.68 / 16.4 / 4.51; q-shares 0.901 / 0.791 / 0.730 / 0.422
    against 0.897 / 0.818 / 0.693 / 0.425.

    v0.6 -> v0.7: Saturn used to land at 8.75 and was honestly ``xfail``ed
    against the 8.0 ceiling.  The cause was NOT "FirstFit fragments large
    gangs" (Saturn's biggest gang is 200 GPU / 25 nodes, and the VC
    carrying 84 % of the gap tops out at 56 GPU) — it was sub-node
    stranding of whole-node capacity: 97 % of that VC's jobs are 1-GPU, and
    first-fit-by-id placement leaves free chips as 1..7-GPU remainders no
    >=8-GPU job can use.  With ``consolidate`` placement Saturn lands at
    6.87 against a published 6.59, and its absolute FIFO JCT at 55,978 s
    against a published 55,984 s.  Every other cluster moved toward
    published too.  The BAND IS UNCHANGED — see ``_JCT_RATIO_BAND`` for why
    it must not be tightened.

    WHAT THIS RUNG DOES NOT PROVE: the bands and both rank invariants are
    also satisfied by the ``spread`` control arm, whose absolute FIFO JCTs
    are 27-99 % off published.  Selecting a placer needs the absolute rung
    in ``test_helios_saturn_placement_model_is_load_bearing``.

    SIZING: uses ``pool_snapshot="max"``.  The plan's Sept-1 snapshot
    mis-sizes Uranus (whose per-VC quota drifts within September) and
    breaks the Uranus-lowest rank and the q-share ordering; sizing each VC
    to its September-max quota recovers both (see the harness notes)."""
    clusters = ("Venus", "Earth", "Saturn", "Uranus")
    jct_ratio: dict[str, float] = {}
    q_ratio: dict[str, float] = {}
    q_share: dict[str, float] = {}
    for cl in clusters:
        fifo = per_vc_replay(
            cl, "2020-09", "fifo", pool_snapshot="max", placement=_PLACEMENT
        )
        sjf = per_vc_replay(
            cl, "2020-09", "sjf", pool_snapshot="max", placement=_PLACEMENT
        )
        assert fifo["placement"] == sjf["placement"] == _PLACEMENT
        # No windowed job may be DROPPED (every VC with jobs must be sized;
        # a VC absent from the pool sizing loses its jobs silently under a
        # snapshot — "max" must lose none) and none may be TRUNCATED (all
        # replayed jobs terminal) — either bias falls exactly on the
        # long-waiting jobs that carry the effect.
        assert fifo["n_dropped"] == 0, (cl, "fifo", fifo["dropped_vcs"], fifo["n_windowed"])
        assert sjf["n_dropped"] == 0, (cl, "sjf", sjf["dropped_vcs"], sjf["n_windowed"])
        assert fifo["n_terminal"] == fifo["n_jobs"], (cl, "fifo", fifo["n_terminal"], fifo["n_jobs"])
        assert sjf["n_terminal"] == sjf["n_jobs"], (cl, "sjf", sjf["n_terminal"], sjf["n_jobs"])
        jct_ratio[cl] = fifo["avg_jct"] / sjf["avg_jct"]
        q_ratio[cl] = fifo["avg_queuing"] / sjf["avg_queuing"]
        q_share[cl] = fifo["avg_queuing"] / fifo["avg_jct"]

    lo, hi = _JCT_RATIO_BAND
    qlo, qhi = _Q_RATIO_BAND
    # Queuing ratio in band for every cluster (all four reproduce here).
    for cl in clusters:
        assert qlo <= q_ratio[cl] <= qhi, (
            f"{cl} queuing ratio {q_ratio[cl]:.2f} outside {_Q_RATIO_BAND} "
            f"(published {_PUB[cl]['q']}); all={q_ratio}"
        )
    # JCT ratio in band for EVERY cluster (v0.7: no xfail — Saturn's
    # v0.6 overshoot was a placement-model gap, now closed).
    for cl in clusters:
        assert lo <= jct_ratio[cl] <= hi, (
            f"{cl} JCT ratio {jct_ratio[cl]:.2f} outside {_JCT_RATIO_BAND} "
            f"(published {_PUB[cl]['jct']}); all={jct_ratio}"
        )

    # Cross-cluster rank: Saturn the strongest SJF-over-FIFO JCT effect,
    # Uranus the weakest (plan §2 V1(f)); q-share ordering Saturn > Venus >
    # Earth > Uranus.  Both reproduce under Sept-max sizing.
    assert jct_ratio["Saturn"] == max(jct_ratio.values()), jct_ratio
    assert jct_ratio["Uranus"] == min(jct_ratio.values()), jct_ratio
    assert (
        q_share["Saturn"] > q_share["Venus"] > q_share["Earth"] > q_share["Uranus"]
    ), q_share

    # The AGGREGATE quantity that actually selects the placement model (the
    # module docstring): mean absolute relative error of the four JCT ratios
    # against published.  Measured 3.4 % for the shipped `consolidate`
    # against 27.4 % for `first_fit` and 29.5 % for `spread`, so a 10 % bar
    # separates them by ~2.7x while staying far looser than the measurement.
    # Asserted here because it costs nothing extra (the ratios are already
    # computed) and because it is the claim a reader is asked to trust.
    mean_abs_err = sum(
        abs(jct_ratio[cl] - _PUB[cl]["jct"]) / _PUB[cl]["jct"] for cl in clusters
    ) / len(clusters)
    assert mean_abs_err < 0.10, (mean_abs_err, jct_ratio)


@pytest.mark.trace_full
@pytest.mark.skipif(
    not os.environ.get("FLEETSIM_HELIOS_FULL"),
    reason="set FLEETSIM_HELIOS_FULL=1 to download and replay the full Helios trace",
)
def test_helios_saturn_placement_model_is_load_bearing() -> None:
    """The placement-model choice is TESTED, not assumed: replay Saturn
    under all three distinguishable placers — ``first_fit`` (the v0.6
    baseline / engine default), the shipped ``consolidate``, and the
    ``spread`` control arm — and assert what each one does.

    Three claims:

    1. ``first_fit``'s JCT ratio really is OUT of band (8.75 > 8.0) and
       ``consolidate``'s is in it (6.87) — the v0.6 gap and its closure.
    2. Under the SHIPPED configuration, ``consolidate``'s absolute FIFO JCT
       lands within +/-20 % of the published 55,984 s (measured 55,978 s)
       while ``first_fit``'s (75,329 s, +35 %) and ``spread``'s (81,096 s,
       +45 %) do not.
    3. The ratio band alone is NOT a discriminator: ``spread``'s ratio
       (7.48) sits *inside* [1.3, 8] while being the worst placer on every
       absolute quantity.  Asserted here so nobody later mistakes "in band"
       for "reproduces the paper".

    WHAT (2) IS AND IS NOT.  It is *corroboration* under the shipped
    configuration, not the discriminator: measured on the real trace, under
    the descending-id FIFO tie-break of docs/validation.md §4.5 — one knob,
    nothing to do with placement — the same comparison INVERTS (``first_fit``
    62,549 s = +11.7 %, ratio 7.26; ``consolidate`` 63,347 s = +13.2 %, ratio
    7.78, both in band and both inside +/-20 %).  So these asserts pin the
    shipped result, and the quantity that actually selects the placer is the
    four-cluster mean absolute ratio error in the module docstring (3.4 % vs
    27.4 % vs 29.5 %), which survives that perturbation, plus the mechanism
    metric.  Do not read a passing (2) as "Saturn's absolute JCT proves the
    placer".

    ``best_fit`` is deliberately not replayed: it is bit-identical to
    ``consolidate`` on this single-level fleet (asserted at unit scale in
    ``tests/test_placement.py``, and confirmed on the full trace in the
    v0.7a sweep).  Saturn only — it carries the effect; each FIFO replay of
    it costs ~3 minutes, so this rung runs ~10 minutes.
    """
    lo, hi = _JCT_RATIO_BAND
    ratios: dict[str, float] = {}
    fifo_jct: dict[str, float] = {}
    for placement in ("first_fit", "consolidate", "spread"):
        fifo = per_vc_replay(
            "Saturn", "2020-09", "fifo", pool_snapshot="max", placement=placement
        )
        sjf = per_vc_replay(
            "Saturn", "2020-09", "sjf", pool_snapshot="max", placement=placement
        )
        assert fifo["n_dropped"] == sjf["n_dropped"] == 0
        assert fifo["n_terminal"] == fifo["n_jobs"]
        assert sjf["n_terminal"] == sjf["n_jobs"]
        ratios[placement] = fifo["avg_jct"] / sjf["avg_jct"]
        fifo_jct[placement] = fifo["avg_jct"]

    # (1) the v0.6 out-of-band baseline, and its closure.
    assert ratios["first_fit"] > hi, ratios
    assert lo <= ratios["consolidate"] <= hi, ratios
    assert ratios["consolidate"] < ratios["first_fit"], ratios

    # (2) the ABSOLUTE rung, under the SHIPPED configuration.  Asserted
    # loosely (+/-20%) because the paper's analysis window is unpublished and
    # the capacity model is a choice (§1, §4.4); a tighter bar would assert
    # luck.  See the docstring for why this rung corroborates rather than
    # selects — it inverts under the §4.5 tie-break perturbation.
    def _err(placement: str) -> float:
        return abs(fifo_jct[placement] - _PUB_SATURN_FIFO_JCT) / _PUB_SATURN_FIFO_JCT

    assert _err("consolidate") < 0.20, fifo_jct
    assert _err("first_fit") > 0.20, fifo_jct
    assert _err("spread") > 0.20, fifo_jct
    assert _err("consolidate") < _err("first_fit"), fifo_jct
    assert _err("consolidate") < _err("spread"), fifo_jct

    # (3) "in band" is NOT "reproduces the paper": the control arm passes
    # the band while being the worst placer on the absolute quantity.
    assert lo <= ratios["spread"] <= hi, ratios
    assert fifo_jct["spread"] > fifo_jct["first_fit"] > fifo_jct["consolidate"], fifo_jct
