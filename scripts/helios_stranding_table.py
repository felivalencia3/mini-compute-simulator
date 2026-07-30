#!/usr/bin/env python
"""Regenerate the placement MECHANISM table in docs/validation.md §4.2.3.

The table reports the v0.7 ``stranded_whole_nodes`` metric — HEALTHY leaves
holding free chips no whole-node gang can claim — time-averaged over a
Helios cluster's per-VC FIFO replays and summed across its VCs, so it reads
as *expected partially-occupied nodes cluster-wide at a random instant*.

Two denominators are printed for a reason stated in the doc: a worse placer
gets a LONGER adaptive horizon, so its extra near-idle tail samples dilute a
whole-run mean and understate the effect.  The comparable number is the one
over the fixed 26-day replay window.

Requires the real Helios ``data.zip`` in the trace cache (~36 MB; the
``FLEETSIM_HELIOS_FULL`` rungs fetch it).  Saturn under three placers is
~9 minutes of CPU.

Usage
-----
    .venv/bin/python scripts/helios_stranding_table.py
    .venv/bin/python scripts/helios_stranding_table.py --cluster Venus \
        --placement first_fit --placement consolidate
"""

from __future__ import annotations

import argparse
import json
import sys

from fleetsim.validation.harness import per_vc_replay

#: The V1 replay window, in seconds — 2020-09-01 .. 2020-09-26 inclusive.
WINDOW_S = 26 * 86_400.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster", default="Saturn")
    ap.add_argument("--month", default="2020-09")
    ap.add_argument("--scheduler", default="fifo")
    ap.add_argument(
        "--placement",
        action="append",
        default=None,
        help="repeatable; defaults to consolidate, first_fit, spread",
    )
    ap.add_argument("--prefix-s", type=float, default=WINDOW_S)
    ap.add_argument("--json", action="store_true", help="dump raw results too")
    args = ap.parse_args(argv)
    placers = args.placement or ["consolidate", "first_fit", "spread"]

    rows = []
    for placement in placers:
        out = per_vc_replay(
            args.cluster,
            args.month,
            args.scheduler,
            pool_snapshot="max",
            placement=placement,
            frag_prefix_s=args.prefix_s,
        )
        assert out["n_dropped"] == 0, out["dropped_vcs"]
        assert out["n_terminal"] == out["n_jobs"], (out["n_terminal"], out["n_jobs"])
        rows.append((placement, out))

    pool = int(rows[0][1]["stranding"]["pool_nodes_total"])
    days = args.prefix_s / 86_400.0
    print(
        f"\n{args.cluster} {args.scheduler.upper()} — stranded_whole_nodes "
        f"(of {pool} nodes), {days:.0f}-day window vs whole run\n"
    )
    head = f"| {'placer':<12} | {'FIFO JCT (s)':>13} | {'partial nodes':>13} | {'stranded chips':>14} | {'partial, whole run':>18} |"
    print(head)
    print("|" + "|".join("-" * (len(c) + 2) for c in head.split("|")[1:-1]) + "|")
    for placement, out in rows:
        s = out["stranding"]
        print(
            f"| {placement:<12} | {out['avg_jct']:13.2f} | {s['nodes_prefix']:13.2f}"
            f" | {s['chips_prefix']:14.1f} | {s['nodes_run']:18.2f} |"
        )
    print()
    if args.json:
        json.dump(
            {
                p: {
                    "avg_jct": o["avg_jct"],
                    "avg_queuing": o["avg_queuing"],
                    "n_queuing": o["n_queuing"],
                    "stranding": o["stranding"],
                    "per_vc": o["per_vc"],
                }
                for p, o in rows
            },
            sys.stdout,
            indent=2,
        )
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
