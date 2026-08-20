"""The comfort/savings frontier.

"Occupants will hate being too warm" is the first thing a facility manager says,
and the honest answer is not a promise -- it is this curve. The operator picks a
comfort ceiling; the system reports the lowest demand it can then commit to and
what that is worth. Nobody has to trust a number they did not choose the inputs
for.

This is also the answer to "how do I get more savings": widen the band. The
controller cannot conjure flexibility that the comfort budget forbids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.baselines import NoControl
from control.mpc import MPCConfig
from eval.find_target import search
from eval.run_month import build_context, run_controller
from sim.thermal import Comfort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--ceilings", type=float, nargs="+", default=[24.0, 25.0, 26.0, 27.0, 28.0])
    ap.add_argument("--comfort-budget-pct", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results/frontier.json")
    args = ap.parse_args()

    rows = []
    for ceiling in args.ceilings:
        comfort = Comfort(t_max=ceiling)
        ctx = build_context(args.building, args.start, args.end, comfort=comfort)
        base = run_controller(ctx, NoControl()).metrics
        res = search(ctx, args.comfort_budget_pct, lo_frac=0.45, iters=args.iters,
                     cfg=MPCConfig())
        chosen = res.get("chosen")
        row = {
            "comfort_ceiling_c": ceiling,
            "band_width_k": ceiling - comfort.t_min,
            "uncontrolled_peak_kw": res["uncontrolled_peak_kw"],
            "uncontrolled_bill_inr": base["bill_inr"],
            "target_kw": res.get("target_kw"),
            "shave_pct": res.get("shave_pct"),
            "bill_inr": chosen["bill_inr"] if chosen else None,
            "saving_inr": (base["bill_inr"] - chosen["bill_inr"]) if chosen else None,
            "comfort_violation_pct": chosen["comfort_violation_pct"] if chosen else None,
        }
        rows.append(row)
        print(f"  ceiling {ceiling:4.1f} C (band {row['band_width_k']:.0f} K): "
              f"target {row['target_kw']:6.0f} kW ({row['shave_pct']:4.1f}% shave)  "
              f"saving Rs {row['saving_inr']:9,.0f}  comfort {row['comfort_violation_pct']:.2f}%")

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
