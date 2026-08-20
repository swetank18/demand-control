"""Find the tightest demand target a building can actually hold for a month.

This is the commercially interesting number, not a tuning knob. An Indian HT
consumer's demand charge is levied on contract demand with a floor, so the
question a facility manager actually asks is "how low can I commit to?" -- and
committing too low is expensive, because exceeding it is what sets the bill.

So: the operator fixes the comfort budget, and we bisect on the demand target to
find the lowest ceiling the controller holds for the whole month without
spending more comfort than the budget allows. The chance constraint is what
makes that commitment safe; a mean-forecast controller asked to hold the same
number breaches it, which is the point of the results table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.mpc import ChanceConstrainedMPC, MPCConfig
from eval.run_month import build_context, run_controller, uncontrolled_peak_kw


def evaluate_target(ctx, target_kw: float, cfg: MPCConfig | None = None,
                    risk_quantile: str = "q95", solar_quantile: str = "q05") -> dict:
    c = ChanceConstrainedMPC(
        ctx["tariff"], ctx["fc_quantile"], cfg or MPCConfig(),
        risk_quantile=risk_quantile, solar_quantile=solar_quantile,
        demand_target_kw=target_kw, name=f"target={target_kw:.0f}",
    )
    r = run_controller(ctx, c)
    m = r.metrics
    m["target_kw"] = target_kw
    return m


def search(
    ctx,
    comfort_budget_pct: float = 2.0,
    lo_frac: float = 0.55,
    hi_frac: float = 1.00,
    iters: int = 6,
    cfg: MPCConfig | None = None,
) -> dict:
    peak = uncontrolled_peak_kw(ctx)
    lo, hi = lo_frac * peak, hi_frac * peak
    trace = []

    # hi must be holdable; if it is not, the comfort budget itself is infeasible
    m_hi = evaluate_target(ctx, hi, cfg)
    trace.append(m_hi)
    if m_hi["ceiling_breaches"] > 0 or m_hi["comfort_violation_pct"] > comfort_budget_pct:
        return {"uncontrolled_peak_kw": peak, "target_kw": hi, "feasible": False, "trace": trace}

    best = m_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = evaluate_target(ctx, mid, cfg)
        trace.append(m)
        ok = m["ceiling_breaches"] == 0 and m["comfort_violation_pct"] <= comfort_budget_pct
        if ok:
            best, hi = m, mid
        else:
            lo = mid
    return {
        "uncontrolled_peak_kw": peak,
        "target_kw": best["target_kw"],
        "shave_pct": 100.0 * (1 - best["target_kw"] / peak),
        "comfort_budget_pct": comfort_budget_pct,
        "feasible": True,
        "chosen": best,
        "trace": trace,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--comfort-budget-pct", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--lo-frac", type=float, default=0.55,
                    help="lower end of the bisection, as a fraction of the uncontrolled peak")
    ap.add_argument("--hi-frac", type=float, default=1.00)
    ap.add_argument("--pv-kwp", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/demand_targets.json")
    args = ap.parse_args()

    ctx = build_context(args.building, args.start, args.end, pv_kwp=args.pv_kwp)
    res = search(ctx, args.comfort_budget_pct, lo_frac=args.lo_frac,
                 hi_frac=args.hi_frac, iters=args.iters)
    res["building"] = args.building
    res["window"] = [args.start, args.end]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing[args.building] = res
    args.out.write_text(json.dumps(existing, indent=2))

    print(f"\n{args.building}: uncontrolled peak {res['uncontrolled_peak_kw']:.0f} kW")
    if res["feasible"]:
        print(f"  tightest holdable target: {res['target_kw']:.0f} kW "
              f"({res['shave_pct']:.1f}% shave) at <= {args.comfort_budget_pct}% comfort violation")
    else:
        print("  no holdable target within the comfort budget")
    print("\n  target_kW  breaches  comfort%   bill_INR")
    for m in sorted(res["trace"], key=lambda x: -x["target_kw"]):
        print(f"  {m['target_kw']:9.0f}  {m['ceiling_breaches']:8d}  {m['comfort_violation_pct']:7.2f}  {m['bill_inr']:11,.0f}")


if __name__ == "__main__":
    main()
