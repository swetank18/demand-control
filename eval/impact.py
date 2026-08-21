"""Impact, computed. Nothing in this file is asserted.

Section 7. Every figure here is produced by running the system, and every
assumption that turns a measured quantity into a rupee is a named constant at
the top of the file that gets printed alongside the number it produced. An
honest figure with its assumption printed beside it beats a bigger one a judge
can poke.

Four tiers, plus the study Section 4 asks for.

Section 4  the static-derating comparison. Tune a fixed margin until it breaches
           exactly as often as we do, then compare the capacity each one leaves
           usable. The difference is what the model is worth, in kilowatts.
Tier 1     technical: breaches, peak reduction, share of the achievable benefit.
Tier 2     operational: how much EV charging each controller can absorb before
           the ceiling goes, converted into years of deferred upgrade.
Tier 3     financial: demand charge, time-of-day energy, avoided capital.
Tier 4     scale: the arithmetic from one building to one distribution utility,
           given as arithmetic rather than as a market-size claim.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.baselines import NoControl, RuleBased
from control.mpc import ChanceConstrainedMPC, MPCConfig
from eval.run_month import build_context, run_controller
from forecast.baselines import REGISTRY
from forecast.sources import TensorForecast
from sim.thermal import EVFleet

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
OURS = "lightgbm_quantile"


# --- assumptions, named, printed with every number they touch ---------------

ASSUMPTIONS = {
    "transformer_upgrade_inr_per_kva": {
        "value": 3_500.0,
        "why": "indicative turnkey cost of distribution transformer capacity in India, "
               "including civil and switchgear works. Order of magnitude only; a real "
               "society would price it from a quotation.",
    },
    "discount_rate": {
        "value": 0.10,
        "why": "nominal discount rate for present-valuing a deferred capital outlay.",
    },
    "ev_load_growth_pct_per_year": {
        "value": 30.0,
        "why": "compound annual growth in on-site EV charging energy. India's EV stock "
               "has been growing faster than this; 30% is deliberately conservative "
               "because a higher rate makes the deferral look better, not worse.",
    },
    "billing_months_per_year": {
        "value": 12.0,
        "why": "the demand charge is a monthly maximum, so a month of saving annualises "
               "by twelve. This assumes June is a representative month, which it is not: "
               "it is the hottest, so this figure is an upper bound and is labelled as one.",
    },
    "buildings_per_feeder": {
        "value": 8.0,
        "why": "commercial buildings of this size on one 11 kV distribution feeder. "
               "Stated as an assumption, not a survey.",
    },
    "feeders_per_discom": {
        "value": 5_000.0,
        "why": "order of magnitude for a mid-sized state distribution utility.",
    },
}


def A(key: str) -> float:
    return float(ASSUMPTIONS[key]["value"])


# --- Section 4: what a static margin costs you ------------------------------

def static_margin_study(
    ctx: dict, target_kw: float, cfg: MPCConfig, ours_row: dict,
    percentiles=(0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00),
) -> dict:
    """Sweep a forecast-free fixed margin and find the one that is as safe as we are.

    The engineer's alternative to a model is a static derating: never plan on more
    than a fixed allowance for the uncontrollable load. Sized for the worst evening
    of the year it is far too conservative on the other 360; sized for a typical
    evening it fails on the bad one, and the bad one is the only one that matters.

    So we sweep it, hold everything else fixed, and read off the margin that
    matches our breach count. The gap in usable headroom at equal safety is the
    value of the forecast, in the units the problem is actually about.
    """
    from eval.ablation import usable_headroom_kw
    from forecast.baselines import ConstantMargin
    from eval.forecast_eval import _slice, supervised
    from forecast.splits import SPLIT

    sup = supervised(ctx["building"])
    train = _slice(sup, SPLIT.train_start, SPLIT.train_end)
    index = ctx["exog"].index
    n = len(index)

    # ConstantMargin's own quantiles are the marginal training distribution, which
    # is one particular derating (its q95). The sweep needs the *allowance* to move,
    # so the risk column is set to the p-th percentile of the training load and the
    # rest of the set is capped at it to stay monotone. p = 0.95 reproduces
    # ConstantMargin exactly, which is the row that appears in the ablation table.
    marginal = ConstantMargin().fit(train, None).levels

    rows = []
    for p in percentiles:
        allowance = float(np.quantile(train["y"], p))
        levels = {q: min(v, allowance) for q, v in marginal.items()}
        levels[0.95] = allowance
        tensor = pd.DataFrame({
            "origin": np.repeat(index, 1), "target_time": index, "horizon": 1,
            **{f"q{int(q * 100):02d}": np.full(n, levels[q]) for q in sorted(levels)},
            "actual": ctx["exog"]["base_kw"].to_numpy(),
        })
        # a constant is horizon-free, so replicate it across the planning horizon
        full = pd.concat([tensor.assign(horizon=h,
                                        target_time=index + pd.Timedelta(minutes=15 * h))
                          for h in range(1, cfg.horizon + 1)], ignore_index=True)
        fc = TensorForecast(full, index, ctx["pvq"], horizon=cfg.horizon)
        ctrl = ChanceConstrainedMPC(ctx["tariff"], fc, cfg, risk_quantile="q95",
                                    solar_quantile="q05", name=f"static p{int(p * 100)}",
                                    demand_target_kw=target_kw)
        r = run_controller(ctx, ctrl, label=f"static p{int(p * 100)}")
        rows.append({
            "percentile": p,
            "allowance_kw": allowance,
            "breaches": r.metrics["ceiling_breaches"],
            "peak_kva": r.metrics["peak_kva"],
            "bill_inr": r.metrics["bill_inr"],
            "comfort_violation_pct": r.metrics["comfort_violation_pct"],
            **usable_headroom_kw(fc, n, target_kw),
        })
        print(f"     static p{int(p * 100):>3}  allowance {rows[-1]['allowance_kw']:6.1f} kW  "
              f"breaches {rows[-1]['breaches']:3d}  headroom {rows[-1]['usable_headroom_kw']:6.1f} kW  "
              f"Rs {rows[-1]['bill_inr']:11,.0f}")

    # the safest static margin that is no worse than us on breaches
    matched = [r for r in rows if r["breaches"] <= ours_row["ceiling_breaches"]]
    best = max(matched, key=lambda r: r["usable_headroom_kw"]) if matched else None
    out = {"sweep": rows, "ours": {k: ours_row[k] for k in
                                   ("ceiling_breaches", "usable_headroom_kw", "bill_inr", "peak_kva")}}
    if best:
        out["matched"] = best
        out["recovered_headroom_kw"] = ours_row["usable_headroom_kw"] - best["usable_headroom_kw"]
        out["recovered_headroom_pct"] = (
            100.0 * out["recovered_headroom_kw"] / max(best["usable_headroom_kw"], 1e-9))
        out["bill_gap_inr"] = best["bill_inr"] - ours_row["bill_inr"]
        out["statement"] = (
            f"A forecast-free static margin tuned to breach no more often than we do "
            f"(p{int(best['percentile'] * 100)}, {best['breaches']} breaches) leaves "
            f"{best['usable_headroom_kw']:.1f} kW usable. We leave "
            f"{ours_row['usable_headroom_kw']:.1f} kW at {ours_row['ceiling_breaches']} breaches. "
            f"The forecast recovers {out['recovered_headroom_kw']:.1f} kW "
            f"({out['recovered_headroom_pct']:+.0f}%) of capacity at equal safety."
        )
    return out


# --- Tier 2: how much EV can each controller absorb -------------------------

def ev_headroom_sweep(
    building: str, start: str, end: str, target_kw: float, cfg: MPCConfig,
    levels_kwh=(220, 500, 800, 1100, 1400, 1700, 2000, 2400), pv_kwp: float = 150.0,
) -> dict:
    """Raise the site's EV charging energy until each controller breaks the ceiling.

    This is the operational number, and it is the one a building owner or a
    managing committee actually understands: not "we saved 4%" but "you can add
    this much EV charging before you have to pay for a bigger connection."

    The uncoordinated controller is the status quo -- charge on arrival, flat
    out. It fails first. The gap between where it fails and where we fail,
    converted through the site's EV load growth rate, is years of deferred
    capital.
    """
    print(f"\n== EV headroom sweep | {building}")
    rows = []
    for kwh in levels_kwh:
        # the charger has to be able to deliver it inside the 9-hour window
        ev = EVFleet(max_kw=max(60.0, kwh / 6.0), required_kwh_per_day=float(kwh))
        ctx = build_context(building, start, end, pv_kwp=pv_kwp, ev=ev)
        ctx["demand_target_kw"] = target_kw

        tensor = MODELS / building / "tensors" / f"{OURS}.parquet"
        if not tensor.exists():
            tensor = MODELS / building / "forecast_test.parquet"
        fc = TensorForecast(pd.read_parquet(tensor), ctx["exog"].index, ctx["pvq"])

        entries = {
            "uncoordinated": NoControl(),
            "rule_based": RuleBased(ctx["tariff"]),
            "ours": ChanceConstrainedMPC(ctx["tariff"], fc, cfg, risk_quantile="q95",
                                         solar_quantile="q05", name="ours",
                                         demand_target_kw=target_kw),
        }
        row = {"ev_kwh_per_day": float(kwh), "ev_max_kw": ev.max_kw}
        for name, ctrl in entries.items():
            r = run_controller(ctx, ctrl, label=name)
            row[f"{name}_breaches"] = r.metrics["ceiling_breaches"]
            row[f"{name}_peak_kva"] = r.metrics["peak_kva"]
            row[f"{name}_bill_inr"] = r.metrics["bill_inr"]
            row[f"{name}_comfort_pct"] = r.metrics["comfort_violation_pct"]
        rows.append(row)
        print(f"   {kwh:>5.0f} kWh/day   uncoordinated {row['uncoordinated_breaches']:>3d}  "
              f"rule-based {row['rule_based_breaches']:>3d}  ours {row['ours_breaches']:>3d}  breaches")

    def first_failure(key: str) -> float | None:
        for r in rows:
            if r[f"{key}_breaches"] > 0:
                return r["ev_kwh_per_day"]
        return None

    limits = {k: first_failure(k) for k in ("uncoordinated", "rule_based", "ours")}
    out = {"sweep": rows, "first_breach_kwh_per_day": limits}

    g = A("ev_load_growth_pct_per_year") / 100.0
    if limits["uncoordinated"] and limits["ours"] and limits["ours"] > limits["uncoordinated"]:
        years = float(np.log(limits["ours"] / limits["uncoordinated"]) / np.log(1 + g))
        out["deferral_years"] = years
        out["deferral_statement"] = (
            f"Uncoordinated charging breaches the ceiling at "
            f"{limits['uncoordinated']:.0f} kWh/day of EV load; ours holds to "
            f"{limits['ours']:.0f} kWh/day. At {A('ev_load_growth_pct_per_year'):.0f}% annual "
            f"growth in EV charging that gap is {years:.1f} years before the connection "
            f"has to be upgraded."
        )
    elif limits["ours"] is None and limits["uncoordinated"] is not None:
        out["deferral_statement"] = (
            f"Uncoordinated charging breaches at {limits['uncoordinated']:.0f} kWh/day. "
            f"Ours did not breach anywhere in the swept range (up to "
            f"{levels_kwh[-1]} kWh/day), so the deferral is longer than this sweep can measure."
        )
    return out


# --- Tiers 1, 3, 4 ----------------------------------------------------------

def tier1(abl: pd.DataFrame, nocontrol: dict) -> dict:
    ours = abl[abl.key == OURS].iloc[0]
    ceil = abl[abl.key == "perfect_foresight"]
    ceil = ceil.iloc[0] if len(ceil) else None

    out = {
        "uncontrolled_peak_kva": nocontrol["peak_kva"],
        "our_peak_kva": float(ours["peak_kva"]),
        "peak_reduction_kva": nocontrol["peak_kva"] - float(ours["peak_kva"]),
        "peak_reduction_pct": 100.0 * (nocontrol["peak_kva"] - float(ours["peak_kva"]))
                              / max(nocontrol["peak_kva"], 1e-9),
        "uncontrolled_breaches": nocontrol["ceiling_breaches"],
        "our_breaches": int(ours["ceiling_breaches"]),
        "our_comfort_violation_pct": float(ours["comfort_violation_pct"]),
    }
    if ceil is not None:
        span = nocontrol["bill_inr"] - float(ceil["bill_inr"])
        out["achievable_saving_inr"] = span
        out["captured_saving_inr"] = nocontrol["bill_inr"] - float(ours["bill_inr"])
        out["pct_of_achievable_captured"] = (
            100.0 * out["captured_saving_inr"] / span if abs(span) > 1e-9 else None)
    return out


def tier3(abl: pd.DataFrame, nocontrol: dict, tariff, deferral_years: float | None) -> dict:
    ours = abl[abl.key == OURS].iloc[0]
    dc_rate = tariff.demand_charge_per_kva
    peak_cut = nocontrol["peak_kva"] - float(ours["peak_kva"])

    demand_saving = peak_cut * dc_rate
    energy_saving = nocontrol["energy_charge"] - float(ours["energy_charge"])
    total_saving = nocontrol["bill_inr"] - float(ours["bill_inr"])

    out = {
        "formula_demand": "peak kVA reduction x INR/kVA/month",
        "demand_charge_saving_inr_per_month": demand_saving,
        "demand_charge_rate_inr_per_kva": dc_rate,
        "energy_charge_saving_inr_per_month": energy_saving,
        "total_bill_saving_inr_per_month": total_saving,
        "total_bill_saving_pct": 100.0 * total_saving / max(nocontrol["bill_inr"], 1e-9),
        "annualised_upper_bound_inr": total_saving * A("billing_months_per_year"),
        "annualised_caveat": ASSUMPTIONS["billing_months_per_year"]["why"],
    }
    if deferral_years:
        upgrade = tariff.contract_demand_kva * A("transformer_upgrade_inr_per_kva")
        r = A("discount_rate")
        out["avoided_capital"] = {
            "formula": "contract kVA x INR/kVA, present-valued over the deferral",
            "upgrade_cost_inr": upgrade,
            "deferral_years": deferral_years,
            "npv_of_deferral_inr": float(upgrade * (1 - (1 + r) ** -deferral_years)),
            "assumptions": {k: ASSUMPTIONS[k] for k in
                            ("transformer_upgrade_inr_per_kva", "discount_rate")},
        }
    return out


def tier4(tier3_out: dict) -> dict:
    per_building = tier3_out["total_bill_saving_inr_per_month"]
    per_feeder = per_building * A("buildings_per_feeder")
    per_discom = per_feeder * A("feeders_per_discom")
    return {
        "arithmetic": "saving per building per month x buildings per feeder x feeders per discom",
        "per_building_inr_per_month": per_building,
        "per_feeder_inr_per_month": per_feeder,
        "per_discom_inr_per_month": per_discom,
        "per_discom_inr_per_year_upper_bound": per_discom * A("billing_months_per_year"),
        "assumptions": {k: ASSUMPTIONS[k] for k in ("buildings_per_feeder", "feeders_per_discom")},
        "note": (
            "This is arithmetic, not a market size. It assumes every building on every "
            "feeder is like this one, has a demand charge, and is instrumented. The point "
            "of the multiplication is the shape of the number, not its precision."
        ),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--pv-kwp", type=float, default=150.0)
    ap.add_argument("--skip-ev-sweep", action="store_true")
    ap.add_argument("--skip-static-sweep", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "impact.json")
    args = ap.parse_args()

    abl_path = RESULTS / f"ablation_{args.building}.json"
    if not abl_path.exists():
        raise SystemExit("run eval/ablation.py first; impact is computed from its rows")
    payload = json.loads(abl_path.read_text())["none"]
    abl = pd.DataFrame(payload["rows"])
    target = payload["meta"]["demand_target_kw"]
    cfg = MPCConfig()

    ctx = build_context(args.building, args.start, args.end, pv_kwp=args.pv_kwp)
    ctx["demand_target_kw"] = target
    ctx["building"] = args.building
    nc = run_controller(ctx, NoControl(), label="No control").metrics
    print(f"\n== impact | {args.building} | target {target:.0f} kW")
    print(f"   status quo: peak {nc['peak_kva']:.1f} kVA, "
          f"{nc['ceiling_breaches']} breaches, Rs {nc['bill_inr']:,.0f}")

    ours = abl[abl.key == OURS].iloc[0].to_dict()

    out = {"building": args.building, "window": [args.start, args.end],
           "demand_target_kw": target, "assumptions": ASSUMPTIONS,
           "status_quo": {k: nc[k] for k in ("peak_kva", "peak_kw", "bill_inr",
                                             "energy_charge", "demand_charge",
                                             "ceiling_breaches", "comfort_violation_pct")}}

    if not args.skip_static_sweep:
        print("\n== static margin sweep (Section 4)")
        out["static_margin_study"] = static_margin_study(ctx, target, cfg, ours)
        if "statement" in out["static_margin_study"]:
            print("\n   " + out["static_margin_study"]["statement"])

    ev = {}
    if not args.skip_ev_sweep:
        ev = ev_headroom_sweep(args.building, args.start, args.end, target, cfg,
                               pv_kwp=args.pv_kwp)
        out["ev_headroom"] = ev
        if ev.get("deferral_statement"):
            print("\n   " + ev["deferral_statement"])

    out["tier1_technical"] = tier1(abl, nc)
    out["tier3_financial"] = tier3(abl, nc, ctx["tariff"], ev.get("deferral_years"))
    out["tier4_scale"] = tier4(out["tier3_financial"])

    args.out.write_text(json.dumps(out, indent=2, default=float))
    t1, t3, t4 = out["tier1_technical"], out["tier3_financial"], out["tier4_scale"]
    print(f"\n   Tier 1  peak {t1['peak_reduction_kva']:.1f} kVA lower "
          f"({t1['peak_reduction_pct']:.1f}%), breaches {t1['uncontrolled_breaches']} -> {t1['our_breaches']}, "
          f"{t1.get('pct_of_achievable_captured') or float('nan'):.0f}% of achievable saving captured")
    print(f"   Tier 3  Rs {t3['total_bill_saving_inr_per_month']:,.0f}/month "
          f"({t3['total_bill_saving_pct']:.1f}%), of which demand charge "
          f"Rs {t3['demand_charge_saving_inr_per_month']:,.0f}")
    print(f"   Tier 4  Rs {t4['per_discom_inr_per_month']:,.0f}/month across one discom "
          f"(arithmetic, not a market size)")
    print(f"\n   -> {args.out}")


if __name__ == "__main__":
    main()
