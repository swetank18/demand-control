"""Export the simulation into a seed bundle for the web app.

The web app must never recompute money. Everything it shows is produced here,
by the same bill engine the results table uses, and shipped as data. That keeps
the deployed demo and the paper numbers identical by construction.
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

from tariff.bill import compute_bill, price_series, window_series
from tariff.schema import Tariff

RESULTS = ROOT / "results"

SCENARIOS = {
    "none": ("Normal month", "June 2017 as it actually happened, no injected failure."),
    "heatwave": ("Heatwave", "Outdoor temperature +6 °C and base load +25% for three days, beyond anything in the forecaster's training window."),
    "sensor_dropout": ("Sensor dropout", "The forecast input freezes for two hours. The controller keeps acting on its last good estimate, which is what happens when a BMS point dies."),
    "outage": ("Grid outage", "Grid import forced to zero for two hours; only critical load is served. The controller must ride through on thermal mass."),
}

CONTROLLERS = {
    "No control": ("no_control", "#9aa5b1", "A deadband thermostat at a fixed 24 °C setpoint, water heater always hot, EVs charging flat out on arrival. Price-blind and peak-blind: what an uninstrumented building does today.", 0),
    "Rule based schedule": ("rule_based", "#7a5195", "Tariff-aware but risk-blind: pre-cools ahead of the peak window, throttles through it, pushes deferrable load into cheap hours. No optimisation, no forecast.", 1),
    "MPC on mean forecast": ("mpc_mean", "#2f6690", "The same MILP as ours, planning the demand ceiling on the *median* forecast instead of the 95th percentile. The only line of difference.", 2),
    "Ours: quantile + chance constrained": ("ours", "#c1440e", "Substitutes base-load q95 and solar q05 into the demand-ceiling constraint. A chance constraint by quantile substitution.", 3),
    "MPC on perfect forecast (16 h)": ("mpc_oracle_rolling", "#7b9e89", "Ours, but handed the true future base load. Still limited to a 16-hour rolling horizon, so it is not the achievable bound.", 4),
    "Perfect foresight oracle": ("oracle", "#2a9d8f", "One MILP over the entire billing month with perfect foresight. The genuine upper bound, and the denominator for '% of oracle savings'.", 5),
}


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace(":", "").replace("+", "")


def series_payload(h: pd.DataFrame, tariff: Tariff) -> dict:
    """Columnar, block-averaged, rounded. Columnar because 24 of these ship to a
    browser and objects-per-row would triple the payload for no benefit."""
    rule = f"{tariff.billing_interval_minutes}min"
    blocks = h["grid_kw"].resample(rule).mean()

    # running bill, computed here so the client never does arithmetic on money
    price = price_series(h.index, tariff)
    energy_cum = (h["grid_kw"].clip(lower=0) * 0.25 * price).cumsum()
    floor = tariff.contract_demand_kva * tariff.billing_demand_floor_pct / 100.0
    running_kva = (blocks / 0.95).cummax()
    demand_cum = np.maximum(running_kva, floor) * tariff.demand_charge_per_kva

    agg = pd.DataFrame({
        "grid_kw": blocks,
        "t_indoor": h["t_indoor"].resample(rule).mean(),
        "t_lo": h["t_lo"].resample(rule).min(),
        "t_hi": h["t_hi"].resample(rule).max(),
        "t_out": h["t_out"].resample(rule).mean(),
        "base_kw": h["base_kw"].resample(rule).mean(),
        "hvac_kw": h["hvac_kw"].resample(rule).mean(),
        "pv_kw": h["pv_kw"].resample(rule).mean(),
        "viol_k": h["comfort_violation_k"].resample(rule).max(),
        "energy_cum": energy_cum.resample(rule).last(),
    }).dropna()
    agg["bill_cum"] = agg["energy_cum"] + demand_cum.reindex(agg.index).ffill()

    return {
        "t": [int(ts.timestamp()) for ts in agg.index],
        "grid_kw": [round(float(v), 1) for v in agg.grid_kw],
        "t_indoor": [round(float(v), 2) for v in agg.t_indoor],
        "t_lo": [round(float(v), 1) for v in agg.t_lo],
        "t_hi": [round(float(v), 1) for v in agg.t_hi],
        "t_out": [round(float(v), 1) for v in agg.t_out],
        "base_kw": [round(float(v), 1) for v in agg.base_kw],
        "hvac_kw": [round(float(v), 1) for v in agg.hvac_kw],
        "pv_kw": [round(float(v), 1) for v in agg.pv_kw],
        "viol_k": [round(float(v), 2) for v in agg.viol_k],
        "bill_cum": [round(float(v), 0) for v in agg.bill_cum],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--out", type=Path, default=Path("/home/swetank/hackit/ampcast/seed"))
    args = ap.parse_args()
    out = args.out
    (out / "series").mkdir(parents=True, exist_ok=True)

    man = json.loads((ROOT / "data/cache/manifest.json").read_text())
    bp = man["buildings"][args.building]
    tariff = Tariff.load(ROOT / "tariff/orders/tnerc_2026.json")
    tariff = dataclasses.replace(tariff, contract_demand_kva=bp["contract_demand_kva"])

    res = pd.read_csv(RESULTS / f"results_{args.building}.csv")
    target = float(res["demand_target_kw"].dropna().iloc[0])

    # ---- runs + series ---------------------------------------------------
    runs, index = [], []
    for _, r in res.iterrows():
        name = r["controller"]
        if name not in CONTROLLERS:
            continue
        key, color, blurb, order = CONTROLLERS[name]
        f = RESULTS / f"history_{args.building}_{r['stress']}_{_slug(name)}.parquet"
        if not f.exists():
            continue
        h = pd.read_parquet(f)
        h.index = pd.to_datetime(h.index)
        bill = compute_bill(h["grid_kw"], tariff, 0.95)
        sid = f"{r['stress']}__{key}"
        (out / "series" / f"{sid}.json").write_text(json.dumps(series_payload(h, tariff)))
        index.append(sid)
        runs.append({
            "id": sid, "scenario": r["stress"], "controller": key, "controller_label": name,
            "color": color, "blurb": blurb, "order": int(order),
            "demand_target_kw": round(target, 1),
            "bill_inr": round(float(r["bill_inr"]), 2),
            "energy_charge": round(float(r["energy_charge"]), 2),
            "demand_charge": round(float(r["demand_charge"]), 2),
            "energy_kwh": round(float(r["energy_kwh"]), 1),
            "peak_kw": round(float(r["peak_kw"]), 2),
            "peak_kva": round(float(r["peak_kva"]), 2),
            "peak_at": str(r["peak_at"]),
            "ceiling_breaches": int(r["ceiling_breaches"]),
            "worst_breach_kw": round(float(r["worst_breach_kw"]), 2),
            "first_breach_at": None if pd.isna(r.get("first_breach_at")) else str(r["first_breach_at"]),
            "comfort_violation_pct": round(float(r["comfort_violation_pct"]), 3),
            "comfort_kelvin_hours": round(float(r["comfort_kelvin_hours"]), 2),
            "pct_of_oracle_savings": None if pd.isna(r.get("pct_of_oracle_savings")) else round(float(r["pct_of_oracle_savings"]), 2),
            "solve_ms_mean": None if pd.isna(r.get("solve_ms_mean")) else round(float(r["solve_ms_mean"]), 2),
            "energy_by_window": {k: {kk: round(float(vv), 2) for kk, vv in v.items()}
                                 for k, v in bill.energy_by_window.items()},
        })

    # ---- everything else --------------------------------------------------
    cal = json.loads((RESULTS / "calibration_summary.json").read_text())
    frontier = json.loads((RESULTS / "frontier.json").read_text())
    mv = json.loads((RESULTS / "mv_baseline_length.json").read_text())
    targets = json.loads((RESULTS / "demand_targets.json").read_text())

    bundle = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "building": {
            "id": args.building, "label": bp["label"], "site": man["site"],
            "sqm": bp["sqm"], "contract_demand_kva": bp["contract_demand_kva"],
            "hvac_capacity_kw": bp["hvac_capacity_kw"],
            "ua_w_per_m2k": round(bp["thermal"]["ua_w_per_m2k"], 3),
            "time_constant_h": round(bp["thermal"]["time_constant_h"], 1),
            "c_kwh_per_k": round(bp["thermal"]["c_kwh_per_k"], 1),
            "why_chosen": bp["why_chosen"],
            "hvac_share_of_meter": round(bp["hvac_share_of_meter"], 4),
        },
        "tariff": {
            "state": tariff.state, "category": tariff.category, "order_ref": tariff.order_ref,
            "energy_rate": tariff.energy_rate,
            "demand_charge_per_kva": tariff.demand_charge_per_kva,
            "billing_interval_minutes": tariff.billing_interval_minutes,
            "contract_demand_kva": tariff.contract_demand_kva,
            "billing_demand_floor_pct": tariff.billing_demand_floor_pct,
            "electricity_duty_pct": tariff.electricity_duty_pct,
            "tod_windows": [dataclasses.asdict(w) for w in tariff.tod_windows],
        },
        "scenarios": [{"key": k, "label": v[0], "description": v[1]} for k, v in SCENARIOS.items()],
        "demand_target_kw": round(target, 1),
        "demand_target_search": targets.get(args.building, {}),
        "runs": runs,
        "series_index": index,
        "calibration": cal,
        "frontier": frontier,
        "mv_baseline_length": mv,
        "window": {"start": "2017-06-01T00:00:00", "end": "2017-06-30T23:45:00"},
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))

    n = sum(f.stat().st_size for f in (out / "series").glob("*.json"))
    print(f"  bundle.json      {len(json.dumps(bundle))/1024:8.1f} KB  ({len(runs)} runs)")
    print(f"  series/*.json    {n/1024:8.1f} KB  ({len(index)} series)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
