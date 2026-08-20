"""Generate the results table. Every number on every slide comes from here.

Nothing in this file computes money. It calls the bill engine, which is the
single source of truth, and formats what comes back.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.mpc import MPCConfig
from eval.run_month import (build_context, run_controller, standard_controllers,
                            uncontrolled_peak_kw)
from sim.stress import STRESSES, StaleForecast


def oracle_savings_captured(rows: list[dict]) -> None:
    """Fraction of the *achievable* saving captured, not the margin over a strawman.

    Judges who own buildings want to know how much of the money on the table you
    actually picked up. Beating a strawman by 20% means nothing if the oracle
    was 60% better.
    """
    base = next((r for r in rows if r["controller"] == "No control"), None)
    orac = next((r for r in rows if "oracle" in r["controller"].lower()), None)
    if base is None or orac is None:
        return
    span = base["bill_inr"] - orac["bill_inr"]
    for r in rows:
        r["pct_of_oracle_savings"] = (
            100.0 * (base["bill_inr"] - r["bill_inr"]) / span if abs(span) > 1e-9 else np.nan
        )


def run_table(
    building: str, start: str, end: str, demand_target_kw: float | None,
    stress_name: str = "none", pv_kwp: float = 150.0, cfg: MPCConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    stress = STRESSES[stress_name]() if stress_name != "none" else None
    ctx = build_context(building, start, end, pv_kwp=pv_kwp, stress=stress)

    if demand_target_kw is None:
        tgt_file = ROOT / "results/demand_targets.json"
        if tgt_file.exists():
            d = json.loads(tgt_file.read_text()).get(building)
            if d and d.get("feasible"):
                demand_target_kw = d["target_kw"]
    if demand_target_kw is None:
        demand_target_kw = 0.85 * uncontrolled_peak_kw(ctx)
    ctx["demand_target_kw"] = demand_target_kw

    controllers = standard_controllers(ctx, cfg, demand_target_kw)
    if stress_name == "sensor_dropout":
        s = STRESSES["sensor_dropout"]()
        for c in controllers:
            if hasattr(c, "fc"):
                c.fc = StaleForecast(c.fc, ctx["exog"].index, s.start, s.hours)

    rows, histories = [], {}
    for c in controllers:
        r = run_controller(ctx, c)
        rows.append(r.metrics)
        histories[r.name] = r.history
        print(f"    {r.name:<40} Rs {r.metrics['bill_inr']:11,.0f}  "
              f"peak {r.metrics['peak_kw']:6.1f} kW  breaches {r.metrics['ceiling_breaches']:3d}  "
              f"comfort {r.metrics['comfort_violation_pct']:5.2f}%")

    oracle_savings_captured(rows)
    df = pd.DataFrame(rows)
    meta = {
        "building": building, "window": [start, end], "stress": stress_name,
        "demand_target_kw": demand_target_kw, "pv_kwp": pv_kwp,
        "tariff": ctx["tariff"].order_ref, "contract_demand_kva": ctx["tariff"].contract_demand_kva,
        "stress_detail": ctx["stress"],
    }
    return df, {"meta": meta, "histories": histories}


COLS = [
    ("controller", "Controller", "{}"),
    ("bill_inr", "Bill INR", "{:,.0f}"),
    ("peak_kva", "Peak kVA", "{:.1f}"),
    ("ceiling_breaches", "Ceiling breaches", "{:d}"),
    ("comfort_violation_pct", "Comfort violated %", "{:.2f}"),
    ("pct_of_oracle_savings", "% of oracle savings", "{:.1f}"),
]


def to_markdown(df: pd.DataFrame, meta: dict) -> str:
    lines = [
        f"### {meta['building']}  |  {meta['window'][0]} to {meta['window'][1]}  |  stress: {meta['stress']}",
        "",
        f"Demand target {meta['demand_target_kw']:.0f} kW, contract demand {meta['contract_demand_kva']:.0f} kVA.",
        f"Tariff: {meta['tariff']}",
        "",
        "| " + " | ".join(c[1] for c in COLS) + " |",
        "| " + " | ".join("---" for _ in COLS) + " |",
    ]
    for _, r in df.iterrows():
        cells = []
        for key, _, fmt in COLS:
            v = r.get(key)
            cells.append("-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
        name = cells[0]
        if name.startswith("Ours"):
            cells[0] = f"**{name}**"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--stress", nargs="+", default=["none"], choices=list(STRESSES))
    ap.add_argument("--demand-target-kw", type=float, default=None)
    ap.add_argument("--pv-kwp", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    md_parts, all_rows = [], []
    for stress in args.stress:
        print(f"\n== {args.building} | stress={stress}")
        df, extra = run_table(args.building, args.start, args.end,
                              args.demand_target_kw, stress, args.pv_kwp)
        md_parts.append(to_markdown(df, extra["meta"]))
        df.insert(0, "stress", stress)
        df.insert(0, "building", args.building)
        all_rows.append(df)
        for name, h in extra["histories"].items():
            safe = name.replace(" ", "_").replace(":", "").replace("+", "")
            h.to_parquet(args.out / f"history_{args.building}_{stress}_{safe}.parquet")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(args.out / f"results_{args.building}.csv", index=False)
    md = "\n\n".join(md_parts)
    (args.out / f"results_{args.building}.md").write_text(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
