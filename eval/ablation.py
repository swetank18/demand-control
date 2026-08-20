"""The ablation. Same optimiser, same simulator, same tariff, same month.
Only the forecaster changes.

This is Section 6.4, and it is the single most important experiment in the
project. The round-1 feedback was that no model was visible and that the
reasoning was vague. The reason it was invisible is that nothing in the pitch
would have changed if the forecaster had been replaced by a constant. So we
replace it with a constant, and with five other things, and measure what the
transformer and the bill do.

The design rule is that exactly one thing varies. The demand target, the MILP
settings, the RC parameters, the comfort budget, the PV array, the solar
quantiles, the tariff and the random seed are identical on every row. What
changes is the file the base-load quantiles are read from. Anything else moving
would give a judge a second explanation for the difference, and the argument
would be dead.

Two axes come out on one table: forecast quality on the left, control outcome on
the right. That layout is the argument.
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

from control.mpc import ChanceConstrainedMPC, MPCConfig
from eval.run_month import build_context, run_controller, uncontrolled_peak_kw
from forecast.baselines import REGISTRY
from forecast.metrics import score_all
from forecast.sources import TensorForecast
from sim.stress import STRESSES

MODELS = ROOT / "models"
RESULTS = ROOT / "results"

#: Reported in this order. The two ends bracket the table: the static margin is
#: what a distribution engineer does today with no model at all, and perfect
#: foresight is the bound on what any model could ever be worth.
ORDER = [
    "static_margin", "persistence", "seasonal_naive", "climatology",
    "linear_quantile", "lightgbm_quantile", "neural_quantile", "perfect_foresight",
]
OURS = "lightgbm_quantile"


def usable_headroom_kw(fc: TensorForecast, n: int, target_kw: float, horizon: int = 4) -> dict:
    """Controllable power the ceiling constraint actually permits, in kW.

    This is the quantity Section 4 is about. The constraint is

        base_q95[t] + controllable[t] - solar_q05[t]  <=  D_target

    so the headroom handed to the optimiser at each interval is

        D_target - base_q95[t] + solar_q05[t]

    A forecaster that is uncertain, or that is simply a large constant, spends
    that headroom on margin it does not need. The kilowatts it gives back are the
    value of the model stated in the units the problem is actually about --
    which is a better answer to "why is there a model" than any metric.

    Evaluated one hour ahead (``horizon`` = 4 steps), the point at which the
    optimiser is committing the current interval.
    """
    q95 = np.array([fc.base(k, horizon, "q95")[-1] for k in range(n - horizon)])
    q05_pv = np.array([fc.pv(k, horizon, "q05")[-1] for k in range(n - horizon)])
    head = target_kw - q95 + q05_pv
    return {
        "usable_headroom_kw": float(np.mean(np.maximum(head, 0.0))),
        "usable_headroom_p10_kw": float(np.quantile(head, 0.10)),
        "headroom_negative_pct": float(100.0 * np.mean(head < 0)),
    }


def forecast_scores(tensor: pd.DataFrame, window: tuple[str, str]) -> dict:
    """Score the tensor over exactly the ablation window, so the left-hand
    columns describe the forecasts the optimiser actually consumed."""
    t = tensor[(tensor["target_time"] >= pd.Timestamp(window[0]))
               & (tensor["target_time"] <= pd.Timestamp(window[1]))]
    preds = {q: t[f"q{int(q * 100):02d}"].to_numpy() for q in (0.05, 0.25, 0.50, 0.75, 0.95)}
    s = score_all(t["actual"].to_numpy(), preds).as_dict()
    return {k: s[k] for k in ("n", "pinball_mean", "crps", "coverage_90", "below_q95",
                              "calibration_error", "sharpness_90", "mae_median")}


def run(
    building: str, start: str, end: str, keys: list[str] | None = None,
    demand_target_kw: float | None = None, stress_name: str = "none",
    pv_kwp: float = 150.0, cfg: MPCConfig | None = None, seed: int = 0,
) -> dict:
    keys = keys or [k for k in ORDER if (MODELS / building / "tensors" / f"{k}.parquet").exists()]
    missing = [k for k in keys if not (MODELS / building / "tensors" / f"{k}.parquet").exists()]
    if missing:
        raise SystemExit(f"no forecast tensor for {missing}; run eval/forecast_eval.py first")

    stress = STRESSES[stress_name]() if stress_name != "none" else None
    ctx = build_context(building, start, end, pv_kwp=pv_kwp, stress=stress)

    if demand_target_kw is None:
        f = RESULTS / "demand_targets.json"
        d = json.loads(f.read_text()).get(building) if f.exists() else None
        demand_target_kw = d["target_kw"] if d and d.get("feasible") else 0.85 * uncontrolled_peak_kw(ctx)
    ctx["demand_target_kw"] = demand_target_kw
    cfg = cfg or MPCConfig()

    index = ctx["exog"].index
    rows = []
    print(f"\n== ablation | {building} | {start} to {end} | stress={stress_name}")
    print(f"   demand target {demand_target_kw:.1f} kW held fixed on every row; "
          f"only the base-load forecast changes")

    for key in keys:
        tensor = pd.read_parquet(MODELS / building / "tensors" / f"{key}.parquet")
        fc = TensorForecast(tensor, index, ctx["pvq"])
        ctrl = ChanceConstrainedMPC(
            ctx["tariff"], fc, cfg, risk_quantile="q95", solar_quantile="q05",
            name=REGISTRY[key].name, demand_target_kw=demand_target_kw,
        )
        r = run_controller(ctx, ctrl, label=REGISTRY[key].name, seed=seed)
        row = {
            "key": key, "forecaster": REGISTRY[key].name,
            **forecast_scores(tensor, (start, end)),
            **{k: r.metrics[k] for k in (
                "bill_inr", "energy_charge", "demand_charge", "energy_kwh",
                "peak_kw", "peak_kva", "billed_kva", "peak_at", "ceiling_breaches",
                "worst_breach_kw", "first_breach_at", "comfort_violation_pct",
                "comfort_kelvin_hours", "load_factor")},
            **usable_headroom_kw(fc, len(index), demand_target_kw),
        }
        row["solve_ms_mean"] = r.metrics.get("solve_ms_mean")
        rows.append(row)
        print(f"   {row['forecaster']:<28} pinball {row['pinball_mean']:7.3f}  "
              f"cov {row['coverage_90']:.3f}  breaches {row['ceiling_breaches']:3d}  "
              f"peak {row['peak_kva']:6.1f} kVA  Rs {row['bill_inr']:11,.0f}  "
              f"headroom {row['usable_headroom_kw']:6.1f} kW")

    df = pd.DataFrame(rows)
    ours = df[df.key == OURS].iloc[0] if (df.key == OURS).any() else None
    if ours is not None:
        df["bill_vs_ours"] = df["bill_inr"] - ours["bill_inr"]
        df["breaches_vs_ours"] = df["ceiling_breaches"] - ours["ceiling_breaches"]
        df["headroom_vs_ours_kw"] = df["usable_headroom_kw"] - ours["usable_headroom_kw"]

    meta = {
        "building": building, "window": [start, end], "stress": stress_name,
        "demand_target_kw": demand_target_kw, "pv_kwp": pv_kwp,
        "tariff": ctx["tariff"].order_ref,
        "contract_demand_kva": ctx["tariff"].contract_demand_kva,
        "held_fixed": [
            "demand target", "MILP horizon and settings", "RC parameters",
            "comfort budget", "PV array and solar quantiles", "tariff", "seed",
        ],
        "varied": "the base-load quantile forecast only",
    }
    return {"meta": meta, "rows": df.to_dict("records")}


ABL_COLS = [
    ("forecaster", "Forecaster feeding the optimiser", "{}"),
    ("pinball_mean", "Pinball", "{:.3f}"),
    ("coverage_90", "Cov 90%", "{:.3f}"),
    ("ceiling_breaches", "Breaches", "{:d}"),
    ("peak_kva", "Peak kVA", "{:.1f}"),
    ("bill_inr", "Bill INR", "{:,.0f}"),
    ("usable_headroom_kw", "Usable headroom kW", "{:.1f}"),
    ("comfort_violation_pct", "Comfort violated %", "{:.2f}"),
]


def to_markdown(res: dict) -> str:
    m = res["meta"]
    lines = [
        f"### Ablation — {m['building']}, {m['window'][0]} to {m['window'][1]}"
        + (f", stress: {m['stress']}" if m["stress"] != "none" else ""),
        "",
        f"Demand target **{m['demand_target_kw']:.0f} kW**, held fixed on every row. "
        f"Contract demand {m['contract_demand_kva']:.0f} kVA. Tariff: {m['tariff']}.",
        "",
        "Held fixed: " + ", ".join(m["held_fixed"]) + ". Varied: " + m["varied"] + ".",
        "",
        "Forecast quality on the left, control outcome on the right. A breach is a "
        "billing block whose average exceeded the target; each one is a permanent "
        "monthly cost, which is why the count and not the average is the safety metric.",
        "",
        "| " + " | ".join(c[1] for c in ABL_COLS) + " |",
        "| " + " | ".join("---" for _ in ABL_COLS) + " |",
    ]
    for r in res["rows"]:
        cells = []
        for key, _, fmt in ABL_COLS:
            v = r.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            elif r["key"] == "static_margin" and key in ("pinball_mean", "coverage_90"):
                # a constant is not a forecast; scoring it as one flatters it
                cells.append(f"{fmt.format(v)}*")
            else:
                cells.append(fmt.format(v))
        if r["key"] == OURS:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    ours = next((r for r in res["rows"] if r["key"] == OURS), None)
    const = next((r for r in res["rows"] if r["key"] == "static_margin"), None)
    if ours and const:
        lines += [
            "",
            f"\\* the static margin is one number for the whole month, so its pinball loss "
            f"and coverage describe a constant rather than a forecast.",
            "",
            f"**Replacing the forecaster with a constant** costs "
            f"{const['ceiling_breaches'] - ours['ceiling_breaches']:+d} ceiling breaches, "
            f"₹{const['bill_inr'] - ours['bill_inr']:+,.0f} on the month, and "
            f"{const['usable_headroom_kw'] - ours['usable_headroom_kw']:+.1f} kW of usable "
            f"headroom ({100 * (const['usable_headroom_kw'] - ours['usable_headroom_kw']) / max(ours['usable_headroom_kw'], 1e-9):+.0f}%).",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--stress", nargs="+", default=["none"])
    ap.add_argument("--demand-target-kw", type=float, default=None)
    ap.add_argument("--pv-kwp", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    payload, md = {}, []
    for stress in args.stress:
        res = run(args.building, args.start, args.end, args.keys,
                  args.demand_target_kw, stress, args.pv_kwp)
        payload[stress] = res
        md.append(to_markdown(res))
    (args.out / f"ablation_{args.building}.json").write_text(json.dumps(payload, indent=2, default=float))
    (args.out / "ablation.md").write_text("\n\n".join(md) + "\n")
    print("\n" + "\n\n".join(md))


if __name__ == "__main__":
    main()
