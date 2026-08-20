"""Calibration report for the quantile forecaster.

Calibration is the safety property of this system. The controller substitutes
q95 into the demand-ceiling constraint; if q95 is not really a 95th percentile
then the ceiling is defended in name only. So this module exists to make the
claim falsifiable: coverage table, pinball loss, reliability diagram, and
coverage by horizon (because the controller leans hardest on the far end).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def load(building: str, models: Path) -> tuple[pd.DataFrame, dict]:
    d = models / building
    df = pd.read_parquet(d / "forecast_test.parquet")
    meta = json.loads((d / "meta.json").read_text())
    return df, meta


def reliability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in QUANTILES:
        col = f"q{int(q*100):02d}"
        rows.append({
            "nominal": q,
            "empirical": float((df["actual"] <= df[col]).mean()),
            "pinball": float(np.mean(np.maximum(q * (df["actual"] - df[col]),
                                                (q - 1) * (df["actual"] - df[col])))),
        })
    return pd.DataFrame(rows)


def report(building: str, models: Path, out_dir: Path) -> dict:
    df, meta = load(building, models)
    rel = reliability(df)
    cov90 = float(((df["actual"] >= df["q05"]) & (df["actual"] <= df["q95"])).mean())
    byh = df.groupby("horizon").apply(
        lambda g: pd.Series({
            "coverage_90": float(((g["actual"] >= g["q05"]) & (g["actual"] <= g["q95"])).mean()),
            "below_q95": float((g["actual"] <= g["q95"]).mean()),
            "width": float((g["q95"] - g["q05"]).mean()),
            "mae": float((g["actual"] - g["q50"]).abs().mean()),
        }),
        include_groups=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(rel["nominal"], rel["empirical"], "o-", color="#c1440e", lw=2, label="observed")
    ax.set_xlabel("nominal quantile"); ax.set_ylabel("empirical coverage")
    ax.set_title(f"Reliability diagram\n{building}")
    ax.grid(alpha=0.3); ax.legend()
    for _, r in rel.iterrows():
        ax.annotate(f"{r.empirical:.3f}", (r.nominal, r.empirical),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)

    ax = axes[1]
    ax.plot(byh.index * 0.25, byh["coverage_90"], color="#1b4965", lw=2)
    ax.axhline(0.90, color="k", ls="--", lw=1)
    ax.axhspan(0.85, 0.95, color="#8ecae6", alpha=0.25, label="acceptance band")
    ax.set_xlabel("forecast horizon (hours)"); ax.set_ylabel("90% interval coverage")
    ax.set_title("Coverage by horizon"); ax.set_ylim(0.7, 1.0)
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ax.plot(byh.index * 0.25, byh["width"], color="#c1440e", lw=2, label="q95 - q05 width")
    ax.plot(byh.index * 0.25, byh["mae"], color="#1b4965", lw=2, label="MAE of median")
    ax.set_xlabel("forecast horizon (hours)"); ax.set_ylabel("kW")
    ax.set_title("Interval width and error grow with horizon")
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{building}.png", dpi=130)
    plt.close(fig)

    summary = {
        "building": building,
        "n": int(len(df)),
        "coverage_90": cov90,
        "acceptance_pass": bool(0.85 <= cov90 <= 0.95),
        "reliability": rel.to_dict("records"),
        "coverage_by_horizon": {int(k): float(v) for k, v in byh["coverage_90"].items()},
        "below_q95_by_horizon": {int(k): float(v) for k, v in byh["below_q95"].items()},
        "worst_horizon_coverage": float(byh["coverage_90"].min()),
        "mae_median_kw": float((df["actual"] - df["q50"]).abs().mean()),
        "mean_interval_width_kw": float((df["q95"] - df["q05"]).mean()),
        "adaptive_conformal": meta.get("adaptive_conformal"),
        "weather_assumption": meta.get("weather_assumption"),
    }
    (out_dir / f"calibration_{building}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="+", default=None)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    buildings = args.buildings or sorted(p.name for p in args.models.iterdir() if p.is_dir())
    # Merge rather than replace: running this for one building must not silently
    # drop the other three from the summary the docs are checked against.
    summary_path = args.out / "calibration_summary.json"
    all_s = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print(f"{'building':<24}{'cov90':>8}{'worst-h':>9}{'MAE kW':>9}{'width':>8}  pass")
    for b in buildings:
        s = report(b, args.models, args.out)
        all_s[b] = s
        print(f"{b:<24}{s['coverage_90']:8.3f}{s['worst_horizon_coverage']:9.3f}"
              f"{s['mae_median_kw']:9.1f}{s['mean_interval_width_kw']:8.1f}  {'PASS' if s['acceptance_pass'] else 'FAIL'}")
    summary_path.write_text(json.dumps(all_s, indent=2))


if __name__ == "__main__":
    main()
