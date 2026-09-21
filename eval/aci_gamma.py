"""Is the adaptive layer's result at each tier a property of its step size?

ACI has one free parameter, gamma, the rate at which the per-horizon offsets
move -- and it is stated in the units of the series. The audit runs it at
0.35 everywhere, which on a 100 kW building with 60 kW intervals moves the
bound by about 0.6% of the interval per step and on a 3,000 MW city with
750 MW intervals by 0.05%: the same number is a twelve-times slower layer at
system scale. So the sweep is in relative terms, gamma = kappa x the mean
split-conformal interval width at the audited lead, for every audited
series, with nothing retrained: the split-conformal predictions are in
results/conformal_year_<series>.parquet and ACI is a pass over them in time
order. Reports the rolling in-band share and the by-fold range at each
kappa, the same statistics the audit reports at its default, plus the
default absolute gamma for reference.

Outputs results/aci_gamma.json; eval/paper_tables.py reads it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.conformal_audit import (BAND_90, BAND_Q95, QUANTILES, ROLL_WINDOW_ORIGINS,
                                  _band_stats, by_month)
from forecast.conformal import adaptive_conformal, rolling_coverage
from forecast.features import HORIZON_STEPS

RESULTS = ROOT / "results"
KAPPAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
DEFAULT_GAMMA = 0.35
SERIES = ("Fox_office_Gaylord", "IN_Delhi")
LEAD = 4


def split_width(year: pd.DataFrame, lead: int = LEAD) -> float:
    sl = year[year["horizon"] == lead]
    return float((sl["split_q95"] - sl["split_q05"]).mean())


def replay(year: pd.DataFrame, gamma: float, lead: int = LEAD) -> dict:
    year = year.sort_values(["target_time", "horizon"]).reset_index(drop=True)
    h, y = year["horizon"].to_numpy(), year["y"].to_numpy()
    for q in (0.05, 0.95):
        col = f"split_q{int(q*100):02d}"
        if gamma == 0.0:
            year[f"aci_q{int(q*100):02d}"] = year[col].to_numpy()
        else:
            year[f"aci_q{int(q*100):02d}"], _ = adaptive_conformal(
                year[col].to_numpy(), y, h, q, gamma=gamma, n_horizons=HORIZON_STEPS)
    for q in (0.25, 0.50, 0.75):      # by_month only reads q05/q95; keep the schema whole
        year[f"aci_q{int(q*100):02d}"] = year[f"split_q{int(q*100):02d}"]
    sl = year[year["horizon"] == lead]
    ys = sl["y"].to_numpy()
    lo, hi = sl["aci_q05"].to_numpy(), sl["aci_q95"].to_numpy()
    cov = rolling_coverage(((ys >= lo) & (ys <= hi)).astype(float), ROLL_WINDOW_ORIGINS)
    blw = rolling_coverage((ys <= hi).astype(float), ROLL_WINDOW_ORIGINS)
    months = by_month(year)
    return {
        "gamma": gamma,
        "band_90": _band_stats(cov.tolist(), BAND_90),
        "band_q95": _band_stats(blw.tolist(), BAND_Q95),
        "by_fold_cov90_min": float(min(r["aci_cov90"] for r in months)),
        "by_fold_cov90_max": float(max(r["aci_cov90"] for r in months)),
        "by_fold_cov90_mean": float(np.mean([r["aci_cov90"] for r in months])),
        "mean_width_kw": float((sl["aci_q95"] - sl["aci_q05"]).mean()),
    }


def main() -> None:
    out = {}
    for key in SERIES:
        p = RESULTS / f"conformal_year_{key}.parquet"
        if not p.exists():
            continue
        year = pd.read_parquet(p)
        W = split_width(year)
        rows = []
        for k in KAPPAS:
            r = replay(year, k * W)
            r["kappa"] = k
            rows.append(r)
        ref = replay(year, DEFAULT_GAMMA)
        ref["kappa"] = DEFAULT_GAMMA / W
        out[key] = {"split_width": W, "default_gamma": DEFAULT_GAMMA, "default": ref, "sweep": rows}
        print(f"== {key}   split interval width at lead {LEAD}: {W:.1f}; default gamma {DEFAULT_GAMMA} = kappa {DEFAULT_GAMMA / W:.4f}")
        for r in rows + [ref]:
            print(f"   kappa {r['kappa']:<7.4f} gamma {r['gamma']:<7.3f} cov90 in-band {r['band_90']['in_band_pct']:5.1f}%  "
                  f"by fold [{r['by_fold_cov90_min']:.3f}, {r['by_fold_cov90_max']:.3f}] "
                  f"mean {r['by_fold_cov90_mean']:.3f}  width {r['mean_width_kw']:.0f}")
    (RESULTS / "aci_gamma.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
