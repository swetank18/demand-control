"""Interpretability: SHAP, feature-group ablation, and the evening we got wrong.

Section 6.7. Three artefacts, and the third one is the one that earns
credibility.

*SHAP.* Which features the boosters actually lean on, measured rather than
assumed. Expect recent lags and the same-time-of-day mean to dominate, with
temperature behind them. If something silly is at the top, that is a bug found
before a judge finds it.

*Feature-group ablation.* Drop a group, retrain, measure the damage. SHAP says
what the model used; this says what it needed. They are different questions and
the second one is the one that tells you whether the weather feed is worth
paying for.

*The worked error case.* The single evening in the test month where the forecast
was worst, what the q95 margin did about it, and why it happened. Showing your
worst case builds more credibility than showing your best, and it is the honest
answer to "what if the model is wrong".
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

from eval.forecast_eval import _slice, series_of, supervised
from forecast.baselines import LightGBMQuantile
from forecast.conformal import calibrate
from forecast.features import FEATURE_COLS
from forecast.metrics import score_all
from forecast.splits import SPLIT

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = ROOT / "models"

#: Feature groups for the drop-one-out study. Grouped rather than one-at-a-time
#: because the individual lags are near-substitutes: drop ``lag_1`` alone and
#: ``lag_2`` absorbs it, and you conclude the lags do not matter.
GROUPS: dict[str, list[str]] = {
    "recent lags": ["lag_1", "lag_2", "lag_4", "lag_8", "last"],
    "daily and weekly lags": ["lag_96", "lag_192", "lag_672", "tod_mean_4w"],
    "rolling statistics": ["rmean_4", "rmean_96", "rmean_672",
                           "rstd_4", "rstd_96", "rstd_672"],
    "weather now": ["t_out_origin", "dtemp_1h"],
    "weather forecast": ["t_out_fut", "cdd_fut", "cloud_fut"],
    "calendar": ["hod_sin", "hod_cos", "dow_sin", "dow_cos",
                 "is_weekend", "is_holiday", "doy_sin", "doy_cos", "solar_elev"],
}


def _fit(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], n_estimators: int,
         seed: int = 0) -> tuple[dict, dict]:
    """Fit on a column subset and score on validation, uncalibrated.

    Validation rather than test: the drop-one-out study is a diagnostic, and
    burning the test block on six extra retrains is exactly the discipline
    failure the frozen split exists to prevent.
    """
    import forecast.baselines as bl

    old = bl.FEATURE_COLS
    try:
        bl.FEATURE_COLS = cols
        m = LightGBMQuantile(n_estimators=n_estimators, seed=seed).fit(train, None, valid=valid)
        pred = m.predict(valid)
    finally:
        bl.FEATURE_COLS = old
    return score_all(valid["y"].to_numpy(), pred).as_dict(), m.best_iteration


def feature_ablation(building: str, n_estimators: int = 250, sample: int = 600_000,
                     seed: int = 0) -> dict:
    """Retrain without each group and report the degradation.

    Subsampled and shortened relative to the production fit -- this is a
    comparison between seven runs under identical settings, so the absolute
    numbers are not the point and the settings are stated so nobody quotes them
    as the headline.
    """
    sup = supervised(building)
    tr = _slice(sup, SPLIT.train_start, SPLIT.train_end)
    va = _slice(sup, SPLIT.valid_start, SPLIT.valid_end)
    if sample and len(tr) > sample:
        tr = tr.sample(sample, random_state=seed)

    print(f"\n== feature-group ablation | {building} | "
          f"{len(tr):,} training rows, {n_estimators} rounds, scored on validation")
    full, _ = _fit(tr, va, list(FEATURE_COLS), n_estimators, seed)
    print(f"   {'all features':<26} pinball {full['pinball_mean']:7.3f}")

    rows = []
    for name, cols in GROUPS.items():
        kept = [c for c in FEATURE_COLS if c not in cols]
        s, _ = _fit(tr, va, kept, n_estimators, seed)
        d = s["pinball_mean"] - full["pinball_mean"]
        rows.append({
            "group": name, "dropped": cols, "n_features_kept": len(kept),
            "pinball_mean": s["pinball_mean"],
            "degradation": d,
            "degradation_pct": 100.0 * d / max(full["pinball_mean"], 1e-9),
            "coverage_90": s["coverage_90"], "mae_median": s["mae_median"],
        })
        print(f"   drop {name:<21} pinball {s['pinball_mean']:7.3f}  "
              f"({d:+.3f}, {rows[-1]['degradation_pct']:+.1f}%)")

    rows.sort(key=lambda r: -r["degradation"])
    return {"building": building, "settings": {"n_estimators": n_estimators,
                                               "train_rows": int(len(tr)), "seed": seed,
                                               "scored_on": "validation"},
            "full": full, "groups": rows}


def shap_report(building: str, n_estimators: int = 250, sample: int = 400_000,
                explain: int = 20_000, seed: int = 0) -> dict:
    """Mean |SHAP| per feature for the q50 and q95 boosters.

    Both, because they are different models with different jobs: the median
    booster predicts the load, the q95 booster predicts how badly the load
    could surprise you, and it is the second one whose output enters the
    constraint.
    """
    import shap

    sup = supervised(building)
    tr = _slice(sup, SPLIT.train_start, SPLIT.train_end)
    va = _slice(sup, SPLIT.valid_start, SPLIT.valid_end)
    if sample and len(tr) > sample:
        tr = tr.sample(sample, random_state=seed)
    X = va[FEATURE_COLS].sample(min(explain, len(va)), random_state=seed)

    m = LightGBMQuantile(quantiles=(0.50, 0.95), n_estimators=n_estimators, seed=seed)
    m.fit(tr, None, valid=va)

    out = {}
    for q in (0.50, 0.95):
        values = shap.TreeExplainer(m.models[q]).shap_values(X)
        imp = pd.Series(np.abs(values).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
        out[f"q{int(q * 100):02d}"] = {
            "mean_abs_shap": {k: float(v) for k, v in imp.items()},
            "top": list(imp.head(8).index),
        }
    return {"building": building, "n_explained": int(len(X)),
            "settings": {"n_estimators": n_estimators, "train_rows": int(len(tr)), "seed": seed},
            "shap": out}


def shap_figure(rep: dict, abl: dict, out: Path, top: int = 14) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    ax = axes[0]
    imp = pd.Series(rep["shap"]["q95"]["mean_abs_shap"]).sort_values().tail(top)
    med = pd.Series(rep["shap"]["q50"]["mean_abs_shap"]).reindex(imp.index)
    y = np.arange(len(imp))
    ax.barh(y + 0.20, imp.values, height=0.38, color="#c1440e", label="q95 booster")
    ax.barh(y - 0.20, med.values, height=0.38, color="#2f6690", label="q50 booster")
    ax.set_yticks(y, imp.index, fontsize=9)
    ax.set_xlabel("mean |SHAP| (kW)")
    ax.set_title("What the model uses\nmean absolute SHAP on held-out data",
                 loc="left", fontsize=11.5, weight="bold")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(loc="lower right")

    ax = axes[1]
    g = sorted(abl["groups"], key=lambda r: r["degradation"])
    names = [r["group"] for r in g]
    vals = [r["degradation"] for r in g]
    colors = ["#c1440e" if v > 0 else "#94a3b8" for v in vals]
    ax.barh(np.arange(len(g)), vals, color=colors)
    ax.set_yticks(np.arange(len(g)), names, fontsize=9)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("increase in pinball loss when the group is removed (kW)")
    ax.set_title("What the model needs\ndrop the group, retrain, measure the damage",
                 loc="left", fontsize=11.5, weight="bold")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle(f"Interpretability — {rep['building']}", fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def worst_evening(building: str, target_kw: float | None = None,
                  key: str = "lightgbm_quantile") -> dict:
    """Find the day the forecast was worst, and report what the margin did.

    The metric is the largest single exceedance of q95 by the actual, because
    that is the failure the chance constraint exists to prevent -- not the
    largest absolute error, which could be a harmless overprediction.
    """
    path = MODELS / building / "tensors" / f"{key}.parquet"
    if not path.exists():
        path = MODELS / building / "forecast_test.parquet"
    t = pd.read_parquet(path)
    t["exceedance"] = t["actual"] - t["q95"]
    t["day"] = pd.DatetimeIndex(t["target_time"]).normalize()

    per_day = t.groupby("day").agg(
        worst_exceedance_kw=("exceedance", "max"),
        mean_abs_error_kw=("actual", "size"),
    )
    per_day["mean_abs_error_kw"] = t.groupby("day").apply(
        lambda d: float((d["actual"] - d["q50"]).abs().mean()), include_groups=False)
    per_day["q95_hit_rate"] = t.groupby("day").apply(
        lambda d: float((d["actual"] <= d["q95"]).mean()), include_groups=False)
    worst = per_day.sort_values("worst_exceedance_kw", ascending=False).iloc[0]
    day = worst.name

    d = t[t["day"] == day].sort_values(["target_time", "horizon"])
    at = d.loc[d["exceedance"].idxmax()]
    band = d[d["horizon"] == 4].sort_values("target_time")

    out = {
        "building": building, "day": str(day.date()),
        "worst_exceedance_kw": float(worst["worst_exceedance_kw"]),
        "worst_exceedance_at": str(at["target_time"]),
        "worst_exceedance_horizon_steps": int(at["horizon"]),
        "actual_kw": float(at["actual"]), "q95_kw": float(at["q95"]),
        "q50_kw": float(at["q50"]),
        "margin_q95_minus_q50_kw": float(at["q95"] - at["q50"]),
        "error_of_the_median_kw": float(at["actual"] - at["q50"]),
        "day_q95_hit_rate": float(worst["q95_hit_rate"]),
        "day_mae_kw": float(worst["mean_abs_error_kw"]),
        "month_worst_day_rank_note": (
            "the worst single interval of the worst day in the test month"),
        "series": {
            "t": [str(x) for x in band["target_time"]],
            "actual": band["actual"].round(2).tolist(),
            "q05": band["q05"].round(2).tolist(),
            "q50": band["q50"].round(2).tolist(),
            "q95": band["q95"].round(2).tolist(),
        },
    }
    if target_kw is not None:
        out["demand_target_kw"] = float(target_kw)
        out["headroom_at_worst_kw"] = float(target_kw - at["q95"])
        out["would_have_breached_planning_on_median"] = bool(
            at["actual"] > target_kw >= at["q50"])
    return out


def error_case_figure(case: dict, out: Path) -> None:
    s = case["series"]
    t = pd.to_datetime(s["t"])
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.fill_between(t, s["q05"], s["q95"], color="#c1440e", alpha=0.16,
                    label="q05 to q95, one hour ahead")
    ax.plot(t, s["q50"], color="#c1440e", lw=1.6, ls="--", label="q50 (median forecast)")
    ax.plot(t, s["q95"], color="#c1440e", lw=1.2, alpha=0.7)
    ax.plot(t, s["actual"], color="#0f172a", lw=2.0, label="actual")

    bad = pd.Timestamp(case["worst_exceedance_at"])
    ax.scatter([bad], [case["actual_kw"]], s=120, color="#d00000", zorder=6,
               marker="v", edgecolor="k", linewidth=0.7)
    ax.annotate(
        f"worst interval of the month\nactual {case['actual_kw']:.0f} kW vs "
        f"q95 {case['q95_kw']:.0f} kW  ({case['worst_exceedance_kw']:+.0f} kW)",
        xy=(bad, case["actual_kw"]), xytext=(14, 26), textcoords="offset points",
        fontsize=9.5, color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef2f2", ec="#fecaca"))

    if case.get("demand_target_kw"):
        ax.axhline(case["demand_target_kw"], color="#d00000", ls="--", lw=2.0)
        ax.annotate("demand ceiling", xy=(0.012, case["demand_target_kw"]),
                    xycoords=("axes fraction", "data"), color="#d00000",
                    va="bottom", fontsize=9.5, weight="bold")

    ax.set_title(
        f"The day we were most wrong — {case['building']}, {case['day']}\n"
        f"the median was out by {case['error_of_the_median_kw']:.0f} kW; the q95 margin "
        f"was carrying {case['margin_q95_minus_q50_kw']:.0f} kW and absorbed all but "
        f"{case['worst_exceedance_kw']:.0f} kW of it",
        loc="left", fontsize=11.5, weight="bold")
    ax.set_ylabel("base load (kW)")
    ax.grid(alpha=0.28)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--n-estimators", type=int, default=250)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    target = None
    f = RESULTS / "demand_targets.json"
    if f.exists():
        d = json.loads(f.read_text()).get(args.building)
        target = d["target_kw"] if d and d.get("feasible") else None

    payload: dict = {}
    abl = feature_ablation(args.building, n_estimators=args.n_estimators)
    payload["feature_ablation"] = abl

    if not args.skip_shap:
        rep = shap_report(args.building, n_estimators=args.n_estimators)
        payload["shap"] = rep
        shap_figure(rep, abl, FIGS / "07_interpretability.png")
        print("\n   SHAP top features")
        for q in ("q50", "q95"):
            print(f"     {q}: " + ", ".join(rep["shap"][q]["top"][:6]))

    case = worst_evening(args.building, target)
    payload["worst_case"] = case
    error_case_figure(case, FIGS / "08_worst_case.png")
    print(f"\n   worst evening: {case['day']} at {case['worst_exceedance_at'][11:16]}, "
          f"actual {case['actual_kw']:.0f} kW vs q95 {case['q95_kw']:.0f} kW "
          f"({case['worst_exceedance_kw']:+.0f} kW), day q95 hit rate "
          f"{case['day_q95_hit_rate']:.3f}")

    (args.out / "interpretability.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n   -> {args.out / 'interpretability.json'}")


if __name__ == "__main__":
    main()
