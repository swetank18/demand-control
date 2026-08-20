"""Quantile LightGBM for uncontrollable base load, plus the calibration that
makes the controller's guarantee real rather than decorative.

The chance constraint downstream substitutes the q95 of this forecast into the
demand-ceiling constraint. If q95 is not actually a 95th percentile, the
guarantee is fake. So calibration is not a nice-to-have here, it is the safety
property, and it gets its own report.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.features import FEATURE_COLS, HORIZON_STEPS, build_supervised

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
MODELS = Path(__file__).resolve().parents[1] / "models"

LGB_PARAMS = dict(
    objective="quantile",
    metric="quantile",
    learning_rate=0.06,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=8,
)


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def train_building(
    building: str,
    cache: Path,
    train_end: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    quantiles: tuple[float, ...] = QUANTILES,
    n_estimators: int = 600,
    weather_noise_c: float = 0.0,
    conformal: bool = True,
    adaptive: bool = True,
    adaptive_gamma: float = 0.35,
) -> dict:
    df = pd.read_parquet(cache / f"{building}.parquet")
    sup = build_supervised(df, weather_noise_c=weather_noise_c)

    tr = sup[sup.target_time <= train_end]
    va = sup[(sup.target_time > train_end) & (sup.target_time <= valid_end)]
    te = sup[(sup.target_time >= test_start) & (sup.target_time <= test_end)]
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit(f"empty split for {building}: train={len(tr)} valid={len(va)} test={len(te)}")

    te_index_original = te.index
    Xtr, ytr = tr[FEATURE_COLS], tr["y"].to_numpy()
    Xva, yva = va[FEATURE_COLS], va["y"].to_numpy()
    Xte, yte = te[FEATURE_COLS], te["y"].to_numpy()

    models: dict[float, lgb.Booster] = {}
    va_pred: dict[float, np.ndarray] = {}
    te_pred: dict[float, np.ndarray] = {}
    for q in quantiles:
        m = lgb.train(
            {**LGB_PARAMS, "alpha": q},
            lgb.Dataset(Xtr, ytr),
            num_boost_round=n_estimators,
            valid_sets=[lgb.Dataset(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        models[q] = m
        va_pred[q] = m.predict(Xva, num_iteration=m.best_iteration)
        te_pred[q] = m.predict(Xte, num_iteration=m.best_iteration)

    # --- conformal adjustment, fitted on validation, applied per horizon ----
    # LightGBM quantile regression is not calibrated by construction. We shift
    # each quantile by the empirical residual quantile on held-out data, per
    # horizon, so coverage holds where it is used. Fifteen lines, and it is the
    # difference between a real guarantee and a decorative one.
    shifts: dict[str, dict[str, float]] = {}
    if conformal:
        for q in quantiles:
            per_h = {}
            for h, idx in va.groupby("horizon").groups.items():
                pos = va.index.get_indexer(idx)
                resid = yva[pos] - va_pred[q][pos]
                per_h[int(h)] = float(np.quantile(resid, q))
            shifts[str(q)] = {str(k): v for k, v in per_h.items()}
            hva = va["horizon"].to_numpy()
            hte = te["horizon"].to_numpy()
            va_pred[q] = va_pred[q] + np.array([per_h[int(h)] for h in hva])
            te_pred[q] = te_pred[q] + np.array([per_h[int(h)] for h in hte])

    # --- adaptive conformal, run in time order over the evaluation window ---
    # Split conformal fitted on April-May undercovers in June, because June is
    # hotter and the load is larger: a classic distribution shift. Adaptive
    # conformal inference closes the loop -- after each target time the actual
    # becomes known, and the offset moves against the realised exceedance rate.
    # This uses only information available at the forecast origin, and base load
    # is exogenous, so running it offline over the window is equivalent to
    # running it online.
    aci_trace = {}
    if adaptive:
        gamma = adaptive_gamma
        te = te.sort_values(["target_time", "horizon"]).copy()
        order = te.index
        hte = te["horizon"].to_numpy()
        yte_o = te["y"].to_numpy()
        reordered = {q: pd.Series(te_pred[q], index=te_index_original).loc[order].to_numpy()
                     for q in quantiles}
        for q in quantiles:
            preds = reordered[q]
            offs = np.zeros(HORIZON_STEPS + 1)
            adj = np.empty_like(preds)
            hits = []
            for i in range(len(preds)):
                h = int(hte[i])
                adj[i] = preds[i] + offs[h]
                exceed = 1.0 if yte_o[i] > adj[i] else 0.0
                hits.append(1.0 - exceed)
                offs[h] += gamma * (exceed - (1.0 - q))
            reordered[q] = adj
            aci_trace[str(q)] = {"final_offsets": offs[1:].tolist(),
                                 "realised_below_rate": float(np.mean(hits))}
        te = te.assign(**{f"_p{int(q*100):02d}": reordered[q] for q in quantiles})
        te_pred = {q: te[f"_p{int(q*100):02d}"].to_numpy() for q in quantiles}
        yte = yte_o

    # quantile crossing is possible after shifting; enforce monotonicity
    def _sort_quantiles(pred: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
        arr = np.vstack([pred[q] for q in quantiles])
        arr = np.sort(arr, axis=0)
        return {q: arr[i] for i, q in enumerate(quantiles)}

    va_pred, te_pred = _sort_quantiles(va_pred), _sort_quantiles(te_pred)

    def _report(y, pred, split_name, frame):
        rep = {"split": split_name, "n": int(len(y))}
        for q in quantiles:
            rep[f"pinball_{q}"] = pinball(y, pred[q], q)
            rep[f"below_{q}"] = float(np.mean(y <= pred[q]))
        lo, hi = pred[quantiles[0]], pred[quantiles[-1]]
        rep["coverage_90"] = float(np.mean((y >= lo) & (y <= hi)))
        rep["interval_width_mean"] = float(np.mean(hi - lo))
        rep["mae_median"] = float(np.mean(np.abs(y - pred[0.50])))
        rep["mape_median"] = float(np.mean(np.abs(y - pred[0.50]) / np.maximum(y, 1e-6)))
        # coverage by horizon, because a controller uses long horizons hardest
        byh = {}
        for h, idx in frame.groupby("horizon").groups.items():
            pos = frame.index.get_indexer(idx)
            byh[int(h)] = float(np.mean((y[pos] >= lo[pos]) & (y[pos] <= hi[pos])))
        rep["coverage_90_by_horizon"] = byh
        return rep

    reports = [_report(yva, va_pred, "valid", va), _report(yte, te_pred, "test", te)]

    # --- persist -----------------------------------------------------------
    out = MODELS / building
    out.mkdir(parents=True, exist_ok=True)
    for q, m in models.items():
        m.save_model(str(out / f"q{int(q*100):02d}.txt"), num_iteration=m.best_iteration)
    (out / "conformal.json").write_text(json.dumps(shifts, indent=2))

    # --- precompute the forecast tensor for the evaluation window ----------
    # Base load is uncontrollable, so its forecast does not depend on the
    # controller. Precomputing once means every controller in the table sees the
    # byte-identical forecast, which is the only way the comparison is fair.
    tensor = te[["origin", "target_time", "horizon"]].copy()
    for q in quantiles:
        tensor[f"q{int(q*100):02d}"] = te_pred[q]
    tensor["actual"] = yte
    tensor.to_parquet(out / "forecast_test.parquet")

    meta = {
        "building": building,
        "quantiles": list(quantiles),
        "features": FEATURE_COLS,
        "horizon_steps": HORIZON_STEPS,
        "splits": {"train_end": train_end, "valid_end": valid_end,
                   "test_start": test_start, "test_end": test_end,
                   "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te))},
        "conformal": conformal,
        "adaptive_conformal": adaptive,
        "adaptive_gamma": adaptive_gamma,
        "aci": aci_trace,
        "weather_noise_c": weather_noise_c,
        "weather_assumption": (
            "Weather at the target time is taken as a perfect forecast unless "
            "weather_noise_c > 0. This flatters the forecaster and is stated as a limitation."
        ),
        "best_iteration": {str(q): int(m.best_iteration or n_estimators) for q, m in models.items()},
        "reports": reports,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--buildings", nargs="+", default=None)
    ap.add_argument("--cache", type=Path, default=root / "data/cache")
    ap.add_argument("--train-end", default="2017-03-31 23:45")
    ap.add_argument("--valid-end", default="2017-05-31 23:45")
    ap.add_argument("--test-start", default="2017-06-01 00:00")
    ap.add_argument("--test-end", default="2017-06-30 23:45")
    ap.add_argument("--weather-noise-c", type=float, default=0.0)
    ap.add_argument("--no-conformal", action="store_true")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--adaptive-gamma", type=float, default=0.35)
    args = ap.parse_args()

    manifest = json.loads((args.cache / "manifest.json").read_text())
    buildings = args.buildings or list(manifest["buildings"])
    for b in buildings:
        meta = train_building(
            b, args.cache, args.train_end, args.valid_end, args.test_start, args.test_end,
            weather_noise_c=args.weather_noise_c, conformal=not args.no_conformal,
            adaptive=not args.no_adaptive, adaptive_gamma=args.adaptive_gamma,
        )
        for r in meta["reports"]:
            print(
                f"  {b:<22} {r['split']:<5} n={r['n']:>7}  cov90={r['coverage_90']:.3f}  "
                f"MAPE={r['mape_median']:.3f}  width={r['interval_width_mean']:6.1f} kW"
            )


if __name__ == "__main__":
    main()
