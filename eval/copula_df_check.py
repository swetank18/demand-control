"""Is the copula's agreement with the realised horizon risk an artefact of how
its one free parameter was chosen?

The t-copula's degrees of freedom are selected on the validation block by
matching predicted to realised horizon exceedance (forecast/trajectories.py,
fit_tail_df). The paper then reports how well the copula predicts realised
horizon exceedance on the held-out month. The split is honest, but the
selection criterion and the judged statistic are the same quantity, and a
reviewer is entitled to ask whether a criterion that knew nothing about that
statistic would have done as well. Two such criteria, on every window:

* **pseudo-likelihood** -- the textbook route: the degrees of freedom that
  maximise the t-copula log-likelihood of the validation normal scores, with
  the correlation matrix held at the fitted one. This spends its resolution on
  the body of the distribution and is blind to the upper tail.
* **no fitting at all** -- the value selected on the Phoenix office, applied
  unchanged to every Indian window.

For each, the copula's predicted horizon exceedance on the held-out month is
compared with the realised value and its day-block bootstrap interval, beside
the tail-matched value already in results/horizon_risk_*.json.

Outputs results/copula_df_check.json; eval/paper_tables.py turns it into a
table. Needs the models/ directory (boosters and the fitted copula).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal, multivariate_t, norm, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.horizon_risk import pivot_paths
from eval.windows import primary_results
from forecast.predict import QuantileModels
from forecast.trajectories import (DF_GRID, LEVELS, CopulaModel, _EPS, _pivot,
                                   nearest_psd, path_exceedance, value_to_z)

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CACHE = ROOT / "data/cache"
HORIZONS = (8, 16, 32, 64)
PHOENIX_DF = 7.0


def validation_scores(key: str, valid_start: str, valid_end: str, horizon: int = 64) -> np.ndarray:
    """(n_origins, H) normal scores of the validation block, the same matrix
    the copula's correlation was fitted from."""
    df = pd.read_parquet(CACHE / f"{key.split('@')[0]}.parquet")
    lead = pd.Timestamp(valid_start) - pd.Timedelta(days=10)
    tensor = QuantileModels(MODELS / key).predict_tensor(df.loc[lead:valid_end], window_start=valid_start)
    qcols = [f"q{int(q*100):02d}" for q in LEVELS]
    piv = _pivot(tensor, ["actual"] + qcols, horizon)
    y = piv["actual"]
    lad = np.stack([piv[c] for c in qcols], axis=-1)
    ok = ~np.isnan(y).any(axis=1) & ~np.isnan(lad).any(axis=(1, 2))
    y, lad = y[ok], lad[ok]
    z = np.empty_like(y)
    for h in range(y.shape[1]):
        z[:, h] = value_to_z(lad[:, h, :], y[:, h], LEVELS)
    return np.clip(z, ndtri(_EPS), ndtri(1 - _EPS))


def copula_loglik(z: np.ndarray, corr: np.ndarray, df: float | None) -> float:
    """Mean log copula density of the scores under a t (or Gaussian) copula
    with the given correlation. The marginals are already uniform by
    construction (z = Phi^-1(u)), so this is the pseudo-likelihood."""
    R = nearest_psd(corr)
    if df is None:
        return float(np.mean(multivariate_normal(mean=np.zeros(len(R)), cov=R, allow_singular=True).logpdf(z)
                             - norm.logpdf(z).sum(axis=1)))
    u = np.clip(ndtr(z), _EPS, 1 - _EPS)
    x = student_t.ppf(u, df)
    return float(np.mean(multivariate_t(loc=np.zeros(len(R)), shape=R, df=df, allow_singular=True).logpdf(x)
                         - student_t.logpdf(x, df).sum(axis=1)))


def predicted_exceedance(piv: dict, corr: np.ndarray, df: float | None, horizons=HORIZONS,
                         n_paths: int = 400, n_origins: int = 300, seed: int = 0) -> dict:
    """The copula's predicted horizon exceedance on the test month, exactly as
    eval/horizon_risk.marginal_vs_joint computes it: same origins, same paths."""
    rng = np.random.default_rng(seed)
    lad = np.stack([piv[f"q{int(q*100):02d}"] for q in LEVELS], axis=-1)
    q95 = piv["q95"]
    take = rng.choice(lad.shape[0], size=min(n_origins, lad.shape[0]), replace=False)
    m = CopulaModel(corr=corr, horizon=64, levels=LEVELS, df=df)
    out = {}
    for H in horizons:
        out[str(H)] = float(np.mean([path_exceedance(m.sample(lad[i, :H, :], n_paths, rng), q95[i, :H])
                                     for i in take]))
    return out


def one_window(path: Path) -> dict:
    p = json.loads(path.read_text())
    key = f"{p['building']}{p.get('tag', '')}"
    cop = CopulaModel.load(MODELS / key / "copula")
    vs, ve = cop.meta["df_fitted_on"]
    z = validation_scores(key, vs, ve)
    ll = {str(df): copula_loglik(z, cop.corr, df) for df in DF_GRID}
    finite = {k: v for k, v in ll.items() if np.isfinite(v)}
    best_ml = max(finite, key=finite.get)
    df_ml = None if best_ml == "None" else float(best_ml)

    tensor = pd.read_parquet(ROOT / p["tensor"])
    s, e = p["window"]
    tensor = tensor[(tensor["target_time"] >= pd.Timestamp(s)) & (tensor["target_time"] <= pd.Timestamp(e))]
    piv = pivot_paths(tensor)

    rows = {r["H"]: r for r in p["marginal_vs_joint"]}
    realised = {str(H): rows[H]["empirical_horizon"] for H in HORIZONS}
    ci = {str(H): rows[H].get("empirical_horizon_ci") for H in HORIZONS}
    tail = {str(H): rows[H]["copula_predicted"] for H in HORIZONS}
    ml = predicted_exceedance(piv, cop.corr, df_ml)
    fixed = tail if cop.df == PHOENIX_DF else predicted_exceedance(piv, cop.corr, PHOENIX_DF)

    def inside(pred):
        lo, hi = ci["64"]
        return bool(lo <= pred["64"] <= hi)

    return {
        "key": key, "building": p["building"], "tag": p.get("tag", ""),
        "n_valid_origins": int(len(z)),
        "df_tail": cop.df, "df_ml": df_ml, "df_fixed": PHOENIX_DF,
        "loglik": ll,
        "realised": realised, "realised_ci": ci,
        "predicted": {"tail": tail, "ml": ml, "fixed": fixed},
        "inside_64": {"tail": inside(tail), "ml": inside(ml), "fixed": inside(fixed)},
        "abs_err_64": {k: abs(v["64"] - realised["64"]) for k, v in
                       (("tail", tail), ("ml", ml), ("fixed", fixed))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS / "copula_df_check.json")
    ap.add_argument("--only", default=None, help="one window key, e.g. IIITD_Girls@2017")
    args = ap.parse_args()
    files = [RESULTS / "horizon_risk_Fox_office_Gaylord.json"] + primary_results(RESULTS)
    done = {}
    if args.out.exists():
        done = {r["key"]: r for r in json.loads(args.out.read_text())["windows"]}
    out = []
    for f in files:
        key = f.stem.replace("horizon_risk_", "")
        if args.only and key != args.only:
            continue
        if key in done:
            out.append(done[key]); print(f"   {key}: cached"); continue
        r = one_window(f)
        out.append(r)
        print(f"   {key:<26} df tail {r['df_tail']!s:>4}  ml {r['df_ml']!s:>4}  | H=64 realised "
              f"{r['realised']['64']:.3f} [{r['realised_ci']['64'][0]:.2f},{r['realised_ci']['64'][1]:.2f}]"
              f"  tail {r['predicted']['tail']['64']:.3f}  ml {r['predicted']['ml']['64']:.3f}"
              f"  fixed {r['predicted']['fixed']['64']:.3f}", flush=True)
        payload = {"windows": out, "df_grid": [None if d is None else d for d in DF_GRID],
                   "phoenix_df": PHOENIX_DF, "horizons": list(HORIZONS)}
        args.out.write_text(json.dumps(payload, indent=2))
    n = len(out)
    for k in ("tail", "ml", "fixed"):
        ins = sum(r["inside_64"][k] for r in out)
        mae = float(np.mean([r["abs_err_64"][k] for r in out]))
        print(f"{k:<6} inside the realised 95% interval on {ins} of {n}; mean |error| at H=64 {mae:.3f}")


if __name__ == "__main__":
    main()
