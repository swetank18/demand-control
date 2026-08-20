"""The calibration layer, shared by every forecaster in the benchmark.

Why it is factored out here rather than living inside the LightGBM trainer:
if only our model gets a conformal wrapper and the baselines do not, then the
benchmark measures the wrapper, not the model, and the ablation downstream
inherits that confound. Every row of the comparison therefore goes through this
same code. What differs between rows is the point forecast underneath it.

Two stages, in this order.

**Split conformal.** Each quantile is shifted by the empirical residual quantile
measured on the validation block, per horizon. LightGBM quantile regression is
not calibrated by construction and neither is a persistence residual; this makes
the nominal level mean something on data the model has not seen.

**Adaptive conformal (ACI).** Split conformal fitted on April-May undercovers in
June because June is hotter and the load is larger -- ordinary distribution
shift. ACI closes the loop: after each target time the actual becomes known and
the offset moves against the realised exceedance rate. Base load is exogenous,
so running it offline over the window is equivalent to running it online, and
it uses only information available before each origin.

The guarantee this buys is the sentence worth having in a technical Q&A:
coverage is distribution-free and finite-sample, not assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_split_shifts(
    y: np.ndarray, pred: dict[float, np.ndarray], horizon: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-quantile, per-horizon shift fitted on a calibration block.

    Per horizon rather than pooled because a 15-minute-ahead forecast and a
    16-hour-ahead forecast have residual distributions that differ by an order
    of magnitude, and the controller leans hardest on the far end.
    """
    y = np.asarray(y, float)
    horizon = np.asarray(horizon, int)
    shifts: dict[str, dict[str, float]] = {}
    for q, p in pred.items():
        per_h: dict[str, float] = {}
        for h in np.unique(horizon):
            m = horizon == h
            per_h[str(int(h))] = float(np.quantile(y[m] - np.asarray(p, float)[m], q))
        shifts[str(q)] = per_h
    return shifts


def apply_split_shifts(
    pred: dict[float, np.ndarray], horizon: np.ndarray, shifts: dict[str, dict[str, float]],
) -> dict[float, np.ndarray]:
    horizon = np.asarray(horizon, int)
    out = {}
    for q, p in pred.items():
        per_h = shifts.get(str(q), {})
        out[q] = np.asarray(p, float) + np.array([per_h.get(str(int(h)), 0.0) for h in horizon])
    return out


def adaptive_conformal(
    pred: np.ndarray, y: np.ndarray, horizon: np.ndarray, q: float,
    gamma: float = 0.35, n_horizons: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Online quantile tracking over rows already sorted into time order.

    Returns the adjusted predictions and the final per-horizon offsets. The
    update is the standard ACI step: raise the bound when the actual exceeded
    it, lower it slowly otherwise, at a rate that makes the long-run exceedance
    rate converge to 1 - q regardless of whether the model is any good.
    """
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    horizon = np.asarray(horizon, int)
    n_h = int(n_horizons or horizon.max()) + 2
    offs = np.zeros(n_h)
    out = np.empty_like(pred)
    for i in range(len(pred)):
        h = horizon[i]
        out[i] = pred[i] + offs[h]
        exceed = 1.0 if y[i] > out[i] else 0.0
        offs[h] += gamma * (exceed - (1.0 - q))
    return out, offs


def calibrate(
    valid: pd.DataFrame, test: pd.DataFrame,
    pred_valid: dict[float, np.ndarray], pred_test: dict[float, np.ndarray],
    split: bool = True, adaptive: bool = True, gamma: float = 0.35,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], pd.DataFrame, dict]:
    """Run the full layer. ``valid`` and ``test`` carry columns ``y``,
    ``horizon`` and ``target_time`` and are aligned with the prediction arrays.

    Returns calibrated validation predictions, calibrated test predictions, the
    test frame in the time order the predictions are now in, and a record of
    what was fitted (for the model card).
    """
    quantiles = sorted(pred_valid)
    record: dict = {"split_conformal": split, "adaptive_conformal": adaptive, "gamma": gamma}

    pv = {q: np.asarray(pred_valid[q], float).copy() for q in quantiles}
    pt = {q: np.asarray(pred_test[q], float).copy() for q in quantiles}

    if split:
        shifts = fit_split_shifts(valid["y"].to_numpy(), pv, valid["horizon"].to_numpy())
        pv = apply_split_shifts(pv, valid["horizon"].to_numpy(), shifts)
        pt = apply_split_shifts(pt, test["horizon"].to_numpy(), shifts)
        record["shifts"] = shifts

    test_ordered = test
    if adaptive:
        order = np.lexsort((test["horizon"].to_numpy(), test["target_time"].to_numpy()))
        test_ordered = test.iloc[order]
        h = test_ordered["horizon"].to_numpy()
        y = test_ordered["y"].to_numpy()
        aci = {}
        for q in quantiles:
            adj, offs = adaptive_conformal(pt[q][order], y, h, q, gamma=gamma)
            pt[q] = adj
            aci[str(q)] = {"final_offsets": offs[1:-1].tolist()}
        record["aci"] = aci

    # independently shifted quantiles can cross; a crossed ladder is not a
    # distribution, so sort pointwise before anything downstream reads it
    def _sort(d: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
        arr = np.sort(np.vstack([d[q] for q in quantiles]), axis=0)
        return {q: arr[i] for i, q in enumerate(quantiles)}

    return _sort(pv), _sort(pt), test_ordered, record
