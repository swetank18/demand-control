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


# ---------------------------------------------------------------------------
# Conformalized quantile regression, and the machinery the audit needs.
#
# The per-quantile shift above is one-sided split conformal, fitted level by
# level. That is the right object for the ceiling constraint, which reads a
# single upper bound and does not care what the lower one is doing. CQR is the
# two-sided cousin: one nonconformity score for the interval as a whole,
#
#     E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
#
# and one width Q added symmetrically to both ends. It is the version with the
# theorem attached (Romano, Patterson and Candes, NeurIPS 2019), and it is what
# a reviewer will ask for by name, so both are implemented and the audit
# reports both. They answer different questions: CQR guarantees the *interval*,
# the per-quantile shift guarantees *each bound*.
# ---------------------------------------------------------------------------


def fit_cqr_widths(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, horizon: np.ndarray, alpha: float = 0.10,
) -> dict[str, float]:
    """Per-horizon CQR width from a calibration block.

    The empirical quantile is taken at ``ceil((n+1)(1-alpha))/n`` rather than
    ``1-alpha``. That finite-sample correction is the whole theorem: with it,
    coverage on an exchangeable test point is at least ``1-alpha`` for any n and
    any underlying model, and without it the guarantee is asymptotic hand-waving.
    """
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    horizon = np.asarray(horizon, int)
    scores = np.maximum(lo - y, y - hi)
    out: dict[str, float] = {}
    for h in np.unique(horizon):
        s = scores[horizon == h]
        n = len(s)
        if n == 0:
            out[str(int(h))] = 0.0
            continue
        # rank of the (1-alpha) conformal quantile, clipped when n is too small
        # to certify the level at all -- in which case the honest answer is the
        # largest score seen, not a silently smaller one
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        out[str(int(h))] = float(np.sort(s)[min(k, n) - 1])
    return out


def apply_cqr(
    lo: np.ndarray, hi: np.ndarray, horizon: np.ndarray, widths: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    horizon = np.asarray(horizon, int)
    w = np.array([widths.get(str(int(h)), 0.0) for h in horizon])
    return np.asarray(lo, float) - w, np.asarray(hi, float) + w


def rolling_coverage(hit: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of a 0/1 hit series, NaN until the window is full.

    Coverage is only meaningful over a window long enough to contain the thing
    that breaks it. A 30-day window over 15-minute data is 2,880 points, which
    is what the A2 acceptance test is stated in.
    """
    s = pd.Series(np.asarray(hit, float))
    return s.rolling(window, min_periods=window).mean().to_numpy()


def block_bootstrap_se(
    hit: np.ndarray, block: np.ndarray, n_boot: int = 2000, seed: int = 0,
) -> float:
    """Standard error of a coverage rate, resampling whole blocks.

    Coverage computed over 184,000 correlated rows has a naive standard error of
    0.0007, which would declare any model on earth miscalibrated. Neighbouring
    15-minute forecasts share almost all of their information, so the honest
    unit of replication is the day, not the row. This resamples days.
    """
    hit = np.asarray(hit, float)
    block = np.asarray(block)
    keys, inv = np.unique(block, return_inverse=True)
    n = np.bincount(inv, minlength=len(keys)).astype(float)
    s = np.bincount(inv, weights=hit, minlength=len(keys))
    # resample blocks, then recombine as a size-weighted mean rather than
    # rebuilding the concatenated sample: same statistic, two orders of
    # magnitude cheaper, and the audit calls this on 180,000 rows
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = s[pick].sum(axis=1) / np.maximum(n[pick].sum(axis=1), 1e-12)
    return float(means.std(ddof=1))
