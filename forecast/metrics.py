"""Proper scoring rules for probabilistic forecasts.

One module, imported by the trainer, the benchmark, the calibration report and
the ablation, so that a pinball loss quoted on slide 4 and a pinball loss quoted
on slide 7 are the same arithmetic. There is no second implementation anywhere
in this repo, deliberately.

A note on what these are for. Point accuracy (MAE, MAPE) is the number everyone
asks for and it is not the operative one here: the controller substitutes q95
into a hard constraint, so what matters is whether q95 is really a 95th
percentile (coverage) and how tight it is while staying so (sharpness). Pinball
loss scores both at once, which is why it is the horizontal axis of the frontier
plot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QuantileDict = dict


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at level ``q``. Lower is better; 0 is exact."""
    d = np.asarray(y, float) - np.asarray(yhat, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def pinball_mean(y: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """Pinball averaged over the reported quantile levels: the headline score."""
    return float(np.mean([pinball(y, preds[q], q) for q in sorted(preds)]))


def crps_from_quantiles(y: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """CRPS approximated from a finite quantile set.

    CRPS = 2 * integral over alpha of the pinball loss at level alpha. With only
    five reported levels the integral is a trapezoid over those levels, so this
    *underestimates* the true CRPS in the tails beyond q05 and q95. It is used
    for ranking models scored on the identical grid, which is what it is valid
    for, and never quoted as an absolute.
    """
    qs = np.array(sorted(preds), float)
    pl = np.array([pinball(y, preds[q], q) for q in qs])
    return float(2.0 * np.trapz(pl, qs))


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Winkler interval score for a central (1-alpha) interval.

    Width, plus a penalty of 2/alpha times the miss distance. It is the score
    that refuses to be gamed by widening: a lazy interval from zero to infinity
    covers everything and scores terribly.
    """
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    w = hi - lo
    w = w + np.where(y < lo, 2.0 / alpha * (lo - y), 0.0)
    w = w + np.where(y > hi, 2.0 / alpha * (y - hi), 0.0)
    return float(np.mean(w))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical hit rate of an interval. For a nominal 90% interval this should
    land between 0.85 and 0.95 on data the model has never seen."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def below_rate(y: np.ndarray, q: np.ndarray) -> float:
    """Fraction of actuals at or below a quantile. For q95 this is the one that
    matters operationally: 1 - this is how often reality broke through the bound
    the optimiser planned against."""
    return float(np.mean(np.asarray(y, float) <= np.asarray(q, float)))


def pit(y: np.ndarray, preds: dict[float, np.ndarray]) -> np.ndarray:
    """Probability integral transform, one value per observation.

    Linear interpolation of the predictive CDF through the reported quantiles,
    clipped outside them. Useful for eyeballing, but note the clipping: with
    five reported levels roughly a tenth of a *correctly specified* sample lands
    exactly on 0.0 or 1.0 because there is no quantile out there to interpolate
    against. So the uniformity statistic below is computed on bins instead, and
    the histogram figure is drawn from :func:`pit_bins`, not from this.
    """
    qs = np.array(sorted(preds), float)
    grid = np.vstack([np.asarray(preds[q], float) for q in qs])       # (K, n)
    y = np.asarray(y, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = np.interp(y[i], grid[:, i], qs, left=0.0, right=1.0)
    return out


def pit_bins(y: np.ndarray, preds: dict[float, np.ndarray]) -> dict:
    """The PIT histogram computed exactly, with the reported quantiles as edges.

    Bin k holds the actuals falling between two adjacent predicted quantiles,
    and under perfect calibration its share of the sample equals the gap between
    those two levels. No interpolation is involved, so this is exact at the
    resolution the forecaster actually reports -- which is the honest resolution
    to draw the histogram at.

    Returns the edges (including the two open tails), the observed share per bin
    and the nominal share, all as plain lists so the figure code and the JSON
    report read the identical numbers.
    """
    y = np.asarray(y, float)
    qs = sorted(preds)
    edges = [0.0] + list(qs) + [1.0]
    nominal = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

    below = [np.asarray(y <= preds[q], bool) for q in qs]
    observed = []
    prev = np.zeros(len(y), bool)
    for b in below:
        observed.append(float(np.mean(b & ~prev)))
        prev = prev | b
    observed.append(float(np.mean(~prev)))                            # above the top quantile
    return {"edges": edges, "levels": list(qs), "observed": observed, "nominal": nominal}


def pit_deviation(y: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """Total-variation distance between the PIT histogram and flat. 0 is perfect.

    Half the summed absolute gap between observed and nominal bin shares, so it
    reads as "this fraction of the probability mass sits in the wrong bin".
    Overconfidence shows up as mass piled into the two tail bins; the sign is
    recovered from :func:`pit_bins` when it matters.
    """
    b = pit_bins(y, preds)
    return float(0.5 * np.sum(np.abs(np.array(b["observed"]) - np.array(b["nominal"]))))


def calibration_error(y: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """Mean absolute gap between nominal and empirical coverage across levels.

    This is the horizontal axis of frontier plot C, where it isolates
    calibration from sharpness: a model can have a fine pinball loss and still
    be systematically overconfident, and it is overconfidence, not error, that
    breaks a chance constraint.
    """
    return float(np.mean([abs(below_rate(y, preds[q]) - q) for q in sorted(preds)]))


@dataclass
class Scores:
    """Everything the benchmark reports for one forecaster on one split."""

    n: int
    pinball_mean: float
    pinball_by_q: dict
    crps: float
    winkler_90: float
    coverage_90: float
    below_q95: float
    calibration_error: float
    pit_deviation: float
    sharpness_90: float
    mae_median: float
    rmse_median: float
    mape_median: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "pinball_mean": self.pinball_mean,
            "pinball_by_q": self.pinball_by_q,
            "crps": self.crps,
            "winkler_90": self.winkler_90,
            "coverage_90": self.coverage_90,
            "below_q95": self.below_q95,
            "calibration_error": self.calibration_error,
            "pit_deviation": self.pit_deviation,
            "sharpness_90": self.sharpness_90,
            "mae_median": self.mae_median,
            "rmse_median": self.rmse_median,
            "mape_median": self.mape_median,
        }


def score_all(y: np.ndarray, preds: dict[float, np.ndarray]) -> Scores:
    """Score a quantile forecast. ``preds`` maps level -> prediction array."""
    y = np.asarray(y, float)
    qs = sorted(preds)
    lo, hi = preds[qs[0]], preds[qs[-1]]
    alpha = qs[0] + (1.0 - qs[-1])
    med = preds[min(qs, key=lambda q: abs(q - 0.5))]
    resid = y - med
    return Scores(
        n=int(len(y)),
        pinball_mean=pinball_mean(y, preds),
        pinball_by_q={str(q): pinball(y, preds[q], q) for q in qs},
        crps=crps_from_quantiles(y, preds),
        winkler_90=winkler(y, lo, hi, alpha),
        coverage_90=coverage(y, lo, hi),
        below_q95=below_rate(y, preds[qs[-1]]),
        calibration_error=calibration_error(y, preds),
        pit_deviation=pit_deviation(y, preds),
        sharpness_90=float(np.mean(hi - lo)),
        mae_median=float(np.mean(np.abs(resid))),
        rmse_median=float(np.sqrt(np.mean(resid**2))),
        mape_median=float(np.mean(np.abs(resid) / np.maximum(np.abs(y), 1e-6))),
    )


def enforce_monotone(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Sort quantiles pointwise. Independently fitted quantile models can cross,
    and a crossed q95 below q75 is not a distribution, it is a bug."""
    qs = sorted(preds)
    arr = np.sort(np.vstack([np.asarray(preds[q], float) for q in qs]), axis=0)
    return {q: arr[i] for i, q in enumerate(qs)}
