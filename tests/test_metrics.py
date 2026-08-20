"""The scoring rules, checked against hand computations and known limits.

These are the numbers the whole evidence layer is denominated in. If pinball
loss is wrong then the frontier plot is a plot of nothing, so the arithmetic is
pinned here rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecast.metrics import (calibration_error, coverage, crps_from_quantiles,
                              enforce_monotone, pinball, pit_bins, pit_deviation,
                              score_all, winkler)

QS = (0.05, 0.25, 0.50, 0.75, 0.95)


def test_pinball_against_hand_computation():
    # under-forecast by 10 at q=0.9: loss = 0.9 * 10
    assert pinball(np.array([100.0]), np.array([90.0]), 0.9) == pytest.approx(9.0)
    # over-forecast by 10 at q=0.9: loss = 0.1 * 10
    assert pinball(np.array([100.0]), np.array([110.0]), 0.9) == pytest.approx(1.0)
    # the asymmetry is the whole point: a high quantile is punished ninefold for
    # being too low, which is what pushes q95 above the mean
    assert pinball(np.array([0.0]), np.array([0.0]), 0.5) == 0.0


def test_pinball_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(50.0, 10.0, 200_000)
    for q in (0.05, 0.5, 0.95):
        truth = float(np.quantile(y, q))
        best = pinball(y, np.full_like(y, truth), q)
        for delta in (-3.0, -1.0, 1.0, 3.0):
            assert pinball(y, np.full_like(y, truth + delta), q) > best


def test_winkler_against_hand_computation():
    y, lo, hi, alpha = np.array([100.0]), np.array([80.0]), np.array([120.0]), 0.10
    assert winkler(y, lo, hi, alpha) == pytest.approx(40.0)          # inside: just the width
    y = np.array([130.0])                                            # 10 above the top
    assert winkler(y, lo, hi, alpha) == pytest.approx(40.0 + 2 / 0.10 * 10.0)
    y = np.array([70.0])                                             # 10 below the bottom
    assert winkler(y, lo, hi, alpha) == pytest.approx(40.0 + 2 / 0.10 * 10.0)


def test_winkler_refuses_to_reward_a_lazy_wide_interval():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 20_000)
    tight = winkler(y, np.full_like(y, -1.645), np.full_like(y, 1.645), 0.10)
    lazy = winkler(y, np.full_like(y, -50.0), np.full_like(y, 50.0), 0.10)
    assert coverage(y, np.full_like(y, -50.0), np.full_like(y, 50.0)) == 1.0
    assert lazy > tight, "a 100-wide interval scored better than a calibrated one"


def test_crps_is_zero_for_a_perfect_forecast_and_grows_with_error():
    y = np.linspace(10.0, 90.0, 500)
    perfect = {q: y.copy() for q in QS}
    assert crps_from_quantiles(y, perfect) == pytest.approx(0.0, abs=1e-12)
    biased = {q: y + 5.0 for q in QS}
    worse = {q: y + 15.0 for q in QS}
    assert crps_from_quantiles(y, worse) > crps_from_quantiles(y, biased) > 0.0


def test_pit_is_flat_for_a_correctly_specified_forecast():
    rng = np.random.default_rng(2)
    n = 60_000
    mu = rng.normal(100.0, 20.0, n)
    y = mu + rng.normal(0.0, 8.0, n)
    from scipy.stats import norm

    good = {q: mu + norm.ppf(q) * 8.0 for q in QS}
    over = {q: mu + norm.ppf(q) * 3.0 for q in QS}      # too sharp: U-shaped PIT

    assert pit_deviation(y, good) < 0.01
    assert pit_deviation(y, over) > 10 * pit_deviation(y, good)

    # bins must be a partition of the sample, and the overconfident forecaster
    # must pile mass into the two tail bins
    b_good, b_over = pit_bins(y, good), pit_bins(y, over)
    assert sum(b_good["observed"]) == pytest.approx(1.0)
    assert sum(b_good["nominal"]) == pytest.approx(1.0)
    np.testing.assert_allclose(b_good["observed"], b_good["nominal"], atol=0.01)
    tails = b_over["observed"][0] + b_over["observed"][-1]
    assert tails > 0.25, "an overconfident forecaster should pile actuals into the tails"


def test_calibration_error_sees_bias_that_pinball_partly_hides():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, 50_000)
    from scipy.stats import norm

    good = {q: np.full_like(y, norm.ppf(q)) for q in QS}
    shifted = {q: np.full_like(y, norm.ppf(q) + 0.5) for q in QS}
    assert calibration_error(y, good) < 0.01
    assert calibration_error(y, shifted) > 0.10


def test_enforce_monotone_uncrosses_quantiles():
    preds = {0.05: np.array([10.0]), 0.5: np.array([4.0]), 0.95: np.array([7.0])}
    fixed = enforce_monotone(preds)
    assert fixed[0.05][0] <= fixed[0.5][0] <= fixed[0.95][0]
    assert sorted(v[0] for v in preds.values()) == [v[0] for v in fixed.values()]


def test_score_all_reports_a_complete_and_sane_row():
    rng = np.random.default_rng(4)
    from scipy.stats import norm

    mu = rng.normal(200.0, 30.0, 20_000)
    y = mu + rng.normal(0.0, 10.0, 20_000)
    s = score_all(y, {q: mu + norm.ppf(q) * 10.0 for q in QS})
    assert s.n == 20_000
    assert 0.88 <= s.coverage_90 <= 0.92
    assert 0.94 <= s.below_q95 <= 0.96
    assert s.sharpness_90 == pytest.approx(2 * norm.ppf(0.95) * 10.0, rel=1e-3)
    assert s.mae_median < s.rmse_median
    assert set(s.as_dict()) >= {"pinball_mean", "crps", "winkler_90", "pit_deviation"}
