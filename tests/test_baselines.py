"""The baselines and the calibration layer, pinned.

The benchmark's credibility rests on the references being real. A persistence
row that is secretly broken makes our model look good for the wrong reason, and
that is worse than losing honestly. So each one is checked against its own
definition, and the linear quantile fit is checked against scikit-learn's linear
programming solver rather than against itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import conformal
from forecast.baselines import (Climatology, ConstantMargin, LinearQuantile,
                                PerfectForesight, Persistence, SeasonalNaive)
from forecast.features import FEATURE_COLS, build_supervised

QS = (0.05, 0.25, 0.50, 0.75, 0.95)


@pytest.fixture(scope="module")
def frames():
    idx = pd.date_range("2016-01-04", periods=180 * 96, freq="15min")
    rng = np.random.default_rng(7)
    hod = idx.hour + idx.minute / 60.0
    t_out = 24 + 9 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 0.5, len(idx))
    y = (150 + 70 * np.exp(-0.5 * ((hod - 14) / 3.0) ** 2) - 30 * (idx.dayofweek >= 5)
         + 3.0 * np.maximum(0.0, t_out - 18.0) + rng.normal(0, 5.0, len(idx)))
    df = pd.DataFrame({"base_kw": y, "t_out": t_out,
                       "cloud": rng.uniform(0, 0.5, len(idx))}, index=idx)
    sup = build_supervised(df, horizon_steps=8)
    cut_a, cut_b = idx[120 * 96], idx[150 * 96]
    tr = sup[sup.target_time <= cut_a]
    va = sup[(sup.target_time > cut_a) & (sup.target_time <= cut_b)]
    te = sup[sup.target_time > cut_b]
    return df["base_kw"], tr, va, te


def test_persistence_is_the_value_at_the_origin(frames):
    series, tr, va, te = frames
    p = Persistence().fit(tr, series).predict(te)
    np.testing.assert_allclose(p[0.5], series.reindex(pd.DatetimeIndex(te["origin"])).to_numpy())
    assert all(np.array_equal(p[0.5], p[q]) for q in QS), "a point baseline must not invent spread"


def test_seasonal_naive_reads_exactly_one_week_before_the_target(frames):
    series, tr, va, te = frames
    p = SeasonalNaive().fit(tr, series).predict(te)
    want = series.reindex(pd.DatetimeIndex(te["target_time"]) - pd.Timedelta(days=7)).to_numpy()
    ok = ~np.isnan(want)
    np.testing.assert_allclose(p[0.5][ok], want[ok])


def test_seasonal_naive_refuses_a_lookup_that_reaches_the_origin(frames):
    """The assertion inside the baseline is load bearing, so prove it fires."""
    series, tr, va, te = frames
    with pytest.raises(AssertionError, match="at or after the forecast origin"):
        SeasonalNaive(days=0).fit(tr, series).predict(te)


def test_climatology_is_calibrated_in_sample_but_not_sharp(frames):
    series, tr, va, te = frames
    m = Climatology().fit(tr, series)
    p = m.predict(tr)
    y = tr["y"].to_numpy()
    cov = float(np.mean((y >= p[0.05]) & (y <= p[0.95])))
    # It never *under*-covers on its own data -- it is the empirical distribution
    # of that slot. (In-sample coverage of an empirical quantile is biased above
    # nominal at small per-slot sample sizes, which is what this fixture has.)
    assert cov >= 0.88, "climatology under-covered on the data it was fitted to"
    # the fixture's irreducible noise is sigma = 5 kW, so a 90% band cannot be
    # much tighter than 2 * 1.645 * sigma without lying
    assert float(np.mean(p[0.95] - p[0.05])) > 2 * 1.645 * 5.0 * 0.85
    assert (p[0.05] <= p[0.50]).all() and (p[0.50] <= p[0.95]).all()

    # ...and it is not sharp, because it conditions on nothing but the clock.
    # Note it is *not* automatically worse than persistence -- on a building with
    # a strong weekly timetable it beats persistence outright, on this fixture and
    # on the real meter data both. That is why it is in the table: "beats
    # persistence" is a claim worth almost nothing.
    perfect = PerfectForesight().predict(tr)
    assert float(np.mean(np.abs(y - p[0.50]))) > float(np.mean(np.abs(y - perfect[0.50])))


def test_constant_margin_really_is_constant(frames):
    series, tr, va, te = frames
    p = ConstantMargin().fit(tr, series).predict(te)
    for q in QS:
        assert len(np.unique(p[q])) == 1
    assert p[0.95][0] > p[0.05][0]


def test_perfect_foresight_is_the_truth(frames):
    series, tr, va, te = frames
    p = PerfectForesight().predict(te)
    for q in QS:
        np.testing.assert_allclose(p[q], te["y"].to_numpy())


def test_linear_quantile_matches_the_reference_lp_solver(frames):
    """Adam on the pinball subgradient must land where the LP lands.

    Checked on a subsample because the LP formulation does not scale to the real
    training set -- which is precisely why the production fit uses Adam.
    """
    sklearn = pytest.importorskip("sklearn.linear_model")
    series, tr, va, te = frames
    sub = tr.sample(4000, random_state=0)
    cols = ["last", "tod_mean_4w", "t_out_fut", "horizon"]

    X = sub[cols].to_numpy(float)
    y = sub["y"].to_numpy(float)
    for q in (0.25, 0.5, 0.95):
        ref = sklearn.QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
        ours = LinearQuantile(quantiles=(q,), epochs=600, batch=1024, lr=0.1, seed=0)
        # fit on the same columns only
        import forecast.baselines as bl
        old = bl.FEATURE_COLS
        try:
            bl.FEATURE_COLS = cols
            ours.fit(sub, series)
            got = ours.predict(sub)[q]
        finally:
            bl.FEATURE_COLS = old
        want = ref.predict(X)
        # agreement in predictions, which is what is used, rather than in
        # coefficients, which are not identified when features are collinear
        assert float(np.mean(np.abs(got - want))) < 0.02 * float(np.std(y)), (
            f"q={q}: mean |ours - LP| = {np.mean(np.abs(got - want)):.3f}")


# --- the shared calibration layer -----------------------------------------

def test_split_conformal_fixes_a_biased_forecaster(frames):
    series, tr, va, te = frames
    # A forecaster that is 25 kW low and noisy at every level. Noise matters:
    # conformal builds the spread out of the residual distribution, so a
    # noiseless forecaster gives it nothing to build from.
    # One noise draw shared across levels, because this is a *point* forecaster
    # whose spread the conformal layer has to construct. Five independent draws
    # would be five different forecasters, and the monotone sort at the end
    # would then return order statistics rather than calibrated levels.
    rng = np.random.default_rng(11)
    pv = va["y"].to_numpy() - 25.0 + rng.normal(0, 12, len(va))
    pt = te["y"].to_numpy() - 25.0 + rng.normal(0, 12, len(te))
    biased_va = {q: pv.copy() for q in QS}
    biased_te = {q: pt.copy() for q in QS}
    _, cal, te_ord, rec = conformal.calibrate(va, te, biased_va, biased_te,
                                              split=True, adaptive=False)
    y = te_ord["y"].to_numpy()
    for q in QS:
        assert abs(float(np.mean(y <= cal[q])) - q) < 0.05, f"level {q} still off after conformal"
    assert rec["split_conformal"]


def test_adaptive_conformal_tracks_a_shifted_test_block(frames):
    """Split conformal alone undercovers when the test block is hotter than the
    calibration block. This is that failure, and the fix, in one test."""
    series, tr, va, te = frames
    rng = np.random.default_rng(12)
    pv = va["y"].to_numpy() + rng.normal(0, 8, len(va))
    pt = te["y"].to_numpy() - 30.0 + rng.normal(0, 8, len(te))    # a shift ACI has to find
    va_pred = {q: pv.copy() for q in QS}
    te_pred = {q: pt.copy() for q in QS}

    _, split_only, ord_a, _ = conformal.calibrate(va, te, va_pred, dict(te_pred),
                                                  split=True, adaptive=False)
    _, with_aci, ord_b, _ = conformal.calibrate(va, te, va_pred, dict(te_pred),
                                                split=True, adaptive=True)
    hit_split = float(np.mean(ord_a["y"].to_numpy() <= split_only[0.95]))
    hit_aci = float(np.mean(ord_b["y"].to_numpy() <= with_aci[0.95]))
    assert hit_split < 0.60, "the test setup did not actually break split conformal"
    assert hit_aci > hit_split + 0.25, "adaptive conformal failed to recover coverage"


def test_calibration_never_returns_crossed_quantiles(frames):
    series, tr, va, te = frames
    rng = np.random.default_rng(0)
    noisy_va = {q: va["y"].to_numpy() + rng.normal(0, 40, len(va)) for q in QS}
    noisy_te = {q: te["y"].to_numpy() + rng.normal(0, 40, len(te)) for q in QS}
    # deliberately crossed on the way in: five independent draws cross constantly
    _, cal, _, _ = conformal.calibrate(va, te, noisy_va, noisy_te)
    stack = np.vstack([cal[q] for q in QS])
    assert (np.diff(stack, axis=0) >= -1e-9).all(), "quantiles crossed after calibration"
