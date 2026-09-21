"""The bootstrap behind every interval in the horizon tables, checked against
the two cases where the right answer is known.

On independent rows the day block must recover the binomial standard error. On
rows that are constant within a day it must recover the standard error of the
*day* mean -- ten times larger over 100 days of 96 rows -- because that is the
whole reason a naive interval on ~2,800 origins would be wrong. And the audit's
``block_bootstrap_se`` must not have moved, since a number in the paper rests
on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.horizon_risk import horizon_bootstrap, pivot_paths, with_bootstrap
from forecast.conformal import block_bootstrap_means, block_bootstrap_se

N_DAYS, PER_DAY = 100, 96


def _days() -> np.ndarray:
    return np.repeat(np.arange(N_DAYS), PER_DAY)


def test_iid_rows_recover_the_binomial_se():
    rng = np.random.default_rng(0)
    p = 0.3
    hit = (rng.random(N_DAYS * PER_DAY) < p).astype(float)
    se = block_bootstrap_means(hit, _days(), n_boot=4000, seed=1).std(ddof=1)
    binom = np.sqrt(p * (1 - p) / len(hit))
    assert 0.8 * binom < se < 1.25 * binom


def test_within_day_dependence_is_not_averaged_away():
    rng = np.random.default_rng(0)
    day_hit = (rng.random(N_DAYS) < 0.5).astype(float)
    hit = np.repeat(day_hit, PER_DAY)  # every row in a day says the same thing
    se = block_bootstrap_means(hit, _days(), n_boot=4000, seed=1).std(ddof=1)
    per_day = np.sqrt(day_hit.mean() * (1 - day_hit.mean()) / N_DAYS)
    naive = np.sqrt(hit.mean() * (1 - hit.mean()) / len(hit))
    assert 0.8 * per_day < se < 1.25 * per_day
    assert se > 5 * naive


def test_se_is_the_std_of_the_means():
    rng = np.random.default_rng(3)
    hit = (rng.random(N_DAYS * PER_DAY) < 0.4).astype(float)
    a = block_bootstrap_se(hit, _days(), n_boot=500, seed=9)
    b = float(block_bootstrap_means(hit, _days(), n_boot=500, seed=9).std(ddof=1))
    assert a == b


def _synthetic_piv(n_days: int = 30, H: int = 64, seed: int = 0) -> dict:
    """A forecast tensor whose errors are AR(1) along the horizon, so the
    horizon event clusters the way real load errors do."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2017-06-01", periods=n_days * 96, freq="15min")
    rows = []
    for o in origins:
        e = np.empty(H)
        e[0] = rng.normal()
        for h in range(1, H):
            e[h] = 0.8 * e[h - 1] + np.sqrt(1 - 0.64) * rng.normal()
        y = 100 + 10 * e
        for h in range(H):
            rows.append((o, h + 1, o + pd.Timedelta(minutes=15 * (h + 1)), y[h],
                         100 - 16.4, 100 - 6.7, 100.0, 100 + 6.7, 100 + 16.4))
    df = pd.DataFrame(rows, columns=["origin", "horizon", "target_time", "actual",
                                     "q05", "q25", "q50", "q75", "q95"])
    return pivot_paths(df, horizon=H)


def test_interval_brackets_the_point_estimate_and_widens_with_horizon():
    piv = _synthetic_piv()
    assert "origin" in piv and len(piv["origin"]) == piv["actual"].shape[0]
    prev = 0.0
    for H in (1, 16, 64):
        b = horizon_bootstrap(piv, H, seed=0)
        emp = float(np.mean((piv["actual"][:, :H] > piv["q95"][:, :H]).any(axis=1)))
        lo, hi = b["empirical_horizon_ci"]
        assert lo <= emp <= hi
        assert b["n_blocks"] == 30
        assert hi - lo >= prev  # more steps, more event, wider interval
        prev = hi - lo
    # per-step rate on a correctly specified quantile sits on 0.05
    b1 = horizon_bootstrap(piv, 64, seed=0)
    assert b1["per_step_ci"][0] <= 0.05 <= b1["per_step_ci"][1]


def test_with_bootstrap_annotates_every_row_and_moves_no_point_estimate():
    piv = _synthetic_piv()
    rows = [{"H": H, "empirical_horizon": 0.123} for H in (1, 64)]
    out = with_bootstrap(rows, piv)
    for r in out:
        assert r["empirical_horizon"] == 0.123
        for k in ("per_step_ci", "empirical_horizon_ci", "empirical_horizon_block_ci",
                  "empirical_horizon_se", "n_blocks", "block_sensitivity"):
            assert k in r
        assert set(r["block_sensitivity"]) == {"2", "3"}


def test_audit_shift_moves_whole_years_and_keeps_the_shape():
    from eval.conformal_audit import shift_years
    assert shift_years("2017-03-31 23:45", 5) == "2012-03-31 23:45:00"
    assert shift_years("2016-06-30 23:45", 5) == "2011-06-30 23:45:00"
    assert shift_years("2017-06-01", 0) == "2017-06-01"


def test_audit_country_from_prefix(tmp_path):
    from eval.conformal_audit import country_of
    assert country_of("IN_Delhi", tmp_path) == "IN"
    assert country_of("CN_Hainan", tmp_path) == "CN"
    assert country_of("Fox_office_Gaylord", tmp_path) == "US"
    assert country_of("IIITD_Campus", tmp_path) == "IN"
