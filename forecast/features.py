"""Feature construction for the base-load forecaster.

One model handles every horizon: the horizon index is a feature and the training
rows are (origin, horizon) pairs. That is cheaper than 32 separate models and it
shares statistical strength across horizons.

Leakage discipline: every feature must be knowable at the forecast *origin*.
Lags are therefore taken relative to the origin, never relative to the target.
The single exception is weather, where we assume a perfect forecast; that
assumption is stated in the calibration report and can be stressed with
``weather_noise_c``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEPS_PER_HOUR = 4
# 16 hours. Long enough that a controller standing at midnight can see the next
# afternoon's peak, which is the precondition for pre-cooling into thermal mass.
# An 8-hour horizon cannot: it reaches 08:00 and the peak is at 14:00.
HORIZON_STEPS = 64

# Lags in steps, relative to the forecast origin.
ORIGIN_LAGS = [1, 2, 4, 8, 96, 192, 672]
ROLL_WINDOWS = [4, 96, 672]

# Site is Tempe, Arizona. US federal holidays 2016-2017.
US_HOLIDAYS = {
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
}


def _cyc(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    a = 2 * np.pi * x / period
    return np.sin(a), np.cos(a)


def build_supervised(
    df: pd.DataFrame,
    horizon_steps: int = HORIZON_STEPS,
    target_col: str = "base_kw",
    weather_noise_c: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a long frame of (origin, horizon) rows with features and target.

    Columns produced:
      origin, horizon, y  plus the feature block.
    """
    df = df.sort_index()
    y = df[target_col].astype(float)
    rng = np.random.default_rng(seed)

    # --- origin-time features (known at t) -------------------------------
    feats = {}
    for L in ORIGIN_LAGS:
        feats[f"lag_{L}"] = y.shift(L)
    for W in ROLL_WINDOWS:
        r = y.shift(1).rolling(W)
        feats[f"rmean_{W}"] = r.mean()
        feats[f"rstd_{W}"] = r.std()
    feats["last"] = y.shift(0)                     # the value at the origin itself
    feats["t_out_origin"] = df["t_out"]
    feats["dtemp_1h"] = df["t_out"] - df["t_out"].shift(4)
    X0 = pd.DataFrame(feats, index=df.index)

    # same-time-of-day mean over the trailing 4 weeks: the strongest single
    # predictor for a building on a weekly schedule
    tod = y.groupby([y.index.dayofweek, y.index.hour, y.index.minute])
    X0["tod_mean_4w"] = tod.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    frames = []
    weather = df["t_out"].astype(float)
    cloud = df.get("cloud", pd.Series(0.0, index=df.index)).astype(float)

    for h in range(1, horizon_steps + 1):
        tgt_time = df.index + pd.Timedelta(minutes=15 * h)
        ok = tgt_time.isin(df.index)
        rows = pd.DataFrame(index=df.index[ok])
        rows["origin"] = rows.index
        rows["target_time"] = tgt_time[ok]
        rows["horizon"] = h
        rows["y"] = y.reindex(rows["target_time"]).to_numpy()

        blk = X0.loc[rows.index]
        for c in blk.columns:
            rows[c] = blk[c].to_numpy()

        # weather at the target time = "the forecast"
        t_fut = weather.reindex(rows["target_time"]).to_numpy()
        if weather_noise_c > 0:
            t_fut = t_fut + rng.normal(0.0, weather_noise_c * np.sqrt(h / horizon_steps), size=len(t_fut))
        rows["t_out_fut"] = t_fut
        rows["cdd_fut"] = np.maximum(0.0, t_fut - 18.0)
        rows["cloud_fut"] = cloud.reindex(rows["target_time"]).to_numpy()

        tt = pd.DatetimeIndex(rows["target_time"])
        hod = tt.hour + tt.minute / 60.0
        s, c = _cyc(hod.to_numpy(), 24.0)
        rows["hod_sin"], rows["hod_cos"] = s, c
        s, c = _cyc(tt.dayofweek.to_numpy(), 7.0)
        rows["dow_sin"], rows["dow_cos"] = s, c
        rows["is_weekend"] = (tt.dayofweek >= 5).astype(int)
        rows["is_holiday"] = tt.normalize().strftime("%Y-%m-%d").isin(US_HOLIDAYS).astype(int)
        rows["doy_sin"], rows["doy_cos"] = _cyc(tt.dayofyear.to_numpy(), 365.0)
        # a solar term: matters once a site has PV, harmless otherwise
        rows["solar_elev"] = np.maximum(0.0, np.sin(np.pi * (hod.to_numpy() - 6.0) / 12.0))
        frames.append(rows)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna()
    return out


FEATURE_COLS = (
    [f"lag_{L}" for L in ORIGIN_LAGS]
    + [f"rmean_{W}" for W in ROLL_WINDOWS]
    + [f"rstd_{W}" for W in ROLL_WINDOWS]
    + [
        "last", "t_out_origin", "dtemp_1h", "tod_mean_4w", "horizon",
        "t_out_fut", "cdd_fut", "cloud_fut",
        "hod_sin", "hod_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_holiday", "doy_sin", "doy_cos", "solar_elev",
    ]
)
