"""Run the trained quantile models over an arbitrary (possibly stressed) series.

Why this exists: the evaluation harness can inject a heatwave that lifts base
load 25% above anything in the training window. If the controller keeps reading
a forecast tensor precomputed on the *unstressed* history, then every controller
is equally blind and the stress test measures nothing -- both the mean-forecast
and the q95 controller are wrong by the same unmodelled amount.

A real forecaster would not be blind like that. It sees the elevated load in its
own lag features within an hour, and its conformal layer widens as errors
accumulate. So under stress we re-run inference on the stressed inputs. The
models are not retrained; only the inputs change, which is exactly what happens
in the field.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.features import FEATURE_COLS, HORIZON_STEPS, build_supervised

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


class QuantileModels:
    """Loaded boosters plus their calibration layers."""

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.boosters = {q: lgb.Booster(model_file=str(d / f"q{int(q*100):02d}.txt")) for q in QUANTILES}
        self.shifts = json.loads((d / "conformal.json").read_text()) if (d / "conformal.json").exists() else {}
        meta = json.loads((d / "meta.json").read_text())
        self.adaptive = bool(meta.get("adaptive_conformal", True))
        self.gamma = float(meta.get("adaptive_gamma", 0.35))
        self.horizon = int(meta.get("horizon_steps", HORIZON_STEPS))

    def predict_tensor(
        self,
        df: pd.DataFrame,
        window_start: str | pd.Timestamp | None = None,
        adaptive: bool | None = None,
    ) -> pd.DataFrame:
        """``df`` must carry enough history before ``window_start`` for the lags
        (at least 7 days). Returns the same shape as ``forecast_test.parquet``."""
        sup = build_supervised(df, horizon_steps=self.horizon)
        if window_start is not None:
            sup = sup[sup["target_time"] >= pd.Timestamp(window_start)]
        if sup.empty:
            raise ValueError("no rows to forecast; is there enough history before the window?")
        sup = sup.sort_values(["target_time", "horizon"]).reset_index(drop=True)

        X = sup[FEATURE_COLS]
        preds = {q: b.predict(X) for q, b in self.boosters.items()}

        # split-conformal shift, per horizon, fitted at training time
        h = sup["horizon"].to_numpy().astype(int)
        for q in QUANTILES:
            per_h = self.shifts.get(str(q))
            if per_h:
                preds[q] = preds[q] + np.array([per_h.get(str(int(x)), 0.0) for x in h])

        # adaptive conformal, walked forward in time order
        use_adaptive = self.adaptive if adaptive is None else adaptive
        if use_adaptive:
            y = sup["y"].to_numpy()
            for q in QUANTILES:
                p = preds[q]
                offs = np.zeros(self.horizon + 2)
                out = np.empty_like(p)
                for i in range(len(p)):
                    hi = h[i]
                    out[i] = p[i] + offs[hi]
                    exceed = 1.0 if y[i] > out[i] else 0.0
                    offs[hi] += self.gamma * (exceed - (1.0 - q))
                preds[q] = out

        arr = np.sort(np.vstack([preds[q] for q in QUANTILES]), axis=0)
        tensor = sup[["origin", "target_time", "horizon"]].copy()
        for i, q in enumerate(QUANTILES):
            tensor[f"q{int(q*100):02d}"] = arr[i]
        tensor["actual"] = sup["y"].to_numpy()
        return tensor
