"""Track C, Stage 2: train the forecaster where the decision actually is.

The measurement in ``eval/decision_regret.py`` is that forecast error is worth
wildly different amounts at different moments. At 03:00 on a Sunday the ceiling
constraint is slack, and the forecast could be 10 kW wrong in either direction
without moving a single setpoint: the finite difference comes back at exactly
zero. Four hours before a hot Tuesday afternoon it is worth tens of rupees per
kilowatt. Pinball loss weights those two intervals identically, because a proper
scoring rule scores the *forecast*, and what is being bought here is a
*decision*.

Stage 2 is the cheap response, and the plan is explicit that it may be enough on
its own: keep pinball loss, keep the optimiser, keep every guarantee, and simply
tell the booster where to spend its capacity. No differentiation, no optimiser in
the training loop, about forty lines.

**Why this is safe, which is not obvious.** Reweighting a quantile regression
generally moves what it estimates. It does not here, and the reason is worth
stating precisely: every covariate below is a function of the *features* --
lead time, hour, weekend, and the model's own median prediction. A weight that
depends only on x leaves the conditional quantile at each x unchanged, because
w(x) factors out of the conditional objective. What it changes is where a
finite-capacity model spends its finite capacity: which splits a tree of 63
leaves bothers to make. So this is not a bias-variance trade dressed up as a
method -- it reallocates approximation error from intervals that cost nothing to
intervals that cost money, and the conformal layer downstream is undisturbed.

A weight that depended on y would be a different and much less defensible
object, and would void that argument. It is not done, deliberately.

The knob is ``mix``. At 0 the weights are uniform and the model is byte-for-byte
the one the repo already ships, which is what makes the frontier sweep an
ablation. At 1 the weights are proportional to measured sensitivity.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

#: Everything here must be computable for a training row from information known
#: at the forecast origin. ``base_q50_kw`` is the model's own median prediction,
#: which is a function of the features and therefore admissible; the realised
#: load is not, and is deliberately absent.
COVARIATES = ["lead_h", "hour", "is_weekend", "base_q50_kw"]

LGB_PARAMS = dict(
    objective="regression_l1",     # sensitivity is heavy-tailed and mostly zero;
    metric="l1",                   # squared error would fit the few huge blocks
    learning_rate=0.06,            # and ignore the shape everywhere else
    num_leaves=15,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=8,
)


class SensitivityModel:
    """Predicts ₹ per kW of forecast error from origin-time covariates.

    A small gradient-boosted regressor rather than a hand-written rule. The
    first attempt at this was a kernel on "how close is the ceiling to binding",
    which encoded a plausible intuition and turned out to have the sign
    backwards: measured sensitivity is *highest* where the constraint has room,
    because that is where the optimiser still has a choice to change. Fitting
    the measurement instead of asserting a shape avoids repeating that.
    """

    def __init__(self, booster: lgb.Booster, meta: dict):
        self.booster = booster
        self.meta = meta

    # -- fitting ----------------------------------------------------------
    @classmethod
    def fit(cls, sens: pd.DataFrame, covariates: list[str] | None = None,
            n_estimators: int = 300, seed: int = 0) -> "SensitivityModel":
        cov = covariates or COVARIATES
        missing = [c for c in cov if c not in sens.columns]
        if missing:
            raise ValueError(f"decision_sensitivity.parquet is missing {missing}; "
                             "re-run eval/decision_regret.py")
        d = sens.dropna(subset=cov + ["abs_d_cost_d_kw"])
        X, y = d[cov], d["abs_d_cost_d_kw"].to_numpy()

        # Held out by origin, not by row. Blocks from the same origin share a
        # solve and are strongly dependent, so a random row split would report a
        # flattering R^2 that says nothing about a new day.
        origins = d["k0"].unique()
        rng = np.random.default_rng(seed)
        holdout = set(rng.choice(origins, size=max(1, len(origins) // 4), replace=False))
        te = d["k0"].isin(holdout).to_numpy()

        booster = lgb.train(
            LGB_PARAMS, lgb.Dataset(X[~te], y[~te]), num_boost_round=n_estimators,
            valid_sets=[lgb.Dataset(X[te], y[te])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        pred = booster.predict(X[te], num_iteration=booster.best_iteration)
        ss_res = float(np.sum((y[te] - pred) ** 2))
        ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
        meta = {
            "covariates": cov,
            "n_rows": int(len(d)), "n_origins": int(len(origins)),
            "n_holdout_origins": int(len(holdout)),
            "holdout_r2": 1.0 - ss_res / max(ss_tot, 1e-9),
            "holdout_spearman": float(pd.Series(pred).corr(
                pd.Series(y[te]).reset_index(drop=True), method="spearman")),
            "mean_abs_d_cost_d_kw": float(y.mean()),
            "share_zero": float((y < 1e-9).mean()),
            "importance": dict(zip(cov, booster.feature_importance("gain").tolist())),
            "best_iteration": int(booster.best_iteration or n_estimators),
        }
        return cls(booster, meta)

    # -- use --------------------------------------------------------------
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        cov = self.meta["covariates"]
        return np.maximum(0.0, self.booster.predict(
            frame[cov], num_iteration=self.meta["best_iteration"]))

    def weights(self, frame: pd.DataFrame, mix: float = 1.0, cap: float = 8.0) -> np.ndarray:
        """Training weights, mean 1, for the rows of ``frame``.

        ``mix`` interpolates from uniform (0) to fully sensitivity-proportional
        (1). ``cap`` bounds the ratio between the heaviest and lightest row: a
        handful of blocks carry sensitivities two orders of magnitude above the
        median, and letting them through unbounded would train the model on
        about forty afternoons.
        """
        if mix <= 0:
            return np.ones(len(frame))
        s = self.predict(frame)
        s = s / max(s.mean(), 1e-12)
        w = (1.0 - mix) + mix * s
        w = np.clip(w, 1.0 / cap, cap)
        return w / w.mean()

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.booster.save_model(str(path.with_suffix(".txt")),
                                num_iteration=self.meta["best_iteration"])
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SensitivityModel":
        path = Path(path)
        return cls(lgb.Booster(model_file=str(path.with_suffix(".txt"))),
                   json.loads(path.with_suffix(".json").read_text()))


def covariate_frame(sup: pd.DataFrame, q50: np.ndarray) -> pd.DataFrame:
    """Build the sensitivity model's covariates for a supervised training frame.

    ``q50`` is the median prediction for each row, from the model being trained.
    That makes the weighting mildly circular -- the weights depend on a first
    pass -- which is why the trainer computes them once from the *unweighted*
    model and then holds them fixed. Iterating to a fixed point is possible and
    was not worth the complexity for the effect size.
    """
    t = pd.to_datetime(sup["target_time"])
    return pd.DataFrame({
        "lead_h": sup["horizon"].to_numpy() * 0.25,
        "hour": t.dt.hour.to_numpy(),
        "is_weekend": (t.dt.dayofweek >= 5).astype(int).to_numpy(),
        "base_q50_kw": np.asarray(q50, float),
    }, index=sup.index)


def training_weights(
    sup: pd.DataFrame, q50: np.ndarray, model: SensitivityModel,
    mix: float = 1.0, cap: float = 8.0,
) -> np.ndarray:
    return model.weights(covariate_frame(sup, q50), mix=mix, cap=cap)
