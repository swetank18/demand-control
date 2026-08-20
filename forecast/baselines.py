"""The forecasters we have to beat, implemented properly.

A model with no baseline is a number with no meaning. Section 6.3 of the plan
lists six references plus ours; they are all here, and two of them are not
strawmen: seasonal naive is genuinely hard to beat on a building running a
weekly schedule, and climatology is the correct no-skill reference for a
*quantile* forecast because it is the unconditional distribution.

Every baseline emits raw quantiles and then goes through the identical
``forecast.conformal`` layer, so what the benchmark compares is point-forecast
skill rather than who got a calibration wrapper.

Leakage discipline is the same as the feature builder's: nothing here may read a
value at or after the target time. The seasonal-naive lookup reaches back seven
days from the *target*, which for a 16-hour horizon is still six days before the
origin; ``tests/test_baselines.py`` asserts that rather than trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecast.features import FEATURE_COLS

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


class Baseline:
    """One forecaster. ``fit`` sees the training block only."""

    name = "baseline"
    #: shown in the benchmark table so a reader can see what each row is
    definition = ""
    #: "point" baselines emit one number replicated across levels; the spread is
    #: then created by the shared conformal layer from validation residuals
    kind = "point"

    def fit(self, train: pd.DataFrame, series: pd.Series) -> "Baseline":
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        raise NotImplementedError

    def _spread(self, point: np.ndarray) -> dict[float, np.ndarray]:
        return {q: np.asarray(point, float) for q in QUANTILES}


# ---------------------------------------------------------------------------
# The references
# ---------------------------------------------------------------------------

class Persistence(Baseline):
    name = "Persistence"
    definition = "the forecast is the value observed at the forecast origin"

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        return self._spread(frame["last"].to_numpy())


class SeasonalNaive(Baseline):
    """Same time of day, one week earlier. Surprisingly strong on a building
    that runs a weekly timetable, and it embarrasses a weak learned model."""

    name = "Seasonal naive"
    definition = "the value at the same clock time exactly one week before the target"

    def __init__(self, days: int = 7):
        self.days = days
        self.series: pd.Series | None = None

    def fit(self, train: pd.DataFrame, series: pd.Series) -> "SeasonalNaive":
        self.series = series.astype(float)
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        lookup = pd.DatetimeIndex(frame["target_time"]) - pd.Timedelta(days=self.days)
        assert (lookup <= pd.DatetimeIndex(frame["origin"])).all(), (
            "seasonal-naive lookup landed at or after the forecast origin"
        )
        v = self.series.reindex(lookup).to_numpy()
        # a gap in the record falls back to the value at the origin rather than
        # to NaN, which is what an operational naive forecaster would do
        v = np.where(np.isnan(v), frame["last"].to_numpy(), v)
        return self._spread(v)


class Climatology(Baseline):
    """The unconditional distribution for that slot of the week, measured on the
    training block. This is the honest no-skill reference for a probabilistic
    forecast: it is already calibrated, it just is not sharp."""

    name = "Climatology"
    definition = "training-set quantiles for that weekday and time of day"
    kind = "quantile"

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES):
        self.quantiles = quantiles
        self.table: pd.DataFrame | None = None
        self.fallback: dict[float, float] = {}

    def fit(self, train: pd.DataFrame, series: pd.Series) -> "Climatology":
        t = pd.DatetimeIndex(train["target_time"])
        g = pd.DataFrame({"y": train["y"].to_numpy(),
                          "dow": t.dayofweek, "slot": t.hour * 4 + t.minute // 15})
        self.table = g.groupby(["dow", "slot"])["y"].quantile(list(self.quantiles)).unstack()
        self.fallback = {q: float(np.quantile(train["y"], q)) for q in self.quantiles}
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        t = pd.DatetimeIndex(frame["target_time"])
        key = pd.MultiIndex.from_arrays([t.dayofweek, t.hour * 4 + t.minute // 15])
        out = {}
        for q in self.quantiles:
            v = self.table[q].reindex(key).to_numpy()
            out[q] = np.where(np.isnan(v), self.fallback[q], v)
        return out


class ConstantMargin(Baseline):
    """No forecast at all: one number for every interval of every day, set to a
    high percentile of the training block. This is the static derating an
    engineer applies today, expressed as a forecaster so it can be dropped into
    the same optimiser and the same table."""

    name = "Static margin (no forecast)"
    definition = "a single constant, the training-set p95, for every interval"
    kind = "quantile"

    def __init__(self, percentile: float = 0.95):
        self.percentile = percentile
        self.value = 0.0
        self.levels: dict[float, float] = {}

    def fit(self, train: pd.DataFrame, series: pd.Series) -> "ConstantMargin":
        self.value = float(np.quantile(train["y"], self.percentile))
        self.levels = {q: float(np.quantile(train["y"], q)) for q in QUANTILES}
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        n = len(frame)
        return {q: np.full(n, self.levels[q]) for q in QUANTILES}


class PerfectForesight(Baseline):
    """The ceiling. Every quantile is the truth, so the chance constraint
    collapses to a deterministic one. Not a model; a bound on what any model
    could be worth."""

    name = "Perfect foresight"
    definition = "the actual value (upper bound on achievable skill)"
    kind = "quantile"

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        y = frame["y"].to_numpy(dtype=float)
        return {q: y.copy() for q in QUANTILES}


# ---------------------------------------------------------------------------
# Linear quantile regression
# ---------------------------------------------------------------------------

@dataclass
class _Standardiser:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, X: np.ndarray) -> "_Standardiser":
        self.mean = X.mean(axis=0)
        self.scale = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


class LinearQuantile(Baseline):
    """Pinball-loss linear regression, one model per quantile.

    Fitted by Adam on the exact (non-smoothed) pinball loss rather than by
    linear programming: the LP formulation of quantile regression on four
    million rows is not something to run under time pressure, and subgradient
    descent on a convex objective reaches the same optimum. Checked against
    scikit-learn's LP solver on a subsample in ``tests/test_baselines.py``.

    Its job in the table is to say how much of our skill is the nonlinearity and
    how much was available from the same features with a straight line.
    """

    name = "Linear quantile"
    definition = "pinball-loss linear regression on the identical feature matrix"
    kind = "quantile"

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES, epochs: int = 20,
                 batch: int = 8192, lr: float = 0.1, seed: int = 0):
        self.quantiles = quantiles
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.std = _Standardiser()
        self.w: dict[float, np.ndarray] = {}
        self.b: dict[float, float] = {}

    def fit(self, train: pd.DataFrame, series: pd.Series) -> "LinearQuantile":
        X = train[FEATURE_COLS].to_numpy(dtype=np.float64)
        y = train["y"].to_numpy(dtype=np.float64)
        Z = self.std.fit(X)(X)
        rng = np.random.default_rng(self.seed)
        n, d = Z.shape
        for q in self.quantiles:
            w = np.zeros(d)
            b = float(np.quantile(y, q))
            mw, vw = np.zeros(d), np.zeros(d)
            mb = vb = 0.0
            step = 0
            for _ in range(self.epochs):
                perm = rng.permutation(n)
                for s in range(0, n, self.batch):
                    idx = perm[s : s + self.batch]
                    Zi, yi = Z[idx], y[idx]
                    r = yi - (Zi @ w + b)
                    # subgradient of the pinball loss
                    g = np.where(r > 0, -q, 1.0 - q)
                    gw = Zi.T @ g / len(idx)
                    gb = float(g.mean())
                    step += 1
                    mw = 0.9 * mw + 0.1 * gw
                    vw = 0.999 * vw + 0.001 * gw**2
                    mb = 0.9 * mb + 0.1 * gb
                    vb = 0.999 * vb + 0.001 * gb**2
                    mhw = mw / (1 - 0.9**step)
                    vhw = vw / (1 - 0.999**step)
                    mhb = mb / (1 - 0.9**step)
                    vhb = vb / (1 - 0.999**step)
                    w -= self.lr * mhw / (np.sqrt(vhw) + 1e-8)
                    b -= self.lr * mhb / (np.sqrt(vhb) + 1e-8)
            self.w[q], self.b[q] = w, b
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        Z = self.std(frame[FEATURE_COLS].to_numpy(dtype=np.float64))
        return {q: Z @ self.w[q] + self.b[q] for q in self.quantiles}


# ---------------------------------------------------------------------------
# The deep learning benchmark
# ---------------------------------------------------------------------------

class NeuralQuantile(Baseline):
    """An N-BEATS-style residual MLP stack with a multi-quantile pinball head.

    This exists to settle the buzzword question with a measurement instead of an
    opinion. It is trained on the same features, the same split and the same
    calibration layer as everything else, so whichever way it lands the
    comparison means something. If it wins we ship it and say why; if it loses,
    that is the expected result at this data scale and we now have the table to
    prove we checked rather than avoided it.

    All quantiles come out of one network with a shared trunk, which is both
    cheaper and less prone to crossing than five independent fits.
    """

    name = "Neural quantile"
    definition = "residual MLP stack, shared trunk, pinball head on all five levels"
    kind = "quantile"

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES, width: int = 256,
                 blocks: int = 3, epochs: int = 6, batch: int = 4096, lr: float = 1e-3,
                 seed: int = 0, device: str = "cpu"):
        self.quantiles = quantiles
        self.width, self.blocks, self.epochs = width, blocks, epochs
        self.batch, self.lr, self.seed, self.device = batch, lr, seed, device
        self.std = _Standardiser()
        self.net = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.history: list[dict] = []

    def _build(self, d: int):
        import torch
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.f = nn.Sequential(
                    nn.Linear(width, width), nn.ReLU(),
                    nn.Linear(width, width), nn.ReLU(),
                )

            def forward(self, x):
                return x + self.f(x)

        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(d, self.width), nn.ReLU(),
            *[Block(self.width) for _ in range(self.blocks)],
            nn.Linear(self.width, len(self.quantiles)),
        )

    def fit(self, train: pd.DataFrame, series: pd.Series,
            valid: pd.DataFrame | None = None) -> "NeuralQuantile":
        import torch

        X = train[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = train["y"].to_numpy(dtype=np.float32)
        Z = self.std.fit(X)(X).astype(np.float32)
        self.y_mean, self.y_scale = float(y.mean()), float(max(y.std(), 1e-6))
        yn = (y - self.y_mean) / self.y_scale

        torch.manual_seed(self.seed)
        self.net = self._build(Z.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        qs = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)

        Zt = torch.from_numpy(Z)
        yt = torch.from_numpy(yn)
        n = len(Zt)
        g = torch.Generator().manual_seed(self.seed)

        Zv = yv = None
        if valid is not None:
            Zv = torch.from_numpy(
                self.std(valid[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32))
            yv = torch.from_numpy(
                ((valid["y"].to_numpy(dtype=np.float32) - self.y_mean) / self.y_scale))

        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot, seen = 0.0, 0
            self.net.train()
            for s in range(0, n, self.batch):
                idx = perm[s : s + self.batch]
                zb, yb = Zt[idx].to(self.device), yt[idx].to(self.device)
                pred = self.net(zb)
                d = yb.unsqueeze(1) - pred
                loss = torch.maximum(qs * d, (qs - 1.0) * d).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
                seen += len(idx)
            sched.step()
            rec = {"epoch": ep + 1, "train_pinball": tot / seen}
            if Zv is not None:
                self.net.eval()
                with torch.no_grad():
                    vl, vn = 0.0, 0
                    for s in range(0, len(Zv), 65536):
                        zb, yb = Zv[s : s + 65536], yv[s : s + 65536]
                        d = yb.unsqueeze(1) - self.net(zb)
                        vl += float(torch.maximum(qs * d, (qs - 1.0) * d).mean()) * len(zb)
                        vn += len(zb)
                    rec["valid_pinball"] = vl / vn
            self.history.append(rec)
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        import torch

        Z = self.std(frame[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32)
        self.net.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(Z), 65536):
                outs.append(self.net(torch.from_numpy(Z[s : s + 65536]).to(self.device)).cpu().numpy())
        P = np.vstack(outs) * self.y_scale + self.y_mean
        return {q: P[:, i].astype(float) for i, q in enumerate(self.quantiles)}


# ---------------------------------------------------------------------------

class LightGBMQuantile(Baseline):
    """Ours. One booster per quantile on the shared feature matrix."""

    name = "LightGBM quantile (ours)"
    definition = "gradient-boosted trees, one booster per quantile"
    kind = "quantile"

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES, n_estimators: int = 600,
                 params: dict | None = None, seed: int = 0):
        from forecast.train import LGB_PARAMS

        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.params = {**LGB_PARAMS, **(params or {}), "seed": seed}
        self.models: dict[float, object] = {}
        self.best_iteration: dict[str, int] = {}

    def fit(self, train: pd.DataFrame, series: pd.Series,
            valid: pd.DataFrame | None = None) -> "LightGBMQuantile":
        import lightgbm as lgb

        Xtr, ytr = train[FEATURE_COLS], train["y"].to_numpy()
        cb, vs = [], []
        if valid is not None:
            vs = [lgb.Dataset(valid[FEATURE_COLS], valid["y"].to_numpy())]
            cb = [lgb.early_stopping(50, verbose=False)]
        for q in self.quantiles:
            m = lgb.train({**self.params, "alpha": q}, lgb.Dataset(Xtr, ytr),
                          num_boost_round=self.n_estimators, valid_sets=vs, callbacks=cb)
            self.models[q] = m
            self.best_iteration[str(q)] = int(m.best_iteration or self.n_estimators)
        return self

    def predict(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        X = frame[FEATURE_COLS]
        return {q: m.predict(X, num_iteration=m.best_iteration) for q, m in self.models.items()}


#: The benchmark, in the order it is reported. Perfect foresight and the static
#: margin bracket the table: nothing can beat the first, and the second is what
#: the site does today.
REGISTRY: dict[str, type[Baseline]] = {
    "static_margin": ConstantMargin,
    "persistence": Persistence,
    "seasonal_naive": SeasonalNaive,
    "climatology": Climatology,
    "linear_quantile": LinearQuantile,
    "lightgbm_quantile": LightGBMQuantile,
    "neural_quantile": NeuralQuantile,
    "perfect_foresight": PerfectForesight,
}
