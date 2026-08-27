"""Track C, Stage 3: train the forecaster on the decision, not on the score.

Stage 2 told the booster *where* to spend its capacity and left the loss alone.
Stage 3 changes the loss: the gradient the model descends is the gradient of the
realised rupee cost of the schedule the optimiser produces from its output.

**The obstacle.** The optimiser is a MILP and MILPs are not differentiable --
the integer variables kill the gradient, and even relaxed, the argmin of a
linear program is piecewise constant in its right-hand side, so the exact
derivative is zero almost everywhere and undefined on the measure-zero set where
anything happens. The plan lists three ways around this. Option 1 relaxes the
binaries and differentiates through the resulting convex program with
``cvxpylayers``. Option 3 is the SPO+ surrogate, which is built for parameters
in the *objective*; here the forecast enters the *right-hand side*, so it does
not apply without reformulation.

**What is implemented is Option 2**, perturbed optimisers (Berthet et al.,
*Learning with Differentiable Perturbed Optimizers*). Smooth the problem by
integrating over noise in the parameters and estimate the gradient of the
smoothed objective from solutions at perturbed inputs. Concretely, a two-sided
directional estimate: draw a random direction over the block-level forecast,
solve the real optimiser at plus and minus a step along it, evaluate both
resulting schedules against the true future, and project the difference back
onto the direction. Two solves buy an unbiased estimate of the smoothed
gradient in every coordinate at once, which is what makes this affordable at
16 hours and 32 blocks.

It has three properties the other options do not. The genuine solver stays in
the loop, binaries included, so the gradient describes the optimiser that will
actually be deployed rather than a relaxation of it. It needs no new dependency.
And it degrades gracefully: as the smoothing radius shrinks the estimate gets
noisier rather than wrong, so the failure mode is slow learning, not a
confidently incorrect direction.

**The anchor.** The decision gradient is mixed with the ordinary pinball
gradient at weight ``lam``. That is not timidity. A pure decision loss is
indifferent to the forecast everywhere the constraint is slack -- which
``eval/decision_regret.py`` measures at a large share of blocks -- so nothing
stops it wandering arbitrarily far off there, and the conformal layer downstream
would then be calibrating a surface that is nonsense over most of its domain.
The pinball term keeps the model a forecast; the decision term bends it where
bending pays.

**Only q95 is tuned.** The ceiling constraint reads one bound. Tuning the whole
ladder would spend four times the compute moving quantiles that no constraint
consumes, and would disturb the interval the calibration report is about.

**Honest framing of the split.** April is the decision-training block, May is
the conformal calibration block, June is the test month. The baseline in the
frontier study is calibrated on May alone as well, so the comparison isolates
the training signal rather than the calibration window. That is a different
split from the one the shipped model uses, which calibrates on April and May
together, and the numbers here are therefore not comparable to
`results/ablation.md` -- only to each other.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from control.decision import DecisionLoss
from control.mpc import ChanceConstrainedMPC, MPCConfig
from forecast.sources import QCOLS, ForecastSource
from sim.thermal import BuildingSim

LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


class ArrayForecast(ForecastSource):
    """A forecast source backed by an array the caller can rewrite in place.

    The tuner changes the model between boosting rounds, so the source the
    optimiser reads has to change with it. ``TensorForecast`` is built once from
    a parquet and is the wrong shape for that.
    """

    name = "array"

    def __init__(self, grid: np.ndarray, pv_quantiles, horizon: int):
        # grid is (n_origins, H, K) in the order of LEVELS
        self.grid = grid
        self.pvq = pv_quantiles
        self.max_horizon = horizon
        self._qi = {qn: i for i, qn in enumerate(QCOLS)}

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        row = self.grid[k0, :H, self._qi[q]]
        if np.isnan(row).any():
            row = np.where(np.isnan(row), np.nanmedian(self.grid[k0, :H, 2]), row)
        return row

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.pvq.forecast(k0, H, q)


@dataclass
class DecisionEnv:
    """Everything needed to price a forecast in rupees at a given origin.

    Holds the plant, the tariff, the true future, and a pre-computed state
    trajectory so that a gradient at origin k starts from the state the
    controller would actually be in at origin k. Sampling origins with a
    made-up state would put the gradient in the wrong place: the whole question
    is whether the *ceiling* binds, and whether it binds depends on how much
    thermal mass has already been charged.
    """

    tariff: object
    params: object
    exog: pd.DataFrame
    index: pd.DatetimeIndex
    pvq: object
    target_kw: float
    cfg: MPCConfig
    seed: int = 0
    #: The source the optimiser reads. The tuner rewrites its q95 layer between
    #: boosting rounds, which is the mechanism by which the gradient describes
    #: the model as it currently stands rather than as it was initialised.
    fc: "ArrayForecast | None" = None
    loss: DecisionLoss = field(init=False)
    trace: list = field(init=False)

    def __post_init__(self) -> None:
        sim = BuildingSim(self.params, self.exog, seed=self.seed)
        self.full_exog = sim.exog
        self.loss = DecisionLoss(
            self.tariff, self.params, sim.exog, demand_target_kw=self.target_kw,
            comfort_penalty=self.cfg.comfort_penalty,
            peak_shaping_cost=self.cfg.peak_shaping_cost)
        self.trace = []

    def build_trace(self, fc: ForecastSource) -> None:
        """One closed-loop pass to record the state at every origin."""
        from eval.decision_regret import state_trace

        ctrl = ChanceConstrainedMPC(self.tariff, fc, self.cfg, risk_quantile="q95",
                                    solar_quantile="q05", demand_target_kw=self.target_kw)
        ctx = {"params": self.params, "exog": self.exog, "tariff": self.tariff}
        self.trace = state_trace(ctx, ctrl, seed=self.seed)

    def gradient(
        self, fc: ArrayForecast, k0: int, snap: dict, sigma_kw: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float] | None:
        """Two-sided directional gradient of realised cost w.r.t. the q95 path.

        Returns a per-step gradient of length H, in INR per kW, and the baseline
        cost. The direction is drawn per *block* rather than per step because
        the ceiling constraint only sees block averages; a per-step direction
        would spend most of its magnitude in a subspace the optimiser cannot
        respond to, which is variance for nothing.
        """
        from sim.thermal import BuildingSim

        sim = BuildingSim(self.params, self.exog, seed=self.seed)
        sim.k = int(k0)
        sim.state.t_indoor = float(snap["t_indoor"])
        sim.state.t_tank = float(snap["t_tank"])
        sim.state.ev_delivered_kwh = float(snap["ev_delivered_kwh"])
        sim.state.soc = float(snap["soc"])
        obs = sim.observe()

        H = min(self.cfg.horizon, len(self.index) - k0, fc.max_horizon)
        if H <= 2:
            return None
        probe = ChanceConstrainedMPC(self.tariff, fc, self.cfg, risk_quantile="q95",
                                     solar_quantile="q05", demand_target_kw=self.target_kw)
        probe.d_committed_kw = float(snap["d_committed_kw"])
        probe._block_progress = list(snap["block_progress"])
        blocks = probe._billing_blocks(self.index[k0:k0 + H], H)

        u_block = rng.standard_normal(len(blocks))
        step_dir = np.zeros(H)
        for bi, b in enumerate(blocks):
            step_dir[np.asarray(b["steps"], int)] = u_block[bi]

        costs = {}
        for sign in (+1.0, -1.0):
            shifted = _DirectionShift(fc, sign * sigma_kw * step_dir)
            pr = ChanceConstrainedMPC(self.tariff, shifted, self.cfg, risk_quantile="q95",
                                      solar_quantile="q05", demand_target_kw=self.target_kw)
            pr.d_committed_kw = probe.d_committed_kw
            pr._block_progress = list(probe._block_progress)
            r = pr.solve(sim, obs)
            if not r.plan:
                return None
            costs[sign] = self.loss.evaluate(k0, r.plan, obs["t_indoor"])["total_inr"]

        # Berthet's perturbed-optimiser estimate, two-sided: the scalar
        # (J+ - J-) / (2*sigma) times the direction is an unbiased estimate of
        # the gradient of the Gaussian-smoothed objective in every coordinate.
        scale = (costs[+1.0] - costs[-1.0]) / (2.0 * sigma_kw)
        return scale * step_dir, 0.5 * (costs[+1.0] + costs[-1.0])


class _DirectionShift:
    """``fc`` with a per-step offset added to every quantile level."""

    def __init__(self, inner: ForecastSource, offset: np.ndarray):
        self.inner = inner
        self.offset = np.asarray(offset, float)
        self.max_horizon = getattr(inner, "max_horizon", 10**9)

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        out = np.asarray(self.inner.base(k0, H, q), float).copy()
        n = min(H, len(self.offset))
        out[:n] += self.offset[:n]
        return out

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.inner.pv(k0, H, q)


# ---------------------------------------------------------------------------
# the tuner
# ---------------------------------------------------------------------------

def tune_q95(
    booster: lgb.Booster,
    sup: pd.DataFrame,
    feature_cols: list[str],
    env: DecisionEnv,
    origin_pos: dict,
    n_rounds: int = 60,
    lam: float = 0.5,
    learning_rate: float = 0.03,
    batch_origins: int = 24,
    sigma_kw: float = 8.0,
    seed: int = 0,
    verbose_every: int = 10,
) -> tuple[lgb.Booster, list[dict]]:
    """Continue boosting ``booster`` on a mixture of pinball and decision loss.

    ``origin_pos`` maps an origin's index in the evaluation window to the row
    positions of that origin's (horizon 1..H) rows in ``sup``. That map is what
    connects a gradient computed in the optimiser's coordinates back to the rows
    the booster is fitted on, and getting it wrong is silent -- the model would
    still train, on gradients attached to the wrong intervals.
    """
    q = 0.95
    X, y = sup[feature_cols], sup["y"].to_numpy()
    rng = np.random.default_rng(seed)
    origins = np.array(sorted(origin_pos))
    trace_by_k = {int(s["k"]): s for s in env.trace}
    usable = np.array([k for k in origins if k in trace_by_k])
    if len(usable) < batch_origins:
        raise ValueError(f"only {len(usable)} usable origins for decision gradients")

    log: list[dict] = []
    state = {"round": 0, "grad_rms": 1.0}
    t0 = time.perf_counter()

    def fobj(preds: np.ndarray, dataset: lgb.Dataset):
        # pinball gradient: -q where the actual is above the prediction, 1-q below
        g_pin = np.where(y > preds, -q, 1.0 - q)
        g_dec = np.zeros_like(preds)

        picks = rng.choice(usable, size=batch_origins, replace=False)
        hits, costs = 0, []
        # rebuild the source from the *current* predictions so the optimiser sees
        # the model as it stands this round, not as it stood at initialisation
        for k0 in picks:
            pos = origin_pos[int(k0)]
            env.fc.grid[int(k0), : len(pos["rows"]), 4] = preds[pos["rows"]]
        for k0 in picks:
            out = env.gradient(env.fc, int(k0), trace_by_k[int(k0)], sigma_kw, rng)
            if out is None:
                continue
            gstep, c = out
            pos = origin_pos[int(k0)]
            n = min(len(gstep), len(pos["rows"]))
            g_dec[pos["rows"][:n]] = gstep[:n]
            hits += 1
            costs.append(c)

        rms = float(np.sqrt(np.mean(g_dec[g_dec != 0] ** 2))) if hits else 0.0
        if rms > 0:
            # Normalise so ``lam`` is a relative weight rather than a unit
            # conversion between rupees-per-kW and the dimensionless pinball
            # subgradient. Without this, lam would have to be retuned for every
            # building and every tariff.
            state["grad_rms"] = 0.9 * state["grad_rms"] + 0.1 * rms
            g_dec = g_dec / max(state["grad_rms"], 1e-9)

        grad = (1.0 - lam) * g_pin + lam * g_dec
        state["round"] += 1
        if costs and (state["round"] % verbose_every == 0 or state["round"] == 1):
            rec = {"round": state["round"], "mean_cost_inr": float(np.mean(costs)),
                   "grad_rms": rms, "origins": hits,
                   "wall_s": time.perf_counter() - t0}
            log.append(rec)
            print(f"     round {rec['round']:>3}  batch cost Rs {rec['mean_cost_inr']:9,.0f}  "
                  f"|g| {rec['grad_rms']:8.3f}  {rec['wall_s']:5.0f}s")
        return grad, np.ones_like(preds)

    # LightGBM 4.x takes a custom objective through ``params``, not through a
    # ``fobj`` argument -- that keyword was removed in 4.0 and passing it is
    # silently ignored by ``**kwargs`` in some wrappers, which trains a model on
    # the default objective and looks like the method not working.
    tuned = lgb.train(
        {"objective": fobj, "learning_rate": learning_rate, "num_leaves": 63,
         "min_data_in_leaf": 200, "feature_fraction": 0.85, "verbose": -1,
         "num_threads": 8},
        lgb.Dataset(X, y, free_raw_data=False),
        num_boost_round=n_rounds,
        init_model=booster,
    )
    return tuned, log


def build_origin_map(sup: pd.DataFrame, index: pd.DatetimeIndex, horizon: int) -> dict:
    """origin position in ``index`` -> the row positions of its horizon rows.

    Only origins whose whole horizon fits inside the window are kept. A partial
    horizon would give the optimiser a shorter planning window than the one it
    is being trained for, and the gradient would describe a different problem.
    """
    pos = pd.Series(np.arange(len(index)), index=index)
    o = pos.reindex(pd.to_datetime(sup["origin"])).to_numpy()
    h = sup["horizon"].to_numpy().astype(int)
    ok = ~np.isnan(o)
    out: dict[int, dict] = {}
    order = np.argsort(o[ok] * (horizon + 1) + h[ok])
    rows_all = np.arange(len(sup))[ok][order]
    o_s, h_s = o[ok][order].astype(int), h[ok][order]
    starts = np.flatnonzero(np.r_[True, o_s[1:] != o_s[:-1]])
    ends = np.r_[starts[1:], len(o_s)]
    for a, b in zip(starts, ends):
        k0 = int(o_s[a])
        if k0 + horizon >= len(index):
            continue
        hh = h_s[a:b]
        if len(hh) < horizon or hh[0] != 1:
            continue
        out[k0] = {"rows": rows_all[a:b][:horizon], "h": hh[:horizon]}
    return out
