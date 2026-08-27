"""Joint trajectories, because marginal quantiles do not compose over a horizon.

The hole this closes, stated precisely. The forecaster reports a quantile per
timestep. The demand-ceiling constraint spans 64 timesteps. A marginal 95th
percentile says that at *each* step, taken alone, the actual exceeds the bound
5% of the time. It says nothing whatever about the probability that *some* step
in the window exceeds it, which is the event the constraint is actually trying
to prevent. Under independence that probability is 1 - 0.95^64, about 96%. Real
load is heavily autocorrelated so the truth is far lower, but it is nowhere near
5% and marginals alone cannot tell you what it is.

It is worse for the billing case. The monthly demand charge is set by the
*maximum* over roughly 2,880 intervals, and the distribution of a maximum is a
function of the joint law, not of the marginals. A per-interval quantile is
close to silent about it.

The fix is to stop reasoning about quantiles and start sampling paths. This
module takes the calibrated marginals the repo already trusts -- conformalised,
audited, unchanged -- and adds the dependence structure back on top with a
Gaussian copula. Nothing about the marginal calibration is disturbed: at the
reported levels the inverse CDF below is exact, so a path drawn from this and
read at one timestep still has the q95 that Track A certified.

Route 2 in the plan was a generative forecaster that emits paths natively
(DeepAR, a normalising flow). This is Route 1, and it is first for a reason:
it reuses every hour already spent on the marginals, and it makes the joint
structure a separate, inspectable object rather than something entangled in a
network's weights.

**What the Gaussian copula does and does not buy.** It reproduces the rank
correlation across horizons, which is the dominant effect and the one the
independence argument gets catastrophically wrong. It does not reproduce tail
dependence: a Gaussian copula is asymptotically independent in the tails, so
the probability that steps 30 and 31 are *both* extreme is understated. For a
ceiling constraint that is the conservative direction for the wrong reason, and
it is stated in the limitations rather than papered over. A t-copula is the
usual next step and is a one-parameter change to :func:`fit_copula`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
#: PIT values are clipped away from 0 and 1 before the normal-score transform,
#: because ndtri(0) is -inf and one infinite residual would destroy a whole
#: correlation matrix.
_EPS = 1e-4


# ---------------------------------------------------------------------------
# the marginal, as a monotone map between the value and its normal score
# ---------------------------------------------------------------------------

def _ladder_z(levels=LEVELS) -> np.ndarray:
    return ndtri(np.asarray(levels, float))


def _monotone(ladder: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Enforce a strictly increasing quantile ladder along the last axis.

    Independently fitted quantiles can cross even after the pointwise sort
    upstream, and a flat pair makes the interpolation below non-invertible.
    """
    out = np.maximum.accumulate(np.asarray(ladder, float), axis=-1)
    bump = eps * np.arange(out.shape[-1])
    return out + bump


def quantile_to_value(ladder: np.ndarray, z: np.ndarray, levels=LEVELS) -> np.ndarray:
    """Inverse CDF: normal score -> load, by interpolating the quantile ladder.

    Linear in *normal-score* space rather than in probability space, which is
    what makes the extrapolation beyond q05 and q95 behave. Probability-space
    extrapolation past the outermost reported level either clips (so no sampled
    path is ever more extreme than q95, which would make the whole exercise
    circular) or runs away. Interpolating in z and continuing the outermost
    slope is the standard construction, and it is exact at the five reported
    levels -- so a sample read at one timestep has precisely the calibrated
    marginal that Track A certified.

    ``ladder`` is (..., K) and ``z`` broadcasts against (...,).
    """
    lz = _ladder_z(levels)
    lad = _monotone(ladder)
    K = lad.shape[-1]
    shape = np.broadcast_shapes(lad.shape[:-1], np.shape(z))
    q = np.broadcast_to(lad, shape + (K,)).reshape(-1, K)
    zz = np.broadcast_to(np.asarray(z, float), shape).reshape(-1)

    # Segment index in 0..K. The knots are the same for every row -- they are
    # the normal scores of the reported levels -- so one searchsorted places the
    # whole batch, and the two extrapolated tails fall out as segments 0 and K.
    j = np.clip(np.searchsorted(lz, zz), 1, K - 1)
    z0, z1 = lz[j - 1], lz[j]
    q0 = np.take_along_axis(q, (j - 1)[:, None], axis=1)[:, 0]
    q1 = np.take_along_axis(q, j[:, None], axis=1)[:, 0]
    out = q0 + (q1 - q0) * (zz - z0) / (z1 - z0)
    return out.reshape(shape)


def value_to_z(ladder: np.ndarray, y: np.ndarray, levels=LEVELS) -> np.ndarray:
    """Forward transform: load -> normal score. The exact inverse of
    :func:`quantile_to_value`, including the two extrapolated tails."""
    lz = _ladder_z(levels)
    lad = _monotone(ladder)
    K = lad.shape[-1]
    shape = np.broadcast_shapes(lad.shape[:-1], np.shape(y))
    q = np.broadcast_to(lad, shape + (K,)).reshape(-1, K)
    yy = np.broadcast_to(np.asarray(y, float), shape).reshape(-1)

    # Here the knots differ row by row, so the segment is found by counting how
    # many ladder entries the value clears rather than by a shared searchsorted.
    j = np.clip((yy[:, None] > q).sum(axis=1), 1, K - 1)
    q0 = np.take_along_axis(q, (j - 1)[:, None], axis=1)[:, 0]
    q1 = np.take_along_axis(q, j[:, None], axis=1)[:, 0]
    z0, z1 = lz[j - 1], lz[j]
    out = z0 + (z1 - z0) * (yy - q0) / (q1 - q0)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# the copula
# ---------------------------------------------------------------------------

def nearest_psd(c: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Project onto the correlation matrices: clip eigenvalues, rescale the
    diagonal back to one.

    An empirical correlation matrix estimated from overlapping windows with
    missing cells is routinely indefinite by a hair, and a Cholesky factorisation
    of an indefinite matrix does not fail loudly, it fails at the worst moment.
    """
    c = np.asarray(c, float)
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, ridge)
    out = (v * w) @ v.T
    d = np.sqrt(np.clip(np.diag(out), 1e-12, None))
    return np.clip(out / np.outer(d, d), -1.0, 1.0)


@dataclass
class CopulaModel:
    """A correlation matrix over horizon steps, plus an optional tail parameter.

    ``df`` selects the family. ``None`` is a Gaussian copula. A finite value is
    a t-copula with that many degrees of freedom, which is the same correlation
    structure with upper-tail dependence added: extremes at neighbouring
    horizons co-occur instead of decorrelating.

    That distinction is not cosmetic here, and the measurement is in
    ``results/horizon_risk.md``. The event the ceiling constraint is about is
    "does *any* step in the window break the bound", which is a statement about
    the joint upper tail. A Gaussian copula is asymptotically independent up
    there, so it scatters exceedances across more distinct trajectories than
    reality does and materially over-states that probability -- over-states it
    even when fitted on the evaluation month itself, so it is a property of the
    family and not a train/test artefact. Lower ``df`` binds the tails together.
    """

    corr: np.ndarray
    horizon: int
    levels: tuple = LEVELS
    df: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.corr = np.asarray(self.corr, float)
        self._chol = np.linalg.cholesky(nearest_psd(self.corr))

    def sample_z(self, n: int, H: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` correlated normal-score paths of length ``H``.

        For H shorter than the fitted horizon the leading block is used, which
        is the correct conditional-free marginalisation of a Gaussian: dropping
        variables from a joint normal leaves the remaining block's covariance
        untouched.
        """
        if H > self.horizon:
            raise ValueError(f"copula fitted to horizon {self.horizon}, asked for {H}")
        L = np.linalg.cholesky(nearest_psd(self.corr[:H, :H]))
        z = rng.standard_normal((n, H)) @ L.T
        if self.df is None:
            return z
        # t-copula: one chi-square draw *shared across the whole path*, which is
        # exactly the mechanism that produces tail dependence -- a small draw
        # inflates every step of that trajectory together, so an extreme
        # afternoon is extreme for several consecutive steps rather than for one.
        from scipy.stats import t as _t
        nu = float(self.df)
        w = np.sqrt(nu / rng.chisquare(nu, size=(n, 1)))
        u = np.clip(_t.cdf(z * w, nu), _EPS, 1.0 - _EPS)
        return ndtri(u)

    def sample(self, ladder: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` joint load trajectories from an (H, K) quantile ladder.

        Returns (n, H). Read down a column and you recover the calibrated
        marginal; read across a row and you get a path whose autocorrelation
        looks like the forecast errors this was fitted on.
        """
        ladder = np.asarray(ladder, float)
        H = ladder.shape[0]
        z = self.sample_z(n, H, rng)
        return quantile_to_value(np.broadcast_to(ladder, (n, H, ladder.shape[-1])),
                                 z, self.levels)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), corr=self.corr)
        path.with_suffix(".json").write_text(json.dumps(
            {"horizon": self.horizon, "levels": list(self.levels),
             "df": self.df, "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CopulaModel":
        path = Path(path)
        d = json.loads(path.with_suffix(".json").read_text())
        corr = np.load(path.with_suffix(".npz"))["corr"]
        return cls(corr=corr, horizon=int(d["horizon"]), levels=tuple(d["levels"]),
                   df=d.get("df"), meta=d.get("meta", {}))


def _pivot(frame: pd.DataFrame, cols: list[str], horizon: int) -> dict[str, np.ndarray]:
    """(n_origins, H) matrices keyed by column, one row per forecast origin."""
    f = frame[frame["horizon"] <= horizon]
    out = {}
    for c in cols:
        p = f.pivot_table(index="origin", columns="horizon", values=c, aggfunc="first")
        p = p.reindex(columns=range(1, horizon + 1))
        out[c] = p.to_numpy()
        out["_index"] = p.index
    return out


def fit_copula(
    frame: pd.DataFrame, horizon: int = 64, levels=LEVELS, shrink: float = 0.05,
    min_origins: int = 200,
) -> CopulaModel:
    """Fit the horizon dependence on a calibration block.

    ``frame`` carries ``origin``, ``horizon``, ``actual`` and the quantile
    columns ``q05..q95`` -- the same shape the forecast tensor already has, so
    this reads the artefact the trainer already writes.

    The rows are forecast origins and the columns are horizon steps, so one row
    is one whole predicted path against what actually happened. That is the
    right unit: what is being estimated is how a forecast that is high at
    +2 hours behaves at +3 hours, and only paths from a common origin carry
    that information.

    ``shrink`` pulls the estimate toward the identity. With 64 horizons and a
    few thousand origins the sample correlation is estimable but not well
    conditioned, and a Cholesky of a nearly singular matrix produces sample
    paths with no variation left in the far horizons.
    """
    qcols = [f"q{int(q*100):02d}" for q in levels]
    piv = _pivot(frame, ["actual"] + qcols, horizon)
    y = piv["actual"]                                    # (n, H)
    lad = np.stack([piv[c] for c in qcols], axis=-1)     # (n, H, K)

    ok = ~np.isnan(y).any(axis=1) & ~np.isnan(lad).any(axis=(1, 2))
    y, lad = y[ok], lad[ok]
    if len(y) < min_origins:
        raise ValueError(f"only {len(y)} complete origins; need {min_origins}")

    z = np.empty_like(y)
    for h in range(y.shape[1]):
        z[:, h] = value_to_z(lad[:, h, :], y[:, h], levels)
    z = np.clip(z, ndtri(_EPS), ndtri(1 - _EPS))

    c = np.corrcoef(z, rowvar=False)
    c = (1.0 - shrink) * c + shrink * np.eye(c.shape[0])
    corr = nearest_psd(c)

    return CopulaModel(
        corr=corr, horizon=horizon, levels=tuple(levels),
        meta={
            "n_origins": int(len(y)),
            "shrink": shrink,
            "mean_z": float(z.mean()),
            "sd_z": float(z.std()),
            # the number that makes the point: adjacent horizons are almost the
            # same forecast error, which is exactly why treating them as
            # independent is wrong by a mile
            "corr_lag1": float(np.mean(np.diag(corr, 1))),
            "corr_lag4": float(np.mean(np.diag(corr, 4))) if horizon > 4 else None,
            "corr_lag32": float(np.mean(np.diag(corr, 32))) if horizon > 32 else None,
        },
    )


def fit_ratio_copula(ratio: np.ndarray, horizon: int, shrink: float = 0.05) -> CopulaModel:
    """Dependence for a scalar series with no quantile ladder of its own.

    Used for the PV forecast, whose uncertainty comes from the site's measured
    cloud record rather than from a fitted model. The series is rank-transformed
    to normal scores, its autocorrelation is read off at lags 0..H-1, and the
    Toeplitz matrix that implies is projected onto the correlation cone. Toeplitz
    rather than free because there is one series here, not thousands of
    independent paths, and an unconstrained (H, H) estimate from a single
    realisation would be mostly noise.
    """
    r = np.asarray(ratio, float)
    r = r[np.isfinite(r)]
    if len(r) < 10 * horizon:
        return CopulaModel(corr=np.eye(horizon), horizon=horizon,
                           meta={"note": "too little history; independence assumed"})
    # rank -> uniform -> normal score
    u = (pd.Series(r).rank(method="average").to_numpy() - 0.5) / len(r)
    z = ndtri(np.clip(u, _EPS, 1 - _EPS))
    z = (z - z.mean()) / max(z.std(), 1e-9)
    ac = np.array([1.0] + [float(np.corrcoef(z[:-k], z[k:])[0, 1]) for k in range(1, horizon)])
    ac = (1.0 - shrink) * ac
    ac[0] = 1.0
    from scipy.linalg import toeplitz
    return CopulaModel(corr=nearest_psd(toeplitz(ac)), horizon=horizon,
                       meta={"n": int(len(r)), "corr_lag1": float(ac[1]),
                             "shrink": shrink, "structure": "toeplitz"})


# ---------------------------------------------------------------------------
# what the samples are for: the statistics marginals cannot produce
# ---------------------------------------------------------------------------

def path_exceedance(paths: np.ndarray, bound: np.ndarray) -> float:
    """Fraction of sampled paths that break through ``bound`` at any step.

    This is the horizon-level violation probability the constraint is really
    about, and it is the number a marginal quantile cannot produce. Compare it
    with 1 - 0.95^H (what independence would imply) and with the empirical rate
    from held-out actuals: the three together are the whole Track B argument on
    one line.
    """
    return float(np.mean((np.asarray(paths, float) > np.asarray(bound, float)[None, :]).any(axis=1)))


def block_max(paths: np.ndarray, steps_per_block: int) -> np.ndarray:
    """Peak block average per path -- the quantity the demand charge bills on.

    The tariff meters a 30-minute average, so the maximum that matters is a
    maximum of block means, not of instantaneous values. Taking the max of raw
    15-minute samples would overstate it.
    """
    p = np.asarray(paths, float)
    n, H = p.shape
    m = (H // steps_per_block) * steps_per_block
    if m == 0:
        return p.max(axis=1)
    blocks = p[:, :m].reshape(n, -1, steps_per_block).mean(axis=2)
    return blocks.max(axis=1)


def marginal_check(paths: np.ndarray, ladder: np.ndarray, levels=LEVELS) -> dict:
    """Do the sampled paths still have the calibrated marginals?

    The copula is only allowed to add dependence. If it has moved the per-step
    quantiles then it has quietly undone Track A, and every coverage number in
    the audit would now be describing a forecast the controller no longer uses.
    Reported as the largest absolute error, in kW, at each reported level.
    """
    p = np.asarray(paths, float)
    lad = np.asarray(ladder, float)
    out = {}
    for i, q in enumerate(levels):
        emp = np.quantile(p, q, axis=0)
        out[f"q{int(q*100):02d}_max_abs_err_kw"] = float(np.max(np.abs(emp - lad[:, i])))
    return out


def reduce_scenarios(
    paths: np.ndarray, k: int, weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(representative paths, weights)``. See :func:`select_scenarios`."""
    idx, w = select_scenarios(paths, k, weights)
    return np.asarray(paths, float)[idx], w


def select_scenarios(
    paths: np.ndarray, k: int, weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast forward selection: pick ``k`` representative paths and reweight.

    Dupacova, Growe-Kuska and Romisch. Greedily choose the path that most
    reduces the transportation distance to the discarded ones, then give each
    survivor the probability mass of everything nearest to it.

    Why bother. The scenario MILP carries one binary per scenario, so 200
    scenarios is 200 binaries and a branch-and-bound tree that does not close
    inside a control step. Sampling 200 and reducing to 20 is a strictly better
    use of the same budget than sampling 20: the reduction keeps the *shape* of
    the tail, which is the part the ceiling constraint is about, whereas 20 raw
    draws frequently contain no bad afternoon at all.

    The distance is Euclidean over the path, which weights every step equally.
    That is the right choice here because the constraint is applied at every
    step; a peak-only distance would collapse scenarios that differ in when the
    peak lands, and when it lands is what the schedule has to respond to.
    """
    p = np.asarray(paths, float)
    S = p.shape[0]
    if k >= S:
        w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights, float)
        return np.arange(S), w / w.sum()
    w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights, float) / np.sum(weights)

    # Gram-trick distances rather than an (S, S, H) broadcast: this runs inside
    # every control step, and materialising S**2 * H floats there is the
    # difference between a 20-minute month and a 3-hour one.
    sq = (p * p).sum(axis=1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (p @ p.T), 0.0))

    chosen: list[int] = []
    remaining = list(range(S))
    # first pick: the path minimising the weighted distance to all others
    j = int(np.argmin((w[None, :] * d).sum(axis=1)))
    chosen.append(j)
    remaining.remove(j)
    best = d[:, j].copy()
    while len(chosen) < k:
        # for each candidate, the distance every remaining path would have to
        # its nearest kept scenario if this candidate were added -- one matvec,
        # not a Python loop over candidates
        rem = np.array(remaining)
        m = np.minimum(best[rem][:, None], d[np.ix_(rem, rem)])
        red = (w[rem][:, None] * m).sum(axis=0)
        j = int(rem[int(np.argmin(red))])
        chosen.append(j)
        remaining.remove(j)
        best = np.minimum(best, d[:, j])

    chosen_arr = np.array(sorted(chosen))
    nearest = chosen_arr[np.argmin(d[:, chosen_arr], axis=1)]
    new_w = np.array([w[nearest == c].sum() for c in chosen_arr])
    return chosen_arr, new_w / new_w.sum()


DF_GRID: tuple[float | None, ...] = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, None)


def fit_tail_df(
    frame: pd.DataFrame, corr: np.ndarray, horizon: int = 64, levels=LEVELS,
    grid=DF_GRID, horizons=(8, 16, 32, 64), n_paths: int = 300,
    n_origins: int = 300, seed: int = 0,
) -> tuple[float | None, list[dict]]:
    """Choose the copula family by the statistic it will be used for.

    Everything else about this fit is a moment condition on the *body* of the
    distribution -- correlations of normal scores, which barely move between a
    Gaussian and a t. The quantity the ceiling constraint depends on lives in
    the joint upper tail, so that is what the one free parameter is fitted to:
    grid-search the degrees of freedom to match the realised probability that
    the q95 bound breaks somewhere in a window of H steps.

    This is calibration to the decision-relevant statistic, and it is fitted on
    the validation block, never on the evaluation month. A likelihood fit would
    be the textbook route and would spend all its resolution on the body, where
    the two families agree and nothing downstream cares.
    """
    qcols = [f"q{int(q*100):02d}" for q in levels]
    piv = _pivot(frame, ["actual"] + qcols, horizon)
    y = piv["actual"]
    lad = np.stack([piv[c] for c in qcols], axis=-1)
    ok = ~np.isnan(y).any(axis=1) & ~np.isnan(lad).any(axis=(1, 2))
    y, lad = y[ok], lad[ok]
    q95 = lad[:, :, -1]

    rng = np.random.default_rng(seed)
    take = rng.choice(len(y), size=min(n_origins, len(y)), replace=False)
    target = {H: float(np.mean((y[:, :H] > q95[:, :H]).any(axis=1))) for H in horizons}

    trace = []
    for df in grid:
        m = CopulaModel(corr=corr, horizon=horizon, levels=tuple(levels), df=df)
        r = np.random.default_rng(seed + 1)
        pred = {}
        for H in horizons:
            pred[H] = float(np.mean([
                path_exceedance(m.sample(lad[i, :H, :], n_paths, r), q95[i, :H]) for i in take]))
        gap = float(np.mean([abs(pred[H] - target[H]) for H in horizons]))
        trace.append({"df": df, "mean_abs_gap": gap,
                      "predicted": {str(H): pred[H] for H in horizons},
                      "empirical": {str(H): target[H] for H in horizons}})

    best = min(trace, key=lambda r: r["mean_abs_gap"])
    return best["df"], trace


# ---------------------------------------------------------------------------
# fitting against this repo's artefacts
# ---------------------------------------------------------------------------

def fit_from_models(
    building: str, models: "str | Path", cache: "str | Path",
    valid_start: str = "2017-04-01", valid_end: str = "2017-05-31 23:45",
    horizon: int = 64, shrink: float = 0.05, fit_df: bool = True,
) -> CopulaModel:
    """Fit the base-load copula on the validation block, never on the test month.

    The trainer only persists the *test* forecast tensor, so the calibration
    predictions are regenerated here by running the frozen boosters forward over
    April and May. Fitting the dependence structure on June would be leakage of
    exactly the kind the temporal split exists to prevent -- and a subtle kind,
    because the marginals would still look honest while the joint statistics
    quietly knew the answer.
    """
    from pathlib import Path

    from forecast.predict import QuantileModels

    cache, models = Path(cache), Path(models)
    df = pd.read_parquet(cache / f"{building}.parquet")
    lead = pd.Timestamp(valid_start) - pd.Timedelta(days=10)
    tensor = QuantileModels(models / building).predict_tensor(
        df.loc[lead:valid_end], window_start=valid_start)
    m = fit_copula(tensor, horizon=horizon, shrink=shrink)
    if fit_df:
        best, trace = fit_tail_df(tensor, m.corr, horizon=horizon)
        m = CopulaModel(corr=m.corr, horizon=horizon, levels=m.levels, df=best,
                        meta={**m.meta, "df": best, "df_trace": trace,
                              "df_fitted_on": [valid_start, valid_end]})
    return m


def load_or_fit(
    building: str, models: "str | Path", cache: "str | Path", refit: bool = False, **kw,
) -> CopulaModel:
    """Cached fit at ``models/<building>/copula``. Cheap, but it needs a full
    inference pass over two months, and the studies below call it repeatedly."""
    from pathlib import Path

    path = Path(models) / building / "copula"
    if not refit and path.with_suffix(".json").exists():
        return CopulaModel.load(path)
    m = fit_from_models(building, models, cache, **kw)
    m.save(path)
    return m
