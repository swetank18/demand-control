"""The copula pseudo-likelihood behind Table copula-df, checked where the
answer is known: scores drawn from a t-copula must prefer the t, scores drawn
from a Gaussian must prefer the Gaussian. If this failed, the finding that the
likelihood criterion picks the Gaussian on real load would be a bug rather
than a result."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import t as student_t

from eval.copula_df_check import copula_loglik

H, N = 16, 4000


def _corr() -> np.ndarray:
    i = np.arange(H)
    return 0.8 ** np.abs(i[:, None] - i[None, :])


def _t_scores(nu: float, rng) -> np.ndarray:
    L = np.linalg.cholesky(_corr())
    z = rng.standard_normal((N, H)) @ L.T
    w = np.sqrt(nu / rng.chisquare(nu, size=(N, 1)))
    u = np.clip(student_t.cdf(z * w, nu), 1e-6, 1 - 1e-6)
    return ndtri(u)


def test_t_scores_prefer_the_t():
    rng = np.random.default_rng(0)
    z = _t_scores(4.0, rng)
    ll = {nu: copula_loglik(z, _corr(), nu) for nu in (3.0, 4.0, 7.0, 25.0, None)}
    assert ll[4.0] > ll[None]
    assert max(ll, key=ll.get) in (3.0, 4.0, 7.0)


def test_gaussian_scores_prefer_the_gaussian():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((N, H)) @ np.linalg.cholesky(_corr()).T
    ll = {nu: copula_loglik(z, _corr(), nu) for nu in (3.0, 7.0, 25.0, None)}
    assert ll[None] > ll[3.0]
    assert ll[25.0] > ll[3.0]
