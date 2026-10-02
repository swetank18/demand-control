"""The paired contrast between risk levels, checked where the answer is known.

Two levels with identical outcomes must differ by exactly zero with a zero-width
interval. A level that adds hits on top of another's must show exactly that
difference, and the matched-pair interval must be narrower than the two
unpaired ones it replaces -- that is the only reason to pair. Origins one level
skipped (a failed solve) must be dropped from the pairing, not misaligned. And
the sign-flip test that decides "resolved" must give the p-value worked out by
hand, including the thin case that made a bootstrap overconfident.
"""
from __future__ import annotations

import numpy as np

from eval.horizon_risk import paired_contrasts, signflip_p

N_DAYS, PER_DAY = 30, 12


def _row(eps, flags, target="tight", origins=None):
    origins = list(range(len(flags))) if origins is None else origins
    days = [f"2017-06-{1 + k // PER_DAY:02d}" for k in origins]
    rate = float(np.mean(flags))
    return {"target": target, "epsilon": eps, "commit_flags": list(map(int, flags)),
            "commit_origins": origins, "commit_days": days,
            "commit_violation_rate": rate, "commit_violation_ci": [rate - 0.1, rate + 0.1]}


def _base(p=0.1, seed=0):
    rng = np.random.default_rng(seed)
    day = rng.random(N_DAYS) < p * 2            # hits cluster by day
    return np.repeat(day, PER_DAY) & (rng.random(N_DAYS * PER_DAY) < 0.5)


def test_identical_levels_differ_by_exactly_zero():
    f = _base()
    out = paired_contrasts([_row(0.05, f), _row(0.10, f)])
    (c,) = out
    assert c["kind"] == "adjacent" and c["diff"] == 0.0
    assert c["diff_ci"] == [0.0, 0.0] and c["p_signflip"] == 1.0 and not c["resolved"]


def test_added_hits_show_up_exactly_and_pairing_narrows_the_interval():
    a = _base()
    rng = np.random.default_rng(5)
    b = a | (rng.random(a.size) < 0.15)         # b breaches wherever a does, and more
    out = paired_contrasts([_row(0.05, a), _row(0.35, b)])
    (c,) = out
    assert abs(c["diff"] - (b.mean() - a.mean())) < 1e-12
    assert c["resolved"] and c["diff_ci"][0] > 0 and c["p_signflip"] < 0.001
    assert c["n_down"] == 0 and c["n_up"] == int((b & ~a).sum())

    # what laying the two unpaired day-block intervals side by side would give
    from forecast.conformal import block_bootstrap_means
    days = np.repeat(np.arange(N_DAYS), PER_DAY)
    hw = [np.subtract(*np.quantile(block_bootstrap_means(x.astype(float), days, seed=0),
                                   [0.975, 0.025])) / 2 for x in (a, b)]
    assert (c["diff_ci"][1] - c["diff_ci"][0]) / 2 < max(hw)


def test_origins_one_level_skipped_are_dropped_not_misaligned():
    f = np.zeros(N_DAYS * PER_DAY, int)
    f[::7] = 1
    keep = [k for k in range(f.size) if k != 3]   # level a failed to solve at origin 3
    a = _row(0.05, f[keep], origins=keep)
    b = _row(0.10, f)
    (c,) = paired_contrasts([a, b])
    assert c["n_pairs"] == f.size - 1 and c["diff"] == 0.0


def test_every_kind_is_reported_per_target():
    f = _base()
    rows = []
    for t in ("tight", "nominal"):
        rows += [_row(None, f, t)] + [_row(e, f, t) for e in (0.05, 0.10, 0.20, 0.35)]
    out = paired_contrasts(rows)
    for t in ("tight", "nominal"):
        kinds = [c["kind"] for c in out if c["target"] == t]
        assert kinds.count("adjacent") == 3
        assert kinds.count("ends") == 1 and kinds.count("marginal_vs_tightest") == 1
    ends = [c for c in out if c["kind"] == "ends"][0]
    assert (ends["from"], ends["to"]) == (0.05, 0.35)


def test_signflip_p_matches_the_hand_count():
    # five days each one breach up, nothing down: of 2^5 sign assignments only
    # all-plus and all-minus reach |5|, so p = 2/32 -- not resolved at 5%,
    # although every bootstrap resample containing one of those days is positive
    assert signflip_p(np.array([1, 1, 1, 1, 1, 0, 0])) == 2 / 32
    # six days: 2/64, resolved
    assert signflip_p(np.array([1, 1, 1, 1, 1, 1])) == 2 / 64
    # a day against the direction counts against it: |2+1+1+1-1| = 4. Over
    # magnitudes {2,1,1,1,1}, totals of +-6 take one assignment each and +-4
    # take four each (the 2 positive, one of the four 1s negative), so 10 of 32
    assert abs(signflip_p(np.array([2, 1, 1, 1, -1])) - 10 / 32) < 1e-12
    assert signflip_p(np.zeros(4)) == 1.0


def test_thin_contrast_is_not_called_resolved_on_the_bootstrap_alone():
    # five extra breaches, one per day on five of thirty days: the bootstrap
    # interval excludes zero, the exact test does not, and the exact test decides
    a = np.zeros(N_DAYS * PER_DAY, bool)
    b = a.copy()
    b[[PER_DAY * d for d in (2, 9, 15, 21, 27)]] = True
    (c,) = paired_contrasts([_row(0.05, a), _row(0.10, b)])
    assert c["p_signflip"] == 2 / 32 and not c["resolved"]
