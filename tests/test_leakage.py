"""The leakage checks, written before any of the round-2 training was run.

Section 6.1 of the plan: a random split on time-series data leaks the future
into training through neighbouring timestamps, and a reviewer who spots it
discards the whole evaluation. These tests make that failure impossible to
commit rather than merely unlikely.

The load-bearing one is ``test_no_feature_reads_the_future``. It does not
inspect the feature list and hope; it rewrites the future of the series and
asserts that not one feature value at an earlier origin moves. Add a feature
that peeks and this test fails, whatever the feature is called.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.features import FEATURE_COLS, build_supervised
from forecast.splits import ROLLING_FOLDS, SPLIT

ROOT = Path(__file__).resolve().parents[1]

#: The one declared exception to origin-time causality: weather at the target
#: time is treated as a perfect forecast. Stated in the model card and in the
#: limitations, and stressable via ``--weather-noise-c``.
WEATHER_AT_TARGET = {"t_out_fut", "cdd_fut", "cloud_fut"}

#: ``horizon`` is a feature but also the second index level in these tests, so
#: it is compared implicitly by aligning on the index rather than as a column.
COMPARED = [c for c in FEATURE_COLS if c != "horizon"]


def _series(days: int = 40, seed: int = 0) -> pd.DataFrame:
    """A synthetic building: daily shape, weekly shape, weather response, noise.

    Synthetic on purpose. These tests assert a property of the feature builder,
    not of any building, and they must run in a clone with no data downloaded.
    """
    idx = pd.date_range("2016-01-04", periods=days * 96, freq="15min")
    rng = np.random.default_rng(seed)
    hod = idx.hour + idx.minute / 60.0
    t_out = 22 + 8 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 0.4, len(idx))
    load = (
        120
        + 60 * np.exp(-0.5 * ((hod - 14) / 3.5) ** 2)
        - 25 * (idx.dayofweek >= 5)
        + 2.5 * np.maximum(0.0, t_out - 18.0)
        + rng.normal(0, 3.0, len(idx))
    )
    return pd.DataFrame(
        {"base_kw": load, "t_out": t_out, "cloud": rng.uniform(0, 0.6, len(idx))}, index=idx
    )


# --- the split itself ------------------------------------------------------

def test_split_blocks_are_ordered_and_disjoint():
    b = [
        (SPLIT.train_start, SPLIT.train_end),
        (SPLIT.valid_start, SPLIT.valid_end),
        (SPLIT.test_start, SPLIT.test_end),
    ]
    for lo, hi in b:
        assert pd.Timestamp(lo) < pd.Timestamp(hi)
    for (_, prev_hi), (next_lo, _) in zip(b, b[1:]):
        assert pd.Timestamp(prev_hi) < pd.Timestamp(next_lo), "blocks overlap in time"


def test_split_masks_partition_without_overlap():
    times = pd.Series(pd.date_range(SPLIT.train_start, SPLIT.test_end, freq="1h"))
    masks = {p: SPLIT.mask(times, p).to_numpy() for p in ("train", "valid", "test")}
    stacked = np.vstack(list(masks.values()))
    assert stacked.sum(axis=0).max() <= 1, "a timestamp belongs to two splits"
    assert stacked.any(axis=1).all(), "a split selected nothing"


def test_rolling_folds_expand_and_never_train_on_their_own_validation():
    for train_end, valid_end in ROLLING_FOLDS:
        assert pd.Timestamp(train_end) < pd.Timestamp(valid_end)
        assert pd.Timestamp(valid_end) <= pd.Timestamp(SPLIT.valid_end), (
            "a rolling fold validates inside the untouched test window"
        )
    ends = [pd.Timestamp(t) for t, _ in ROLLING_FOLDS]
    assert ends == sorted(ends) and len(set(ends)) == len(ends), "folds do not expand"


# --- the feature builder ---------------------------------------------------

def test_no_feature_reads_the_future():
    """Rewrite the future of the load series; no earlier feature may move."""
    df = _series()
    cut = df.index[len(df) // 2]

    tampered = df.copy()
    tampered.loc[tampered.index > cut, "base_kw"] *= 3.0

    a = build_supervised(df, horizon_steps=8).set_index(["origin", "horizon"]).sort_index()
    b = build_supervised(tampered, horizon_steps=8).set_index(["origin", "horizon"]).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    past = a.index.get_level_values("origin") <= cut
    assert past.sum() > 1000, "not enough pre-cut rows to make this test meaningful"
    np.testing.assert_allclose(
        a.loc[past, COMPARED].to_numpy(),
        b.loc[past, COMPARED].to_numpy(),
        rtol=0,
        atol=0,
        err_msg="a feature at an origin before the cut moved when the future changed",
    )

    # and the test is not vacuous: after the cut, features must move
    future = ~past
    assert not np.allclose(
        a.loc[future, COMPARED].to_numpy(), b.loc[future, COMPARED].to_numpy()
    ), "tampering changed nothing anywhere, so this test proves nothing"


def test_weather_is_the_only_declared_look_ahead():
    """Perturbing future weather may move the weather-at-target block and nothing else."""
    df = _series()
    cut = df.index[len(df) // 2]
    tampered = df.copy()
    tampered.loc[tampered.index > cut, "t_out"] += 10.0

    a = build_supervised(df, horizon_steps=8).set_index(["origin", "horizon"]).sort_index()
    b = build_supervised(tampered, horizon_steps=8).set_index(["origin", "horizon"]).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    past = a.index.get_level_values("origin") <= cut

    moved = {
        c for c in COMPARED
        if not np.allclose(a.loc[past, c].to_numpy(), b.loc[past, c].to_numpy())
    }
    assert moved <= WEATHER_AT_TARGET, (
        f"features other than the declared weather forecast read future weather: {moved - WEATHER_AT_TARGET}"
    )
    assert moved, "future weather changed nothing, so the weather feature is not wired up"


def test_target_is_the_value_h_steps_after_the_origin():
    """Guards the other direction: features honest, target misaligned by one step."""
    df = _series(days=20)
    sup = build_supervised(df, horizon_steps=4)
    want = df["base_kw"].reindex(
        pd.DatetimeIndex(sup["origin"]) + pd.to_timedelta(sup["horizon"] * 15, unit="m")
    ).to_numpy()
    np.testing.assert_allclose(sup["y"].to_numpy(), want, rtol=0, atol=1e-9)


def test_lag_one_equals_the_previous_observation():
    df = _series(days=20)
    sup = build_supervised(df, horizon_steps=2)
    origins = pd.DatetimeIndex(sup["origin"])
    np.testing.assert_allclose(
        sup["lag_1"].to_numpy(),
        df["base_kw"].reindex(origins - pd.Timedelta(minutes=15)).to_numpy(),
        rtol=0, atol=1e-9,
    )
    np.testing.assert_allclose(
        sup["last"].to_numpy(), df["base_kw"].reindex(origins).to_numpy(), rtol=0, atol=1e-9
    )


# --- what actually got trained --------------------------------------------

@pytest.mark.parametrize("d", sorted((ROOT / "models").glob("*/meta.json")) if (ROOT / "models").exists() else [])
def test_trained_artefacts_used_the_frozen_split(d: Path):
    """A model in ``models/`` that was trained on different dates is a stale
    artefact quoting itself into the deck. Fail rather than let it."""
    meta = json.loads(d.read_text())
    s = meta.get("splits", {})
    if not s:
        pytest.skip(f"{d.parent.name} carries no split record")
    assert s["train_end"] == SPLIT.train_end
    assert s["valid_end"] == SPLIT.valid_end
    assert s["test_start"] == SPLIT.test_start
    assert s["test_end"] == SPLIT.test_end
    assert pd.Timestamp(s["valid_end"]) < pd.Timestamp(s["test_start"])
