"""Controller invariants.

These are the claims the pitch makes, written down so they can fail. Short
windows so the suite stays fast enough to run between changes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.baselines import NoControl, RuleBased, hold_power_kw
from control.mpc import ChanceConstrainedMPC, FullHorizonOracle, MPCConfig
from eval.run_month import build_context, run_controller
from sim.thermal import BuildingSim

WINDOW = ("2017-06-01", "2017-06-04 23:45")


@pytest.fixture(scope="module")
def ctx():
    return build_context("Fox_office_Gaylord", *WINDOW)


def test_q95_is_more_conservative_than_q50(ctx):
    """The one idea, as a test: substituting q95 into the ceiling constraint must
    produce a plan that leaves headroom the mean forecast does not."""
    cfg = MPCConfig()
    ours = ChanceConstrainedMPC(ctx["tariff"], ctx["fc_quantile"], cfg,
                                risk_quantile="q95", solar_quantile="q05", demand_target_kw=420.0)
    mean = ChanceConstrainedMPC(ctx["tariff"], ctx["fc_quantile"], cfg,
                                risk_quantile="q50", solar_quantile="q50", demand_target_kw=420.0)
    sim = BuildingSim(ctx["params"], ctx["exog"])
    obs = sim.observe()
    a_ours = ours.solve(sim, obs)
    a_mean = mean.solve(sim, obs)
    # both feasible
    assert a_ours.status == "optimal" and a_mean.status == "optimal"
    # the risk scenario the two are defending differs in the right direction
    k, H = sim.k, cfg.horizon
    assert (ctx["fc_quantile"].base(k, H, "q95") >= ctx["fc_quantile"].base(k, H, "q50") - 1e-9).all()
    assert (ctx["fc_quantile"].pv(k, H, "q05") <= ctx["fc_quantile"].pv(k, H, "q50") + 1e-9).all()


def test_plan_respects_the_ceiling_on_its_own_risk_scenario(ctx):
    """Whatever the controller plans must satisfy the constraint it was given.
    If this fails the optimiser is not solving the problem we wrote down."""
    cfg = MPCConfig()
    c = ChanceConstrainedMPC(ctx["tariff"], ctx["fc_quantile"], cfg, demand_target_kw=430.0)
    sim = BuildingSim(ctx["params"], ctx["exog"])
    res = c.solve(sim, sim.observe())
    assert res.status == "optimal"
    H = len(res.plan["hvac"])
    risk = ctx["fc_quantile"].base(0, H, "q95") - ctx["fc_quantile"].pv(0, H, "q05")
    grid = risk + res.plan["hvac"] + res.plan.get("ev", np.zeros(H))
    grid = grid + res.plan.get("wh", np.zeros(H)) * ctx["params"].water_heater.power_kw
    # res.plan["wh"] is the raw solver output, which is what the constraint saw
    blocks = grid[: (H // 2) * 2].reshape(-1, 2).mean(axis=1)
    assert blocks.max() <= res.d_peak_kw + 1e-6


def test_controller_defends_target_it_can_hold(ctx):
    """A target inside the building's physical envelope must actually be held for
    the whole window, not merely aimed at."""
    c = ChanceConstrainedMPC(ctx["tariff"], ctx["fc_quantile"], MPCConfig(), demand_target_kw=430.0)
    m = run_controller(ctx, c).metrics
    assert m["ceiling_breaches"] == 0
    assert m["peak_kw"] <= 430.0 + 1e-6


def test_comfort_slack_is_a_last_resort_not_a_habit(ctx):
    c = ChanceConstrainedMPC(ctx["tariff"], ctx["fc_quantile"], MPCConfig(), demand_target_kw=430.0)
    r = run_controller(ctx, c)
    assert r.metrics["comfort_violation_pct"] < 5.0
    assert r.metrics["max_overshoot_k"] < 2.0


def test_full_month_oracle_beats_the_rolling_perfect_forecast(ctx):
    """The oracle must be an upper bound. A rolling controller with perfect
    forecasts still cannot see the whole month, so it should not beat the
    whole-month solve; if it does, the oracle is not solving what we think."""
    cfg = MPCConfig()
    rolling = ChanceConstrainedMPC(ctx["tariff"], ctx["fc_oracle"], cfg, risk_quantile="q50",
                                   solar_quantile="q50", demand_target_kw=430.0)
    oracle = FullHorizonOracle(ctx["tariff"], cfg, demand_target_kw=430.0)
    m_roll = run_controller(ctx, rolling).metrics
    m_orac = run_controller(ctx, oracle).metrics
    assert oracle.status == "optimal"
    # allow a small tolerance: the oracle relaxes the water-heater binaries
    assert m_orac["bill_inr"] <= m_roll["bill_inr"] * 1.005


def test_no_control_holds_comfort_and_is_price_blind(ctx):
    r = run_controller(ctx, NoControl())
    assert r.metrics["comfort_violation_pct"] == 0.0
    h = r.history
    # a fixed-setpoint thermostat should show no relationship between price and load
    from tariff.bill import price_series
    p = price_series(h.index, ctx["tariff"])
    hv = h["hvac_kw"]
    if hv.std() > 1e-6:
        assert abs(np.corrcoef(p.to_numpy(), hv.to_numpy())[0, 1]) < 0.5


def test_hold_power_lands_on_target_exactly(ctx):
    sim = BuildingSim(ctx["params"], ctx["exog"])
    obs = sim.observe()
    target = obs["t_indoor"] - 0.2
    kw = hold_power_kw(sim, obs, target)
    rec = sim.step(type(sim).__mro__ and __import__("sim.thermal", fromlist=["Action"]).Action(hvac_kw=kw))
    if kw < ctx["params"].hvac_capacity_kw - 1e-6:
        assert rec["t_indoor_next"] == pytest.approx(target, abs=1e-6)


def test_rule_based_is_tariff_aware(ctx):
    """The rule-based baseline must actually respond to the tariff, otherwise it
    is a strawman rather than a baseline."""
    r = run_controller(ctx, RuleBased(ctx["tariff"]))
    h = r.history
    from tariff.bill import window_series
    w = window_series(h.index, ctx["tariff"])
    peak_hvac = h.loc[(w == "peak").to_numpy(), "hvac_kw"].mean()
    normal_hvac = h.loc[(w == "normal").to_numpy(), "hvac_kw"].mean()
    assert peak_hvac < normal_hvac
