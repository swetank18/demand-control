"""The simulator is the plant. If its physics is wrong every result is wrong,
so the RC update is checked against a closed-form solution, not just eyeballed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.thermal import (DT_H, Action, BuildingParams, BuildingSim, Comfort,
                         WaterHeater, clear_sky_ghi)
from sim.stress import GridOutage, Heatwave


def _params(**kw):
    base = dict(
        id="T", label="test", sqm=5000.0, r_k_per_kw=1 / 10.0, c_kwh_per_k=200.0,
        cop=3.0, hvac_capacity_kw=150.0, contract_demand_kva=300.0,
        # gains switched off so the envelope can be checked against the analytic
        # solution in isolation
        internal_gain_fraction=0.0, occupancy_w_per_m2=0.0,
    )
    base.update(kw)
    return BuildingParams(**base)


def _exog(n=96, t_out=35.0, base=0.0):
    idx = pd.date_range("2017-06-01", periods=n, freq="15min")
    return pd.DataFrame({"base_kw": base, "t_out": t_out, "cloud": 0.0}, index=idx)


def test_free_float_matches_exponential_decay():
    """With no gains and no cooling, indoor temperature must relax towards outdoor
    with time constant tau = R*C. Checked against the analytic solution."""
    p = _params()
    tau = p.r_k_per_kw * p.c_kwh_per_k     # 20 hours
    sim = BuildingSim(p, _exog(n=96 * 3, t_out=35.0), t_indoor_init=20.0)
    while not sim.done:
        sim.step(Action(hvac_kw=0.0))
    h = sim.history()
    t_hours = np.arange(len(h)) * DT_H
    analytic = 35.0 - (35.0 - 20.0) * np.exp(-t_hours / tau)
    # discrete Euler lags the exact solution slightly; 0.15 K is well inside that
    assert np.max(np.abs(h["t_indoor"].to_numpy() - analytic)) < 0.15


def test_cooling_removes_exactly_cop_times_electrical():
    """Steady state: the HVAC power that holds temperature must satisfy
    hvac * COP == UA * (T_out - T_in). Any other answer means the energy balance
    is wrong and every kWh downstream is wrong with it."""
    p = _params()
    sim = BuildingSim(p, _exog(n=96 * 4, t_out=35.0), t_indoor_init=25.0)
    hold = p.ua_kw_per_k * (35.0 - 25.0) / p.cop
    while not sim.done:
        sim.step(Action(hvac_kw=hold))
    h = sim.history()
    assert h["t_indoor"].std() < 1e-6
    assert h["t_indoor"].iloc[-1] == pytest.approx(25.0, abs=1e-6)


def test_actions_are_clipped_not_trusted():
    p = _params()
    sim = BuildingSim(p, _exog(n=8))
    rec = sim.step(Action(hvac_kw=10_000.0))
    assert rec["hvac_kw"] == pytest.approx(p.hvac_capacity_kw)


def test_grid_never_negative_and_pv_offsets():
    p = _params(internal_gain_fraction=0.0)
    sim = BuildingSim(p, _exog(n=96, base=50.0))
    while not sim.done:
        sim.step(Action(hvac_kw=0.0))
    h = sim.history()
    assert (h["grid_kw"] >= 0).all()
    assert h["grid_kw"].max() <= 50.0 + 1e-9


def test_water_heater_tank_loses_heat_and_recovers():
    p = _params(water_heater=WaterHeater(daily_draw_kwh=0.0))
    sim = BuildingSim(p, _exog(n=96))
    for _ in range(20):
        sim.step(Action(wh_on=0))
    cooled = sim.state.t_tank
    for _ in range(20):
        sim.step(Action(wh_on=1))
    assert sim.state.t_tank > cooled


def test_comfort_band_widens_when_unoccupied():
    c = Comfort()
    occupied = pd.Timestamp("2017-06-01 10:00")     # Thursday
    night = pd.Timestamp("2017-06-01 03:00")
    weekend = pd.Timestamp("2017-06-03 10:00")      # Saturday
    assert c.band(occupied) == (c.t_min, c.t_max)
    assert c.band(night) == (c.unoccupied_t_min, c.unoccupied_t_max)
    assert c.band(weekend) == (c.unoccupied_t_min, c.unoccupied_t_max)


def test_clear_sky_is_zero_at_night_and_peaks_at_noon():
    idx = pd.date_range("2017-06-21", periods=96, freq="15min")
    g = clear_sky_ghi(idx, 33.42)
    assert g.between_time("00:00", "03:00").max() == 0.0
    assert 900 < g.max() < 1200
    assert 11 <= g.idxmax().hour <= 13


def test_heatwave_raises_temperature_and_load():
    e = _exog(n=96 * 6, t_out=30.0, base=100.0)
    e.index = pd.date_range("2017-06-14", periods=len(e), freq="15min")
    out, rep = Heatwave(start="2017-06-15 00:00", days=2).apply(e)
    assert rep["applied"]
    win = out.loc["2017-06-15 12:00":"2017-06-15 18:00"]
    assert win["t_out"].max() > 34.0
    assert win["base_kw"].max() > 110.0
    # outside the window nothing moved
    assert out.loc["2017-06-14 12:00", "t_out"] == pytest.approx(30.0)


def test_outage_zeroes_grid_availability_and_sheds_load():
    e = _exog(n=96 * 3, base=100.0)
    e.index = pd.date_range("2017-06-22", periods=len(e), freq="15min")
    out, rep = GridOutage(start="2017-06-22 14:00", hours=2.0).apply(e)
    assert rep["applied"]
    win = out.loc["2017-06-22 14:00":"2017-06-22 15:45"]
    assert (win["grid_available"] == 0).all()
    assert win["base_kw"].max() <= 100.0 * 0.35 + 1e-9
