"""Baseline controllers. Three of the five rows in the results table come from
here; the other two are the MILP with different forecasts fed into it.

Keeping the strawmen honest matters. "No control" is a real deadband thermostat
with immediate EV charging, which is what an actual building does today -- not a
deliberately stupid agent that exists to lose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sim.thermal import DT_H, Action, BuildingSim



def hold_power_kw(sim: BuildingSim, obs: dict, target_c: float) -> float:
    """Electrical HVAC power that lands the indoor temperature on ``target_c`` at
    the end of this step.

    Inverting the RC update rather than using a proportional gain is deliberate:
    a plain P controller parks at a steady-state offset above setpoint and books
    phantom comfort violations, which would flatter every controller that comes
    after it in the table. Real commissioned VAV loops do not behave that way.
    Assumes the local loop can estimate its own internal gain, which is what a
    building management system trend log gives you.
    """
    p = sim.p
    q_int = sim.internal_gain_kw(min(sim.k, len(sim.index) - 1))
    t_in = obs["t_indoor"]
    envelope = (obs["t_out"] - t_in) * p.ua_kw_per_k
    # required thermal extraction so that T_next == target
    q_needed = envelope + q_int - (target_c - t_in) * p.c_kwh_per_k / DT_H
    return float(np.clip(q_needed / p.cop, 0.0, p.hvac_capacity_kw))


class Controller:
    name = "base"

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def act(self, sim: BuildingSim, obs: dict) -> Action:
        raise NotImplementedError


# ---------------------------------------------------------------------------

class NoControl(Controller):
    """Status quo. A deadband thermostat aiming at the middle of the comfort band,
    the water heater kept hot at all times, and EVs charging flat out the moment
    they plug in. Price-blind, peak-blind."""

    name = "No control"

    def __init__(self, setpoint_c: float = 24.0):
        self.setpoint_c = setpoint_c

    def act(self, sim: BuildingSim, obs: dict) -> Action:
        p = sim.p
        # one fixed setpoint all year, which is what an uninstrumented building runs
        target = min(max(self.setpoint_c, obs["t_lo"]), obs["t_hi"])
        hvac = hold_power_kw(sim, obs, target)

        wh_on = 0
        if p.water_heater is not None:
            wh = p.water_heater
            wh_on = int(obs["t_tank"] < wh.t_max - 1.0)

        ev_kw = 0.0
        if p.ev is not None and sim._ev_window(obs["t"]):
            remaining = p.ev.required_kwh_per_day - obs["ev_delivered_kwh"]
            ev_kw = float(np.clip(remaining / DT_H, 0.0, p.ev.max_kw))

        return Action(hvac_kw=hvac, wh_on=wh_on, ev_kw=ev_kw)


class RuleBased(Controller):
    """What a good facilities engineer does without any optimisation: pre-cool
    ahead of the expensive window, coast through it, and push deferrable load
    into the cheap window. Tariff-aware, forecast-blind, risk-blind."""

    name = "Rule based schedule"

    def __init__(self, tariff, precool_h: float = 3.0):
        self.tariff = tariff
        self.precool_h = precool_h
        self._peak_starts = [
            w.start for w in tariff.tod_windows if w.multiplier == max(x.multiplier for x in tariff.tod_windows)
        ]

    def _minutes_to_peak(self, ts: pd.Timestamp) -> float:
        now = ts.hour * 60 + ts.minute
        best = 24 * 60
        for s in self._peak_starts:
            h, m = s.split(":")
            start = int(h) * 60 + int(m)
            d = (start - now) % (24 * 60)
            best = min(best, d)
        return best

    def act(self, sim: BuildingSim, obs: dict) -> Action:
        p = sim.p
        ts = obs["t"]
        window = self.tariff.window_for(ts.hour * 60 + ts.minute).name
        is_peak = window == "peak"
        to_peak_h = self._minutes_to_peak(ts) / 60.0

        # pre-cool towards the bottom of the band, coast towards the top in peak
        if is_peak:
            target = obs["t_hi"] - 0.2
        elif to_peak_h <= self.precool_h:
            target = obs["t_lo"] + 0.2
        else:
            target = 0.5 * (obs["t_lo"] + obs["t_hi"])

        target = min(max(target, obs["t_lo"]), obs["t_hi"])
        hvac = hold_power_kw(sim, obs, target)
        if is_peak:
            hvac = min(hvac, 0.6 * p.hvac_capacity_kw)   # hard throttle in the peak window

        wh_on = 0
        if p.water_heater is not None:
            wh = p.water_heater
            # only heat outside peak unless the tank is genuinely cold
            wh_on = int(obs["t_tank"] < (wh.t_min + 2.0)) if is_peak else int(obs["t_tank"] < wh.t_max - 1.0)

        ev_kw = 0.0
        if p.ev is not None and sim._ev_window(ts):
            remaining = p.ev.required_kwh_per_day - obs["ev_delivered_kwh"]
            hours_left = max(sim.p.ev.depart_h - (ts.hour + ts.minute / 60.0), DT_H)
            if is_peak:
                rate = remaining / max(hours_left, 1e-6) * 0.3    # trickle through peak
            else:
                rate = remaining / max(hours_left, 1e-6) * 1.5    # get ahead while it is cheap
            ev_kw = float(np.clip(rate, 0.0, p.ev.max_kw))

        return Action(hvac_kw=hvac, wh_on=wh_on, ev_kw=ev_kw)
