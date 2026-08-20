"""Stress injection. The demo is not about the model being accurate; it is about
the system behaving when the model is wrong.

Three switches, each one button in the UI, each producing a condition that is
absent from the forecaster's training window by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class Stress:
    name = "none"

    def apply(self, exog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        return exog, {"stress": "none"}


@dataclass
class Heatwave(Stress):
    """Outdoor temperature raised sharply and base load pushed beyond anything in
    the training window. This is the condition under which a mean-forecast
    controller's demand ceiling is wrong in the expensive direction."""

    name: str = "heatwave"
    start: str = "2017-06-15 00:00"
    days: int = 3
    delta_c: float = 6.0
    base_load_multiplier: float = 1.25

    def apply(self, exog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        out = exog.copy()
        t0 = pd.Timestamp(self.start)
        t1 = t0 + pd.Timedelta(days=self.days)
        m = (out.index >= t0) & (out.index < t1)
        if not m.any():
            return out, {"stress": self.name, "applied": False, "reason": "window outside run"}
        # ramp in and out over 6 hours so the step is not physically absurd
        ramp = np.ones(m.sum())
        n_ramp = min(24, len(ramp) // 3)
        if n_ramp > 0:
            ramp[:n_ramp] = np.linspace(0, 1, n_ramp)
            ramp[-n_ramp:] = np.linspace(1, 0, n_ramp)
        out.loc[m, "t_out"] = out.loc[m, "t_out"] + self.delta_c * ramp
        out.loc[m, "base_kw"] = out.loc[m, "base_kw"] * (1 + (self.base_load_multiplier - 1) * ramp)
        out.loc[m, "cloud"] = 0.0
        return out, {
            "stress": self.name, "applied": True,
            "window": [str(t0), str(t1)],
            "delta_c": self.delta_c, "base_load_multiplier": self.base_load_multiplier,
            "peak_t_out_c": float(out.loc[m, "t_out"].max()),
        }


@dataclass
class SensorDropout(Stress):
    """The forecast input goes stale. The controller keeps running on its last
    good estimate, which is what actually happens when a BMS point dies."""

    name: str = "sensor_dropout"
    start: str = "2017-06-20 12:00"
    hours: float = 2.0

    def apply(self, exog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        # The plant is untouched: reality does not stop when the sensor does.
        # The dropout is applied to what the *controller* sees, via
        # ``StaleForecast`` below, so the simulator keeps the true trajectory.
        return exog, {
            "stress": self.name, "applied": True,
            "window": [self.start, str(pd.Timestamp(self.start) + pd.Timedelta(hours=self.hours))],
            "note": "applied to the controller's inputs, not to the plant",
        }


@dataclass
class GridOutage(Stress):
    """Grid import forced to zero. The controller must ride through on storage and
    thermal mass while prioritising critical load. CityLearn 2023 scored agents
    under outages; a controller that only works when the grid is up is not a
    controller a building owner wants."""

    name: str = "outage"
    start: str = "2017-06-22 14:00"
    hours: float = 2.0
    critical_load_fraction: float = 0.35

    def apply(self, exog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        out = exog.copy()
        t0 = pd.Timestamp(self.start)
        t1 = t0 + pd.Timedelta(hours=self.hours)
        m = (out.index >= t0) & (out.index < t1)
        if not m.any():
            return out, {"stress": self.name, "applied": False, "reason": "window outside run"}
        out["grid_available"] = 1.0
        out.loc[m, "grid_available"] = 0.0
        # only critical load survives an outage; the rest is simply not served
        out.loc[m, "base_kw"] = out.loc[m, "base_kw"] * self.critical_load_fraction
        return out, {
            "stress": self.name, "applied": True,
            "window": [str(t0), str(t1)],
            "critical_load_fraction": self.critical_load_fraction,
        }


class StaleForecast:
    """Wraps a forecast source and freezes it during a window.

    Deliberately not a no-op: the controller keeps acting on a forecast it
    believes, which is the realistic failure. A system whose safety depends on
    fresh data is a system that is unsafe on the day the data stops.
    """

    def __init__(self, inner, index: pd.DatetimeIndex, start: str, hours: float):
        self.inner = inner
        self.index = index
        t0 = pd.Timestamp(start)
        t1 = t0 + pd.Timedelta(hours=hours)
        self.mask = (index >= t0) & (index < t1)
        pos = np.flatnonzero(self.mask)
        self.freeze_at = int(pos[0]) - 1 if len(pos) and pos[0] > 0 else 0
        self.name = f"{getattr(inner, 'name', 'forecast')} (stale {t0}..{t1})"

    def _k(self, k0: int) -> int:
        return self.freeze_at if (k0 < len(self.mask) and self.mask[k0]) else k0

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.inner.base(self._k(k0), H, q)

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.inner.pv(self._k(k0), H, q)


STRESSES = {
    "none": Stress,
    "heatwave": Heatwave,
    "sensor_dropout": SensorDropout,
    "outage": GridOutage,
}
