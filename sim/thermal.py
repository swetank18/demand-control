"""Closed-loop building simulator: RC envelope, water heater tank, EV, battery.

Why a simulator at all: a static CSV cannot test a controller, because the moment
the controller acts the recorded data is wrong. So the meter gives us the
uncontrollable base load and the weather, and physics gives us everything the
controller can move.

Every equation here is at 15-minute resolution and is deliberately simple enough
to read in one sitting. Where a number is assumed rather than measured it is
named in ``BuildingParams`` and echoed into the results manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

DT_H = 0.25          # simulation timestep, hours
STEPS_PER_DAY = 96


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Comfort:
    """The operator's comfort budget. This is an input, not something we choose."""

    t_min: float = 22.0
    t_max: float = 26.0
    occupied_days: tuple[int, ...] = (0, 1, 2, 3, 4)
    occupied_start_h: float = 8.0
    occupied_end_h: float = 20.0
    unoccupied_t_max: float = 30.0     # band widens when nobody is in
    unoccupied_t_min: float = 18.0

    def band(self, ts: pd.Timestamp) -> tuple[float, float]:
        occupied = ts.dayofweek in self.occupied_days and self.occupied_start_h <= (
            ts.hour + ts.minute / 60.0
        ) < self.occupied_end_h
        if occupied:
            return self.t_min, self.t_max
        return self.unoccupied_t_min, self.unoccupied_t_max

    def band_series(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        lo, hi = zip(*[self.band(ts) for ts in index])
        return pd.DataFrame({"t_lo": lo, "t_hi": hi}, index=index)


@dataclass(frozen=True)
class WaterHeater:
    """Deferrable thermal load with its own tank. On/off, so it is a binary in the MILP."""

    power_kw: float = 30.0
    tank_kwh_per_k: float = 4.0        # thermal capacity of the tank
    ua_kw_per_k: float = 0.05          # standing loss
    t_min: float = 55.0
    t_max: float = 70.0
    t_ambient: float = 25.0
    efficiency: float = 0.98
    daily_draw_kwh: float = 120.0      # hot water actually used per day


@dataclass(frozen=True)
class EVFleet:
    """Aggregate charger. Energy must be delivered by departure, rate is free."""

    max_kw: float = 60.0
    required_kwh_per_day: float = 220.0
    arrive_h: float = 9.0
    depart_h: float = 18.0
    active_days: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 200.0
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    efficiency: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_init: float = 0.50


@dataclass(frozen=True)
class PV:
    """Synthetic rooftop PV. There is no solar meter at this site, so this is a
    clear-sky model attenuated by measured cloud cover -- a design scenario, not
    a measurement. Set kwp = 0 to remove it entirely."""

    kwp: float = 0.0
    latitude: float = 33.42
    temp_coeff_per_k: float = -0.004
    system_efficiency: float = 0.80


@dataclass(frozen=True)
class BuildingParams:
    id: str
    label: str
    sqm: float
    r_k_per_kw: float
    c_kwh_per_k: float
    cop: float
    hvac_capacity_kw: float
    contract_demand_kva: float
    power_factor: float = 0.95
    internal_gain_fraction: float = 0.85   # share of base-load electricity that ends up as heat
    occupancy_w_per_m2: float = 5.0        # sensible gain from people when occupied
    unoccupied_occupancy_fraction: float = 0.15
    solar_gain_m2: float = 0.0             # effective solar aperture (SHGC x glazed area)
    comfort: Comfort = field(default_factory=Comfort)
    water_heater: WaterHeater | None = None
    ev: EVFleet | None = None
    battery: Battery | None = None
    pv: PV = field(default_factory=PV)
    ua_source: str = ""

    @property
    def ua_kw_per_k(self) -> float:
        return 1.0 / self.r_k_per_kw

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, building_id: str, **overrides) -> "BuildingParams":
        m = json.loads(Path(manifest_path).read_text())
        p = m["buildings"][building_id]
        th = p["thermal"]
        base = cls(
            id=building_id,
            label=p.get("label", building_id),
            sqm=p["sqm"],
            r_k_per_kw=th["r_k_per_kw"],
            c_kwh_per_k=th["c_kwh_per_k"],
            cop=th["cop"],
            hvac_capacity_kw=p["hvac_capacity_kw"],
            contract_demand_kva=p["contract_demand_kva"],
            ua_source=th.get("ua_source", ""),
            # Deliberately zero. UA was fitted against outdoor temperature, and
            # temperature and irradiance are strongly correlated, so the fitted
            # envelope conductance already absorbs the solar-driven cooling load.
            # Adding an explicit aperture on top would double-count it. The term is
            # kept so a site with a measured UA can switch it on.
            solar_gain_m2=0.0,
        )
        return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# exogenous signals
# ---------------------------------------------------------------------------

def clear_sky_ghi(index: pd.DatetimeIndex, latitude: float) -> pd.Series:
    """Very standard clear-sky beam+diffuse proxy, W/m2. Good enough for a PV
    envelope; we are not doing solar resource assessment."""
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    lat = np.radians(latitude)
    hour_angle = np.radians(15.0 * (hour - 12.0))
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cos_z = np.clip(cos_z, 0.0, None)
    ghi = 1100.0 * cos_z ** 1.15
    return pd.Series(ghi, index=index, name="ghi")


def pv_output_kw(index: pd.DatetimeIndex, cloud: pd.Series, t_out: pd.Series, pv: PV) -> pd.Series:
    if pv.kwp <= 0:
        return pd.Series(0.0, index=index, name="pv_kw")
    ghi = clear_sky_ghi(index, pv.latitude)
    # BDG2 cloudCoverage is in oktas (0-9); the 0.75 coefficient is the usual
    # linear cloud attenuation used for GHI derating
    clear_frac = 1.0 - 0.75 * (cloud.reindex(index).fillna(0.0).clip(0, 8) / 8.0) ** 3
    t_cell = t_out.reindex(index).ffill() + 0.03 * ghi
    derate = 1.0 + pv.temp_coeff_per_k * (t_cell - 25.0)
    out = pv.kwp * (ghi / 1000.0) * clear_frac * derate * pv.system_efficiency
    return out.clip(lower=0.0).rename("pv_kw")


def solar_gain_kw(index: pd.DatetimeIndex, cloud: pd.Series, params: BuildingParams) -> pd.Series:
    """Thermal solar gain through the envelope, kW."""
    ghi = clear_sky_ghi(index, params.pv.latitude)
    clear_frac = 1.0 - 0.75 * (cloud.reindex(index).fillna(0.0).clip(0, 8) / 8.0) ** 3
    return (ghi / 1000.0 * clear_frac * params.solar_gain_m2).rename("solar_gain_kw")


def occupancy_gain_kw(index: pd.DatetimeIndex, params: BuildingParams) -> pd.Series:
    """People are a heat source: 5 W/m2 sensible when occupied is the conventional
    office figure (roughly 0.05 persons/m2 at 100 W each)."""
    occupied = np.array(
        [params.comfort.band(ts) == (params.comfort.t_min, params.comfort.t_max) for ts in index]
    )
    frac = np.where(occupied, 1.0, params.unoccupied_occupancy_fraction)
    return pd.Series(frac * params.occupancy_w_per_m2 * params.sqm / 1000.0,
                     index=index, name="occ_gain_kw")


# ---------------------------------------------------------------------------
# actions and state
# ---------------------------------------------------------------------------

@dataclass
class Action:
    hvac_kw: float = 0.0          # electrical
    wh_on: int = 0
    ev_kw: float = 0.0
    batt_charge_kw: float = 0.0
    batt_discharge_kw: float = 0.0


@dataclass
class State:
    t_indoor: float
    t_tank: float
    ev_delivered_kwh: float
    soc: float


class BuildingSim:
    """The plant. The controller never sees inside this object except through
    ``observe()``, which is what makes the closed-loop results meaningful."""

    def __init__(
        self,
        params: BuildingParams,
        exog: pd.DataFrame,
        t_indoor_init: float = 24.0,
        seed: int = 0,
    ):
        required = {"base_kw", "t_out", "cloud"}
        missing = required - set(exog.columns)
        if missing:
            raise ValueError(f"exog missing columns: {missing}")
        self.p = params
        self.index = exog.index
        self.exog = exog.copy()
        self.exog["pv_kw"] = pv_output_kw(self.index, exog["cloud"], exog["t_out"], params.pv)
        self.exog["solar_gain_kw"] = solar_gain_kw(self.index, exog["cloud"], params)
        self.exog["occ_gain_kw"] = occupancy_gain_kw(self.index, params)
        self.rng = np.random.default_rng(seed)
        self.t_indoor_init = t_indoor_init
        self.reset()

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        wh = self.p.water_heater
        b = self.p.battery
        self.k = 0
        self.state = State(
            t_indoor=self.t_indoor_init,
            t_tank=(wh.t_max - 2.0) if wh else 0.0,
            ev_delivered_kwh=0.0,
            soc=b.soc_init if b else 0.0,
        )
        self.log: list[dict] = []

    @property
    def now(self) -> pd.Timestamp:
        return self.index[self.k]

    @property
    def done(self) -> bool:
        return self.k >= len(self.index)

    # -- physics ----------------------------------------------------------
    def internal_gain_kw(self, k: int) -> float:
        row = self.exog.iloc[k]
        return (
            self.p.internal_gain_fraction * float(row.base_kw)
            + float(row.solar_gain_kw)
            + float(row.occ_gain_kw)
        )

    def observe(self) -> dict:
        row = self.exog.iloc[self.k]
        lo, hi = self.p.comfort.band(self.now)
        return {
            "t": self.now,
            "k": self.k,
            "t_indoor": self.state.t_indoor,
            "t_out": float(row.t_out),
            "base_kw": float(row.base_kw),
            "pv_kw": float(row.pv_kw),
            "t_tank": self.state.t_tank,
            "ev_delivered_kwh": self.state.ev_delivered_kwh,
            "soc": self.state.soc,
            "t_lo": lo,
            "t_hi": hi,
        }

    def _ev_window(self, ts: pd.Timestamp) -> bool:
        ev = self.p.ev
        if ev is None:
            return False
        h = ts.hour + ts.minute / 60.0
        return ts.dayofweek in ev.active_days and ev.arrive_h <= h < ev.depart_h

    def step(self, a: Action) -> dict:
        """Advance one 15-minute step. Actions are clipped to what the equipment
        can physically do -- a controller that asks for the impossible gets the
        possible, and the violation shows up in the log."""
        p, s = self.p, self.state
        row = self.exog.iloc[self.k]
        ts = self.now

        hvac = float(np.clip(a.hvac_kw, 0.0, p.hvac_capacity_kw))

        # --- envelope ----------------------------------------------------
        q_int = self.internal_gain_kw(self.k)
        q_cool_thermal = hvac * p.cop
        dT = (DT_H / p.c_kwh_per_k) * (
            (float(row.t_out) - s.t_indoor) * p.ua_kw_per_k + q_int - q_cool_thermal
        )
        t_next = s.t_indoor + dT

        # --- water heater -------------------------------------------------
        wh_kw = 0.0
        if p.water_heater is not None:
            wh = p.water_heater
            wh_kw = wh.power_kw * (1 if a.wh_on else 0)
            # draw is spread over the occupied day
            draw_kw = wh.daily_draw_kwh / 12.0 if 7 <= ts.hour < 19 else 0.0
            dT_tank = (DT_H / wh.tank_kwh_per_k) * (
                wh_kw * wh.efficiency - wh.ua_kw_per_k * (s.t_tank - wh.t_ambient) - draw_kw
            )
            s.t_tank = float(np.clip(s.t_tank + dT_tank, 5.0, 95.0))

        # --- EV -----------------------------------------------------------
        ev_kw = 0.0
        if p.ev is not None:
            if self._ev_window(ts):
                ev_kw = float(np.clip(a.ev_kw, 0.0, p.ev.max_kw))
                s.ev_delivered_kwh += ev_kw * DT_H
            elif ts.hour == p.ev.depart_h and ts.minute == 0:
                s.ev_delivered_kwh = 0.0   # fleet leaves, counter resets

        # --- battery --------------------------------------------------------
        ch = dis = 0.0
        if p.battery is not None:
            b = p.battery
            ch = float(np.clip(a.batt_charge_kw, 0.0, b.max_charge_kw))
            dis = float(np.clip(a.batt_discharge_kw, 0.0, b.max_discharge_kw))
            e = s.soc * b.capacity_kwh + (ch * b.efficiency - dis / b.efficiency) * DT_H
            e = float(np.clip(e, b.soc_min * b.capacity_kwh, b.soc_max * b.capacity_kwh))
            s.soc = e / b.capacity_kwh

        grid = float(row.base_kw) + hvac + wh_kw + ev_kw + ch - dis - float(row.pv_kw)
        grid = max(grid, 0.0)

        lo, hi = p.comfort.band(ts)
        rec = {
            "t": ts, "t_indoor": s.t_indoor, "t_indoor_next": t_next, "t_out": float(row.t_out),
            "base_kw": float(row.base_kw), "hvac_kw": hvac, "wh_kw": wh_kw, "ev_kw": ev_kw,
            "batt_charge_kw": ch, "batt_discharge_kw": dis, "pv_kw": float(row.pv_kw),
            "grid_kw": grid, "t_lo": lo, "t_hi": hi,
            "comfort_violation_k": max(0.0, s.t_indoor - hi) + max(0.0, lo - s.t_indoor),
            "t_tank": s.t_tank, "soc": s.soc, "ev_delivered_kwh": s.ev_delivered_kwh,
        }
        self.log.append(rec)

        s.t_indoor = float(np.clip(t_next, -10.0, 60.0))
        self.k += 1
        return rec

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.log).set_index("t")
