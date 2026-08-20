"""Tariff object definition.

A tariff is data, not code. Swapping states means swapping this JSON, and every
downstream module (bill engine, optimizer objective, UI) reads it from here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _hhmm_to_minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass(frozen=True)
class ToDWindow:
    """One time-of-day slice of the day.

    ``multiplier`` is applied to the base energy rate. Windows may wrap past
    midnight (start > end), which is how most night/off-peak slabs are written.
    """

    name: str
    start: str
    end: str
    multiplier: float

    def covers(self, minute_of_day: int) -> bool:
        a, b = _hhmm_to_minutes(self.start), _hhmm_to_minutes(self.end)
        if a == b:
            return True  # degenerate 24h window
        if a < b:
            return a <= minute_of_day < b
        return minute_of_day >= a or minute_of_day < b  # wraps midnight


@dataclass(frozen=True)
class PowerFactorRule:
    """Reactive-power adjustment on the *total demand+energy* charge.

    Below ``target`` the consumer pays ``penalty_pct_per_point`` percent for each
    0.01 of shortfall; above ``rebate_threshold`` they earn
    ``rebate_pct_per_point`` per 0.01, capped by ``max_rebate_pct``.
    """

    target: float = 0.90
    penalty_pct_per_point: float = 1.0
    rebate_threshold: float = 0.95
    rebate_pct_per_point: float = 0.5
    max_rebate_pct: float = 5.0


@dataclass(frozen=True)
class Tariff:
    state: str
    category: str
    order_ref: str
    energy_rate: float                 # INR per kWh, base (multiplier 1.0)
    tod_windows: tuple[ToDWindow, ...]
    demand_charge_per_kva: float       # INR per kVA of billed demand, per month
    billing_interval_minutes: int      # demand averaging block: 15 or 30
    contract_demand_kva: float
    power_factor: PowerFactorRule = field(default_factory=PowerFactorRule)
    billing_demand_floor_pct: float = 0.0   # % of contract demand billed as floor
    fixed_charge: float = 0.0               # flat INR per month
    electricity_duty_pct: float = 0.0       # % on energy charge
    currency: str = "INR"
    notes: str = ""

    # ---- lookup helpers -------------------------------------------------
    def window_for(self, minute_of_day: int) -> ToDWindow:
        for w in self.tod_windows:
            if w.covers(minute_of_day):
                return w
        raise ValueError(f"no ToD window covers minute {minute_of_day}; tariff is not a partition of the day")

    def rate_for(self, minute_of_day: int) -> float:
        return self.energy_rate * self.window_for(minute_of_day).multiplier

    def validate(self) -> None:
        """A tariff must partition the day, or the bill is ambiguous."""
        if self.billing_interval_minutes not in (15, 30, 60):
            raise ValueError("billing_interval_minutes must be 15, 30 or 60")
        for minute in range(0, 1440):
            hits = [w.name for w in self.tod_windows if w.covers(minute)]
            if len(hits) != 1:
                hh, mm = divmod(minute, 60)
                raise ValueError(
                    f"ToD windows must partition the day: {hh:02d}:{mm:02d} matched {hits or 'nothing'}"
                )
        if self.energy_rate <= 0 or self.demand_charge_per_kva < 0:
            raise ValueError("rates must be non-negative and energy_rate positive")

    # ---- io -------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tariff":
        pf = d.get("power_factor") or {}
        t = cls(
            state=d["state"],
            category=d["category"],
            order_ref=d["order_ref"],
            energy_rate=float(d["energy_rate"]),
            tod_windows=tuple(ToDWindow(**w) for w in d["tod_windows"]),
            demand_charge_per_kva=float(d["demand_charge_per_kva"]),
            billing_interval_minutes=int(d["billing_interval_minutes"]),
            contract_demand_kva=float(d["contract_demand_kva"]),
            power_factor=PowerFactorRule(**pf) if pf else PowerFactorRule(),
            billing_demand_floor_pct=float(d.get("billing_demand_floor_pct", 0.0)),
            fixed_charge=float(d.get("fixed_charge", 0.0)),
            electricity_duty_pct=float(d.get("electricity_duty_pct", 0.0)),
            currency=d.get("currency", "INR"),
            notes=d.get("notes", ""),
        )
        t.validate()
        return t

    @classmethod
    def load(cls, path: str | Path) -> "Tariff":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
