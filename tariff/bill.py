"""Deterministic bill engine. The single source of truth for every rupee.

Nothing else in this repo is allowed to compute money. The optimizer's objective
is an approximation of this engine; the evaluation table, the UI counter and the
slides all call ``compute_bill``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import Tariff


@dataclass
class Bill:
    """A month's bill, decomposed the way the utility decomposes it."""

    energy_kwh: float
    energy_charge: float
    energy_by_window: dict[str, dict[str, float]]   # window -> {kwh, charge, rate}
    peak_demand_kva: float
    peak_demand_kw: float
    peak_demand_at: pd.Timestamp | None
    billed_demand_kva: float
    demand_charge: float
    power_factor: float
    pf_adjustment: float          # negative = rebate, positive = penalty
    electricity_duty: float
    fixed_charge: float
    total: float
    currency: str = "INR"
    meta: dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "energy_kwh": self.energy_kwh,
                "energy_charge": self.energy_charge,
                "peak_demand_kw": self.peak_demand_kw,
                "peak_demand_kva": self.peak_demand_kva,
                "billed_demand_kva": self.billed_demand_kva,
                "demand_charge": self.demand_charge,
                "pf_adjustment": self.pf_adjustment,
                "electricity_duty": self.electricity_duty,
                "fixed_charge": self.fixed_charge,
                "total": self.total,
            }
        )

    def explain(self) -> str:
        lines = [
            f"Bill  [{self.currency}]",
            f"  Energy      {self.energy_kwh:12,.1f} kWh    {self.energy_charge:14,.2f}",
        ]
        for name, d in self.energy_by_window.items():
            lines.append(
                f"    {name:<8}  {d['kwh']:10,.1f} kWh @ {d['rate']:.4f}  {d['charge']:12,.2f}"
            )
        at = self.peak_demand_at.strftime("%Y-%m-%d %H:%M") if self.peak_demand_at is not None else "n/a"
        lines += [
            f"  Peak demand {self.peak_demand_kw:12,.2f} kW  ({at})",
            f"  Billed kVA  {self.billed_demand_kva:12,.2f}          {self.demand_charge:14,.2f}",
            f"  PF {self.power_factor:.3f} adjustment                 {self.pf_adjustment:14,.2f}",
            f"  Electricity duty                        {self.electricity_duty:14,.2f}",
            f"  Fixed charge                            {self.fixed_charge:14,.2f}",
            f"  {'TOTAL':<38}{self.total:14,.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# price vectors (also consumed by the optimizer)
# ---------------------------------------------------------------------------

def price_series(index: pd.DatetimeIndex, tariff: Tariff) -> pd.Series:
    """INR/kWh at each timestamp."""
    minutes = index.hour * 60 + index.minute
    return pd.Series([tariff.rate_for(int(m)) for m in minutes], index=index, name="price")


def window_series(index: pd.DatetimeIndex, tariff: Tariff) -> pd.Series:
    minutes = index.hour * 60 + index.minute
    return pd.Series([tariff.window_for(int(m)).name for m in minutes], index=index, name="window")


def _infer_dt_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("need at least two timestamps to infer the interval")
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    dt = float(np.median(deltas))
    if not np.allclose(deltas, dt, rtol=0, atol=1.0):
        raise ValueError("power series must be on a regular grid")
    return dt / 3600.0


def demand_blocks(power_kw: pd.Series, tariff: Tariff) -> pd.Series:
    """Average kW over each billing block. Demand charges are levied on the
    *average* over the block, not the instantaneous peak -- that distinction is
    worth real money and is the reason a 15-minute spike can be ridden out."""
    rule = f"{tariff.billing_interval_minutes}min"
    return power_kw.resample(rule, label="left", closed="left").mean().dropna()


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

def compute_bill(
    power_kw: pd.Series,
    tariff: Tariff,
    power_factor: float | pd.Series = 0.95,
    apply_demand_floor: bool = True,
) -> Bill:
    """Bill a 15-minute (or finer) grid-import series over a billing month.

    ``power_kw`` is *grid import* in kW, indexed by timestamp. Negative values
    (export) are clipped to zero for energy billing: net metering is out of
    scope and pretending otherwise would flatter the result.
    """
    if not isinstance(power_kw.index, pd.DatetimeIndex):
        raise TypeError("power_kw must be indexed by a DatetimeIndex")
    power_kw = power_kw.sort_index().astype(float)
    dt_h = _infer_dt_hours(power_kw.index)
    imp = power_kw.clip(lower=0.0)

    # --- energy charge, split by ToD window --------------------------------
    price = price_series(power_kw.index, tariff)
    window = window_series(power_kw.index, tariff)
    kwh = imp * dt_h
    charge = kwh * price

    energy_by_window: dict[str, dict[str, float]] = {}
    for w in tariff.tod_windows:
        m = (window == w.name).to_numpy()
        energy_by_window[w.name] = {
            "kwh": float(kwh[m].sum()),
            "charge": float(charge[m].sum()),
            "rate": tariff.energy_rate * w.multiplier,
            "multiplier": w.multiplier,
        }
    energy_kwh = float(kwh.sum())
    energy_charge = float(charge.sum())

    # --- demand charge -----------------------------------------------------
    blocks = demand_blocks(imp, tariff)
    pf_scalar = float(np.mean(power_factor)) if not np.isscalar(power_factor) else float(power_factor)
    pf_scalar = min(max(pf_scalar, 0.5), 1.0)
    kva_blocks = blocks / pf_scalar
    if len(kva_blocks):
        peak_kva = float(kva_blocks.max())
        peak_kw = float(blocks.max())
        peak_at = kva_blocks.idxmax()
    else:
        peak_kva = peak_kw = 0.0
        peak_at = None

    floor = tariff.contract_demand_kva * tariff.billing_demand_floor_pct / 100.0 if apply_demand_floor else 0.0
    billed_kva = max(peak_kva, floor)
    demand_charge = billed_kva * tariff.demand_charge_per_kva

    # --- power factor adjustment on (energy + demand) ----------------------
    pfr = tariff.power_factor
    base_for_pf = energy_charge + demand_charge
    if pf_scalar < pfr.target:
        points = round((pfr.target - pf_scalar) * 100, 6)
        pf_adjustment = base_for_pf * (pfr.penalty_pct_per_point / 100.0) * points
    elif pf_scalar > pfr.rebate_threshold:
        points = round((pf_scalar - pfr.rebate_threshold) * 100, 6)
        pct = min(pfr.rebate_pct_per_point * points, pfr.max_rebate_pct)
        pf_adjustment = -base_for_pf * pct / 100.0
    else:
        pf_adjustment = 0.0

    duty = energy_charge * tariff.electricity_duty_pct / 100.0
    total = energy_charge + demand_charge + pf_adjustment + duty + tariff.fixed_charge

    return Bill(
        energy_kwh=energy_kwh,
        energy_charge=energy_charge,
        energy_by_window=energy_by_window,
        peak_demand_kva=peak_kva,
        peak_demand_kw=peak_kw,
        peak_demand_at=peak_at,
        billed_demand_kva=billed_kva,
        demand_charge=demand_charge,
        power_factor=pf_scalar,
        pf_adjustment=float(pf_adjustment),
        electricity_duty=float(duty),
        fixed_charge=tariff.fixed_charge,
        total=float(total),
        currency=tariff.currency,
        meta={
            "interval_hours": dt_h,
            "n_intervals": int(len(power_kw)),
            "demand_floor_kva": floor,
            "start": str(power_kw.index[0]),
            "end": str(power_kw.index[-1]),
            "order_ref": tariff.order_ref,
        },
    )


def marginal_demand_cost(peak_kva_before: float, peak_kva_after: float, tariff: Tariff) -> float:
    """What one more kVA of monthly peak actually costs. Used in the pitch."""
    floor = tariff.contract_demand_kva * tariff.billing_demand_floor_pct / 100.0
    b = max(peak_kva_before, floor)
    a = max(peak_kva_after, floor)
    return (a - b) * tariff.demand_charge_per_kva
