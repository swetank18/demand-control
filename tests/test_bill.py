"""Acceptance test for the bill engine: a week computed by hand, to the rupee.

The expected numbers below are written out longhand on purpose. If you change
the tariff JSON these break, and they are supposed to.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tariff.bill import compute_bill, demand_blocks, marginal_demand_cost, price_series
from tariff.schema import Tariff, ToDWindow

TARIFF = Tariff.load(Path(__file__).resolve().parents[1] / "tariff/orders/tnerc_2026.json")


def _week(kw: float = 100.0) -> pd.Series:
    idx = pd.date_range("2025-06-02", periods=96 * 7, freq="15min")  # Mon 00:00, 7 days
    return pd.Series(kw, index=idx)


def test_hand_computed_week_to_the_rupee():
    """100 kW flat for 7 days on TNERC HT-I-A.

    Hand arithmetic:
      windows/day: solar 05-10 = 5h, normal 10-18 = 8h, peak 18-22 = 4h, night 22-05 = 7h  (= 24h)
      kWh/day:     500,  800,  400,  700          -> x7 = 3500, 5600, 2800, 4900   (16,800 kWh)
      rates:       6.35*0.80 = 5.0800   6.35*1.00 = 6.3500   6.35*1.25 = 7.9375   6.35*0.90 = 5.7150
      energy:      3500*5.0800 =  17,780.00
                   5600*6.3500 =  35,560.00
                   2800*7.9375 =  22,225.00
                   4900*5.7150 =  28,003.50   -> 103,568.50
      demand:      every 30-min block averages 100 kW; at pf 0.95 -> 105.2631... kVA
                   floor = 90% of 250 kVA contract = 225 kVA, which binds
                   225 * 608 = 136,800.00
      pf 0.95:     inside [target 0.90, rebate threshold 0.95] -> 0.00
      duty:        5% of energy = 5,178.425
      TOTAL        103,568.50 + 136,800.00 + 0 + 5,178.425 = 245,546.925
    """
    bill = compute_bill(_week(100.0), TARIFF, power_factor=0.95)

    assert bill.energy_kwh == pytest.approx(16_800.0, abs=1e-6)
    assert bill.energy_by_window["solar"]["charge"] == pytest.approx(17_780.00, abs=0.01)
    assert bill.energy_by_window["normal"]["charge"] == pytest.approx(35_560.00, abs=0.01)
    assert bill.energy_by_window["peak"]["charge"] == pytest.approx(22_225.00, abs=0.01)
    assert bill.energy_by_window["night"]["charge"] == pytest.approx(28_003.50, abs=0.01)
    assert bill.energy_charge == pytest.approx(103_568.50, abs=0.01)

    assert bill.peak_demand_kw == pytest.approx(100.0, abs=1e-9)
    assert bill.peak_demand_kva == pytest.approx(100.0 / 0.95, abs=1e-9)
    assert bill.billed_demand_kva == pytest.approx(225.0, abs=1e-9)   # floor binds
    assert bill.demand_charge == pytest.approx(136_800.00, abs=0.01)

    assert bill.pf_adjustment == pytest.approx(0.0, abs=1e-9)
    assert bill.electricity_duty == pytest.approx(5_178.425, abs=0.01)
    assert bill.total == pytest.approx(245_546.925, abs=0.01)


def test_one_bad_tuesday_sets_the_month():
    """The emotional hook, as an assertion.

    A single 30-minute block pushed to 400 kW on one Tuesday afternoon costs the
    marginal demand charge for the whole month, and almost nothing in energy.
    """
    base = _week(200.0)
    spiked = base.copy()
    tue = pd.Timestamp("2025-06-03 14:00")
    spiked.loc[tue : tue + pd.Timedelta(minutes=45)] = 400.0   # two 15-min steps -> one 30-min block

    b0 = compute_bill(base, TARIFF, 0.95)
    b1 = compute_bill(spiked, TARIFF, 0.95)

    extra_kwh = b1.energy_kwh - b0.energy_kwh
    assert extra_kwh == pytest.approx(200.0, abs=1e-6)          # 200 kW extra for 1.0 h
    extra_energy_cost = b1.energy_charge - b0.energy_charge
    assert extra_energy_cost == pytest.approx(200.0 * 6.35, abs=0.01)   # normal window

    # Peak kVA goes 200/0.95 = 210.5 -> 400/0.95 = 421.1, but the 225 kVA billing
    # floor binds in the base case, so only the part above the floor is new money.
    floor = TARIFF.contract_demand_kva * TARIFF.billing_demand_floor_pct / 100.0
    expected_demand_delta = (max(400.0 / 0.95, floor) - max(200.0 / 0.95, floor)) * TARIFF.demand_charge_per_kva
    assert expected_demand_delta == pytest.approx(119_200.0, abs=0.01)
    assert (b1.demand_charge - b0.demand_charge) == pytest.approx(expected_demand_delta, abs=0.01)
    assert marginal_demand_cost(b0.peak_demand_kva, b1.peak_demand_kva, TARIFF) == pytest.approx(
        expected_demand_delta, abs=0.01
    )

    # The punchline, and the number that goes on slide 2: one extra hour of load
    # costs INR 1,270 in energy and INR 119,200 in demand charge -- a factor of 94.
    ratio = (b1.demand_charge - b0.demand_charge) / extra_energy_cost
    assert ratio == pytest.approx(93.86, abs=0.05)
    assert ratio > 90


def test_demand_is_block_average_not_instantaneous():
    """A spike narrower than the billing block is diluted. This is exactly the
    headroom the controller trades against comfort, so it must be modelled."""
    s = _week(200.0)
    s.loc[pd.Timestamp("2025-06-03 14:00")] = 400.0   # one 15-min step inside a 30-min block
    b = compute_bill(s, TARIFF, 0.95)
    assert b.peak_demand_kw == pytest.approx(300.0, abs=1e-9)  # (400+200)/2


def test_power_factor_penalty_and_rebate():
    w = _week(300.0)
    good = compute_bill(w, TARIFF, power_factor=0.99)
    bad = compute_bill(w, TARIFF, power_factor=0.85)
    assert good.pf_adjustment < 0 and bad.pf_adjustment > 0
    # 0.85 is 5 points below the 0.90 target -> 5% penalty on energy+demand
    base = bad.energy_charge + bad.demand_charge
    assert bad.pf_adjustment == pytest.approx(base * 0.05, rel=1e-9)
    # 0.99 is 4 points above the 0.95 threshold -> 2% rebate
    gbase = good.energy_charge + good.demand_charge
    assert good.pf_adjustment == pytest.approx(-gbase * 0.02, rel=1e-9)


def test_export_is_not_paid_for():
    s = _week(100.0)
    s.iloc[10:20] = -50.0
    b = compute_bill(s, TARIFF, 0.95)
    assert b.energy_kwh == pytest.approx(16_800.0 - 10 * 0.25 * 100.0, abs=1e-6)


def test_tariff_must_partition_the_day():
    with pytest.raises(ValueError):
        Tariff.from_dict(
            {
                "state": "X", "category": "y", "order_ref": "z", "energy_rate": 5.0,
                "tod_windows": [{"name": "a", "start": "00:00", "end": "12:00", "multiplier": 1.0}],
                "demand_charge_per_kva": 100.0, "billing_interval_minutes": 30,
                "contract_demand_kva": 100.0,
            }
        )


def test_second_state_loads_and_shifts_the_incentive():
    """Portability claim, as a test: same engine, different JSON, different answer."""
    tn = TARIFF
    mh = Tariff.load(Path(__file__).resolve().parents[1] / "tariff/orders/msedcl_2026.json")
    noon = 12 * 60
    assert tn.window_for(noon).name == "normal"
    assert mh.window_for(noon).name == "solar"      # midday is cheap in MH, not in TN
    w = _week(150.0)
    assert compute_bill(w, tn, 0.95).total != compute_bill(w, mh, 0.95).total
