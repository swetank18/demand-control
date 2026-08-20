"""Closed-loop evaluation over a full billing month.

Every controller sees the identical weather, the identical base load and the
identical forecasts. The only thing that varies is the decision rule. Numbers
come out of the bill engine, never out of the optimiser's objective -- the
optimiser is an approximation of the bill, and reporting its own objective back
would be marking our own homework.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.baselines import NoControl, RuleBased
from control.mpc import ChanceConstrainedMPC, FullHorizonOracle, MPCConfig
from forecast.sources import OracleForecast, PVQuantiles, TensorForecast
from sim.thermal import (Action, Battery, BuildingParams, BuildingSim, Comfort,
                         EVFleet, PV, WaterHeater, pv_output_kw)
from tariff.bill import compute_bill, demand_blocks
from tariff.schema import Tariff


@dataclass
class RunResult:
    name: str
    history: pd.DataFrame
    bill: object
    metrics: dict = field(default_factory=dict)


def build_context(
    building: str,
    month_start: str,
    month_end: str,
    tariff_path: Path = ROOT / "tariff/orders/tnerc_2026.json",
    cache: Path = ROOT / "data/cache",
    models: Path = ROOT / "models",
    pv_kwp: float = 150.0,
    with_battery: bool = False,
    comfort: Comfort | None = None,
    stress=None,
) -> dict:
    df = pd.read_parquet(cache / f"{building}.parquet")
    history = df.loc[: month_start].iloc[:-1]
    # Carry enough history before the window for the forecaster's 7-day lags, so
    # a stressed run can be re-forecast rather than read from a tensor that was
    # computed on data the stress never touched.
    ext_start = pd.Timestamp(month_start) - pd.Timedelta(days=10)
    ext = df.loc[ext_start:month_end, ["base_kw", "t_out", "cloud"]].copy()

    tariff = Tariff.load(tariff_path)
    params = BuildingParams.from_manifest(
        cache / "manifest.json", building,
        water_heater=WaterHeater(), ev=EVFleet(), pv=PV(kwp=pv_kwp),
        battery=Battery() if with_battery else None,
        **({"comfort": comfort} if comfort else {}),
    )
    tariff = dataclasses.replace(tariff, contract_demand_kva=params.contract_demand_kva)

    stress_report = {}
    if stress is not None:
        # Stress the extended frame once, then slice: the plant and the
        # forecaster then see one consistent version of reality.
        ext, stress_report = stress.apply(ext)
    exog = ext.loc[month_start:month_end].copy()

    pvq = PVQuantiles(exog.index, exog["cloud"], exog["t_out"], params.pv,
                      history_cloud=history["cloud"], history_index=history.index)
    pv_actual = pv_output_kw(exog.index, exog["cloud"], exog["t_out"], params.pv).to_numpy()

    # A stress that moves base load or weather invalidates the precomputed
    # tensor. Re-running inference on the stressed inputs is what a deployed
    # forecaster would do; leaving it stale would make every controller equally
    # blind and the stress test would measure nothing.
    perturbs_forecast = stress is not None and stress_report.get("applied") and (
        stress.name in ("heatwave", "outage")
    )
    if perturbs_forecast:
        from forecast.predict import QuantileModels

        tensor = QuantileModels(models / building).predict_tensor(ext, window_start=month_start)
    else:
        tensor = models / building / "forecast_test.parquet"
    fc_quantile = TensorForecast(tensor, exog.index, pvq)
    fc_oracle = OracleForecast(exog["base_kw"].to_numpy(), pv_actual)

    return dict(
        building=building, params=params, tariff=tariff, exog=exog,
        fc_quantile=fc_quantile, fc_oracle=fc_oracle, pvq=pvq,
        stress=stress_report, month_start=month_start, month_end=month_end,
    )


def uncontrolled_peak_kw(ctx: dict) -> float:
    """Peak of the status-quo building. The demand target is expressed relative to
    this, so the ask ("shave 15%") is stated in terms the operator recognises."""
    r = run_controller(ctx, NoControl(), label="__probe__")
    return float(r.bill.peak_demand_kw)


def run_controller(ctx: dict, controller, label: str | None = None, seed: int = 0) -> RunResult:
    params, tariff = ctx["params"], ctx["tariff"]
    sim = BuildingSim(params, ctx["exog"], seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()

    block_min = tariff.billing_interval_minutes
    steps_per_block = block_min // 15
    block_buf: list[float] = []
    t0 = time.perf_counter()

    while not sim.done:
        obs = sim.observe()
        if hasattr(controller, "note_block_progress"):
            controller.note_block_progress(block_buf)
        action = controller.act(sim, obs)
        rec = sim.step(action)
        # feed the realised billing block back to the controller: once a peak is
        # set for the month, there is nothing left to protect below it
        block_buf.append(rec["grid_kw"])
        if len(block_buf) == steps_per_block:
            if hasattr(controller, "note_realised"):
                controller.note_realised(float(np.mean(block_buf)))
            block_buf = []

    wall = time.perf_counter() - t0
    h = sim.history()
    bill = compute_bill(h["grid_kw"], tariff, params.power_factor)

    blocks = demand_blocks(h["grid_kw"], tariff)
    target = getattr(controller, "demand_target_kw", None) or ctx.get("demand_target_kw")
    viol = h["comfort_violation_k"]
    occupied = (h["t_hi"] - h["t_lo"]) < 6.0

    metrics = {
        "controller": label or getattr(controller, "name", type(controller).__name__),
        "bill_inr": bill.total,
        "energy_charge": bill.energy_charge,
        "demand_charge": bill.demand_charge,
        "energy_kwh": bill.energy_kwh,
        "peak_kw": bill.peak_demand_kw,
        "peak_kva": bill.peak_demand_kva,
        "billed_kva": bill.billed_demand_kva,
        "peak_at": str(bill.peak_demand_at),
        "comfort_violation_pct": float((viol > 0.1).mean() * 100.0),
        "comfort_violation_pct_occupied": float((viol[occupied] > 0.1).mean() * 100.0) if occupied.any() else 0.0,
        "comfort_kelvin_hours": float(viol.sum() * 0.25),
        "max_overshoot_k": float(viol.max()),
        "load_factor": float(h["grid_kw"].mean() / max(blocks.max(), 1e-9)),
        "demand_target_kw": float(target) if target else np.nan,
        # A "breach" is a billing block whose average exceeded the demand target.
        # This is the safety metric: each breach is a permanent monthly cost.
        "ceiling_breaches": int((blocks > target + 1e-6).sum()) if target else 0,
        "worst_breach_kw": float(max(0.0, blocks.max() - target)) if target else 0.0,
        # When the ceiling first goes is the whole mechanism: the demand charge is
        # a monthly maximum, so the first breach spends the money and every later
        # block below it is then free. Reporting the time makes that visible
        # instead of hiding it inside the total.
        "first_breach_at": (str(blocks[blocks > target + 1e-6].index[0])
                            if target and (blocks > target + 1e-6).any() else None),
        "wall_seconds": wall,
        "ev_shortfall_kwh": float(max(0.0, 0.0)),
    }
    if hasattr(controller, "solve_times") and controller.solve_times:
        st = np.array(controller.solve_times)
        metrics["solve_ms_mean"] = float(st.mean() * 1000)
        metrics["solve_ms_p95"] = float(np.quantile(st, 0.95) * 1000)
        metrics["infeasible_solves"] = int(sum(1 for s in controller.statuses if s != "optimal"))
    return RunResult(metrics["controller"], h, bill, metrics)


def standard_controllers(ctx: dict, cfg: MPCConfig | None = None,
                         demand_target_kw: float | None = None) -> list:
    """The five rows of the results table.

    All three optimising controllers are handed the *same* demand target and the
    same solver settings. The only difference between the mean-forecast row and
    ours is which quantile goes into the ceiling constraint, and the only
    difference between ours and the oracle is whether the forecast is real. That
    makes the table an ablation, not a beauty contest.
    """
    cfg = cfg or MPCConfig()
    tariff = ctx["tariff"]
    tgt = demand_target_kw if demand_target_kw is not None else ctx.get("demand_target_kw")
    return [
        NoControl(),
        RuleBased(tariff),
        ChanceConstrainedMPC(tariff, ctx["fc_quantile"], cfg, risk_quantile="q50",
                             solar_quantile="q50", name="MPC on mean forecast",
                             demand_target_kw=tgt),
        ChanceConstrainedMPC(tariff, ctx["fc_quantile"], cfg, risk_quantile="q95",
                             solar_quantile="q05", name="Ours: quantile + chance constrained",
                             demand_target_kw=tgt),
        ChanceConstrainedMPC(tariff, ctx["fc_oracle"], cfg, risk_quantile="q50",
                             solar_quantile="q50", name="MPC on perfect forecast (16 h)",
                             demand_target_kw=tgt),
        FullHorizonOracle(tariff, cfg, demand_target_kw=tgt),
    ]
