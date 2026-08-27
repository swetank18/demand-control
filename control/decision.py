"""What the forecast is actually worth: the cost of the decision it produced.

The forecaster is trained on pinball loss. Nobody is billed in pinball loss.
The repo's own ablation already measured that the two are related and not
identical -- ₹2,228 per unit of pinball, R² 0.95, so a fifth of the variance in
what the model is worth is *not* explained by the score it was fitted on. This
module makes the other quantity measurable, which is the precondition for
everything in Track C.

**Decision regret.** Fix the plant state. Solve the MPC with forecast f and get
a committed schedule. Solve it again with the true future and get the schedule
you would have committed had you known. Evaluate *both* schedules against what
actually happened, and take the difference. That is regret: the rupees the
forecast error cost, holding everything else constant. It is computable today,
with no differentiation machinery anywhere, and it is Stage 1 of the plan.

**Why the schedules must start from the same state.** An obvious mistake is to
run two closed-loop months and difference the bills. That measures the
compounding of a thousand different states as much as it measures the forecast,
and it cannot attribute anything to a particular interval. Here the state is
held fixed at each origin and only the forecast varies, so the number is the
marginal value of the forecast at that moment -- which is the object Stage 2
needs, because it is defined per interval.

**What the evaluation includes.** The same four terms the MPC optimises: energy
at the tariff's time-of-day rate, the demand charge on the excess above the
target, the comfort penalty, and the small shaping cost on the planned ceiling.
The uncontrollable part of the energy bill is included and simply cancels in the
difference. The envelope is re-simulated under the *true* internal gains, so a
plan built on a forecast that was too low is charged for the comfort violation
it actually causes, not for the one it thought it would cause.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sim.thermal import DT_H, BuildingParams


@dataclass
class DecisionLoss:
    """Realised cost of a committed plan, evaluated against the true future.

    Deliberately a separate implementation from the MILP's objective rather than
    a call back into it. The optimiser's objective is an approximation of the
    bill built for tractability; scoring a plan with the same approximation that
    chose it would be marking our own homework, which is the rule the repo
    already applies to the closed-loop table.
    """

    tariff: object
    params: BuildingParams
    exog: pd.DataFrame          # true base_kw, pv_kw, t_out, occ_gain_kw, solar_gain_kw
    demand_target_kw: float
    comfort_penalty: float = 5_000.0
    peak_shaping_cost: float = 2.0

    def __post_init__(self) -> None:
        p = self.params
        self.index = self.exog.index
        self.ua = p.ua_kw_per_k
        self.cap_c = p.c_kwh_per_k
        self.a = 1.0 - DT_H * self.ua / self.cap_c
        self.b = DT_H * p.cop / self.cap_c
        self.steps_per_block = max(1, self.tariff.billing_interval_minutes // 15)
        self._price = np.array([self.tariff.rate_for(ts.hour * 60 + ts.minute)
                                for ts in self.index])
        band = p.comfort.band_series(self.index)
        self._t_lo = band["t_lo"].to_numpy()
        self._t_hi = band["t_hi"].to_numpy()

    # -- the true trajectory ------------------------------------------------
    def _const(self, sl: slice) -> np.ndarray:
        p = self.params
        e = self.exog
        q_int = (p.internal_gain_fraction * e["base_kw"].to_numpy()[sl]
                 + e["occ_gain_kw"].to_numpy()[sl] + e["solar_gain_kw"].to_numpy()[sl])
        return (DT_H / self.cap_c) * (e["t_out"].to_numpy()[sl] * self.ua + q_int)

    def controllable_kw(self, plan: dict, H: int) -> np.ndarray:
        """Total controllable draw per step implied by a plan."""
        p = self.params
        out = np.asarray(plan["hvac"], float)[:H].copy()
        if "wh" in plan and p.water_heater is not None:
            out = out + np.asarray(plan["wh"], float)[:H] * p.water_heater.power_kw
        if "ev" in plan:
            out = out + np.asarray(plan["ev"], float)[:H]
        if "bc" in plan:
            out = out + np.asarray(plan["bc"], float)[:H]
        if "bd" in plan:
            out = out - np.asarray(plan["bd"], float)[:H]
        return out

    def evaluate(self, k0: int, plan: dict, t_indoor0: float,
                 d_ref: float | None = None) -> dict:
        """Cost of committing ``plan`` at origin ``k0``, under the true future."""
        H = min(len(plan["hvac"]), len(self.index) - k0)
        sl = slice(k0, k0 + H)
        const = self._const(sl)
        hvac = np.asarray(plan["hvac"], float)[:H]

        # envelope replayed under the true internal gains, not the forecast ones
        T = np.empty(H)
        prev = t_indoor0
        for t in range(H):
            prev = self.a * prev - self.b * hvac[t] + const[t]
            T[t] = prev
        slack = (np.maximum(0.0, T - self._t_hi[sl]) + np.maximum(0.0, self._t_lo[sl] - T))

        ctrl = self.controllable_kw(plan, H)
        grid = (self.exog["base_kw"].to_numpy()[sl] - self.exog["pv_kw"].to_numpy()[sl] + ctrl)

        # block averages, on whole blocks only: a partial block at the end of the
        # horizon has not been metered yet and charging for it would penalise
        # plans purely for where the window happened to stop
        m = (H // self.steps_per_block) * self.steps_per_block
        blocks = (grid[:m].reshape(-1, self.steps_per_block).mean(axis=1)
                  if m else np.array([grid.mean()]))
        peak = float(blocks.max())
        ref = self.demand_target_kw if d_ref is None else d_ref
        excess = max(0.0, peak - ref)

        energy = float(np.sum(self._price[sl] * np.maximum(grid, 0.0)) * DT_H)
        demand = excess * self.tariff.demand_charge_per_kva / max(self.params.power_factor, 1e-6)
        comfort = float(self.comfort_penalty * slack.sum())
        return {
            "energy_inr": energy,
            "demand_inr": demand,
            "comfort_inr": comfort,
            "shaping_inr": self.peak_shaping_cost * peak,
            "total_inr": energy + demand + comfort + self.peak_shaping_cost * peak,
            "peak_block_kw": peak,
            "excess_kw": excess,
            "comfort_kelvin_steps": float(slack.sum()),
            "max_overshoot_k": float(np.max(np.maximum(0.0, T - self._t_hi[sl]))) if H else 0.0,
            "H": int(H),
        }


def regret(loss: DecisionLoss, k0: int, plan: dict, oracle_plan: dict,
           t_indoor0: float) -> dict:
    """``cost(plan) - cost(oracle_plan)``, both under the truth, same state.

    Non-negative up to solver tolerance: the oracle solves the same problem with
    the true parameters, so nothing can beat it except by rounding. A negative
    value large enough to notice means the two solves did not start from the
    same state, which is the failure mode this signature is shaped to prevent.
    """
    a = loss.evaluate(k0, plan, t_indoor0)
    b = loss.evaluate(k0, oracle_plan, t_indoor0)
    return {
        "regret_inr": a["total_inr"] - b["total_inr"],
        "regret_energy": a["energy_inr"] - b["energy_inr"],
        "regret_demand": a["demand_inr"] - b["demand_inr"],
        "regret_comfort": a["comfort_inr"] - b["comfort_inr"],
        "cost_inr": a["total_inr"],
        "oracle_cost_inr": b["total_inr"],
        "peak_block_kw": a["peak_block_kw"],
        "oracle_peak_block_kw": b["peak_block_kw"],
    }


def block_sensitivity(
    loss: DecisionLoss, controller, sim, obs, delta_kw: float = 10.0,
) -> dict:
    """How much realised cost moves when the forecast for one block moves.

    A central finite difference per billing block: shift the forecast that block
    sees by ±delta, re-solve, and evaluate both resulting plans against the same
    true future. The result is ₹ per kW of forecast error, per block, which is
    the quantity Stage 2 needs -- "where does forecast error actually change the
    schedule" -- as a number rather than an intuition.

    Per block rather than per step because the ceiling constraint only ever sees
    block averages; perturbing a single 15-minute step moves the constraint by
    half as much and costs twice as many solves for the same information.

    Blocks where the ceiling is slack come back at zero, and that is the finding,
    not a defect: forecast error at 03:00 on a Sunday changes no decision and is
    worth nothing to get right. That is the asymmetry pinball loss is blind to,
    because pinball weights every interval the same.
    """
    from control.mpc import ChanceConstrainedMPC

    k0 = sim.k
    base = controller.solve(sim, obs)
    if not base.plan:
        return {"ok": False}
    H = len(base.plan["hvac"])
    blocks = controller._billing_blocks(sim.index[k0:k0 + H], H)
    j0 = loss.evaluate(k0, base.plan, obs["t_indoor"])["total_inr"]

    out = []
    for bi, b in enumerate(blocks):
        costs = {}
        for sign in (+1.0, -1.0):
            shifted = _ShiftedForecast(controller.fc, b["steps"], sign * delta_kw)
            probe = ChanceConstrainedMPC(
                controller.tariff, shifted, controller.cfg,
                risk_quantile=controller.risk_quantile,
                solar_quantile=controller.solar_quantile,
                demand_target_kw=controller.demand_target_kw)
            probe.d_committed_kw = controller.d_committed_kw
            probe._block_progress = list(controller._block_progress)
            r = probe.solve(sim, obs)
            costs[sign] = (loss.evaluate(k0, r.plan, obs["t_indoor"])["total_inr"]
                           if r.plan else j0)
        out.append({
            "block": bi,
            "steps": b["steps"],
            "lead_steps": int(b["steps"][0]),
            "d_cost_d_kw": (costs[+1.0] - costs[-1.0]) / (2.0 * delta_kw),
            "abs_d_cost_d_kw": abs(costs[+1.0] - costs[-1.0]) / (2.0 * delta_kw),
        })
    return {"ok": True, "k0": int(k0), "base_cost_inr": j0, "blocks": out}


class _ShiftedForecast:
    """A forecast source with a constant offset added to selected horizon steps.

    Wraps rather than mutates, so the probe cannot leave a residue in the source
    the real controller is reading. The offset is applied to every quantile
    level, which is the right perturbation: what is being measured is the
    sensitivity to the forecast being *wrong*, not to it becoming wider.
    """

    def __init__(self, inner, steps: list[int], delta: float):
        self.inner = inner
        self._steps = np.asarray(steps, int)
        self._delta = float(delta)
        self.max_horizon = getattr(inner, "max_horizon", 10**9)

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        out = np.asarray(self.inner.base(k0, H, q), float).copy()
        s = self._steps[self._steps < H]
        out[s] += self._delta
        return out

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.inner.pv(k0, H, q)
