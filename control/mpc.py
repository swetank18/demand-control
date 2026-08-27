"""Chance-constrained model predictive control, as a MILP.

The whole argument of this project is one substitution. In the constraint that
holds grid import under the monthly demand ceiling we use the **95th percentile**
of the base-load forecast and the **5th percentile** of solar, not the means:

    base_q95[t] + controllable[t] - solar_q05[t]  <=  D_peak

That is a chance constraint implemented by quantile substitution. It is a few
characters of code. It is also the entire reason the controller does not blow
the monthly demand charge, and it is why the calibration work upstream is load
bearing rather than decorative: if q95 is not really a 95th percentile, this
guarantee is theatre.

Everything else follows the standard hierarchical template: forecast, then a
deterministic optimiser, then the plant. No learned policy anywhere near a
money decision.

Where risk goes and where expectation goes:
  * demand ceiling -> q95. Breaching it costs a full month of demand charge.
  * energy cost    -> q50. Costing energy at q95 would systematically overstate
                     the bill and distort the trade-off.
  * thermal comfort-> q50, with slack. Comfort is a soft constraint by design;
                     the operator sets the budget and violations are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp, Bounds
from scipy.sparse import coo_matrix

from sim.thermal import DT_H, Action, BuildingSim

BIG_SLACK_PENALTY = 5_000.0     # INR per K per step of comfort violation


@dataclass
class MPCConfig:
    horizon: int = 64                  # 16 hours at 15 minutes
    comfort_penalty: float = BIG_SLACK_PENALTY
    tank_penalty: float = 2_000.0
    ev_penalty: float = 800.0          # INR per kWh of undelivered charge
    terminal_soc_penalty: float = 50.0
    # Only the first step of the plan is ever applied, so the on/off decision has
    # to be exact only near the front of the horizon. Keeping the tail continuous
    # is standard move-blocking: it cuts branch-and-bound from 32 binaries to a
    # handful and costs nothing in the action actually taken.
    binary_steps: int = 8
    relax_binaries: bool = False       # fully continuous water heater, for speed
    solve_every: int = 1               # re-solve cadence in steps
    time_limit_s: float = 5.0
    mip_gap: float = 0.005
    # deliberate model mismatch, for the robustness study
    ua_scale: float = 1.0
    c_scale: float = 1.0
    # Shaping cost on the planned ceiling, INR per kW. Small enough to be noise in
    # the bill, large enough to break the degeneracy described in
    # ``ChanceConstrainedMPC``: without it, every block below the committed peak
    # is free and the optimiser fills it.
    peak_shaping_cost: float = 2.0
    # Whether a peak already set this month dissolves the target.
    #
    # Strictly, this month's bill only sees the monthly maximum, so once 450 kW
    # has been recorded there is no bill-reduction left below 450 kW and the
    # greedy policy is to stop defending. That policy is also useless: it gives
    # up permanently after one bad block, which is precisely the block a
    # mis-specified forecast produces. And it misreads what the target is. The
    # demand target is a *forward commitment* -- contract demand, renegotiated
    # between billing cycles -- so exceeding it is costly every time it happens,
    # not once. We therefore defend the target continuously by default.
    ratchet_to_committed: bool = False


class _VarIndex:
    """Flat variable layout for the MILP.

    ``extra`` appends named blocks after the standard ones. The scenario
    formulation in ``control/scenario_mpc.py`` needs a per-scenario indicator
    and a couple of risk variables; giving it a hook here is what lets it reuse
    this entire solver rather than fork a second copy of the plant model, which
    would then drift.
    """

    def __init__(self, H: int, has_wh: bool, has_ev: bool, has_batt: bool,
                 extra: dict[str, int] | None = None):
        self.H = H
        self.slices: dict[str, slice] = {}
        n = 0

        def add(name: str, size: int) -> None:
            nonlocal n
            self.slices[name] = slice(n, n + size)
            n += size

        add("hvac", H)
        add("T", H)
        add("s_hi", H)
        add("s_lo", H)
        if has_wh:
            add("wh", H)
            add("Ttank", H)
            add("s_tank", H)
        if has_ev:
            add("ev", H)
            add("ev_short", 1)
        if has_batt:
            add("bc", H)
            add("bd", H)
            add("soc", H)
            add("s_soc", 1)
        add("Dpeak", 1)
        add("excess", 1)
        for name, size in (extra or {}).items():
            if size > 0:
                add(name, size)
        self.n = n
        self.has_wh, self.has_ev, self.has_batt = has_wh, has_ev, has_batt

    def has(self, name: str) -> bool:
        return name in self.slices

    def __call__(self, name: str) -> slice:
        return self.slices[name]


@dataclass
class MPCResult:
    action: Action
    status: str
    objective: float
    d_peak_kw: float
    plan: dict[str, np.ndarray] = field(default_factory=dict)


class ChanceConstrainedMPC:
    """The controller. ``risk_quantile='q50'`` turns it into the mean-forecast
    baseline, and feeding it the actuals turns it into the perfect-foresight
    oracle -- same code path, so the table compares like with like."""

    def __init__(
        self,
        tariff,
        forecast: "ForecastSource",
        cfg: MPCConfig | None = None,
        risk_quantile: str = "q95",
        solar_quantile: str = "q05",
        name: str | None = None,
        demand_target_kw: float | None = None,
    ):
        self.tariff = tariff
        self.fc = forecast
        self.cfg = cfg or MPCConfig()
        self.risk_quantile = risk_quantile
        self.solar_quantile = solar_quantile
        self.name = name or f"MPC[{risk_quantile}]"
        # The demand target is the operator's demand limit, in kW at the site
        # power factor. A receding horizon of 8 hours cannot choose a *monthly*
        # peak on its own -- it has no way to know that today's convenient peak
        # will be beaten on the 24th -- so the month's ceiling is an input, the
        # same way the comfort band is an input. Defending it under forecast
        # uncertainty is exactly what the chance constraint is for.
        self.demand_target_kw = demand_target_kw
        self.d_committed_kw = 0.0
        # Power already metered in the billing block that is currently in
        # progress. A 30-minute block is two control steps, so half of it can
        # already be spent by the time we solve; constraining only the remaining
        # steps lets the realised block average exceed the ceiling we thought we
        # were defending.
        self._block_progress: list[float] = []
        self._cache: MPCResult | None = None
        self._age = 0
        self.solve_times: list[float] = []
        self.statuses: list[str] = []
        #: The ceiling the optimiser committed to at each step, one entry per
        #: simulator step. The risk formulations make their promise about *this*
        #: number -- "no block in the window exceeds D_peak with probability
        #: 1 - eps" -- and not about the operator's monthly target, which the
        #: plant may simply be unable to hold. Measuring the promise against the
        #: target conflates a formulation that is working with a building that
        #: is short of capacity, so the runner records both.
        self.planned_dpeak: list[float] = []

    # -- interface used by the runner ------------------------------------
    def reset(self) -> None:
        self.d_committed_kw = 0.0
        self._block_progress = []
        self._cache = None
        self._age = 0
        self.solve_times.clear()
        self.statuses.clear()
        self.planned_dpeak.clear()

    def note_block_progress(self, realised_kw: list[float]) -> None:
        """Power already metered in the billing block still in progress."""
        self._block_progress = list(realised_kw)

    def note_realised(self, block_avg_kw: float) -> None:
        """Tell the controller what the month has already committed to. Once a
        peak has been set there is nothing left to protect below it, and a
        controller that does not know this leaves money on the table."""
        self.d_committed_kw = max(self.d_committed_kw, block_avg_kw)

    def act(self, sim: BuildingSim, obs: dict) -> Action:
        import time

        if self._cache is not None and self._age < self.cfg.solve_every:
            self._age += 1
            self.planned_dpeak.append(self._cache.d_peak_kw)
            return self._advance_cached()
        t0 = time.perf_counter()
        res = self.solve(sim, obs)
        self.solve_times.append(time.perf_counter() - t0)
        self.statuses.append(res.status)
        self.planned_dpeak.append(res.d_peak_kw)
        self._cache = res
        self._age = 1
        return res.action

    def _advance_cached(self) -> Action:
        p = self._cache.plan
        i = min(self._age - 1, len(p["hvac"]) - 1)
        return Action(
            hvac_kw=float(p["hvac"][i]),
            wh_on=int(round(p["wh"][i])) if "wh" in p else 0,
            ev_kw=float(p["ev"][i]) if "ev" in p else 0.0,
            batt_charge_kw=float(p["bc"][i]) if "bc" in p else 0.0,
            batt_discharge_kw=float(p["bd"][i]) if "bd" in p else 0.0,
        )

    # -- extension points -------------------------------------------------
    # The scenario formulation differs from this one in exactly one place: the
    # rows that hold grid import under the ceiling. Everything else -- the RC
    # envelope, the tank, the EV window, the battery, the objective, the
    # fallback -- is identical and must stay identical, or the comparison
    # between the two stops being a comparison of risk formulations and becomes
    # a comparison of two separately maintained plant models.

    def _extra_vars(self, H: int) -> dict[str, int]:
        """Extra variable blocks appended to the layout. Empty for this one."""
        return {}

    def _extra_bounds(self, V: "_VarIndex", lo, hi, integrality, H: int) -> None:
        pass

    def _extra_objective(self, V: "_VarIndex", c, H: int) -> None:
        pass

    def _billing_blocks(self, idx, H: int) -> list[dict]:
        """Group horizon steps into tariff billing blocks.

        A block is billed on its average over the *whole* block, so the divisor
        is the block length and not the number of steps still under our control.
        Anything already metered in the block in progress is a fixed head start,
        carried here as ``head`` so both formulations account for it the same
        way -- a half-spent block that the optimiser thinks it owns entirely is
        how a defended ceiling gets breached by arithmetic rather than by
        forecast error.
        """
        bi = self.tariff.billing_interval_minutes
        block_id = np.array([(ts.hour * 60 + ts.minute) // bi for ts in idx])
        day_id = np.array([ts.dayofyear for ts in idx])
        keyed: dict[tuple[int, int], list[int]] = {}
        for t in range(H):
            keyed.setdefault((int(day_id[t]), int(block_id[t])), []).append(t)

        steps_per_block = max(1, bi // 15)
        first_key = (int(day_id[0]), int(block_id[0]))
        out = []
        for key, steps in keyed.items():
            elapsed = self._block_progress if key == first_key else []
            n = steps_per_block if len(steps) + len(elapsed) >= steps_per_block else len(steps)
            out.append({"key": key, "steps": steps, "n": n,
                        "head": (sum(elapsed) / n) if elapsed else 0.0})
        return out

    def _block_coeffs(self, V: "_VarIndex", steps: list[int], n: int, p) -> dict[int, float]:
        """Coefficients of the controllable variables in one block average."""
        hv = V("hvac").start
        coeffs: dict[int, float] = {}
        for t in steps:
            coeffs[hv + t] = coeffs.get(hv + t, 0.0) + 1.0 / n
            if V.has_wh:
                j = V("wh").start + t
                coeffs[j] = coeffs.get(j, 0.0) + p.water_heater.power_kw / n
            if V.has_ev:
                j = V("ev").start + t
                coeffs[j] = coeffs.get(j, 0.0) + 1.0 / n
            if V.has_batt:
                j = V("bc").start + t
                coeffs[j] = coeffs.get(j, 0.0) + 1.0 / n
                j = V("bd").start + t
                coeffs[j] = coeffs.get(j, 0.0) - 1.0 / n
        return coeffs

    def _add_demand_rows(self, add_row, V, blocks, base_risk, pv_risk, p, k0, H) -> None:
        """The ceiling, one row per billing block, at a single risk quantile.

        Substituting q95 makes each row individually a 95% chance constraint.
        Whether 32 of those compose into anything is the question Track B exists
        to answer, and the answer is no -- see ``control/scenario_mpc.py``.
        """
        dpi = V("Dpeak").start
        for b in blocks:
            coeffs = {dpi: -1.0, **self._block_coeffs(V, b["steps"], b["n"], p)}
            rhs = -b["head"] - sum((base_risk[t] - pv_risk[t]) / b["n"] for t in b["steps"])
            add_row(coeffs, -np.inf, rhs)

    # -- the optimisation -------------------------------------------------
    def solve(self, sim: BuildingSim, obs: dict) -> MPCResult:
        p = sim.p
        cfg = self.cfg
        k0 = sim.k
        # Never plan further than the forecaster can actually see. Padding a
        # forecast out to a longer horizon would put invented numbers into the
        # constraint that carries the safety claim.
        H = min(cfg.horizon, len(sim.index) - k0, getattr(self.fc, "max_horizon", cfg.horizon))
        if H <= 1:
            return MPCResult(Action(), "horizon-exhausted", 0.0, self.d_committed_kw)

        idx = sim.index[k0 : k0 + H]
        has_wh = p.water_heater is not None
        has_ev = p.ev is not None
        has_batt = p.battery is not None
        V = _VarIndex(H, has_wh, has_ev, has_batt, self._extra_vars(H))

        # ---- exogenous forecasts ----------------------------------------
        base_risk = self.fc.base(k0, H, self.risk_quantile)
        base_exp = self.fc.base(k0, H, "q50")
        pv_risk = self.fc.pv(k0, H, self.solar_quantile)
        pv_exp = self.fc.pv(k0, H, "q50")
        t_out = sim.exog["t_out"].to_numpy()[k0 : k0 + H]
        occ_gain = sim.exog["occ_gain_kw"].to_numpy()[k0 : k0 + H]
        sol_gain = sim.exog["solar_gain_kw"].to_numpy()[k0 : k0 + H]
        q_int = p.internal_gain_fraction * base_exp + occ_gain + sol_gain

        price = np.array([self.tariff.rate_for(ts.hour * 60 + ts.minute) for ts in idx])
        band = p.comfort.band_series(idx)
        t_lo, t_hi = band["t_lo"].to_numpy(), band["t_hi"].to_numpy()

        # ---- model of the plant (optionally mismatched on purpose) -------
        ua = p.ua_kw_per_k * cfg.ua_scale
        cap_c = p.c_kwh_per_k * cfg.c_scale
        a = 1.0 - DT_H * ua / cap_c
        b = DT_H * p.cop / cap_c
        const = (DT_H / cap_c) * (t_out * ua + q_int)

        # ---- objective ---------------------------------------------------
        c = np.zeros(V.n)
        c[V("hvac")] = price * DT_H
        c[V("s_hi")] = cfg.comfort_penalty
        c[V("s_lo")] = cfg.comfort_penalty
        if has_wh:
            c[V("wh")] = price * DT_H * p.water_heater.power_kw
            c[V("s_tank")] = cfg.tank_penalty
        if has_ev:
            c[V("ev")] = price * DT_H
            c[V("ev_short")] = cfg.ev_penalty
        if has_batt:
            c[V("bc")] = price * DT_H
            c[V("bd")] = -price * DT_H * 0.999   # discharge displaces import
            c[V("s_soc")] = cfg.terminal_soc_penalty
        # The demand charge bites only above the reference: the month's target,
        # or whatever peak has already been committed, whichever is higher. Money
        # already spent is not a reason to keep spending, but it is also not
        # recoverable, so there is nothing to protect below it.
        d_ref = (max(self.demand_target_kw or 0.0, self.d_committed_kw)
                 if cfg.ratchet_to_committed else (self.demand_target_kw or self.d_committed_kw))
        c[V("excess")] = self.tariff.demand_charge_per_kva / max(p.power_factor, 1e-6)
        c[V("Dpeak")] = cfg.peak_shaping_cost
        self._extra_objective(V, c, H)

        # constant part of the energy cost (uncontrollable), dropped: it does not
        # change the argmin. Reported separately by the bill engine.

        # ---- bounds -------------------------------------------------------
        lo = np.zeros(V.n)
        hi = np.full(V.n, np.inf)
        hi[V("hvac")] = p.hvac_capacity_kw
        lo[V("T")] = -20.0
        hi[V("T")] = 60.0
        if has_wh:
            hi[V("wh")] = 1.0
            lo[V("Ttank")] = 0.0
            hi[V("Ttank")] = 95.0
        if has_ev:
            ev = p.ev
            in_win = np.array([
                (ts.dayofweek in ev.active_days)
                and (ev.arrive_h <= ts.hour + ts.minute / 60.0 < ev.depart_h)
                for ts in idx
            ])
            hi[V("ev")] = np.where(in_win, ev.max_kw, 0.0)
        if has_batt:
            bt = p.battery
            hi[V("bc")] = bt.max_charge_kw
            hi[V("bd")] = bt.max_discharge_kw
            lo[V("soc")] = bt.soc_min
            hi[V("soc")] = bt.soc_max
        lo[V("Dpeak")] = max(self.d_committed_kw, 0.0) if cfg.ratchet_to_committed else 0.0

        integrality = np.zeros(V.n)
        if has_wh and not cfg.relax_binaries:
            nb = max(0, min(cfg.binary_steps, H))
            integrality[V("wh").start : V("wh").start + nb] = 1
        self._extra_bounds(V, lo, hi, integrality, H)

        # Sparse triplets, not dense rows. Each constraint touches a handful of
        # the variables; materialising a dense row per constraint is fine for a
        # 64-step horizon and impossible for a whole-month oracle, where the
        # dense matrix would be terabytes.
        ri: list[int] = []
        ci: list[int] = []
        vv: list[float] = []
        lb: list[float] = []
        ub: list[float] = []

        def add_row(coeffs: dict[int, float], low: float, up: float) -> None:
            r = len(lb)
            for j, v in coeffs.items():
                ri.append(r); ci.append(j); vv.append(v)
            lb.append(low)
            ub.append(up)

        Ti = V("T").start
        hv = V("hvac").start

        # ---- envelope dynamics -------------------------------------------
        # T[0] is the state after applying hvac[0] to the measured temperature
        add_row({Ti + 0: 1.0, hv + 0: b}, a * obs["t_indoor"] + const[0], a * obs["t_indoor"] + const[0])
        for t in range(1, H):
            add_row({Ti + t: 1.0, Ti + t - 1: -a, hv + t: b}, const[t], const[t])

        # ---- comfort with slack --------------------------------------------
        shi, slo = V("s_hi").start, V("s_lo").start
        for t in range(H):
            add_row({Ti + t: 1.0, shi + t: -1.0}, -np.inf, t_hi[t])
            add_row({Ti + t: 1.0, slo + t: 1.0}, t_lo[t], np.inf)

        # ---- water heater tank ----------------------------------------------
        if has_wh:
            wh = p.water_heater
            wi, tki, tsi = V("wh").start, V("Ttank").start, V("s_tank").start
            aw = 1.0 - DT_H * wh.ua_kw_per_k / wh.tank_kwh_per_k
            bw = DT_H * wh.power_kw * wh.efficiency / wh.tank_kwh_per_k
            draw = np.array([wh.daily_draw_kwh / 12.0 if 7 <= ts.hour < 19 else 0.0 for ts in idx])
            cw = (DT_H / wh.tank_kwh_per_k) * (wh.ua_kw_per_k * wh.t_ambient - draw)
            add_row({tki: 1.0, wi: -bw}, aw * obs["t_tank"] + cw[0], aw * obs["t_tank"] + cw[0])
            for t in range(1, H):
                add_row({tki + t: 1.0, tki + t - 1: -aw, wi + t: -bw}, cw[t], cw[t])
            for t in range(H):
                add_row({tki + t: 1.0, tsi + t: 1.0}, wh.t_min, np.inf)
                add_row({tki + t: 1.0, tsi + t: -1.0}, -np.inf, wh.t_max)

        # ---- EV energy by departure ------------------------------------------
        if has_ev:
            ev = p.ev
            evi, esi = V("ev").start, V("ev_short").start
            hnow = idx[0].hour + idx[0].minute / 60.0
            if in_win.any():
                remaining = max(ev.required_kwh_per_day - obs["ev_delivered_kwh"], 0.0)
                # only demand what physically fits in the remaining window
                fits = float(in_win.sum()) * DT_H * ev.max_kw
                need = min(remaining, fits)
                add_row({**{evi + t: DT_H for t in range(H) if in_win[t]}, esi: 1.0}, need, np.inf)
            else:
                add_row({esi: 1.0}, 0.0, 0.0)

        # ---- battery ---------------------------------------------------------
        if has_batt:
            bt = p.battery
            bci, bdi, sci, ssi = V("bc").start, V("bd").start, V("soc").start, V("s_soc").start
            k_ch = DT_H * bt.efficiency / bt.capacity_kwh
            k_di = DT_H / (bt.efficiency * bt.capacity_kwh)
            add_row({sci: 1.0, bci: -k_ch, bdi: k_di}, obs["soc"], obs["soc"])
            for t in range(1, H):
                add_row({sci + t: 1.0, sci + t - 1: -1.0, bci + t: -k_ch, bdi + t: k_di}, 0.0, 0.0)
            # terminal condition, softened: end the horizon no worse than you started
            add_row({sci + H - 1: 1.0, ssi: 1.0}, min(obs["soc"], bt.soc_max), np.inf)

        # ---- the demand ceiling, on billing-block averages -------------------
        # This is the chance constraint. base_risk is q95, pv_risk is q05.
        dpi = V("Dpeak").start
        blocks = self._billing_blocks(idx, H)
        self._add_demand_rows(add_row, V, blocks, base_risk, pv_risk, p, k0, H)

        # excess >= Dpeak - d_ref, excess >= 0 (bounds already give the latter)
        add_row({V("excess").start: 1.0, dpi: -1.0}, -d_ref, np.inf)

        A = coo_matrix((vv, (ri, ci)), shape=(len(lb), V.n)).tocsc()
        cons = LinearConstraint(A, np.array(lb), np.array(ub))
        res = milp(
            c=c,
            constraints=cons,
            bounds=Bounds(lo, hi),
            integrality=integrality,
            options={"time_limit": cfg.time_limit_s, "mip_rel_gap": cfg.mip_gap, "presolve": True},
        )

        if res.x is None:
            # Never let the plant go uncontrolled: fall back to holding the band.
            from control.baselines import hold_power_kw

            target = min(max(obs["t_indoor"], obs["t_lo"]), obs["t_hi"])
            return MPCResult(
                Action(hvac_kw=hold_power_kw(sim, obs, target), wh_on=0, ev_kw=0.0),
                f"infeasible:{res.status}", float("nan"), self.d_committed_kw,
            )

        x = res.x
        plan = {"hvac": x[V("hvac")], "T": x[V("T")]}
        if has_wh:
            # Kept raw. Only the first ``binary_steps`` entries are integral; the
            # relaxed tail is never applied, and rounding it here would make the
            # stored plan violate the very constraint it was solved under.
            plan["wh"] = x[V("wh")]
        if has_ev:
            plan["ev"] = x[V("ev")]
        if has_batt:
            plan["bc"], plan["bd"] = x[V("bc")], x[V("bd")]

        action = Action(
            hvac_kw=float(plan["hvac"][0]),
            wh_on=int(round(plan["wh"][0])) if has_wh else 0,
            ev_kw=float(plan["ev"][0]) if has_ev else 0.0,
            batt_charge_kw=float(plan["bc"][0]) if has_batt else 0.0,
            batt_discharge_kw=float(plan["bd"][0]) if has_batt else 0.0,
        )
        return MPCResult(action, "optimal", float(res.fun), float(x[dpi]), plan)


class FullHorizonOracle:
    """The genuine upper bound: one MILP over the entire billing month.

    The rolling controller with perfect forecasts is *not* the achievable
    optimum -- it still cannot see past its 16-hour window, so it cannot decide
    on the 3rd that the peak worth defending is the one on the 24th. A monthly
    demand charge is a global coupling across the whole month, and only a
    whole-month problem can price it correctly.

    This is what the "% of oracle savings captured" column is measured against.
    Water-heater binaries are relaxed to continuous to keep ~23,000 variables
    tractable; the relaxation can only *help* the oracle, which is the safe
    direction for an upper bound.
    """

    name = "Perfect foresight oracle"

    def __init__(self, tariff, cfg: MPCConfig | None = None, demand_target_kw: float | None = None):
        self.tariff = tariff
        self.cfg = cfg or MPCConfig()
        self.demand_target_kw = demand_target_kw
        self.plan: dict[str, np.ndarray] | None = None
        self.status = "unsolved"
        self.solve_times: list[float] = []
        self.statuses: list[str] = []
        self._i = 0

    def reset(self) -> None:
        self.plan = None
        self._i = 0
        self.solve_times.clear()
        self.statuses.clear()

    def note_realised(self, block_avg_kw: float) -> None:
        pass

    def note_block_progress(self, realised_kw: list[float]) -> None:
        pass

    def act(self, sim: BuildingSim, obs: dict) -> Action:
        if self.plan is None:
            import time

            t0 = time.perf_counter()
            self._solve_month(sim, obs)
            self.solve_times.append(time.perf_counter() - t0)
            self.statuses.append(self.status)
        i = min(self._i, len(self.plan["hvac"]) - 1)
        self._i += 1
        p = self.plan
        return Action(
            hvac_kw=float(p["hvac"][i]),
            wh_on=int(round(p["wh"][i])) if "wh" in p else 0,
            ev_kw=float(p["ev"][i]) if "ev" in p else 0.0,
            batt_charge_kw=float(p["bc"][i]) if "bc" in p else 0.0,
            batt_discharge_kw=float(p["bd"][i]) if "bd" in p else 0.0,
        )

    def _solve_month(self, sim: BuildingSim, obs: dict) -> None:
        from forecast.sources import OracleForecast

        n = len(sim.index) - sim.k
        fc = OracleForecast(
            sim.exog["base_kw"].to_numpy(), sim.exog["pv_kw"].to_numpy()
        )
        cfg = replace(self.cfg, horizon=n, binary_steps=0, relax_binaries=True,
                      time_limit_s=max(self.cfg.time_limit_s, 120.0))
        inner = ChanceConstrainedMPC(
            self.tariff, fc, cfg, risk_quantile="q50", solar_quantile="q50",
            demand_target_kw=self.demand_target_kw,
        )
        res = inner.solve(sim, obs)
        self.status = res.status
        if res.plan:
            self.plan = res.plan
        else:
            self.plan = {"hvac": np.zeros(n), "wh": np.zeros(n), "ev": np.zeros(n)}
