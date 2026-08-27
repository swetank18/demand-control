"""Scenario-based MPC: a chance constraint over the whole horizon, not per step.

What is wrong with the controller this subclasses. It substitutes q95 into one
constraint per billing block, which makes each block individually a 95%
guarantee. The thing an operator is actually buying is that the *month* stays
under the ceiling, and 32 marginal guarantees do not compose into one joint
guarantee. They do not even come close: with 32 blocks in a horizon and even
mild dependence, the probability that at least one breaks through is many times
the 5% each row promises. That is a correctness bug in the formulation, not a
tuning problem, and no amount of better calibration fixes it -- a perfectly
calibrated marginal is still a marginal.

The fix is sample average approximation, which is the standard construction in
stochastic programming and old enough to be uncontroversial. Draw S joint
trajectories from the copula in ``forecast/trajectories.py``, write the ceiling
constraint once per scenario per block, and cap how many scenarios may violate:

    for each scenario s, for each block b:
        grid[s,b]  <=  D_peak + M * z[s]
    sum_s p[s] * z[s]  <=  epsilon

``z[s]`` is one binary per *scenario*, not per block. That is the whole point:
excusing a scenario excuses its entire trajectory, so what the cardinality
constraint limits is the fraction of futures in which anything at all goes
wrong. That is a joint chance constraint over the horizon, which is what the
constraint was always supposed to be.

The controllable decisions stay here-and-now variables shared across every
scenario. They have to: one schedule gets committed and the plant does not get
to pick which future it is in. Only the violation indicators are recourse.

**Two formulations, and when to use which.**

``joint-cc`` is the above, exactly. It is the honest object and it carries S
binaries into branch-and-bound.

``cvar`` replaces the indicator with a conditional-value-at-risk bound on the
worst-case block excess. CVaR at level 1-epsilon being non-positive is a
conservative convex approximation of the chance constraint (Rockafellar and
Uryasev; Nemirovski and Shapiro for the approximation argument): every solution
it admits is feasible for the chance constraint, and it admits fewer of them.
It has no binaries at all, so it solves in milliseconds and never times out.
Conservative means the ceiling is defended harder than asked and some headroom
is left unused -- which is the right direction to be wrong in, and is why this
is the default for whole-month runs.
"""
from __future__ import annotations

import numpy as np

from control.mpc import ChanceConstrainedMPC, MPCConfig


class ScenarioMPC(ChanceConstrainedMPC):
    """Same plant, same objective, same solver. One block of rows differs.

    Everything that is not the risk formulation is inherited, deliberately. If
    the scenario controller carried its own copy of the RC envelope or the tank
    dynamics, then a difference in the results table could be a difference in
    the plant model, and the comparison would be worthless.
    """

    def __init__(
        self,
        tariff,
        forecast,
        cfg: MPCConfig | None = None,
        epsilon: float = 0.05,
        mode: str = "cvar",
        name: str | None = None,
        demand_target_kw: float | None = None,
        solar_quantile: str = "q05",
    ):
        if mode not in ("joint-cc", "cvar"):
            raise ValueError(f"mode must be 'joint-cc' or 'cvar', got {mode!r}")
        if not hasattr(forecast, "scenarios"):
            raise TypeError("ScenarioMPC needs a forecast source with .scenarios(k0, H)")
        super().__init__(
            tariff, forecast, cfg,
            # The parent still reads these for the energy-cost terms and for the
            # infeasible fallback. The ceiling no longer uses them at all: risk
            # now lives in the scenario set, not in a quantile label.
            risk_quantile="q50", solar_quantile=solar_quantile,
            name=name or f"Scenario MPC[{mode}, eps={epsilon:g}]",
            demand_target_kw=demand_target_kw,
        )
        self.epsilon = float(epsilon)
        self.mode = mode
        self.S = int(forecast.reduce_to or forecast.n_scenarios)
        #: filled in each solve, so the study code can report what the optimiser
        #: was actually looking at without re-drawing the scenarios
        self.last_scenarios: dict | None = None

    # -- extra variables ---------------------------------------------------
    def _extra_vars(self, H: int) -> dict[str, int]:
        if self.mode == "joint-cc":
            return {"z": self.S}
        return {"g": self.S, "w": self.S, "eta": 1}

    def _extra_bounds(self, V, lo, hi, integrality, H: int) -> None:
        if self.mode == "joint-cc":
            lo[V("z")], hi[V("z")] = 0.0, 1.0
            integrality[V("z")] = 1
        else:
            # g is the per-scenario worst block excess over the planned ceiling.
            # It must be free: a scenario that comfortably clears the ceiling has
            # a negative excess, and clamping that at zero would turn CVaR into a
            # robust worst-case constraint and give away most of the headroom.
            lo[V("g")], hi[V("g")] = -np.inf, np.inf
            lo[V("w")], hi[V("w")] = 0.0, np.inf
            lo[V("eta")], hi[V("eta")] = -np.inf, np.inf

    def _extra_objective(self, V, c, H: int) -> None:
        if self.mode == "joint-cc":
            # A nominal price on excusing a scenario. Not a risk preference --
            # it is far too small to trade against the demand charge. It breaks
            # ties so the solver excuses only the scenarios it must, which makes
            # the reported violation count mean something and gives
            # branch-and-bound a direction to prune in.
            c[V("z")] = 1e-3

    # -- the ceiling, rewritten --------------------------------------------
    def _add_demand_rows(self, add_row, V, blocks, base_risk, pv_risk, p, k0, H) -> None:
        base_s, pv_s, weights = self.fc.scenarios(k0, H)
        net = base_s - pv_s                                    # (S, H)
        S = net.shape[0]
        dpi = V("Dpeak").start

        # per-scenario, per-block exogenous contribution to the block average
        block_net = np.array([
            [net[s, b["steps"]].sum() / b["n"] + b["head"] for b in blocks] for s in range(S)
        ])                                                      # (S, B)
        coeffs_per_block = [self._block_coeffs(V, b["steps"], b["n"], p) for b in blocks]

        self.last_scenarios = {
            "k0": int(k0), "H": int(H), "S": int(S),
            "block_net_max": float(block_net.max()),
            "block_net_p95": float(np.quantile(block_net.max(axis=1), 0.95)),
            "weights": weights.tolist(),
        }

        if self.mode == "joint-cc":
            zi = V("z").start
            # Big-M large enough to make an excused row vacuous and no larger.
            # An M of 1e6 would be "safe" and would also hand the LP relaxation a
            # z of 1e-4 on every scenario, which satisfies the cardinality
            # constraint for free and makes the whole formulation collapse.
            controllable_max = self._max_controllable_kw(p)
            M = float(block_net.max() + controllable_max) + 1.0
            for s in range(S):
                for j, b in enumerate(blocks):
                    row = {dpi: -1.0, zi + s: -M, **coeffs_per_block[j]}
                    add_row(row, -np.inf, -block_net[s, j])
            # at most an epsilon fraction of the probability mass may violate
            add_row({zi + s: float(weights[s]) for s in range(S)}, -np.inf, self.epsilon)
            return

        # -- CVaR ----------------------------------------------------------
        gi, wi, ei = V("g").start, V("w").start, V("eta").start
        for s in range(S):
            for j, b in enumerate(blocks):
                # g[s] >= block_avg[s,j] - Dpeak
                row = {dpi: -1.0, gi + s: -1.0, **coeffs_per_block[j]}
                add_row(row, -np.inf, -block_net[s, j])
            # w[s] >= g[s] - eta   (w >= 0 comes from the bounds)
            add_row({gi + s: 1.0, ei: -1.0, wi + s: -1.0}, -np.inf, 0.0)
        # eta + (1/epsilon) * E[w] <= 0, the Rockafellar-Uryasev form of
        # CVaR_{1-epsilon}(g) <= 0
        row = {ei: 1.0}
        for s in range(S):
            row[wi + s] = float(weights[s]) / self.epsilon
        add_row(row, -np.inf, 0.0)

    @staticmethod
    def _max_controllable_kw(p) -> float:
        """Every controllable kW the plant can draw at once. Only used to size
        the big-M, where an over-estimate costs solve time and an under-estimate
        would silently make an excused scenario still binding."""
        total = float(p.hvac_capacity_kw)
        if p.water_heater is not None:
            total += float(p.water_heater.power_kw)
        if p.ev is not None:
            total += float(p.ev.max_kw)
        if p.battery is not None:
            total += float(p.battery.max_charge_kw)
        return total


def horizon_violation_rate(paths: np.ndarray, ceiling: float, steps_per_block: int) -> float:
    """Fraction of trajectories in which *any* billing block clears ``ceiling``.

    The quantity the joint chance constraint controls, and the one the closed-
    loop acceptance test in ``eval/horizon_risk.py`` is measured against.
    """
    p = np.asarray(paths, float)
    n, H = p.shape
    m = (H // steps_per_block) * steps_per_block
    if m == 0:
        return float(np.mean((p > ceiling).any(axis=1)))
    blocks = p[:, :m].reshape(n, -1, steps_per_block).mean(axis=2)
    return float(np.mean((blocks > ceiling).any(axis=1)))
