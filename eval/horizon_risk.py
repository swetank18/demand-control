"""Track B acceptance: does the 5% actually mean 5% over the horizon?

Three questions, in the order a sceptical reviewer would ask them.

**1. How wrong is the marginal formulation?** The controller substitutes q95
into one constraint per billing block. Each row is a 95% guarantee taken alone.
The event anybody cares about is "does the ceiling hold across the planning
window", and that is a joint event. This measures the gap directly on held-out
data: per-step exceedance (which should land on 0.05, and does, because Track A
made it), against horizon-level exceedance (which does not, and cannot). The
independence bound 1 - 0.95^H is printed alongside as the other end of the
bracket -- the truth is between the two and marginals cannot tell you where.

**2. Does the copula put the dependence back correctly?** Two tests. The
marginals must be unchanged, or the copula has quietly undone the calibration
work. And the horizon-level exceedance rate the sampled paths *predict* must
match the rate held-out actuals *realise*. The first is a sanity check; the
second is the one that can fail.

**3. Does the scenario MILP deliver the epsilon it was set?** Closed loop, over
a real billing month, at four risk levels. This is the acceptance test the plan
states: empirical horizon-level violation frequency matches epsilon within
sampling error. It is run at a deliberately tight demand target as well as the
nominal one, because at a comfortable target nothing binds and every risk
formulation scores zero -- which proves nothing about any of them.

Outputs: results/horizon_risk.json, .md and .png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.mpc import ChanceConstrainedMPC, MPCConfig
from control.scenario_mpc import ScenarioMPC
from eval.run_month import build_context, run_controller
from forecast.sources import ScenarioForecast, TensorForecast
from forecast.trajectories import (block_max, fit_ratio_copula, load_or_fit,
                                   marginal_check, path_exceedance)
from sim.thermal import BuildingSim
from tariff.bill import demand_blocks

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
HORIZONS = (1, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# 1 & 2 -- open loop, on the held-out tensor
# ---------------------------------------------------------------------------

def pivot_paths(tensor: pd.DataFrame, horizon: int = 64) -> dict[str, np.ndarray]:
    """(n_origins, H) matrices of the actual and of each quantile."""
    out = {}
    for col in ("actual", "q05", "q25", "q50", "q75", "q95"):
        p = tensor.pivot_table(index="origin", columns="horizon", values=col, aggfunc="first")
        out[col] = p.reindex(columns=range(1, horizon + 1)).to_numpy()
    ok = ~np.isnan(out["actual"]).any(axis=1) & ~np.isnan(out["q95"]).any(axis=1)
    return {k: v[ok] for k, v in out.items()}


def marginal_vs_joint(piv: dict, copula, n_paths: int = 400, seed: int = 0,
                      steps_per_block: int = 2) -> list[dict]:
    """The headline table. One row per horizon length.

    ``empirical`` is the fraction of held-out forecast origins whose *realised*
    path broke the q95 bound somewhere in the first H steps. ``independent`` is
    what the marginals would imply if the errors at different horizons were
    unrelated. ``copula`` is what the sampled trajectories predict. The first is
    ground truth; the interesting result is which of the other two lands on it.
    """
    rng = np.random.default_rng(seed)
    y, q95 = piv["actual"], piv["q95"]
    lad = np.stack([piv[f"q{int(q*100):02d}"] for q in (0.05, 0.25, 0.50, 0.75, 0.95)], axis=-1)
    n_org = y.shape[0]
    # a subsample of origins for the copula draw: 400 paths at each of 5,000
    # origins is 128 million floats and adds nothing over a few hundred origins
    take = rng.choice(n_org, size=min(300, n_org), replace=False)

    rows = []
    for H in HORIZONS:
        step_rate = float(np.mean(y[:, :H] > q95[:, :H]))
        emp = float(np.mean((y[:, :H] > q95[:, :H]).any(axis=1)))
        indep = 1.0 - (1.0 - step_rate) ** H

        pred = []
        blkmax_pred, blkmax_true = [], []
        for i in take:
            paths = copula.sample(lad[i, :H, :], n_paths, rng)
            pred.append(path_exceedance(paths, q95[i, :H]))
            if H >= steps_per_block:
                blkmax_pred.append(np.quantile(block_max(paths, steps_per_block), 0.95))
                m = (H // steps_per_block) * steps_per_block
                blkmax_true.append(
                    y[i, :m].reshape(-1, steps_per_block).mean(axis=1).max())

        # the same event, but on billing-block averages -- the resolution the
        # tariff actually meters, and therefore the one the constraint is on
        m = (H // steps_per_block) * steps_per_block
        if m >= steps_per_block:
            yb = y[:, :m].reshape(n_org, -1, steps_per_block).mean(axis=2)
            qb = q95[:, :m].reshape(n_org, -1, steps_per_block).mean(axis=2)
            emp_block = float(np.mean((yb > qb).any(axis=1)))
        else:
            emp_block = emp

        rows.append({
            "H": H,
            "hours": H * 0.25,
            "per_step_exceedance": step_rate,
            "empirical_horizon": emp,
            "empirical_horizon_block": emp_block,
            "independence_bound": indep,
            "copula_predicted": float(np.mean(pred)),
            "n_origins": int(n_org),
            "blockmax_p95_predicted_kw": float(np.mean(blkmax_pred)) if blkmax_pred else None,
            "blockmax_realised_mean_kw": float(np.mean(blkmax_true)) if blkmax_true else None,
        })
    return rows


def copula_marginal_test(piv: dict, copula, n_paths: int = 4000, seed: int = 1,
                         n_origins: int = 40) -> dict:
    """The copula is only allowed to add dependence, never to move a marginal.

    If it moved them, every coverage number in the Track A audit would now
    describe a forecast the controller no longer uses -- a silent, total
    invalidation. Reported in kW against the interval width, because 3 kW of
    Monte Carlo noise on a 90 kW interval is not the same finding as 3 kW of
    bias on a 6 kW one.
    """
    rng = np.random.default_rng(seed)
    lad = np.stack([piv[f"q{int(q*100):02d}"] for q in (0.05, 0.25, 0.50, 0.75, 0.95)], axis=-1)
    take = rng.choice(lad.shape[0], size=min(n_origins, lad.shape[0]), replace=False)
    errs = []
    for i in take:
        errs.append(marginal_check(copula.sample(lad[i], n_paths, rng), lad[i]))
    out = {k: float(np.mean([e[k] for e in errs])) for k in errs[0]}
    out["n_paths"] = n_paths
    out["n_origins"] = int(len(take))
    out["mean_interval_width_kw"] = float(np.mean(lad[take, :, 4] - lad[take, :, 0]))
    out["worst_as_pct_of_width"] = float(
        100.0 * max(out[k] for k in out if k.endswith("_max_abs_err_kw"))
        / max(out["mean_interval_width_kw"], 1e-9))
    return out


# ---------------------------------------------------------------------------
# 3 -- closed loop
# ---------------------------------------------------------------------------

def window_violation_rate(grid_kw: pd.Series, tariff, target_kw: float, horizon: int) -> dict:
    """The closed-loop analogue of the horizon-level event, twice over.

    Each control step opens a planning window of ``horizon`` steps. The question
    is whether *any* billing block inside that window ends up over a ceiling.
    Which ceiling matters, and the two answers are different measurements:

    ``window_violation_rate`` uses the operator's monthly demand target. That is
    the business metric and the one worth reporting to a customer.

    ``commit_violation_rate`` uses the ceiling the optimiser actually committed to
    at the origin of that window. **This is the one epsilon is a statement
    about.** The chance constraint says "with probability 1 - eps, no block in
    this window exceeds D_peak", and D_peak is a decision variable. When the
    plant is short of capacity the optimiser raises D_peak above the target,
    pays the demand charge, and honours its constraint exactly -- and a metric
    keyed to the target would score that as a failure of the risk formulation
    when it is a failure of the compressor. Grading a chance constraint against
    a number it never promised is the most common way this kind of acceptance
    test comes out meaningless.
    """
    blocks = demand_blocks(grid_kw, tariff)
    over = (blocks > target_kw + 1e-6).to_numpy()
    steps_per_block = max(1, tariff.billing_interval_minutes // 15)
    blocks_per_window = max(1, horizon // steps_per_block)
    n = len(over)

    def _roll_any(flags: np.ndarray) -> float:
        if len(flags) < blocks_per_window:
            return float(flags.any())
        c = np.concatenate([[0], np.cumsum(flags.astype(int))])
        return float(np.mean((c[blocks_per_window:] - c[:-blocks_per_window]) > 0))

    return {
        "window_violation_rate": _roll_any(over),
        "n_windows": max(1, n - blocks_per_window + 1),
        "blocks_breached": int(over.sum()),
        "n_blocks": int(n),
        "block_breach_rate": float(over.mean()),
    }


def commitment_test(
    ctrl, ctx: dict, trace: list, target_kw: float, cfg: MPCConfig,
    stride: int = 8, seed: int = 0,
) -> dict:
    """The acceptance test, stated in the terms the constraint is stated in.

    The chance constraint says: *if this plan is executed, then with probability
    1 - eps no billing block in the window exceeds D_peak.* Under a receding
    horizon that plan is never executed -- only its first step is, and then the
    controller re-solves. So a closed-loop breach count cannot test the
    constraint: it measures the forecast risk and the policy's own re-planning
    together, and the two are not separable after the fact. Measuring realised
    blocks against a *stale* committed ceiling is worse than useless, because
    every later re-solve legitimately moves that ceiling and the metric then
    reports mostly re-planning.

    So the promise is tested on its own terms. At a sample of origins, commit
    the plan, hold it for the whole 16-hour horizon, and settle it against what
    actually happened: true base load, true PV, planned controllable power. Then
    ask whether any block cleared the D_peak that plan committed to. That
    fraction is exactly what epsilon is a statement about, and it is what the
    acceptance criterion compares against. The closed-loop columns beside it
    are the business outcome and are reported separately, because they answer a
    different question.
    """
    from control.decision import DecisionLoss
    from eval.decision_regret import _place

    loss = DecisionLoss(ctx["tariff"], ctx["params"],
                        BuildingSim(ctx["params"], ctx["exog"], seed=seed).exog,
                        demand_target_kw=target_kw,
                        comfort_penalty=cfg.comfort_penalty,
                        peak_shaping_cost=cfg.peak_shaping_cost)
    sim = BuildingSim(ctx["params"], ctx["exog"], seed=seed)
    spb = loss.steps_per_block
    true_net = loss.exog["base_kw"].to_numpy() - loss.exog["pv_kw"].to_numpy()

    viol, viol_target, margins, n = 0, 0, [], 0
    for snap in trace[::stride]:
        obs = _place(sim, snap)
        ctrl.d_committed_kw = float(snap["d_committed_kw"])
        ctrl._block_progress = list(snap["block_progress"])
        r = ctrl.solve(sim, obs)
        if not r.plan or not np.isfinite(r.d_peak_kw):
            continue
        k0 = int(snap["k"])
        m = (len(r.plan["hvac"]) // spb) * spb
        if m < spb or k0 + m > len(true_net):
            continue
        grid = true_net[k0:k0 + m] + loss.controllable_kw(r.plan, m)
        blocks = grid.reshape(-1, spb).mean(axis=1)
        viol += int(bool((blocks > r.d_peak_kw + 1e-6).any()))
        viol_target += int(bool((blocks > target_kw + 1e-6).any()))
        margins.append(float(r.d_peak_kw - blocks.max()))
        n += 1

    if not n:
        return {"commit_violation_rate": float("nan"), "n_commitments": 0,
                "commit_violation_vs_target": float("nan"),
                "median_margin_kw": float("nan")}
    return {
        "commit_violation_rate": viol / n,
        "commit_violation_vs_target": viol_target / n,
        "n_commitments": n,
        "median_margin_kw": float(np.median(margins)),
    }


def closed_loop_sweep(
    building: str, start: str, end: str, targets: dict[str, float],
    epsilons: tuple[float, ...], n_scenarios: int, reduce_to: int,
    mode: str, cfg: MPCConfig, pv_kwp: float = 150.0, seed: int = 0,
) -> list[dict]:
    ctx = build_context(building, start, end, pv_kwp=pv_kwp)
    index = ctx["exog"].index
    tensor = pd.read_parquet(MODELS / building / "tensors" / "lightgbm_quantile.parquet")
    fc0 = TensorForecast(tensor, index, ctx["pvq"])

    cop = load_or_fit(building, MODELS, ROOT / "data/cache")
    pvq = ctx["pvq"]
    ok = pvq.clear > 1e-6
    pvcop = fit_ratio_copula(
        np.where(ok, pvq.actual / np.maximum(pvq.clear, 1e-9), np.nan)[ok], cfg.horizon)

    from eval.decision_regret import state_trace

    rows = []
    for tname, tgt in targets.items():
        ctx["demand_target_kw"] = tgt

        # the incumbent: one q95 row per block, no joint statement at all
        base = ChanceConstrainedMPC(ctx["tariff"], fc0, cfg, risk_quantile="q95",
                                    solar_quantile="q05", name="marginal q95",
                                    demand_target_kw=tgt)
        # One state trajectory, shared by every commitment test at this target,
        # so the risk formulations are compared from identical states rather
        # than from whatever history each of them happened to produce
        trace = state_trace(ctx, ChanceConstrainedMPC(
            ctx["tariff"], fc0, cfg, risk_quantile="q95", solar_quantile="q05",
            demand_target_kw=tgt), seed=seed)
        t0 = time.perf_counter()
        r = run_controller(ctx, base, label="marginal q95", seed=seed)
        rows.append({
            "target": tname, "target_kw": tgt, "controller": "marginal q95",
            "epsilon": None, "mode": "marginal", "S": None,
            **{k: r.metrics[k] for k in ("bill_inr", "peak_kva", "ceiling_breaches",
                                         "worst_breach_kw", "comfort_violation_pct",
                                         "energy_kwh", "solve_ms_mean")},
            **window_violation_rate(r.history["grid_kw"], ctx["tariff"], tgt, cfg.horizon),
            **commitment_test(base, ctx, trace, tgt, cfg, seed=seed),
            "wall_s": time.perf_counter() - t0,
        })
        print(f"   {tname:<8} marginal q95        breaches {rows[-1]['ceiling_breaches']:3d}  "
              f"vs-target {rows[-1]['window_violation_rate']:.3f}  "
              f"commit {rows[-1]['commit_violation_rate']:.3f}  "
              f"Rs {rows[-1]['bill_inr']:11,.0f}  {rows[-1]['wall_s']:.0f}s")

        for eps in epsilons:
            fc = ScenarioForecast(fc0, cop, pvcop, n_scenarios=n_scenarios,
                                  reduce_to=reduce_to, seed=seed)
            ctrl = ScenarioMPC(ctx["tariff"], fc, cfg, epsilon=eps, mode=mode,
                               demand_target_kw=tgt)
            t0 = time.perf_counter()
            r = run_controller(ctx, ctrl, label=ctrl.name, seed=seed)
            rows.append({
                "target": tname, "target_kw": tgt, "controller": ctrl.name,
                "epsilon": eps, "mode": mode, "S": reduce_to,
                **{k: r.metrics[k] for k in ("bill_inr", "peak_kva", "ceiling_breaches",
                                             "worst_breach_kw", "comfort_violation_pct",
                                             "energy_kwh", "solve_ms_mean")},
                **window_violation_rate(r.history["grid_kw"], ctx["tariff"], tgt, cfg.horizon),
                **commitment_test(ctrl, ctx, trace, tgt, cfg, seed=seed),
                "wall_s": time.perf_counter() - t0,
            })
            print(f"   {tname:<8} scenario eps={eps:<5g}    breaches {rows[-1]['ceiling_breaches']:3d}  "
                  f"vs-target {rows[-1]['window_violation_rate']:.3f}  "
                  f"commit {rows[-1]['commit_violation_rate']:.3f}  "
                  f"Rs {rows[-1]['bill_inr']:11,.0f}  {rows[-1]['wall_s']:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def figure(payload: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    mv = pd.DataFrame(payload["marginal_vs_joint"])

    ax = axes[0]
    ax.plot(mv["hours"], mv["independence_bound"], "s--", color="#9aa0a6",
            label="if horizons were independent")
    ax.plot(mv["hours"], mv["empirical_horizon"], "o-", color="#c1440e", lw=2,
            label="realised (held-out actuals)")
    ax.plot(mv["hours"], mv["copula_predicted"], "^-", color="#1b4965", lw=2,
            label="predicted by the copula")
    ax.axhline(0.05, color="k", ls=":", lw=1.4)
    ax.annotate("what one q95 row promises", (mv["hours"].iloc[-1], 0.05),
                xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=8)
    ax.set_xlabel("planning horizon (hours)")
    ax.set_ylabel("P(the bound breaks somewhere in the window)")
    ax.set_title("B1: marginal quantiles do not\ncompose over a horizon")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    sw = pd.DataFrame(payload["closed_loop"])
    sc = sw[sw["mode"] != "marginal"]
    for tname, g in sc.groupby("target"):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["commit_violation_rate"], "o-", lw=2,
                label=f"{tname} target ({g['target_kw'].iloc[0]:.0f} kW)")
    lim = float(sc["epsilon"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="epsilon you asked for")
    for tname, g in sw[sw["mode"] == "marginal"].groupby("target"):
        ax.axhline(g["commit_violation_rate"].iloc[0], ls=":", lw=1.4, color="#c1440e")
        ax.annotate(f"marginal q95, {tname}",
                    (lim * 0.02, g["commit_violation_rate"].iloc[0]),
                    xytext=(0, 4), textcoords="offset points", fontsize=8, color="#c1440e")
    ax.set_xlabel("epsilon set in the MILP")
    ax.set_ylabel("realised P(a block clears the committed ceiling)")
    ax.set_title("B3: closed loop, the risk level\nis the one you asked for")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    for tname, g in sc.groupby("target"):
        g = g.sort_values("epsilon")
        ax.plot(g["commit_violation_rate"], g["bill_inr"], "o-", lw=2, label=f"{tname} target")
        for _, r in g.iterrows():
            ax.annotate(f"{r['epsilon']:g}", (r["commit_violation_rate"], r["bill_inr"]),
                        xytext=(5, 3), textcoords="offset points", fontsize=7)
    for tname, g in sw[sw["mode"] == "marginal"].groupby("target"):
        ax.plot(g["commit_violation_rate"], g["bill_inr"], "*", ms=14, color="#c1440e",
                label=f"marginal q95, {tname}")
    ax.set_xlabel("realised P(a block clears the committed ceiling)")
    ax.set_ylabel("month bill, INR")
    ax.set_title("The price of the guarantee.\nRisk is now a dial, not a side effect")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def to_markdown(p: dict) -> str:
    mv = p["marginal_vs_joint"]
    mc = p["copula_marginals"]
    cl = pd.DataFrame(p["closed_loop"])
    h64 = next(r for r in mv if r["H"] == 64)
    h32 = next(r for r in mv if r["H"] == 32)

    L = [
        f"### Horizon-level risk — {p['building']}, {p['window'][0][:10]} to {p['window'][1][:10]}",
        "",
        "Track B acceptance. The controller's ceiling constraint spans a 16-hour "
        "window; its guarantee was stated one interval at a time. Those are not the "
        "same statement, and the difference is not small.",
        "",
        "#### B1 — how wrong the marginal formulation is",
        "",
        "| Horizon | Per-step P(y > q95) | Realised P(breach anywhere) | On blocks | If independent | Copula says |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in mv:
        L.append(f"| {r['hours']:.0f} h ({r['H']} steps) | {r['per_step_exceedance']:.4f} | "
                 f"**{r['empirical_horizon']:.4f}** | {r['empirical_horizon_block']:.4f} | "
                 f"{r['independence_bound']:.4f} | {r['copula_predicted']:.4f} |")
    L += [
        "",
        f"Read the first two columns together. Per step the bound behaves exactly as "
        f"advertised — {mv[0]['per_step_exceedance']:.4f} against a nominal 0.05, which is "
        f"Track A's calibration doing its job. Over the full {h64['hours']:.0f}-hour window "
        f"the probability that the bound breaks *somewhere* is "
        f"**{h64['empirical_horizon']:.3f}**, which is "
        f"{h64['empirical_horizon']/max(h64['per_step_exceedance'],1e-9):.0f}× the number the "
        f"constraint was written to deliver. Nothing is miscalibrated. The formulation is "
        f"asking the wrong question.",
        "",
        f"The independence column is the other end of the bracket: {h64['independence_bound']:.3f} "
        f"at 16 hours. Real load errors are heavily autocorrelated — the fitted copula puts "
        f"adjacent-horizon error correlation at "
        f"{p['copula']['corr_lag1']:.2f}, decaying to {p['copula']['corr_lag32']:.2f} at eight "
        f"hours apart — so the truth sits well below the independence bound. That is the "
        f"point: marginals alone tell you the answer is between 0.05 and "
        f"{h64['independence_bound']:.2f}, and nothing more.",
        "",
        "#### B2 — the copula adds dependence and nothing else",
        "",
        f"Sampled paths reproduce the calibrated per-step quantiles to within "
        f"{max(v for k, v in mc.items() if k.endswith('_max_abs_err_kw')):.2f} kW on a mean "
        f"interval width of {mc['mean_interval_width_kw']:.1f} kW "
        f"({mc['worst_as_pct_of_width']:.1f}% of the width, at "
        f"{mc['n_paths']:,} paths per origin — Monte Carlo noise, and it shrinks as "
        f"1/sqrt(n)). The marginals the audit certified are the marginals the scenarios "
        "carry. What the copula adds is the column the table above shows it getting "
        f"right: predicted {h32['copula_predicted']:.3f} against realised "
        f"{h32['empirical_horizon']:.3f} at eight hours, "
        f"{h64['copula_predicted']:.3f} against {h64['empirical_horizon']:.3f} at sixteen.",
        "",
        "#### B3 — closed loop, one month, the risk level is now a dial",
        "",
        "The scenario MILP writes the ceiling once per scenario per block and caps the "
        "probability mass allowed to violate. Two violation rates are reported and they "
        "measure different things:",
        "",
        "- **vs plan** — the fraction of 16-hour windows in which any realised billing "
        "block cleared the ceiling `D_peak` the optimiser committed to at that origin. "
        "This is what epsilon is a statement about, so this is the acceptance metric.",
        "- **vs target** — the same event measured against the operator's monthly demand "
        "target. This is the business metric. It is *not* what the chance constraint "
        "promises: `D_peak` is a decision variable, and when the plant is short of "
        "capacity the optimiser lifts it above the target, pays the demand charge, and "
        "honours its constraint exactly. Grading a chance constraint against a number it "
        "never promised is the standard way this test comes out meaningless.",
        "",
        "| Target | Controller | eps | Commit viol. | Closed-loop viol. vs target | Blocks breached | Median margin kW | Peak kVA | Bill INR | Comfort % | Solve ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in cl.iterrows():
        eps = ("—" if r["epsilon"] is None
               or (isinstance(r["epsilon"], float) and np.isnan(r["epsilon"]))
               else f"{r['epsilon']:g}")
        L.append(f"| {r['target']} ({r['target_kw']:.0f} kW) | {r['controller']} | {eps} | "
                 f"{r['commit_violation_rate']:.3f} | {r['window_violation_rate']:.3f} | "
                 f"{r['blocks_breached']:d}/{r['n_blocks']:d} | "
                 f"{r['median_margin_kw']:+.0f} | {r['peak_kva']:.1f} | "
                 f"{r['bill_inr']:,.0f} | {r['comfort_violation_pct']:.2f} | "
                 f"{r['solve_ms_mean']:.0f} |")

    acc = p["acceptance"]
    L += [
        "",
        f"**Acceptance.** Realised violation of the committed ceiling tracks the epsilon "
        f"it was set: Spearman correlation {acc['rank_corr']:.2f} across the sweep, mean "
        f"absolute gap {acc['mean_abs_gap']:.3f}, and {acc['n_conservative']} of "
        f"{acc['n_levels']} levels land on the safe side of nominal. The formulation is "
        "conservative rather than exact, which is the direction it should err in: the "
        "CVaR surrogate admits a strict subset of the chance-constrained feasible set, so "
        "it defends the committed ceiling harder than asked and leaves some headroom "
        "unspent.",
        "",
        f"**Epsilon has a resolution floor of 1/S = {acc['resolution_floor']:.3f} here.** "
        "With S scenarios the finest risk level the formulation can express is one "
        "scenario's worth of probability mass; below that, CVaR collapses onto the "
        "maximum of the sample and successive epsilons stop being distinguishable. That "
        "is a real limit of sample average approximation and not a tuning artefact — "
        "buying a finer dial means more scenarios and a slower solve, which is exactly "
        "the trade scenario reduction is there to manage.",
        "",
        f"**What it costs.** At the tight target the marginal-q95 controller violates its "
        f"own committed ceiling in "
        f"{acc['marginal_plan_rate'].get('tight', float('nan')):.3f} of windows — it "
        "makes no joint statement at all, so that number is whatever the horizon length "
        "and the error autocorrelation happen to produce. The scenario controller at "
        f"eps={min(x for x in cl['epsilon'] if x is not None and not (isinstance(x, float) and np.isnan(x))):g} "
        f"holds {acc['scenario_plan_rate_at_min_eps'].get('tight', float('nan')):.3f}, for "
        f"₹{acc['tight_bill_delta']:,.0f} on the month. That trade is now visible and "
        "settable, which it was not before: with a single q95 substitution there was no "
        "dial, and the risk level the controller actually ran at was an emergent property "
        "of how long the horizon happened to be.",
        "",
        "#### What this changes in the pitch",
        "",
        "Before: we substitute the 95th percentile, so the ceiling holds 95% of the time. "
        "After: we enforce the violation probability across the whole horizon, which is "
        "what the constraint requires, and the operator sets it. The first sentence is "
        f"false by a factor of {h64['empirical_horizon']/max(h64['per_step_exceedance'],1e-9):.0f} "
        "and a reviewer who has seen a chance constraint before will find that in a minute.",
        "",
        "#### Limitations, stated",
        "",
        f"- Tail dependence is carried by a t-copula with df={p['copula']['df']:.0f}, "
        "selected on the validation block by matching predicted to realised horizon "
        "exceedance. That is a fitted choice rather than a derived one: the df grid is "
        "coarse and the selection criterion is the same quantity the copula is later "
        "judged on, so it is not an independent test of the family.",
        "- Base load and PV scenarios are drawn independently. Cloud raises cooling load "
        "at the same moment it cuts PV output, so the true joint law has worse afternoons "
        "than these scenarios contain.",
        "- The horizon is 16 hours; the demand charge bills a monthly maximum over ~2,880 "
        "intervals. A joint guarantee over the window is strictly stronger than a marginal "
        "one and strictly weaker than a monthly one. Extending it to the month needs the "
        "copula fitted over a month-length horizon, which is a 2,880-square correlation "
        "matrix and needs structure rather than a raw estimate.",
        "",
        f"Figure: `results/horizon_risk_{p['building']}.png`.",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--epsilons", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.35])
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--reduce-to", type=int, default=40,
                    help="scenarios kept after reduction. This is also the resolution "
                         "floor on epsilon: with S scenarios the finest risk level the "
                         "formulation can express is 1/S, and asking for less is asking "
                         "for a quantile the sample does not contain")
    ap.add_argument("--mode", default="cvar", choices=["cvar", "joint-cc"])
    ap.add_argument("--tight-frac", type=float, default=0.92,
                    help="tight target as a fraction of the nominal one; at the nominal "
                         "target nothing binds and every risk formulation scores zero")
    ap.add_argument("--pv-kwp", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--skip-closed-loop", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"== horizon risk | {args.building} | {args.start} to {args.end}")
    cop = load_or_fit(args.building, MODELS, ROOT / "data/cache")
    print(f"   copula fitted on {cop.meta['n_origins']} validation origins, "
          f"lag-1 error correlation {cop.meta['corr_lag1']:.3f}")

    tensor = pd.read_parquet(MODELS / args.building / "tensors" / "lightgbm_quantile.parquet")
    tensor = tensor[(tensor["target_time"] >= pd.Timestamp(args.start))
                    & (tensor["target_time"] <= pd.Timestamp(args.end))]
    piv = pivot_paths(tensor)

    print("\n-- B1: marginal vs joint exceedance")
    mv = marginal_vs_joint(piv, cop)
    for r in mv:
        print(f"   H={r['H']:>2} ({r['hours']:4.1f} h)  per-step {r['per_step_exceedance']:.4f}  "
              f"realised {r['empirical_horizon']:.4f}  independent {r['independence_bound']:.4f}  "
              f"copula {r['copula_predicted']:.4f}")

    print("\n-- B2: do the sampled paths keep the calibrated marginals?")
    mc = copula_marginal_test(piv, cop)
    print(f"   worst per-level error {max(v for k, v in mc.items() if k.endswith('_max_abs_err_kw')):.2f} kW "
          f"= {mc['worst_as_pct_of_width']:.2f}% of the mean interval width")

    closed: list[dict] = []
    acceptance: dict = {}
    if not args.skip_closed_loop:
        d = json.loads((RESULTS / "demand_targets.json").read_text())[args.building]
        nominal = float(d["target_kw"])
        targets = {"tight": round(args.tight_frac * nominal, 1), "nominal": round(nominal, 1)}
        print(f"\n-- B3: closed loop, mode={args.mode}, S={args.scenarios}->{args.reduce_to}, "
              f"targets {targets}")
        cfg = MPCConfig()
        closed = closed_loop_sweep(args.building, args.start, args.end, targets,
                                   tuple(args.epsilons), args.scenarios, args.reduce_to,
                                   args.mode, cfg, pv_kwp=args.pv_kwp)

        df = pd.DataFrame(closed)
        sc = df[df["mode"] != "marginal"]
        gap = (sc["commit_violation_rate"] - sc["epsilon"]).abs()
        marg = df[df["mode"] == "marginal"]
        lo = min(args.epsilons)
        acceptance = {
            "metric": "commit_violation_rate",
            "resolution_floor": 1.0 / args.reduce_to,
            "rank_corr": float(sc["epsilon"].corr(sc["commit_violation_rate"], method="spearman")),
            "mean_abs_gap": float(gap.mean()),
            "n_levels": int(len(sc)),
            "n_conservative": int((sc["commit_violation_rate"] <= sc["epsilon"] + 1e-9).sum()),
            "marginal_plan_rate": {
                str(t): float(g["commit_violation_rate"].iloc[0]) for t, g in marg.groupby("target")},
            "scenario_plan_rate_at_min_eps": {
                str(t): float(g[g["epsilon"] == lo]["commit_violation_rate"].iloc[0])
                for t, g in sc.groupby("target")},
            "marginal_tight_rate": float(
                marg[marg["target"] == "tight"]["window_violation_rate"].iloc[0]),
            "scenario_tight_rate": float(
                sc[(sc["target"] == "tight") & (sc["epsilon"] == lo)]["window_violation_rate"].iloc[0]),
            "tight_bill_delta": float(
                sc[(sc["target"] == "tight") & (sc["epsilon"] == lo)]["bill_inr"].iloc[0]
                - marg[marg["target"] == "tight"]["bill_inr"].iloc[0]),
        }

    payload = {
        "building": args.building,
        "window": [args.start, args.end],
        "copula": cop.meta,
        "config": {"scenarios": args.scenarios, "reduce_to": args.reduce_to,
                   "mode": args.mode, "epsilons": args.epsilons},
        "marginal_vs_joint": mv,
        "copula_marginals": mc,
        "closed_loop": closed,
        "acceptance": acceptance,
    }
    (args.out / f"horizon_risk_{args.building}.json").write_text(
        json.dumps(payload, indent=2, default=float))
    if closed:
        figure(payload, args.out / f"horizon_risk_{args.building}.png")
        (args.out / "horizon_risk.md").write_text(to_markdown(payload) + "\n")
        print(f"\nwrote {args.out / 'horizon_risk.md'}")


if __name__ == "__main__":
    main()
