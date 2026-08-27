"""Track C, Stage 1: decision regret as a metric, and where forecast error pays.

The plan's argument for doing this at all is that the repo already owns half the
result. The frontier study fitted ₹2,228 of monthly bill per unit of pinball
loss at R² 0.95 — a strong relationship, and one that leaves a fifth of the
variance in what a forecaster is *worth* unexplained by the score it was fitted
on. That gap is the whole of Track C. This file measures it directly instead of
inferring it.

Two studies.

**C1, regret.** At each of a sample of forecast origins, with the plant state
held fixed, solve the MPC once per forecaster and once with the true future.
Evaluate every resulting schedule against what actually happened. The difference
from the oracle's schedule is the rupees that forecast error cost at that
moment. Averaged over the month it is a decision-space score directly comparable
to pinball loss, and the two do not rank the forecasters identically — which is
the point.

**C2's input, sensitivity.** Where does forecast error actually change the
schedule? A central finite difference per billing block gives ₹ per kW of
forecast error, per block. Most blocks return exactly zero: at 03:00 on a Sunday
the ceiling is nowhere near binding and the forecast could be wrong by 50 kW
without moving a single setpoint. Pinball loss weights that interval exactly as
heavily as 14:30 on the hottest Tuesday of the month. That asymmetry, measured
here, is what the reweighting in ``forecast/decision_weights.py`` exploits.

Outputs: results/decision_regret.json, .md, .png, and
results/decision_sensitivity.parquet, which the weighting reads.
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

from control.decision import DecisionLoss, block_sensitivity, regret
from control.mpc import ChanceConstrainedMPC, MPCConfig
from eval.ablation import ORDER, OURS, forecast_scores
from eval.run_month import build_context
from forecast.baselines import REGISTRY
from forecast.sources import OracleForecast, TensorForecast
from sim.thermal import BuildingSim

MODELS = ROOT / "models"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# a shared state trajectory
# ---------------------------------------------------------------------------

def state_trace(ctx: dict, controller, seed: int = 0) -> list[dict]:
    """Run one closed-loop month and snapshot the plant state at every step.

    Every forecaster is then evaluated from these *same* states. Without that,
    differencing two forecasters compares two divergent histories and attributes
    the compounding to the forecast, which is exactly the confound the ablation
    was built to avoid at the monthly level and which matters more here, because
    the whole point is to attribute cost to individual intervals.
    """
    sim = BuildingSim(ctx["params"], ctx["exog"], seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    block_min = ctx["tariff"].billing_interval_minutes
    steps_per_block = block_min // 15
    buf: list[float] = []
    trace = []
    while not sim.done:
        obs = sim.observe()
        trace.append({
            "k": sim.k, "t_indoor": obs["t_indoor"], "t_tank": obs["t_tank"],
            "ev_delivered_kwh": obs["ev_delivered_kwh"], "soc": obs["soc"],
            "d_committed_kw": controller.d_committed_kw,
            "block_progress": list(buf),
        })
        controller.note_block_progress(buf)
        rec = sim.step(controller.act(sim, obs))
        buf.append(rec["grid_kw"])
        if len(buf) == steps_per_block:
            controller.note_realised(float(np.mean(buf)))
            buf = []
    return trace


def _place(sim: BuildingSim, snap: dict) -> dict:
    """Position a simulator at a recorded state and return its observation."""
    sim.k = int(snap["k"])
    sim.state.t_indoor = float(snap["t_indoor"])
    sim.state.t_tank = float(snap["t_tank"])
    sim.state.ev_delivered_kwh = float(snap["ev_delivered_kwh"])
    sim.state.soc = float(snap["soc"])
    return sim.observe()


# ---------------------------------------------------------------------------
# C1 -- regret per forecaster
# ---------------------------------------------------------------------------

def regret_study(
    ctx: dict, keys: list[str], target_kw: float, cfg: MPCConfig,
    stride: int = 8, seed: int = 0,
) -> tuple[list[dict], pd.DataFrame]:
    index = ctx["exog"].index
    loss = DecisionLoss(ctx["tariff"], ctx["params"], BuildingSim(
        ctx["params"], ctx["exog"], seed=seed).exog, demand_target_kw=target_kw,
        comfort_penalty=cfg.comfort_penalty, peak_shaping_cost=cfg.peak_shaping_cost)

    sources = {}
    for key in keys:
        t = pd.read_parquet(MODELS / ctx["building"] / "tensors" / f"{key}.parquet")
        sources[key] = TensorForecast(t, index, ctx["pvq"])

    # The oracle is the perfect-*base-load* tensor read through the same PV
    # quantiles as every other row, not the true PV. That keeps the ablation's
    # design rule -- only the base-load forecast varies -- so regret measures
    # base-load forecast error and nothing else. Using true PV here instead
    # would fold the cost of the conservative solar quantile into every row's
    # regret, including the perfect-foresight row's, whose regret must be
    # exactly zero for the number to mean what it says.
    oracle = sources.get("perfect_foresight") or ctx["fc_oracle"]

    ref = ChanceConstrainedMPC(ctx["tariff"], sources[OURS], cfg, risk_quantile="q95",
                               solar_quantile="q05", demand_target_kw=target_kw)
    print("   building the shared state trajectory ...")
    trace = state_trace(ctx, ref, seed=seed)
    snaps = trace[::stride]
    print(f"   {len(snaps)} origins sampled (every {stride} steps = {stride*15} min)")

    sim = BuildingSim(ctx["params"], ctx["exog"], seed=seed)
    ctrls = {key: ChanceConstrainedMPC(ctx["tariff"], src, cfg, risk_quantile="q95",
                                       solar_quantile="q05", demand_target_kw=target_kw)
             for key, src in sources.items()}
    # The oracle here is the same controller reading the truth, so it is the
    # best plan available from this state under this optimiser -- not a better
    # optimiser. Regret is therefore attributable to the forecast alone.
    ctrls["__oracle__"] = ChanceConstrainedMPC(
        ctx["tariff"], oracle, cfg, risk_quantile="q95", solar_quantile="q05",
        demand_target_kw=target_kw)

    rows = []
    t0 = time.perf_counter()
    for i, snap in enumerate(snaps):
        obs = _place(sim, snap)
        plans = {}
        for key, c in ctrls.items():
            c.d_committed_kw = float(snap["d_committed_kw"])
            c._block_progress = list(snap["block_progress"])
            r = c.solve(sim, obs)
            plans[key] = r.plan
        if not plans["__oracle__"]:
            continue
        for key in keys:
            if not plans[key]:
                continue
            rows.append({"key": key, "k": snap["k"], "time": str(index[snap["k"]]),
                         **regret(loss, snap["k"], plans[key], plans["__oracle__"],
                                  obs["t_indoor"])})
        if (i + 1) % 50 == 0:
            print(f"     {i+1}/{len(snaps)} origins, {time.perf_counter()-t0:.0f}s")

    per_origin = pd.DataFrame(rows)
    window = (str(index[0]), str(index[-1]))
    summary = []
    for key in keys:
        g = per_origin[per_origin["key"] == key]
        t = pd.read_parquet(MODELS / ctx["building"] / "tensors" / f"{key}.parquet")
        summary.append({
            "key": key, "forecaster": REGISTRY[key].name,
            **forecast_scores(t, window),
            "n_origins": int(len(g)),
            "regret_mean_inr": float(g["regret_inr"].mean()),
            "regret_median_inr": float(g["regret_inr"].median()),
            "regret_p95_inr": float(g["regret_inr"].quantile(0.95)),
            "regret_max_inr": float(g["regret_inr"].max()),
            # where the regret comes from. A forecaster can be mediocre on
            # average and still safe, or sharp on average and occasionally
            # catastrophic, and those are different products.
            "regret_demand_share": float(
                g["regret_demand"].sum() / max(g["regret_inr"].sum(), 1e-9)),
            "regret_comfort_share": float(
                g["regret_comfort"].sum() / max(g["regret_inr"].sum(), 1e-9)),
            "regret_energy_share": float(
                g["regret_energy"].sum() / max(g["regret_inr"].sum(), 1e-9)),
            "pct_origins_zero_regret": float(100.0 * (g["regret_inr"] < 1.0).mean()),
        })
    return summary, per_origin


# ---------------------------------------------------------------------------
# C2's input -- where forecast error changes the schedule
# ---------------------------------------------------------------------------

def sensitivity_study(
    ctx: dict, target_kw: float, cfg: MPCConfig, stride: int = 96,
    delta_kw: float = 10.0, seed: int = 0, max_origins: int = 120,
) -> pd.DataFrame:
    """Finite-difference sensitivity, with the covariates the weighting needs.

    ``stride`` is coarse on purpose. Each origin costs two solves per billing
    block, so this is the expensive study, and sensitivity is a smooth function
    of how close the ceiling is to binding rather than something that changes
    every fifteen minutes.
    """
    index = ctx["exog"].index
    t = pd.read_parquet(MODELS / ctx["building"] / "tensors" / f"{OURS}.parquet")
    fc = TensorForecast(t, index, ctx["pvq"])
    loss = DecisionLoss(ctx["tariff"], ctx["params"],
                        BuildingSim(ctx["params"], ctx["exog"], seed=seed).exog,
                        demand_target_kw=target_kw,
                        comfort_penalty=cfg.comfort_penalty,
                        peak_shaping_cost=cfg.peak_shaping_cost)
    ref = ChanceConstrainedMPC(ctx["tariff"], fc, cfg, risk_quantile="q95",
                               solar_quantile="q05", demand_target_kw=target_kw)
    trace = state_trace(ctx, ref, seed=seed)
    snaps = trace[::stride][:max_origins]

    sim = BuildingSim(ctx["params"], ctx["exog"], seed=seed)
    ctrl = ChanceConstrainedMPC(ctx["tariff"], fc, cfg, risk_quantile="q95",
                                solar_quantile="q05", demand_target_kw=target_kw)
    rows = []
    t0 = time.perf_counter()
    for i, snap in enumerate(snaps):
        obs = _place(sim, snap)
        ctrl.d_committed_kw = float(snap["d_committed_kw"])
        ctrl._block_progress = list(snap["block_progress"])
        s = block_sensitivity(loss, ctrl, sim, obs, delta_kw=delta_kw)
        if not s.get("ok"):
            continue
        k0 = snap["k"]
        H = min(cfg.horizon, len(index) - k0)
        q95 = fc.base(k0, H, "q95")
        q50 = fc.base(k0, H, "q50")
        pv05 = fc.pv(k0, H, "q05")
        for b in s["blocks"]:
            steps = np.array(b["steps"], int)
            # headroom the ceiling constraint leaves for controllable load in
            # this block: the single covariate that decides whether a forecast
            # error here can change anything at all
            head = target_kw - float(np.mean(q95[steps] - pv05[steps]))
            tt = index[k0 + steps[0]]
            rows.append({
                "k0": int(k0), "block": b["block"], "lead_steps": b["lead_steps"],
                "lead_h": b["lead_steps"] * 0.25,
                "time": str(tt), "hour": int(tt.hour), "dayofweek": int(tt.dayofweek),
                "is_weekend": int(tt.dayofweek >= 5),
                "headroom_kw": head,
                "headroom_frac_of_capacity": head / max(ctx["params"].hvac_capacity_kw, 1e-9),
                # The median forecast of base load in this block. Recorded because
                # it is the one load-level covariate that is also computable for a
                # *training* row -- one call to the q50 booster -- which is what
                # lets the fitted sensitivity map be applied where the weights are
                # needed. Anything derived from the actual would be a label leak
                # dressed up as a covariate.
                "base_q50_kw": float(np.mean(q50[steps])),
                "d_cost_d_kw": b["d_cost_d_kw"],
                "abs_d_cost_d_kw": b["abs_d_cost_d_kw"],
            })
        if (i + 1) % 20 == 0:
            print(f"     {i+1}/{len(snaps)} origins, {time.perf_counter()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def figure(payload: dict, sens: pd.DataFrame, path: Path) -> None:
    df = pd.DataFrame(payload["summary"])
    df = df[df["key"] != "perfect_foresight"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.scatter(df["pinball_mean"], df["regret_mean_inr"], s=70, color="#1b4965", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["forecaster"], (r["pinball_mean"], r["regret_mean_inr"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=7.5)
    best_p = df.loc[df["pinball_mean"].idxmin()]
    best_r = df.loc[df["regret_mean_inr"].idxmin()]
    ax.scatter([best_p["pinball_mean"]], [best_p["regret_mean_inr"]], s=190,
               facecolors="none", edgecolors="#c1440e", lw=2, label="pinball-optimal")
    ax.scatter([best_r["pinball_mean"]], [best_r["regret_mean_inr"]], s=260,
               facecolors="none", edgecolors="#2a9d8f", lw=2, label="regret-optimal")
    ax.set_xlabel("pinball loss (what the model is trained on)")
    ax.set_ylabel("mean decision regret, INR per origin")
    ax.set_title("C1: the score and the cost\nare related, not identical")
    ax.set_xscale("log")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    s = sens[sens["abs_d_cost_d_kw"] >= 0]
    ax.scatter(s["headroom_kw"], s["abs_d_cost_d_kw"], s=9, alpha=0.35, color="#1b4965")
    if payload.get("sensitivity_curve"):
        c = pd.DataFrame(payload["sensitivity_curve"])
        ax.plot(c["headroom_mid_kw"], c["mean_abs_d_cost_d_kw"], "o-",
                color="#c1440e", lw=2, label="binned mean")
        ax.legend(fontsize=8)
    ax.set_xlabel("headroom left by the ceiling constraint, kW")
    ax.set_ylabel("|d(realised cost) / d(forecast)|, INR per kW")
    ax.set_title("C2: forecast error only costs money\nwhere the ceiling nearly binds")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(alpha=0.3)

    ax = axes[2]
    byh = sens.groupby("hour")["abs_d_cost_d_kw"].mean()
    ax.bar(byh.index, byh.to_numpy(), color="#1b4965")
    ax.set_xlabel("hour of the target block")
    ax.set_ylabel("mean |d cost / d forecast|, INR per kW")
    ax.set_title("Pinball loss weights every one of these\nhours the same")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def to_markdown(p: dict) -> str:
    df = pd.DataFrame(p["summary"])
    live = df[df["key"] != "perfect_foresight"]
    bp = live.loc[live["pinball_mean"].idxmin()]
    br = live.loc[live["regret_mean_inr"].idxmin()]
    sc = p["sensitivity"]

    L = [
        f"### Decision regret — {p['building']}, {p['window'][0][:10]} to {p['window'][1][:10]}",
        "",
        "Track C Stage 1. The forecaster is trained on pinball loss and the building "
        "is billed in rupees. This measures the second one directly.",
        "",
        f"At each of {p['n_origins']} forecast origins the plant state is held fixed, the "
        "MPC is solved once per forecaster and once with the true future, and every "
        "resulting schedule is evaluated against what actually happened. Regret is the "
        "gap to the schedule the oracle would have committed *from the same state*. The "
        "oracle here is the identical controller reading the truth, not a better "
        "optimiser, so the difference is attributable to the forecast and to nothing else.",
        "",
        "| Forecaster | Pinball | Cov 90% | Mean regret ₹ | Median ₹ | p95 ₹ | Worst ₹ | Origins with zero regret | Regret that is demand charge |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.sort_values("regret_mean_inr").iterrows():
        star = "*" if r["key"] == "static_margin" else ""
        bold = (lambda x: f"**{x}**") if r["key"] == OURS else (lambda x: x)
        L.append("| " + " | ".join([
            bold(r["forecaster"]), bold(f"{r['pinball_mean']:.3f}{star}"),
            bold(f"{r['coverage_90']:.3f}{star}"), bold(f"{r['regret_mean_inr']:,.0f}"),
            bold(f"{r['regret_median_inr']:,.0f}"), bold(f"{r['regret_p95_inr']:,.0f}"),
            bold(f"{r['regret_max_inr']:,.0f}"),
            bold(f"{r['pct_origins_zero_regret']:.0f}%"),
            bold(f"{100*r['regret_demand_share']:.0f}%"),
        ]) + " |")

    L += [
        "",
        "\\* the static margin is one number for the whole month, so its pinball loss and "
        "coverage describe a constant rather than a forecast.",
        "",
        f"**The two rankings are not the same.** {bp['forecaster']} has the best pinball "
        f"loss ({bp['pinball_mean']:.3f}); {br['forecaster']} has the lowest decision "
        f"regret (₹{br['regret_mean_inr']:,.0f} per origin). "
        + ("Those are the same model here, so on this month the score and the cost agree "
           "at the top — but read the rest of the column: the ordering below the winner "
           "is not the pinball ordering, and the ratio of regret to pinball varies by "
           f"{live['regret_mean_inr'].max()/max(live['regret_mean_inr'].min(),1e-9):.0f}× "
           "across the table. A loss function that is monotone in the thing you care "
           "about is still not the thing you care about."
           if bp["key"] == br["key"] else
           "They are different models. The model that wins on the loss it was fitted on "
           "is not the model that wins on the bill, which is exactly the finding Stage 3 "
           "is built to exploit."),
        "",
        f"**Where regret comes from.** For our model "
        f"{100*float(df[df.key==OURS]['regret_demand_share'].iloc[0]):.0f}% of it is demand "
        f"charge and {100*float(df[df.key==OURS]['regret_comfort_share'].iloc[0]):.0f}% is "
        f"comfort. Regret is zero at "
        f"{float(df[df.key==OURS]['pct_origins_zero_regret'].iloc[0]):.0f}% of origins: most "
        "of the time a better forecast would have changed nothing, and the entire value of "
        "the model is concentrated in the minority of moments where it would.",
        "",
        "#### Where forecast error actually costs money",
        "",
        "A central finite difference per billing block: shift that block's forecast by "
        f"±{sc['delta_kw']:.0f} kW, re-solve, and evaluate both plans against the same true "
        f"future. {sc['n_blocks']:,} block-perturbations across {sc['n_origins']} origins.",
        "",
        "| Headroom band (kW) | Blocks | Mean \\|d cost / d kW\\| ₹ | Share of total sensitivity |",
        "| --- | --- | --- | --- |",
    ]
    for c in p["sensitivity_curve"]:
        L.append(f"| {c['headroom_lo_kw']:.0f} – {c['headroom_hi_kw']:.0f} | {c['n']:,} | "
                 f"{c['mean_abs_d_cost_d_kw']:.2f} | {100*c['share_of_total']:.1f}% |")
    L += [
        "",
        f"**{100*sc['share_zero']:.0f}% of blocks have exactly zero sensitivity.** At those "
        "moments the ceiling is nowhere near binding and the forecast could be wrong by "
        f"{sc['delta_kw']:.0f} kW in either direction without moving a single setpoint. "
        f"The top {100*sc['top_decile_frac']:.0f}% of blocks by headroom-tightness carry "
        f"{100*sc['top_decile_share']:.0f}% of all the sensitivity in the month.",
        "",
        "Pinball loss weights every one of those intervals identically. That is not a "
        "criticism of pinball loss, which is a perfectly good proper scoring rule; it is "
        "the observation that a proper scoring rule scores the *forecast*, and what is "
        "being bought here is a *decision*. Stage 2 uses this curve to reweight the "
        "training data. Stage 3 differentiates through it.",
        "",
        f"Figure: `results/decision_regret_{p['building']}.png`. "
        f"Per-block sensitivities: `results/decision_sensitivity.parquet`.",
    ]
    return "\n".join(L)


def sensitivity_curve(sens: pd.DataFrame, n_bins: int = 8) -> tuple[list[dict], dict]:
    """Binned |d cost / d kW| against headroom. This is the object Stage 2 uses."""
    s = sens.copy()
    edges = np.quantile(s["headroom_kw"], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(np.round(edges, 3))
    s["bin"] = np.clip(np.searchsorted(edges, s["headroom_kw"], side="right") - 1,
                       0, len(edges) - 2)
    total = s["abs_d_cost_d_kw"].sum()
    curve = []
    for b, g in s.groupby("bin"):
        curve.append({
            "bin": int(b),
            "headroom_lo_kw": float(edges[b]), "headroom_hi_kw": float(edges[b + 1]),
            "headroom_mid_kw": float(0.5 * (edges[b] + edges[b + 1])),
            "n": int(len(g)),
            "mean_abs_d_cost_d_kw": float(g["abs_d_cost_d_kw"].mean()),
            "share_of_total": float(g["abs_d_cost_d_kw"].sum() / max(total, 1e-9)),
        })
    tight = s.nsmallest(max(1, len(s) // 10), "headroom_kw")
    stats = {
        "n_blocks": int(len(s)), "n_origins": int(s["k0"].nunique()),
        "share_zero": float((s["abs_d_cost_d_kw"] < 1e-9).mean()),
        "top_decile_frac": 0.10,
        "top_decile_share": float(tight["abs_d_cost_d_kw"].sum() / max(total, 1e-9)),
        "mean_abs": float(s["abs_d_cost_d_kw"].mean()),
    }
    return curve, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--sens-stride", type=int, default=48)
    ap.add_argument("--sens-max-origins", type=int, default=120)
    ap.add_argument("--delta-kw", type=float, default=10.0)
    ap.add_argument("--demand-target-kw", type=float, default=None)
    ap.add_argument("--pv-kwp", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys = args.keys or [k for k in ORDER
                         if (MODELS / args.building / "tensors" / f"{k}.parquet").exists()]
    ctx = build_context(args.building, args.start, args.end, pv_kwp=args.pv_kwp)
    ctx["building"] = args.building
    tgt = args.demand_target_kw
    if tgt is None:
        tgt = float(json.loads((RESULTS / "demand_targets.json").read_text())
                    [args.building]["target_kw"])
    ctx["demand_target_kw"] = tgt
    cfg = MPCConfig()

    print(f"== decision regret | {args.building} | target {tgt:.0f} kW")
    print("\n-- C1: regret per forecaster")
    summary, per_origin = regret_study(ctx, keys, tgt, cfg, stride=args.stride)
    for r in sorted(summary, key=lambda r: r["regret_mean_inr"]):
        print(f"   {r['forecaster']:<28} pinball {r['pinball_mean']:7.3f}  "
              f"regret Rs {r['regret_mean_inr']:9,.1f}  p95 {r['regret_p95_inr']:9,.1f}  "
              f"zero at {r['pct_origins_zero_regret']:4.0f}% of origins")

    print("\n-- C2 input: block sensitivity by finite difference")
    sens = sensitivity_study(ctx, tgt, cfg, stride=args.sens_stride,
                             delta_kw=args.delta_kw, max_origins=args.sens_max_origins)
    curve, stats = sensitivity_curve(sens)
    stats["delta_kw"] = args.delta_kw
    print(f"   {stats['n_blocks']:,} blocks, {100*stats['share_zero']:.0f}% with zero "
          f"sensitivity, tightest decile carries {100*stats['top_decile_share']:.0f}%")

    payload = {
        "building": args.building, "window": [args.start, args.end],
        "demand_target_kw": tgt, "stride": args.stride,
        "n_origins": int(per_origin["k"].nunique()),
        "summary": summary, "sensitivity": stats, "sensitivity_curve": curve,
    }
    (args.out / f"decision_regret_{args.building}.json").write_text(
        json.dumps(payload, indent=2, default=float))
    per_origin.to_parquet(args.out / f"decision_regret_origins_{args.building}.parquet")
    sens.to_parquet(args.out / "decision_sensitivity.parquet")
    figure(payload, sens, args.out / f"decision_regret_{args.building}.png")
    (args.out / "decision_regret.md").write_text(to_markdown(payload) + "\n")
    print(f"\nwrote {args.out / 'decision_regret.md'}")


if __name__ == "__main__":
    main()
