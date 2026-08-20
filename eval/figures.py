"""Static figures for the deck, and the fallback if the UI dies on stage.

Reads only from results/, so these cannot disagree with the table.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
B = "Fox_office_Gaylord"
OURS = "Ours: quantile + chance constrained"
MEAN = "MPC on mean forecast"
BASE = "No control"
COLORS = {BASE: "#9aa5b1", MEAN: "#2f6690", OURS: "#c1440e"}


def hist(stress: str, name: str) -> pd.DataFrame:
    f = RESULTS / f"history_{B}_{stress}_{name.replace(' ', '_').replace(':', '').replace('+', '')}.parquet"
    h = pd.read_parquet(f)
    h.index = pd.to_datetime(h.index)
    return h


def ceiling_figure(stress: str, target: float, title: str, out: Path, zoom=None) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for name in (BASE, MEAN, OURS):
        blk = hist(stress, name)["grid_kw"].resample("30min").mean()
        if zoom:
            blk = blk.loc[zoom[0]:zoom[1]]
        ax.plot(blk.index, blk.values, lw=1.3, color=COLORS[name], label=name)
        over = blk[blk > target]
        if len(over):
            ax.scatter(over.index, over.values, s=60, color=COLORS[name], zorder=5,
                       marker="v", edgecolor="k", linewidth=0.6)
    ax.axhline(target, color="#d00000", ls="--", lw=2.2)
    ax.annotate("demand ceiling", xy=(0.012, target), xycoords=("axes fraction", "data"),
                color="#d00000", va="bottom", ha="left", fontsize=10, weight="bold")
    ax.set_ylabel("grid import (kW, 30-min billing blocks)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.name}")


def frontier_figure(out: Path) -> None:
    d = pd.DataFrame(json.loads((RESULTS / "frontier.json").read_text()))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(d.comfort_ceiling_c, d.saving_inr / 1000, "o-", color="#c1440e", lw=2.2, ms=8)
    ax1.set_xlabel("occupied comfort ceiling (°C)")
    ax1.set_ylabel("monthly saving vs no control (₹ thousands)", color="#c1440e")
    ax1.tick_params(axis="y", labelcolor="#c1440e")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.step(d.comfort_ceiling_c, d.target_kw, where="mid", color="#1b4965", lw=2, ls="--")
    ax2.set_ylabel("demand ceiling the system will commit to (kW)", color="#1b4965")
    ax2.tick_params(axis="y", labelcolor="#1b4965")
    ax1.set_title("Comfort is the lever, and it belongs to the operator")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.name}")


def mv_figure(out: Path) -> None:
    d = pd.DataFrame(json.loads((RESULTS / "mv_baseline_length.json").read_text()))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(d))
    lo = d["reported"] - d["ci"].str[0]
    hi = d["ci"].str[1] - d["reported"]
    ax.errorbar(x, d["reported"] / 1000, yerr=[lo / 1000, hi / 1000], fmt="o",
                color="#1b4965", capsize=6, ms=9, lw=2, label="reported saving (95% CI)")
    ax.axhline(d["true"].iloc[0] / 1000, color="#c1440e", lw=2.2, ls="--",
               label=f"true saving ₹{d['true'].iloc[0]:,.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels(d["baseline"])
    ax.set_xlabel("baseline period used for measurement and verification")
    ax.set_ylabel("saving (₹ thousands)")
    ax.set_title("Below twelve months of baseline, M&V under-reports and its error bars miss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    res = pd.read_csv(RESULTS / f"results_{B}.csv")
    target = float(res["demand_target_kw"].dropna().iloc[0])
    figs = RESULTS / "figures"
    figs.mkdir(exist_ok=True)
    ceiling_figure("none", target, "Normal month — ours holds the ceiling, the mean-forecast controller does not",
                   figs / "01_normal_month.png")
    ceiling_figure("heatwave", target,
                   "Heatwave: +6 °C and base load +25% for three days",
                   figs / "02_heatwave.png")
    ceiling_figure("heatwave", target,
                   "Heatwave, 14–19 June — every triangle sets the demand charge for the whole month",
                   figs / "03_heatwave_zoom.png", zoom=("2017-06-14", "2017-06-19"))
    frontier_figure(figs / "04_comfort_frontier.png")
    mv_figure(figs / "05_mv_baseline_length.png")


if __name__ == "__main__":
    main()
