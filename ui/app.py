"""Replay dashboard. One screen, four scenarios, a running rupee counter.

Deliberately not a product: no settings page, no login, no navigation. The only
job is to make the demo legible from the back of a room, and to do it fast --
a live demo that recomputes a month of MILP solves on every click is a demo that
dies on stage. Runs are precomputed by ``eval/table.py`` and loaded from disk;
the UI recomputes only if a scenario is missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tariff.bill import compute_bill, price_series
from tariff.schema import Tariff

st.set_page_config(page_title="Tariff-native demand control", layout="wide")

RESULTS = ROOT / "results"
OURS = "Ours: quantile + chance constrained"
BASE = "No control"
MEAN = "MPC on mean forecast"
ORACLE = "Perfect foresight oracle"

COLORS = {
    BASE: "#9aa5b1",
    "Rule based schedule": "#7a5195",
    MEAN: "#2f6690",
    OURS: "#c1440e",
    "MPC on perfect forecast (16 h)": "#7b9e89",
    ORACLE: "#2a9d8f",
}
SCENARIOS = {
    "none": "Normal month",
    "heatwave": "Heatwave  (+6 °C, base load +25%, 3 days)",
    "sensor_dropout": "Sensor dropout  (forecast frozen 2 h)",
    "outage": "Grid outage  (2 h, critical load only)",
}


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace(":", "").replace("+", "")


@st.cache_data(show_spinner=False)
def load_results(building: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RESULTS / f"results_{building}.csv")
    hist: dict[tuple[str, str], pd.DataFrame] = {}
    for stress in SCENARIOS:
        for name in COLORS:
            f = RESULTS / f"history_{building}_{stress}_{_slug(name)}.parquet"
            if f.exists():
                h = pd.read_parquet(f)
                h.index = pd.to_datetime(h.index)
                hist[(stress, name)] = h
    return df, hist


@st.cache_data(show_spinner=False)
def load_tariff(building: str) -> Tariff:
    import dataclasses

    t = Tariff.load(ROOT / "tariff/orders/tnerc_2026.json")
    man = json.loads((ROOT / "data/cache/manifest.json").read_text())
    return dataclasses.replace(
        t, contract_demand_kva=man["buildings"][building]["contract_demand_kva"]
    )


BUILDING = "Fox_office_Gaylord"
try:
    results, histories = load_results(BUILDING)
except FileNotFoundError:
    st.error("No precomputed results. Run:  python eval/table.py --stress none heatwave sensor_dropout outage")
    st.stop()
tariff = load_tariff(BUILDING)
target = float(results["demand_target_kw"].dropna().iloc[0])

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Demand control")
st.sidebar.caption(f"{BUILDING} · June 2017 · {tariff.state} {tariff.category}")
stress = st.sidebar.radio("Scenario", list(SCENARIOS), format_func=SCENARIOS.get)
available = [n for n in COLORS if (stress, n) in histories]
show = st.sidebar.multiselect("Controllers on the chart", available,
                              [n for n in (BASE, MEAN, OURS) if n in available])
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Demand ceiling **{target:.0f} kW**, found by bisection as the tightest "
    "target the controller holds for a whole month inside a 2% comfort budget. "
    "The comfort band is an operator input, not something the optimiser chooses."
)

rows = results[results.stress == stress].set_index("controller")
ours, base = rows.loc[OURS], rows.loc[BASE]
mean_row = rows.loc[MEAN] if MEAN in rows.index else None

st.title("We do not forecast electricity. We forecast the bill.")
st.caption(
    f"Peak window 18:00–22:00 at {tariff.energy_rate * 1.25:.2f} ₹/kWh · demand charge "
    f"₹{tariff.demand_charge_per_kva:,.0f}/kVA/month on the highest "
    f"{tariff.billing_interval_minutes}-minute block of the month"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bill, ours", f"₹{ours.bill_inr:,.0f}", f"{ours.bill_inr - base.bill_inr:,.0f} vs no control")
c2.metric("Peak demand", f"{ours.peak_kva:.0f} kVA", f"{ours.peak_kva - base.peak_kva:+.0f} kVA")
c3.metric("Ceiling breaches", f"{int(ours.ceiling_breaches)}",
          None if mean_row is None else f"{int(ours.ceiling_breaches - mean_row.ceiling_breaches):+d} vs mean-forecast MPC",
          delta_color="inverse")
c4.metric("Comfort violated", f"{ours.comfort_violation_pct:.2f}%", "budget 2.00%", delta_color="off")

# ---------------------------------------------------------------- replay
idx = histories[(stress, OURS)].index
st.markdown("#### Month replay")
pos = st.slider("Replay to", 0, len(idx) - 1, len(idx) - 1, 4, format="")
upto = idx[: pos + 1]
st.caption(f"showing up to **{idx[pos]:%d %B, %H:%M}**")

fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
rule = f"{tariff.billing_interval_minutes}min"

ax = axes[0]
for name in show:
    blk = histories[(stress, name)].loc[upto, "grid_kw"].resample(rule).mean()
    ax.plot(blk.index, blk.values, lw=1.1, color=COLORS[name], label=name)
    over = blk[blk > target]
    if len(over):
        ax.scatter(over.index, over.values, s=26, color=COLORS[name], zorder=5,
                   marker="v", edgecolor="k", linewidth=0.4)
ax.axhline(target, color="#d00000", ls="--", lw=2)
ax.text(idx[0], target, "  demand ceiling", color="#d00000", va="bottom", fontsize=9, weight="bold")
ax.set_ylabel("grid import (kW, billing blocks)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.25)
ax.set_title("Every triangle above the red line sets the demand charge for the whole month")

ax = axes[1]
floor = tariff.contract_demand_kva * tariff.billing_demand_floor_pct / 100.0
for name in show:
    h = histories[(stress, name)].loc[upto]
    energy = (h["grid_kw"].clip(lower=0) * 0.25 * price_series(h.index, tariff)).cumsum()
    blk = h["grid_kw"].resample(rule).mean()
    running_peak = (blk / 0.95).cummax().reindex(h.index, method="ffill")
    demand = np.maximum(running_peak, floor) * tariff.demand_charge_per_kva
    ax.plot(h.index, energy + demand, lw=1.5, color=COLORS[name], label=name)
ax.set_ylabel("running bill (₹)")
ax.grid(alpha=0.25)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}L"))

ax = axes[2]
h0 = histories[(stress, OURS)].loc[upto]
ax.fill_between(h0.index, h0["t_lo"], h0["t_hi"], color="#8ecae6", alpha=0.30, label="comfort band")
for name in show:
    h = histories[(stress, name)].loc[upto]
    ax.plot(h.index, h["t_indoor"], lw=0.9, color=COLORS[name])
    bad = h[h["comfort_violation_k"] > 0.1]
    if len(bad):
        ax.scatter(bad.index, bad["t_indoor"], s=10, color=COLORS[name], marker="x")
ax.set_ylabel("indoor °C")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
st.pyplot(fig)

# ---------------------------------------------------------------- table
st.markdown("#### Results — every number from the bill engine")
tbl = results[results.stress == stress][[
    "controller", "bill_inr", "energy_charge", "demand_charge", "peak_kva",
    "ceiling_breaches", "comfort_violation_pct", "pct_of_oracle_savings",
]].copy()
tbl.columns = ["Controller", "Bill ₹", "Energy ₹", "Demand ₹", "Peak kVA",
               "Ceiling breaches", "Comfort violated %", "% of oracle savings"]
for c in ["Bill ₹", "Energy ₹", "Demand ₹"]:
    tbl[c] = tbl[c].round(0).astype(int)
st.dataframe(tbl.round(2), width='stretch', hide_index=True)

col1, col2 = st.columns(2)
with col1:
    p = RESULTS / f"calibration_{BUILDING}.png"
    if p.exists():
        st.markdown("#### Calibration is the safety property")
        st.image(str(p))
        st.caption(
            "The controller substitutes q95 into the demand-ceiling constraint. "
            "If q95 is not really a 95th percentile, the guarantee is theatre — so it is measured."
        )
with col2:
    mv = RESULTS / "mv_baseline_length.json"
    if mv.exists():
        st.markdown("#### Would you believe the saving?")
        d = pd.DataFrame(json.loads(mv.read_text()))
        st.dataframe(
            pd.DataFrame({
                "Baseline period": d["baseline"],
                "Reported saving ₹": d["reported"].round(0).astype(int),
                "% of truth": d["pct_of_truth"].round(0).astype(int),
                "CI covers truth": np.where(d["inside"], "yes", "NO"),
            }),
            width='stretch', hide_index=True,
        )
        st.caption(
            f"True saving ₹{d['true'].iloc[0]:,.0f}. With less than twelve months of baseline, "
            "measurement and verification under-reports the saving and its error bars miss the truth."
        )

fr = RESULTS / "frontier.json"
if fr.exists():
    st.markdown("#### Comfort is the lever, and it is yours")
    d = pd.DataFrame(json.loads(fr.read_text()))
    st.dataframe(
        pd.DataFrame({
            "Comfort ceiling °C": d["comfort_ceiling_c"],
            "Band (K)": d["band_width_k"],
            "Demand target kW": d["target_kw"].round(0).astype(int),
            "Saving ₹": d["saving_inr"].round(0).astype(int),
            "Comfort violated %": d["comfort_violation_pct"].round(2),
        }),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Widening the occupied ceiling from 24 °C to 28 °C nearly doubles the saving. "
        "Violations fall as the band widens: more room to manoeuvre means the controller "
        "is cornered less often."
    )

with st.expander("What is real and what is not"):
    f = ROOT / "docs/limitations.md"
    st.markdown(f.read_text() if f.exists() else "See README.")
