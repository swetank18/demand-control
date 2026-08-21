"""The frontier: what a unit of forecast quality is worth in breaches and rupees.

Section 6.5, and the closest thing here to a research contribution. The ablation
table already shows that the forecaster matters. The frontier says *how much*,
by plotting the two axes of that table against each other and asking whether the
relationship is monotonic.

Three plots.

A. pinball loss against transformer-ceiling breaches
B. pinball loss against the monthly bill
C. calibration error against breaches

C is the one worth arguing about. A and B mix sharpness and calibration
together, so a model can move along them by being generally more accurate. C
isolates calibration, and calibration is the thing a chance constraint actually
consumes: a forecaster can have a fine average error and still be systematically
overconfident, and it is overconfidence, not error, that puts load through a
ceiling.

If the curves are monotonic we have measured an exchange rate between model
quality and money. If they are flat that is also a finding and it gets reported
as one -- see the risk register. Either way the plot is made from the ablation
JSON and nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

OURS = "lightgbm_quantile"
CEILING = "perfect_foresight"
NOMODEL = "static_margin"

STYLE = {
    OURS: dict(color="#c1440e", s=190, marker="*", zorder=6, edgecolor="k", linewidth=0.7),
    CEILING: dict(color="#2a9d8f", s=120, marker="D", zorder=5, edgecolor="k", linewidth=0.6),
    NOMODEL: dict(color="#6b7280", s=130, marker="s", zorder=5, edgecolor="k", linewidth=0.6),
}
DEFAULT = dict(color="#2f6690", s=90, marker="o", zorder=4, edgecolor="k", linewidth=0.5)

PANELS = [
    dict(x="pinball_mean", y="ceiling_breaches",
         xlab="Pinball loss on the month (kW)", ylab="Ceiling breaches (30-min blocks)",
         title="A. Forecast quality buys transformer safety",
         ylog=True, drop=()),
    dict(x="pinball_mean", y="bill_inr",
         xlab="Pinball loss on the month (kW)", ylab="Monthly bill (INR)",
         title="B. Forecast quality buys money",
         ylog=False, drop=()),
    # Perfect foresight is dropped here, not hidden: every quantile equals the
    # actual, so its "calibration error" is 0.5 by arithmetic and plotting it
    # would put a meaningless point at the far right of the axis.
    dict(x="calibration_error", y="ceiling_breaches",
         xlab="Calibration error (mean |nominal - empirical|)",
         ylab="Ceiling breaches (30-min blocks)",
         title="C. Calibration, isolated from sharpness",
         ylog=True, drop=(CEILING,)),
    dict(x="pinball_mean", y="usable_headroom_kw",
         xlab="Pinball loss on the month (kW)", ylab="Usable headroom (kW)",
         title="D. Capacity claimed — but only safety makes it real",
         ylog=False, drop=()),
]

#: Panel D does not come out monotonic and that is the finding, not a defect.
#: Persistence claims more headroom than climatology because its q95 simply
#: tracks the last observed value, so it hands the optimiser capacity that is not
#: there -- and then breaches 87 times using it. Headroom is only a benefit at
#: equal safety, which is what the static-margin study in eval/impact.py measures
#: and what this panel deliberately does not.
PANEL_D_NOTE = (
    "Panel D is not monotonic, and that is the finding. Headroom without safety is not "
    "a benefit: persistence claims more capacity than climatology because its q95 simply "
    "follows the last observed value, and it then breaches the ceiling 87 times spending "
    "capacity that was never there. The like-for-like comparison is the static-margin "
    "study in eval/impact.py, which matches breach counts first and compares headroom second."
)


def load(building: str, stress: str = "none") -> pd.DataFrame:
    payload = json.loads((RESULTS / f"ablation_{building}.json").read_text())
    return pd.DataFrame(payload[stress]["rows"]), payload[stress]["meta"]


def monotonicity(df: pd.DataFrame, x: str, y: str) -> dict:
    """Spearman rank correlation over the rows that are genuinely forecasters.

    The static margin is excluded: it is a constant, so its 'pinball loss' is
    not a forecast score and including it would manufacture the correlation we
    are trying to test for. Perfect foresight is excluded for the same reason
    from the other end -- it is a bound, not a model.
    """
    sub = df[~df.key.isin([NOMODEL, CEILING])]
    if len(sub) < 3:
        return {"n": int(len(sub)), "rho": None, "p": None}
    rho, p = spearmanr(sub[x], sub[y])
    return {"n": int(len(sub)), "rho": float(rho), "p": float(p),
            "monotonic": bool(abs(rho) > 0.6 and p < 0.10)}


#: Vertical offsets, in points, cycled by plotting order. Four lanes rather than
#: two because on panels A and C every competent forecaster lands on the same y
#: (zero breaches) within a few kW of the next, and two lanes still overprint.
LABEL_LANES = (8, -16, 20, -28)


def _place(ax, x, y, text, i: int, n: int) -> None:
    """Annotate without collisions, by cycling the offset around the point.

    Not a general label-repel algorithm -- there are eight points and a deck to
    build. Cycling the vertical offset by index, and flipping the last two to the
    left so they do not run off the axis, is enough to separate the cluster of
    good forecasters that otherwise print on top of one another.
    """
    dx, dy = 9, LABEL_LANES[i % len(LABEL_LANES)]
    ha = "left"
    if i >= n - 2:                       # rightmost points would run off the axis
        dx, ha = -9, "right"
    # points sitting on the floor of the axis (zero breaches, and there are five
    # of them) have no room below, so those lanes are folded upward
    frac = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))[1]
    if frac < 0.18 and dy < 0:
        dy = -dy + 14            # into lanes of its own, not on top of 8 and 20
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8.4, color="#0f172a", ha=ha,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72))


def figure(df: pd.DataFrame, meta: dict, out: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))
    stats = {}

    for ax, spec in zip(axes.ravel(), PANELS):
        xk, yk = spec["x"], spec["y"]
        d = df[~df.key.isin(spec["drop"])].dropna(subset=[xk, yk]).sort_values(xk)

        # the trend line runs through the forecasters only; the static margin is
        # not on the same axis of variation and would bend the line by itself
        curve = d[~d.key.isin([NOMODEL])]
        ax.plot(curve[xk], curve[yk], color="#cbd5e1", lw=1.3, zorder=2)

        n = len(d)
        for i, (_, r) in enumerate(d.reset_index(drop=True).iterrows()):
            ax.scatter(r[xk], r[yk], **STYLE.get(r["key"], DEFAULT))
            _place(ax, r[xk], r[yk], r["forecaster"].replace(" (ours)", ""), i, n)

        if spec["ylog"]:
            # breaches run 0 to 87; a linear axis collapses everything below ten
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(-0.35, max(120, d[yk].max() * 1.6))
            ax.set_yticks([0, 1, 10, 100])
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        if not spec["ylog"]:
            ylo, yhi = float(d[yk].min()), float(d[yk].max())
            ypad = 0.10 * (yhi - ylo or 1.0)
            ax.set_ylim(ylo - ypad, yhi + ypad)

        lo, hi = float(d[xk].min()), float(d[xk].max())
        pad = 0.14 * (hi - lo or 1.0)
        ax.set_xlim(lo - pad, hi + pad * 1.6)

        s = monotonicity(df[~df.key.isin(spec["drop"])], xk, yk)
        stats[f"{xk}__{yk}"] = s
        ax.set_xlabel(spec["xlab"])
        ax.set_ylabel(spec["ylab"])
        ax.set_title(spec["title"], loc="left", fontsize=11.5, weight="bold", pad=22)
        ax.grid(alpha=0.26)

        # The statistic sits above the axes rather than inside them. Anywhere
        # inside collides with a point label on at least one of the four panels,
        # and a legend box parked on top of the data is how a plot stops being read.
        if s.get("rho") is not None:
            verdict = "monotonic" if s.get("monotonic") else "not monotonic"
            ax.text(0.0, 1.012,
                    f"Spearman rho = {s['rho']:+.2f}    p = {s['p']:.3f}    n = {s['n']}    ({verdict})",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8,
                    color="#475569")

    fig.text(0.5, 0.017, textwrap.fill(PANEL_D_NOTE, 132), ha="center", va="bottom",
             fontsize=8.6, color="#7f1d1d", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec="#fecaca"))

    fig.suptitle(
        f"What forecast quality is worth downstream — {meta['building']}, "
        f"{meta['window'][0][:10]} to {meta['window'][1][:10]}\n"
        f"Same optimiser, same simulator, same {meta['demand_target_kw']:.0f} kW target on "
        f"every point. Only the forecaster changes.",
        fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0.005, 0.075, 0.995, 0.952))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return stats


def exchange_rate(df: pd.DataFrame) -> dict:
    """Rupees and breaches per unit of pinball loss, fitted across the forecasters.

    A slope, not a correlation, so it can be quoted: 'one kW of pinball loss
    costs this much per month on this building'. Stated with its own caveat --
    it is a within-building, within-month fit over six points, so it is an
    order of magnitude, not a constant of nature.
    """
    sub = df[~df.key.isin([NOMODEL, CEILING])].dropna(subset=["pinball_mean"])
    out = {}
    for target in ("bill_inr", "ceiling_breaches", "usable_headroom_kw"):
        if len(sub) >= 3:
            slope, intercept = np.polyfit(sub["pinball_mean"], sub[target], 1)
            pred = slope * sub["pinball_mean"] + intercept
            ss_res = float(((sub[target] - pred) ** 2).sum())
            ss_tot = float(((sub[target] - sub[target].mean()) ** 2).sum())
            out[target] = {
                "per_unit_pinball": float(slope),
                "intercept": float(intercept),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else None,
                "n": int(len(sub)),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    ap.add_argument("--stress", default="none")
    ap.add_argument("--out", type=Path, default=FIGS / "06_model_frontier.png")
    args = ap.parse_args()

    df, meta = load(args.building, args.stress)
    stats = figure(df, meta, args.out)
    rates = exchange_rate(df)

    payload = {"building": args.building, "stress": args.stress, "meta": meta,
               "monotonicity": stats, "exchange_rate": rates,
               "panel_d_note": PANEL_D_NOTE,
               "figure": str(args.out.relative_to(ROOT))}
    (RESULTS / "model_frontier.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{args.building} | {args.stress}")
    for k, v in stats.items():
        x, y = k.split("__")
        if v.get("rho") is not None:
            verdict = "monotonic" if v.get("monotonic") else "not clearly monotonic"
            print(f"  {x:>18} -> {y:<22} rho {v['rho']:+.2f}  p {v['p']:.3f}   {verdict}")
    for k, v in rates.items():
        print(f"  d({k})/d(pinball) = {v['per_unit_pinball']:+,.1f} per kW   (R2 {v['r2']:.2f}, n={v['n']})")
    print(f"  figure -> {args.out}")


if __name__ == "__main__":
    main()
