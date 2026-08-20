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
    ("pinball_mean", "ceiling_breaches",
     "Pinball loss on the month (kW)", "Ceiling breaches (30-min blocks)",
     "A. Forecast quality buys transformer safety"),
    ("pinball_mean", "bill_inr",
     "Pinball loss on the month (kW)", "Monthly bill (INR)",
     "B. Forecast quality buys money"),
    ("calibration_error", "ceiling_breaches",
     "Calibration error (mean |nominal - empirical|)", "Ceiling breaches (30-min blocks)",
     "C. Calibration, isolated from sharpness"),
    ("pinball_mean", "usable_headroom_kw",
     "Pinball loss on the month (kW)", "Usable headroom (kW)",
     "D. ...and it buys back capacity"),
]


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


def figure(df: pd.DataFrame, meta: dict, out: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    stats = {}

    for ax, (xk, yk, xlab, ylab, title) in zip(axes.ravel(), PANELS):
        d = df.dropna(subset=[xk, yk]).sort_values(xk)
        curve = d[~d.key.isin([NOMODEL])]
        ax.plot(curve[xk], curve[yk], color="#94a3b8", lw=1.4, ls="-", zorder=2, alpha=0.9)

        for _, r in d.iterrows():
            ax.scatter(r[xk], r[yk], **STYLE.get(r["key"], DEFAULT))
            ax.annotate(r["forecaster"].replace(" (ours)", ""), (r[xk], r[yk]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8.2,
                        color="#0f172a")

        s = monotonicity(df, xk, yk)
        stats[f"{xk}__{yk}"] = s
        if s.get("rho") is not None:
            ax.text(0.98, 0.04, f"Spearman rho = {s['rho']:+.2f}  (p = {s['p']:.3f}, n = {s['n']})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec="#cbd5e1"))

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=11.5, weight="bold")
        ax.grid(alpha=0.28)

    fig.suptitle(
        f"What forecast quality is worth downstream — {meta['building']}, "
        f"{meta['window'][0][:10]} to {meta['window'][1][:10]}\n"
        f"Same optimiser, same simulator, same {meta['demand_target_kw']:.0f} kW target on "
        f"every point. Only the forecaster changes.",
        fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
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
