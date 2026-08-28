"""Emit the paper's figures as vector PDF, on the same contract as the tables.

Every figure here is drawn from a results JSON. Nothing is hand-placed, nothing
is traced from a screenshot, and re-running this after a new study is the whole
update path for the artwork. `main.tex` \\includegraphics{}es what this writes.

Four figures, one per result:

  fig_horizon      the marginal-versus-horizon bracket, which is the paper's
                   headline and which a six-row table understates
  fig_calibration  coverage by aggregation level, and the China replication
  fig_null         the null, and the one relationship that is not null
  fig_aci          the remedy: adaptive conformal holding the level across a
                   walk-forward year that split conformal does not

House style is deliberately plain: serif to sit with the body text, no grid
except a hairline where a reader has to compare across a wide axis, no colour
carrying meaning that the shape does not also carry, and every axis starting
where the quantity starts rather than where the data happens to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
OUT = ROOT / "docs/paper/figures"
OURS = "lightgbm_quantile"

INK = "#1a1a1a"
MID = "#6b6b6b"
LIGHT = "#b8b8b8"
ACCENT = "#a02c2c"          # India, and the row a reader must not miss
COOL = "#2c5a7a"            # the China panel

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.grid": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.pdf")


# --------------------------------------------------------------------------
def fig_horizon() -> None:
    """The bracket. A table of six rows makes this look like a list of numbers;
    drawn, it is one gap opening up and two substitutes missing it in opposite
    directions, which is the actual claim."""
    p = RESULTS / "horizon_risk_Fox_office_Gaylord.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    m = d["marginal_vs_joint"]
    H = np.array([r["H"] for r in m], float)
    per = np.array([r["per_step_exceedance"] for r in m])
    emp = np.array([r["empirical_horizon"] for r in m])
    ind = np.array([r["independence_bound"] for r in m])
    cop = np.array([r["copula_predicted"] for r in m])
    boole = np.minimum(1.0, per * H)

    fig, ax = plt.subplots(figsize=(5.2, 3.5))

    # The band is the paper's first sentence drawn: knowing every marginal is
    # calibrated pins the horizon risk only to somewhere in here.
    ax.fill_between(H, 0.05, ind, color=LIGHT, alpha=0.20, lw=0)
    ax.annotate("what marginal calibration\nalone leaves undetermined",
                xy=(1.12, 0.80), fontsize=6.8, color=MID, ha="left", va="center",
                linespacing=1.3)

    ax.plot(H, boole, color=LIGHT, ls=(0, (1, 1.6)), lw=1.1,
            label=r"Boole $\min(1,\,H\hat\alpha)$ — valid, vacuous")
    ax.plot(H, ind, color=MID, ls="--", lw=1.1, marker="s", ms=2.8,
            label=r"independence $1-(1-\hat\alpha)^{H}$ — not a bound")
    ax.plot(H, cop, color=COOL, ls="-.", lw=1.1, marker="^", ms=3.0,
            label="copula, predicted")
    ax.plot(H, emp, color=INK, lw=1.8, marker="o", ms=3.6,
            label="realised horizon exceedance")
    ax.plot(H, per, color=ACCENT, lw=1.2, marker="D", ms=2.8,
            label=r"realised per-step $\hat\alpha$ (nominal $0.05$)")
    ax.axhline(0.05, color=ACCENT, ls=":", lw=0.9)

    for val, y_off, col, weight in ((ind[-1], 6, MID, "normal"),
                                    (emp[-1], -13, INK, "bold"),
                                    (cop[-1], 7, COOL, "normal")):
        ax.annotate(f"{val:.3f}", xy=(H[-1], val), xytext=(-4, y_off),
                    textcoords="offset points", ha="right", fontsize=7.2,
                    color=col, fontweight=weight)
    ax.annotate("per-step calibration is not the problem: "
                f"$\\hat\\alpha$ holds at {per.min():.3f}–{per.max():.3f} throughout",
                xy=(1.05, 0.010), fontsize=6.8, color=ACCENT, ha="left")

    ax.set_xscale("log", base=2)
    ax.set_xticks(H)
    ax.set_xticklabels([f"{int(h)}\n{r['hours']:g} h" for h, r in zip(H, m)])
    ax.set_xlim(0.9, 74)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("planning horizon $H$ (15-minute steps)")
    ax.set_ylabel("P(ceiling exceeded somewhere in the window)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False,
              ncol=2, handlelength=2.6, labelspacing=0.35, columnspacing=1.6,
              borderpad=0.0)
    _despine(ax)
    _save(fig, "fig_horizon")


# --------------------------------------------------------------------------
def _study():
    import eval.comparative_report as R
    df = R.load()
    df = df[df.get("error").isna()] if "error" in df.columns else df
    return df.merge(R.climate_covariates(), on="id", how="left").drop_duplicates("id")


def fig_calibration(df) -> None:
    """One dot per supply. The tier effect is the whole finding, so the figure
    is built to make a reader see three populations rather than read three
    means."""
    pops = [
        ("Buildings\n(BDG2, metered)", df[df.tier == 1], INK),
        ("System demand\n(metered)",
         df[(df.tier == 2) & (~df.reconstructed.astype(bool))], ACCENT),
        ("System demand\n(reconstructed)",
         df[(df.tier == 2) & (df.reconstructed.astype(bool))], COOL),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.axvspan(0.50, 0.85, color=LIGHT, alpha=0.20, lw=0)
    ax.axvline(0.90, color=INK, lw=0.9, ls="--")
    ax.annotate("nominal 0.90", xy=(0.905, 2.72), fontsize=6.8, color=INK,
                ha="left")
    ax.annotate("below 0.85", xy=(0.70, 2.72), fontsize=6.8, color=MID,
                ha="center")

    rng = np.random.default_rng(0)
    for i, (name, g, col) in enumerate(pops):
        c = g[f"{OURS}_cov90"].dropna().to_numpy()
        y = i + rng.uniform(-0.15, 0.15, len(c))
        ax.scatter(c, y, s=18, facecolor="none", edgecolor=col, linewidths=0.9,
                   zorder=3, clip_on=False)
        ax.plot([c.mean(), c.mean()], [i - 0.26, i + 0.26], color=col, lw=1.8,
                zorder=4)
        # The means go in a column of their own on the right, where they can be
        # read off against each other instead of hunting for them in the cloud.
        ax.text(1.005, i, f"mean {c.mean():.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.2, color=col)

    for label, gid, i in (("Delhi", "IN_Delhi", 1),
                          ("Heilongjiang", "CN_Heilongjiang", 2)):
        row = df[df.id == gid]
        if row.empty:
            continue
        x = float(row[f"{OURS}_cov90"].iloc[0])
        ax.annotate(label, xy=(x, i - 0.16), xytext=(x, i - 0.52), ha="center",
                    fontsize=6.8, color=pops[i][2],
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=pops[i][2]))

    ax.set_yticks(range(len(pops)))
    ax.set_yticklabels([p[0] for p in pops], linespacing=1.3)
    ax.set_ylim(-0.75, 3.05)  # buildings at the bottom, aggregation upward
    ax.set_xlim(0.53, 0.99)
    ax.set_xlabel("empirical coverage of the nominal 90% interval")
    _despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_calibration")


def fig_null(df) -> None:
    """Three scatters. Two are the null and its replication; the third is the
    relationship that is real, drawn at the same size so the contrast is the
    figure rather than a sentence about the figure."""
    b = df[df.tier == 1].dropna(subset=[f"{OURS}_skill", "cdd_share"])
    cn = df[(df.tier == 2) & (df.reconstructed.astype(bool))].dropna(
        subset=[f"{OURS}_skill", "cdd_share"])
    bh = df[df.tier == 1].dropna(subset=["hvac_share", "cdd_share"])

    panels = [
        (b, "cdd_share", f"{OURS}_skill", INK,
         "(a) 18 buildings", "cooling-degree-day share", "forecast skill"),
        (cn, "cdd_share", f"{OURS}_skill", COOL,
         "(b) 6 Chinese provinces", "cooling-degree-day share", "forecast skill"),
        (bh, "cdd_share", "hvac_share", ACCENT,
         "(c) the same 18 buildings", "cooling-degree-day share",
         "controllable (HVAC) fraction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    for j, (ax, (g, xk, yk, col, title, xl, yl)) in enumerate(zip(axes, panels)):
        x = g[xk].astype(float).to_numpy()
        y = g[yk].astype(float).to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=19, facecolor="none", edgecolor=col, linewidths=0.9)
        xs = np.linspace(0, 1, 2)
        k, c = np.polyfit(x, y, 1)
        # Solid where the relationship is real, dashed where the fit is being
        # drawn only so a reader can see that it is flat.
        ax.plot(xs, k * xs + c, color=col, lw=1.0,
                ls="-" if abs(r) > 0.4 else (0, (3, 2)))
        ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
        ax.set_title(f"{title}\n$r$ = {r:+.3f},  $n$ = {len(g)}", pad=5,
                     linespacing=1.4, color=ACCENT if j == 2 else INK)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-0.04, 1.04)
        # (a) and (b) share a y-axis so the two nulls are read on one scale;
        # (c) measures a different quantity and keeps its own.
        if j < 2:
            ax.set_ylim(-0.55, 0.72)
        _despine(ax)
    fig.subplots_adjust(wspace=0.40)
    _save(fig, "fig_null")


# --------------------------------------------------------------------------
def fig_aci() -> None:
    """The remedy, at building scale. Split conformal is the layer with the
    theorem; the theorem's hypothesis is what breaks, and this is the twelve
    months in which it breaks and the adaptive layer does not."""
    p = RESULTS / "conformal_audit_Fox_office_Gaylord.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    m = d["year"]["by_month"]
    x = np.arange(len(m))
    lab = [r["month"][2:] for r in m]

    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.axhspan(0.85, 0.95, color=LIGHT, alpha=0.22, lw=0)
    ax.axhline(0.90, color=INK, lw=0.8, ls="--")

    for key, col, ls, mk, name in (
            ("raw_cov90", LIGHT, (0, (1, 1.6)), "s", "raw LightGBM quantiles"),
            ("split_cov90", MID, "--", "^", "split conformal"),
            ("aci_cov90", ACCENT, "-", "o", "split + adaptive (ACI)")):
        y = [r[key] for r in m]
        ax.plot(x, y, color=col, ls=ls, lw=1.3 if key == "aci_cov90" else 1.0,
                marker=mk, ms=3.0, label=name)

    band = d["year"]["band_90"]
    ax.set_title(
        "share of the year inside 0.85\u20130.95:      "
        f"raw {band['raw']['in_band_pct']:.0f}%      "
        f"split conformal {band['split']['in_band_pct']:.0f}%      "
        f"ACI {band['aci']['in_band_pct']:.0f}%",
        loc="left", fontsize=7.2, color=MID, pad=6)

    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=45, ha="right")
    ax.set_ylim(0.20, 1.02)
    ax.set_ylabel("coverage of the\nnominal 90% interval", linespacing=1.3)
    ax.set_xlabel("walk-forward test month (trained strictly on the past)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), frameon=False,
              ncol=3, handlelength=2.4, labelspacing=0.3, borderpad=0.0,
              columnspacing=1.8)
    _despine(ax)
    _save(fig, "fig_aci")


def main() -> None:
    print("emitting figures:")
    fig_horizon()
    df = _study()
    fig_calibration(df)
    fig_null(df)
    fig_aci()
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
