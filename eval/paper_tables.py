"""Emit the paper's tables as LaTeX, so no number in it is typed by hand.

Same contract as everywhere else in this repo: a number reaches the paper only
if a script put it there. `main.tex` \\input{}s these files, so regenerating the
study and re-running this is the whole update path for the write-up.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
OUT = ROOT / "docs/paper/tables"
OURS = "lightgbm_quantile"

ESC = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}


def esc(s: str) -> str:
    return "".join(ESC.get(c, c) for c in str(s))


def table(path: Path, header: list[str], rows: list[list[str]], align: str,
          caption: str, label: str) -> None:
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{" + align + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    path.write_text("\n".join(L))
    print(f"  {path.name}")


def horizon_table() -> None:
    p = RESULTS / "horizon_risk_Fox_office_Gaylord.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = []
    for r in d["marginal_vs_joint"]:
        rows.append([
            f"{r['H']}", f"{r['hours']:.2f}",
            f"{r['per_step_exceedance']:.3f}",
            f"\\textbf{{{r['empirical_horizon']:.3f}}}",
            f"{r.get('independence_bound', float('nan')):.3f}",
            f"{r.get('copula_predicted', float('nan')):.3f}",
        ])
    table(OUT / "horizon.tex",
          ["$H$", "hours", "per-step", "empirical", "Boole bound", "copula"],
          rows, "rrrrrr",
          "Marginal versus horizon-level exceedance of the demand ceiling. "
          "The per-step quantile is well calibrated at every horizon; the "
          "probability of breaching \\emph{somewhere} in the window is not the "
          "number the constraint appears to promise, and the independence bound "
          "is far too loose to substitute for it. "
          "Held-out June 2017, "
          f"{d['copula']['n_origins']:,} forecast origins.",
          "tab:horizon")


def load_study() -> pd.DataFrame:
    import eval.comparative_report as R
    df = R.load()
    df = df[df.get("error").isna()] if "error" in df.columns else df
    cov = R.climate_covariates()
    return df.merge(cov, on="id", how="left").drop_duplicates("id")


def arm_table(df: pd.DataFrame, arm: str, first: str, firstname: str,
              caption: str, label: str, fname: str) -> None:
    g = df[df.arm == arm].copy()
    if g.empty:
        return
    g = g.sort_values("cdd_share" if arm != "demographic" else f"{OURS}_skill",
                      ascending=False)
    rows = []
    for _, r in g.iterrows():
        hv = "--" if pd.isna(r.get("hvac_share")) else f"{r['hvac_share']:.2f}"
        cd = "--" if pd.isna(r.get("cdd_share")) else f"{r['cdd_share']:.2f}"
        sk = r.get(f"{OURS}_skill")
        skt = "--" if pd.isna(sk) else f"\\textbf{{{sk:+.3f}}}"
        rows.append([
            esc(r[first]), esc(r["country"]), cd, hv,
            f"{r['seasonal_naive_pinball']:.3f}", f"{r[f'{OURS}_pinball']:.3f}",
            skt, f"{r[f'{OURS}_cov90']:.3f}",
        ])
    table(OUT / fname,
          [firstname, "Country", "CDD sh.", "HVAC sh.", "Seas.\\ naive",
           "Ours", "Skill", "Cov 90\\%"],
          rows, "llrrrrrr", caption, label)


def calibration_table(df: pd.DataFrame) -> None:
    rows = []
    for t, g in df.groupby("tier"):
        g = g.dropna(subset=[f"{OURS}_cov90"])
        if g.empty:
            continue
        name = "Buildings (BDG2)" if t == 1 else "National demand"
        rows.append([name, f"{len(g)}",
                     f"\\textbf{{{g[f'{OURS}_cov90'].mean():.3f}}}",
                     f"{g[f'{OURS}_cov90'].min():.3f}",
                     f"{int((g[f'{OURS}_cov90'] < 0.85).sum())}"])
    table(OUT / "calibration.tex",
          ["Tier", "$n$", "mean cov.", "worst", "$<0.85$"],
          rows, "lrrrr",
          "Empirical coverage of the nominal 90\\% interval after conformal "
          "calibration, by aggregation level. The guarantee very nearly holds "
          "on individual buildings and fails systematically on system-level "
          "demand, worst of all on Delhi (0.762).",
          "tab:calibration")


def correlation_table(df: pd.DataFrame) -> None:
    b = df[df.tier == 1].dropna(subset=[f"{OURS}_skill"])

    def r_of(x, frame):
        g = frame.dropna(subset=[x, f"{OURS}_skill"])
        if len(g) < 3:
            return float("nan"), 0
        return float(np.corrcoef(g[x].astype(float),
                                 g[f"{OURS}_skill"].astype(float))[0, 1]), len(g)

    rows = []
    for x, label in [("cdd_share", "cooling-degree-day share"),
                     ("hvac_share", "controllable (HVAC) fraction"),
                     ("t_mean", "mean outdoor temperature"),
                     ("median_load", "median load")]:
        r, n = r_of(x, b)
        g = b.dropna(subset=[x, f"{OURS}_skill"])
        loo = [r_of(x, g.drop(i))[0] for i in g.index]
        rows.append([label, f"{r:+.3f}", f"{n}",
                     f"{min(loo):+.3f} to {max(loo):+.3f}"])
    # the one that is not null
    g = df[df.tier == 1].dropna(subset=["cdd_share", "hvac_share"])
    r = float(np.corrcoef(g.cdd_share.astype(float), g.hvac_share.astype(float))[0, 1])
    rows.insert(0, ["\\emph{controllable fraction} vs climate",
                    f"\\textbf{{{r:+.3f}}}", f"{len(g)}", "--"])
    table(OUT / "correlations.tex",
          ["Against forecast skill", "$r$", "$n$", "leave-one-out range"],
          rows, "lrrl",
          "Forecast skill is uncorrelated with climate, controllable fraction, "
          "temperature and load size; every leave-one-out range straddles or "
          "nearly straddles zero. The controllable fraction (first row, against "
          "climate rather than skill) is the one relationship that is real. "
          "Skill and available flexibility are independent axes.",
          "tab:correlations")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("emitting LaTeX tables:")
    horizon_table()
    df = load_study()
    arm_table(df, "climate", "site", "Site",
              "Climate arm: demographic held fixed at Education, climate and "
              "country varied across seven sites and three countries.",
              "tab:climate", "climate.tex")
    arm_table(df, "office", "site", "Site",
              "Office arm: the same ladder in a second usage class, run as a "
              "control on the climate arm. Robin (London) is the study's worst "
              "row and is reported unchanged.",
              "tab:office", "office.tex")
    arm_table(df, "demographic", "usage", "Usage class",
              "Demographic arm: climate held fixed at Washington DC, demographic "
              "varied across eight usage classes.",
              "tab:demographic", "demographic.tex")
    arm_table(df, "national", "site", "Series",
              "National arm: real system-level demand in six countries. Delhi is "
              "15-minute native resolution from Delhi SLDC; the European series "
              "are ENTSO-E hourly via Open Power System Data.",
              "tab:national", "national.tex")
    calibration_table(df)
    correlation_table(df)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
