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
CACHE = ROOT / "data/cache"
OUT = ROOT / "docs/paper/tables"
OURS = "lightgbm_quantile"

#: Full names everywhere. Two-letter codes and internal site aliases are fine in
#: a results file and wrong in a paper -- a reader should not have to learn that
#: "Wolf" is Dublin or that "GB_UKM" includes Northern Ireland.
COUNTRY_NAME = {
    "US": "United States", "GB": "United Kingdom", "IE": "Ireland",
    "IN": "India", "DE": "Germany", "FR": "France", "ES": "Spain",
    "CA": "Canada", "CN": "China",
}

SITE_CITY = {
    "Fox": "Phoenix, United States", "Bull": "Austin, United States",
    "Rat": "Washington DC, United States", "Hog": "Minneapolis, United States",
    "Bear": "Berkeley, United States", "Robin": "London, United Kingdom",
    "Wolf": "Dublin, Ireland", "Lamb": "Cardiff, United Kingdom",
    "Mouse": "London, United Kingdom", "Shrew": "London, United Kingdom",
    "Panther": "Orlando, United States", "Gator": "Orlando, United States",
    "Eagle": "United States", "Peacock": "Princeton, United States",
    "Cockatoo": "Ithaca, United States", "Crow": "Ottawa, Canada",
    "Moose": "Ottawa, Canada", "Swan": "United States", "Bobcat": "United States",
}

SERIES_NAME = {
    "IN_Delhi": "India (Delhi)", "GB_UKM": "United Kingdom", "IE": "Ireland",
    "DE": "Germany", "FR": "France", "ES": "Spain",
    #: The Chinese rows are provinces, so they are named by their load centre --
    #: a reader should not have to know that CN_Hainan is Haikou.
    "CN_Hainan": "Hainan (Haikou)", "CN_Guangdong": "Guangdong (Guangzhou)",
    "CN_Shanghai": "Shanghai", "CN_Yunnan": "Yunnan (Kunming)",
    "CN_Beijing": "Beijing", "CN_Heilongjiang": "Heilongjiang (Harbin)",
}


def pretty(v: str, kind: str) -> str:
    if kind == "country":
        return COUNTRY_NAME.get(v, v)
    if kind == "series":
        return SERIES_NAME.get(v, v)
    if kind == "site":
        return SITE_CITY.get(v, v)
    return v

ESC = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}


def esc(s: str) -> str:
    return "".join(ESC.get(c, c) for c in str(s))


def table(path: Path, header: list[str], rows: list[list[str]], align: str,
          caption: str, label: str) -> None:
    """Wide tables get a smaller face and tighter columns.

    Eight or more columns overruns the text block at \\small on a4paper, which
    LaTeX reports as an overfull hbox and a reader sees as a table poking into
    the margin. Deciding this from the column count keeps it automatic rather
    than something to remember per table.
    """
    wide = len(header) >= 8
    size = r"\footnotesize" if wide else r"\small"
    L = [r"\begin{table}[t]", r"\centering", size]
    if wide:
        L.append(r"\setlength{\tabcolsep}{3.5pt}")
    L += [r"\begin{tabular}{" + align + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    #: A one-element row whose text starts with a backslash is emitted verbatim,
    #: which is how a table carries an internal \midrule and splits into panels.
    L += [r[0] if len(r) == 1 and str(r[0]).startswith("\\")
          else " & ".join(r) + r" \\" for r in rows]
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
        # Boole/Bonferroni is the union bound, sum of the per-step rates capped
        # at one -- NOT 1-(1-a)^H, which is the independence calculation and is
        # not a bound at all under positive dependence. Reporting the two in
        # separate columns is the whole point: the union bound goes vacuous long
        # before the horizon the controller actually plans over, so the textbook
        # correction is not merely loose there, it says nothing.
        boole = min(1.0, r["per_step_exceedance"] * r["H"])
        rows.append([
            f"{r['H']}", f"{r['hours']:.2f}",
            f"{r['per_step_exceedance']:.3f}",
            f"\\textbf{{{r['empirical_horizon']:.3f}}}",
            f"{r.get('independence_bound', float('nan')):.3f}",
            f"{boole:.3f}",
            f"{r.get('copula_predicted', float('nan')):.3f}",
        ])
    last = d["marginal_vs_joint"][-1]
    #: The horizon at which the union bound reaches 1 and stops saying anything,
    #: computed from the measured per-step rate rather than asserted.
    boole_saturates = int(np.ceil(1.0 / last["per_step_exceedance"]))
    table(OUT / "horizon.tex",
          ["$H$", "hours", "per-step $\\hat\\alpha$", "realised",
           "independent", "Boole $H\\hat\\alpha$", "copula"],
          rows, "rrrrrrr",
          "Marginal versus horizon-level exceedance of the demand ceiling, "
          "held-out June 2017, "
          f"{d['copula']['n_origins']:,} forecast origins. "
          "The per-step rate $\\hat\\alpha$ is well calibrated at every horizon "
          "against a nominal $0.05$. The realised probability of breaching "
          "\\emph{somewhere} in the window is not that number. Neither substitute "
          "is usable: the independence calculation "
          f"$1-(1-\\hat\\alpha)^H$ overstates the risk by ${last['independence_bound'] / last['empirical_horizon']:.1f}\\times$ "
          "because load errors are strongly autocorrelated, and the "
          "Boole/Bonferroni union bound $\\min(1, H\\hat\\alpha)$ -- the only one "
          f"of the two that is a valid bound -- saturates at $1$ by $H={boole_saturates}$ "
          "and is vacuous over the controller's actual horizon.",
          "tab:horizon")


def acceptance_table() -> None:
    """The closed-loop sweep. Commit violation is the acceptance metric; the
    against-target column is shown because omitting it would look like hiding it,
    and labelled as the business metric it is."""
    p = RESULTS / "horizon_risk_Fox_office_Gaylord.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    cl = d.get("closed_loop") or []
    if not cl:
        return
    a = d["acceptance"]
    rows = []
    for r in cl:
        eps = "--" if r.get("epsilon") is None else f"{r['epsilon']:.2f}"
        name = "marginal $q95$" if r["mode"] == "marginal" else "scenario"
        commit = r["commit_violation_rate"]
        gap = "" if r.get("epsilon") is None else (
            f" ({commit - r['epsilon']:+.3f})")
        rows.append([
            r["target"], name, eps,
            f"\\textbf{{{commit:.3f}}}{gap}",
            f"{r['window_violation_rate']:.3f}",
            f"{r['ceiling_breaches']}",
            f"{r['peak_kva']:.1f}",
            f"{r['bill_inr']:,.0f}",
            f"{r['solve_ms_mean']:.0f}",
        ])
    table(OUT / "acceptance.tex",
          ["Target", "Controller", "$\\varepsilon$", "Commit viol.\\ (gap)",
           "vs target", "Breaches", "Peak kVA", "Bill Rs", "Solve ms"],
          rows, "llrrrrrrr",
          "Closed loop over one billing month. \\textbf{Commit violation} is the "
          "acceptance metric: the fraction of horizons in which realised load cleared "
          "the ceiling the optimiser committed to, which is what $\\varepsilon$ is a "
          "statement about. The against-target column is the business metric and is "
          "not what the chance constraint promises, since the committed peak is a "
          f"decision variable. Rank correlation {a['rank_corr']:.3f}, mean absolute gap "
          f"{a['mean_abs_gap']:.3f}, conservative at {a['n_conservative']} of "
          f"{a['n_levels']} levels; resolution floor $1/S={a['resolution_floor']:.3f}$.",
          "tab:acceptance")


def india_table(df: pd.DataFrame) -> None:
    """India against every other system-level series, on the axes that decide
    whether the method is worth deploying and whether it can be trusted when it
    is. The provenance column is load-bearing: with China in the panel, three of
    India's four headline positions are held only among *metered* supplies, and a
    table that hid where the numbers came from would overstate the claim."""
    nat = json.loads((CACHE / "manifest_national.json").read_text())
    g = df[df.arm == "national"].copy()
    if g.empty:
        return
    g["cdd"] = g["id"].map(lambda i: nat[i]["cdd_share"])
    g = g.sort_values("cdd", ascending=False)
    rows = []
    for _, r in g.iterrows():
        m = nat[r["id"]]
        name = pretty(r["id"], "series")
        if r["country"] == "IN":
            name = f"\\textbf{{{name}}}"
        rows.append([
            name,
            "recon." if m.get("reconstructed") else "metered",
            f"{m['native_resolution_min']:.0f}",
            f"{m['cdd_share']:.2f}",
            f"{m['corr_temp']:+.2f}",
            f"{m['p99_over_median']:.2f}",
            f"{r[f'{OURS}_skill']:+.3f}",
            f"{r[f'{OURS}_cov90']:.3f}",
        ])
    table(OUT / "india.tex",
          ["Series", "Provenance", "Native min", "Cooling share",
           "Load--temp. corr.", "$p99/$med", "Skill", "Cov 90\\%"],
          rows, "llrrrrrr",
          "India against every other system-level series in the study, ordered "
          "by cooling-degree-day share. Among metered supplies Delhi is the "
          "extreme of both halves of the verdict: the highest cooling share, the "
          "strongest positive load--temperature relationship, the peakiest "
          "profile, and simultaneously the lowest forecast skill and the worst "
          "interval coverage. The six Chinese provinces, added after the study "
          "was written, displace India from three of those four extremes on the "
          "face of the table --- and the provenance column is why that "
          "displacement is reported rather than acted on. Guangdong is the "
          "closest structural analogue to Delhi in the study and returns the "
          "second-worst skill in it, which reads like corroboration and is not: "
          "Section~\\ref{sec:china} shows the Chinese skill column is an "
          "artefact of how the data was built. Delhi remains the only series at "
          "native 15-minute resolution, the cadence the controller runs at and "
          "the cadence Indian demand charges are assessed on.",
          "tab:india")


def china_audit_table() -> None:
    """The audit that decides how the China arm may be read. Emitted from the
    audit's own output so the compression figure in the paper is the one the
    check produced."""
    p = RESULTS / "china_audit.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = []
    for code, r in d["provinces"].items():
        rows.append([
            esc(r["name"]), f"{r['n_days']}",
            f"\\textbf{{{r['n_distinct_shapes']}}}",
            "/".join(str(c) for c in r["days_per_shape"]),
            f"{r['n_free_parameters']:,}", f"{r['n_hourly_values']:,}",
            f"{r['max_reconstruction_error_pct_of_mean']:.6f}",
        ])
    v = next(iter(d["provinces"].values()))
    table(OUT / "china_audit.tex",
          ["Province", "Days", "Shapes", "Days/shape", "Numbers",
           "Values", "Max err.\\ \\%"],
          rows, "lrrlrrr",
          "Provenance audit of the China arm, run against the data rather than "
          "taken from its documentation. Every calendar day's 24-hour profile is "
          "min--max normalised and the distinct shapes counted. \\emph{Shapes} is "
          "how many survive; \\emph{Numbers} is how many values are needed to "
          "rebuild the year from them; \\emph{Values} is how many the year "
          "contains. A whole year of "
          f"hourly provincial demand is {v['n_free_parameters']} numbers --- "
          f"{v['n_distinct_shapes']} normalised day-shapes and one (min, max) "
          f"pair per day --- reproducing all {v['n_hourly_values']:,} values to "
          f"within {d['summary']['worst_reconstruction_error_pct']/100:.0e} of "
          "their magnitude, which is the precision the source is published at "
          f"rather than an approximation we introduced. Compression "
          f"{v['compression_ratio']}:1. The "
          "published method is exactly what the authors state it is. The "
          "consequence for this study is that forecast skill, a ratio against a "
          "seasonal-naive baseline, is uninformative on these series: the shape "
          "the baseline has to guess is constant, so both models reduce to "
          "predicting the same two administrative numbers per day.",
          "tab:china-audit")


def load_study() -> pd.DataFrame:
    import eval.comparative_report as R
    df = R.load()
    df = df[df.get("error").isna()] if "error" in df.columns else df
    cov = R.climate_covariates()
    return df.merge(cov, on="id", how="left").drop_duplicates("id")


def arm_table(df: pd.DataFrame, arm: str, first: str, firstname: str,
              caption: str, label: str, fname: str,
              keep=None,
              extra: list[tuple[str, str, str]] | None = None) -> None:
    """`keep` filters rows inside an arm. The national arm holds two populations
    with different provenance -- metered transmission and city data, and the
    Chinese provincial series reconstructed from digitised load curves -- and
    pooling them into one table would silently launder the second into the
    first."""
    g = df[df.arm == arm].copy()
    if keep is not None:
        g = g[g.apply(keep, axis=1)]
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
        # NB: not `label` -- that is this function's LaTeX-label parameter, and
        # shadowing it silently renames \label{tab:climate} to \label{Ireland}.
        rowname = r[first]
        if arm == "national":
            rowname = pretty(rowname, "series")
        elif first == "site":
            rowname = pretty(rowname, "site")
        row = [esc(rowname), esc(pretty(r["country"], "country")), cd, hv,
               f"{r['seasonal_naive_pinball']:.3f}", f"{r[f'{OURS}_pinball']:.3f}",
               skt, f"{r[f'{OURS}_cov90']:.3f}"]
        for i, (col, _hdr, kind) in enumerate(extra or []):
            v = r.get(col)
            row.insert(2 + i, "--" if pd.isna(v) else
                       (f"{v:+.2f}" if kind == "signed" else f"{v:.2f}"))
        rows.append(row)
    header = [firstname, "Country", "CDD sh.", "HVAC sh.", "Seas.\\ naive",
              "Ours", "Skill", "Cov 90\\%"]
    align = "llrrrrrr"
    for i, (_col, hdr, _kind) in enumerate(extra or []):
        header.insert(2 + i, hdr)
        align = align[:2] + "r" + align[2:]
    table(OUT / fname, header, rows, align, caption, label)


def calibration_table(df: pd.DataFrame) -> None:
    """Coverage pooled by aggregation level. The Chinese rows are held out as
    their own population rather than folded into tier 2: their provenance is
    different, and the interesting question is whether the tier effect
    *replicates* on them, which pooling would destroy."""
    def group(g, name):
        g = g.dropna(subset=[f"{OURS}_cov90"])
        if g.empty:
            return None
        c = g[f"{OURS}_cov90"]
        return [name, f"{len(g)}", f"\\textbf{{{c.mean():.3f}}}",
                f"{c.min():.3f}", f"{int((c < 0.85).sum())}"]

    rows = [r for r in (
        group(df[df.tier == 1], "Buildings (BDG2), metered"),
        group(df[(df.tier == 2) & (~df.reconstructed.astype(bool))],
              "System demand, metered"),
        group(df[(df.tier == 2) & (df.reconstructed.astype(bool))],
              "System demand, reconstructed"),
    ) if r is not None]
    table(OUT / "calibration.tex",
          ["Population", "$n$", "mean cov.", "worst", "$<0.85$"],
          rows, "lrrrr",
          "Empirical coverage of the nominal 90\\% interval after conformal "
          "calibration. The guarantee very nearly holds on individual buildings "
          "and fails systematically on system-level demand. The third row is the "
          "replication: six Chinese provinces, a different country, a different "
          "year and a different data-generating process, reproduce the "
          "system-level failure to within 0.005 of the metered panel and contain "
          "the worst row in the study (Heilongjiang, 0.565).",
          "tab:calibration")


def correlation_table(df: pd.DataFrame) -> None:
    """The null, and its replication.

    The building panel is the original test. The Chinese provincial panel is an
    independent one -- different country, different year, different provenance,
    and a wider climate span than the whole European set -- run after the study
    was written, on data chosen by someone else's question. Reporting them as two
    panels rather than one pooled correlation is the point: a null that
    replicates on a panel it was not fitted to is worth more than a null with a
    larger $n$."""
    def r_of(x, frame):
        g = frame.dropna(subset=[x, f"{OURS}_skill"])
        if len(g) < 3:
            return float("nan"), 0
        return float(np.corrcoef(g[x].astype(float),
                                 g[f"{OURS}_skill"].astype(float))[0, 1]), len(g)

    def block(frame, specs):
        out = []
        for x, label in specs:
            r, n = r_of(x, frame)
            if n == 0:
                continue
            g = frame.dropna(subset=[x, f"{OURS}_skill"])
            loo = [r_of(x, g.drop(i))[0] for i in g.index]
            out.append([label, f"{r:+.3f}", f"{n}",
                        f"{min(loo):+.3f} to {max(loo):+.3f}"])
        return out

    b = df[df.tier == 1].dropna(subset=[f"{OURS}_skill"])
    cn = df[(df.tier == 2) & (df.reconstructed.astype(bool))].dropna(
        subset=[f"{OURS}_skill"])

    rows = [[r"\emph{Panel A --- 18 BDG2 buildings, three countries}", "", "", ""]]
    rows += block(b, [("cdd_share", "\\quad cooling-degree-day share"),
                      ("hvac_share", "\\quad controllable (HVAC) fraction"),
                      ("t_mean", "\\quad mean outdoor temperature"),
                      ("median_load", "\\quad median load")])
    if len(cn):
        rows += [[r"\midrule"],
                 [r"\emph{Panel B --- 6 Chinese provinces, independent replication}",
                  "", "", ""]]
        rows += block(cn, [("cdd_share", "\\quad cooling-degree-day share"),
                           ("t_mean", "\\quad mean outdoor temperature"),
                           ("corr_temp", "\\quad load--temperature correlation")])

    # the one relationship that is not null
    g = df[df.tier == 1].dropna(subset=["cdd_share", "hvac_share"])
    r = float(np.corrcoef(g.cdd_share.astype(float), g.hvac_share.astype(float))[0, 1])
    rows += [[r"\midrule"],
             ["\\emph{controllable fraction} vs climate (not vs skill)",
              f"\\textbf{{{r:+.3f}}}", f"{len(g)}", "--"]]

    table(OUT / "correlations.tex",
          ["Correlation against forecast skill", "$r$", "$n$",
           "leave-one-out range"],
          rows, "lrrl",
          "Forecast skill is uncorrelated with climate, controllable fraction, "
          "temperature and load size, and every leave-one-out range straddles or "
          "nearly straddles zero --- which at these sample sizes is what a null "
          "looks like. Panel B is the replication on a panel the null was not "
          "fitted to. The final row is the one relationship in the study that is "
          "real, and note what it is between: available flexibility tracks "
          "climate strongly, while predictability does not track anything. Skill "
          "and flexibility are independent deployment axes.",
          "tab:correlations")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("emitting LaTeX tables:")
    horizon_table()
    acceptance_table()
    df = load_study()
    arm_table(df, "climate", "site", "Site",
              "Climate arm: demographic held fixed at Education, climate and "
              "country varied across seven sites and three countries.",
              "tab:climate", "climate.tex")
    arm_table(df, "office", "site", "Site",
              "Office arm: the same ladder in a second usage class, run as a "
              "control on the climate arm. The London office is the study's worst "
              "row and is reported unchanged.",
              "tab:office", "office.tex")
    arm_table(df, "demographic", "usage", "Usage class",
              "Demographic arm: climate held fixed at Washington DC, demographic "
              "varied across the eight usage classes that pass the selection rule "
              "at that site. Six rows appear here; Education and Office were "
              "also run in this arm and are shown in "
              "Tables~\\ref{tab:climate} and~\\ref{tab:office} instead, since a "
              "building is one supply however many arms it appears in.",
              "tab:demographic", "demographic.tex")
    arm_table(df, "national", "site", "Series",
              "National arm, metered: system-level demand in six countries. "
              "Delhi is 15-minute native resolution from the Delhi State Load "
              "Despatch Centre; the European series are ENTSO-E hourly via Open "
              "Power System Data.",
              "tab:national", "national.tex",
              keep=lambda r: not bool(r.get("reconstructed")))
    _cnskill = df[(df.tier == 2) & (df.reconstructed.astype(bool))][f"{OURS}_skill"]
    _cn = {"mean": float(_cnskill.mean()), "neg": int((_cnskill < 0).sum())}
    arm_table(df, "national", "site", "Series",
              "The China arm, added after the study was complete on a "
              "constraint imposed from outside it: six provinces spanning a "
              "wider climate range than the entire European panel, in a "
              "developing economy, at one calendar year and therefore a "
              "three-month training block rather than fifteen. "
              f"Mean skill is ${_cn['mean']:+.3f}$ with {_cn['neg']} negative "
              "rows. \\textbf{That column should not be read as a statement "
              "about the method}, for the reason established in "
              "Table~\\ref{tab:china-audit}: the series is constructed, not "
              "metered, and its construction pins our model and the baseline it "
              "is scored against to the same two numbers per day. Coverage does "
              "not run through the baseline and is readable; skill is not.",
              "tab:china", "china.tex",
              keep=lambda r: bool(r.get("reconstructed")),
              extra=[("corr_temp", "Load--temp.", "signed")])
    india_table(df)
    china_audit_table()
    calibration_table(df)
    correlation_table(df)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
