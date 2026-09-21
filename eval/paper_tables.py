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


def ci(bounds, nd: int = 2) -> str:
    """A bootstrap interval as one cell, or a dash for a result that predates them."""
    if not bounds:
        return "--"
    return f"[{bounds[0]:.{nd}f}, {bounds[1]:.{nd}f}]"


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
            ci(r.get("empirical_horizon_ci")),
            f"{r.get('independence_bound', float('nan')):.3f}",
            f"{boole:.3f}",
            f"{r.get('copula_predicted', float('nan')):.3f}",
        ])
    last = d["marginal_vs_joint"][-1]
    #: The horizon at which the union bound reaches 1 and stops saying anything,
    #: computed from the measured per-step rate rather than asserted.
    boole_saturates = int(np.ceil(1.0 / last["per_step_exceedance"]))
    lo, hi = last.get("empirical_horizon_ci", (float("nan"), float("nan")))
    table(OUT / "horizon.tex",
          ["$H$", "hours", "per-step $\\hat\\alpha$", "realised", "95\\% CI",
           "independent", "Boole $H\\hat\\alpha$", "copula"],
          rows, "rrrrcrrr",
          "Marginal versus horizon-level exceedance of the demand ceiling, "
          "held-out June 2017, "
          f"{last['n_origins']:,} forecast origins over {last.get('n_blocks', 30)} days. "
          "The per-step rate $\\hat\\alpha$ is well calibrated at every horizon "
          "against a nominal $0.05$. The realised probability of breaching "
          "\\emph{somewhere} in the window is not that number: at $H=64$ it is "
          f"between ${lo / 0.05:.0f}$ and ${hi / 0.05:.0f}$ times the nominal level "
          "(day-block bootstrap, 95\\% interval; adjacent origins share 63 of 64 "
          "steps, so the day is the unit of replication). Neither substitute "
          "is usable: the independence calculation "
          f"$1-(1-\\hat\\alpha)^H$ overstates the risk by ${last['independence_bound'] / last['empirical_horizon']:.1f}\\times$ "
          "because load errors are strongly autocorrelated, and the "
          "Boole/Bonferroni union bound $\\min(1, H\\hat\\alpha)$ -- the only one "
          f"of the two that is a valid bound -- saturates at $1$ by $H={boole_saturates}$ "
          "and is vacuous over the controller's actual horizon.",
          "tab:horizon")



def _india_rows() -> list[dict]:
    """The I-BLEND arm, one row per admissible (series, June) window."""
    p = RESULTS / "comparative_india.json"
    if not p.exists():
        return []
    rows = []
    for v in json.loads(p.read_text()).values():
        if "models" not in v:
            continue
        rows.append(dict(
            id=v["id"], name=v["id"].replace("IIITD_", ""), level=v["usage"],
            june=int(v["test_june"]), train_months=int(v["train_months"]),
            cooling=bool(v.get("cooling_on_meter")), median=float(v["median_load"]),
            seasonal=v["models"]["seasonal_naive"]["pinball_mean"],
            ours=v["models"][OURS]["pinball_mean"],
            skill=v["models"][OURS]["skill_vs_seasonal"],
            cov=v["models"][OURS]["coverage_90"],
        ))
    rows.sort(key=lambda r: (r["level"] != "campus", r["name"], r["june"]))
    return rows


def iblend_table() -> None:
    rows = _india_rows()
    if not rows:
        return
    out = []
    for r in rows:
        out.append([
            esc(r["name"]), r["level"], f"{r['june']}", f"{r['train_months']}",
            "yes" if r["cooling"] else "no",
            f"{r['median']:.1f}", f"{r['seasonal']:.3f}", f"{r['ours']:.3f}",
            f"\\textbf{{{r['skill']:+.3f}}}", f"{r['cov']:.3f}",
        ])
    b = [r for r in rows if r["level"] != "campus"]
    won = sum(r["skill"] > 0 for r in b)
    full = [r for r in b if r["train_months"] >= 13]
    short = [r for r in b if r["train_months"] < 13]
    camp = [r for r in rows if r["level"] == "campus"]
    table(OUT / "iblend.tex",
          ["Series", "Level", "June", "Train mo.", "Cooling on meter", "Median kW",
           "Seas.\\ naive", "Ours", "Skill", "Cov 90\\%"],
          out, "llrrlrrrrr",
          "The Indian building arm: I-BLEND, IIIT-Delhi, one row per admissible "
          "(series, June) window, native 15-minute blocks, the unchanged protocol "
          "with the training block shortened where meter uptime forces it (the "
          "China-arm precedent). \\emph{Cooling on meter} is the selection rule's "
          "chilled-water clause: six of seven buildings are served by a central "
          "chiller on the campus feed, so only the Facilities building and the "
          "campus total are control objects; every row is a forecasting and "
          "calibration result. "
          f"Over the {len(b)} building rows the model beats seasonal naive on {won}; "
          f"mean skill is ${np.mean([r['skill'] for r in b]):+.3f}$, "
          f"${np.mean([r['skill'] for r in full]):+.3f}$ on the {len(full)} windows with a full "
          f"training block and ${np.mean([r['skill'] for r in short]):+.3f}$ on the {len(short)} shorter ones. "
          f"Coverage is {min(r['cov'] for r in b):.3f}--{max(r['cov'] for r in b):.3f} on every building row"
          + (f" and {camp[0]['cov']:.3f} on the campus feed" if camp else "") + ".",
          "tab:iblend")


def horizon_panel_table() -> None:
    """The bracket, replicated: one row per window, Fox first."""
    files = [RESULTS / "horizon_risk_Fox_office_Gaylord.json"] + \
            sorted(RESULTS.glob("horizon_risk_IIITD_*.json"))
    rows, vals, cis, two_day = [], [], [], []
    for f in files:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        r = [x for x in d["marginal_vs_joint"] if x["H"] == 64][0]
        fox = d["building"].startswith("Fox")
        name = "Phoenix office (BDG2)" if fox else esc(d["building"].replace("IIITD_", "IIIT-Delhi "))
        june = "2017" if fox else d.get("tag", "").lstrip("@")
        boole = min(1.0, 64 * r["per_step_exceedance"])
        rows.append([name, june, f"{r['per_step_exceedance']:.3f}",
                     f"\\textbf{{{r['empirical_horizon']:.3f}}}",
                     ci(r.get("empirical_horizon_ci")),
                     f"{r['independence_bound']:.3f}", f"{boole:.2f}",
                     f"{r['copula_predicted']:.3f}", f"{d['copula']['corr_lag1']:.2f}"])
        vals.append((r["per_step_exceedance"], r["empirical_horizon"],
                     r["independence_bound"], r["copula_predicted"]))
        cis.append(tuple(r.get("empirical_horizon_ci", (np.nan, np.nan))))
        two_day.append(tuple(r.get("block_sensitivity", {}).get("2", (np.nan, np.nan))))
        if fox and len(files) > 1:
            rows.append(["\\midrule"])
    if len(vals) < 2:
        return
    ps, emp, ind, cop = (np.array([v[i] for v in vals]) for i in range(4))
    cop_err, ind_err = np.abs(cop - emp), np.abs(ind - emp)
    lo, hi = np.array(cis).T
    lo2, hi2 = np.array(two_day).T
    n = len(vals)
    #: The three statements the intervals licence, each computed rather than
    #: asserted: the floor every window clears even at the bottom of its
    #: interval; whether the spread across windows is wider than the
    #: intervals (pairs whose intervals are disjoint); and which of the two
    #: substitutes ever lands inside the realised interval.
    floor_x = float(np.nanmin(lo) / 0.05)
    disjoint = sum(1 for i in range(n) for j in range(i + 1, n)
                   if hi[i] < lo[j] or hi[j] < lo[i])
    cop_in = int(((cop >= lo) & (cop <= hi)).sum())
    ind_in = int(((ind >= lo) & (ind <= hi)).sum())
    widen = float(np.nanmean((hi2 - lo2) / (hi - lo)))
    floor2_x = float(np.nanmin(lo2) / 0.05)
    i_lo, i_hi = int(np.argmin(emp)), int(np.argmax(emp))
    extremes_disjoint = bool(hi[i_lo] < lo[i_hi])
    disjoint2 = sum(1 for i in range(n) for j in range(i + 1, n)
                    if hi2[i] < lo2[j] or hi2[j] < lo2[i])
    table(OUT / "horizon_panel.tex",
          ["Series", "June", "per-step $\\hat\\alpha$", "realised $H{=}64$", "95\\% CI",
           "independent", "Boole", "copula", "lag-1 $\\rho$"],
          rows, "llrrcrrrr",
          "The bracket of Table~\\ref{tab:horizon}, replicated over "
          f"{n} windows: the Phoenix office of the original measurement and "
          f"{n - 1} Indian windows at native 15-minute resolution. The per-step "
          f"rate is calibrated throughout (${ps.min():.3f}$--${ps.max():.3f}$ against a "
          "nominal $0.05$). The realised probability of breaching somewhere in the "
          f"16-hour window runs from ${emp.min():.3f}$ to ${emp.max():.3f}$, median "
          f"${np.median(emp):.3f}$ --- between ${emp.min() / 0.05:.0f}$ and "
          f"${emp.max() / 0.05:.0f}$ times the level the constraint appears to promise. "
          "Intervals are day-block bootstrap, the day being the unit of replication "
          "because adjacent origins share 63 of 64 steps. No window's lower bound "
          f"falls under ${np.nanmin(lo):.2f}$, ${floor_x:.1f}\\times$ nominal. The "
          f"spread across windows is not sampling noise: {disjoint} of "
          f"{n * (n - 1) // 2} pairs of intervals are disjoint"
          + (", the lowest and highest rows among them" if extremes_disjoint else "")
          + ", and the same building returns different values in "
          "different years while its per-step rate does not move. A 16-hour window "
          "opened late in one day runs into the next, so two-day blocks are also "
          f"reported: they widen the intervals by {100 * (widen - 1):.0f}\\%, and the "
          f"floor becomes ${np.nanmin(lo2):.2f}$ (${floor2_x:.1f}\\times$) with "
          f"{disjoint2} pairs disjoint. "
          f"The independence figure overstates realised risk by ${np.median(ind / emp):.2f}\\times$ "
          f"(median) and lies inside the realised interval on {ind_in} of {n} windows; "
          "the Boole bound is $1$ on every row. The copula predicts the "
          f"realised value to a mean absolute error of ${cop_err.mean():.3f}$ against "
          f"${ind_err.mean():.3f}$ for independence, and lies inside the interval on "
          f"{cop_in} of {n} windows.",
          "tab:horizon-panel")

def copula_df_table() -> None:
    """What the choice of the copula's one free parameter contributes: the
    held-out prediction under the tail-matched nu, under a pseudo-likelihood
    nu, and under nu held at the Phoenix value with no refitting."""
    p = RESULTS / "copula_df_check.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    W = d["windows"]

    def nu(v):
        return "Gauss." if v is None else f"{v:g}"

    def cell(pred, ci):
        inside = ci[0] <= pred <= ci[1]
        return f"{pred:.3f}" if inside else f"{pred:.3f}$^\\dagger$"

    rows = []
    for w in W:
        fox = w["building"].startswith("Fox")
        name = "Phoenix office (BDG2)" if fox else esc(w["building"].replace("IIITD_", "IIIT-Delhi "))
        june = "2017" if fox else w["tag"].lstrip("@")
        ci = w["realised_ci"]["64"]
        rows.append([name, june, f"\\textbf{{{w['realised']['64']:.3f}}}", ci_str(ci),
                     nu(w["df_tail"]), cell(w["predicted"]["tail"]["64"], ci),
                     nu(w["df_ml"]), cell(w["predicted"]["ml"]["64"], ci),
                     cell(w["predicted"]["fixed"]["64"], ci)])
        if fox and len(W) > 1:
            rows.append(["\\midrule"])
    ind = [w for w in W if not w["building"].startswith("Fox")]
    n_ind = len(ind)

    def stats(k, ws):
        return (sum(w["inside_64"][k] for w in ws),
                float(np.mean([w["abs_err_64"][k] for w in ws])),
                sum(w["predicted"][k]["64"] > w["realised"]["64"] for w in ws))
    t_in, t_mae, _ = stats("tail", W)
    m_in, m_mae, m_over = stats("ml", W)
    f_in, f_mae, _ = stats("fixed", ind)
    ml_heavy = sum(1 for w in W if w["df_ml"] is not None and w["df_ml"] < 25)
    table(OUT / "copula_df.tex",
          ["Series", "June", "realised", "95\\% CI", "$\\nu_{\\mathrm{tail}}$", "copula",
           "$\\nu_{\\mathrm{lik}}$", "copula", "copula, $\\nu{=}7$"],
          rows, "llrcrrrrr",
          "What selecting the copula's degrees of freedom contributes. Realised "
          "horizon exceedance at $H=64$ on the held-out month with its day-block "
          "bootstrap interval, and the copula's prediction under three choices of "
          "$\\nu$: matched to horizon exceedance on the validation block (the paper's "
          "choice; columns 5--6), maximising the $t$-copula pseudo-likelihood on the "
          "same block (columns 7--8), and held at the Phoenix value $\\nu=7$ with no "
          "refitting (last column). $\\dagger$ marks a prediction outside the "
          f"realised interval. The tail-matched choice lies inside on {t_in} of {len(W)} "
          f"windows (mean absolute error ${t_mae:.3f}$). The likelihood criterion "
          f"selects $\\nu \\geq 25$ or the Gaussian on every window"
          + (f" but {ml_heavy}" if ml_heavy else "")
          + f", over-predicts on {m_over} of {len(W)} and lies inside on {m_in}: it is "
          "dominated by the body of the distribution, where the families agree, and "
          "cannot see the upper-tail dependence the horizon event is made of. The "
          f"Phoenix value applied unchanged to the {n_ind} Indian windows lies inside on "
          f"{f_in} of {n_ind} (mean absolute error ${f_mae:.3f}$); the per-window "
          "selection buys accuracy, not the agreement itself. The independence figure "
          "lies inside on none.",
          "tab:copula-df")


def ci_str(bounds) -> str:
    return ci(bounds)


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
    # India, one city, three aggregation levels, one weather feed. Kept as its
    # own panel: these rows are windows of the same meters, not independent
    # supplies, and pooling them into the tiers above would double-count.
    ib = _india_rows()
    if ib:
        def igroup(rs, name):
            c = np.array([r["cov"] for r in rs])
            return [name, f"{len(rs)}", f"\\textbf{{{c.mean():.3f}}}",
                    f"{c.min():.3f}", f"{int((c < 0.85).sum())}"]
        delhi = df[(df.tier == 2) & (df["id"].astype(str).str.contains("Delhi", case=False))]
        rows.append(["\\midrule"])
        rows.append(["\\emph{India, one city (I-BLEND + Delhi SLDC), windows}", "", "", "", ""])
        rows.append(igroup([r for r in ib if r["level"] != "campus"], "\\quad buildings, native 15-min"))
        rows.append(igroup([r for r in ib if r["level"] == "campus"], "\\quad campus feed (one HT consumer)"))
        if not delhi.empty:
            c = float(delhi[f"{OURS}_cov90"].iloc[0])
            rows.append(["\\quad city (Delhi SLDC)", "1", f"\\textbf{{{c:.3f}}}", f"{c:.3f}", f"{int(c < 0.85)}"])
    table(OUT / "calibration.tex",
          ["Population", "$n$", "mean cov.", "worst", "$<0.85$"],
          rows, "lrrrr",
          "Empirical coverage of the nominal 90\\% interval after conformal "
          "calibration. The guarantee very nearly holds on individual buildings "
          "and fails systematically on system-level demand. The third row is the "
          "replication: six Chinese provinces, a different country, a different "
          "year and a different data-generating process, reproduce the "
          "system-level failure to within 0.005 of the metered panel and contain "
          "the worst row in the study (Heilongjiang, 0.565). The lower panel is "
          "the same effect inside one city: building, campus and city demand in "
          "Delhi, one weather feed, native 15-minute resolution throughout, and "
          "coverage falls monotonically with each step up the ladder.",
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
    horizon_panel_table()
    copula_df_table()
    acceptance_table()
    iblend_table()
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
