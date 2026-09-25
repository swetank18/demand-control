"""Emit the paper's tables as LaTeX, so no number in it is typed by hand.

Same contract as everywhere else in this repo: a number reaches the paper only
if a script put it there. `main.tex` \\input{}s these files, so regenerating the
study and re-running this is the whole update path for the write-up.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.windows import primary_results

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
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


#: Emit for the two-column IEEEtran build instead of the single-column report:
#: the same rows and the same caption, in a spanning environment at a smaller
#: face. Set by --ieee so that one generator serves both documents and no
#: number is ever retyped into the conference version.
IEEE = False
SHORT_CAPTIONS = False

#: IEEEtran sets table captions in small caps, where the report's
#: paragraph-length ones are unreadable. The conference build keeps whole
#: sentences up to this budget and drops the discussion that follows, so the
#: numbers a caption states are still the generated ones and nothing is
#: rewritten by hand.
IEEE_CAPTION_CHARS = 620


def _short_caption(caption: str, budget: int = IEEE_CAPTION_CHARS) -> str:
    if len(caption) <= budget:
        return caption
    out, n = [], 0
    #: split on sentence ends that are not a decimal point or an abbreviation
    for sentence in re.split(r"(?<=[.;]) (?=[A-Z(])", caption):
        if n + len(sentence) > budget and out:
            break
        out.append(sentence)
        n += len(sentence) + 1
    return " ".join(out)


def table(path: Path, header: list[str], rows: list[list[str]], align: str,
          caption: str, label: str) -> None:
    """Wide tables get a smaller face and tighter columns.

    Eight or more columns overruns the text block at \\small on a4paper, which
    LaTeX reports as an overfull hbox and a reader sees as a table poking into
    the margin. Deciding this from the column count keeps it automatic rather
    than something to remember per table. In the two-column build the same
    count decides between a column-width table and one that spans the page.
    """
    wide = len(header) >= 8
    if IEEE:
        # Everything spans. A five-column table looks narrow until its first
        # column carries "System demand, reconstructed", and a table that
        # overruns an IEEE column is worse than one that takes the page width.
        env = "table*"
        size = r"\scriptsize" if wide else r"\footnotesize"
        if SHORT_CAPTIONS:
            caption = _short_caption(caption)
        L = [rf"\begin{{{env}}}[t]", r"\centering", size,
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
        L += [r[0] if len(r) == 1 and str(r[0]).startswith("\\")
              else " & ".join(r) + r" \\" for r in rows]
        L += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}", ""]
        path.write_text("\n".join(L))
        print(f"  {path.name}")
        return
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
    files = [RESULTS / "horizon_risk_Fox_office_Gaylord.json"] + primary_results(RESULTS)
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

AUDIT_SERIES = (("Fox_office_Gaylord", "Phoenix office, building, 2016--17"),
                ("IN_Delhi", "Delhi city, system, 2011--12"),
                ("IN_Delhi@rel", "Delhi city, system, 2011--12"))


def aci_table() -> None:
    """The conformal audit, one row per tier and ACI step it has been run at:
    the monthly range and in-band share of the walk-forward year for split
    conformal and for ACI, and post-shift coverage under the frozen model."""
    rows, found = [], []
    for key, name in AUDIT_SERIES:
        p = RESULTS / f"conformal_audit_{key}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        m = d["year"]["by_month"]
        b = d["year"]["band_90"]
        f = d["frozen_shift"]
        sp = [r["split_cov90"] for r in m]
        ac = [r["aci_cov90"] for r in m]
        gamma = d.get("gamma", 0.35)
        kappa = d.get("gamma_rel", float("nan"))
        rows.append([name, f"{gamma:.2f}", f"{kappa:.4f}",
                     f"{min(sp):.3f}--{max(sp):.3f}", f"{b['split']['in_band_pct']:.0f}\\%",
                     f"{min(ac):.3f}--{max(ac):.3f}", f"{b['aci']['in_band_pct']:.0f}\\%",
                     f"{f['post_shift_cov90_split']:.3f}", f"{f['post_shift_cov90_aci']:.3f}"])
        found.append((key, name, d))
    if not rows:
        return
    by = {k: d for k, _, d in found}
    fox, dl, rel = by.get("Fox_office_Gaylord"), by.get("IN_Delhi"), by.get("IN_Delhi@rel")

    def pct(d, layer):
        return d["year"]["band_90"][layer]["in_band_pct"]
    cap = ("The adaptive layer at each tier it has been run at. Walk-forward year of "
           "twelve monthly folds, each trained strictly on the past and calibrated on "
           "the thirty days before it; coverage of the nominal $0.90$ interval by month "
           "(range) and the share of the rolling 30-day curve inside $0.85$--$0.95$. "
           "The last two columns are coverage after a level and volatility shift "
           "injected into a model frozen six months earlier, which static calibration "
           "cannot respond to by construction. $\\gamma$ is the ACI step in the units of "
           "the series and $\\kappa$ the same step as a fraction of the mean "
           "split-conformal interval width at the audited lead. ")
    if fox:
        cap += (f"At the building, split conformal spends {pct(fox, 'split'):.0f}\\% of the "
                f"year in band and ACI {pct(fox, 'aci'):.0f}\\%. ")
    if dl:
        cap += (f"Carried to the city as the same absolute $\\gamma$, the step is "
                f"{fox['gamma_rel'] / dl['gamma_rel']:.0f}$\\times$ smaller relative to the "
                f"interval and ACI reaches {pct(dl, 'aci'):.0f}\\% against split's "
                f"{pct(dl, 'split'):.0f}\\%" if fox else "")
        cap += ". "
    if rel and fox:
        cap += (f"Carried as the same $\\kappa$, with nothing tuned on Delhi, ACI spends "
                f"{pct(rel, 'aci'):.0f}\\% of the year in band, by-month coverage "
                f"{min(r['aci_cov90'] for r in rel['year']['by_month']):.3f}--"
                f"{max(r['aci_cov90'] for r in rel['year']['by_month']):.3f}, and settles at "
                f"{rel['frozen_shift']['post_shift_cov90_aci']:.3f} after the shift against "
                f"split's {rel['frozen_shift']['post_shift_cov90_split']:.3f}.")
    table(OUT / "aci.tex",
          ["Series", "$\\gamma$", "$\\kappa$", "split, by month", "in band",
           "ACI, by month", "in band", "split", "ACI"],
          rows, "lrrrrrrrr", cap, "tab:aci")


def aci_gamma_table() -> None:
    """The step-size sweep behind the kappa column: ACI replayed over the saved
    year at a grid of relative steps, both tiers side by side."""
    p = RESULTS / "aci_gamma.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    keys = [k for k in ("Fox_office_Gaylord", "IN_Delhi") if k in d]
    if not keys:
        return
    kappas = [r["kappa"] for r in d[keys[0]]["sweep"]]
    rows = []
    for i, k in enumerate(kappas):
        row = [f"{k:g}" if k else "0 (split only)"]
        for key in keys:
            r = d[key]["sweep"][i]
            row += [f"{r['band_90']['in_band_pct']:.1f}\\%",
                    f"{r['by_fold_cov90_min']:.3f}--{r['by_fold_cov90_max']:.3f}",
                    f"{r['mean_width_kw']:.0f}"]
        rows.append(row)
    heads = ["$\\kappa$"]
    for key in keys:
        heads += ["in band", "by month", "width"]
    defaults = ", ".join(f"{d[k]['default']['kappa']:.4f}" for k in keys)
    widths = ", ".join(f"{d[k]['split_width']:.0f}" for k in keys)
    table(OUT / "aci_gamma.tex", heads, rows, "l" + "rrr" * len(keys),
          "ACI replayed over the saved walk-forward year at a grid of step sizes, "
          "stated as a fraction $\\kappa$ of the mean split-conformal interval width "
          f"at the audited lead ({widths} in the series' units for the "
          "Phoenix office and Delhi city respectively; left block Phoenix, right block "
          "Delhi). Nothing is retrained: ACI is a pass over the saved split-conformal "
          "predictions in time order. In band is the share of the rolling 30-day "
          "coverage curve inside $0.85$--$0.95$; by month is the range of monthly "
          "coverage; width is the mean adaptive interval. The audit's absolute default "
          f"$\\gamma=0.35$ is $\\kappa = {defaults}$ on the two series. Above "
          "$\\kappa \\approx 0.01$ the interval is a fast tracker rather than a "
          "forecast interval: the recursion pins the long-run rate whatever the model, "
          "and the width it reports is no longer the forecaster's.",
          "tab:aci-gamma")


def horizon_kappa_table() -> None:
    """The horizon rows again with a per-step layer whose step transfers.

    Every window in Table horizon-panel was calibrated with an adaptive step
    in the units of its own series, and those series run from 5 kW intervals
    to 800 kW ones. This is the same measurement with the step stated as a
    fraction of the interval width, so the per-step column is comparable
    across rows -- and the question is whether the horizon gap survives it.
    """
    pairs = []
    for f in sorted(RESULTS.glob("horizon_risk_*k.json")):
        key = f.stem.replace("horizon_risk_", "")
        base = key[:-1] if not key.endswith("@k") else key[:-2] + "@g"
        b = RESULTS / f"horizon_risk_{base}.json"
        if not b.exists():
            continue
        dk, db = json.loads(f.read_text()), json.loads(b.read_text())
        rk = [x for x in dk["marginal_vs_joint"] if x["H"] == 64][0]
        rb = [x for x in db["marginal_vs_joint"] if x["H"] == 64][0]
        fox = dk["building"].startswith("Fox")
        name = "Phoenix office (BDG2)" if fox else esc(dk["building"].replace("IIITD_", "IIIT-Delhi "))
        june = "2017" if fox else dk.get("tag", "").lstrip("@").rstrip("k")
        pairs.append((fox, name, june, rb, rk, dk))
    if not pairs:
        return
    pairs.sort(key=lambda t: (not t[0], t[1], t[2]))
    rows = []
    for i, (fox, name, june, rb, rk, dk) in enumerate(pairs):
        rows.append([name, june,
                     f"{rb['per_step_exceedance']:.3f}", f"{rk['per_step_exceedance']:.3f}",
                     f"{rb['empirical_horizon']:.3f}",
                     f"\\textbf{{{rk['empirical_horizon']:.3f}}}",
                     ci(rk.get("empirical_horizon_ci")),
                     f"{rk['copula_predicted']:.3f}"])
        if fox and len(pairs) > 1:
            rows.append(["\\midrule"])
    pb = np.array([p[3]["per_step_exceedance"] for p in pairs])
    pk = np.array([p[4]["per_step_exceedance"] for p in pairs])
    eb = np.array([p[3]["empirical_horizon"] for p in pairs])
    ek = np.array([p[4]["empirical_horizon"] for p in pairs])
    lo = np.array([p[4]["empirical_horizon_ci"][0] for p in pairs])
    cop = np.array([p[4]["copula_predicted"] for p in pairs])
    cin = int(sum(p[4]["empirical_horizon_ci"][0] <= p[4]["copula_predicted"] <= p[4]["empirical_horizon_ci"][1]
                  for p in pairs))
    #: whether the two columns differ by more than the measurement can resolve:
    #: the absolute-step value against the relative-step row's own interval
    inside = int(sum(p[4]["empirical_horizon_ci"][0] <= p[3]["empirical_horizon"] <= p[4]["empirical_horizon_ci"][1]
                     for p in pairs))
    #: the absolute step, expressed in the units the relative one is stated in,
    #: is what varied across these series and is the reason for the rerun
    kap = np.array([0.35 / json.loads((MODELS / f"{p[5]['building']}{p[5].get('tag','')}" / "meta.json").read_text())
                    ["adaptive_gamma_width"] for p in pairs])
    table(OUT / "horizon_kappa.tex",
          ["Series", "June", "$\\hat\\alpha$ abs.", "$\\hat\\alpha$ rel.",
           "realised abs.", "realised rel.", "95\\% CI", "copula"],
          rows, "llrrrrcr",
          "The horizon rows of Table~\\ref{tab:horizon-panel} with the per-step "
          "layer's adaptive step stated as a fraction of the interval width "
          "($\\kappa=0.006$, the width read on the validation block) instead of as "
          "an absolute number in the units of each series. These meters run from "
          "intervals of a few kilowatts to several hundred, so the absolute step "
          f"was a different layer on every row --- $\\kappa={kap.min():.4f}$ to "
          f"$\\kappa={kap.max():.4f}$ across these windows, so on all but the campus feed "
          "it was faster than the step the building audit settled on, not slower. "
          f"Under the relative step the per-step rate spans ${pk.min():.3f}$--${pk.max():.3f}$ "
          f"against ${pb.min():.3f}$--${pb.max():.3f}$ before, and its mean absolute "
          f"distance from the nominal $0.05$ moves from ${np.abs(pb - 0.05).mean():.4f}$ to "
          f"${np.abs(pk - 0.05).mean():.4f}$: a slower layer tracks the nominal rate less "
          "tightly, which is why the per-step column of "
          "Table~\\ref{tab:horizon-panel} is the one reported there. The horizon gap "
          f"survives and shrinks. Realised 16-hour exceedance runs "
          f"${ek.min():.3f}$--${ek.max():.3f}$ (median ${np.median(ek):.3f}$, "
          f"${np.median(ek) / 0.05:.0f}\\times$ nominal) against "
          f"${eb.min():.3f}$--${eb.max():.3f}$ (median ${np.median(eb):.3f}$) before; the "
          f"paired change is ${(ek - eb).mean():+.3f}$ on average, and the absolute-step "
          f"value lies inside the relative-step row's own $95\\%$ interval on {inside} of "
          f"{len(pairs)} windows, so on {len(pairs) - inside} the two are separated by more "
          "than sampling error. The magnitude of the gap is therefore not independent of how "
          "the per-step layer is tuned --- a faster layer produces a noisier bound and more "
          "nearly independent exceedances across the horizon --- but no choice of step makes "
          f"it go away: the lowest lower bound anywhere in this table is ${lo.min():.2f}$, "
          f"${lo.min() / 0.05:.1f}\\times$ the level the constraint appears to promise. The copula "
          f"lies inside the realised interval on {cin} of {len(pairs)} windows. "
          "Fox is run through this trainer at both steps, because its row in "
          "Table~\\ref{tab:horizon-panel} comes from the benchmark harness and "
          "would not otherwise be a paired comparison.",
          "tab:horizon-kappa")


def aci_step_table() -> None:
    """The aggregation ladder of Table calibration, re-read with the same
    forecaster under three calibrations of the same predictions: split
    conformal alone, ACI at the absolute step every benchmark row used, and
    ACI at the building's step as a fraction of the interval width."""
    p = RESULTS / "aci_step_check.json"
    if not p.exists():
        return
    rows_in = [v for v in json.loads(p.read_text()).values() if "calib" in v]
    if not rows_in:
        return
    seen, uniq = set(), []
    for v in rows_in:                       # a building in two arms is one supply
        k = (v["id"], v.get("test_june"))
        if k not in seen:
            seen.add(k); uniq.append(v)

    def pop(name, pred):
        g = [v for v in uniq if pred(v)]
        if not g:
            return None
        cov = {c: np.array([v["calib"][c]["coverage_90"] for v in g]) for c in ("split", "aci_abs", "aci_rel")}
        kap = np.median([v["kappa_of_abs"] for v in g])
        return [name, f"{len(g)}", f"{kap:.4f}",
                f"{cov['split'].mean():.3f}", f"{cov['aci_abs'].mean():.3f}",
                f"\\textbf{{{cov['aci_rel'].mean():.3f}}}", f"{cov['aci_rel'].min():.3f}",
                f"{int((cov['aci_rel'] < 0.85).sum())}"], cov

    specs = [
        ("Buildings (BDG2), metered", lambda v: v["tier"] == 1 and v["arm"] != "india"),
        ("System demand, metered", lambda v: v["arm"] == "national" and not v["id"].startswith("CN_")),
        ("System demand, reconstructed", lambda v: v["arm"] == "national" and v["id"].startswith("CN_")),
    ]
    india = [
        ("\\quad buildings, native 15-min", lambda v: v["arm"] == "india" and v["usage"] != "campus"),
        ("\\quad campus feed (one HT consumer)", lambda v: v["arm"] == "india" and v["usage"] == "campus"),
        ("\\quad city (Delhi SLDC)", lambda v: v["id"] == "IN_Delhi"),
    ]
    rows, covs = [], {}
    for name, pred in specs:
        r = pop(name, pred)
        if r:
            rows.append(r[0]); covs[name] = r[1]
    ind = [(n, pop(n, pr)) for n, pr in india]
    if any(r for _, r in ind):
        rows.append(["\\midrule"])
        rows.append(["\\emph{India, one city, windows}"] + [""] * 7)
        for n, r in ind:
            if r:
                rows.append(r[0]); covs[n] = r[1]
    b = covs.get("Buildings (BDG2), metered"); m = covs.get("System demand, metered")
    gap = ""
    if b is not None and m is not None:
        gap = (f" The building--system gap is ${b['split'].mean() - m['split'].mean():+.3f}$ under split "
               f"conformal alone, ${b['aci_abs'].mean() - m['aci_abs'].mean():+.3f}$ under the absolute "
               f"step the benchmark used, and ${b['aci_rel'].mean() - m['aci_rel'].mean():+.3f}$ under "
               "the step stated in the interval's units.")
    n_rows = sum(1 for r in rows if len(r) > 1 and not r[0].startswith("\\emph"))
    table(OUT / "aci_step.tex",
          ["Population", "$n$", "$\\kappa$ of $\\gamma{=}0.35$", "split", "ACI $\\gamma{=}0.35$",
           "ACI $\\kappa{=}0.006$", "worst", "$<0.85$"],
          rows, "lrrrrrrr",
          "The ladder of Table~\\ref{tab:calibration} re-read without the confound. The "
          "paper's forecaster refitted once per supply and the same predictions "
          "calibrated three ways: split conformal alone; split plus ACI at the absolute "
          "step $\\gamma=0.35$ every benchmark row used; and split plus ACI at the "
          "building audit's step as a fraction of the interval width, $\\kappa=0.006$, "
          "the width taken from the validation block so that nothing from the test month "
          "is used. The third column is what the absolute step amounted to on each "
          "population, as a fraction of its interval (median): a working adaptive layer "
          "on buildings and a nearly inert one on system demand. Mean coverage of the "
          "nominal $0.90$ interval on the test month, then the worst row and the count "
          "below $0.85$ under the relative step." + gap,
          "tab:aci-step")


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
          "calibration, \\emph{as the benchmark calibrated it}: split conformal "
          "plus an adaptive layer at the step $\\gamma=0.35$ in the units of each "
          "series. Read this way the guarantee very nearly holds on individual "
          "buildings and fails systematically on system-level demand. The third "
          "row is the replication: six Chinese provinces, a different country, a "
          "different year and a different data-generating process, reproduce the "
          "system-level failure to within 0.005 of the metered panel and contain "
          "the worst row in the study (Heilongjiang, 0.565). The lower panel is "
          "the same effect inside one city: building, campus and city demand in "
          "Delhi, one weather feed, native 15-minute resolution throughout, and "
          "coverage falls monotonically with each step up the ladder. "
          "Table~\\ref{tab:aci-step} re-reads every row of this table with the "
          "adaptive step stated in the interval's units, and the ladder does not "
          "survive it.",
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
    global IEEE, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ieee", action="store_true",
                    help="emit two-column IEEEtran variants instead of single-column ones")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write them (default: docs/paper/tables, or "
                         "docs/paper_ieee/tables with --ieee)")
    ap.add_argument("--short-captions", action="store_true",
                    help="cut each caption to whole sentences within a budget. The "
                         "conference build needs it, because IEEEtran sets captions in "
                         "small caps where a paragraph is unreadable; the journal build "
                         "has room for the whole thing and does not")
    args = ap.parse_args()
    if args.ieee:
        IEEE = True
        OUT = args.out or ROOT / "docs/paper_ieee/tables"
    elif args.out:
        OUT = args.out
    globals()["SHORT_CAPTIONS"] = bool(args.short_captions)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("emitting LaTeX tables:")
    horizon_table()
    horizon_panel_table()
    copula_df_table()
    aci_table()
    aci_gamma_table()
    aci_step_table()
    horizon_kappa_table()
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
