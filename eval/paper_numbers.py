"""Check the numbers the paper's *prose* states against the results files.

Tables and figures cannot drift: a script writes them. Prose can. A sentence
saying "rank correlation 0.957" is typed once and then the study is rerun, and
nothing complains. This is the list of every figure the text asserts, with
where it comes from, so a reader of this file can see what the paper is
claiming and a run of it can catch a claim that has gone stale.

Add a line here whenever the text states a new number. Run before submitting.

    ../.venv/bin/python eval/paper_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def h64(d: dict) -> dict:
    return [x for x in d["marginal_vs_joint"] if x["H"] == 64][0]


def _dedup(store: dict) -> list[dict]:
    out, seen = [], set()
    for v in store.values():
        key = (v.get("id"), v.get("test_june"))
        if "calib" in v and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def claims() -> list[tuple[str, float, float, float]]:
    """(what the text says, the figure it states, the figure in results, tolerance)."""
    fox = load("horizon_risk_Fox_office_Gaylord.json")
    f64, fa = h64(fox), fox["acceptance"]
    camp = load("horizon_risk_IIITD_Campus@2017.json")
    c64, ca = h64(camp), camp["acceptance"]
    step = _dedup(load("aci_step_check.json"))
    kapc = load("horizon_risk_IIITD_Campus@2017k.json")

    def pop(pred, col):
        g = [v for v in step if pred(v)]
        return float(np.mean([v["calib"][col]["coverage_90"] for v in g]))

    bld = lambda v: v["tier"] == 1 and v["arm"] != "india"
    sysm = lambda v: v["arm"] == "national" and not v["id"].startswith("CN_")
    recon = lambda v: v["arm"] == "national" and v["id"].startswith("CN_")

    panel = [load(p.name) for p in
             [R / "horizon_risk_Fox_office_Gaylord.json"]
             + sorted(R.glob("horizon_risk_IIITD_*.json"))
             if "@" not in p.name or p.name.split("@")[1][:-5].isdigit()]
    emp = np.array([h64(d)["empirical_horizon"] for d in panel])
    per = np.array([h64(d)["per_step_exceedance"] for d in panel])
    lo = np.array([h64(d)["empirical_horizon_ci"][0] for d in panel])

    kap = [load(p.name) for p in sorted(R.glob("horizon_risk_*k.json"))]
    empk = np.array([h64(d)["empirical_horizon"] for d in kap])

    return [
        # Section: the horizon gap
        ("Fox realised horizon exceedance 0.405", 0.405, f64["empirical_horizon"], 5e-4),
        ("Fox independence figure 0.983", 0.983, f64["independence_bound"], 5e-4),
        ("Fox copula prediction 0.497", 0.497, f64["copula_predicted"], 5e-4),
        ("panel windows = 14", 14, len(panel), 0),
        ("panel realised min 0.393", 0.393, emp.min(), 5e-4),
        ("panel realised max 0.818", 0.818, emp.max(), 5e-4),
        ("panel realised median 0.565", 0.565, float(np.median(emp)), 5e-4),
        ("panel per-step min 0.032", 0.032, per.min(), 5e-4),
        ("panel per-step max 0.079", 0.079, per.max(), 6e-4),
        ("panel lowest bootstrap bound 0.28", 0.28, lo.min(), 5e-3),
        # Section: the horizon gap, robustness to the step
        ("kappa windows = 14", 14, len(kap), 0),
        ("kappa realised min 0.254", 0.254, empk.min(), 5e-4),
        ("kappa realised max 0.660", 0.660, empk.max(), 5e-4),
        ("kappa realised median 0.440", 0.440, float(np.median(empk)), 5e-4),
        # Section: the dial
        ("Fox rank correlation 0.957", 0.957, fa["rank_corr"], 5e-4),
        ("Fox mean absolute gap 0.057", 0.057, fa["mean_abs_gap"], 5e-4),
        ("Fox conservative at 5 of 8", 5, fa["n_conservative"], 0),
        ("Fox tight bill delta +3339", 3339.0, fa["tight_bill_delta"], 1.0),
        ("campus rank correlation 0.988", 0.988, ca["rank_corr"], 5e-4),
        ("campus mean absolute gap 0.100", 0.100, ca["mean_abs_gap"], 5e-4),
        ("campus conservative at 0 of 8", 0, ca["n_conservative"], 0),
        ("campus bill delta -3329", -3329.0, ca["tight_bill_delta"], 1.0),
        ("campus marginal commit 0.297", 0.2972, ca["marginal_plan_rate"]["tight"], 5e-4),
        ("campus per-step 0.079", 0.079, c64["per_step_exceedance"], 6e-4),
        ("resolution floor 1/S = 0.025", 0.025, fa["resolution_floor"], 1e-9),
        # Section: the dial, tested against its own explanation
        ("campus at kappa, mean abs gap 0.083", 0.083, kapc["acceptance"]["mean_abs_gap"], 5e-4),
        ("campus at kappa, rank corr 0.988", 0.988, kapc["acceptance"]["rank_corr"], 5e-4),
        ("campus at kappa, conservative 0 of 8", 0, kapc["acceptance"]["n_conservative"], 0),
        ("campus at kappa, per-step 0.069", 0.069, h64(kapc)["per_step_exceedance"], 6e-4),
        ("campus at kappa, commit at eps=0.05 0.158",
         0.158, kapc["acceptance"]["scenario_plan_rate_at_min_eps"]["tight"], 5e-4),
        # Section: what degrades with aggregation
        ("buildings, split only 0.820", 0.820, pop(bld, "split"), 5e-4),
        ("system metered, split only 0.791", 0.791, pop(sysm, "split"), 5e-4),
        ("system reconstructed, split only 0.785", 0.785, pop(recon, "split"), 5e-4),
        ("buildings, relative step 0.898", 0.898, pop(bld, "aci_rel"), 5e-4),
        ("system metered, relative step 0.914", 0.914, pop(sysm, "aci_rel"), 5e-4),
        ("system reconstructed, relative step 0.925", 0.925, pop(recon, "aci_rel"), 5e-4),
    ]


def main() -> int:
    rows = claims()
    bad = 0
    for text, stated, actual, tol in rows:
        ok = abs(stated - actual) <= tol
        bad += not ok
        print(f"{'  ' if ok else '! '}{text:<42} text {stated:>10.4f}   results {actual:>10.4f}")
    print(f"\n{len(rows) - bad} of {len(rows)} prose figures agree with results/")
    if bad:
        print("A '!' row means the text says something the study no longer produces.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
