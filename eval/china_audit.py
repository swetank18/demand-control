"""Audit the provenance of the China arm against the data itself.

The China series was added to the study late, on a constraint imposed from
outside it, and it is the only tier-2 source whose documentation describes a
*construction* rather than a measurement. Wu and Kan state their method plainly:
the daily maximum and minimum come from National Development and Reform
Commission reporting, a representative workday profile and a representative
holiday profile are traced pixel by pixel off published load curves with
WebPlotDigitizer, and every day of the year is then that profile rescaled to
that day's envelope.

Taking a third party's stated method on trust is exactly the kind of thing this
repository does not do elsewhere, so this script checks it. The test is direct:
min--max normalise every calendar day's 24-hour profile and count how many
distinct shapes a whole year contains. If the documentation is accurate the
answer is two, and the entire series is recoverable from

    2 shapes x 24 hours  +  365 days x (min, max)  =  778 numbers

against 8,760 hourly values. It is, exactly, in every province tested.

Why this matters enough to be its own script. Forecast skill in this study is a
ratio against each series' own seasonal-naive baseline. On data whose shape is
literally constant, seasonal naive recovers that shape exactly and its only
error is in the daily envelope -- and so is ours. Both models are pinned to the
same two administrative numbers per day, the ratio between them collapses
towards zero, and a skill figure computed on this data says almost nothing about
a forecaster. That has to be established before the China skill column is read,
not after, which is why this runs before the arm is reported.

Writes results/china_audit.json and results/china_audit.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data/raw"
RESULTS = ROOT / "results"

#: the six provinces the study actually uses, in the ladder order of the table
PROVINCES = {
    "HI": "Hainan (Haikou)", "GD": "Guangdong (Guangzhou)", "SH": "Shanghai",
    "BJ": "Beijing", "YN": "Yunnan (Kunming)", "HL": "Heilongjiang (Harbin)",
}

#: Two provinces share the abbreviation ``HB`` (Hebei and Hubei) and pandas
#: renames the second ``HB.1``. None of the six above collides, but selecting by
#: name without knowing that is how a study silently reports the wrong province.
COLLIDING = {"HB"}


def audit_province(s: pd.Series) -> dict:
    days = [g for _, g in s.groupby(s.index.normalize())
            if len(g) == 24 and g.max() > g.min()]
    norm = [(g.to_numpy() - g.min()) / (g.max() - g.min()) for g in days]

    # Group on a rounded key but keep the *unrounded* first member of each group
    # as the exemplar. Rebuilding from a rounded shape would inject 5e-7 of our
    # own error into the answer and make an exact reconstruction look
    # approximate -- which is what a first pass of this script did.
    shapes, keys = [], {}
    for p_ in norm:
        k = np.round(p_, 6).tobytes()
        if k not in keys:
            keys[k] = len(shapes)
            shapes.append(p_)
    shapes = np.vstack(shapes)

    # rebuild the year from the distinct shapes plus each day's envelope
    rebuilt, actual, assign = [], [], []
    for g, p_ in zip(days, norm):
        k = keys[np.round(p_, 6).tobytes()]
        assign.append(k)
        rebuilt.append(shapes[k] * (g.max() - g.min()) + g.min())
        actual.append(g.to_numpy())
    rebuilt, actual = np.concatenate(rebuilt), np.concatenate(actual)
    err = float(np.abs(rebuilt - actual).max() / actual.mean() * 100)

    counts = np.bincount(assign, minlength=len(shapes)).tolist()
    n_params = len(shapes) * 24 + 2 * len(days)
    return {
        "n_days": len(days),
        "n_distinct_shapes": int(len(shapes)),
        "days_per_shape": counts,
        "n_free_parameters": int(n_params),
        "n_hourly_values": int(24 * len(days)),
        "compression_ratio": round(24 * len(days) / n_params, 2),
        "max_reconstruction_error_pct_of_mean": err,
    }


def main() -> None:
    path = RAW / "china_provincial_2018.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run data/national.py first")
    d = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    idx = pd.date_range("2018-01-01 00:00", periods=len(d), freq="h")

    out = {}
    print("China provenance audit -- distinct normalised day-shapes per year")
    for code, name in PROVINCES.items():
        if code in COLLIDING:
            continue
        s = pd.Series(pd.to_numeric(d[code], errors="coerce").to_numpy(),
                      index=idx).dropna()
        r = audit_province(s)
        out[code] = {"name": name, **r}
        print(f"  {name:<24} {r['n_days']:>4} days -> "
              f"{r['n_distinct_shapes']} shapes, "
              f"{r['n_free_parameters']} numbers reproduce "
              f"{r['n_hourly_values']} values, "
              f"max error {r['max_reconstruction_error_pct_of_mean']:.4f}% of mean")

    every = {v["n_distinct_shapes"] for v in out.values()}
    exact = max(v["max_reconstruction_error_pct_of_mean"] for v in out.values())
    summary = {
        "all_provinces_have_same_shape_count": len(every) == 1,
        "shape_count": sorted(every)[0] if len(every) == 1 else None,
        "worst_reconstruction_error_pct": exact,
        #: The day-shapes are not bit-identical: the source publishes about ten
        #: significant figures, so two days' normalised profiles agree to that
        #: and no further. 1e-6 % of the mean is 1e-8 relative, comfortably
        #: inside the published precision and far outside any real mismatch.
        "verdict": ("exact to published precision" if exact < 1e-6
                    else "approximate"),
    }
    (RESULTS / "china_audit.json").write_text(
        json.dumps({"provinces": out, "summary": summary}, indent=1))

    v = next(iter(out.values()))
    md = [
        "### China arm — provenance audit\n",
        "The Chinese provincial series is the only tier-2 source in the study "
        "whose documentation describes a construction rather than a measurement. "
        "Wu and Kan take the daily maximum and minimum from NDRC reporting, trace "
        "a representative workday profile and a representative holiday profile "
        "off published load curves with WebPlotDigitizer, and rescale one of "
        "those two profiles to each day's envelope. This checks that claim "
        "against the data.\n",
        "| Province | Days | Distinct normalised day-shapes | Days per shape | "
        "Numbers needed | Hourly values | Max reconstruction error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for code, r in out.items():
        md.append(
            f"| {r['name']} | {r['n_days']} | **{r['n_distinct_shapes']}** | "
            f"{r['days_per_shape']} | {r['n_free_parameters']} | "
            f"{r['n_hourly_values']} | "
            f"{r['max_reconstruction_error_pct_of_mean']:.6f}% of mean |")
    md += [
        "",
        f"The documentation is accurate and the construction is exact. A whole "
        f"year of hourly demand in each province is {v['n_free_parameters']} "
        f"numbers — {v['n_distinct_shapes']} normalised 24-hour shapes and one "
        f"(min, max) pair per day — reproducing all {v['n_hourly_values']} "
        f"values to within {summary['worst_reconstruction_error_pct'] / 100:.1e} "
        f"of their magnitude -- the precision at which the source is published, "
        f"not an approximation we introduced. A compression of "
        f"{v['compression_ratio']}:1.\n",
        "**What this does to the skill column.** Skill here is a ratio against "
        "each series' own seasonal-naive baseline. When every workday shares one "
        "shape, seasonal naive recovers that shape exactly and its whole error is "
        "the daily envelope; so is ours. Both models are pinned to the same two "
        "administrative numbers per day, the ratio collapses toward zero, and a "
        "skill figure on this data is a statement about the data rather than "
        "about a forecaster. The China skill column is reported for completeness "
        "and is not evidence either way about the method.\n",
        "**What survives.** Coverage is a property of interval width against "
        "realised error and does not run through the baseline, so the tier-2 "
        "under-coverage result can be read on this arm — and it replicates, on "
        "an error process unlike any other in the study. So does the null "
        "between forecast skill and climate.\n",
    ]
    (RESULTS / "china_audit.md").write_text("\n".join(md))
    print(f"\n-> results/china_audit.json, results/china_audit.md")


if __name__ == "__main__":
    main()
