"""Which sites can supply a building comparable to the ones we already study?

The repo's claim rests on four buildings at one site in Tempe, Arizona. The
question a reviewer actually asks is whether the method is general or whether it
is a fit to Phoenix. Answering that needs comparable buildings elsewhere, and
"comparable" has to mean the same stated rule, not a hand-pick.

So this applies `data/buildings.json`'s selection rule verbatim to all nineteen
BDG2 sites:

  no district chilled-water meter   (the electricity meter must contain the
                                     cooling we intend to control)
  coverage > 97% in the split window
  median load 40-800 kW
  positive load-vs-outdoor-temperature correlation

The fourth clause is doing something the original rule never had to think about.
At Fox every candidate is cooling-dominated, so "positive correlation" is a
sanity check. Applied to Minneapolis or Dublin it is a *filter that removes the
entire site*, because those buildings burn electricity when it is cold. That is
not a bug in the rule; it is the rule discovering that the premise underneath it
is regional. So both are reported: buildings that pass, and buildings that fail
only on the correlation clause, which are the heating-dominated ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

# the split window, imported rather than retyped
import sys
sys.path.insert(0, str(ROOT.parent))
from forecast.splits import SPLIT

WINDOW = (pd.Timestamp(SPLIT.train_start), pd.Timestamp(SPLIT.test_end))

MIN_COVERAGE = 0.97
LOAD_BAND_KW = (40.0, 800.0)

COUNTRY = {"Europe/London": "UK", "Europe/Dublin": "Ireland"}
# Crow and Moose sit on Ottawa coordinates but carry a US/Eastern tz string.
CANADA_SITES = {"Crow", "Moose"}


def country_of(site: str, tz: str) -> str:
    if site in CANADA_SITES:
        return "Canada"
    return COUNTRY.get(tz, "USA")


def load_meta() -> pd.DataFrame:
    m = pd.read_csv(RAW / "metadata.csv", low_memory=False)
    m["country"] = [country_of(s, t) for s, t in zip(m.site_id, m.timezone)]
    return m


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"])
    w = w[(w.timestamp >= WINDOW[0]) & (w.timestamp <= WINDOW[1])]
    return w


def climate_table(w: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-site climate. CDD/HDD are what decide whether our premise holds."""
    rows = []
    for site, g in w.groupby("site_id"):
        t = g.set_index("timestamp")["airTemperature"].dropna()
        if t.empty:
            continue
        daily = t.resample("D").mean().dropna()
        rows.append({
            "site": site,
            "country": country_of(site, meta.loc[meta.site_id == site, "timezone"].iloc[0]
                                  if (meta.site_id == site).any() else ""),
            "t_mean": t.mean(),
            "t_p99": t.quantile(0.99),
            "t_p01": t.quantile(0.01),
            # base 18 C, the convention
            "cdd": float(np.maximum(daily - 18.0, 0).sum()),
            "hdd": float(np.maximum(18.0 - daily, 0).sum()),
            "n_days": len(daily),
        })
    c = pd.DataFrame(rows).set_index("site")
    c["cdd_share"] = c.cdd / (c.cdd + c.hdd)
    return c.sort_values("cdd_share", ascending=False)


def screen(meta: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Apply the selection rule to every metered building in BDG2."""
    cand = meta[meta.electricity.notna()].copy()
    ids = [b for b in cand.building_id]

    # one pass over the meter file, only the columns we need
    head = pd.read_csv(RAW / "electricity_cleaned.csv", nrows=0).columns.tolist()
    ids = [b for b in ids if b in head]
    df = pd.read_csv(
        RAW / "electricity_cleaned.csv",
        usecols=["timestamp"] + ids,
        parse_dates=["timestamp"], index_col="timestamp",
    ).sort_index()
    df = df.loc[WINDOW[0]:WINDOW[1]]

    wt = {s: g.set_index("timestamp")["airTemperature"] for s, g in w.groupby("site_id")}

    rows = []
    meta_i = cand.set_index("building_id")
    for bid in ids:
        s = df[bid]
        cov = float(s.notna().mean())
        med = float(s.median()) if s.notna().any() else np.nan
        site = meta_i.loc[bid, "site_id"]
        corr = np.nan
        if site in wt and s.notna().sum() > 500:
            t = wt[site].reindex(s.index)
            ok = s.notna() & t.notna()
            if ok.sum() > 500:
                corr = float(np.corrcoef(s[ok], t[ok])[0, 1])
        rows.append({
            "building_id": bid,
            "site": site,
            "country": meta_i.loc[bid, "country"],
            "usage": meta_i.loc[bid, "primaryspaceusage"],
            "sqm": meta_i.loc[bid, "sqm"],
            "has_chilledwater": pd.notna(meta_i.loc[bid, "chilledwater"]),
            "coverage": cov,
            "median_kw": med,
            "p99_over_median": float(s.quantile(0.99) / med) if med and med > 0 else np.nan,
            "corr_temp": corr,
        })
    r = pd.DataFrame(rows)

    r["pass_chw"] = ~r.has_chilledwater
    r["pass_cov"] = r.coverage > MIN_COVERAGE
    r["pass_load"] = r.median_kw.between(*LOAD_BAND_KW)
    r["pass_corr"] = r.corr_temp > 0
    r["passes_rule"] = r.pass_chw & r.pass_cov & r.pass_load & r.pass_corr
    # fails ONLY the correlation clause -> heating-dominated but otherwise usable
    r["heating_dominated"] = r.pass_chw & r.pass_cov & r.pass_load & ~r.pass_corr
    return r


def main() -> None:
    meta = load_meta()
    w = load_weather()
    clim = climate_table(w, meta)
    r = screen(meta, w)

    out = ROOT / "site_survey.csv"
    r.to_csv(out, index=False)

    per_site = r.groupby("site").agg(
        country=("country", "first"),
        metered=("building_id", "size"),
        passes=("passes_rule", "sum"),
        heating=("heating_dominated", "sum"),
        med_corr=("corr_temp", "median"),
    )
    per_site = per_site.join(clim[["t_mean", "cdd", "hdd", "cdd_share"]])
    per_site = per_site.sort_values("cdd_share", ascending=False)

    print("=" * 100)
    print("BDG2 sites against the Fox selection rule   (window %s to %s)"
          % (WINDOW[0].date(), WINDOW[1].date()))
    print("=" * 100)
    print(per_site.to_string(float_format=lambda x: f"{x:.2f}"))
    print()
    print("passes      = no chilled-water meter, coverage >97%, median 40-800 kW, positive load-temp corr")
    print("heating     = passes everything EXCEPT the positive-correlation clause")
    print()
    print("-" * 100)
    print("Demographic spread among buildings that pass, by country")
    print("-" * 100)
    p = r[r.passes_rule]
    print(pd.crosstab(p.usage, p.country).to_string())
    print()
    print(f"total passing the rule: {int(r.passes_rule.sum())} across "
          f"{p.site.nunique()} sites and {p.country.nunique()} countries")
    print(f"heating-dominated but otherwise usable: {int(r.heating_dominated.sum())} across "
          f"{r[r.heating_dominated].site.nunique()} sites")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
