"""Tier 1, India: I-BLEND, the IIIT-Delhi campus (Rashid, Singh & Singh, Sci. Data 2019).

The paper's Section 9 said the thing India must fix first is that no Indian
building-level load dataset exists. That was wrong, and this file is the
correction: I-BLEND is 52 months (2013-08 to 2017-12) of one-minute mains power
for seven buildings on the IIIT-Delhi campus, plus the three transformers that
feed the whole campus, CC0, on figshare. It is the only open Indian building
corpus at a resolution finer than the 15- and 30-minute blocks Indian demand
charges are assessed on, which means -- unlike every BDG2 building in this
study -- the billing blocks here are formed by *averaging* rather than by
interpolating up from hourly data that has already smoothed the peak.

Three levels of aggregation, one city, one weather feed:

  building   the seven mains meters. Academic, Library, Lecture, Dining and both
             dormitories are on a *central chiller plant that is not on their
             mains meter*; Facilities runs seven window ACs and is the one
             building whose meter contains its own cooling. So at building level
             this corpus is a forecasting and calibration benchmark, not a
             control one, and the selection rule's chilled-water clause says so.
  campus     the sum of the three transformer feeds. An Indian campus is billed
             as one HT consumer with one contract demand and one monthly demand
             charge, so this is the object the tariff actually sees -- and it
             contains the chiller. This is where the closed loop can run.
  city       Delhi SLDC, already in data/national.py.

**The selection rule, applied and reported rather than bent.** The rule in
data/buildings.json was written for BDG2: no district chilled-water meter,
>97% coverage in the split window, median load 40-800 kW, positive load-vs-
temperature correlation. Applied here, every building fails the size clause
(the largest median is 25 kW) and six of seven fail the chilled-water clause.
The manifest records each clause for each series. What we do about it is the
same thing the paper did when the correlation clause turned out to be a Phoenix
assumption in Dublin: state the clause that does not travel, and say what is
readable on the rows that fail it. Coverage and the horizon gap do not need a
chiller on the meter. A savings claim does.

**Windows.** The protocol is 15 months train, April-May calibrate, June test.
Meter uptime varies by feed, so for each series and each candidate June the
script finds the earliest train start from January of the previous year that
gives >=97% coverage through June 30, requiring at least three months of
training as the China arm did. Series with no admissible June are kept in the
cache and marked inadmissible in the manifest.

Usage:
    python data/iblend.py            # fetch (589 MB, once), build, write manifest
    python data/iblend.py --survey   # print the rule table only
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "iblend"
CACHE = ROOT / "cache"
sys.path.insert(0, str(ROOT))
from national import open_meteo, _download                      # noqa: E402
from prepare import (                                            # noqa: E402
    ASSUMED_COP, ASSUMED_C_KJ_PER_M2_K, UA_SANITY_BAND_W_PER_M2K, FALLBACK_UA_W_PER_M2K,
    T_DESIGN_INDOOR, DESIGN_MARGIN, _fit_changepoint,
)

FIGSHARE_ZIP = "https://ndownloader.figshare.com/files/10797959"   # article 6007637, CC0
LAT, LON = 28.5463, 77.2732                                        # Okhla Phase III, New Delhi
TZ = "Asia/Kolkata"

#: Floor areas from the dataset paper (Table 1), ft2 -> m2.
SQFT = {"Academic": 61308, "Lecture": 17548, "Library": 26401, "Facilities": 9769,
        "Mess": 50191, "Boys": 72745, "Girls": 38126}
SQM = {k: v * 0.092903 for k, v in SQFT.items()}
SQM["Campus"] = sum(SQM.values())

#: series -> the columns of all_buildings_power.csv (or the transformer file) it sums.
#: A dormitory is one building on two feeds (mains + UPS backup), so it is their
#: sum, valid only where both are present.
SERIES = {
    "IIITD_Academic":   {"cols": ["Academic"],                  "area": "Academic",   "cooling_on_meter": False},
    "IIITD_Library":    {"cols": ["Library"],                   "area": "Library",    "cooling_on_meter": False},
    "IIITD_Lecture":    {"cols": ["Lecture"],                   "area": "Lecture",    "cooling_on_meter": False},
    "IIITD_Dining":     {"cols": ["Mess"],                      "area": "Mess",       "cooling_on_meter": False},
    "IIITD_Facilities": {"cols": ["Facilities"],                "area": "Facilities", "cooling_on_meter": True},
    "IIITD_Boys":       {"cols": ["Boys_main", "Boys_backup"],  "area": "Boys",       "cooling_on_meter": False},
    "IIITD_Girls":      {"cols": ["Girls_main", "Girls_backup"], "area": "Girls",     "cooling_on_meter": False},
    "IIITD_Campus":     {"cols": ["transfomer_1", "transfomer_2", "transfomer_3"], "area": "Campus",
                         "cooling_on_meter": True, "file": "all_transformer_power.csv"},
}
MIN_MINUTES_PER_BLOCK = 8          # a 15-min block is valid if >= 8 of its minutes were metered
COVERAGE_FLOOR = 0.97              # the rule's coverage clause
MEDIAN_BAND_KW = (40.0, 800.0)     # the rule's size clause
CANDIDATE_JUNES = (2014, 2015, 2016, 2017)
MIN_TRAIN_MONTHS = 3               # the China-arm precedent


def fetch() -> Path:
    d = RAW / "energy_dataset"
    if (d / "all_buildings_power.csv").exists():
        return d
    z = _download(FIGSHARE_ZIP, RAW / "energy_dataset.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(RAW)
    return d


def _read_minutes(path: Path) -> pd.DataFrame:
    """One-minute watts, IST wall clock, reindexed to a complete minute grid."""
    d = pd.read_csv(path)
    ts = pd.to_datetime(d.timestamp, unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    d = d.drop(columns="timestamp").set_index(ts).sort_index()
    d = d[~d.index.duplicated()]
    full = pd.date_range(d.index.min().floor("min"), d.index.max().ceil("min"), freq="min")
    return d.reindex(full)


def to_blocks(watts: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum of feeds, 15-minute mean kW, NaN where any feed is short of minutes."""
    kw = watts[cols] / 1000.0
    mean = kw.resample("15min").mean()
    n = kw.resample("15min").count()
    ok = (n >= MIN_MINUTES_PER_BLOCK).all(axis=1)
    return mean.sum(axis=1).where(ok)


def admissible_windows(valid: pd.Series) -> list[dict]:
    """For each candidate June, the earliest train start that clears the coverage
    floor through June 30, with at least MIN_TRAIN_MONTHS of training."""
    out = []
    for y in CANDIDATE_JUNES:
        test_end = pd.Timestamp(f"{y}-06-30 23:45")
        if valid.index.max() < test_end:
            continue
        found = None
        for m in range(0, 13):                            # Jan(y-1) .. Jan(y)
            ts = pd.Timestamp(f"{y - 1}-01-01") + pd.DateOffset(months=m)
            if ts < valid.index.min().normalize():
                continue
            w = valid.loc[ts:test_end]
            if len(w) and w.mean() >= COVERAGE_FLOOR:
                found = ts
                break
        if found is None:
            continue
        train_months = (pd.Timestamp(f"{y}-04-01") - found).days // 30
        if train_months < MIN_TRAIN_MONTHS:
            continue
        out.append({
            "test_june": y, "train_start": str(found), "train_end": f"{y}-03-31 23:45",
            "valid_start": f"{y}-04-01 00:00", "valid_end": f"{y}-05-31 23:45",
            "test_start": f"{y}-06-01 00:00", "test_end": f"{y}-06-30 23:45",
            "train_months": int(train_months),
            "coverage": float(valid.loc[found:test_end].mean()),
            "full_protocol": bool(found <= pd.Timestamp(f"{y - 1}-01-01")),
        })
    return out


def thermal_params(out: pd.DataFrame, slope: float, sqm: float) -> tuple[dict, float, str]:
    """Identical arithmetic to data/prepare.py, so a campus row is comparable to a BDG2 row."""
    ua_kw_per_k = slope * ASSUMED_COP
    ua_w_per_m2k = ua_kw_per_k * 1000.0 / sqm
    ua_source = "fitted from the series' own meter (changepoint slope x COP)"
    if not (UA_SANITY_BAND_W_PER_M2K[0] <= ua_w_per_m2k <= UA_SANITY_BAND_W_PER_M2K[1]):
        ua_kw_per_k = FALLBACK_UA_W_PER_M2K * sqm / 1000.0
        ua_source = (f"HAND-PICKED {FALLBACK_UA_W_PER_M2K} W/m2K: the fitted value "
                     f"({ua_w_per_m2k:.2f} W/m2K) fell outside the believable band")
        ua_w_per_m2k = FALLBACK_UA_W_PER_M2K
    r_k_per_kw = 1.0 / max(ua_kw_per_k, 1e-6)
    c_kwh_per_k = ASSUMED_C_KJ_PER_M2_K * sqm / 3600.0
    t_design = float(out.t_out.quantile(0.995))
    q_int_design = 0.85 * float(out.base_kw.quantile(0.95)) + 0.05 * sqm * 0.100
    hvac_capacity_kw = float(np.ceil(
        DESIGN_MARGIN * (ua_kw_per_k * max(t_design - T_DESIGN_INDOOR, 0.0) + q_int_design)
        / ASSUMED_COP / 5.0) * 5.0)
    thermal = {"ua_kw_per_k": ua_kw_per_k, "ua_w_per_m2k": ua_w_per_m2k, "ua_source": ua_source,
               "r_k_per_kw": r_k_per_kw, "c_kwh_per_k": c_kwh_per_k, "cop": ASSUMED_COP,
               "time_constant_h": r_k_per_kw * c_kwh_per_k}
    sizing = (f"engineering sizing at design outdoor {t_design:.1f} C holding {T_DESIGN_INDOOR:.0f} C: "
              f"envelope {ua_kw_per_k * (t_design - T_DESIGN_INDOOR):.0f} kW + internal {q_int_design:.0f} kW "
              f"thermal, / COP {ASSUMED_COP}, x {DESIGN_MARGIN} margin")
    return thermal, hvac_capacity_kw, sizing


def build(name: str, spec: dict, watts: dict[str, pd.DataFrame], weather: pd.DataFrame) -> dict:
    src = spec.get("file", "all_buildings_power.csv")
    total = to_blocks(watts[src], spec["cols"])
    valid = total.notna()
    idx = total.index

    # ERA5 is UTC; the meter is IST wall clock. Shift the weather, not the meter,
    # so the calendar features see the day the occupants saw.
    w = weather.copy()
    w.index = w.index + pd.Timedelta(hours=5, minutes=30)
    w = w.resample("15min").interpolate(limit=8).reindex(idx).ffill(limit=8).bfill(limit=8)

    # Changepoint on hourly means, as prepare.py does on hourly BDG2 rows.
    hr = pd.concat([total.resample("h").mean().rename("y"), w.t_out.resample("h").mean().rename("t")], axis=1).dropna()
    base_kw, slope, t_balance = _fit_changepoint(hr.y, hr.t)
    cooling = (slope * np.maximum(0.0, w.t_out - t_balance)).clip(lower=0.0)
    cooling = np.minimum(cooling, total * 0.85)
    base = (total - cooling).clip(lower=0.0)
    hvac_hist = (total - base).clip(lower=0.0)

    out = pd.DataFrame({"total_kw": total, "base_kw": base, "hvac_kw_hist": hvac_hist,
                        "t_out": w.t_out, "t_dew": w.t_dew, "cloud": w.cloud}).dropna()
    out.index.name = "timestamp"
    out.to_parquet(CACHE / f"{name}.parquet")

    sqm = SQM[spec["area"]]
    thermal, hvac_capacity_kw, sizing = thermal_params(out, slope, sqm)
    hist_peak_kva = float(out.total_kw.resample("30min").mean().max() / 0.95)
    windows = admissible_windows(valid)
    med = float(out.total_kw.median())
    corr = float(np.corrcoef(hr.y, hr.t)[0, 1])

    # A meter that reads zero most of the time is present but not alive; the
    # Lecture feed is one, and coverage alone would admit it.
    dead_frac = float((out.total_kw <= 0.05).mean())
    alive = med >= 1.0 and dead_frac < 0.5
    if not alive:
        windows = []
    rule = {
        "meter_alive": bool(alive),
        "no_district_chilled_water": bool(spec["cooling_on_meter"]),
        "coverage_ge_97pct_in_some_window": bool(windows),
        "median_40_800_kw": bool(MEDIAN_BAND_KW[0] <= med <= MEDIAN_BAND_KW[1]),
        "positive_load_temp_corr": bool(corr > 0),
    }
    rule["passes_bdg2_rule"] = all(rule.values())
    rule["readable"] = {
        "forecast_skill_and_coverage": bool(windows),
        "horizon_gap": bool(windows),
        "closed_loop_control": bool(windows and spec["cooling_on_meter"]),
    }

    return {
        "id": name, "country": "IN", "tier": 1, "site": "IIITD", "source": "iblend",
        "level": "campus" if name == "IIITD_Campus" else "building",
        "label": name.replace("IIITD_", "IIIT-Delhi "),
        "feeds": spec["cols"], "sqm": sqm,
        "native_resolution_min": 1, "upsampled": False,
        "block_rule": f"15-min mean of 1-min watts; block valid if >= {MIN_MINUTES_PER_BLOCK}/15 minutes on every feed",
        "window": [str(out.index.min()), str(out.index.max())],
        "coverage_overall": float(valid.mean()),
        "n_intervals": int(len(out)),
        "changepoint": {"base_kw": base_kw, "slope_kw_per_k": slope, "t_balance_c": t_balance},
        "thermal": thermal, "hvac_capacity_kw": hvac_capacity_kw, "hvac_sizing": sizing,
        "hvac_capacity_from_meter_kw": float(out.hvac_kw_hist.quantile(0.995)),
        "hist_peak_kva": hist_peak_kva,
        "contract_demand_kva": float(np.ceil(hist_peak_kva * 1.05 / 5) * 5),
        "hvac_share_of_meter": float(out.hvac_kw_hist.sum() / max(out.total_kw.sum(), 1e-9)),
        "median_kw": med, "peak_kw": float(out.total_kw.max()), "dead_block_frac": dead_frac,
        "p99_over_median": float(out.total_kw.quantile(0.99) / max(med, 1e-9)),
        "corr_temp": corr,
        "cooling_on_meter": bool(spec["cooling_on_meter"]),
        "cooling_note": ("seven window AC units, on this meter" if name == "IIITD_Facilities" else
                         "central chiller plant, on the campus feed" if name == "IIITD_Campus" else
                         "served by the central chiller plant, which is NOT on this meter"),
        "rule": rule,
        "windows": windows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true", help="print the rule table, write nothing")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    d = fetch()
    CACHE.mkdir(parents=True, exist_ok=True)
    watts = {f: _read_minutes(d / f) for f in ("all_buildings_power.csv", "all_transformer_power.csv")}
    weather = open_meteo(LAT, LON, "2013-08-01", "2017-12-31", "IN_IIITD")

    manifest = {
        "_": "I-BLEND (IIIT-Delhi), tier 1 India. See the module docstring.",
        "source": {"paper": "Rashid, Singh & Singh, Scientific Data 6:190015 (2019)",
                   "data": "figshare 10.6084/m9.figshare.6007637 (CC0)", "fetched_from": FIGSHARE_ZIP},
        "location": {"lat": LAT, "lon": LON, "tz": TZ, "weather": "ERA5 via Open-Meteo, hourly"},
        "assumptions": {"cop": ASSUMED_COP, "capacitance_kJ_per_m2_K": ASSUMED_C_KJ_PER_M2_K,
                        "decomposition": "changepoint regression on outdoor air temperature",
                        "block_rule": f">= {MIN_MINUTES_PER_BLOCK}/15 minutes present on every feed"},
        "selection_rule": ("identical to data/buildings.json, applied and reported per clause; "
                           "the size clause is a BDG2 artefact and fails every building here"),
        "series": {},
    }
    for name, spec in SERIES.items():
        if args.only and name not in args.only:
            continue
        m = build(name, spec, watts, weather)
        manifest["series"][name] = m
        r = m["rule"]
        wins = ", ".join(f"Jun{w['test_june']}({w['train_months']}mo)" for w in m["windows"]) or "none"
        print(f"  {name:<17} {'alive' if r['meter_alive'] else 'DEAD '} med {m['median_kw']:6.1f} kW  p99/med {min(m['p99_over_median'], 99):5.2f}  "
              f"corr {m['corr_temp']:+.2f}  cool-on-meter {str(r['no_district_chilled_water']):<5} "
              f"size {str(r['median_40_800_kw']):<5} windows: {wins}")

    if not args.survey:
        (CACHE / "manifest_iblend.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nwrote {len(manifest['series'])} series -> {CACHE / 'manifest_iblend.json'}")


if __name__ == "__main__":
    main()
