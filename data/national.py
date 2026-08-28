"""Tier 2: real national and city demand, in the countries the project is about.

BDG2 is the only harmonised multi-country *building* dataset in existence, and
it contains no Indian buildings. That is a hard limit, and pretending otherwise
by relabelling an American building as Indian would be worse than the gap. So
the comparative study is split in two, and this file is the second half.

  Tier 1  building-level, BDG2. Same rule, same pipeline, three countries.
          This is where the *control* claim can be tested, because controlling
          anything requires building-level HVAC.

  Tier 2  system-level demand, real per-country sources including India. No
          building to control here -- but the thing the judge asked to
          benchmark is the *model*, and a quantile load forecaster is exactly
          as testable on a city's demand as on a building's.

Sources, all public and all fetched rather than vendored:

  India   Delhi SLDC, 15-minute, via the Ecohen4/Delhi archive. Two complete
          years (2011, 2012) at 100% coverage, five distribution companies plus
          the total. This is the repo's first real Indian load data.
  Europe  Open Power System Data's ENTSO-E mirror, hourly, no API key, 2015-2020.
          GB, IE, DE, FR, ES.
  Weather Open-Meteo's ERA5 archive, hourly, free, no key.

**The window problem, stated rather than buried.** Delhi is 2011-2012; OPSD
starts in 2015. There is no overlapping year, so absolute pinball loss is not
comparable across the two groups -- different years, different economies,
different weather. Every cross-country claim in the study is therefore made on
*skill relative to that series' own seasonal-naive baseline*, which is a ratio
computed inside one country's own data and is comparable across years. Absolute
figures are printed too, greyed out in the sense that nothing is concluded from
them alone.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CACHE = ROOT / "cache"

DELHI_URL = "https://raw.githubusercontent.com/Ecohen4/Delhi/master/data/Delhi-SLDC-data-15min-master.csv"
OPSD_URL = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
CHINA_URL = ("https://zenodo.org/records/8322210/files/"
             "Appendix%201_Hourly%20electric%20power%20load%20final.csv?download=1")

#: name -> (iso country for holidays, lat, lon, source key, window)
#: Capital-city temperature is a proxy for a national load-weighted temperature.
#: It is a good one for GB and IE and a rougher one for DE/FR/ES, which is said
#: here and repeated in the study.
#: China enters as a within-country climate ladder rather than one national
#: series, because 31 provinces spanning Hainan to Heilongjiang is a wider
#: climate range than the whole European panel and it is a *developing-economy*
#: range, which is the comparison India actually needs.
#:
#: **Provenance warning, and it is not a small one.** Unlike Delhi (metered by
#: the State Load Despatch Centre) and the European series (metered by
#: transmission operators), the Chinese series is *reconstructed*: the authors
#: take each day's maximum and minimum from NDRC reporting, trace a
#: representative workday profile and a representative holiday profile off
#: published load curves with WebPlotDigitizer, and rescale one of those two
#: profiles to each day's envelope. `eval/china_audit.py` checks that against
#: the data rather than taking it on trust, and finds it exact: a whole year of
#: hourly demand in every province is 778 numbers -- two normalised day-shapes
#: and 365 (min, max) pairs -- reproducing all 8,760 values to machine
#: precision.
#:
#: The consequence is specific, and it is not the obvious one. It does not make
#: the series look unusually forecastable to *us*; it makes the seasonal-naive
#: baseline we are scored against recover the shape exactly too. Both models
#: collapse onto predicting the same two administrative numbers per day, so the
#: skill ratio between them collapses toward zero and says nothing about a
#: forecaster. Coverage does not run through the baseline and stays readable.
#: The arm is included with that stated, because excluding the second-largest
#: electricity system on earth would be the worse error, and because what it
#: turned out to measure is worth reporting on its own.
CHINA = {
    "CN_Hainan":       ("HI", 20.04, 110.32, "Haikou -- tropical, the most cooling-driven province"),
    "CN_Guangdong":    ("GD", 23.13, 113.26, "Guangzhou -- subtropical, the closest Chinese analogue to an Indian metro"),
    "CN_Shanghai":     ("SH", 31.23, 121.47, "humid subtropical, hot summer and cold winter"),
    "CN_Yunnan":       ("YN", 25.04, 102.72, "Kunming -- mild highland, little cooling and little heating"),
    "CN_Beijing":      ("BJ", 39.90, 116.41, "cold continental, heating-dominated"),
    "CN_Heilongjiang": ("HL", 45.80, 126.53, "Harbin -- severe cold, the coldest province in the set"),
}

PANEL = {
    "IN_Delhi":  dict(country="IN", lat=28.61, lon=77.21, source="delhi",
                      col="Total", start="2011-01-01", end="2012-12-31",
                      note="Delhi city total, 15-min native, all five discoms summed"),
    "GB_UKM":    dict(country="GB", lat=51.51, lon=-0.13, source="opsd",
                      col="GB_UKM_load_actual_entsoe_transparency",
                      start="2016-01-01", end="2017-06-30",
                      note="Great Britain national, hourly, capital-temperature proxy"),
    "IE":        dict(country="IE", lat=53.35, lon=-6.26, source="opsd",
                      col="IE_load_actual_entsoe_transparency",
                      start="2016-01-01", end="2017-06-30",
                      note="Ireland national, hourly, capital-temperature proxy"),
    "DE":        dict(country="DE", lat=52.52, lon=13.40, source="opsd",
                      col="DE_load_actual_entsoe_transparency",
                      start="2016-01-01", end="2017-06-30",
                      note="Germany national, hourly, capital-temperature proxy is rough"),
    "FR":        dict(country="FR", lat=48.86, lon=2.35, source="opsd",
                      col="FR_load_actual_entsoe_transparency",
                      start="2016-01-01", end="2017-06-30",
                      note="France national, hourly. Electric-heating dominated."),
    "ES":        dict(country="ES", lat=40.42, lon=-3.70, source="opsd",
                      col="ES_load_actual_entsoe_transparency",
                      start="2016-01-01", end="2017-06-30",
                      note="Spain national, hourly, capital-temperature proxy is rough"),
    **{name: dict(country="CN", lat=lat, lon=lon, source="china", col=code,
                  start="2018-01-01", end="2018-12-31",
                  note=f"China, {note}. RECONSTRUCTED from digitised load curves, not metered.")
       for name, (code, lat, lon, note) in CHINA.items()},
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def open_meteo(lat: float, lon: float, start: str, end: str, tag: str) -> pd.DataFrame:
    """Hourly ERA5 temperature/dewpoint/cloud. Cached to disk per location."""
    cache = RAW / f"weather_{tag}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,dew_point_2m,cloud_cover",
        "timezone": "UTC",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
    print(f"  weather {tag} ...", flush=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                j = json.loads(r.read())
            break
        except Exception as e:                       # rate limit or transient
            if attempt == 3:
                raise
            print(f"    retry {attempt+1} ({e})", flush=True)
            time.sleep(5 * (attempt + 1))
    h = j["hourly"]
    w = pd.DataFrame({
        "t_out": h["temperature_2m"],
        "t_dew": h["dew_point_2m"],
        "cloud": [c / 100.0 if c is not None else np.nan for c in h["cloud_cover"]],
    }, index=pd.to_datetime(h["time"]))
    w.index.name = "timestamp"
    w = w.interpolate(limit=6, limit_direction="both")
    w.to_parquet(cache)
    return w


def _to_15min(s: pd.Series) -> pd.Series:
    """Shape-preserving upsample, identical in spirit to data/prepare.py."""
    s = s.dropna()
    x = s.index.astype("int64").to_numpy(dtype=float)
    f = PchipInterpolator(x, s.to_numpy(dtype=float), extrapolate=False)
    idx = pd.date_range(s.index[0], s.index[-1], freq="15min")
    return pd.Series(f(idx.astype("int64").to_numpy(dtype=float)), index=idx).clip(lower=0.0)


def load_delhi(col: str, start: str, end: str) -> pd.Series:
    path = _download(DELHI_URL, RAW / "delhi_sldc_15min.csv")
    d = pd.read_csv(path)
    d = d[d.discom == col].copy()
    d["dt"] = pd.to_datetime(d.date, format="%m/%d/%y", errors="coerce")
    d["ts"] = d.dt + pd.to_timedelta((d.timepoint.astype(int) - 1) * 15, unit="m")
    s = (d.dropna(subset=["ts"]).drop_duplicates("ts").set_index("ts")["mW"]
         .sort_index().astype(float))
    return s.loc[start:end]


def load_china(col: str, start: str, end: str) -> pd.Series:
    """One province from the 2018 reconstruction. Semicolon-delimited, MWh.

    The time column is an hour-of-year index rather than a timestamp; the paper
    states 00:00 1 Jan to 23:00 31 Dec 2018 in UTC+8, so the index is rebuilt
    from that. Two provinces share the abbreviation ``HB`` (Hebei and Hubei) and
    pandas disambiguates the second as ``HB.1``; none of the provinces used here
    is affected, but the collision is why columns are selected by position-safe
    lookup rather than assumed unique.
    """
    path = _download(CHINA_URL, RAW / "china_provincial_2018.csv")
    d = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if col not in d.columns:
        raise SystemExit(f"province {col} not in {list(d.columns)[:8]}...")
    idx = pd.date_range("2018-01-01 00:00", periods=len(d), freq="h")
    s = pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(), index=idx).dropna()
    return s.loc[start:end]


def load_opsd(col: str, start: str, end: str) -> pd.Series:
    path = _download(OPSD_URL, RAW / "opsd_60min.csv")
    d = pd.read_csv(path, usecols=["utc_timestamp", col], parse_dates=["utc_timestamp"])
    s = d.set_index("utc_timestamp")[col].astype(float).sort_index()
    s.index = s.index.tz_localize(None)
    return s.loc[start:end].dropna()


def build(name: str, spec: dict) -> dict:
    loader = {"delhi": load_delhi, "opsd": load_opsd, "china": load_china}[spec["source"]]
    load = loader(spec["col"], spec["start"], spec["end"])
    if load.empty:
        raise SystemExit(f"{name}: no load rows in window")
    native_min = int(pd.Series(load.index).diff().dt.total_seconds().median() // 60)

    w = open_meteo(spec["lat"], spec["lon"], spec["start"], spec["end"], name)

    total = _to_15min(load)
    t_out = _to_15min(w["t_out"]).reindex(total.index).ffill().bfill()
    t_dew = _to_15min(w["t_dew"]).reindex(total.index).ffill().bfill()
    cloud = _to_15min(w["cloud"]).reindex(total.index).ffill().bfill()

    # At system level there is no submetered HVAC to split out and nothing to
    # control, so the forecast target is the whole series. base_kw carries it so
    # that the identical feature builder and identical models apply unchanged.
    out = pd.DataFrame({
        "total_kw": total, "base_kw": total,
        "hvac_kw_hist": 0.0,
        "t_out": t_out, "t_dew": t_dew, "cloud": cloud,
    }).dropna()

    out.to_parquet(CACHE / f"{name}.parquet")
    meta = {
        "id": name, "country": spec["country"], "tier": 2,
        "source": spec["source"], "note": spec["note"],
        "native_resolution_min": native_min,
        "upsampled": native_min > 15,
        "window": [str(out.index.min()), str(out.index.max())],
        "n_intervals": int(len(out)),
        "unit": "MWh" if spec["source"] == "china" else "MW",
        "reconstructed": spec["source"] == "china",
        "mean": float(out.total_kw.mean()), "peak": float(out.total_kw.max()),
        "p99_over_median": float(out.total_kw.quantile(0.99) / out.total_kw.median()),
        "corr_temp": float(np.corrcoef(out.total_kw, out.t_out)[0, 1]),
        "t_mean": float(out.t_out.mean()),
        "cdd": float(np.maximum(out.t_out.resample("D").mean() - 18.0, 0).sum()),
        "hdd": float(np.maximum(18.0 - out.t_out.resample("D").mean(), 0).sum()),
    }
    meta["cdd_share"] = meta["cdd"] / max(meta["cdd"] + meta["hdd"], 1e-9)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    metas = {}
    for name, spec in PANEL.items():
        if args.only and name not in args.only:
            continue
        print(f"{name}")
        m = build(name, spec)
        metas[name] = m
        print(f"  {m['n_intervals']:>7} steps | native {m['native_resolution_min']:>2} min | "
              f"mean {m['mean']:8.0f} {m['unit']} | peak {m['peak']:8.0f} | "
              f"p99/med {m['p99_over_median']:.2f} | corr_T {m['corr_temp']:+.2f} | "
              f"cdd_share {m['cdd_share']:.2f}")

    path = CACHE / "manifest_national.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(metas)
    path.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {len(prev)} national series -> {path}")


if __name__ == "__main__":
    main()
