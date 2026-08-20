"""BDG2 -> modelling-ready 15-minute building files.

Three things happen here and all three are assumptions worth stating out loud:

1. **Hourly to 15 minutes.** BDG2 is hourly; Indian demand charges are set on
   30-minute blocks. We interpolate with a shape-preserving PCHIP so we do not
   invent overshoot. Interpolation *smooths* real sub-hourly variability, so our
   measured peaks are, if anything, understated. That biases our savings claim
   downwards, which is the safe direction.

2. **Decomposition into base load and thermal load.** BDG2 meters whole-building
   electricity. The controller needs something to control, and the forecaster
   needs an uncontrollable target. We split the meter with a per-building
   changepoint regression on outdoor temperature: the weather-driven part is
   treated as HVAC (the simulator regenerates it from physics under control), the
   remainder is uncontrollable base load (the forecaster predicts it).

3. **RC parameters are derived, not invented.** The regression slope gives
   kW per degree, which with an assumed COP gives the envelope conductance UA,
   and R = 1/UA. Thermal capacitance comes from floor area times an assumed
   heavy-construction figure. Both assumptions are named in the sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CACHE = ROOT / "cache"

# --- named assumptions -----------------------------------------------------
ASSUMED_COP = 3.0                  # chiller coefficient of performance
ASSUMED_C_KJ_PER_M2_K = 110.0      # ISO 13790 "medium" construction, kJ/(m2 K).
                                   # The conservative end: less thermal storage to
                                   # exploit means a smaller claimed saving.
BALANCE_POINT_SEARCH = (14.0, 26.0)  # C, grid searched per building
# A fitted envelope conductance outside this band (W per m2 of floor area per K)
# is not believable, and means the meter does not contain the cooling. We fall
# back to a hand-picked value and flag it in the manifest rather than shipping a
# number we do not believe.
UA_SANITY_BAND_W_PER_M2K = (0.5, 5.0)
FALLBACK_UA_W_PER_M2K = 2.0
T_DESIGN_INDOOR = 26.0   # the top of the occupied comfort band
DESIGN_MARGIN = 1.15     # plant is sized 15% above the design-day load


def _read_site_meters(site: str, ids: list[str]) -> pd.DataFrame:
    cols = pd.read_csv(RAW / "electricity_cleaned.csv", nrows=0).columns.tolist()
    missing = [b for b in ids if b not in cols]
    if missing:
        raise SystemExit(f"not in electricity_cleaned.csv: {missing}")
    df = pd.read_csv(
        RAW / "electricity_cleaned.csv",
        usecols=["timestamp"] + ids,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    return df.sort_index()


def _read_site_weather(site: str) -> pd.DataFrame:
    w = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"])
    w = w[w.site_id == site].set_index("timestamp").sort_index()
    keep = ["airTemperature", "dewTemperature", "cloudCoverage", "windSpeed"]
    w = w[keep].astype(float)
    # weather gaps are short; interpolate then hard-fill the edges
    w = w.interpolate(limit=6, limit_direction="both").ffill().bfill()
    return w


def _fit_changepoint(load_kw: pd.Series, t_out: pd.Series) -> tuple[float, float, float]:
    """Least-squares fit of load = base + slope * max(0, T - T_balance).

    Returns (base_kw, slope_kw_per_k, t_balance). Grid search on the balance
    point, closed-form least squares inside.
    """
    df = pd.concat([load_kw.rename("y"), t_out.rename("t")], axis=1).dropna()
    y, t = df.y.to_numpy(), df.t.to_numpy()
    best = (np.inf, 0.0, 0.0, 18.0)
    for tb in np.arange(BALANCE_POINT_SEARCH[0], BALANCE_POINT_SEARCH[1] + 0.01, 0.5):
        x = np.maximum(0.0, t - tb)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = float(np.sum((A @ coef - y) ** 2))
        if resid < best[0] and coef[1] > 0:
            best = (resid, float(coef[0]), float(coef[1]), float(tb))
    _, base, slope, tb = best
    return base, slope, tb


def _to_15min(s: pd.Series) -> pd.Series:
    """Shape-preserving hourly -> 15-minute interpolation (no overshoot)."""
    s = s.dropna()
    x = s.index.astype("int64").to_numpy(dtype=float)
    f = PchipInterpolator(x, s.to_numpy(dtype=float), extrapolate=False)
    new_idx = pd.date_range(s.index[0], s.index[-1], freq="15min")
    out = pd.Series(f(new_idx.astype("int64").to_numpy(dtype=float)), index=new_idx)
    return out.clip(lower=0.0)


def prepare(config_path: Path, years: tuple[int, ...] = (2016, 2017)) -> None:
    cfg = json.loads(config_path.read_text())
    site = cfg["site"]
    ids = [b["id"] for b in cfg["buildings"]]
    CACHE.mkdir(parents=True, exist_ok=True)

    meters = _read_site_meters(site, ids)
    weather = _read_site_weather(site)
    meta = pd.read_csv(RAW / "metadata.csv", low_memory=False).set_index("building_id")

    manifest = {"site": site, "years": list(years), "assumptions": {
        "cop": ASSUMED_COP,
        "capacitance_kJ_per_m2_K": ASSUMED_C_KJ_PER_M2_K,
        "upsampling": "PCHIP hourly->15min, shape preserving, clipped at zero",
        "decomposition": "changepoint regression on outdoor air temperature",
    }, "buildings": {}}

    for b in cfg["buildings"]:
        bid = b["id"]
        raw = meters[bid]
        raw = raw[raw.index.year.isin(years)]
        if raw.notna().mean() < 0.9:
            print(f"  ! {bid}: coverage {raw.notna().mean():.1%} over {years}, skipping")
            continue
        # short gaps only; a long gap should be visible, not papered over
        raw = raw.interpolate(limit=3, limit_direction="both")

        t_hr = weather["airTemperature"].reindex(raw.index)
        base_kw, slope, t_balance = _fit_changepoint(raw, t_hr)

        # hourly decomposition, then upsample each part
        cooling_hr = (slope * np.maximum(0.0, t_hr - t_balance)).clip(lower=0.0)
        cooling_hr = np.minimum(cooling_hr, raw * 0.85)     # HVAC never exceeds 85% of the meter
        base_hr = (raw - cooling_hr).clip(lower=0.0)

        total = _to_15min(raw)
        base = _to_15min(base_hr).reindex(total.index).ffill()
        hvac_hist = (total - base).clip(lower=0.0)
        t_out = _to_15min(t_hr).reindex(total.index).ffill()
        dew = _to_15min(weather["dewTemperature"].reindex(raw.index)).reindex(total.index).ffill()
        cloud = _to_15min(weather["cloudCoverage"].reindex(raw.index).fillna(0)).reindex(total.index).ffill()

        out = pd.DataFrame(
            {"total_kw": total, "base_kw": base, "hvac_kw_hist": hvac_hist,
             "t_out": t_out, "t_dew": dew, "cloud": cloud}
        ).dropna()

        sqm = float(meta.loc[bid, "sqm"]) if bid in meta.index and not pd.isna(meta.loc[bid, "sqm"]) else np.nan
        if not np.isfinite(sqm) or sqm <= 0:
            sqm = 5000.0
        # electrical kW/K * COP = thermal kW/K = effective envelope conductance
        ua_kw_per_k = slope * ASSUMED_COP
        ua_w_per_m2k = ua_kw_per_k * 1000.0 / sqm
        ua_source = "fitted from the building's own meter (changepoint slope x COP)"
        if not (UA_SANITY_BAND_W_PER_M2K[0] <= ua_w_per_m2k <= UA_SANITY_BAND_W_PER_M2K[1]):
            ua_kw_per_k = FALLBACK_UA_W_PER_M2K * sqm / 1000.0
            ua_w_per_m2k = FALLBACK_UA_W_PER_M2K
            ua_source = (
                f"HAND-PICKED {FALLBACK_UA_W_PER_M2K} W/m2K: the fitted value "
                f"({slope * ASSUMED_COP * 1000.0 / sqm:.2f} W/m2K) fell outside the believable band"
            )
        r_k_per_kw = 1.0 / max(ua_kw_per_k, 1e-6)
        c_kwh_per_k = ASSUMED_C_KJ_PER_M2_K * sqm / 3600.0   # kJ/K -> kWh/K
        hist_peak_kva = float(out.total_kw.resample("30min").mean().max() / 0.95)

        # Size the plant the way an engineer would, not from the fitted residual:
        # the changepoint only sees cooling load *above* the balance point, so
        # sizing off it leaves the chiller unable to hold the band on a design day.
        t_design = float(out.t_out.quantile(0.995))
        q_int_design = (
            0.85 * float(out.base_kw.quantile(0.95))      # lighting and plug load as heat
            + 0.05 * sqm * 0.100                          # people, 5 W/m2 sensible
        )
        hvac_capacity_kw = float(
            np.ceil(
                DESIGN_MARGIN
                * (ua_kw_per_k * max(t_design - T_DESIGN_INDOOR, 0.0) + q_int_design)
                / ASSUMED_COP
                / 5.0
            )
            * 5.0
        )

        params = {
            "id": bid, "label": b.get("label", bid), "role": b.get("role", "campus"),
            "why_chosen": b.get("why", ""),
            "sqm": sqm,
            "changepoint": {"base_kw": base_kw, "slope_kw_per_k": slope, "t_balance_c": t_balance},
            "thermal": {
                "ua_kw_per_k": ua_kw_per_k, "ua_w_per_m2k": ua_w_per_m2k,
                "ua_source": ua_source,
                "r_k_per_kw": r_k_per_kw,
                "c_kwh_per_k": c_kwh_per_k, "cop": ASSUMED_COP,
                "time_constant_h": r_k_per_kw * c_kwh_per_k,
            },
            "hvac_capacity_kw": hvac_capacity_kw,
            "hvac_sizing": (
                f"engineering sizing at design outdoor {t_design:.1f} C holding {T_DESIGN_INDOOR:.0f} C: "
                f"envelope {ua_kw_per_k * (t_design - T_DESIGN_INDOOR):.0f} kW + internal {q_int_design:.0f} kW "
                f"thermal, / COP {ASSUMED_COP}, x {DESIGN_MARGIN} margin"
            ),
            "hvac_capacity_from_meter_kw": float(out.hvac_kw_hist.quantile(0.995)),
            "hist_peak_kva": hist_peak_kva,
            "contract_demand_kva": float(np.ceil(hist_peak_kva * 1.05 / 5) * 5),
            "hvac_share_of_meter": float(out.hvac_kw_hist.sum() / out.total_kw.sum()),
            "coverage": float(raw.notna().mean()),
            "n_intervals": int(len(out)),
        }
        out.to_parquet(CACHE / f"{bid}.parquet")
        manifest["buildings"][bid] = params
        print(
            f"  {bid:<24} {len(out):>6} steps | base {out.base_kw.mean():6.1f} kW | "
            f"hvac {params['hvac_share_of_meter']:5.1%} | UA {ua_w_per_m2k:4.2f} W/m2K | "
            f"tau {params['thermal']['time_constant_h']:5.1f} h | hvac cap {hvac_capacity_kw:4.0f} kW | "
            f"contract {params['contract_demand_kva']:.0f} kVA"
        )

    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest['buildings'])} buildings -> {CACHE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "buildings.json")
    ap.add_argument("--years", type=int, nargs="+", default=[2016, 2017])
    args = ap.parse_args()
    prepare(args.config, tuple(args.years))
