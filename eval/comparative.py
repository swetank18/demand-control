"""The comparative benchmark: one model, one protocol, many climates and countries.

Every number this repo has published comes from four buildings at one site in
Tempe, Arizona. The question that invites is whether the method is general or
whether it is a fit to Phoenix, and it is not answered by asserting generality.
It is answered by running the identical pipeline somewhere else and reporting
what happens, including where it stops working.

Three arms, each isolating one factor.

  climate       demographic held fixed (Education, the one usage class that
                survives the selection rule at every site), climate and country
                varied across seven sites and three countries.
  demographic   climate held fixed (Rat, Washington DC), demographic varied
                across eight usage classes -- including Lodging/residential,
                which is what Samanvay actually targets and which the existing
                evidence base does not contain at all.
  national      real system-level demand in six countries including India,
                because BDG2 contains no Indian buildings and the honest way to
                say so is to go and get Indian data from somewhere else.

**Why skill, not pinball.** Absolute pinball loss is not comparable across
series: a 3,000 MW city and a 60 kW building are different units, and Delhi's
window (2011-12) does not overlap Europe's (2016-17). So the headline metric is
*skill against that series' own seasonal-naive baseline*,

    skill = 1 - pinball(model) / pinball(seasonal_naive)

which is a ratio computed entirely inside one series' own data and its own
period. Absolute figures are reported alongside and nothing is concluded from
them across rows.

Seasonal naive is the right denominator rather than a strawman: it is what a
site can do on day one with no model at all, and this repo's own ablation
already showed it reaching zero ceiling breaches on a calm month. Beating it is
the thing that has to be earned.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast import conformal
from forecast.baselines import REGISTRY, Baseline
from forecast.calendars import country_of_building
from forecast.features import HORIZON_STEPS, build_supervised
from forecast.metrics import score_all
from forecast.splits import SPLIT

CACHE = ROOT / "data/cache"
RESULTS = ROOT / "results"

#: Kept deliberately small. These five span "no forecast at all" to "ours", and
#: the two in between are the references that actually threaten us.
MODELS = ("static_margin", "persistence", "climatology", "seasonal_naive", "lightgbm_quantile")
SKILL_REF = "seasonal_naive"


def shifted_split(years: int):
    """SPLIT moved by whole years, preserving its shape exactly.

    Delhi's archive is 2011-2012 and BDG2's window is 2016-2017. Rather than
    invent a different protocol for India, the same 15-month train / 2-month
    validate / 1-month test structure is shifted back five years, so the test
    month is June in both cases and the seasonal position is identical.
    """
    def sh(s: str) -> str:
        t = pd.Timestamp(s)
        return str(t.replace(year=t.year - years))
    return replace(
        SPLIT,
        train_start=sh(SPLIT.train_start), train_end=sh(SPLIT.train_end),
        valid_start=sh(SPLIT.valid_start), valid_end=sh(SPLIT.valid_end),
        test_start=sh(SPLIT.test_start), test_end=sh(SPLIT.test_end),
    )


def china_split():
    """China is a single calendar year, so the 15-month training block does not
    exist. The calendar *position* is preserved instead -- validate on April and
    May, test on June, exactly as every other row -- and the training block is
    whatever precedes it, three months rather than fifteen.

    This biases against us and is the safe direction to err. Less training data
    hurts a learned model more than it hurts seasonal naive, so a China skill
    figure is a lower bound on what the model would achieve with a full history,
    not an inflated one.
    """
    return replace(
        SPLIT,
        train_start="2018-01-01 00:00", train_end="2018-03-31 23:45",
        valid_start="2018-04-01 00:00", valid_end="2018-05-31 23:45",
        test_start="2018-06-01 00:00", test_end="2018-06-30 23:45",
    )


def supervised_for(series_id: str, country: str, rebuild: bool = False) -> pd.DataFrame:
    """Cache key carries the country, because the holiday feature depends on it."""
    path = CACHE / f"sup_cmp_{series_id}_{country}_h{HORIZON_STEPS}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    df = pd.read_parquet(CACHE / f"{series_id}.parquet")
    sup = build_supervised(df, country=country)
    sup.to_parquet(path)
    return sup


def _slice(sup: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    t = sup["target_time"]
    return sup[(t >= pd.Timestamp(lo)) & (t <= pd.Timestamp(hi))]


def run_series(series_id: str, country: str, split, models=MODELS, seed: int = 0) -> dict:
    sup = supervised_for(series_id, country)
    series = pd.read_parquet(CACHE / f"{series_id}.parquet")["base_kw"].astype(float)

    tr = _slice(sup, split.train_start, split.train_end)
    va = _slice(sup, split.valid_start, split.valid_end)
    ev = _slice(sup, split.test_start, split.test_end)
    if min(len(tr), len(va), len(ev)) == 0:
        return {"id": series_id, "error": f"empty block train={len(tr)} valid={len(va)} test={len(ev)}"}

    scale = float(np.median(np.abs(series)))          # for unit-free reporting
    out = {"id": series_id, "country": country, "n_train": len(tr), "n_test": len(ev),
           "median_load": scale, "split": split.describe(), "models": {}}

    for key in models:
        cls = REGISTRY[key]
        model: Baseline = cls(seed=seed) if key in {"linear_quantile", "neural_quantile", "lightgbm_quantile"} else cls()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(tr, series, valid=va)
            except TypeError:
                model.fit(tr, series)
            p_va, p_ev = model.predict(va), model.predict(ev)
            cal_va, cal_ev, ev_ord, rec = conformal.calibrate(
                va, ev, p_va, p_ev, split=True, adaptive=True)
        sc = score_all(ev_ord["y"].to_numpy(), cal_ev)
        d = sc.as_dict()
        d["fit_seconds"] = round(time.perf_counter() - t0, 2)
        d["pinball_pct_of_median"] = 100.0 * d["pinball_mean"] / max(scale, 1e-9)
        out["models"][key] = d

    ref = out["models"].get(SKILL_REF, {}).get("pinball_mean")
    for key, d in out["models"].items():
        d["skill_vs_seasonal"] = (1.0 - d["pinball_mean"] / ref) if ref else np.nan
    return out


def panel_rows() -> list[dict]:
    """Every series to run, with its arm, country and split."""
    spec = json.loads((ROOT / "data/comparative.json").read_text())
    rows, seen = [], set()
    for arm, items in spec["arms"].items():
        for b in items:
            key = (arm, b["id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(arm=arm, id=b["id"], site=b["site"], usage=b["usage"],
                             country=country_of_building(b["id"]), tier=1, split=SPLIT))
    nat_path = CACHE / "manifest_national.json"
    if nat_path.exists():
        for name, m in json.loads(nat_path.read_text()).items():
            # Delhi's archive predates BDG2's window by five years; same protocol,
            # shifted, so the test month is June for every row in the table.
            if m["source"] == "delhi":
                sp = shifted_split(5)
            elif m["source"] == "china":
                sp = china_split()
            else:
                sp = SPLIT
            rows.append(dict(arm="national", id=name, site=name, usage="system demand",
                             country=m["country"], tier=2, split=sp))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None, help="climate demographic office national")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--out", default="comparative")
    ap.add_argument("--skip-done", action="store_true",
                    help="resume: leave series already present in the output file alone")
    args = ap.parse_args()

    rows = panel_rows()
    if args.arms:
        rows = [r for r in rows if r["arm"] in args.arms]
    if args.only:
        rows = [r for r in rows if r["id"] in args.only]

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.out}.json"
    store = json.loads(path.read_text()) if path.exists() else {}

    if args.skip_done:
        rows = [r for r in rows if f"{r['arm']}/{r['id']}" not in store]
        print(f"resuming: {len(rows)} series still to run", flush=True)

    for i, r in enumerate(rows, 1):
        tag = f"{r['arm']}/{r['id']}"
        print(f"[{i}/{len(rows)}] {tag}", flush=True)
        try:
            res = run_series(r["id"], r["country"], r["split"], models=tuple(args.models))
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            store[tag] = {"id": r["id"], "error": f"{type(e).__name__}: {e}", **{k: r[k] for k in ("arm","site","usage","country","tier")}}
            path.write_text(json.dumps(store, indent=2, default=str))
            continue
        res.update({k: r[k] for k in ("arm", "site", "usage", "country", "tier")})
        store[tag] = res
        path.write_text(json.dumps(store, indent=2, default=str))
        if "models" in res:
            ours = res["models"].get("lightgbm_quantile", {})
            sn = res["models"].get("seasonal_naive", {})
            print(f"    ours pinball {ours.get('pinball_mean', float('nan')):8.3f} "
                  f"| seasonal {sn.get('pinball_mean', float('nan')):8.3f} "
                  f"| skill {ours.get('skill_vs_seasonal', float('nan')):+.3f} "
                  f"| cov90 {ours.get('coverage_90', float('nan')):.3f}", flush=True)

    print(f"\nwrote {len(store)} series -> {path}")


if __name__ == "__main__":
    main()
