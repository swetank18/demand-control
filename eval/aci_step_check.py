"""Does the aggregation ladder survive an adaptive step stated in the
interval's units?

Every row of the comparative benchmark was calibrated with split conformal
plus ACI at gamma = 0.35 -- a number in the units of the series. On a 100 kW
building that is a working adaptive layer; on a 3,000 MW city or a 60 GW
grid it is a layer that does not move, so the system-level rows of the
calibration table were, in effect, split conformal alone while the building
rows had the adaptive layer working. The conformal audit found exactly this
on Delhi (results/conformal_audit_IN_Delhi@rel.json). Contribution 3, that
coverage degrades with aggregation, was measured across that confound.

This refits the paper's own forecaster on every row of the panel, once, and
calibrates the same predictions three ways: split conformal alone; split plus
ACI at the absolute gamma the paper used; and split plus ACI at gamma =
kappa x W, where W is the mean split-conformal 90% interval width at one hour
ahead on the *validation* block -- known before the test month, so nothing
leaks -- and kappa = 0.006 is the step the building audit ran at. Coverage of
the test month under each is what the ladder is then read from.

Resumable: one entry per row in results/aci_step_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.comparative import CACHE, _slice, panel_rows, supervised_for
from forecast import conformal
from forecast.baselines import REGISTRY
from forecast.metrics import score_all

RESULTS = ROOT / "results"
KAPPA = 0.006
GAMMA_ABS = 0.35
LEAD = 4
MODEL = "lightgbm_quantile"


def validation_width(valid: pd.DataFrame, pred_valid_split: dict, lead: int = LEAD) -> float:
    m = valid["horizon"].to_numpy() == lead
    return float(np.mean(pred_valid_split[0.95][m] - pred_valid_split[0.05][m]))


def one_row(r: dict, seed: int = 0) -> dict:
    sup = supervised_for(r["id"], r["country"])
    series = pd.read_parquet(CACHE / f"{r['id']}.parquet")["base_kw"].astype(float)
    sp = r["split"]
    tr, va, ev = (_slice(sup, a, b) for a, b in
                  ((sp.train_start, sp.train_end), (sp.valid_start, sp.valid_end),
                   (sp.test_start, sp.test_end)))
    t0 = time.perf_counter()
    model = REGISTRY[MODEL](seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(tr, series, valid=va)
        except TypeError:
            model.fit(tr, series)
        p_va, p_ev = model.predict(va), model.predict(ev)
    # split alone, to read the width the relative step is stated in
    cal_va, cal_ev_split, ev_ord, _ = conformal.calibrate(va, ev, p_va, p_ev, split=True, adaptive=False)
    W = validation_width(va, cal_va)
    out = {"id": r["id"], "arm": r["arm"], "usage": r["usage"], "country": r["country"],
           "tier": r["tier"], "tag": r["tag"], "median_load": float(np.median(np.abs(series))),
           "valid_width_lead4": W, "gamma_abs": GAMMA_ABS, "kappa": KAPPA,
           "gamma_rel": KAPPA * W, "kappa_of_abs": GAMMA_ABS / W,
           "fit_seconds": round(time.perf_counter() - t0, 1), "calib": {}}
    for k in ("test_june", "train_months", "cooling_on_meter"):
        if k in r:
            out[k] = r[k]
    y = ev_ord["y"].to_numpy()
    for name, kw in (("split", dict(adaptive=False)),
                     ("aci_abs", dict(adaptive=True, gamma=GAMMA_ABS)),
                     ("aci_rel", dict(adaptive=True, gamma=KAPPA * W))):
        _, cal_ev, ev_o, _ = conformal.calibrate(va, ev, p_va, p_ev, split=True, **kw)
        s = score_all(ev_o["y"].to_numpy(), cal_ev).as_dict()
        out["calib"][name] = {k: s[k] for k in ("coverage_90", "below_q95", "pinball_mean",
                                                 "calibration_error") if k in s}
        out["calib"][name]["mean_width"] = float(np.mean(cal_ev[0.95] - cal_ev[0.05]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()
    rows = panel_rows()
    for r in rows:
        r.setdefault("tag", f"{r['arm']}/{r['id']}")
    if args.arms:
        rows = [r for r in rows if r["arm"] in args.arms]
    if args.only:
        rows = [r for r in rows if r["id"] in args.only]
    path = RESULTS / "aci_step_check.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in rows if r["tag"] not in store]
    print(f"{len(todo)} of {len(rows)} rows to run", flush=True)
    for i, r in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {r['tag']}", flush=True)
        try:
            res = one_row(r)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            store[r["tag"]] = {"id": r["id"], "arm": r["arm"], "error": f"{type(e).__name__}: {e}"}
            path.write_text(json.dumps(store, indent=2, default=str))
            continue
        store[r["tag"]] = res
        path.write_text(json.dumps(store, indent=2, default=str))
        c = res["calib"]
        print(f"    W {res['valid_width_lead4']:9.1f}  kappa of 0.35 = {res['kappa_of_abs']:.4f} | cov90 "
              f"split {c['split']['coverage_90']:.3f}  aci@0.35 {c['aci_abs']['coverage_90']:.3f}  "
              f"aci@kappa {c['aci_rel']['coverage_90']:.3f}   ({res['fit_seconds']:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
