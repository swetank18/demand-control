"""The forecast benchmark: every model, every baseline, one script, one table.

This is Section 6.3 and 6.2 of the plan. It answers "there is a trained model
and it is good" with held-out numbers against six references rather than with an
adjective, and it answers "how do you know you are not overfitting" with a
temporal split plus rolling-origin folds whose *spread* is reported, not just
their mean.

Two evaluations, deliberately different in what they are for.

*Headline.* Train on the training block, calibrate on validation, evaluate once
on the June test block. That is the number the deck quotes.

*Rolling origin.* Eight expanding-window folds ending each month from October to
May, each one trained and calibrated only on what preceded it. A single split on
seasonal data is fragile: a model that is excellent in February and poor in May
is not one you want holding a demand ceiling, and only the folds reveal that.

Everything downstream reads the forecast tensors this writes, so the ablation in
``eval/ablation.py`` is running the identical forecasts that were scored here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast import conformal
from forecast.baselines import QUANTILES, REGISTRY, Baseline
from forecast.features import HORIZON_STEPS, build_supervised
from forecast.metrics import score_all
from forecast.splits import ROLLING_FOLDS, SPLIT

CACHE = ROOT / "data/cache"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------

def supervised(building: str, cache: Path = CACHE, rebuild: bool = False) -> pd.DataFrame:
    """Build (or reuse) the long (origin, horizon) frame for one building.

    Cached because it takes about a minute and eight studies want the identical
    rows. The cache key includes the horizon so a changed horizon cannot silently
    read a stale frame.
    """
    path = cache / f"supervised_{building}_h{HORIZON_STEPS}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    df = pd.read_parquet(cache / f"{building}.parquet")
    sup = build_supervised(df)
    sup.to_parquet(path)
    return sup


def series_of(building: str, cache: Path = CACHE) -> pd.Series:
    return pd.read_parquet(cache / f"{building}.parquet")["base_kw"].astype(float)


def _slice(sup: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    t = sup["target_time"]
    return sup[(t >= pd.Timestamp(lo)) & (t <= pd.Timestamp(hi))]


def fit_predict(
    key: str, sup: pd.DataFrame, series: pd.Series,
    train_lo: str, train_hi: str, valid_lo: str, valid_hi: str, eval_lo: str, eval_hi: str,
    adaptive: bool = True, seed: int = 0,
) -> dict:
    """Fit one baseline on one fold and return its calibrated evaluation tensor.

    The evaluation block is never seen by ``fit`` and never seen by the
    conformal fit; the adaptive layer sees each actual only after that target
    time has passed, which is what it would see deployed.
    """
    cls = REGISTRY[key]
    model: Baseline = cls(seed=seed) if key in {"linear_quantile", "neural_quantile", "lightgbm_quantile"} else cls()

    tr = _slice(sup, train_lo, train_hi)
    va = _slice(sup, valid_lo, valid_hi)
    ev = _slice(sup, eval_lo, eval_hi)
    if min(len(tr), len(va), len(ev)) == 0:
        raise SystemExit(f"empty block for {key}: train={len(tr)} valid={len(va)} eval={len(ev)}")

    t0 = time.perf_counter()
    try:
        model.fit(tr, series, valid=va)          # trainers that use early stopping
    except TypeError:
        model.fit(tr, series)
    fit_s = time.perf_counter() - t0

    p_va = model.predict(va)
    p_ev = model.predict(ev)

    # Perfect foresight is a bound, not a forecast; calibrating it would shift
    # the truth off itself and make the ceiling row of the table meaningless.
    if key == "perfect_foresight":
        cal_va, cal_ev, ev_ord, record = p_va, p_ev, ev, {"calibrated": False}
    else:
        cal_va, cal_ev, ev_ord, record = conformal.calibrate(
            va, ev, p_va, p_ev, split=True, adaptive=adaptive)

    scores = score_all(ev_ord["y"].to_numpy(), cal_ev)
    raw = score_all(ev["y"].to_numpy(), p_ev)

    tensor = ev_ord[["origin", "target_time", "horizon"]].copy()
    for q in sorted(cal_ev):
        tensor[f"q{int(q * 100):02d}"] = cal_ev[q]
    tensor["actual"] = ev_ord["y"].to_numpy()

    return {
        "key": key,
        "name": model.name,
        "definition": model.definition,
        "fit_seconds": fit_s,
        "scores": scores.as_dict(),
        "scores_uncalibrated": raw.as_dict(),
        "calibration": {k: v for k, v in record.items() if k not in ("shifts", "aci")},
        "tensor": tensor,
        "model": model,
    }


def by_horizon(tensor: pd.DataFrame) -> dict:
    """Coverage and error as the horizon grows, because the controller leans
    hardest on the far end and an average hides that."""
    g = tensor.groupby("horizon")
    return {
        "coverage_90": {int(h): float(v) for h, v in
                        (((g.apply(lambda d: ((d.actual >= d.q05) & (d.actual <= d.q95)).mean(),
                                   include_groups=False)))).items()},
        "below_q95": {int(h): float(v) for h, v in
                      g.apply(lambda d: (d.actual <= d.q95).mean(), include_groups=False).items()},
        "mae": {int(h): float(v) for h, v in
                g.apply(lambda d: (d.actual - d.q50).abs().mean(), include_groups=False).items()},
        "width": {int(h): float(v) for h, v in
                  g.apply(lambda d: (d.q95 - d.q05).mean(), include_groups=False).items()},
    }


# ---------------------------------------------------------------------------

def headline(building: str, keys: list[str], adaptive: bool = True,
             save_tensors: bool = True) -> dict:
    sup = supervised(building)
    series = series_of(building)
    print(f"\n== {building}  |  {SPLIT.describe()}")
    print(f"   rows: train/valid/test = "
          f"{len(_slice(sup, SPLIT.train_start, SPLIT.train_end)):,} / "
          f"{len(_slice(sup, SPLIT.valid_start, SPLIT.valid_end)):,} / "
          f"{len(_slice(sup, SPLIT.test_start, SPLIT.test_end)):,}")

    out = {"building": building, "split": SPLIT.as_dict(), "models": {}}
    tdir = MODELS / building / "tensors"
    if save_tensors:
        tdir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        r = fit_predict(key, sup, series,
                        SPLIT.train_start, SPLIT.train_end,
                        SPLIT.valid_start, SPLIT.valid_end,
                        SPLIT.test_start, SPLIT.test_end, adaptive=adaptive)
        s = r["scores"]
        print(f"   {r['name']:<28} pinball {s['pinball_mean']:7.3f}  "
              f"cov90 {s['coverage_90']:.3f}  q95-hit {s['below_q95']:.3f}  "
              f"width {s['sharpness_90']:6.1f} kW  MAE {s['mae_median']:6.2f}  "
              f"({r['fit_seconds']:.0f}s)")
        if save_tensors:
            r["tensor"].to_parquet(tdir / f"{key}.parquet")
        entry = {k: v for k, v in r.items() if k not in ("tensor", "model")}
        entry["by_horizon"] = by_horizon(r["tensor"])
        if key == "neural_quantile":
            entry["training_curve"] = r["model"].history
        if key == "lightgbm_quantile":
            entry["best_iteration"] = r["model"].best_iteration
        out["models"][key] = entry
    return out


def rolling(building: str, keys: list[str], folds=ROLLING_FOLDS, adaptive: bool = True) -> dict:
    """Expanding-window walk-forward. Each fold validates on the month after the
    one it stopped training on, and the calibration block is the month before
    that -- so no fold ever calibrates on what it is scored on."""
    sup = supervised(building)
    series = series_of(building)
    out = {"building": building, "folds": [], "summary": {}}
    print(f"\n== rolling origin | {building} | {len(folds)} folds")

    for train_hi, valid_hi in folds:
        # the month being scored is (train_hi, valid_hi]; the calibration block
        # is the month before train_hi, held out of that fold's training
        cal_lo = (pd.Timestamp(train_hi) - pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d %H:%M")
        fit_hi = (pd.Timestamp(cal_lo) - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")
        row = {"train_end": fit_hi, "calibrate": [cal_lo, train_hi], "score": [train_hi, valid_hi],
               "models": {}}
        for key in keys:
            r = fit_predict(key, sup, series, SPLIT.train_start, fit_hi,
                            cal_lo, train_hi, train_hi, valid_hi, adaptive=adaptive)
            row["models"][key] = r["scores"]
        best = min(row["models"], key=lambda k: row["models"][k]["pinball_mean"]
                   if k != "perfect_foresight" else 1e18)
        print(f"   score {row['score'][0][:7]} -> {row['score'][1][:7]}   best: {best}")
        out["folds"].append(row)

    for key in keys:
        p = np.array([f["models"][key]["pinball_mean"] for f in out["folds"]])
        c = np.array([f["models"][key]["coverage_90"] for f in out["folds"]])
        out["summary"][key] = {
            "pinball_mean": float(p.mean()), "pinball_std": float(p.std()),
            "pinball_min": float(p.min()), "pinball_max": float(p.max()),
            "coverage_mean": float(c.mean()), "coverage_std": float(c.std()),
            "coverage_worst": float(c.min()),
            "folds_won": int(sum(
                1 for f in out["folds"]
                if min((k for k in f["models"] if k != "perfect_foresight"),
                       key=lambda k: f["models"][k]["pinball_mean"]) == key)),
        }
    return out


# ---------------------------------------------------------------------------

BENCH_COLS = [
    ("name", "Forecaster", "{}", None),
    ("pinball_mean", "Pinball", "{:.3f}", "min"),
    ("crps", "CRPS", "{:.3f}", "min"),
    ("winkler_90", "Winkler 90", "{:.1f}", "min"),
    ("mae_median", "MAE kW", "{:.2f}", "min"),
    ("coverage_90", "Cov 90%", "{:.3f}", None),
    ("below_q95", "q95 hit", "{:.3f}", None),
    ("sharpness_90", "Width kW", "{:.1f}", None),
    ("calibration_error", "Cal err", "{:.4f}", "min"),
]


def to_markdown(res: dict, roll: dict | None) -> str:
    rows = [{"name": m["name"], "definition": m["definition"], **m["scores"]}
            for m in res["models"].values() if m["key"] != "perfect_foresight"]
    ceiling = [m for m in res["models"].values() if m["key"] == "perfect_foresight"]

    best = {}
    for key, _, _, opt in BENCH_COLS:
        if opt == "min" and rows:
            best[key] = min(r[key] for r in rows)

    s = res["split"]
    lines = [
        f"### Forecast benchmark — {res['building']}",
        "",
        f"Trained {s['train_start'][:10]} to {s['train_end'][:10]}, "
        f"calibrated {s['valid_start'][:10]} to {s['valid_end'][:10]}, "
        f"tested once on {s['test_start'][:10]} to {s['test_end'][:10]}. "
        f"n = {rows[0]['n']:,} (origin, horizon) pairs, horizons out to "
        f"{HORIZON_STEPS // 4} hours.",
        "",
        "Every row goes through the identical split-conformal plus adaptive-conformal "
        "layer, so what is compared is the forecast underneath it.",
        "",
        "| " + " | ".join(c[1] for c in BENCH_COLS) + " |",
        "| " + " | ".join("---" for _ in BENCH_COLS) + " |",
    ]
    # The ceiling row is a degenerate distribution: every quantile equals the
    # actual. Coverage is then trivially 1 and "calibration error" reads 0.5 by
    # arithmetic rather than by any defect, so those cells are blanked instead of
    # printed as if they meant something.
    ceiling_rows = [{"name": m["name"], **m["scores"], "_degenerate": True} for m in ceiling]
    DEGENERATE_BLANK = {"coverage_90", "below_q95", "calibration_error"}

    for r in rows + ceiling_rows:
        cells = []
        for key, _, fmt, opt in BENCH_COLS:
            v = r.get(key)
            if r.get("_degenerate") and key in DEGENERATE_BLANK:
                cells.append("n/a")
                continue
            txt = "-" if v is None else fmt.format(v)
            if opt == "min" and key in best and isinstance(v, float) and abs(v - best[key]) < 1e-12:
                txt = f"**{txt}**"
            cells.append(txt)
        if "ours" in str(r["name"]).lower():
            cells[0] = f"**{cells[0]}**"
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["",
              "Perfect foresight is a bound, not a forecaster: it is not put through the "
              "calibration layer, and its coverage and calibration cells are marked n/a "
              "because a distribution with zero width makes them arithmetic rather than "
              "evidence.",
              "", "Definitions:", ""]
    for m in res["models"].values():
        lines.append(f"- **{m['name']}** — {m['definition']}")

    if roll:
        lines += [
            "", f"#### Rolling origin, {len(roll['folds'])} expanding-window folds", "",
            "Mean and spread across folds. The spread is the point: a forecaster that "
            "is excellent in February and poor in May cannot be trusted with a ceiling.",
            "",
            "| Forecaster | Pinball mean | Pinball spread | Worst fold | Coverage mean | Worst coverage | Folds won |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key, sm in roll["summary"].items():
            if key == "perfect_foresight":
                continue
            nm = res["models"][key]["name"] if key in res["models"] else key
            lines.append(
                f"| {nm} | {sm['pinball_mean']:.3f} | ±{sm['pinball_std']:.3f} | "
                f"{sm['pinball_max']:.3f} | {sm['coverage_mean']:.3f} | "
                f"{sm['coverage_worst']:.3f} | {sm['folds_won']}/{len(roll['folds'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="+", default=["Fox_office_Gaylord"])
    ap.add_argument("--baselines", nargs="+", default=list(REGISTRY))
    ap.add_argument("--rolling", action="store_true", help="also run walk-forward folds")
    ap.add_argument("--rolling-baselines", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=len(ROLLING_FOLDS))
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    md = []
    for b in args.buildings:
        res = headline(b, args.baselines, adaptive=not args.no_adaptive)
        roll = None
        if args.rolling:
            keys = args.rolling_baselines or [k for k in args.baselines if k != "perfect_foresight"]
            roll = rolling(b, keys, folds=ROLLING_FOLDS[: args.folds],
                           adaptive=not args.no_adaptive)
            res["rolling"] = roll
        (args.out / f"forecast_benchmark_{b}.json").write_text(json.dumps(res, indent=2, default=float))
        md.append(to_markdown(res, roll))
    (args.out / "forecast_benchmark.md").write_text("\n\n".join(md) + "\n")
    print("\n" + "\n\n".join(md))


if __name__ == "__main__":
    main()
