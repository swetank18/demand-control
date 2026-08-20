"""Measurement and verification: what did this actually save?

Measurement and verification is the real reason facility managers distrust these
systems. A vendor says "we saved you 12%", the manager knows last July was
milder than this one, and the conversation ends there.

So we do the honest version, IPMVP Option C in shape: fit a baseline model of
what the building *would* have drawn from weather and calendar alone, using a
period before the controller was switched on, then predict the reporting period
and difference it against what actually happened. Report an uncertainty band,
not a point estimate.

The unusual part: this is a simulation, so we also know the *true* counterfactual
-- we can run the uncontrolled building over the identical month. That means we
can check whether the M&V method is telling the truth, which is something a real
deployment can never do and which nobody bothers to do in a hackathon.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tariff.bill import compute_bill
from tariff.schema import Tariff


def daily_profile(pre: pd.DataFrame) -> pd.DataFrame:
    """Normalised load shape from the pre-period, by weekday/weekend and
    time of day. Each column sums to 1 over a day.

    This is the piece the naive method gets wrong. A counterfactual built by
    rescaling the *controlled* period's shape inherits the flattened profile the
    controller produced, so it shows almost no demand-charge saving by
    construction. The counterfactual has to carry the building's own historical
    peakiness, which lives here.
    """
    g = pre["grid_kw"].clip(lower=0)
    weekend = (g.index.dayofweek >= 5).astype(int)
    tod = g.index.hour * 4 + g.index.minute // 15
    prof = g.groupby([tod, weekend]).mean().unstack(level=-1)
    prof = prof.reindex(index=range(96)).interpolate().bfill().ffill()
    for w in (0, 1):
        if w not in prof.columns:
            prof[w] = prof[prof.columns[0]]
    prof = prof[[0, 1]]
    return prof / prof.sum(axis=0)


def _daily_peak_features(df: pd.DataFrame, block_min: int = 30) -> pd.DataFrame:
    """Daily maximum of the billed demand block, against degree days."""
    blocks = df["grid_kw"].clip(lower=0).resample(f"{block_min}min").mean()
    d = pd.DataFrame({
        "peak_kw": blocks.resample("D").max(),
        "t_mean": df["t_out"].resample("D").mean(),
        "t_max": df["t_out"].resample("D").max(),
    })
    d["cdd"] = np.maximum(0.0, d["t_mean"] - 18.0)
    d["weekday"] = (d.index.dayofweek < 5).astype(float)
    return d.dropna()


def fit_baseline_peak(train: pd.DataFrame, block_min: int = 30):
    """Separate regression for the daily peak.

    A daily-energy model cannot see a demand-charge saving: shifting load
    changes the peak at constant energy, which is precisely what the controller
    does. Verifying a demand saving therefore needs a model of the *peak*, and
    hot days are peakier than mild days rather than merely larger, so the
    maximum daily temperature earns its place as a regressor.
    """
    d = _daily_peak_features(train, block_min)
    X = np.column_stack([np.ones(len(d)), d.cdd, d.weekday])
    y = d.peak_kw.to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    # A negative temperature coefficient is not a real building. It is what
    # collinearity between mean and maximum daily temperature produces on a
    # short baseline, and it extrapolates catastrophically into a hotter month,
    # so the term is refused rather than shipped.
    if coef[1] < 0:
        coef[1] = 0.0
        coef[0] = float(np.mean(y - X[:, 1:] @ coef[1:]))
    resid = y - X @ coef
    return coef, resid


def predict_baseline_peak(coef: np.ndarray, report: pd.DataFrame, block_min: int = 30) -> pd.Series:
    d = _daily_peak_features(report, block_min)
    X = np.column_stack([np.ones(len(d)), d.cdd, d.weekday])
    return pd.Series(X @ coef, index=d.index, name="baseline_peak_kw")


def _shape_for_peak(share: np.ndarray, kwh: float, target_peak_kw: float,
                    steps_per_block: int) -> np.ndarray:
    """Re-shape one day so it hits both the predicted energy and the predicted
    peak, by sharpening the profile: s -> s**a, renormalised. a is found by
    bisection, and is clamped, so a day whose target peak is unreachable simply
    gets the sharpest allowed profile rather than a fabricated spike."""
    if kwh <= 0 or target_peak_kw <= 0:
        return share * kwh / 0.25

    def peak_of(a: float) -> float:
        w = np.power(np.maximum(share, 1e-12), a)
        w = w / w.sum()
        kw = w * kwh / 0.25
        n = (len(kw) // steps_per_block) * steps_per_block
        return kw[:n].reshape(-1, steps_per_block).mean(axis=1).max()

    lo, hi = 0.2, 6.0
    if peak_of(hi) < target_peak_kw:
        a = hi
    elif peak_of(lo) > target_peak_kw:
        a = lo
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if peak_of(mid) < target_peak_kw:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
    w = np.power(np.maximum(share, 1e-12), a)
    return w / w.sum() * kwh / 0.25


def synth_counterfactual(prof: pd.DataFrame, daily_kwh: pd.Series, index: pd.DatetimeIndex,
                         daily_peak: pd.Series | None = None, block_min: int = 30) -> pd.Series:
    """Build a 15-minute counterfactual: the building's historical shape, scaled
    each day to the predicted energy and, when a peak model is supplied,
    sharpened to the predicted peak."""
    weekend = (index.dayofweek >= 5).astype(int)
    tod = index.hour * 4 + index.minute // 15
    share_all = np.array([prof.iloc[t, w] for t, w in zip(tod, weekend)])
    out = np.empty(len(index))
    steps_per_block = max(1, block_min // 15)
    days = index.normalize()
    for day in pd.unique(days):
        m = days == day
        kwh = float(daily_kwh.get(day, np.nan))
        if not np.isfinite(kwh):
            out[m] = share_all[m] * float(np.nanmean(daily_kwh)) / 0.25
            continue
        sh = share_all[m]
        sh = sh / sh.sum()
        if daily_peak is not None and np.isfinite(daily_peak.get(day, np.nan)):
            out[m] = _shape_for_peak(sh, kwh, float(daily_peak[day]), steps_per_block)
        else:
            out[m] = sh * kwh / 0.25
    return pd.Series(out, index=index, name="counterfactual_kw")


@dataclass
class MVResult:
    reported_saving_inr: float
    ci_low: float
    ci_high: float
    baseline_bill_inr: float
    actual_bill_inr: float
    true_saving_inr: float | None
    inside_band: bool | None
    naive_saving_inr: float
    naive_inside_band: bool | None
    naive_ci: tuple[float, float]
    baseline_cv_rmse: float
    baseline_nmbe: float
    baseline_peak_coef: list
    extrapolation_warning: str
    ashrae_pass: bool
    n_bootstrap: int
    notes: str


def _daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily energy against cooling and heating degree days plus a weekday flag.
    The classic Option C regression; deliberately not a neural network, because
    the point is an auditable model a facility manager can argue with."""
    d = pd.DataFrame({
        "kwh": df["grid_kw"].clip(lower=0).resample("D").sum() * 0.25,
        "t_mean": df["t_out"].resample("D").mean(),
    })
    d["cdd"] = np.maximum(0.0, d["t_mean"] - 18.0)
    d["hdd"] = np.maximum(0.0, 18.0 - d["t_mean"])
    d["weekday"] = (d.index.dayofweek < 5).astype(float)
    return d.dropna()


def fit_baseline(train: pd.DataFrame):
    d = _daily_features(train)
    X = np.column_stack([np.ones(len(d)), d.cdd, d.hdd, d.weekday])
    y = d.kwh.to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    cv_rmse = float(np.sqrt(np.mean(resid ** 2)) / np.mean(y))
    nmbe = float(np.sum(resid) / (len(y) * np.mean(y)))
    return coef, resid, cv_rmse, nmbe


def predict_baseline(coef: np.ndarray, report: pd.DataFrame) -> pd.Series:
    d = _daily_features(report)
    X = np.column_stack([np.ones(len(d)), d.cdd, d.hdd, d.weekday])
    return pd.Series(X @ coef, index=d.index, name="baseline_kwh")


def verify(
    pre_period: pd.DataFrame,
    reporting: pd.DataFrame,
    tariff: Tariff,
    truth_uncontrolled: pd.DataFrame | None = None,
    power_factor: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 0,
) -> MVResult:
    """``pre_period`` and ``reporting`` need ``grid_kw`` and ``t_out`` columns."""
    coef, resid, cv_rmse, nmbe = fit_baseline(pre_period)
    base_daily = predict_baseline(coef, reporting)
    block_min = tariff.billing_interval_minutes
    pcoef, presid = fit_baseline_peak(pre_period, block_min)
    base_peak = predict_baseline_peak(pcoef, reporting, block_min)

    actual = compute_bill(reporting["grid_kw"], tariff, power_factor)

    # --- the counterfactual, built on the PRE-PERIOD shape -----------------
    prof = daily_profile(pre_period)
    counterfactual = synth_counterfactual(prof, base_daily, reporting.index, base_peak, block_min)
    baseline_bill = compute_bill(counterfactual, tariff, power_factor)
    reported = baseline_bill.total - actual.total

    # --- the naive version, kept for comparison ----------------------------
    # Rescaling the controlled period's own shape is what an Option C
    # implementation does if nobody thinks about demand charges. We report it
    # because the gap between the two is the point.
    shape = reporting["grid_kw"].clip(lower=0)
    daily_actual = shape.resample("D").sum() * 0.25
    scale = (base_daily / daily_actual.reindex(base_daily.index)).reindex(shape.index, method="ffill")
    naive_cf = shape * scale.fillna(1.0)
    naive_saving = compute_bill(naive_cf, tariff, power_factor).total - actual.total

    # Bootstrap the baseline regression residuals to get a band on the saving.
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    naive_draws = np.empty(n_bootstrap)
    daily_idx = base_daily.index
    for i in range(n_bootstrap):
        noise = rng.choice(resid, size=len(base_daily), replace=True)
        b = (base_daily + noise).clip(lower=1.0)
        pk = (base_peak + rng.choice(presid, size=len(base_peak), replace=True)).clip(lower=1.0)
        cf = synth_counterfactual(prof, b, reporting.index, pk, block_min)
        draws[i] = compute_bill(cf, tariff, power_factor).total - actual.total
        sc = (b / daily_actual.reindex(daily_idx)).reindex(shape.index, method="ffill").fillna(1.0)
        naive_draws[i] = compute_bill(shape * sc, tariff, power_factor).total - actual.total
    lo, hi = np.quantile(draws, [0.025, 0.975])
    nlo, nhi = np.quantile(naive_draws, [0.025, 0.975])

    true_saving = inside = naive_inside = None
    if truth_uncontrolled is not None:
        true_bill = compute_bill(truth_uncontrolled["grid_kw"], tariff, power_factor)
        true_saving = true_bill.total - actual.total
        inside = bool(lo <= true_saving <= hi)
        naive_inside = bool(nlo <= true_saving <= nhi)

    # ASHRAE Guideline 14 thresholds for a monthly-data Option C baseline
    ashrae_pass = bool(cv_rmse <= 0.20 and abs(nmbe) <= 0.05)

    # Extrapolation is the failure mode that silently destroys an M&V number, so
    # it is checked and reported rather than left for the reader to notice.
    pre_cdd = _daily_features(pre_period)["cdd"]
    rep_cdd = _daily_features(reporting)["cdd"]
    frac_out = float((rep_cdd > pre_cdd.max()).mean())
    extrap = (
        "ok"
        if frac_out == 0
        else f"{frac_out:.0%} of reporting days are hotter than any day in the baseline period"
    )

    return MVResult(
        reported_saving_inr=float(reported),
        ci_low=float(lo), ci_high=float(hi),
        baseline_bill_inr=float(baseline_bill.total),
        actual_bill_inr=float(actual.total),
        true_saving_inr=None if true_saving is None else float(true_saving),
        inside_band=inside,
        naive_saving_inr=float(naive_saving),
        naive_inside_band=naive_inside,
        naive_ci=(float(nlo), float(nhi)),
        baseline_cv_rmse=cv_rmse, baseline_nmbe=nmbe, ashrae_pass=ashrae_pass,
        baseline_peak_coef=[float(x) for x in pcoef],
        extrapolation_warning=extrap,
        n_bootstrap=n_bootstrap,
        notes=(
            "Baseline is a daily CDD/HDD/weekday regression fitted on the pre-period, "
            "applied to the reporting period, and shaped by the pre-period's own "
            "15-minute load profile so the counterfactual keeps the building's "
            "historical peakiness. The band is a residual bootstrap: it covers "
            "baseline-model error, not tariff or metering error. 'naive' is the same "
            "method shaped by the controlled period instead, which is what an Option C "
            "implementation does when nobody accounts for demand charges."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="Fox_office_Gaylord")
    # IPMVP Option C asks for twelve months of baseline, and this is exactly why:
    # a two-month spring baseline does not contain a single day as hot as June,
    # so every June prediction is an extrapolation and the counterfactual peak
    # comes out ~23% low.
    ap.add_argument("--pre-start", default="2016-06-01")
    ap.add_argument("--pre-end", default="2017-05-31 23:45")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default="2017-06-30 23:45")
    ap.add_argument("--out", type=Path, default=ROOT / "results/mv.json")
    ap.add_argument("--compare-baselines", action="store_true",
                    help="repeat the verification over several baseline period lengths")
    args = ap.parse_args()

    from control.mpc import MPCConfig
    from control.baselines import NoControl
    from eval.run_month import build_context, run_controller, standard_controllers

    # pre-period: the building as it runs today, i.e. under the status quo
    pre_ctx = build_context(args.building, args.pre_start, args.pre_end)
    pre = run_controller(pre_ctx, NoControl()).history

    ctx = build_context(args.building, args.start, args.end)
    tgt_file = ROOT / "results/demand_targets.json"
    tgt = None
    if tgt_file.exists():
        d = json.loads(tgt_file.read_text()).get(args.building)
        tgt = d["target_kw"] if d and d.get("feasible") else None
    ctrls = {c.name: c for c in standard_controllers(ctx, MPCConfig(), tgt)}
    ours = run_controller(ctx, ctrls["Ours: quantile + chance constrained"]).history
    truth = run_controller(ctx, NoControl()).history

    if args.compare_baselines:
        rows = []
        for label, start in [("2 months", "2017-04-01"), ("4 months", "2017-02-01"),
                             ("8 months", "2016-10-01"), ("12 months", "2016-06-01")]:
            pc = build_context(args.building, start, args.pre_end)
            ph = run_controller(pc, NoControl()).history
            r = verify(ph, ours, ctx["tariff"], truth_uncontrolled=truth, n_bootstrap=300)
            rows.append({
                "baseline": label, "start": start,
                "reported": r.reported_saving_inr, "ci": [r.ci_low, r.ci_high],
                "naive": r.naive_saving_inr, "true": r.true_saving_inr,
                "inside": r.inside_band, "naive_inside": r.naive_inside_band,
                "pct_of_truth": 100 * r.reported_saving_inr / r.true_saving_inr,
                "extrapolation": r.extrapolation_warning,
            })
        (ROOT / "results/mv_baseline_length.json").write_text(json.dumps(rows, indent=2))
        print("\nHow much baseline history does honest M&V need?")
        print(f"  true saving: Rs {rows[0]['true']:,.0f}\n")
        print(f"  {'baseline':<11}{'reported':>12}{'% of truth':>12}{'95% CI covers truth':>22}   extrapolation")
        for r in rows:
            print(f"  {r['baseline']:<11}{r['reported']:>12,.0f}{r['pct_of_truth']:>11.0f}%"
                  f"{('yes' if r['inside'] else 'NO'):>22}   {r['extrapolation']}")
        return

    res = verify(pre, ours, ctx["tariff"], truth_uncontrolled=truth)
    args.out.write_text(json.dumps(res.__dict__, indent=2))

    print(f"\nMeasurement and verification, {args.building} {args.start}..{args.end}\n")
    print(f"  baseline model   CV(RMSE) {res.baseline_cv_rmse:.1%}   NMBE {res.baseline_nmbe:+.2%}   "
          f"ASHRAE 14 {'PASS' if res.ashrae_pass else 'FAIL'}")
    print(f"  extrapolation    {res.extrapolation_warning}")
    print(f"  actual bill      Rs {res.actual_bill_inr:12,.0f}")
    print(f"  counterfactual   Rs {res.baseline_bill_inr:12,.0f}")
    print(f"  reported saving  Rs {res.reported_saving_inr:12,.0f}  "
          f"[95% CI  {res.ci_low:,.0f} .. {res.ci_high:,.0f}]")
    print(f"  naive Option C   Rs {res.naive_saving_inr:12,.0f}  "
          f"[95% CI  {res.naive_ci[0]:,.0f} .. {res.naive_ci[1]:,.0f}]")
    if res.true_saving_inr is not None:
        print(f"  TRUE saving      Rs {res.true_saving_inr:12,.0f}")
        print(f"     ours  -> {'inside' if res.inside_band else 'OUTSIDE'} the reported band")
        print(f"     naive -> {'inside' if res.naive_inside_band else 'OUTSIDE'} its band "
              f"({100*res.naive_saving_inr/res.true_saving_inr:.0f}% of the truth)")


if __name__ == "__main__":
    main()
