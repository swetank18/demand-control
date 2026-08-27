### Conformal audit — Fox_office_Gaylord

Track A acceptance. The repo already ran split conformal and adaptive conformal inference; what it did not have was evidence that either works. Three questions: does the guarantee survive a change of calibration block, does it hold out of sample across a year, and does it survive a shift the model was not retrained for.

**The short answer, because it is not the expected one.** Split conformal on its own does *not* deliver its nominal level on this data, and the failure is not marginal — month-to-month coverage of the nominal-90% interval ranges from 0.45 to 0.94 across the walk-forward year. The finite-sample theorem is not wrong; its hypothesis is. Split conformal guarantees coverage when calibration and test points are exchangeable, and a building's load in August is not exchangeable with its load in July. The adaptive layer is what actually holds the level, and this audit is the measurement that says so.

#### A1 — coverage does not depend on the calibration split

Six disjoint calibration blocks partition May 2017; June is the test month, touched once. Training ends 2017-03-31 and April is the early-stopping block, so no block that selected the model is ever used to calibrate it. A seventh row calibrates on all of May, which separates the effect of *where* the calibration window sits from *how big* it is.

| Calibration block | n | Cov 90% (per-quantile shift) | Cov 90% (CQR) | P(y ≤ q95) | Mean width kW |
| --- | --- | --- | --- | --- | --- |
| May 1-6 | 36,864 | 0.7104 | 0.7474 | 0.8720 | 25.9 |
| May 7-11 | 30,720 | 0.7569 | 0.8206 | 0.9027 | 28.3 |
| May 12-16 | 30,720 | 0.8607 | 0.8798 | 0.8907 | 35.1 |
| May 17-21 | 30,720 | 0.8306 | 0.8771 | 0.8520 | 35.1 |
| May 22-26 | 30,720 | 0.8039 | 0.8069 | 0.8930 | 30.0 |
| May 27-31 | 30,720 | 0.8257 | 0.8536 | 0.8565 | 33.0 |
| **all of May** | 190,464 | **0.8361** | 0.8343 | 0.8806 | 32.2 |

Mean coverage across the six blocks **0.7980** against a nominal 0.90, spread 0.0551, block-bootstrap standard error 0.0311 (resampling 30 whole test days, because 184,320 overlapping 15-minute forecasts are not 184,320 independent observations — the naive standard error here is 0.0007 and would fail every model ever built).

- nominal within one standard error: **FAIL**
- stable across calibration splits (spread ≤ 1 SE): **FAIL**
- P(y ≤ q95) = 0.8778 against nominal 0.95, SE 0.0302: **FAIL**

Calibrating on all of May instead of a fifth of it moves coverage to 0.8361, so calibration-set size accounts for part of the gap and the rest is the May-to-June shift. Neither is sampling noise, which is the point of reporting the block-bootstrap error bar next to them.

CQR reaches 0.8309 coverage at 1.03× the width. It is the construction with the theorem attached (Romano, Patterson and Candès) and is reported for that reason, but the controller reads one bound rather than an interval, and a symmetric width pays for a lower end nothing in the constraint ever looks at.

#### A2 — a walk-forward year

Twelve monthly folds, 2016-07-01 to 2017-06-30, each trained on everything strictly before its month and calibrated on the thirty days immediately before it. Nothing here is in-sample. ACI runs once across the concatenation, so its offsets carry over fold boundaries the way they would in deployment.

| Month | Mean load kW | Cov 90% raw | Cov 90% split | Cov 90% **ACI** | P(y≤q95) raw | split | **ACI** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2016-07 | 109 | 0.331 | 0.865 | **0.897** | 0.609 | 0.906 | **0.938** |
| 2016-08 | 135 | 0.259 | 0.674 | **0.908** | 0.268 | 0.676 | **0.959** |
| 2016-09 | 117 | 0.743 | 0.683 | **0.903** | 0.835 | 0.992 | **0.971** |
| 2016-10 | 114 | 0.730 | 0.936 | **0.893** | 0.993 | 0.986 | **0.940** |
| 2016-11 | 109 | 0.346 | 0.447 | **0.867** | 1.000 | 1.000 | **0.971** |
| 2016-12 | 123 | 0.653 | 0.877 | **0.919** | 0.927 | 0.877 | **0.919** |
| 2017-01 | 121 | 0.971 | 0.911 | **0.928** | 0.971 | 1.000 | **0.968** |
| 2017-02 | 133 | 0.774 | 0.829 | **0.890** | 0.775 | 0.840 | **0.931** |
| 2017-03 | 129 | 0.726 | 0.923 | **0.902** | 0.739 | 0.941 | **0.952** |
| 2017-04 | 124 | 0.805 | 0.898 | **0.893** | 0.832 | 0.935 | **0.951** |
| 2017-05 | 120 | 0.796 | 0.853 | **0.898** | 0.914 | 0.978 | **0.953** |
| 2017-06 | 116 | 0.722 | 0.853 | **0.888** | 0.836 | 0.892 | **0.932** |

| Layer | Rolling 30-day P(y ≤ q95) in band | min | max | mean |
| --- | --- | --- | --- | --- |
| no conformal | 17.0% | 0.2309 | 1.0000 | 0.8187 |
| split conformal | 27.4% | 0.6240 | 1.0000 | 0.9187 |
| split + ACI | 73.2% | 0.8962 | 0.9993 | 0.9499 |

Acceptance band 0.925–0.975 around the nominal 0.95, stated before the run. Coverage is read at a one-hour lead rather than pooled over horizons: pooling averages a 15-minute forecast with a 16-hour one and hides the horizon the controller actually leans on.

**What the monthly table shows, and it is worth reading carefully.** The raw LightGBM quantiles are not calibrated at all out of sample — coverage swings from 0.26 to 0.97. Split conformal narrows that considerably and still does not hold: the exchangeability its theorem needs is broken by season. The failures are directional and the direction matters. In 2016-11 the split-conformal bound sits so high that 100.0% of actuals fall under it — safe, and paying for it in unused headroom. In 2016-08 only 67.6% do, which is the expensive direction: that is a month in which the ceiling constraint was being defended against a bound reality broke through 32% of the time. ACI holds 0.919–0.971 across every month in the year, including that one.

#### A2 — frozen model, synthetic shift

The walk-forward year retrains every month, which absorbs most drift on its own and therefore cannot separate the adaptive layer from the retrain schedule. So: the model is frozen at 2016-12-31 and run to 2017-06-30 with no retraining, and on 2017-03-01 the base load takes a 15% level shift and an added volatility of 8% of the daily mean. The level shift alone would prove little — the lag features absorb it within one step. The volatility shift is the part no point forecast can absorb: the conditional mean is unchanged and the conditional spread is not, so an interval fitted before the shift is too narrow however good the median is.

| | Split conformal (frozen) | Split + ACI |
| --- | --- | --- |
| Post-shift P(y ≤ q95) | 0.8971 | 0.9425 |
| Post-shift 90% coverage | 0.7697 | 0.8849 |
| Time in band after shift | 11.4% | 89.1% |
| Days to return to band | never | 121.44791666666667 |

The trailing window is thirty days long, so nothing can return to band in under thirty days by construction; what the last row compares is the excess over that floor.

#### What this changes in the claim

Before: *our q95 carries a distribution-free finite-sample coverage guarantee.* That sentence is a citation, not a result, and on this data the plain split-conformal version of it is false — its exchangeability hypothesis does not hold across a season.

After: *the bound is held at its nominal level by an online update whose long-run exceedance rate converges regardless of whether the underlying model is any good, and here is the year of out-of-sample months showing it doing so, including one where the static version broke through 32% of the time.* That is a weaker theoretical claim and a much stronger empirical one, and it is the one that survives a hostile question.

It also changes where the credit goes. The adaptive layer was the second item in the calibration stack and easy to read as a refinement on the first. It is not a refinement. On this data it is the part that works.

Figure: `results/conformal_audit_Fox_office_Gaylord.png`.
