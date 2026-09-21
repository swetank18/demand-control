### Conformal audit — IN_Delhi

Track A acceptance. The repo already ran split conformal and adaptive conformal inference; what it did not have was evidence that either works. Three questions: does the guarantee survive a change of calibration block, does it hold out of sample across a year, and does it survive a shift the model was not retrained for.

**The short answer, because it is not the expected one.** Split conformal on its own does *not* deliver its nominal level on this data, and the failure is not marginal — month-to-month coverage of the nominal-90% interval ranges from 0.63 to 0.98 across the walk-forward year. The finite-sample theorem is not wrong; its hypothesis is. Split conformal guarantees coverage when calibration and test points are exchangeable, and a building's load in August is not exchangeable with its load in July. The adaptive layer is what actually holds the level, and this audit is the measurement that says so.

#### A1 — coverage does not depend on the calibration split

Six disjoint calibration blocks partition May 2012; June is the test month, touched once. Training ends 2012-03-31 and April is the early-stopping block, so no block that selected the model is ever used to calibrate it. A seventh row calibrates on all of May, which separates the effect of *where* the calibration window sits from *how big* it is.

| Calibration block | n | Cov 90% (per-quantile shift) | Cov 90% (CQR) | P(y ≤ q95) | Mean width kW |
| --- | --- | --- | --- | --- | --- |
| May 1-6 | 36,864 | 0.3437 | 0.3733 | 0.3437 | 948.1 |
| May 7-11 | 30,720 | 0.5441 | 0.4646 | 0.5443 | 921.0 |
| May 12-16 | 30,720 | 0.3794 | 0.5033 | 0.3794 | 1075.6 |
| May 17-21 | 30,720 | 0.4951 | 0.4112 | 0.4960 | 941.2 |
| May 22-26 | 30,720 | 0.6550 | 0.5793 | 0.6595 | 926.9 |
| May 27-31 | 30,720 | 0.7282 | 0.6880 | 0.7404 | 926.1 |
| **all of May** | 190,464 | **0.6117** | 0.5357 | 0.6118 | 1045.3 |

Mean coverage across the six blocks **0.5242** against a nominal 0.90, spread 0.1507, block-bootstrap standard error 0.0437 (resampling 30 whole test days, because 184,320 overlapping 15-minute forecasts are not 184,320 independent observations — the naive standard error here is 0.0007 and would fail every model ever built).

- nominal within one standard error: **FAIL**
- stable across calibration splits (spread ≤ 1 SE): **FAIL**
- P(y ≤ q95) = 0.5272 against nominal 0.95, SE 0.0437: **FAIL**

Calibrating on all of May instead of a fifth of it moves coverage to 0.6117, so calibration-set size accounts for part of the gap and the rest is the May-to-June shift. Neither is sampling noise, which is the point of reporting the block-bootstrap error bar next to them.

CQR reaches 0.5033 coverage at 1.14× the width. It is the construction with the theorem attached (Romano, Patterson and Candès) and is reported for that reason, but the controller reads one bound rather than an interval, and a symmetric width pays for a lower end nothing in the constraint ever looks at.

#### A2 — a walk-forward year

Twelve monthly folds, 2011-07-01 to 2012-06-30, each trained on everything strictly before its month and calibrated on the thirty days immediately before it. Nothing here is in-sample. ACI runs once across the concatenation, so its offsets carry over fold boundaries the way they would in deployment.

| Month | Mean load kW | Cov 90% raw | Cov 90% split | Cov 90% **ACI** | P(y≤q95) raw | split | **ACI** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2011-07 | 3752 | 0.774 | 0.977 | **0.918** | 0.774 | 0.985 | **0.954** |
| 2011-08 | 3626 | 0.767 | 0.846 | **0.874** | 0.862 | 0.956 | **0.946** |
| 2011-09 | 3495 | 0.754 | 0.948 | **0.935** | 0.953 | 0.976 | **0.969** |
| 2011-10 | 2768 | 0.773 | 0.836 | **0.874** | 0.982 | 0.986 | **0.957** |
| 2011-11 | 2290 | 0.818 | 0.976 | **0.923** | 0.980 | 0.976 | **0.930** |
| 2011-12 | 2376 | 0.878 | 0.924 | **0.882** | 0.979 | 0.926 | **0.943** |
| 2012-01 | 2526 | 0.610 | 0.628 | **0.893** | 0.692 | 0.664 | **0.937** |
| 2012-02 | 2386 | 0.795 | 0.980 | **0.922** | 0.861 | 0.997 | **0.982** |
| 2012-03 | 2279 | 0.666 | 0.755 | **0.880** | 0.883 | 0.937 | **0.932** |
| 2012-04 | 2867 | 0.572 | 0.834 | **0.910** | 0.730 | 0.885 | **0.962** |
| 2012-05 | 3632 | 0.766 | 0.932 | **0.907** | 0.818 | 0.942 | **0.932** |
| 2012-06 | 4220 | 0.419 | 0.633 | **0.895** | 0.424 | 0.641 | **0.931** |

| Layer | Rolling 30-day P(y ≤ q95) in band | min | max | mean |
| --- | --- | --- | --- | --- |
| no conformal | 8.0% | 0.4257 | 0.9917 | 0.8581 |
| split conformal | 41.7% | 0.6122 | 0.9955 | 0.9189 |
| split + ACI | 94.2% | 0.9174 | 0.9903 | 0.9480 |

Acceptance band 0.925–0.975 around the nominal 0.95, stated before the run. Coverage is read at a one-hour lead rather than pooled over horizons: pooling averages a 15-minute forecast with a 16-hour one and hides the horizon the controller actually leans on.

**What the monthly table shows, and it is worth reading carefully.** The raw LightGBM quantiles are not calibrated at all out of sample — coverage swings from 0.42 to 0.88. Split conformal narrows that considerably and still does not hold: the exchangeability its theorem needs is broken by season. The failures are directional and the direction matters. In 2012-02 the split-conformal bound sits so high that 99.7% of actuals fall under it — safe, and paying for it in unused headroom. In 2012-06 only 64.1% do, which is the expensive direction: that is a month in which the ceiling constraint was being defended against a bound reality broke through 36% of the time. ACI holds 0.930–0.982 across every month in the year, including that one.

#### A2 — frozen model, synthetic shift

The walk-forward year retrains every month, which absorbs most drift on its own and therefore cannot separate the adaptive layer from the retrain schedule. So: the model is frozen at 2011-12-31 and run to 2012-06-30 with no retraining, and on 2012-03-01 00:00:00 the base load takes a 15% level shift and an added volatility of 8% of the daily mean. The level shift alone would prove little — the lag features absorb it within one step. The volatility shift is the part no point forecast can absorb: the conditional mean is unchanged and the conditional spread is not, so an interval fitted before the shift is too narrow however good the median is.

| | Split conformal (frozen) | Split + ACI |
| --- | --- | --- |
| Post-shift P(y ≤ q95) | 0.4266 | 0.9280 |
| Post-shift 90% coverage | 0.3579 | 0.8873 |
| Time in band after shift | 0.0% | 59.6% |
| Days to return to band | never | never |

The trailing window is thirty days long, so nothing can return to band in under thirty days by construction; what the last row compares is the excess over that floor.

#### What this changes in the claim

Before: *our q95 carries a distribution-free finite-sample coverage guarantee.* That sentence is a citation, not a result, and on this data the plain split-conformal version of it is false — its exchangeability hypothesis does not hold across a season.

After: *the bound is held at its nominal level by an online update whose long-run exceedance rate converges regardless of whether the underlying model is any good, and here is the year of out-of-sample months showing it doing so, including one where the static version broke through 36% of the time.* That is a weaker theoretical claim and a much stronger empirical one, and it is the one that survives a hostile question.

It also changes where the credit goes. The adaptive layer was the second item in the calibration stack and easy to read as a refinement on the first. It is not a refinement. On this data it is the part that works.

Figure: `results/conformal_audit_IN_Delhi.png`.
