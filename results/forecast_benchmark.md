### Forecast benchmark — Fox_public_Denny

Trained 2016-01-01 to 2017-03-31, calibrated 2017-04-01 to 2017-05-31, tested once on 2017-06-01 to 2017-06-30. n = 184,320 (origin, horizon) pairs, horizons out to 16 hours.

Every row goes through the identical split-conformal plus adaptive-conformal layer, so what is compared is the forecast underneath it.

| Forecaster | Pinball | CRPS | Winkler 90 | MAE kW | Cov 90% | q95 hit | Width kW | Cal err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 4.337 | 9.285 | 66.8 | 14.53 | 0.887 | 0.932 | 53.8 | 0.0097 |
| Persistence | 6.146 | 13.169 | 94.3 | 20.70 | 0.875 | 0.939 | 74.3 | 0.0094 |
| Seasonal naive | 3.037 | 6.371 | 55.7 | 9.40 | 0.878 | 0.939 | 40.2 | 0.0105 |
| Climatology | 3.230 | 6.928 | 48.6 | 10.70 | 0.889 | 0.931 | 38.9 | 0.0100 |
| Linear quantile | 3.448 | 7.360 | 54.8 | 11.47 | 0.895 | 0.942 | 44.3 | **0.0033** |
| **LightGBM quantile (ours)** | **2.576** | **5.430** | **45.5** | **8.11** | 0.897 | 0.939 | 36.8 | 0.0049 |
| Neural quantile | 3.392 | 7.161 | 59.1 | 10.73 | 0.887 | 0.931 | 44.1 | 0.0130 |
| Perfect foresight | 0.000 | 0.000 | 0.0 | 0.00 | n/a | n/a | 0.0 | n/a |

Perfect foresight is a bound, not a forecaster: it is not put through the calibration layer, and its coverage and calibration cells are marked n/a because a distribution with zero width makes them arithmetic rather than evidence.

Definitions:

- **Static margin (no forecast)** — a single constant, the training-set p95, for every interval
- **Persistence** — the forecast is the value observed at the forecast origin
- **Seasonal naive** — the value at the same clock time exactly one week before the target
- **Climatology** — training-set quantiles for that weekday and time of day
- **Linear quantile** — pinball-loss linear regression on the identical feature matrix
- **LightGBM quantile (ours)** — gradient-boosted trees, one booster per quantile
- **Neural quantile** — residual MLP stack, shared trunk, pinball head on all five levels
- **Perfect foresight** — the actual value (upper bound on achievable skill)
