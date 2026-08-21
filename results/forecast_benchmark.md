### Forecast benchmark — Fox_assembly_Dixie

Trained 2016-01-01 to 2017-03-31, calibrated 2017-04-01 to 2017-05-31, tested once on 2017-06-01 to 2017-06-30. n = 184,320 (origin, horizon) pairs, horizons out to 16 hours.

Every row goes through the identical split-conformal plus adaptive-conformal layer, so what is compared is the forecast underneath it.

| Forecaster | Pinball | CRPS | Winkler 90 | MAE kW | Cov 90% | q95 hit | Width kW | Cal err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 9.682 | 21.096 | 123.4 | 34.19 | 0.924 | 0.970 | 116.4 | 0.0158 |
| Persistence | 13.374 | 28.778 | 193.7 | 44.34 | 0.938 | 0.963 | 180.2 | 0.0124 |
| Seasonal naive | 7.275 | 14.797 | 165.5 | 19.87 | 0.919 | 0.959 | 133.0 | 0.0155 |
| Climatology | 5.750 | 12.189 | 97.1 | 18.59 | 0.909 | 0.953 | 78.0 | 0.0117 |
| Linear quantile | 5.574 | 11.937 | **83.8** | 17.76 | 0.901 | 0.947 | 73.6 | 0.0112 |
| **LightGBM quantile (ours)** | **4.927** | **10.410** | 85.1 | **15.54** | 0.913 | 0.947 | 70.1 | **0.0057** |
| Neural quantile | 6.736 | 13.880 | 140.7 | 19.32 | 0.931 | 0.966 | 120.7 | 0.0099 |
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
