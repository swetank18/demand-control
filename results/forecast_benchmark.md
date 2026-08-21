### Forecast benchmark — Fox_office_Gaylord

Trained 2016-01-01 to 2017-03-31, calibrated 2017-04-01 to 2017-05-31, tested once on 2017-06-01 to 2017-06-30. n = 184,320 (origin, horizon) pairs, horizons out to 16 hours.

Every row goes through the identical split-conformal plus adaptive-conformal layer, so what is compared is the forecast underneath it.

| Forecaster | Pinball | CRPS | Winkler 90 | MAE kW | Cov 90% | q95 hit | Width kW | Cal err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 20.307 | 44.341 | 252.5 | 72.28 | 0.852 | 0.924 | 223.4 | 0.0256 |
| Persistence | 27.582 | 59.242 | 404.3 | 89.47 | 0.820 | 0.914 | 314.7 | 0.0259 |
| Seasonal naive | 6.213 | 12.796 | 131.4 | 18.37 | 0.884 | 0.938 | 72.2 | 0.0109 |
| Climatology | 7.314 | 15.849 | 102.1 | 26.75 | 0.863 | 0.924 | 86.5 | 0.0184 |
| Linear quantile | 10.758 | 23.205 | 154.4 | 37.19 | 0.880 | 0.932 | 132.5 | 0.0082 |
| **LightGBM quantile (ours)** | **2.397** | **4.922** | 51.7 | **7.03** | 0.891 | 0.939 | 39.6 | 0.0068 |
| Neural quantile | 2.435 | 5.107 | **45.0** | 7.59 | 0.894 | 0.952 | 33.5 | **0.0044** |
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

#### Rolling origin, 8 expanding-window folds

Mean and spread across folds. The spread is the point: a forecaster that is excellent in February and poor in May cannot be trusted with a ceiling.

| Forecaster | Pinball mean | Pinball spread | Worst fold | Coverage mean | Worst coverage | Folds won |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence | 17.400 | ±2.818 | 22.148 | 0.903 | 0.872 | 0/8 |
| Seasonal naive | 5.193 | ±2.494 | 10.131 | 0.899 | 0.825 | 0/8 |
| Climatology | 5.121 | ±2.139 | 8.751 | 0.902 | 0.846 | 0/8 |
| Linear quantile | 6.840 | ±0.998 | 8.317 | 0.902 | 0.859 | 0/8 |
| LightGBM quantile (ours) | 2.987 | ±1.294 | 5.429 | 0.913 | 0.874 | 7/8 |
| Neural quantile | 3.490 | ±1.755 | 6.479 | 0.896 | 0.823 | 1/8 |
