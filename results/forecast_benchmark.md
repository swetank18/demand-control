### Forecast benchmark — Fox_office_Gaylord

Trained 2016-01-01 to 2017-03-31, calibrated 2017-04-01 to 2017-05-31, tested once on 2017-06-01 to 2017-06-30. n = 184,320 (origin, horizon) pairs, horizons out to 16 hours.

Every row goes through the identical split-conformal plus adaptive-conformal layer, so what is compared is the forecast underneath it.

| Forecaster | Pinball | CRPS | Winkler 90 | MAE kW | Cov 90% | q95 hit | Width kW | Cal err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear quantile | 10.781 | 23.250 | 154.9 | 37.18 | 0.879 | 0.931 | 132.6 | 0.0086 |
| Neural quantile | **2.435** | **5.107** | **45.0** | **7.59** | 0.894 | 0.952 | 33.5 | **0.0044** |

Definitions:

- **Linear quantile** — pinball-loss linear regression on the identical feature matrix
- **Neural quantile** — residual MLP stack, shared trunk, pinball head on all five levels
