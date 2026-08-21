### Ablation — Fox_office_Gaylord, 2017-06-01 to 2017-06-30 23:45

Demand target **467 kW**, held fixed on every row. Contract demand 505 kVA. Tariff: TNERC Tariff Order T.P. No. 1 of 2023 as amended for FY 2025-26; ToD windows per Electricity (Rights of Consumers) Amendment Rules 2023.

Held fixed: demand target, MILP horizon and settings, RC parameters, comfort budget, PV array and solar quantiles, tariff, seed. Varied: the base-load quantile forecast only.

Forecast quality on the left, control outcome on the right. A breach is a billing block whose average exceeded the target; each one is a permanent monthly cost, which is why the count and not the average is the safety metric.

| Forecaster feeding the optimiser | Pinball | Cov 90% | Breaches | Peak kVA | Bill INR | Usable headroom kW | Comfort violated % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 20.307* | 0.852* | 11 | 503.4 | 1,162,491 | 240.9 | 5.62 |
| Persistence | 27.582 | 0.820 | 87 | 556.8 | 1,203,766 | 342.9 | 1.39 |
| Seasonal naive | 6.213 | 0.884 | 0 | 482.8 | 1,150,421 | 339.2 | 1.15 |
| Climatology | 7.314 | 0.863 | 1 | 494.8 | 1,157,695 | 328.8 | 1.15 |
| Linear quantile | 10.758 | 0.880 | 0 | 490.5 | 1,155,519 | 314.1 | 0.97 |
| **LightGBM quantile (ours)** | **2.397** | **0.891** | **0** | **481.9** | **1,149,605** | **354.1** | **0.59** |
| Neural quantile | 2.435 | 0.894 | 0 | 477.5 | 1,146,700 | 359.7 | 0.62 |
| Perfect foresight | 0.000 | 1.000 | 0 | 480.9 | 1,148,785 | 376.0 | 0.66 |

\* the static margin is one number for the whole month, so its pinball loss and coverage describe a constant rather than a forecast.

**Replacing the forecaster with a constant** costs +11 ceiling breaches, ₹+12,885 on the month, and -113.2 kW of usable headroom (-32%).

### Ablation — Fox_office_Gaylord, 2017-06-01 to 2017-06-30 23:45, stress: heatwave

Demand target **467 kW**, held fixed on every row. Contract demand 505 kVA. Tariff: TNERC Tariff Order T.P. No. 1 of 2023 as amended for FY 2025-26; ToD windows per Electricity (Rights of Consumers) Amendment Rules 2023.

Held fixed: demand target, MILP horizon and settings, RC parameters, comfort budget, PV array and solar quantiles, tariff, seed. Varied: the base-load quantile forecast only.

Forecast quality on the left, control outcome on the right. A breach is a billing block whose average exceeded the target; each one is a permanent monthly cost, which is why the count and not the average is the safety metric.

| Forecaster feeding the optimiser | Pinball | Cov 90% | Breaches | Peak kVA | Bill INR | Usable headroom kW | Comfort violated % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 20.307* | 0.852* | 15 | 506.2 | 1,194,198 | 240.8 | 6.08 |
| Persistence | 27.582 | 0.820 | 104 | 556.8 | 1,233,458 | 342.8 | 1.49 |
| Seasonal naive | 6.213 | 0.884 | 0 | 487.4 | 1,183,179 | 339.1 | 1.25 |
| Climatology | 7.314 | 0.863 | 3 | 497.1 | 1,189,003 | 328.7 | 1.25 |
| Linear quantile | 10.758 | 0.880 | 3 | 501.6 | 1,192,089 | 314.0 | 1.08 |
| **LightGBM quantile (ours)** | **2.397** | **0.891** | **2** | **501.6** | **1,191,414** | **354.0** | **0.69** |
| Neural quantile | 2.435 | 0.894 | 2 | 502.1 | 1,191,609 | 359.6 | 0.69 |
| Perfect foresight | 0.000 | 1.000 | 2 | 497.2 | 1,188,476 | 375.9 | 0.73 |

\* the static margin is one number for the whole month, so its pinball loss and coverage describe a constant rather than a forecast.

**Replacing the forecaster with a constant** costs +13 ceiling breaches, ₹+2,784 on the month, and -113.2 kW of usable headroom (-32%).
