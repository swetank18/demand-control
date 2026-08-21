### Ablation — Fox_public_Denny, 2017-06-01 to 2017-06-30 23:45

Demand target **261 kW**, held fixed on every row. Contract demand 290 kVA. Tariff: TNERC Tariff Order T.P. No. 1 of 2023 as amended for FY 2025-26; ToD windows per Electricity (Rights of Consumers) Amendment Rules 2023.

Held fixed: demand target, MILP horizon and settings, RC parameters, comfort budget, PV array and solar quantiles, tariff, seed. Varied: the base-load quantile forecast only.

Forecast quality on the left, control outcome on the right. A breach is a billing block whose average exceeded the target; each one is a permanent monthly cost, which is why the count and not the average is the safety metric.

| Forecaster feeding the optimiser | Pinball | Cov 90% | Breaches | Peak kVA | Bill INR | Usable headroom kW | Comfort violated % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 4.337* | 0.887* | 0 | 271.4 | 914,312 | 136.9 | 1.18 |
| Persistence | 6.146 | 0.875 | 25 | 308.0 | 939,334 | 154.1 | 0.87 |
| Seasonal naive | 3.037 | 0.878 | 0 | 272.1 | 916,188 | 146.6 | 0.73 |
| Climatology | 3.230 | 0.889 | 0 | 269.7 | 913,240 | 146.0 | 0.76 |
| Linear quantile | 3.448 | 0.895 | 0 | 273.7 | 915,566 | 147.9 | 0.80 |
| **LightGBM quantile (ours)** | **2.576** | **0.897** | **0** | **273.8** | **915,706** | **150.4** | **0.76** |
| Neural quantile | 3.392 | 0.887 | 0 | 274.3 | 916,468 | 147.6 | 0.97 |
| Perfect foresight | 0.000 | 1.000 | 0 | 272.6 | 914,948 | 166.8 | 0.76 |

\* the static margin is one number for the whole month, so its pinball loss and coverage describe a constant rather than a forecast.

**Replacing the forecaster with a constant** costs +0 ceiling breaches, ₹-1,394 on the month, and -13.5 kW of usable headroom (-9%).
