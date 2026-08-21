### Ablation — Fox_assembly_Dixie, 2017-06-01 to 2017-06-30 23:45

Demand target **213 kW**, held fixed on every row. Contract demand 295 kVA. Tariff: TNERC Tariff Order T.P. No. 1 of 2023 as amended for FY 2025-26; ToD windows per Electricity (Rights of Consumers) Amendment Rules 2023.

Held fixed: demand target, MILP horizon and settings, RC parameters, comfort budget, PV array and solar quantiles, tariff, seed. Varied: the base-load quantile forecast only.

Forecast quality on the left, control outcome on the right. A breach is a billing block whose average exceeded the target; each one is a permanent monthly cost, which is why the count and not the average is the safety metric.

| Forecaster feeding the optimiser | Pinball | Cov 90% | Breaches | Peak kVA | Bill INR | Usable headroom kW | Comfort violated % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 9.682* | 0.924* | 0 | 223.2 | 563,428 | 117.2 | 1.56 |
| Persistence | 13.374 | 0.938 | 255 | 302.3 | 599,734 | 162.2 | 0.17 |
| Seasonal naive | 7.275 | 0.919 | 7 | 267.5 | 587,993 | 126.8 | 0.45 |
| Climatology | 5.750 | 0.909 | 0 | 222.0 | 573,282 | 146.7 | 0.24 |
| Linear quantile | 5.574 | 0.901 | 0 | 219.2 | 564,533 | 159.2 | 0.35 |
| **LightGBM quantile (ours)** | **4.927** | **0.913** | **1** | **224.6** | **564,469** | **160.1** | **0.35** |
| Neural quantile | 6.736 | 0.931 | 11 | 237.8 | 582,981 | 141.4 | 0.35 |
| Perfect foresight | 0.000 | 1.000 | 0 | 221.8 | 559,229 | 191.7 | 0.59 |

\* the static margin is one number for the whole month, so its pinball loss and coverage describe a constant rather than a forecast.

**Replacing the forecaster with a constant** costs -1 ceiling breaches, ₹-1,041 on the month, and -42.9 kW of usable headroom (-27%).
