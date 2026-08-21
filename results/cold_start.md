### Cold start — Fox_public_Denny held out of training entirely

Trained on `Fox_office_Gaylord`, `Fox_assembly_Dixie`, `Fox_education_Etta`. The held-out building contributes only 14 days of its own history, used to set its scale (109 kW) and fill its lag features. Tested on 2017-06-01 to 2017-06-30.

| Model | Pinball | Cov 90% | q95 hit | Width kW | MAE kW |
| --- | --- | --- | --- | --- | --- |
| Trained on this building (the ceiling for this comparison) | 2.576 | 0.897 | 0.939 | 36.8 | 8.11 |
| **Cold start (trained on the other three, never on this one)** | 3.546 | 0.891 | 0.940 | 40.8 | 12.70 |
| Seasonal naive (needs one week of history and no model at all) | 3.037 | 0.878 | 0.939 | 40.2 | 9.40 |

Held out entirely, the model scores 3.546 pinball on Fox_public_Denny against 2.576 for the same model trained on that building — +38%. It loses to seasonal naive (3.037), which is what the site could do on day one with no model. Coverage on the unseen building is 0.891 against a nominal 0.90.
