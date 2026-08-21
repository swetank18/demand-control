# Judge questions — two-sentence answers, prepared

## The model (asked first in round 2, so answer these cold)

**Where is the model, exactly?**
It produces `base_q95` — the 95th percentile of the next sixteen hours of base
load at 15-minute resolution — and that number is a term in the demand-ceiling
constraint of the MILP, so it decides how much capacity gets handed out at every
interval. Its output is not a chart; every kilowatt of forecast error is a
kilowatt of headroom we either waste or dangerously give away.

**How do you know the model is doing anything?**
Take it out: same optimiser, same month, same tariff, physics, comfort budget, PV
and seed, only the forecaster changes. Replace it with a constant and the same
controller takes 11 more ceiling breaches, pays ₹12,885 more and loses 113 kW of
usable headroom (`results/ablation.md`).

**Why forecast at all, when an engineer would just derate the transformer?**
We swept exactly that — a fixed allowance from the 50th percentile of historical
load to the maximum, everything else held fixed. The setting that is as safe as
we are leaves 198 kW usable where we leave 354: the forecast recovers 156 kW,
79% more capacity at equal safety.

**Why gradient boosting and not deep learning?**
We benchmarked both on identical features with identical pinball loss and the
identical calibration layer: LightGBM 2.397 pinball, the neural quantile net
2.435. At this data scale that is the expected result, and we would rather show
you the table than defend a preference.

**How do you know you are not overfitting?**
Temporal split, never random — train to 2017-03-31, calibrate on April–May, test
once on June — plus 8-fold expanding-window walk-forward so a single lucky split
cannot carry the claim. The leakage check is a unit test that runs before any
training: `tests/test_leakage.py`.

**What is your accuracy?**
Point accuracy is not the operative metric here — coverage is, because the
constraint is built on a quantile, and the 90% interval covers at 0.891 on data
the model never saw. If you want the point number anyway, the median forecast's
MAE is 7.0 kW on a building that runs at 300–500 kW.

**How much does the model quality actually buy, in money?**
Fitted across the whole family of forecasters: ₹2,228 a month per unit of pinball
loss (R² 0.95) and 3.5 ceiling breaches per unit (R² 0.89). That is a measured
exchange rate on this building, not a claim that better models are better.

**What does the model get wrong?**
19 June at 14:30: 290 kW arrived against a q95 of 264, a 26 kW exceedance, and
the day's q95 hit rate fell to 0.596. No breach followed, because there were
203 kW of headroom under the ceiling at that moment — which is the entire point
of planning on a quantile rather than a mean.

## The rest

**Why not reinforcement learning?**
Across CityLearn 2021–2023 at NeurIPS, none of the top-performing teams used RL
for scheduling — the winners used classical optimisation, because there is no
simulation environment rich enough to train a policy on. We have less training
environment than they did, and a MILP gives us a constraint we can actually
guarantee rather than a policy we have to trust.

**Does this work on a building you have not seen?**
We measured it rather than asserting it: one building held out of training
entirely, given only 14 days of its own history to set scale and fill lags. It
costs 38% of pinball against a model trained on that building and it *loses to
seasonal naive* there — but it keeps coverage at 0.891 against a nominal 0.90,
which is the property the controller actually depends on. So day one at a new
site is seasonal naive plus our conformal interval, and a site-specific model
once there is a season of data (`results/cold_start.md`).

**What happens when the forecast is wrong?**
That is the demo — press heatwave and watch the mean-forecast controller cross
the line four times while ours does not. More precisely: the q95 substitution
buys protection against ordinary forecast error, and the adaptive conformal layer
widens the interval online when a regime shifts, which is what keeps June covered
at 0.890 instead of the 0.832 split conformal alone manages.

**How do I know the savings are real and not just a mild month?**
We compute a counterfactual bill from a weather-and-calendar baseline model fitted
on the pre-period, with a bootstrap confidence band, and because this is
simulation we can check the answer against ground truth. We also found that the
usual approach under-reports by 3× on a short baseline — you need twelve months
of baseline data before the confidence band actually covers the truth, and we
would rather tell you that than quote you a number from three.

**What does it cost to deploy?**
In the base case, nothing new: it reads interval data and writes setpoints
through an existing building management system, and the whole optimisation solves
in about 8 milliseconds on a laptop. The costed additions are optional — storage
and PV are modelled but the results above hold without either.

**Occupants will hate being too warm.**
The comfort band is an operator input, not something the optimiser chooses, and
violations are reported in every table rather than hidden. `results/frontier.json`
gives the actual trade: pick your ceiling and we will tell you the lowest demand
you can commit to and what it is worth.

**Which state's tariff? What about mine?**
Tamil Nadu HT-I-A, hand-encoded from the published order and cross-checked by a
parser that reproduces the same JSON from the order text. Maharashtra is encoded
too, and a test asserts the same code gives a different answer — in Maharashtra
midday is the cheap window, which inverts the whole shifting strategy.

**Who buys this?**
The facility manager of a commercial or campus building above 10 kW sanctioned
demand — which is the population the time-of-day mandate already covers — and the
purchase is really the contract-demand decision, because a chance constraint you
can trust is what lets you commit to a lower sanctioned demand safely.

**Your saving is only 6% of peak. Is that worth anything?**
It is ₹171,000 a month on this building against doing nothing, and the honest
framing is that 6% is close to the physical ceiling: the whole-month oracle, with
perfect foresight, only reaches 483 kVA against our 481. We report the gap to the
oracle rather than to a strawman precisely so that this is visible.

**Why does your rule-based baseline do worse than doing nothing?**
Because it pre-cools ahead of the peak window and the pre-cooling sets its own
monthly maximum — 528 kVA against 523 for the uncontrolled building. That is the
best argument we have for why tariff-awareness without risk-awareness is not
merely insufficient, it is harmful.

**What is the single weakest thing here?**
The buildings are American, in Arizona, on an Indian tariff. The climate and
solar geometry are defensible analogues, but the load *shapes* are not Indian —
these buildings peak at 14:00, inside the normal window, so the time-of-day lever
is weaker here than it would be in a real Indian office.

Second weakest, and we would rather say it than have it found: most of the rupees
on this building are not the demand charge. The measured split is ₹138,613 a month
of energy against ₹25,153 of demand charge, and the energy share comes from the
controller using its full comfort band rather than holding a fixed setpoint — any
competent scheduler would get it. The demand-charge line is the smaller one, and
it is the only one that needs the forecast: it is what the ablation moves.
