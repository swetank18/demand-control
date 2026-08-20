# Judge questions — two-sentence answers, prepared

**Why not reinforcement learning?**
Across CityLearn 2021–2023 at NeurIPS, none of the top-performing teams used RL
for scheduling — the winners used classical optimisation, because there is no
simulation environment rich enough to train a policy on. We have less training
environment than they did, and a MILP gives us a constraint we can actually
guarantee rather than a policy we have to trust.

**Does this work on a building you have not seen?**
The forecaster needs about two weeks of meter history to build its lag features,
and the physics parameters are fitted from the building's own meter by changepoint
regression, so cold start is weeks not months. We ran the same pipeline unchanged
across four buildings with very different shapes — an office, an event hall, a
flat campus block and a late-running public building — and all four passed the
same calibration acceptance test.

**What happens when the forecast is wrong?**
That is the demo — press heatwave and watch the mean-forecast controller cross
the line four times while ours does not. More precisely: the q95 substitution
buys protection against ordinary forecast error, and the adaptive conformal layer
widens the interval online when a regime shifts, which is what keeps June covered
at 89.7% instead of 85.3%.

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
is weaker here than it would be in a real Indian office, and essentially all of
our saving comes from the demand charge.
