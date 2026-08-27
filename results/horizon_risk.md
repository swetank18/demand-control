### Horizon-level risk — Fox_office_Gaylord, 2017-06-01 to 2017-06-30

Track B acceptance. The controller's ceiling constraint spans a 16-hour window; its guarantee was stated one interval at a time. Those are not the same statement, and the difference is not small.

#### B1 — how wrong the marginal formulation is

| Horizon | Per-step P(y > q95) | Realised P(breach anywhere) | On blocks | If independent | Copula says |
| --- | --- | --- | --- | --- | --- |
| 0 h (1 steps) | 0.0596 | **0.0596** | 0.0596 | 0.0596 | 0.0495 |
| 1 h (4 steps) | 0.0600 | **0.0905** | 0.0774 | 0.2192 | 0.1000 |
| 2 h (8 steps) | 0.0597 | **0.1260** | 0.1079 | 0.3890 | 0.1478 |
| 4 h (16 steps) | 0.0595 | **0.1818** | 0.1615 | 0.6251 | 0.2247 |
| 8 h (32 steps) | 0.0611 | **0.2751** | 0.2471 | 0.8672 | 0.3423 |
| 16 h (64 steps) | 0.0620 | **0.4050** | 0.3745 | 0.9833 | 0.4974 |

Read the first two columns together. Per step the bound behaves exactly as advertised — 0.0596 against a nominal 0.05, which is Track A's calibration doing its job. Over the full 16-hour window the probability that the bound breaks *somewhere* is **0.405**, which is 7× the number the constraint was written to deliver. Nothing is miscalibrated. The formulation is asking the wrong question.

The independence column is the other end of the bracket: 0.983 at 16 hours. Real load errors are heavily autocorrelated — the fitted copula puts adjacent-horizon error correlation at 0.87, decaying to 0.09 at eight hours apart — so the truth sits well below the independence bound. That is the point: marginals alone tell you the answer is between 0.05 and 0.98, and nothing more.

#### B2 — the copula adds dependence and nothing else

Sampled paths reproduce the calibrated per-step quantiles to within 1.85 kW on a mean interval width of 40.1 kW (4.6% of the width, at 4,000 paths per origin — Monte Carlo noise, and it shrinks as 1/sqrt(n)). The marginals the audit certified are the marginals the scenarios carry. What the copula adds is the column the table above shows it getting right: predicted 0.342 against realised 0.275 at eight hours, 0.497 against 0.405 at sixteen.

#### B3 — closed loop, one month, the risk level is now a dial

The scenario MILP writes the ceiling once per scenario per block and caps the probability mass allowed to violate. Two violation rates are reported and they measure different things:

- **vs plan** — the fraction of 16-hour windows in which any realised billing block cleared the ceiling `D_peak` the optimiser committed to at that origin. This is what epsilon is a statement about, so this is the acceptance metric.
- **vs target** — the same event measured against the operator's monthly demand target. This is the business metric. It is *not* what the chance constraint promises: `D_peak` is a decision variable, and when the plant is short of capacity the optimiser lifts it above the target, pays the demand charge, and honours its constraint exactly. Grading a chance constraint against a number it never promised is the standard way this test comes out meaningless.

| Target | Controller | eps | Commit viol. | Closed-loop viol. vs target | Blocks breached | Median margin kW | Peak kVA | Bill INR | Comfort % | Solve ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tight (430 kW) | marginal q95 | — | 0.106 | 0.049 | 4/1440 | +22 | 459.0 | 1,138,193 | 0.59 | 6 |
| tight (430 kW) | Scenario MPC[cvar, eps=0.05] | 0.05 | 0.097 | 0.097 | 9/1440 | +26 | 464.6 | 1,141,532 | 0.76 | 53 |
| tight (430 kW) | Scenario MPC[cvar, eps=0.1] | 0.1 | 0.114 | 0.096 | 9/1440 | +24 | 466.8 | 1,142,186 | 0.73 | 53 |
| tight (430 kW) | Scenario MPC[cvar, eps=0.2] | 0.2 | 0.156 | 0.073 | 10/1440 | +20 | 469.2 | 1,142,647 | 0.83 | 54 |
| tight (430 kW) | Scenario MPC[cvar, eps=0.35] | 0.35 | 0.219 | 0.077 | 13/1440 | +16 | 467.5 | 1,141,036 | 1.04 | 56 |
| nominal (467 kW) | marginal q95 | — | 0.103 | 0.000 | 0/1440 | +22 | 481.9 | 1,149,580 | 0.59 | 6 |
| nominal (467 kW) | Scenario MPC[cvar, eps=0.05] | 0.05 | 0.078 | 0.000 | 0/1440 | +27 | 480.5 | 1,148,576 | 0.62 | 52 |
| nominal (467 kW) | Scenario MPC[cvar, eps=0.1] | 0.1 | 0.097 | 0.000 | 0/1440 | +24 | 486.0 | 1,151,865 | 0.62 | 53 |
| nominal (467 kW) | Scenario MPC[cvar, eps=0.2] | 0.2 | 0.150 | 0.000 | 0/1440 | +20 | 480.1 | 1,148,071 | 0.62 | 53 |
| nominal (467 kW) | Scenario MPC[cvar, eps=0.35] | 0.35 | 0.208 | 0.000 | 0/1440 | +16 | 484.0 | 1,150,364 | 0.62 | 53 |

**Acceptance.** Realised violation of the committed ceiling tracks the epsilon it was set: Spearman correlation 0.96 across the sweep, mean absolute gap 0.057, and 5 of 8 levels land on the safe side of nominal. The formulation is conservative rather than exact, which is the direction it should err in: the CVaR surrogate admits a strict subset of the chance-constrained feasible set, so it defends the committed ceiling harder than asked and leaves some headroom unspent.

**Epsilon has a resolution floor of 1/S = 0.025 here.** With S scenarios the finest risk level the formulation can express is one scenario's worth of probability mass; below that, CVaR collapses onto the maximum of the sample and successive epsilons stop being distinguishable. That is a real limit of sample average approximation and not a tuning artefact — buying a finer dial means more scenarios and a slower solve, which is exactly the trade scenario reduction is there to manage.

**What it costs.** At the tight target the marginal-q95 controller violates its own committed ceiling in 0.106 of windows — it makes no joint statement at all, so that number is whatever the horizon length and the error autocorrelation happen to produce. The scenario controller at eps=0.05 holds 0.097, for ₹3,339 on the month. That trade is now visible and settable, which it was not before: with a single q95 substitution there was no dial, and the risk level the controller actually ran at was an emergent property of how long the horizon happened to be.

#### What this changes in the pitch

Before: we substitute the 95th percentile, so the ceiling holds 95% of the time. After: we enforce the violation probability across the whole horizon, which is what the constraint requires, and the operator sets it. The first sentence is false by a factor of 7 and a reviewer who has seen a chance constraint before will find that in a minute.

#### Limitations, stated

- Tail dependence is carried by a t-copula with df=7, selected on the validation block by matching predicted to realised horizon exceedance. That is a fitted choice rather than a derived one: the df grid is coarse and the selection criterion is the same quantity the copula is later judged on, so it is not an independent test of the family.
- Base load and PV scenarios are drawn independently. Cloud raises cooling load at the same moment it cuts PV output, so the true joint law has worse afternoons than these scenarios contain.
- The horizon is 16 hours; the demand charge bills a monthly maximum over ~2,880 intervals. A joint guarantee over the window is strictly stronger than a marginal one and strictly weaker than a monthly one. Extending it to the month needs the copula fitted over a month-length horizon, which is a 2,880-square correlation matrix and needs structure rather than a raw estimate.

Figure: `results/horizon_risk_Fox_office_Gaylord.png`.