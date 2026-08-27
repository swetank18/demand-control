# Where the novelty actually is

**Written to be attacked.** A novelty claim that cannot survive a reviewer opening
Google Scholar is worth less than no claim at all. So this document states what
is *not* new first, maps the prior art honestly, and only then says what is left.

---

## 1. What we are not claiming

None of the following is novel, and claiming any of it would be puncturable in
under a minute:

| Component | Status in the literature |
| --- | --- |
| Chance-constrained MPC for building energy | Established. Extensive literature on stochastic MPC for HVAC and storage. |
| **Joint** chance constraints over a horizon | Textbook. Boole/Bonferroni risk allocation, and the scenario approach (Calafiore & Campi). The intractability of joint constraints is a standard result. |
| Quantile gradient boosting for load forecasting | Standard practice. LightGBM quantile regression is a default baseline. |
| Conformal prediction, CQR, adaptive conformal | Established. Romano/Patterson/Candès; Gibbs & Candès for the adaptive online variant. |
| Cross-building / cross-climate transfer learning | Active research area. BDG2 has already been used for transferability assessment across building types and regions. |
| Predict-then-optimize / decision-focused learning | Established. Elmachtoub & Grigas (SPO), Donti et al. |
| Demand-charge management | Long-standing applied literature. |

**We did not invent an algorithm and we are not going to say we did.**

That is not a weakness in the pitch if the actual contribution is stated
precisely. It is a weakness only if the claim is vague enough to be mistaken for
an algorithmic one.

---

## 2. What the project actually is

An instrument, not an algorithm.

Every part is a known part. What did not exist is an end-to-end pipeline in which
**every component can be pinned and exactly one swapped, with the consequence
read out in rupees and ceiling breaches** — the tariff compiler, the quantile
forecaster, the chance-constrained MILP, the RC simulator, the stress injector
and the bill engine, all reproducible from a clean checkout by one script.

That instrument is what makes the three measurements below possible. They are the
contribution.

---

## 3. Contribution 1 — the horizon gap, bracketed on real data

### The claim

A demand ceiling defended by substituting a **marginal** 95th percentile at each
timestep is not a 95% guarantee about the thing anyone is buying. The event that
costs money is *"the ceiling is breached somewhere in the billing window"* — a
maximum over many steps, not a statement about any one step.

Everybody knows this in principle. What nobody in the applied demand-charge
literature reports is **how big the gap is on real meter data, and how badly the
standard conservative bound overshoots it.**

### The measurement

`eval/horizon_risk.py`, Fox_office_Gaylord, held-out June 2017, 5,793 forecast
origins:

| Horizon | Per-step exceedance | **Empirical horizon-level** | Independence (Boole) bound |
| --- | --- | --- | --- |
| 1 step (15 min) | 0.060 | 0.060 | 0.060 |
| 4 steps (1 h) | 0.060 | 0.091 | 0.219 |
| 8 steps (2 h) | 0.060 | 0.126 | 0.389 |
| 16 steps (4 h) | 0.060 | 0.182 | 0.625 |
| 32 steps (8 h) | 0.061 | 0.275 | 0.867 |
| **64 steps (16 h)** | **0.062** | **0.405** | **0.983** |

Read the last row carefully, because all three numbers are simultaneously true
and they are the whole point:

- The per-step quantile is **well calibrated** at 0.062 against a nominal 0.05.
  The forecaster is not broken. Marginal calibration is exactly what it promised.
- The risk of breaching **somewhere in the 16-hour planning window is 0.405** —
  roughly **seven times** the number the constraint appears to promise.
- The conservative bound a careful designer would reach for is **0.983**, which
  is **2.4× the truth** and so loose as to be operationally useless. Allocate
  risk with Boole and you give away most of your usable headroom for nothing.

The reason the truth sits so far below the bound is that load is heavily
autocorrelated — lag-1 correlation 0.871, decaying to 0.086 by lag 32. Boole
assumes the worst case over dependence structures; real buildings are nowhere
near it.

### Why this is worth something

It is a **falsifiable, quantitative statement about a formulation error that is
widespread in applied work**, and it comes with both ends of the bracket. The
practical consequence is concrete in both directions: a designer using marginal
quantiles is under-protected by ~7×, and a designer using the textbook bound is
over-protected by ~2.4× and pays for it in headroom.

### The honest gap

Our fix — a t-copula over the horizon (df 7) feeding a scenario MPC — reproduces
the marginals to within 4.6% of interval width, so it does not undo the
calibration work. But it **over-predicts** risk: 0.497 against an empirical
0.405 at 64 steps. It is conservative, which is the safe direction, and it is
still far tighter than Boole. The closed-loop acceptance test that would show
the scenario MPC actually delivers the ε it is set is **running now and is not
yet part of this claim.**

Until it lands, the honest statement is: *we have measured the problem
precisely and have a fix that is directionally right and demonstrably
conservative.* Not: *we solved it.*

---

## 4. Contribution 2 — conformal calibration degrades with aggregation level

### The claim

Distribution-free coverage guarantees earned on building-level data **do not
survive aggregation**. The nominal 90% interval very nearly holds on individual
buildings and fails systematically on system-level demand — and fails worst on
real Indian data.

### The measurement

24 series, one protocol, from `results/comparative.md`:

| Tier | n | mean coverage | worst | rows below 0.85 |
| --- | --- | --- | --- | --- |
| Buildings (BDG2, 3 countries, 8 usage classes) | 18 | **0.901** | 0.824 | 1 |
| National/city demand (6 countries incl. India) | 6 | **0.829** | **0.762** | **5** |

Delhi is the worst row in the whole study at **0.762** against a nominal 0.90.

### Why this is worth something

This is not a curiosity. **The controller reads its q95 from the same object as
that interval.** If the interval is too narrow, the demand ceiling it defends is
too narrow with it, and the safety property fails precisely where the project
most wants it to hold — on Indian load.

It also has a clean mechanism, and one our own audit had already found at
smaller scale: split conformal requires calibration and test blocks to be
*exchangeable*, and a Delhi May is not exchangeable with a Delhi June because
the monsoon arrives in between. `results/conformal_audit.md` documents the same
failure month-to-month at Fox, where coverage of the nominal 90% interval ranges
from 0.45 to 0.94 across a walk-forward year. Delhi is that failure mode with a
sharper regime change.

### What would falsify it

Running the adaptive conformal layer across the national tier and finding
coverage restored. That is the next experiment, and it is stated as an
experiment rather than a result.

---

## 5. Contribution 3 — skill and controllable load are independent axes

### The claim

"Will this method work in my country?" is two questions, not one, and only one
of them tracks climate.

### The measurement

Across 18 buildings spanning cooling-degree-day share 0.87 (Phoenix) to 0.00
(Dublin):

| Relationship | r | leave-one-out range | verdict |
| --- | --- | --- | --- |
| Controllable fraction vs climate | **+0.489** | — | real |
| Forecast **skill** vs climate | **+0.010** | −0.308 to +0.077 | **null** |
| Forecast skill vs controllable fraction | +0.283 | +0.029 to +0.378 | not robust |
| Forecast skill vs mean temperature | −0.012 | −0.266 to +0.051 | null |

The controllable fraction — the share of the meter that is weather-driven and
therefore movable — runs from **37.6% in Phoenix to 0.8% in London** and tracks
climate strongly. Forecast skill does not track it at all, and the leave-one-out
range crosses the sign line, which at n=18 is what a null result looks like.

We went in expecting the opposite and are reporting the null.

### Why this is worth something

It is a **deployment rule**. It says: do not screen a site by asking whether its
load is predictable. Screen it by asking how much of its load is weather-driven.
A Dublin office is perfectly predictable (skill +0.513) and has 3% controllable
load, so there is nothing to sell. Conflating the two axes — which is the
natural thing to do — gets that backwards.

It also happens to be good news for the target market, because India is at the
favourable end of the axis that matters: Delhi demand correlates **+0.59** with
outdoor temperature, against **−0.64** for France.

---

## 6. Contribution 4 — the negative results, which are the credibility

Three rows in the study argue against us and are printed at the same size as the
rest:

| Row | Result | What it means |
| --- | --- | --- |
| Robin (London) office | skill **−0.463** | our model is *worse* than a baseline needing no training at all |
| Delhi | skill **+0.097**, coverage **0.762** | on real Indian data, barely beats seasonal naive, worst calibration in the study |
| Public services (DC) | skill **+0.052** | near-zero |

All three share a cause: at 1–3% HVAC share, a weather-aware model has nothing
to add over "last week, same time". That is a coherent and testable explanation,
not an excuse.

Reporting these is itself a differentiator. Most submissions in this space report
the configuration that worked.

---

## 7. The claim, in one paragraph

> We did not invent an algorithm. We built an instrument in which every component
> can be pinned and one swapped, and used it to make three measurements that the
> literature asserts but does not quantify: that a per-step 95% guarantee on a
> demand ceiling is a 60% guarantee over a 16-hour window while the textbook
> conservative bound is 2.4× too loose; that conformal coverage degrades with
> aggregation level and fails worst on real Indian demand; and that forecast
> skill and controllable load are independent axes, only one of which tracks
> climate. All three are reproducible from a clean checkout, and three of our own
> results argue against us.

---

## 8. Why this is worth a judge's time

**It is falsifiable.** Every number is produced by a named script and
regenerated by `reproduce.sh`. Nothing is asserted that cannot be checked.

**It reports its own failures.** Three negative results, two defects the study
found in our own code (a hard-coded American holiday calendar that would have
corrupted every cross-country number, and a manifest that silently discarded
sites), and a fix that is honestly labelled conservative rather than correct.

**It is decision-relevant, not metric-relevant.** The output is rupees and
ceiling breaches. The forecaster is judged by what happens downstream when you
remove it — 11 more breaches, ₹12,885 more, 113 kW less usable headroom — not by
its MAE.

**It is about the money that actually moves.** The demand charge is set by a
single worst 30-minute block in a month. That is a risk problem, not a
prediction problem, and almost every project in this space optimises kWh and
never touches it.

**It travels.** The protocol runs unchanged on 19 sites, 4 countries, 8 building
types and 6 national grids. The weights do not transfer and we say so; the method
does, and we show where it stops paying.
