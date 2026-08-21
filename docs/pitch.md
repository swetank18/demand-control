# Pitch — twelve slides, ~8 minutes plus demo

Round 1 was told: there is no model in it, nothing is trained, everything is
vague, and impact is unquantified. This deck is rebuilt around that. The model
leads, the ablation is the loudest slide, and every number below is produced by a
script in `results/`. Nothing is typed by hand. If a number here disagrees with
`results/`, `results/` is right and this file is stale.

Regenerate all of it: `./reproduce.sh`.

---

## 1. The problem, unchanged

An Indian HT commercial bill has three levers. This building, June 2017,
uncontrolled — straight from the bill engine:

| Lever | Amount | Share of bill |
| --- | --- | --- |
| Energy charge (154,250 kWh across four ToD windows) | ₹954,407 | 72.3% |
| **Demand charge (523.3 kVA × ₹608)** | **₹318,175** | **24.1%** |
| Electricity duty | ₹47,720 | 3.6% |
| **Total** | **₹1,320,302** | |

A quarter of the bill is set by **one thirty-minute block** — here, 12:00 on
20 June. One extra hour of load on one Tuesday afternoon costs ₹1,270 in energy
and **₹119,200** in demand charge, a factor of 94. That is a test, not a claim:
`tests/test_bill.py::test_one_bad_tuesday_sets_the_month`.

Fifteen seconds. Then move.

## 2. Why this is a prediction problem — and say the word *model* here

You cannot allocate headroom you cannot measure in advance. To decide at 12:00
what the building is allowed to draw at 14:00, you must know what 14:00 looks
like. **This is where the model enters, and it enters inside a constraint:**

```
base_q95[t]  +  controllable[t]  −  solar_q05[t]   ≤   D_ceiling
```

The forecast's output is not a chart for a human. It is `base_q95` — a number in
a hard constraint in a MILP. Every kilowatt of forecast error is a kilowatt of
headroom we either waste or dangerously give away.

## 3. Why not a static margin — the answer to "why is there a model at all"

Every distribution engineer already knows how to protect a transformer without
machine learning: never plan on more than a fixed allowance. So we swept it.
Each row of `results/impact.json → static_margin_study` is a full simulated
month on a constant derating, everything else held fixed.

| Static derating | Breaches | Usable headroom |
| --- | --- | --- |
| p50 of training load (102 kW) | 13 | 390.0 kW |
| p80 (195 kW) | 5 | 296.5 kW |
| p90 (226 kW) | 12 | 266.2 kW |
| p100 — the training maximum (294 kW) | **0** | 198.1 kW |
| **Ours** | **0** | **354.1 kW** |

**A forecast-free margin that is as safe as we are leaves 198 kW usable. We
leave 354 kW. The forecast recovers 156 kW — 79% more usable capacity at equal
safety.** That is the value of the model, in the units the problem is about.

Note the non-monotonicity in the middle of that sweep (p80 breaches less than
p90). It is real and it is worth volunteering: a fixed margin that is too tight
defers load into a rebound peak later, so "more conservative" does not reliably
mean "safer". That instability is itself an argument against fixed margins.

## 4. The model

- **Target** total base (uncontrollable) load, 15-minute resolution, 64 steps —
  16 hours ahead, direct multi-horizon
- **Output** quantiles at 0.05, 0.25, 0.50, 0.75, 0.95, one booster per quantile
- **Family** LightGBM with `objective='quantile'`, then split conformal plus an
  adaptive online correction
- **Split** train 2016-01-01 → 2017-03-31, calibrate 2017-04-01 → 2017-05-31,
  test once on 2017-06-01 → 2017-06-30. Temporal, never random. The leakage
  check is a unit test: `tests/test_leakage.py`.

Held out, June 2017, n = 184,320 (origin, horizon) pairs. Every row goes through
the identical calibration layer, so what is compared is the forecast underneath:

| Forecaster | Pinball | CRPS | Winkler 90 | MAE kW | Cov 90% | Width kW |
| --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 20.307 | 44.341 | 252.5 | 72.28 | 0.852 | 223.4 |
| Persistence | 27.582 | 59.242 | 404.3 | 89.47 | 0.820 | 314.7 |
| Seasonal naive | 6.213 | 12.796 | 131.4 | 18.37 | 0.884 | 72.2 |
| Climatology | 7.314 | 15.849 | 102.1 | 26.75 | 0.863 | 86.5 |
| Linear quantile | 10.758 | 23.205 | 154.4 | 37.19 | 0.880 | 132.5 |
| **LightGBM quantile (ours)** | **2.397** | **4.922** | 51.7 | **7.03** | 0.891 | 39.6 |
| Neural quantile | 2.435 | 5.107 | **45.0** | 7.59 | 0.894 | 33.5 |
| Perfect foresight | 0.000 | 0.000 | 0.0 | 0.00 | n/a | 0.0 |

**On deep learning, before anyone asks.** The neural quantile net is a residual
MLP with a pinball head on all five levels, trained on the identical features and
put through the identical calibration layer. It lands within 0.04 pinball of the
boosters — the expected result at this data scale. We are not avoiding deep
learning; we benchmarked it, and we can show the table.

**One split is not evidence, so here are eight.** Expanding-window walk-forward,
each fold trained to a month end, calibrated on the month before the score month,
scored on the month after:

| Forecaster | Pinball mean | Spread | Worst fold | Cov mean | Worst cov | Folds won |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence | 17.400 | ±2.818 | 22.148 | 0.903 | 0.872 | 0/8 |
| Seasonal naive | 5.193 | ±2.494 | 10.131 | 0.899 | 0.825 | 0/8 |
| Climatology | 5.121 | ±2.139 | 8.751 | 0.902 | 0.846 | 0/8 |
| Linear quantile | 6.840 | ±0.998 | 8.317 | 0.902 | 0.859 | 0/8 |
| **LightGBM quantile (ours)** | **2.987** | ±1.294 | **5.429** | 0.913 | 0.874 | **7/8** |
| Neural quantile | 3.490 | ±1.755 | 6.479 | 0.896 | 0.823 | 1/8 |

Ours wins **7 of 8 folds**, has the best worst-fold loss (5.429) and holds
coverage at 0.913 on average and 0.874 at its worst. The spread matters as much
as the mean: a forecaster that is excellent in February and poor in May cannot be
trusted with a transformer. Note that persistence's *coverage* is fine across
folds — its intervals are simply enormous, which is exactly the failure mode the
ablation converts into 87 breaches.

## 5. Calibration is the safety property

If q95 is not really a 95th percentile, the guarantee in slide 2 is theatre. So
it is measured on held-out data:

| Building | 90% coverage | P(y ≤ q95) | worst horizon |
| --- | --- | --- | --- |
| Office (Gaylord) | 0.890 | 0.939 | 0.888 |
| Assembly (Dixie) | 0.920 | 0.953 | 0.916 |
| Education (Etta) | 0.926 | 0.960 | 0.922 |
| Public (Denny) | 0.896 | 0.939 | 0.893 |

Show the reliability diagram, then the honest detail — it is the strongest thing
on this slide. Split conformal calibrated on April–May **under-covers June at
0.832**, below our own acceptance floor of 0.85. June is hotter; a textbook
distribution shift. Adaptive conformal closes the loop online and restores
**0.890**, paying 6.4 kW of interval width (32.2 → 38.6 kW) for it. Without that
layer the forecaster fails its own test.

Reproduce the failure: `forecast/train.py --buildings Fox_office_Gaylord --no-adaptive`.

## 6. THE ABLATION — the loudest slide in the deck

**Same optimiser. Same month. Same tariff, physics, comfort budget, PV, seed.
The only thing that changes is the file the base-load quantiles are read from.**

| Forecaster feeding the optimiser | Pinball | Cov 90% | Breaches | Peak kVA | Bill ₹ | Usable headroom kW |
| --- | --- | --- | --- | --- | --- | --- |
| Static margin (no forecast) | 20.307* | 0.852* | 11 | 503.4 | 1,162,491 | 240.9 |
| Persistence | 27.582 | 0.820 | 87 | 556.8 | 1,203,766 | 342.9 |
| Seasonal naive | 6.213 | 0.884 | 0 | 482.8 | 1,150,421 | 339.2 |
| Climatology | 7.314 | 0.863 | 1 | 494.8 | 1,157,695 | 328.8 |
| Linear quantile | 10.758 | 0.880 | 0 | 490.5 | 1,155,519 | 314.1 |
| **LightGBM quantile (ours)** | **2.397** | **0.891** | **0** | **481.9** | **1,149,605** | **354.1** |
| Neural quantile | 2.435 | 0.894 | 0 | 477.5 | 1,146,700 | 359.7 |
| Perfect foresight | 0.000 | 1.000 | 0 | 480.9 | 1,148,785 | 376.0 |

\* a constant is not a forecast; scoring it as one flatters it.

**The line to say out loud:** replace the forecaster with a constant and the same
controller takes **11 more ceiling breaches**, pays **₹12,885 more**, and has
**113 kW less** usable headroom. Nothing else moved.

Two things to volunteer here before a judge finds them:

1. **Seasonal naive already gets to zero breaches.** On a building with a
   timetable, last week's shape is a strong forecast. Our advantage over it is
   not the breach count — it is 15 kW more headroom, a lower peak, a bill ₹816
   lower, and a pinball loss 2.6× better on the same month. Do not overclaim
   here: on a calm month against a building with a timetable, last week's shape
   is a genuinely good forecast, and the cold-start study on slide 8 has it
   beating our *transferred* model outright. It is the reference we would run at
   a new site for a fortnight. What it does not have is margin when the month
   stops being calm — see the heatwave row below.
2. **Persistence claims the second-most headroom in the table and breaches 87
   times.** Headroom without calibration is not a benefit; it is an overdraft.
   That row is why coverage, not accuracy, is the operative metric.

### The same table under a heatwave — and it goes against us

+6 °C and +25% base load for three days, everything else fixed:

| Forecaster | Breaches | Interval width kW |
| --- | --- | --- |
| Static margin (no forecast) | 15 | 223.4 |
| Persistence | 104 | 314.7 |
| **Seasonal naive** | **0** | 72.2 |
| Climatology | 3 | 86.5 |
| **LightGBM quantile (ours)** | **2** | 39.6 |
| Perfect foresight | 2 | 0.0 |

Say it before a judge finds it: **seasonal naive takes zero breaches here and we
take two — and so does a perfect forecast of the unstressed month.** Under a
shock that is in nobody's inputs, what defends the ceiling is interval width, and
theirs is 1.8× ours. Sharpness buys headroom on an ordinary month; it is what
costs you on a day the model could not have known about.

And the caveat that matters: this table measures *margin*, not *adaptation*. The
tensors are computed offline, so no forecaster here can react to the heatwave.
The adaptive layer that would react is the one measured on slide 5 — 0.832 → 0.890
across a real distribution shift.

### And on two more buildings, where it partly does not hold

| Building | Ceiling | Our pinball | Breaches: ours / no model / persistence | Headroom gained | Bill vs no model |
| --- | --- | --- | --- | --- | --- |
| Office tower (Gaylord) | 467 kW | 2.397 | 0 / 11 / 87 | +113.2 kW | ₹12,885 better |
| Assembly hall (Dixie) | 213 kW | 4.927 | 1 / 0 / 255 | +42.9 kW | ₹1,041 worse |
| Public services (Denny) | 261 kW | 2.576 | 0 / 0 / 25 | +13.5 kW | ₹1,394 worse |

Volunteer this. **Only the office tower reproduces the headline** — it is the one
where the ceiling binds and the load is predictable. On the campus buildings the
no-model constant already holds the ceiling and we finish a few hundred rupees
behind, partly because the ceiling was bisected with our own forecaster so we sit
at its tightest point.

What holds everywhere: 113, 43 and 14 kW of capacity recovered, and persistence
destroying the ceiling on all three (87, 255, 25 breaches).

**The claim to make, and only this one:** a forecast buys capacity, a badly
calibrated one is worse than no forecast at all, and whether a good forecast also
buys rupees depends on whether the ceiling binds.

## 7. The frontier — what forecast quality is worth, in rupees

Take the table above and plot it: forecast quality on the x-axis, downstream
outcome on the y. `results/model_frontier.json`, figure
`results/figures/06_model_frontier.png`.

- **₹2,228 per unit of pinball loss per month** (R² 0.95)
- **3.5 ceiling breaches per unit of pinball loss** (R² 0.89)
- Calibration error against breaches is monotonic (Spearman ρ = 0.85, p = 0.03)

That is a measured exchange rate between model quality and money on this
building. Panel D — headroom against pinball — is **not** monotonic, and we
report that as the finding it is: persistence buys headroom it cannot pay for.
The like-for-like comparison is slide 3, where breach counts are matched first.

## 8. Does it work on a building it has never seen?

Cold start. `Fox_public_Denny` is held out of training entirely and contributes
only 14 days of its own history, used to set its scale (109 kW) and fill its
lags — exactly what a new site has on day one.

| Model | Pinball | Cov 90% | q95 hit | MAE kW |
| --- | --- | --- | --- | --- |
| Trained on this building (the ceiling) | 2.576 | 0.897 | 0.939 | 8.11 |
| **Cold start — never saw this building** | **3.546** | **0.891** | **0.940** | **12.70** |
| Seasonal naive (day-one alternative) | 3.037 | 0.878 | 0.939 | 9.40 |

**Say the result, including the part that is against us.** Cold start costs 38%
of pinball against a model trained on the building, and on this building it does
not beat seasonal naive. What survives the transfer is the thing the controller
actually needs: **coverage of 0.891 against a nominal 0.90 on a building the
model has never seen.** So the honest deployment story is: run seasonal naive at
a new site for the first fortnight, run the pooled model with its conformal layer
from day one for the *interval*, and switch to a site-specific model once you
have a season of data. We would rather say that than claim a transfer that the
numbers do not support.

## 9. It is not a black box

- **SHAP** on the q95 booster: `hod_cos`, `dow_sin`, `solar_elev`, `cdd_fut`,
  `rmean_672`, `t_out_fut`. Time-of-day and weekday dominate, weather next.
  Nothing silly is at the top — and this is where we would have found it.
- **Feature-group ablation** — drop a group, retrain, measure the damage:
  calendar **+177%** pinball, weather forecast **+34%**, recent lags **−1.3%**,
  rolling statistics **−6.8%**.
  The two studies disagree and the disagreement is the finding: sixteen hours
  ahead, what the meter read an hour ago carries almost nothing. That is exactly
  why persistence collapses in the ablation.
- **Volunteer the caveat attached to that number.** Our weather feed at the
  target time is the recorded observation, i.e. a perfect weather forecast, and
  the ablation says a third of the model's edge rides on it. A real deployment
  buys an IMD or vendor forecast with its own error, so treat the weather-forecast
  row as an upper bound. `forecast/train.py --weather-noise-c` exists precisely
  to degrade it, and `docs/limitations.md` said so before these results existed.
- **The evening we got wrong.** 19 June, 14:30: actual 290 kW against a q95 of
  264 kW, a 26 kW exceedance, and the day's q95 hit rate fell to 0.596. The
  ceiling held anyway — there were 203 kW of headroom under the target at that
  moment. That is what the margin is for, and showing the worst case buys more
  credibility than showing the best one.

## 10. Impact, computed

Every constant that turns a measured quantity into a rupee is named in
`eval/impact.py` and printed beside the number it produced.

**Tier 1 — technical.** Peak 41.4 kVA lower than uncontrolled (7.9%), breaches
11 → 0, capturing **99.5%** of the saving a perfect-foresight controller gets on
the same month.

**Tier 2 — operational, and the strongest one.** Sweep EV charging energy until
each controller breaks the ceiling: uncoordinated charging breaches at
**220 kWh/day**; ours holds to **500 kWh/day**. At 30% annual growth in EV
charging, that gap is **3.1 years of deferred connection upgrade**, worth
₹455,799 present-valued against a ₹1.77M upgrade.

**Tier 3 — financial.** ₹170,696 a month, 12.9% of the uncontrolled bill. The
split matters and we state it: **₹138,613 energy, ₹25,153 demand charge**, rest
duty. The energy share comes from the controller using its full comfort band
rather than holding a fixed setpoint — any competent scheduler would get it. The
demand-charge line is the smaller number and **it is the only one that needs the
forecast**: it is what the ablation moves.

**Tier 4 — scale, as arithmetic and labelled as such.** One building × 8
buildings per feeder × 5,000 feeders per discom. The point of the multiplication
is the shape of the number, not its precision, and the assumptions are printed
next to it. Anchor it in something real: distribution-transformer metering is
already mandated nationally under RDSS, so the measurement layer this needs is
being installed regardless of whether anyone builds the controller.

## 11. Demo

`docs/demo_script.md`. Ninety seconds, rehearsed. The new moment is the
**forecaster switch** on the model page: swap the optimiser's forecaster live and
watch the ceiling go. It is not an animation — each option replays the run that
produced the row in slide 6.

Fallback figures if the UI dies: `results/figures/`.

## 12. What is real and what is not

Read from `docs/limitations.md`, written before the results existed. Real: the
tariff, the bill engine, the meter data, the calibration, the physics. Not real:
the buildings are American, the source data is hourly upsampled to 15 minutes,
the base/HVAC split is inferred, the PV array is a design scenario.

End on portability: swap the tariff object, rerun. `tariff/orders/msedcl_2026.json`
is a second state, and `tests/test_bill.py` asserts the same code produces a
different answer — in Maharashtra midday is the *cheap* window, which inverts the
strategy.

---

## Open the model section with this, delivered at the person who gave the feedback

> Last time you asked where the model was. It was inside the constraint, and we
> did not show you. Here it is — and here is what happens when we take it out.

---

## Four findings worth volunteering before a judge finds them

1. **The tariff-aware rule-based baseline makes the bill worse on peak.** It
   pre-cools ahead of the peak window and its own pre-cooling sets a higher
   monthly demand (528.2 kVA vs 523.3 for doing nothing). Tariff-awareness
   without risk-awareness is actively harmful.
2. **Most of our rupees are not the demand charge.** ₹138,613 of the ₹170,696 is
   energy, from the comfort band. We say it before anyone checks the split.
3. **Cold start loses to seasonal naive on the held-out building.** Reported in
   full on slide 8, with what we would actually deploy on day one.
4. **Standard M&V would under-report our own saving by 3×.** With a 2-month
   baseline it reports ₹45,957 against a true ₹171,448, and its 95% band excludes
   the truth. Only at twelve months does the band cover it.

## Figures

| File | Use |
| --- | --- |
| `results/figures/06_model_frontier.png` | **slide 7** — model quality against rupees and breaches |
| `results/figures/07_interpretability.png` | slide 9 — SHAP and feature-group ablation |
| `results/figures/08_worst_case.png` | slide 9 — the evening we got wrong |
| `results/calibration_Fox_office_Gaylord.png` | slide 5, reliability diagram |
| `results/figures/03_heatwave_zoom.png` | **the money figure** — 14–19 June, breaches marked |
| `results/figures/01_normal_month.png` | slide 11 backup, normal month |
| `results/figures/04_comfort_frontier.png` | answer to "occupants will hate this" |
| `results/figures/05_mv_baseline_length.png` | answer to "how do I know it is real" |
