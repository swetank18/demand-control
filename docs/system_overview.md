# Aethergrid — what the dashboard shows and what is underneath it

A single document for someone who has five minutes and wants to know what this
system is, what is actually learned, and where every number on the screen came
from. Written to be read top to bottom; the file map at the end is for when you
want to go and look.

---

## 1. The one-sentence version

An Indian commercial building pays for the single highest 30-minute block of the
month at roughly ₹608 per kVA, so a quarter of the bill is decided by one bad
afternoon. We forecast the *distribution* of the building's uncontrollable load,
put its 95th percentile inside the capacity constraint of a mixed-integer
optimiser, and let that optimiser schedule cooling, hot water and EV charging so
the ceiling holds — then we prove the forecast is load bearing by taking it out
and measuring what breaks.

---

## 2. Two repositories

| Repo | What it is |
| --- | --- |
| `demand-control` | Python. The tariff engine, the forecaster, the optimiser, the simulator, the evaluation suite. This is the source of every number. |
| `ampcast` | Next.js 16 + React 19. The dashboard, deployed on Vercel as **aethergrid**. It renders numbers; it never computes money. |

The join between them is one script: `eval/export_web.py` reads `results/` and
writes `ampcast/src/lib/bundle.json` plus `ampcast/public/data/**`. That is
deliberate — if the deployed page and the paper ever disagreed, the pitch would
be dead, so the page is a viewer over the study's own output files.

---

## 3. The dashboard

Three pages, in the order a judge should see them.

### `/model` — the evidence page

Answers "where is the model", which is the question round 1 could not answer.

- **Where the model sits.** A four-column chain: inputs → model → the constraint
  the output enters → the outcome in rupees. Plus the exact split dates.
- **Swap the forecaster.** The centrepiece. Eight forecasters, each one a month
  the Python side already ran with the optimiser, the tariff, the physics and the
  seed held fixed. Picking one replays that month: the top panel is grid import
  against the ceiling with a marker on every breach, the bottom panel is the
  q05–q95 forecast band the controller was planning on with the load that
  actually arrived drawn through it. Our run stays on screen as a ghost so the
  divergence is visible rather than remembered. Transport controls: restart,
  play/pause, 0.5×–4× speed, and a scrub bar. Deep-linkable — `/model#f=persistence`
  opens straight into the failure case.
- **Live calibration readouts.** Rolling 24-hour q95 hit rate and 90% coverage at
  the replay head, so calibration is a property of the run being played rather
  than a diagram in an appendix.
- **Take the model out.** The ablation table, toggleable between the normal month
  and the heatwave.
- **Why not a fixed margin**, **what the quality is worth**, the **benchmark**,
  **walk-forward spread**, **cold start**, **the evening we got wrong**, **SHAP
  and feature-group ablation**, **impact**, and the generated **model card**.

### `/worldsim` — the control instrument

Five controllers across four scenarios (normal, heatwave, sensor dropout, grid
outage). Three stacked panels: grid import against the ceiling, the billed-demand
ratchet that only ever climbs, and indoor temperature inside its comfort band.
Plus a breach ledger that prices every block over the line, and the results table.

### `/method` — how it was done

The one substitution, the calibration evidence and its reliability diagram, where
the demand ceiling came from, the comfort/savings frontier, the measurement-and-
verification study, and an explicit list of what is real and what is not.

### How data reaches the page

```
results/*.json ──► eval/export_web.py ──► ampcast/src/lib/bundle.json      (metrics, tables, model evidence)
                                     └──► ampcast/public/data/series/*.json (24 controller runs)
                                     └──► ampcast/public/data/ablation/*.json (8 forecaster runs + bands)
                                                    │
                        scripts/seed.ts ──► Neon Postgres ──► /api/series/[id]
                                                    │
                                              Next.js (RSC) ──► the page
```

Reads prefer Neon and fall back to the bundled export field by field, so a
database outage degrades to stale-but-correct numbers instead of a blank page.
The eight ablation months (1.5 MB total) are prefetched in the background after
load, which is why switching forecaster on stage is instant.

**A note on the replay's performance**, because it was a real bug: every path is
built once and revealed by animating a clip rectangle. Rebuilding a dozen
thousand-point SVG paths on every animation frame — the obvious implementation —
stutters badly enough that it reads as the page reloading.

---

## 4. What is actually learned

Four learned components. The first is the one the argument rests on.

### M1 · Base-load quantile forecaster — the model

| | |
| --- | --- |
| Target | total uncontrollable load, 15-minute resolution |
| Horizon | 64 steps = 16 hours, direct multi-horizon (horizon index is a feature) |
| Output | quantiles 0.05, 0.25, 0.50, 0.75, 0.95 — one LightGBM booster per quantile, `objective='quantile'` |
| Features | 30, all knowable at the forecast origin: lags at 1/2/4/8/96/192/672 steps, rolling mean and SD over 1 h / 1 day / 1 week, same-time-of-day mean over four weeks, outdoor temperature and its 1-hour change, horizon index, calendar (time of day, weekday, weekend, holiday, day of year), solar elevation |
| Split | train 2016-01-01 → 2017-03-31, calibrate 2017-04-01 → 2017-05-31, test once on June 2017. Temporal, never random; `tests/test_leakage.py` asserts it |
| Held-out | pinball 2.397, MAE 7.03 kW, 90% coverage 0.891 on 184,320 (origin, horizon) pairs |

**Calibration is two stages and it is the safety property.** Split conformal
shifts each quantile by the empirical residual quantile per horizon; adaptive
conformal (ACI) then moves that offset online against the realised exceedance
rate. This is not decoration: split conformal fitted on April–May *undercovers*
June at 0.832, below the 0.85 acceptance floor we set in advance, because June is
hotter. ACI restores 0.890 for 6.4 kW of extra interval width.

**Where the output goes.** `base_q95` is a term in a hard constraint, not a chart:

```
base_q95[t]  +  controllable[t]  −  solar_q05[t]   ≤   D_ceiling
```

### M2 · Per-flat thermal parameters — learned physics

Envelope conductance is fitted per building from its own meter by changepoint
regression against outdoor temperature; capacitance is ISO 13790 "medium" and is
declared as an assumption. This is what lets comfort constraints be calibrated
per building rather than asserted.

### M3 · Solar quantiles

PV output quantiles derived from the site's own measured cloud record. The array
itself is a design scenario; its *uncertainty* is not invented.

### M4 · The benchmark family

Six alternatives are trained and scored through the identical calibration layer
so the comparison is of forecasts, not of plumbing: persistence, seasonal naive,
climatology, linear quantile regression, a residual-MLP neural quantile network
with a pinball head (PyTorch), and a constant "static margin" that stands in for
what a distribution engineer does today. Perfect foresight is the bound.

---

## 5. What is not learned, and deliberately so

The optimiser is a **deterministic mixed-integer linear program**, solved with
HiGHS via `scipy.optimize.milp`, 64-step horizon at 15-minute resolution,
re-solved every step and only the first step applied. It decides cooling power,
water-heater on/off, EV charging and battery dispatch against the demand ceiling,
the tariff's time-of-day windows and a comfort band the *operator* sets. Mean
solve time is 4–8 ms.

Two rules that are worth saying out loud:

- **No language model touches a money decision.** The tariff parser is regex plus
  a human confirmation step that refuses to emit an unreviewed tariff.
- **The bill engine is the single source of truth.** Every rupee on every screen
  comes from `tariff/bill.py`, which is matched to the rupee against a hand
  computation for a full week in `tests/test_bill.py`.

Reinforcement learning was considered and rejected on evidence: across CityLearn
2021–2023 at NeurIPS, none of the top-performing teams used RL for scheduling.

---

## 6. The evidence layer

Every claim has a script, and every script writes a file that the deck quotes.

| Script | Produces | The claim it serves |
| --- | --- | --- |
| `eval/forecast_eval.py` | benchmark table, 8 walk-forward folds | there is a trained model and it is good |
| `forecast/calibrate.py` | reliability, PIT, coverage by horizon | the interval means what it says |
| `eval/ablation.py` | **the ablation**, per-row timelines and bands | the model is necessary |
| `eval/model_frontier.py` | monotonicity and the exchange rate | model quality converts into money |
| `eval/impact.py` | static-margin study, EV sweep, four tiers | what it is worth, with assumptions printed |
| `eval/cold_start.py` | a building held out of training entirely | does it work on a site you have not seen |
| `eval/interpret.py` | SHAP, feature-group retrain, the worst evening | it is not a black box |
| `eval/model_cards.py` | a model card per model, from the artefacts | the card cannot drift from the model |
| `eval/table.py`, `eval/mv.py`, `eval/frontier.py` | controllers × scenarios, M&V, comfort frontier | the control result itself |

Headline numbers, all from those files:

- Replace the forecaster with a constant: **+11 ceiling breaches, ₹12,885, −113 kW** of usable headroom, nothing else changed.
- A forecast-free static margin as safe as we are leaves 198 kW usable; we leave 354 kW. The forecast recovers **156 kW, +79% capacity at equal safety**.
- Across the family of forecasters: **₹2,228 per unit of pinball loss per month** (R² 0.95) and **3.5 breaches per unit** (R² 0.89).
- Uncoordinated EV charging breaches at 220 kWh/day; ours holds to 500 → **3.1 years of deferred upgrade**.

And the results that go against us, which are on the page too: cold start loses
to seasonal naive on an unseen building; seasonal naive already reaches zero
breaches on a calm month and rides out the heatwave better than we do because its
intervals are wider; and most of the rupees on this building are energy from the
comfort band, not the demand charge the model defends.

---

## 7. Reproducing it

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r demand-control/requirements.txt
cd demand-control && ./reproduce.sh          # ~2 h; ROLLING=0 halves it
```

`reproduce.sh` runs twelve stages in dependency order and regenerates every file
in `results/`. The rule it enforces: no number reaches a slide unless a script
produced it. If a judge asks where a figure came from, the answer is a file path.

The dashboard:

```bash
cd ampcast && npm install && npm run dev     # reads the bundle; Neon optional
```

---

## 8. What is real and what is not

Real: the tariff (hand-encoded from a published TNERC order and cross-checked by
a parser), the bill engine, the meter data (Building Data Genome Project 2, four
real buildings, two years), the calibration, and the physics.

Not real, and stated rather than discovered: the buildings are American (Tempe,
Arizona — a climate analogue, not India); the source data is hourly upsampled to
15 minutes, which smooths peaks; the split of the meter into base load and HVAC
is inferred by changepoint regression rather than submetered; the rooftop PV is a
design scenario; the weather at the target time is used as a *perfect* forecast,
which flatters the model by roughly the 34% that the feature-group study
attributes to the weather feed.

`docs/limitations.md` was written before the results existed. Read it before
quoting any number from this repo.

---

## 9. File map

```
demand-control/
  tariff/     schema, deterministic compiler, bill engine, encoded orders (TN, MH)
  forecast/   features, quantile LightGBM, conformal + adaptive conformal, baselines
  control/    the chance-constrained MILP, rule-based and thermostat baselines
  sim/        RC thermal model, water heater, EV, battery, PV, stress injector
  eval/       benchmark, ablation, frontier, cold start, interpretability, impact,
              model cards, figures, web export
  models/     trained artefacts, forecast tensors, generated model cards
  results/    every number in the deck, and the figures made from them
  docs/       pitch, demo script, judge questions, limitations, results summary
  tests/      leakage, scoring rules, bill to the rupee, RC against closed form

ampcast/
  src/app/model      the evidence page
  src/app/worldsim   the control instrument
  src/app/method     how it was done
  src/components/    ForecastSwitch (the replay), StripChart, ModelCharts, MethodCharts
  src/lib/           bundle.json (exported), data access, Neon client, types
  public/data/       series and ablation runs, served statically
  scripts/           schema.sql and the Neon seeder
```
