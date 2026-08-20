# Tariff-Native Demand Control for Smart Buildings

**We do not forecast electricity. We forecast the bill, and we buy it down inside a comfort budget the operator sets.**

An Indian commercial building's electricity bill has three levers, and almost
every student project optimises only the first:

1. **Energy charge** — ₹/kWh, and since April 2024 that rate depends on which
   time-of-day window you consume in. Peak is at least 1.2× normal for
   commercial and industrial consumers; solar hours at least 20% below.
2. **Demand charge** — ₹ per kVA of the single highest 30-minute block in the
   *entire month*. In Tamil Nadu this is roughly ₹608/kVA/month. One bad Tuesday
   afternoon sets the charge for all thirty days.
3. **Power factor** — penalties and rebates on reactive power, almost never
   touched by software projects.

A kWh optimiser sees lever one. The money is spread across all three, and lever
two is a *risk* problem rather than a prediction problem. That is the gap this
builds into.

## The one idea

In the constraint that holds grid import under the monthly demand ceiling, we
substitute the **95th percentile** of the base-load forecast and the **5th
percentile** of solar, instead of the means:

```
base_q95[t] + controllable[t] - solar_q05[t]  <=  D_target
```

That is a chance constraint implemented by quantile substitution. It is a few
characters of code. It is also the entire reason the controller does not blow
the monthly demand charge — and it is why the calibration work upstream is load
bearing rather than decorative. If q95 is not really a 95th percentile, the
guarantee is theatre. So we measure it: see `results/calibration_*.png`.

Where risk goes and where expectation goes:

| Quantity | Planned on | Why |
| --- | --- | --- |
| Demand ceiling | **q95** | breaching costs a full month of demand charge |
| Energy cost | q50 | costing energy at q95 would overstate the bill and distort the trade-off |
| Thermal comfort | q50, with slack | comfort is soft by design; the operator sets the budget and violations are reported, not hidden |

## Two hard rules

- **No language model touches a money decision.** The optimiser is a
  deterministic MILP. The tariff parser is regex + a human confirmation step
  that refuses to emit an unreviewed tariff.
- **The bill engine is the single source of truth.** Every number in every table
  and on every slide comes out of `tariff/bill.py`, which is matched to the
  rupee against a hand computation in `tests/test_bill.py`.

## Architecture

```
                    tariff order text
                           |
                    [tariff compiler]        deterministic, human-confirmed
                           |
                    tariff object (JSON)
                           |
   BDG2 meter data -> [quantile forecaster] -> q05..q95 per interval
                           |                   (LightGBM + adaptive conformal)
                           v
                 [chance-constrained MILP]  <-- comfort budget, demand target
                           |
                    setpoints, schedules
                           |
                      [RC simulator]  <-- stress injector
                           |
                    realised power series
                           |
                     [bill engine] ---> counterfactual bill
                           |
                        [replay UI]
```

## Quickstart

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install lightgbm highspy scikit-learn
cd demand-control

python data/prepare.py                 # BDG2 -> 15-min building files  (~2 s)
python forecast/train.py               # quantile LightGBM + calibration (~4 min)
python forecast/calibrate.py           # reliability diagrams -> results/
python eval/find_target.py             # tightest holdable demand target
python eval/table.py --stress none heatwave sensor_dropout outage
streamlit run ui/app.py
```

Raw BDG2 files are not vendored. To fetch them:

```bash
B=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data
curl -sL -o data/raw/metadata.csv $B/metadata/metadata.csv
curl -sL -o data/raw/weather.csv $B/weather/weather.csv
curl -sL -o data/raw/electricity_cleaned.csv $B/meters/cleaned/electricity_cleaned.csv
```

## Why this shape and not the obvious shape

- **Reinforcement learning loses here.** Across CityLearn 2021, 2022 and 2023 at
  NeurIPS, none of the top-performing teams used RL for scheduling. Winning
  solutions used classical optimisation and heuristics, for want of a dynamic
  simulation environment rich enough to train on. A hackathon has even less of
  that than the competition did.
- **Hierarchical forecast + MPC wins.** The CityLearn 2023 second-place
  framework used LightGBM/kNN/linear forecasting, building-level MPC, rule-based
  control for hot water, and physics-based MPC for battery dispatch. Boring
  parts, careful composition.
- **Novel use cases beat novel models when judges own buildings.** In the
  NYSERDA RTEM hackathon a team won by finding pairs of nearby buildings with
  complementary thermal loads and costing out the transfer. No ML at all.
- **Resilience is now scored.** CityLearn 2023 explicitly evaluated agents under
  power outages alongside normal operation, which is why there is an outage
  button.

We take the winning architecture, add the thing none of them had (a real
tariff), and make the demo about what happens when the model is wrong.

## Data

Building Data Genome Project 2 — 1,636 real buildings, two years of hourly
meter data plus weather. Site **Fox** (Tempe, Arizona) was chosen as the hottest
site in the set (mean 25.1 °C, peak 48.3 °C) and, at 33.4° N, a reasonable
analogue for northern India's solar geometry.

Buildings were selected by a stated rule, not by which ones looked good:
no district chilled-water meter (so the electricity meter actually contains the
cooling we intend to control), >97% coverage, median load 40–800 kW, and a
positive load-vs-temperature correlation.

| Building | Role | Shape |
| --- | --- | --- |
| `Fox_office_Gaylord` | primary | peaky office, p99/median 2.9, weekday peak 14:00 |
| `Fox_assembly_Dixie` | campus | event-driven, p99/median 4.4, night/day 0.21 |
| `Fox_education_Etta` | campus | flat round-the-clock, strongest weather response |
| `Fox_public_Denny` | campus | runs late, night/day 0.78 |

## Repo layout

```
tariff/    schema, deterministic compiler, bill engine, encoded orders (TN, MH)
forecast/  features, quantile LightGBM + adaptive conformal, calibration report
control/   chance-constrained MILP, rule-based and thermostat baselines
sim/       RC thermal model, water heater, EV, battery, PV, stress injector
eval/      closed-loop runner, demand-target search, results table
ui/        one-screen Streamlit replay
tests/     bill to the rupee, RC against closed form, controller invariants
docs/      limitations, written before the results
```

## Honesty

`docs/limitations.md` was written before the results were generated. Read it
before quoting any number from this repo. The short version: the tariff, the
bill engine, the meter data, the calibration and the physics are real; the
buildings are American, the source data is hourly upsampled to 15 minutes, the
meter's split into base load and HVAC is inferred rather than submetered, and
the PV array is a design scenario.
