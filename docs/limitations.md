# What is real and what is not

Written before the results, so it cannot be tuned to flatter them.

## Real

- **The tariff.** Rates, time-of-day windows, the 30-minute demand block, the
  90%-of-contract billing floor and the power-factor rule are hand-encoded from a
  published TNERC order and cross-checked by a text parser that reproduces the
  same JSON (`tariff/compiler.py`). A second state (MSEDCL) is encoded to
  demonstrate portability.
- **The bill engine.** Matched to the rupee against a hand computation for a full
  week (`tests/test_bill.py`). Every number in every table and on every slide
  comes out of it.
- **The meter data.** Building Data Genome Project 2, four real buildings, two
  years, from the same site. Buildings were selected by a stated rule, not by
  which ones looked good.
- **The forecaster and its calibration.** Trained on 2016-01 to 2017-03,
  calibrated on 2017-04 to 2017-05, evaluated on 2017-06. Coverage is measured on
  data the model never saw, including per horizon.
- **The physics.** The RC model is checked against the analytic exponential decay
  and against a steady-state energy balance (`tests/test_sim.py`). Envelope
  conductance is fitted from each building's own meter.

## Not real, and stated as such

- **The buildings are American.** Site "Fox" is Tempe, Arizona: the hottest site
  in BDG2 (mean 25.1 °C, peak 48.3 °C) and at 33.4° N, close to northern India's
  solar geometry. It is a climate analogue, not India. The load *shapes* are
  American too, and this matters in a specific direction: these buildings peak
  around 14:00, inside the tariff's normal window, not inside the 18:00-22:00
  peak window. So the time-of-day arbitrage available here is *weaker* than it
  would be for an Indian office, and essentially all of our saving comes from the
  demand charge. That is the lever the project argues for, so the result is
  conservative rather than flattering, but the reader should know the ToD column
  is understated.
- **Hourly source data upsampled to 15 minutes.** BDG2 is hourly; Indian demand
  charges are set on 30-minute blocks. We interpolate with a shape-preserving
  PCHIP. Interpolation *smooths* real sub-hourly variability, so measured peaks
  are understated and the demand-charge exposure we optimise against is, if
  anything, milder than reality.
- **The split of the meter into base load and HVAC** is a changepoint regression
  on outdoor temperature, not a submetered measurement. Buildings on district
  chilled water were excluded precisely because their electricity meter contains
  no cooling to control, but for the buildings we kept, the split is inferred.
- **Rooftop PV is synthetic.** There is no solar meter at this site. The PV array
  is a design scenario (150 kWp by default, settable to zero) from a clear-sky
  model. Its *uncertainty* is not invented: the forecast quantiles come from the
  site's own measured cloud-cover record.
- **Thermal capacitance is assumed**, at 110 kJ/m²K (ISO 13790 "medium"). The
  conservative end of the range: less thermal storage to exploit means a smaller
  claimed saving. Envelope conductance is fitted, but a fitted value outside
  0.5-5.0 W/m²K would be replaced by a hand-picked 2.0 and flagged in the
  manifest. No building in the final set needed that fallback.
- **The weather forecast is assumed perfect** unless `--weather-noise-c` is set.
  This flatters the forecaster. It does not flatter the *controller*, whose
  robustness claim rests on base-load uncertainty and is tested directly by the
  stress cases.
- **A single building, simulated.** No hardware, no live BMS, one month, one
  tariff, one climate.

## The honest shape of the claim

We do not claim a validated saving for any real building. We claim that on real
meter data, under a real published tariff, with a calibrated forecast and a
deterministic optimiser, substituting the 95th percentile into the demand-ceiling
constraint holds a demand target that the same controller on mean forecasts
breaches — and that the resulting difference in the bill is dominated by the
demand charge, not by energy.
