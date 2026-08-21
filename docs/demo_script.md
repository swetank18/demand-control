# Demo script — two minutes, rehearsed word for word

Two screens, in this order: the **model page** (`/model`) for the ablation, then
**world sim** (`/worldsim`) for the stress scenarios. The first answers "where is
the model"; the second answers "what happens when it is wrong". Do not reverse
them — round 1 failed by showing the second one only.

Have both tabs open before you start. World sim on **Normal month**, replay
slider at the far right, `No control` / `MPC on mean forecast` / `Ours` selected.

---

## Part one — the model page, 40 seconds

**0:00 — where the model sits.** Point at the four-column chain at the top.
> "Load lags, temperature, the weather forecast, the calendar. Out comes a 95th
> percentile of the next sixteen hours. That number goes into one constraint in
> the optimiser, and everything downstream is decided by it. Last time you asked
> where the model was — it was in there, and we did not show you."

**0:12 — the switch.** The forecaster switcher is on `LightGBM quantile (ours)`.
Zero breaches on screen.
> "Same optimiser, same building, same June, same tariff, same seed. Watch the
> bottom panel: that is the forecast band the controller was planning on, and the
> white line is the load that actually arrived, staying inside it."

**0:25 — take it out.** Click **Persistence**.

**Say nothing for three seconds.** Let the eighty-seven breach markers land.

> "Same everything. Only the forecaster changed. Eighty-seven breaches, peak up
> 75 kVA, and — look — it claimed *more* headroom than we did while doing it.
> Headroom without calibration is an overdraft."

Click **Static margin**: eleven breaches.
> "And that is what the site does today with no model at all."

**0:38 — the number that answers 'why forecast'.**
> "Tuned to be exactly as safe as we are, a fixed margin leaves 198 kW usable. We
> leave 354. The forecast recovers 156 kilowatts of capacity at equal safety.
> That is what the model is worth, in the units the problem is about."

Switch tabs.

## Part two — world sim, 80 seconds

**0:00 — the ceiling.**
> "Red line is the demand ceiling: 467 kW. It is not a target we invented — it is
> the tightest number this building can hold for a whole month inside the
> operator's comfort budget. Grey is the building as it runs today. Eleven blocks
> above the line."

Drag the replay slider left to 1 June, then release to the right and let it run.

**0:20 — press Heatwave.**
> "Now a heatwave. Six degrees hotter, base load up 25%, three days — conditions
> that are not in the forecaster's training data."

**0:30 — the mean-forecast controller.**
Point at the orange trace crossing the red line.
> "This is the same optimiser, same tariff, same comfort budget. The only
> difference is that it plans on the *average* forecast."

**Say nothing for three seconds.** Let the triangles above the line sit there.

> "Four breaches. Peak 502 kVA. Each one of those sets the demand charge for
> all thirty days."

**0:50 — ours, same day.**
Point at the dark red trace hugging the line.
> "Ours plans the same load on the 95th percentile instead. Zero breaches, 482
> kVA — and within six hundred rupees of a controller that was *given the actual
> future*. Same comfort: 0.66% versus 0.69%."

**1:10 — press Grid outage.**
> "Two-hour outage, critical load only. Comfort holds, the controller rides
> through on thermal mass, and it does not panic-recover into a new peak
> afterwards — which is the mistake that would cost the month."

**1:20 — cut to the table.**
> "Every number here comes out of the bill engine, which is matched to the rupee
> against a hand-computed week. Bottom row is a whole-month perfect-foresight
> optimum, so 81% means 81% of what was actually available — not 81% better than
> a strawman."

---

**The point being made** is not that the model is accurate. It is that the model
is load bearing — part one — and that the system behaves when the model is wrong
— part two. If you only have sixty seconds, cut part two, not part one.

## If something breaks

- The screen recording made at hour 33 is in `results/`. Switch to it and keep talking.
- If the UI will not start: `results/results_Fox_office_Gaylord.md` and
  `results/ablation.md` have every table, and `results/figures/` has the plots.
- If only the ablation replay is broken, the table on the same page is the same
  eight runs. Read the breach column out loud instead of clicking.
- Do not debug live. The talk is the deliverable.
