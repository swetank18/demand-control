# Demo script — ninety seconds, rehearsed word for word

Start with the UI already open on **Normal month**, replay slider at the far
right, `No control` / `MPC on mean forecast` / `Ours` selected.

---

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

**The point being made** is not that the model is accurate. It is that the system
behaves when the model is wrong.

## If something breaks

- The screen recording made at hour 33 is in `results/`. Switch to it and keep talking.
- If the UI will not start: `results/results_Fox_office_Gaylord.md` has every table.
- Do not debug live. The talk is the deliverable.
