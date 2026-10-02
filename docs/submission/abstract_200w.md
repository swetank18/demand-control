# Submission abstract (199 words)

For the journal's abstract field: Applied Energy caps it (sources give 200 or
250 words; this fits either). Plain text, no markup, so it survives a paste into
a web form. It is the abstract of the Elsevier build
(`docs/paper_elsevier/main.tex`); every number is one the study emits and
`eval/paper_numbers.py` or the tables check. Rewritten 2026-10-02 to cover the
campus closed loop, the matched-pair result and the retracted aggregation
finding, none of which the 25 September version had.

---

Indian commercial demand charges are set by the worst thirty-minute block in a
month, so defending a demand ceiling is a risk problem. The usual response
substitutes a high forecast quantile into the capacity constraint and calls it
a 95% guarantee. We measure what it delivers on fourteen held-out windows from
an American office and an Indian campus. Per step the bound holds, at
0.032-0.079 against a nominal 0.05; over the controller's 16-hour window the
probability of breaching somewhere is 0.39-0.82, median 0.57, eight to sixteen
times the level promised. The independence correction overstates it and is not
a bound; the valid union bound is 1 on every window. A Student-t copula
predicts the realised value to a mean absolute error of 0.065, and a scenario
MILP built on it makes the violation level an operator setting. On two plants
the response is monotone and, on matched commitments, resolved within one
billing month; it is close to exact only where the per-step layer beneath it
is calibrated. Over thirty-seven supplies in eight countries,
distribution-free coverage does not degrade with aggregation, a reading this
study first made, then traced to the adaptive conformal step being stated in
the series' units.

---

**199 words.**
