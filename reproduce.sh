#!/usr/bin/env bash
# Reproduce every number in results/ from scratch.
#
# The rule this file enforces: no number reaches a slide unless a script
# produced it. If a judge asks where a figure came from, the answer is a file
# path, and running this from a clean checkout regenerates all of them.
#
# Runtime is roughly 3 hours end to end. Stages 1-6 are the model and its
# evidence; 7-12 are the control results; P1-P6 are the cross-country study and
# the paper. Set ROLLING=0 to skip the walk-forward folds and PAPER=0 to skip
# the paper track, which are the two long poles.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"
B="${B:-Fox_office_Gaylord}"
ROLLING="${ROLLING:-1}"

step () { echo; echo "== $1"; }

step "1/12  data       (BDG2 -> 15-min building files)"
$PY data/prepare.py

step "2/12  tests      (leakage, scoring rules, bill to the rupee, RC vs closed form)"
$PY -m pytest tests/ -q

step "3/12  forecast   (quantile LightGBM + adaptive conformal, production artefacts)"
$PY forecast/train.py

step "4/12  calibrate  (reliability, PIT, coverage by horizon, acceptance test)"
$PY forecast/calibrate.py

step "5/12  benchmark  (six baselines + ours + the neural net, held-out and walk-forward)"
if [ "$ROLLING" = "1" ]; then
  $PY eval/forecast_eval.py --buildings "$B" --rolling \
      --rolling-baselines persistence seasonal_naive climatology \
                          linear_quantile lightgbm_quantile neural_quantile
else
  $PY eval/forecast_eval.py --buildings "$B"
fi

step "6/12  coldstart  (train on three buildings, evaluate on one never seen)"
$PY eval/cold_start.py

step "7/12  target     (tightest holdable demand ceiling)"
$PY eval/find_target.py --building "$B" --iters 7 --lo-frac 0.45

step "8/12  ABLATION   (same optimiser, eight different forecasters)"
$PY eval/ablation.py --building "$B"

step "9/12  frontier   (what forecast quality is worth in breaches and rupees)"
$PY eval/model_frontier.py --building "$B"

step "10/12 interpret  (SHAP, feature-group ablation, the worst evening)"
$PY eval/interpret.py --building "$B"

step "11/12 impact     (static-margin study, EV headroom sweep, four tiers)"
$PY eval/impact.py --building "$B"

step "12/12 table      (5 controllers x 4 stress scenarios, M&V, comfort frontier)"
$PY eval/table.py --building "$B" --stress none heatwave sensor_dropout outage
$PY eval/mv.py --compare-baselines
$PY eval/frontier.py

step "cards and figures"
$PY eval/model_cards.py --buildings "$B"
$PY eval/figures.py

step "web bundle"
$PY eval/export_web.py || echo "  (export_web skipped)"

# ---------------------------------------------------------------------------
# The paper track. Separate from the twelve stages above because it is a
# different question -- those ask whether the controller works on one building,
# these ask whether anything about it survives leaving that building. Set
# PAPER=0 to skip; the cross-country stage is the long pole at roughly an hour
# and needs network on its first run to fetch the public archives.
# ---------------------------------------------------------------------------
if [ "${PAPER:-1}" = "1" ]; then

  step "P1/6  sources    (Delhi SLDC, ENTSO-E via OPSD, Chinese provincial, ERA5)"
  $PY data/national.py

  step "P2/6  study      (one protocol, thirty supplies, eight countries)"
  # One output file per arm, which is what the report loader unions. The
  # national arm is split by provenance rather than by arm, because the six
  # Chinese provinces are `arm=national` but must never be pooled with the
  # metered series -- see eval/china_audit.py for why.
  $PY eval/comparative.py --arms climate     --out comparative_climate
  $PY eval/comparative.py --arms office      --out comparative_office
  $PY eval/comparative.py --arms demographic --out comparative_demographic
  $PY eval/comparative.py --arms national    --out comparative_national \
      --only IN_Delhi GB_UKM IE DE FR ES
  $PY eval/comparative.py --arms national    --out comparative_china \
      --only CN_Hainan CN_Guangdong CN_Shanghai CN_Yunnan CN_Beijing CN_Heilongjiang
  $PY eval/comparative_report.py

  step "P3/6  provenance (audit the China arm against its own data)"
  $PY eval/china_audit.py

  step "P4/6  horizon    (the marginal-vs-joint bracket, copula, closed loop)"
  $PY eval/horizon_risk.py --building "$B"

  step "P5/6  conformal  (split robustness, walk-forward year, frozen shift)"
  $PY eval/conformal_audit.py --building "$B"

  step "P6/6  paper      (tables, figures, Overleaf bundle)"
  $PY eval/paper_tables.py
  $PY eval/paper_figures.py
  $PY eval/paper_bundle.py

fi

echo
echo "done. results/ now contains every number used in docs/pitch.md and the deck,"
echo "and ../paper/ contains the paper with every table and figure regenerated."
echo "run the demo with:  streamlit run ui/app.py       (or the Next.js app in ../ampcast)"
