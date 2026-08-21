#!/usr/bin/env bash
# Reproduce every number in results/ from scratch.
#
# The rule this file enforces: no number reaches a slide unless a script
# produced it. If a judge asks where a figure came from, the answer is a file
# path, and running this from a clean checkout regenerates all of them.
#
# Runtime is roughly 2 hours end to end. Stages 1-6 are the model and its
# evidence; 7-12 are the control results that were already there. Set
# ROLLING=0 to skip the walk-forward folds, which are half the total.
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

echo
echo "done. results/ now contains every number used in docs/pitch.md and the deck."
echo "run the demo with:  streamlit run ui/app.py       (or the Next.js app in ../ampcast)"
