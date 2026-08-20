#!/usr/bin/env bash
# Reproduce every number in results/ from scratch.
# Total runtime is roughly 50 minutes, dominated by the stress table.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"

echo "== 1/8  data      (BDG2 -> 15-min building files)"
$PY data/prepare.py

echo "== 2/8  forecast  (quantile LightGBM + adaptive conformal)"
$PY forecast/train.py

echo "== 3/8  calibrate (reliability diagrams, acceptance test)"
$PY forecast/calibrate.py

echo "== 4/8  tests     (bill to the rupee, RC vs closed form, controller invariants)"
$PY -m pytest tests/ -q

echo "== 5/8  target    (tightest holdable demand ceiling)"
$PY eval/find_target.py --building Fox_office_Gaylord --iters 7 --lo-frac 0.45

echo "== 6/8  table     (5 controllers x 4 stress scenarios)"
$PY eval/table.py --building Fox_office_Gaylord --stress none heatwave sensor_dropout outage

echo "== 7/8  M&V       (counterfactual bill, baseline-length study)"
$PY eval/mv.py --compare-baselines

echo "== 8/8  frontier  (comfort ceiling vs achievable demand target)"
$PY eval/frontier.py

echo "== figures"
$PY eval/figures.py

echo
echo "done. results/ now contains every number used in docs/pitch.md"
echo "run the demo with:  streamlit run ui/app.py"
