#!/usr/bin/env bash
# Refresh ../ieee_paper_status.md every 5 minutes while reproduce_india.sh runs, then once more.
cd "$(dirname "$0")/.."
while pgrep -f "reproduce_india\.sh" >/dev/null; do
  ../.venv/bin/python eval/status_md.py >/dev/null 2>&1
  sleep 300
done
../.venv/bin/python eval/status_md.py >/dev/null 2>&1
