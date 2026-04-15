#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# predict.sh — Weekly Super Lig gameday prediction
#
# Usage:
#   ./predict.sh             # runs with GAMEDAY set in predict_gameday.py
#   ./predict.sh --gameday 31
#   ./predict.sh --no-md     # skip Obsidian export
#
# What it does:
#   1. Refreshes referee assignments from Sofascore (fast, ~60s)
#   2. Runs the full prediction pipeline (DC + XGBoost + EV + O/U)
#   3. Saves a Markdown note to ~/Documents/Obsidian Vault/Predict May/GD{N}.md
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the project virtualenv so all dependencies are available
source venv/bin/activate

echo "═══════════════════════════════════════════════════════"
echo "  Predict May — weekly pipeline"
echo "  $(date '+%Y-%m-%d %H:%M')"
echo "═══════════════════════════════════════════════════════"

echo ""
echo "▶  Step 1/2 — Refreshing referee data from Sofascore..."
python3 scripts/fetch_referee_data.py

echo ""
echo "▶  Step 2/2 — Running prediction pipeline..."
python3 scripts/predict_gameday.py "$@"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done."
echo "═══════════════════════════════════════════════════════"
