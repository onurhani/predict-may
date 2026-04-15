#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Full weekly pipeline for Predict May
#
# Usage:
#   ./run.sh                  # full run for GAMEDAY set in predict_gameday.py
#   ./run.sh --gameday 31     # override gameday (pass-through to prediction script)
#   ./run.sh --no-push        # skip git commit + push at the end
#   ./run.sh --no-md          # skip Obsidian markdown export
#
# Pipeline:
#   1. Fetch latest match results  (football-data.co.uk → DuckDB)
#   2. Rebuild dbt models          (standings, predictions, SPI)
#   3. Refresh referee data        (Sofascore → data/referee_stats.json)
#   4. Run ML prediction pipeline  (DC + XGBoost + Referee bias → ml_predictions.json + Obsidian MD)
#   5. Monte Carlo season sim      (10k runs, using ML probs for this week → season_projections)
#   6. Export dashboard            (combines everything → docs/data/dashboard.json)
#   7. Git commit + push           (publishes to GitHub Pages)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate project virtualenv
source venv/bin/activate

# ─── Parse args ──────────────────────────────────────────────────────────────
PUSH=true
PREDICT_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        *)         PREDICT_ARGS+=("$arg") ;;
    esac
done

# ─── Header ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         Predict May — full weekly pipeline           ║"
echo "║         $(date '+%Y-%m-%d %H:%M')                         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Fetch current match results ─────────────────────────────────────
echo "▶  [1/6] Fetching latest match results..."
python src/ingestion/fetch_fixtures_current.py
echo ""

# ─── Step 2: Rebuild dbt models ──────────────────────────────────────────────
echo "▶  [2/6] Rebuilding dbt models..."
(cd dbt/predict_may && dbt run --quiet)
echo "   dbt models up to date."
echo ""

# ─── Step 3: Refresh referee data from Sofascore ─────────────────────────────
echo "▶  [3/6] Refreshing referee data from Sofascore..."
python scripts/fetch_referee_data.py
echo ""

# ─── Step 4: Run ML prediction pipeline ──────────────────────────────────────
echo "▶  [4/6] Running ML prediction pipeline (DC + XGBoost + Referee bias)..."
python scripts/predict_gameday.py "${PREDICT_ARGS[@]}"
echo ""

# ─── Step 5: Monte Carlo season simulation ───────────────────────────────────
echo "▶  [5/6] Running Monte Carlo season simulation (10,000 runs)..."
python scripts/simulate_season.py
echo ""

# ─── Step 6: Export dashboard JSON ───────────────────────────────────────────
echo "▶  [6/6] Exporting dashboard JSON..."
python scripts/export_dashboard.py
echo ""

# ─── Step 7: Git commit + push ───────────────────────────────────────────────
if [ "$PUSH" = true ]; then
    echo "▶  Publishing dashboard to GitHub Pages..."
    GAMEDAY=$(python -c "
import json, pathlib
p = pathlib.Path('scripts/ml_predictions.json')
print(json.loads(p.read_text())['gameday'] if p.exists() else '??')
")
    git add docs/data/dashboard.json
    if git diff --cached --quiet; then
        echo "   No changes to dashboard.json — nothing to push."
    else
        git commit -m "data: GD${GAMEDAY} predictions + standings update"
        git push
        echo "   ✓ Dashboard published."
    fi
else
    echo "   Skipping git push (--no-push)."
    echo "   To publish manually:"
    echo "     git add docs/data/dashboard.json"
    echo "     git commit -m 'data: matchday update'"
    echo "     git push"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done.                                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
