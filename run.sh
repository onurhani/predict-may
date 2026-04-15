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
#   7. Push predict-may repo       (keeps source repo in sync)
#   8. Sync + push onurhani.github.io (publishes live to onurhani.github.io/predict-may/)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the GitHub Pages repo that serves the live dashboard
PAGES_REPO="$(cd "$(dirname "$0")/../onurhani.github.io" 2>/dev/null && pwd || true)"
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

# Detect gameday from ml_predictions.json
GAMEDAY=$(python -c "
import json, pathlib
p = pathlib.Path('scripts/ml_predictions.json')
print(json.loads(p.read_text())['gameday'] if p.exists() else '??')
")

# ─── Step 7: Push predict-may repo ───────────────────────────────────────────
if [ "$PUSH" = true ]; then
    echo "▶  [7/8] Pushing predict-may repo..."
    git add docs/data/dashboard.json
    if git diff --cached --quiet; then
        echo "   No changes to predict-may — skipping."
    else
        git commit -m "data: GD${GAMEDAY} predictions + standings update"
        git push
        echo "   ✓ predict-may pushed."
    fi
else
    echo "▶  [7/8] Skipping predict-may push (--no-push)."
fi
echo ""

# ─── Step 8: Sync + push onurhani.github.io (live dashboard) ─────────────────
if [ "$PUSH" = true ]; then
    echo "▶  [8/8] Publishing live dashboard → onurhani.github.io/predict-may/..."
    if [ -z "$PAGES_REPO" ] || [ ! -d "$PAGES_REPO" ]; then
        echo "   ⚠️  onurhani.github.io repo not found at ../onurhani.github.io — skipping."
        echo "   Clone it next to this repo: git clone https://github.com/onurhani/onurhani.github.io.git ../onurhani.github.io"
    else
        cp docs/data/dashboard.json "$PAGES_REPO/predict-may/dashboard.json"
        (
            cd "$PAGES_REPO"
            git add predict-may/dashboard.json
            if git diff --cached --quiet; then
                echo "   No changes — live dashboard already up to date."
            else
                git commit -m "data: GD${GAMEDAY} predictions + standings update"
                git push
                echo "   ✓ Live dashboard published → onurhani.github.io/predict-may/"
            fi
        )
    fi
else
    echo "▶  [8/8] Skipping live publish (--no-push)."
    echo "   To publish manually:"
    echo "     cp docs/data/dashboard.json ../onurhani.github.io/predict-may/dashboard.json"
    echo "     cd ../onurhani.github.io && git add predict-may/dashboard.json && git commit -m 'data: GD${GAMEDAY} update' && git push"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done.                                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
