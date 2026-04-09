#!/bin/bash
set -e

echo "🔄 Starting full pipeline update..."

# Fetch current data from football-data.co.uk
echo "📥 Fetching match data..."
python src/ingestion/fetch_fixtures_current.py

# Rebuild dbt models
echo "🔨 Building dbt models..."
cd dbt/predict_may && dbt run && cd ../..

# Run Monte Carlo simulation
echo "🎲 Running simulation..."
python scripts/simulate_season.py

# Export dashboard JSON
echo "📊 Exporting dashboard data..."
python scripts/export_dashboard.py

echo "✅ Done. Commit docs/data/dashboard.json to update the dashboard."
echo ""
echo "   git add docs/data/dashboard.json"
echo "   git commit -m 'data: matchday XX update'"
echo "   git push"
