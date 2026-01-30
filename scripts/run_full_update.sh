#!/bin/bash
set -e  # Exit on error

echo "🔄 Starting full pipeline update..."

# Update current season data
echo "📥 Fetching current season fixtures..."
python src/ingestion/fetch_fixtures_current.py

# Build models locally
echo "🔨 Building dbt models..."
cd dbt/predict_may
dbt run
cd ../..

# Run simulation
echo "🎲 Running Monte Carlo simulation..."
python scripts/simulate_season.py

# Optional: sync to MotherDuck
read -p "Sync to MotherDuck? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python scripts/sync_to_motherduck.py
    cd dbt/predict_may
    dbt run --target motherduck
    cd ../..
fi

echo "✅ Pipeline update complete!"