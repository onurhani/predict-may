#!/bin/bash
# Activate the predict_may environment

source venv/bin/activate

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "✅ predict_may environment activated!"
echo ""
echo "📊 Available commands:"
echo "  jupyter lab          - Launch Jupyter for analysis"
echo "  code .              - Open project in VS Code"
echo "  dbt run             - Run dbt models"
echo "  dbt test            - Test dbt models"
echo "  python src/ingestion/api_football_fixtures.py - Ingest new data"
echo ""
echo "📁 Project structure:"
echo "  data/               - DuckDB database"
echo "  notebooks/          - Jupyter notebooks"
echo "  dbt/predict_may/    - dbt models"
echo "  src/ingestion/      - Data ingestion scripts"
