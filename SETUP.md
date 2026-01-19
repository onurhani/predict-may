# Predict May - Setup Guide

## Quick Start

### 1. Activate Environment
```bash
cd ~/git/predict_may
source activate.sh
```

### 2. Run Data Ingestion (if needed)
```bash
# Make sure you have API_FOOTBALL_KEY in .env
python src/ingestion/api_football_fixtures.py
```

### 3. Run dbt Models
```bash
cd dbt/predict_may
dbt run
dbt test
```

### 4. Start Analysis
```bash
# Launch Jupyter
jupyter lab

# Or open in VS Code
code .
```

## Using DuckDB in VS Code

1. Open Command Palette (Cmd+Shift+P)
2. Type "SQLTools: Connect"
3. Choose "predict_may" connection
4. Browse tables and run queries!

## Project Workflow

```
1. Ingest data → src/ingestion/api_football_fixtures.py
2. Transform data → dbt run (in dbt/predict_may/)
3. Analyze → notebooks/
4. Visualize → Create charts, export to visualizations/
5. Report → Write up insights in reports/
```

## Useful Commands

```bash
# dbt
dbt run                    # Run all models
dbt run --select staging   # Run only staging models
dbt test                   # Run tests
dbt docs generate          # Generate documentation
dbt docs serve             # View docs in browser

# Python
python -m src.ingestion.api_football_fixtures  # Run ingestion

# Jupyter
jupyter lab                # Launch Jupyter
jupyter notebook list      # See running notebooks
```

## Querying in Python

```python
import duckdb

# Connect to database
con = duckdb.connect('data/football.duckdb')

# Query raw data
df = con.sql("SELECT * FROM raw.fixtures").df()

# Query dbt models (after dbt run)
df = con.sql("SELECT * FROM staging.stg_fixtures").df()
```

## MotherDuck (Optional)

To sync your database to the cloud:

```python
import duckdb
con = duckdb.connect('md:predict_may')
# Your queries automatically sync
```

Get token at: https://motherduck.com
