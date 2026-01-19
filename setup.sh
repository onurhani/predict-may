#!/bin/bash

# Setup script for predict_may project
# Run from: ~/git/predict_may

set +e  # Don't exit on errors

echo "🏈 Setting up predict_may analytics environment..."
echo "=================================================="

# Verify we're in the right place
if [ ! -f "requirements.txt" ] || [ ! -d "dbt" ]; then
    echo "❌ Error: Run this script from ~/git/predict_may directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "✅ Confirmed we're in predict_may project"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Install Python 3.11 if needed
if ! command -v python3.11 &> /dev/null; then
    echo "🐍 Installing Python 3.11..."
    brew install python@3.11
else
    echo "✅ Python 3.11 already installed"
fi

# Install VS Code if needed
if ! command -v code &> /dev/null; then
    echo "📝 Installing Visual Studio Code..."
    brew install --cask visual-studio-code 2>/dev/null || echo "⚠️  VS Code may already be installed"
    sleep 3
else
    echo "✅ VS Code already installed"
fi

echo ""
echo "🔌 Installing VS Code extensions..."
code --install-extension ms-python.python 2>/dev/null || true
code --install-extension ms-toolsai.jupyter 2>/dev/null || true
code --install-extension mtxr.sqltools 2>/dev/null || true
code --install-extension evidence-dev.sqltools-duckdb-driver 2>/dev/null || true
code --install-extension mechatroner.rainbow-csv 2>/dev/null || true
code --install-extension ms-vscode.vscode-data-wrangler 2>/dev/null || true
code --install-extension yzhang.markdown-all-in-one 2>/dev/null || true
code --install-extension innoverio.vscode-dbt-power-user 2>/dev/null || true

echo "✅ Extensions installed"
echo ""

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  venv already exists, skipping creation"
else
    python3.11 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements from requirements.txt
echo "📦 Installing project requirements..."
pip install -r requirements.txt --quiet

# Install additional analytics packages
echo "📦 Installing additional analytics packages..."
pip install \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    plotly \
    streamlit \
    pyarrow \
    openpyxl \
    --quiet

# Install MotherDuck support
pip install 'duckdb[motherduck]' --quiet

echo "✅ All packages installed"
echo ""

# Create additional folders if they don't exist
echo "📁 Setting up project folders..."
mkdir -p data
mkdir -p notebooks
mkdir -p visualizations
mkdir -p reports

# Update .gitignore
echo "📝 Updating .gitignore..."
if [ -f ".gitignore" ]; then
    # Backup existing
    cp .gitignore .gitignore.backup
fi

cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.env

# DuckDB
data/*.duckdb
data/*.duckdb.wal

# dbt
dbt/predict_may/target/
dbt/predict_may/dbt_packages/
dbt/predict_may/logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (optional - you may want to commit some)
# data/*.csv
# data/*.parquet
EOF

echo "✅ .gitignore updated"
echo ""

# Create example notebook
echo "📓 Creating example analysis notebook..."
cat > notebooks/exploratory_analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Predict May - Exploratory Analysis\n",
    "\n",
    "Analyzing Turkish Süper Lig data with DuckDB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import duckdb\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set style\n",
    "sns.set_style('whitegrid')\n",
    "plt.rcParams['figure.figsize'] = (12, 6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Connect to your DuckDB database\n",
    "con = duckdb.connect('../data/football.duckdb')\n",
    "\n",
    "# Check what tables exist\n",
    "con.sql(\"SHOW TABLES\").show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Query raw fixtures\n",
    "fixtures = con.sql(\"\"\"\n",
    "    SELECT * \n",
    "    FROM raw.fixtures \n",
    "    ORDER BY date DESC \n",
    "    LIMIT 10\n",
    "\"\"\").df()\n",
    "\n",
    "fixtures"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Goals per season\n",
    "season_goals = con.sql(\"\"\"\n",
    "    SELECT \n",
    "        season,\n",
    "        COUNT(*) as matches,\n",
    "        SUM(home_goals + away_goals) as total_goals,\n",
    "        ROUND(AVG(home_goals + away_goals), 2) as avg_goals_per_match\n",
    "    FROM raw.fixtures\n",
    "    GROUP BY season\n",
    "    ORDER BY season\n",
    "\"\"\").df()\n",
    "\n",
    "season_goals"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(data=season_goals, x='season', y='avg_goals_per_match')\n",
    "plt.title('Average Goals per Match by Season')\n",
    "plt.ylabel('Goals per Match')\n",
    "plt.xlabel('Season')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Query dbt models (after running dbt)\n",
    "# con.sql(\"SELECT * FROM staging.stg_fixtures LIMIT 10\").df()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✅ Example notebook created"
echo ""

# Create activation script
cat > activate.sh << 'EOF'
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
EOF

chmod +x activate.sh

# Create VS Code workspace settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true
    },
    "python.analysis.extraPaths": [
        "${workspaceFolder}/dbt/predict_may"
    ],
    "[sql]": {
        "editor.defaultFormatter": "dorzey.vscode-sqlfluff"
    }
}
EOF

# Create SQLTools connection config for VS Code
cat > .vscode/sqltools.json << 'EOF'
{
    "connections": [
        {
            "name": "predict_may",
            "driver": "DuckDB",
            "database": "${workspaceFolder}/data/football.duckdb",
            "previewLimit": 100
        }
    ]
}
EOF

echo "✅ VS Code configuration created"
echo ""

# Update requirements.txt with full dependencies
cat > requirements.txt << 'EOF'
# dbt & DuckDB
dbt-core>=1.7
dbt-duckdb>=1.7
duckdb

# Data manipulation
pandas
numpy
pyarrow

# Visualization
matplotlib
seaborn
plotly

# API & utilities
requests
python-dotenv

# Analysis & notebooks
jupyter
jupyterlab

# Reporting
streamlit
EOF

echo "✅ requirements.txt updated"
echo ""

# Create a quick reference guide
cat > SETUP.md << 'EOF'
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
EOF

echo "✅ Setup guide created"
echo ""

echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "📁 Your project is ready at: $(pwd)"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Make sure you have your .env file with API_FOOTBALL_KEY"
echo ""
echo "2. Activate the environment:"
echo "   source activate.sh"
echo ""
echo "3. Start working:"
echo "   jupyter lab              # For analysis"
echo "   code .                   # Open in VS Code"
echo "   cd dbt/predict_may && dbt run  # Run dbt models"
echo ""
echo "📚 See SETUP.md for detailed workflow guide"
echo "📓 Check notebooks/exploratory_analysis.ipynb to get started"
echo ""
echo "=================================================="
