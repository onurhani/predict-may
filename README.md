# Data-driven football analysis and May standings projections

## Overview

**Predict May** is a personal analytics and data journalism project focused on the Turkish Süper Lig. The goal is to build transparent, reproducible models that analyze league dynamics and estimate end-of-season outcomes — especially **how the table might look by May**.

The project is inspired by data-driven football analysis (e.g., FiveThirtyEight's SPI ratings), but prioritizes:
- **Simplicity** – Start with results-only data, avoid unnecessary complexity
- **Explainability** – Every model decision should be understandable
- **Reproducibility** – Open source, documented, and rebuildable by anyone

## Current Status

**✅ Version 1.0 – Operational**

The complete prediction pipeline is working:
- Historical match data (2021-2025)
- Team strength ratings (SPI-based)
- Match outcome probabilities
- Monte Carlo season simulations
- May standings projections

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Source** | [API-Football](https://www.api-football.com/) | Match fixtures and results |
| **Local Database** | [DuckDB](https://duckdb.org/) | Fast analytical database for development |
| **Cloud Database** | [MotherDuck](https://motherduck.com/) | Cloud-hosted DuckDB for sharing/collaboration |
| **Transformations** | [dbt](https://www.getdbt.com/) | Data modeling and SQL transformations |
| **Simulation** | Python (NumPy, Pandas) | Monte Carlo season projections |
| **Exploration** | DBeaver | Database exploration and ad-hoc queries |
| **Version Control** | GitHub | Code repository and collaboration |

## Project Structure

```
predict-may/
├── data/
│   └── football.duckdb           # Local DuckDB database (not committed)
├── dbt/predict_may/
│   ├── models/
│   │   ├── staging/              # Clean, standardized source data
│   │   │   ├── stg_fixtures.sql
│   │   │   └── stg_fixtures_future.sql
│   │   ├── intermediate/         # Team-centric transformations
│   │   │   ├── int_team_matches.sql
│   │   │   ├── int_team_spi.sql
│   │   │   └── int_match_features.sql
│   │   └── marts/                # Analytics-ready models
│   │       ├── team_season_stats.sql
│   │       ├── match_predictions.sql
│   │       ├── match_predictions_future.sql
│   │       ├── current_team_ratings.sql
│   │       └── season_projections.sql
│   ├── dbt_project.yml
│   └── packages.yml
├── scripts/
│   ├── fetch_fixtures_history.py    # One-time: fetch 2021-2024
│   ├── fetch_fixtures_current.py    # Regular: update current season
│   ├── simulate_season.py           # Monte Carlo simulation
│   ├── sync_to_motherduck.py        # Push data to cloud
│   └── run_full_update.sh           # Complete pipeline update
├── .env                             # API keys and tokens (not committed)
├── requirements.txt
└── README.md
```

## Data Pipeline Architecture

```
┌─────────────────┐
│  API-Football   │
│  (Source Data)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Ingestion Scripts (Python)         │
│  - fetch_fixtures_history.py        │
│  - fetch_fixtures_current.py        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  DuckDB (Local)                     │
│  raw.fixtures                       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  dbt Transformations                │
│  staging → intermediate → marts     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Monte Carlo Simulation (Python)    │
│  - 10,000 season simulations        │
│  - Position probabilities           │
│  - Expected final points            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  May Standings Projections          │
│  + Data Journalism / Substack       │
└─────────────────────────────────────┘
```

## Setup & Installation

### Prerequisites
- Python 3.9+
- API-Football API key ([get one here](https://www.api-football.com/))
- MotherDuck account (optional, for cloud features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/onurhani/predict-may.git
   cd predict-may
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   API_FOOTBALL_KEY=your_api_key_here
   MOTHERDUCK_TOKEN=your_motherduck_token_here  # Optional
   ```

4. **Set up dbt profile**
   
   Create/edit `~/.dbt/profiles.yml`:
   ```yaml
   predict_may:
     target: dev
     
     outputs:
       dev:
         type: duckdb
         path: /absolute/path/to/predict-may/data/football.duckdb
         threads: 4
         
       motherduck:
         type: duckdb
         path: "md:predict_may?motherduck_token={{ env_var('MOTHERDUCK_TOKEN') }}"
         threads: 4
   ```

5. **Initial data load**
   ```bash
   # Fetch historical data (2021-2024)
   python scripts/fetch_fixtures_history.py
   
   # Fetch current season (2025)
   python scripts/fetch_fixtures_current.py
   
   # Build dbt models
   cd dbt/predict_may
   dbt run
   dbt test
   
   # Run simulation
   cd ../..
   python scripts/simulate_season.py
   ```

## Usage

### Quick Update (Recommended)

Run the complete pipeline:
```bash
./scripts/run_full_update.sh
```

### Manual Step-by-Step

**Update current season data:**
```bash
python scripts/fetch_fixtures_current.py
```

**Rebuild models:**
```bash
cd dbt/predict_may
dbt run
cd ../..
```

**Generate May projections:**
```bash
python scripts/simulate_season.py
```

**Sync to MotherDuck (optional):**
```bash
python scripts/sync_to_motherduck.py
cd dbt/predict_may
dbt run --target motherduck
```

### Exploring Results

**In DuckDB CLI:**
```bash
duckdb data/football.duckdb
```

```sql
-- View May projections
SELECT * FROM main_marts.season_projections
ORDER BY "Expected Points" DESC;

-- Check upcoming match predictions
SELECT 
    match_date,
    home_team,
    away_team,
    ROUND(prob_home_win * 100, 1) as home_win_pct,
    predicted_result
FROM main_marts.match_predictions_future
ORDER BY match_date
LIMIT 10;

-- Current team strength ratings
SELECT * FROM main_marts.current_team_ratings
ORDER BY spi_rating DESC;
```

**In MotherDuck:**
Visit [app.motherduck.com](https://app.motherduck.com) and run the same queries in the cloud.

## Modeling Approach

### SPI Ratings (Team Strength)
- Based on rolling 5-match form and season-to-date performance
- Weighted combination of goal difference, goals for, and goals against
- **Critical:** Uses only prior matches to avoid data leakage

### Match Predictions
- Home/away win probabilities using logistic regression on SPI difference
- Dynamic draw probability based on match closeness
- All probabilities normalized to sum to 1.0

### Season Simulation
- Monte Carlo method: 10,000 simulations of remaining fixtures
- Each match outcome sampled from probability distribution
- Aggregates results to show:
  - Expected final points
  - Most likely finishing position
  - Top 4 probability (Champions League)
  - Relegation risk

## Model Philosophy

1. **Start Simple** – Results-only data, no advanced metrics yet
2. **Explainable** – Every coefficient and formula should make intuitive sense
3. **Time-Aware** – Proper handling of time series, no future data in predictions
4. **Transparent** – SQL-based transformations, visible in dbt
5. **Iterative** – Build foundational model first, improve incrementally

## Contributing

This is a personal project, but feedback, suggestions, and forks are welcome!

If you find bugs or have ideas for improvements:
1. Open an issue
2. Submit a pull request
3. Reach out on Twitter/X or via email

## License

MIT License – Feel free to explore, fork, or adapt the ideas.

## Acknowledgments

- Inspired by [FiveThirtyEight's SPI ratings](https://projects.fivethirtyeight.com/soccer-predictions/)
- Data provided by [API-Football](https://www.api-football.com/)
- Built with [dbt](https://www.getdbt.com/), [DuckDB](https://duckdb.org/), and [MotherDuck](https://motherduck.com/)

---

**Follow the project:** Updates and analysis coming soon on [Substack](https://substack.com) (link TBD)

**Questions?** Open an issue or reach out!
