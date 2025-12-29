# Predict May  
### Analyze Turkish football with the intention of predicting the standings in May

## Overview
**Predict May** is a personal analytics and data journalism project focused on the Turkish Süper Lig.  
The goal is to build transparent, reproducible models that analyze league dynamics and estimate end-of-season outcomes — especially **how the table might look by May**.

The project is inspired by data-driven football analysis (e.g. FiveThirtyEight), but prioritizes:
- simplicity
- explainability
- open-source reproducibility

## Current Status (v1 – in progress)
The project is currently in **Version 1**, focusing on results-based modeling using historical match data.

Completed so far:
- ✅ Data ingestion into DuckDB
- ✅ Raw fixtures table
- ✅ dbt project setup
- ✅ Staging model (`stg_fixtures`) with:
  - cleaned team names
  - parsed dates
  - computed points
  - deterministic `match_id`

Next steps:
- 🔜 Intermediate team-centric models
- 🔜 Rolling form features
- 🔜 Match-level prediction features
- 🔜 Season simulation & probabilities
- 🔜 Visualizations and analytical articles

## Tech Stack
- **DuckDB** – local analytical database
- **dbt** – data modeling & transformations
- **Python** – ingestion & future modeling
- **DBeaver** – data exploration
- **GitHub** – version control & open source

## Project Structure

├── data/
│ └── football.duckdb # DuckDB database (not committed)
├── src/
│ └── ingestion/ # Data ingestion scripts
├── dbt/
│ ├── models/
│ │ ├── staging/ # Cleaned, standardized models
│ │ ├── intermediate/ # Team-centric & rolling features
│ │ └── marts/ # Prediction-ready views
│ ├── dbt_project.yml
│ └── packages.yml
├── README.md


## Modeling Philosophy
- Start with **results-only data**
- Avoid unnecessary complexity early
- Use **team-centric, time-aware** features
- Prefer SQL + dbt for transparency
- Iterate toward stronger models incrementally

This project intentionally starts simple and improves over time.

## Future Ideas
- SPI / Elo-style team strength models
- Monte Carlo season simulations
- Home/away & form-based adjustments
- Data journalism articles explaining insights
- Public-facing visualizations

## License
MIT License.  
Feel free to explore, fork, or adapt the ideas.
