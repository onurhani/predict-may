"""
Fetch historical fixtures (completed past seasons only)
Run this ONCE initially, then only when archiving completed seasons
"""
import os
import duckdb
import requests
import pandas as pd
from dotenv import load_dotenv
from time import sleep

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
HEADERS = {"x-apisports-key": API_KEY}

DB_PATH = "data/football.duckdb"
LEAGUE_ID = 203  # Super Lig

# Only historical seasons (not current)
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]

def fetch_season_fixtures(season: int) -> pd.DataFrame:
    """Fetch all fixtures for a historical season"""
    print(f"📥 Fetching historical season {season}")

    resp = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={
            "league": LEAGUE_ID,
            "season": season
        }
    )
    resp.raise_for_status()

    fixtures = resp.json()["response"]

    rows = []
    for f in fixtures:
        status = f["fixture"]["status"]["short"]
        
        rows.append({
            "season": season,
            "date": f["fixture"]["date"][:10],
            "home_team": f["teams"]["home"]["name"],
            "away_team": f["teams"]["away"]["name"],
            "home_goals": f["goals"]["home"],
            "away_goals": f["goals"]["away"],
            "status": status,
            "fixture_id": f["fixture"]["id"]
        })

    return pd.DataFrame(rows)

def main():
    con = duckdb.connect(DB_PATH)

    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    con.execute("""
        CREATE TABLE IF NOT EXISTS raw.fixtures (
            season INTEGER,
            date DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_goals INTEGER,
            away_goals INTEGER,
            status VARCHAR,
            fixture_id INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    print("📚 Fetching historical seasons (2021-2024)...\n")

    for season in HISTORICAL_SEASONS:
        df = fetch_season_fixtures(season)
        print(f"→ {len(df)} fixtures")

        if not df.empty:
            # Delete old data for this season (avoid duplicates)
            con.execute("DELETE FROM raw.fixtures WHERE season = ?", [season])
            
            # Insert
            con.register("df", df)
            con.execute("""
                INSERT INTO raw.fixtures 
                (season, date, home_team, away_team, home_goals, away_goals, status, fixture_id) 
                SELECT * FROM df
            """)

        sleep(2)  # Rate limiting

    # Show summary
    summary = con.execute("""
        SELECT 
            season, 
            COUNT(*) as total_fixtures,
            SUM(CASE WHEN status = 'FT' THEN 1 ELSE 0 END) as finished
        FROM raw.fixtures
        WHERE season IN (2021, 2022, 2023, 2024)
        GROUP BY season
        ORDER BY season
    """).fetchdf()
    
    print("\n" + "="*50)
    print("📊 Historical fixtures summary:")
    print(summary.to_string(index=False))
    print("="*50)

    con.close()
    print("\n✅ Historical data ingestion complete")
    print("💡 Next: Run fetch_fixtures_current.py for 2025 season")

if __name__ == "__main__":
    main()