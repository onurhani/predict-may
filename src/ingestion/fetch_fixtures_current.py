"""
Fetch current season fixtures (both finished and upcoming)
Run this FREQUENTLY (daily/weekly) during the season
"""
import os
import duckdb
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
HEADERS = {"x-apisports-key": API_KEY}

DB_PATH = "data/football.duckdb"
LEAGUE_ID = 203  # Super Lig
CURRENT_SEASON = 2025

def fetch_current_season() -> pd.DataFrame:
    """
    Fetch all fixtures for current season (both finished and upcoming)
    """
    print(f"📥 Fetching current season ({CURRENT_SEASON}) fixtures...")

    resp = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={
            "league": LEAGUE_ID,
            "season": CURRENT_SEASON
        }
    )
    resp.raise_for_status()

    fixtures = resp.json()["response"]

    rows = []
    for f in fixtures:
        status = f["fixture"]["status"]["short"]
        
        rows.append({
            "season": CURRENT_SEASON,
            "date": f["fixture"]["date"][:10],
            "home_team": f["teams"]["home"]["name"],
            "away_team": f["teams"]["away"]["name"],
            "home_goals": f["goals"]["home"],  # NULL for future fixtures
            "away_goals": f["goals"]["away"],  # NULL for future fixtures
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

    # Fetch current season
    df = fetch_current_season()
    
    finished_count = (df['status'] == 'FT').sum()
    future_count = df['status'].isin(['NS', 'TBD', 'PST']).sum()
    
    print(f"→ {finished_count} finished matches")
    print(f"→ {future_count} upcoming matches")
    print(f"→ {len(df)} total fixtures")

    # CRITICAL: Delete old current season data to avoid duplicates
    print(f"\n🗑️  Clearing old {CURRENT_SEASON} data...")
    con.execute("DELETE FROM raw.fixtures WHERE season = ?", [CURRENT_SEASON])

    # Insert all fixtures
    if not df.empty:
        print(f"📊 Inserting {len(df)} fixtures...")
        con.register("df", df)
        con.execute("""
            INSERT INTO raw.fixtures 
            (season, date, home_team, away_team, home_goals, away_goals, status, fixture_id)
            SELECT * FROM df
        """)

    # Show current status
    summary = con.execute("""
        SELECT 
            status,
            COUNT(*) as count
        FROM raw.fixtures 
        WHERE season = ?
        GROUP BY status
        ORDER BY status
    """, [CURRENT_SEASON]).fetchdf()

    print("\n" + "="*50)
    print(f"📊 Current season ({CURRENT_SEASON}) status:")
    print(summary.to_string(index=False))
    print("="*50)

    con.close()
    print("\n✅ Current season data updated")
    print("💡 Next: dbt run --target motherduck")

if __name__ == "__main__":
    main()