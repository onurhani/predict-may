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
SEASONS = [2021, 2022, 2023, 2024, 2025]

def fetch_season(season: int) -> pd.DataFrame:
    print(f"📥 Fetching season {season}")

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
        if f["fixture"]["status"]["short"] != "FT":
            continue

        rows.append({
            "season": season,
            "date": f["fixture"]["date"][:10],
            "home_team": f["teams"]["home"]["name"],
            "away_team": f["teams"]["away"]["name"],
            "home_goals": f["goals"]["home"],
            "away_goals": f["goals"]["away"],
        })

    return pd.DataFrame(rows)

def main():
    con = duckdb.connect(DB_PATH)

    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    con.execute("""
        CREATE OR REPLACE TABLE raw.fixtures (
            season INTEGER,
            date DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_goals INTEGER,
            away_goals INTEGER
        )
    """)

    for season in SEASONS:
        df = fetch_season(season)
        print(f"→ {len(df)} matches")

        if not df.empty:
            con.register("df", df)
            con.execute("INSERT INTO raw.fixtures SELECT * FROM df")

        sleep(2)

    con.close()
    print("✅ Ingestion complete")

if __name__ == "__main__":
    main()
