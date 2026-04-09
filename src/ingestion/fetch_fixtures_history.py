"""
Fetch historical fixtures (completed past seasons only)
Run this ONCE initially, then only when archiving completed seasons
Data source: football-data.co.uk (free, no API key required)
"""
import hashlib
import duckdb
import pandas as pd
from datetime import datetime

DB_PATH = "data/football.duckdb"

# football-data.co.uk URLs for completed historical seasons
# 2023-24 had 20 teams (380 matches), 2024-25 had 19 teams (342 matches)
SEASON_URLS = {
    2023: "https://www.football-data.co.uk/mmz4281/2324/T1.csv",
    2024: "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
}


def make_fixture_id(season: int, home_team: str, away_team: str) -> int:
    """Generate a stable surrogate fixture_id from match identifiers (fits INT32)."""
    s = f"{season}:{home_team}:{away_team}"
    return int(hashlib.md5(s.encode()).hexdigest()[:7], 16)  # max 0xFFFFFFF = 268M < INT32 max


def infer_season(date_obj) -> int:
    """
    Turkish Süper Lig runs Aug–May.
    Aug–Dec match belongs to that calendar year's season.
    Jan–Jul match belongs to the prior year's season.
    """
    if date_obj.month >= 8:
        return date_obj.year
    else:
        return date_obj.year - 1


def parse_csv(url: str, expected_season: int) -> pd.DataFrame:
    """Download and parse a football-data.co.uk CSV for one season."""
    print(f"  Downloading {url}")
    df = pd.read_csv(url, encoding="latin-1")

    # Keep only columns we need; skip rows with missing results
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    df = df[df["FTHG"].astype(str).str.strip() != ""]
    df = df[df["FTAG"].astype(str).str.strip() != ""]

    # Parse date: football-data.co.uk uses DD/MM/YYYY
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # Infer season from match date; warn if it doesn't match expected
    df["season"] = df["date"].apply(infer_season)
    unexpected = df[df["season"] != expected_season]
    if not unexpected.empty:
        print(
            f"  ⚠️  {len(unexpected)} rows inferred as season "
            f"{unexpected['season'].unique().tolist()} (expected {expected_season}) — check dates"
        )
    df = df[df["season"] == expected_season]

    df["home_team"] = df["HomeTeam"].str.strip()
    df["away_team"] = df["AwayTeam"].str.strip()
    df["home_goals"] = df["FTHG"].astype(int)
    df["away_goals"] = df["FTAG"].astype(int)
    df["status"] = "FT"
    df["fixture_id"] = df.apply(
        lambda r: make_fixture_id(r["season"], r["home_team"], r["away_team"]), axis=1
    )
    df["date"] = df["date"].dt.date  # convert to plain date

    return df[["season", "date", "home_team", "away_team", "home_goals", "away_goals", "status", "fixture_id"]]


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

    print("📚 Fetching historical seasons (2023–2024)...\n")

    total_inserted = 0
    for season, url in SEASON_URLS.items():
        print(f"Season {season} ({season}-{str(season+1)[2:]})")
        df = parse_csv(url, expected_season=season)
        print(f"  → {len(df)} completed matches")

        if not df.empty:
            con.execute("DELETE FROM raw.fixtures WHERE season = ?", [season])
            con.register("df_hist", df)
            con.execute("""
                INSERT INTO raw.fixtures
                    (season, date, home_team, away_team, home_goals, away_goals, status, fixture_id)
                SELECT season, date, home_team, away_team, home_goals, away_goals, status, fixture_id
                FROM df_hist
            """)
            total_inserted += len(df)

        # Print distinct team names so caller can spot naming inconsistencies
        teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        print(f"  Teams ({len(teams)}): {', '.join(teams)}\n")

    summary = con.execute("""
        SELECT
            season,
            COUNT(*) as total_fixtures,
            SUM(CASE WHEN status = 'FT' THEN 1 ELSE 0 END) as finished
        FROM raw.fixtures
        WHERE season IN (2023, 2024)
        GROUP BY season
        ORDER BY season
    """).fetchdf()

    print("=" * 50)
    print("📊 Historical fixtures summary:")
    print(summary.to_string(index=False))
    print("=" * 50)

    con.close()
    print(f"\n✅ Historical data ingestion complete ({total_inserted} rows)")
    print("💡 Next: Run fetch_fixtures_current.py for the 2025 season")


if __name__ == "__main__":
    main()
