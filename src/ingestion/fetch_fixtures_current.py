"""
Fetch current season fixtures (2024-25 = season 2024)
Run this FREQUENTLY (daily/weekly) during the season.
Data source: football-data.co.uk (free, no API key required)

Completed matches are loaded directly from the CSV.
Upcoming fixtures are generated as skeleton rows (home/away pairs not yet played),
because football-data.co.uk only publishes results, not future schedules.
"""
import hashlib
import duckdb
import pandas as pd
from itertools import permutations

DB_PATH = "data/football.duckdb"
CURRENT_SEASON = 2025  # 2025-26 season
CURRENT_URL = "https://www.football-data.co.uk/mmz4281/2526/T1.csv"


def make_fixture_id(season: int, home_team: str, away_team: str) -> int:
    """Generate a stable surrogate fixture_id from match identifiers (fits INT32)."""
    s = f"{season}:{home_team}:{away_team}"
    return int(hashlib.md5(s.encode()).hexdigest()[:7], 16)  # max 0xFFFFFFF = 268M < INT32 max


def infer_season(date_obj) -> int:
    """Aug–Dec → that calendar year; Jan–Jul → prior year."""
    return date_obj.year if date_obj.month >= 8 else date_obj.year - 1


def fetch_completed(url: str) -> pd.DataFrame:
    """Download the current season CSV and return completed matches."""
    print(f"  Downloading {url}")
    df = pd.read_csv(url, encoding="latin-1")

    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    # Keep only rows that have full-time results
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    df = df[df["FTHG"].astype(str).str.strip() != ""]
    df = df[df["FTAG"].astype(str).str.strip() != ""]

    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["season"] = df["date"].apply(infer_season)
    df = df[df["season"] == CURRENT_SEASON]

    df["home_team"] = df["HomeTeam"].str.strip()
    df["away_team"] = df["AwayTeam"].str.strip()
    df["home_goals"] = df["FTHG"].astype(int)
    df["away_goals"] = df["FTAG"].astype(int)
    df["status"] = "FT"
    df["fixture_id"] = df.apply(
        lambda r: make_fixture_id(r["season"], r["home_team"], r["away_team"]), axis=1
    )
    df["date"] = df["date"].dt.date

    return df[["season", "date", "home_team", "away_team", "home_goals", "away_goals", "status", "fixture_id"]]


def generate_upcoming(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Generate skeleton upcoming fixtures: all home/away pairs not yet played.

    Turkish Süper Lig: 18 teams, double round-robin = 306 matches total.
    We infer the full fixture list from all permutations of the teams that
    appeared in completed matches, then remove already-played pairs.
    """
    teams = sorted(set(completed["home_team"].tolist() + completed["away_team"].tolist()))

    if len(teams) == 0:
        return pd.DataFrame(columns=["season", "date", "home_team", "away_team",
                                     "home_goals", "away_goals", "status", "fixture_id"])

    # All possible (home, away) pairs — each team plays every other team home AND away
    all_pairs = set(permutations(teams, 2))

    # Remove already-played pairs
    played_pairs = set(
        zip(completed["home_team"], completed["away_team"])
    )
    remaining_pairs = all_pairs - played_pairs

    rows = []
    for home_team, away_team in sorted(remaining_pairs):
        rows.append({
            "season": CURRENT_SEASON,
            "date": None,          # Date unknown for future fixtures
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": None,
            "away_goals": None,
            "status": "NS",
            "fixture_id": make_fixture_id(CURRENT_SEASON, home_team, away_team),
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

    print(f"📥 Fetching current season ({CURRENT_SEASON}) fixtures...\n")

    # --- Completed matches ---
    completed = fetch_completed(CURRENT_URL)
    print(f"  → {len(completed)} completed matches (FT)")

    # Print distinct team names for manual consistency check
    teams = sorted(set(completed["home_team"].tolist() + completed["away_team"].tolist()))
    print(f"  Teams ({len(teams)}): {', '.join(teams)}\n")

    # --- Upcoming skeleton fixtures ---
    upcoming = generate_upcoming(completed)
    print(f"  → {len(upcoming)} upcoming fixtures generated (NS)\n")

    # Combine
    all_fixtures = pd.concat([completed, upcoming], ignore_index=True)

    # CRITICAL: Delete old current-season data to avoid duplicates
    print(f"🗑️  Clearing old season {CURRENT_SEASON} data...")
    con.execute("DELETE FROM raw.fixtures WHERE season = ?", [CURRENT_SEASON])

    print(f"📊 Inserting {len(all_fixtures)} fixtures...")
    con.register("df_current", all_fixtures)
    con.execute("""
        INSERT INTO raw.fixtures
            (season, date, home_team, away_team, home_goals, away_goals, status, fixture_id)
        SELECT season, date, home_team, away_team, home_goals, away_goals, status, fixture_id
        FROM df_current
    """)

    summary = con.execute("""
        SELECT
            status,
            COUNT(*) as count
        FROM raw.fixtures
        WHERE season = ?
        GROUP BY status
        ORDER BY status
    """, [CURRENT_SEASON]).fetchdf()

    print("\n" + "=" * 50)
    print(f"📊 Season {CURRENT_SEASON} status breakdown:")
    print(summary.to_string(index=False))
    print("=" * 50)

    con.close()
    print(f"\n✅ Current season data updated")
    print("💡 Next: ./scripts/run_full_update.sh  (or dbt run → simulate_season.py)")


if __name__ == "__main__":
    main()
