"""
Export dashboard data to docs/data/dashboard.json
Run this after simulate_season.py to publish fresh data for the GitHub Pages dashboard.
"""
import json
import os
import urllib.request
import duckdb
import pandas as pd
from datetime import date, datetime

DB_PATH = "data/football.duckdb"
OUTPUT_PATH = "docs/data/dashboard.json"
CURRENT_SEASON = 2025  # 2025-26 season
MATCHDAY_TOTAL = 34

# Sofascore season ID for Süper Lig 2025-26
SOFASCORE_SEASON_ID = 77805

# Map Sofascore team names → DB team names
SOFASCORE_TO_DB = {
    "Fenerbahçe": "Fenerbahce",
    "Beşiktaş JK": "Besiktas",
    "Başakşehir FK": "Buyuksehyr",
    "Göztepe": "Goztep",
    "Çaykur Rizespor": "Rizespor",
    "Fatih Karagümrük": "Karagumruk",
    "Kasımpaşa": "Kasimpasa",
    "Eyüpspor": "Eyupspor",
    "Gençlerbirliği": "Genclerbirligi",
    "Gaziantep FK": "Gaziantep",
}


def build_meta(con) -> dict:
    completed = con.execute("""
        SELECT COUNT(*) as n
        FROM raw.fixtures
        WHERE season = ? AND status = 'FT'
    """, [CURRENT_SEASON]).fetchone()[0]

    remaining = con.execute("""
        SELECT COUNT(*) as n
        FROM raw.fixtures
        WHERE season = ? AND status = 'NS'
    """, [CURRENT_SEASON]).fetchone()[0]

    # Derive matches per matchday from actual team count (N teams → N/2 games/round)
    n_teams = con.execute("""
        SELECT COUNT(DISTINCT home_team)
        FROM raw.fixtures WHERE season = ?
    """, [CURRENT_SEASON]).fetchone()[0]
    matches_per_matchday = max(n_teams // 2, 1)

    matchday_current = round(completed / matches_per_matchday)

    return {
        "updated_at": date.today().isoformat(),
        "matchday_current": matchday_current,
        "matchday_total": MATCHDAY_TOTAL,
        "matches_remaining_total": remaining,
    }


def build_standings(con) -> list:
    df = con.execute("""
        SELECT
            "Team"                 AS team,
            "Current Points"       AS current_pts,
            "Expected Points"      AS expected_pts,
            "Most Likely Position" AS likely_position,
            "Prob Top 4"           AS top4_str,
            "Prob Relegation"      AS relegation_str
        FROM main_marts.season_projections
    """).fetchdf()

    if df.empty:
        return []

    # Parse percentage strings stored as "XX.X%"
    df["top4_pct"] = df["top4_str"].str.rstrip("%").astype(float)
    df["relegation_pct"] = df["relegation_str"].str.rstrip("%").astype(float)

    # Rank by current points (ties share same rank; break by expected pts)
    df = df.sort_values(["current_pts", "expected_pts"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "rank": int(r["rank"]),
            "team": r["team"],
            "current_pts": int(r["current_pts"]),
            "expected_pts": float(r["expected_pts"]),
            "likely_position": int(r["likely_position"]),
            "top4_pct": round(r["top4_pct"], 1),
            "relegation_pct": round(r["relegation_pct"], 1),
        })

    return rows


def build_accuracy(con) -> dict:
    df = con.execute("""
        SELECT
            date_trunc('week', match_date) AS week_start,
            COUNT(*)                        AS total,
            SUM(correct_prediction)         AS correct
        FROM main_marts.match_predictions
        WHERE season = ?
        GROUP BY week_start
        ORDER BY week_start
    """, [CURRENT_SEASON]).fetchdf()

    if df.empty:
        return {
            "overall_pct": 0.0,
            "rolling_4week_pct": 0.0,
            "best_week_pct": 0.0,
            "weekly": [],
        }

    df["weekly_pct"] = (df["correct"] / df["total"] * 100).round(1)

    # 4-week rolling average
    df["rolling_pct"] = (
        df["weekly_pct"].rolling(window=4, min_periods=1).mean().round(1)
    )

    # Week label: "Aug W1", "Sep W3" etc (week of month = ceil(day/7))
    def week_label(ts) -> str:
        d = pd.Timestamp(ts)
        month_abbr = d.strftime("%b")
        week_in_month = (d.day - 1) // 7 + 1
        return f"{month_abbr} W{week_in_month}"

    weekly = []
    for _, row in df.iterrows():
        weekly.append({
            "week_label": week_label(row["week_start"]),
            "weekly_pct": float(row["weekly_pct"]),
            "rolling_pct": float(row["rolling_pct"]),
        })

    overall_pct = round(df["correct"].sum() / df["total"].sum() * 100, 1)
    rolling_4week_pct = float(df["rolling_pct"].iloc[-1]) if len(df) >= 1 else 0.0
    best_week_pct = float(df["weekly_pct"].max())

    return {
        "overall_pct": overall_pct,
        "rolling_4week_pct": rolling_4week_pct,
        "best_week_pct": best_week_pct,
        "weekly": weekly,
    }


def fetch_next_round_sofascore() -> list[dict] | None:
    """Fetch next unplayed round fixtures from Sofascore public API.
    Returns list of {home_team, away_team} dicts with DB-normalised names, or None on failure.
    """
    base = f"https://www.sofascore.com/api/v1/unique-tournament/52/season/{SOFASCORE_SEASON_ID}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    # Get all rounds
    try:
        req = urllib.request.Request(f"{base}/rounds", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            rounds_data = json.loads(resp.read())
    except Exception as e:
        print(f"  ⚠️  Sofascore rounds fetch failed: {e}")
        return None

    rounds = rounds_data.get("rounds", [])
    if not rounds:
        return None

    # Find the first round that has notstarted events
    for rnd in sorted(rounds, key=lambda r: r.get("round", 0)):
        round_num = rnd.get("round")
        try:
            req = urllib.request.Request(
                f"{base}/events/round/{round_num}", headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                events_data = json.loads(resp.read())
        except Exception:
            continue

        events = events_data.get("events", [])
        notstarted = [e for e in events if e.get("status", {}).get("type") == "notstarted"]

        if notstarted:
            print(f"  Sofascore: using round {round_num} ({len(notstarted)} fixtures)")
            fixtures = []
            for ev in notstarted:
                home_raw = ev["homeTeam"]["name"]
                away_raw = ev["awayTeam"]["name"]
                fixtures.append({
                    "home_team": SOFASCORE_TO_DB.get(home_raw, home_raw),
                    "away_team": SOFASCORE_TO_DB.get(away_raw, away_raw),
                })
            return fixtures

    return None


def build_next_matches(con) -> list:
    # Try to get actual next-round fixtures from Sofascore
    sofascore_fixtures = fetch_next_round_sofascore()

    if sofascore_fixtures:
        rows = []
        for fix in sofascore_fixtures:
            home, away = fix["home_team"], fix["away_team"]
            row = con.execute("""
                SELECT
                    home_team,
                    away_team,
                    prob_home_win,
                    prob_draw,
                    prob_away_win,
                    predicted_result
                FROM main_marts.match_predictions_future
                WHERE season = ?
                  AND home_team = ?
                  AND away_team = ?
                LIMIT 1
            """, [CURRENT_SEASON, home, away]).fetchone()

            if row is None:
                # Include the fixture with unknown probabilities
                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "match_date": None,
                    "prob_home": 33,
                    "prob_draw": 34,
                    "prob_away": 33,
                    "predicted_result": "D",
                })
                continue

            raw = [row[2], row[3], row[4]]
            total = sum(raw)
            scaled = [v / total for v in raw] if total > 0 else [1/3, 1/3, 1/3]
            ints = [round(v * 100) for v in scaled]
            diff = 100 - sum(ints)
            if diff != 0:
                ints[ints.index(max(ints))] += diff

            rows.append({
                "home_team": row[0],
                "away_team": row[1],
                "match_date": None,
                "prob_home": ints[0],
                "prob_draw": ints[1],
                "prob_away": ints[2],
                "predicted_result": row[5],
            })
        return rows

    # Fallback: first 9 NS fixtures from DB (dates are NULL so ordering is arbitrary)
    print("  ⚠️  Sofascore unavailable, falling back to DB fixture order")
    df = con.execute("""
        SELECT
            home_team,
            away_team,
            match_date,
            prob_home_win,
            prob_draw,
            prob_away_win,
            predicted_result
        FROM main_marts.match_predictions_future
        WHERE season = ?
        ORDER BY home_team
        LIMIT 9
    """, [CURRENT_SEASON]).fetchdf()

    rows = []
    for _, r in df.iterrows():
        raw = [r["prob_home_win"], r["prob_draw"], r["prob_away_win"]]
        total = sum(raw)
        scaled = [v / total for v in raw] if total > 0 else [1/3, 1/3, 1/3]
        ints = [round(v * 100) for v in scaled]
        diff = 100 - sum(ints)
        if diff != 0:
            ints[ints.index(max(ints))] += diff

        match_date_val = r["match_date"]
        date_str = None if pd.isna(match_date_val) else pd.Timestamp(match_date_val).strftime("%Y-%m-%d")

        rows.append({
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "match_date": date_str,
            "prob_home": ints[0],
            "prob_draw": ints[1],
            "prob_away": ints[2],
            "predicted_result": r["predicted_result"],
        })

    return rows


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    con = duckdb.connect(DB_PATH)

    print("📊 Building dashboard data...")

    meta = build_meta(con)
    print(f"  Meta: matchday {meta['matchday_current']}/{meta['matchday_total']}, "
          f"{meta['matches_remaining_total']} matches remaining")

    standings = build_standings(con)
    print(f"  Standings: {len(standings)} teams")

    accuracy = build_accuracy(con)
    print(f"  Accuracy: {accuracy['overall_pct']}% overall, "
          f"{len(accuracy['weekly'])} weeks")

    next_matches = build_next_matches(con)
    print(f"  Next matches: {len(next_matches)} fixtures")

    con.close()

    dashboard = {
        "meta": meta,
        "accuracy": accuracy,
        "standings": standings,
        "next_matches": next_matches,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Written to {OUTPUT_PATH}")
    print("   git add docs/data/dashboard.json && git commit -m 'data: matchday update' && git push")


if __name__ == "__main__":
    main()
