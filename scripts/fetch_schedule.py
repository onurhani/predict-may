"""
Fetch the full season schedule from Sofascore and write it as a dbt seed CSV.

Run once per season (or when schedule changes significantly):
    python scripts/fetch_schedule.py

Output: dbt/predict_may/seeds/schedule_2526.csv
Columns: season, round_number, home_team, away_team
"""
import csv
import json
import time
import urllib.request

SOFASCORE_SEASON_ID = 77805  # Süper Lig 2025-26
CURRENT_SEASON = 2025
TOURNAMENT_ID = 52  # Turkish Super Lig
OUTPUT_PATH = "dbt/predict_may/seeds/schedule_2526.csv"

# Map Sofascore team names → DB team names (keep in sync with export_dashboard.py)
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

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

BASE = f"https://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{SOFASCORE_SEASON_ID}"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def normalize(name: str) -> str:
    return SOFASCORE_TO_DB.get(name, name)


def main():
    print("Fetching rounds list...")
    rounds_data = fetch_json(f"{BASE}/rounds")
    rounds = sorted(rounds_data.get("rounds", []), key=lambda r: r.get("round", 0))
    print(f"  Found {len(rounds)} rounds")

    rows = []
    for rnd in rounds:
        round_num = rnd.get("round")
        print(f"  Fetching round {round_num}...", end=" ")
        try:
            events_data = fetch_json(f"{BASE}/events/round/{round_num}")
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        events = events_data.get("events", [])
        for ev in events:
            home = normalize(ev["homeTeam"]["name"])
            away = normalize(ev["awayTeam"]["name"])
            rows.append({
                "season": CURRENT_SEASON,
                "round_number": round_num,
                "home_team": home,
                "away_team": away,
            })
        print(f"{len(events)} matches")
        time.sleep(0.3)  # be polite

    # Deduplicate (Sofascore occasionally returns duplicate events per round)
    seen = set()
    deduped = []
    for r in rows:
        key = (r["season"], r["round_number"], r["home_team"], r["away_team"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    if len(deduped) < len(rows):
        print(f"  Deduplicated: {len(rows)} → {len(deduped)} rows")
    rows = deduped

    print(f"\nWriting {len(rows)} rows to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["season", "round_number", "home_team", "away_team"])
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
