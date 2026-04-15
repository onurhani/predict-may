#!/usr/bin/env python3
"""
Fetch referee data for Turkish Super Lig 2025-26 from Sofascore.

For each completed match (GD1–current):
  1. Fetch individual event detail to get referee name + ID.
  2. Accumulate per-referee match outcomes (H/D/A) and card counts.
  3. For each referee, also fetch career-level stats from /referee/{id}/statistics
     and extract Super Lig appearance/card totals.
  4. Save to data/referee_stats.json.

Also saves per-match referee assignments to data/referee_assignments.json so
predict_gameday.py can look up which referee is doing which GD30 match once
Sofascore publishes the assignments.

Usage:
    python3 scripts/fetch_referee_data.py            # fetch all completed rounds
    python3 scripts/fetch_referee_data.py --round 30 # also try to include GD30
"""

import argparse
import json
import time
from pathlib import Path
import urllib.request
import urllib.error

ROOT           = Path(__file__).parent.parent
DATA_DIR       = ROOT / "data"
OUTPUT_STATS   = DATA_DIR / "referee_stats.json"
OUTPUT_ASSIGN  = DATA_DIR / "referee_assignments.json"

TOURNAMENT_ID = 52       # Trendyol Süper Lig
SEASON_ID     = 77805    # 2025-26
TOTAL_ROUNDS  = 34

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}

# Sofascore winnerCode: 1=home, 2=draw, 3=away
WINNER_MAP = {1: "H", 2: "D", 3: "A"}

# League average baselines (2025-26 season, approximate from football-data.co.uk)
# Used as fallback when a referee has < MIN_GAMES matches recorded.
LEAGUE_AVG_HOME = 0.44
LEAGUE_AVG_DRAW = 0.27
LEAGUE_AVG_AWAY = 0.29
MIN_GAMES_FOR_STATS = 5  # refs with fewer games get a blended estimate


def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if attempt < retries - 1:
                time.sleep(1.5)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5)
            else:
                raise
    return {}


def fetch_round_event_ids(round_num: int) -> list[dict]:
    """Return list of {id, homeTeam, awayTeam} for a round."""
    url = (f"https://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}"
           f"/season/{SEASON_ID}/events/round/{round_num}")
    data = fetch_json(url)
    events = data.get("events", [])
    return [
        {
            "event_id": ev["id"],
            "round": round_num,
            "home": ev["homeTeam"]["name"],
            "away": ev["awayTeam"]["name"],
            "winner_code": ev.get("winnerCode"),   # None if not finished
            "status": ev.get("status", {}).get("type", ""),
        }
        for ev in events
    ]


def fetch_event_detail(event_id: int) -> dict:
    """Return full event detail dict (or {})."""
    url = f"https://www.sofascore.com/api/v1/event/{event_id}"
    data = fetch_json(url)
    return data.get("event", {})


def fetch_referee_career_stats(referee_id: int) -> dict:
    """
    Return Super Lig career stats from /referee/{id}/statistics.
    Fields: appearances, yellowCards, redCards, yellowRedCards, penalty
    """
    url = f"https://www.sofascore.com/api/v1/referee/{referee_id}/statistics"
    data = fetch_json(url)
    for entry in data.get("statistics", []):
        if entry.get("uniqueTournament", {}).get("id") == TOURNAMENT_ID:
            return {
                "career_appearances": entry.get("appearances", 0),
                "career_yellow":      entry.get("yellowCards", 0),
                "career_red":         entry.get("redCards", 0),
                "career_yellow_red":  entry.get("yellowRedCards", 0),
                "career_penalties":   entry.get("penalty", 0),
            }
    return {}


def build_referee_stats(assignments: list[dict]) -> dict:
    """
    Given list of per-match assignments (with referee + outcome), compute:
      - matches, home_wins, draws, away_wins
      - draw_rate, home_win_rate, away_win_rate
      - yellow_cards, red_cards, yellow_per_game, red_per_game
    Returns dict keyed by referee_id.
    """
    refs: dict[int, dict] = {}

    for a in assignments:
        ref_id   = a.get("referee_id")
        ref_name = a.get("referee_name")
        outcome  = a.get("outcome")  # H / D / A or None
        yellows  = a.get("yellow_cards", 0) or 0
        reds     = a.get("red_cards", 0) or 0

        if ref_id is None or outcome is None:
            continue

        if ref_id not in refs:
            refs[ref_id] = {
                "referee_id":   ref_id,
                "referee_name": ref_name,
                "matches": 0, "home_wins": 0, "draws": 0, "away_wins": 0,
                "yellow_cards": 0, "red_cards": 0,
            }

        r = refs[ref_id]
        r["matches"]     += 1
        r["yellow_cards"] += yellows
        r["red_cards"]    += reds
        if outcome == "H":
            r["home_wins"] += 1
        elif outcome == "D":
            r["draws"] += 1
        elif outcome == "A":
            r["away_wins"] += 1

    # Compute rates (blend with league average for low-sample refs)
    result = {}
    for ref_id, r in refs.items():
        n = r["matches"]
        w = min(n / MIN_GAMES_FOR_STATS, 1.0)  # blend weight (0→1)

        raw_h = r["home_wins"] / n if n > 0 else LEAGUE_AVG_HOME
        raw_d = r["draws"]     / n if n > 0 else LEAGUE_AVG_DRAW
        raw_a = r["away_wins"] / n if n > 0 else LEAGUE_AVG_AWAY

        result[ref_id] = {
            **r,
            "home_win_rate":    round(w * raw_h + (1 - w) * LEAGUE_AVG_HOME, 4),
            "draw_rate":        round(w * raw_d + (1 - w) * LEAGUE_AVG_DRAW, 4),
            "away_win_rate":    round(w * raw_a + (1 - w) * LEAGUE_AVG_AWAY, 4),
            "yellow_per_game":  round(r["yellow_cards"] / n, 3) if n > 0 else None,
            "red_per_game":     round(r["red_cards"]    / n, 3) if n > 0 else None,
            # Bias ratio vs league average (1.0 = average; >1 = above average)
            "draw_bias":        round((w * raw_d + (1 - w) * LEAGUE_AVG_DRAW) / LEAGUE_AVG_DRAW, 4),
            "home_bias":        round((w * raw_h + (1 - w) * LEAGUE_AVG_HOME) / LEAGUE_AVG_HOME, 4),
            "away_bias":        round((w * raw_a + (1 - w) * LEAGUE_AVG_AWAY) / LEAGUE_AVG_AWAY, 4),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None,
                        help="Max round to fetch (default: all completed rounds)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if data already exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing assignments to avoid re-fetching event details
    existing_assign: list[dict] = []
    existing_event_ids: set[int] = set()
    if OUTPUT_ASSIGN.exists() and not args.force:
        existing_assign = json.loads(OUTPUT_ASSIGN.read_text())
        existing_event_ids = {a["event_id"] for a in existing_assign}
        print(f"Loaded {len(existing_assign)} existing assignments.")

    assignments: list[dict] = list(existing_assign)
    seen_ref_ids: set[int] = set(a["referee_id"] for a in assignments if a.get("referee_id"))
    career_cache: dict[int, dict] = {}

    # Determine which rounds to fetch
    max_round = args.round or TOTAL_ROUNDS
    print(f"Fetching rounds 1–{max_round} from Sofascore...")

    for rnd in range(1, max_round + 1):
        print(f"  Round {rnd}...", end=" ", flush=True)
        try:
            events = fetch_round_event_ids(rnd)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        new_count = 0
        for ev in events:
            eid = ev["event_id"]
            # Skip already fetched, unless it was incomplete before
            if eid in existing_event_ids:
                # Check if it was incomplete (no outcome) — re-fetch if so
                prev = next((a for a in existing_assign if a["event_id"] == eid), None)
                if prev and prev.get("outcome") is not None:
                    continue

            # Only fetch detail for finished matches
            if ev["status"] not in ("finished",):
                if ev["winner_code"] is None:
                    # Not finished, record as pending assignment
                    entry = {
                        "event_id":     eid,
                        "round":        rnd,
                        "home":         ev["home"],
                        "away":         ev["away"],
                        "outcome":      None,
                        "referee_id":   None,
                        "referee_name": None,
                        "yellow_cards": None,
                        "red_cards":    None,
                    }
                    if eid not in existing_event_ids:
                        assignments.append(entry)
                        existing_event_ids.add(eid)
                    continue

            # Fetch event detail
            try:
                detail = fetch_event_detail(eid)
                time.sleep(0.2)
            except Exception as e:
                print(f"\n    Event {eid} detail FAILED: {e}")
                continue

            ref    = detail.get("referee") or {}
            ref_id = ref.get("id")
            ref_nm = ref.get("name")

            # Outcome
            wc      = detail.get("winnerCode") or ev["winner_code"]
            outcome = WINNER_MAP.get(wc)

            # Card counts from homeScore/awayScore (not always available)
            hs = detail.get("homeScore", {})
            as_ = detail.get("awayScore", {})
            # Sofascore doesn't expose card totals directly in event detail in a simple field.
            # Use yellowCards from referee profile (career total is not per-game useful here).
            # For per-game cards we rely on accumulated stats across all matches below.
            # We'll store None and compute from aggregated data.

            entry = {
                "event_id":     eid,
                "round":        rnd,
                "home":         ev["home"],
                "away":         ev["away"],
                "outcome":      outcome,
                "referee_id":   ref_id,
                "referee_name": ref_nm,
                "yellow_cards": None,   # not reliably available at event level
                "red_cards":    None,
            }

            # Fetch career stats for new referees
            if ref_id and ref_id not in seen_ref_ids:
                try:
                    career = fetch_referee_career_stats(ref_id)
                    career_cache[ref_id] = career
                    seen_ref_ids.add(ref_id)
                    time.sleep(0.2)
                except Exception:
                    pass

            # Update or append
            if eid in existing_event_ids:
                for i, a in enumerate(assignments):
                    if a["event_id"] == eid:
                        assignments[i] = entry
                        break
            else:
                assignments.append(entry)
                existing_event_ids.add(eid)
                new_count += 1

        print(f"{len(events)} events, {new_count} new")

    # Build referee stats from assignments
    print("\nBuilding referee statistics...")
    ref_stats = build_referee_stats(assignments)

    # Merge career stats
    for ref_id, career in career_cache.items():
        if ref_id in ref_stats:
            ref_stats[ref_id].update(career)

    # Also fetch career stats for refs already in stats but not in cache
    for ref_id, rs in ref_stats.items():
        if "career_appearances" not in rs:
            try:
                career = fetch_referee_career_stats(ref_id)
                if career:
                    ref_stats[ref_id].update(career)
                time.sleep(0.2)
            except Exception:
                pass

    # Summary
    print(f"  Referees found:  {len(ref_stats)}")
    print(f"  Total matches with referee: {sum(1 for a in assignments if a.get('referee_id'))}")
    print(f"  Matches without referee:    {sum(1 for a in assignments if not a.get('referee_id'))}")

    print("\nReferee stats (this season, sorted by matches):")
    print(f"  {'Name':<28} {'N':>4} {'H%':>6} {'D%':>6} {'A%':>6} {'D-bias':>7} {'H-bias':>7}  Career SL")
    print("  " + "─" * 80)
    for rs in sorted(ref_stats.values(), key=lambda x: -x["matches"]):
        career_ap = rs.get("career_appearances", "?")
        print(f"  {rs['referee_name']:<28} {rs['matches']:>4} "
              f"{rs['home_win_rate']*100:>5.1f}% {rs['draw_rate']*100:>5.1f}% {rs['away_win_rate']*100:>5.1f}%  "
              f"{rs['draw_bias']:>6.3f}x {rs['home_bias']:>6.3f}x  {career_ap}")

    # Save files
    output = {
        "league_avg": {
            "home_win_rate": LEAGUE_AVG_HOME,
            "draw_rate":     LEAGUE_AVG_DRAW,
            "away_win_rate": LEAGUE_AVG_AWAY,
        },
        "referees": ref_stats,
    }
    OUTPUT_STATS.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved referee stats  → {OUTPUT_STATS}")

    OUTPUT_ASSIGN.write_text(json.dumps(assignments, indent=2, ensure_ascii=False))
    print(f"Saved assignments    → {OUTPUT_ASSIGN}")


if __name__ == "__main__":
    main()
