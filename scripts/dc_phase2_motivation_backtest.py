#!/usr/bin/env python3
"""
Dixon-Coles Phase 2 + Phase 3 Backtest
Adds squad market value (AMV) and motivation score modifiers to DC Phase 1 probabilities.
"""

import json
import math
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "transfermarkt")
DC_PREDS_FILE = os.path.join(SCRIPT_DIR, "dc_predictions.json")
DASHBOARD_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "docs", "data", "dashboard.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "dc_phase2_predictions.json")

# ─── Team name mapping: CSV ASCII → Transfermarkt club_id ─────────────────────
CSV_TO_TM_ID = {
    "Galatasaray": 141,
    "Fenerbahce": 36,
    "Besiktas": 114,
    "Trabzonspor": 449,
    "Buyuksehyr": 6890,   # İstanbul Başakşehir (corrected from 2831)
    "Kayserispor": 3205,
    "Konyaspor": 2293,
    "Kasimpasa": 10484,
    "Alanyaspor": 11282,
    "Antalyaspor": 589,
    "Rizespor": 126,
    "Samsunspor": 152,
    "Gaziantep": 2832,
    "Eyupspor": 7160,
    "Genclerbirligi": 820,   # corrected from 38 (Fortuna Düsseldorf)
    "Goztep": 1467,           # Göztepe
    "Karagumruk": 6646,       # Fatih Karagümrük (corrected from 10267)
    "Kocaelispor": 120,
}

# ─── Load football-data.co.uk match results for motivation standings ──────────
SEASONS = ["2122", "2223", "2324", "2425", "2526"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/T1.csv"

# Name normalization for football-data.co.uk team names → our CSV names
FD_NAME_MAP = {
    "Galatasaray": "Galatasaray",
    "Fenerbahce": "Fenerbahce",
    "Besiktas": "Besiktas",
    "Trabzonspor": "Trabzonspor",
    "Istanbul Basaksehir": "Buyuksehyr",
    "Basaksehir": "Buyuksehyr",
    "Kayserispor": "Kayserispor",
    "Konyaspor": "Konyaspor",
    "Kasimpasa": "Kasimpasa",
    "Alanyaspor": "Alanyaspor",
    "Antalyaspor": "Antalyaspor",
    "Rizespor": "Rizespor",
    "Caykur Rizespor": "Rizespor",
    "Samsunspor": "Samsunspor",
    "Gaziantep FK": "Gaziantep",
    "Gaziantep": "Gaziantep",
    "Eyupspor": "Eyupspor",
    "Genclerbirligi": "Genclerbirligi",
    "Goztepe": "Goztep",
    "Karagumruk": "Karagumruk",
    "Fatih Karagumruk": "Karagumruk",
    "Kocaelispor": "Kocaelispor",
    "Adana Demirspor": "Adana Demirspor",
    "Sivasspor": "Sivasspor",
    "Giresunspor": "Giresunspor",
    "Umraniyespor": "Umraniyespor",
    "Hatayspor": "Hatayspor",
    "Ankaragücü": "Ankaragucu",
    "Ankaragucu": "Ankaragucu",
    "Pendikspor": "Pendikspor",
    "Istanbulspor": "Istanbulspor",
    "Bodrum FK": "Bodrum",
    "Erzurumspor": "Erzurumspor",
}


def normalize_fd_name(name):
    """Normalize football-data.co.uk team name to our ASCII format."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return FD_NAME_MAP.get(name, name)


def load_match_data():
    """Load and combine all seasons from football-data.co.uk."""
    all_dfs = []
    for season in SEASONS:
        url = BASE_URL.format(season)
        try:
            df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
            # Standardise column names
            col_map = {}
            for c in df.columns:
                cl = c.strip().lower()
                if cl in ("hometeam", "home"):
                    col_map[c] = "home"
                elif cl in ("awayteam", "away"):
                    col_map[c] = "away"
                elif cl in ("fthg", "hg"):
                    col_map[c] = "hg"
                elif cl in ("ftag", "ag"):
                    col_map[c] = "ag"
                elif cl in ("ftr", "res", "result"):
                    col_map[c] = "result"
                elif cl == "date":
                    col_map[c] = "date"
            df = df.rename(columns=col_map)
            needed = {"home", "away", "hg", "ag", "result", "date"}
            if not needed.issubset(df.columns):
                print(f"  [WARN] Season {season}: missing columns {needed - set(df.columns)}")
                continue
            df["season"] = season
            df = df[["date", "home", "away", "hg", "ag", "result", "season"]].copy()
            df["home"] = df["home"].apply(normalize_fd_name)
            df["away"] = df["away"].apply(normalize_fd_name)
            df = df.dropna(subset=["home", "away", "result"])
            df["hg"] = pd.to_numeric(df["hg"], errors="coerce")
            df["ag"] = pd.to_numeric(df["ag"], errors="coerce")
            df = df.dropna(subset=["hg", "ag"])
            df["hg"] = df["hg"].astype(int)
            df["ag"] = df["ag"].astype(int)
            all_dfs.append(df)
            print(f"  Loaded season {season}: {len(df)} matches")
        except Exception as e:
            print(f"  [ERROR] Season {season}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ─── Phase 2: AMV from Transfermarkt ─────────────────────────────────────────

def load_amv_data():
    """
    Build AMV ratio lookup: {game_id: {club_id: {'xi_value', 'squad_value', 'amv_ratio'}}}
    Also build squad_value per team: {club_id: {'squad_value', 'date'}} for fallback.
    """
    print("\n[Phase 2] Loading AMV data from Transfermarkt...")

    games_df = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    lineups_df = pd.read_csv(
        os.path.join(DATA_DIR, "game_lineups.csv"), low_memory=False
    )
    vals_df = pd.read_csv(os.path.join(DATA_DIR, "player_valuations.csv"))

    # Filter to TR1 2025 season
    tr1_games = games_df[
        (games_df["competition_id"] == "TR1") & (games_df["season"] == 2025)
    ].copy()
    tr1_games["date"] = pd.to_datetime(tr1_games["date"])
    tr1_game_ids = set(tr1_games["game_id"].tolist())

    # Parse matchday number from round string  "N. Matchday" → N
    def parse_matchday(r):
        try:
            return int(str(r).split(".")[0].strip())
        except Exception:
            return None

    tr1_games["matchday"] = tr1_games["round"].apply(parse_matchday)

    print(f"  TR1 2025 games: {len(tr1_games)}")

    # Filter lineups to TR1 2025
    tr1_lineups = lineups_df[lineups_df["game_id"].isin(tr1_game_ids)].copy()
    tr1_lineups["date"] = pd.to_datetime(tr1_lineups["date"])
    print(f"  TR1 2025 lineup entries: {len(tr1_lineups)}")

    # Parse valuations
    vals_df["date"] = pd.to_datetime(vals_df["date"])
    vals_df["market_value_in_eur"] = pd.to_numeric(
        vals_df["market_value_in_eur"], errors="coerce"
    ).fillna(0)
    vals_df["current_club_id"] = pd.to_numeric(
        vals_df["current_club_id"], errors="coerce"
    )
    vals_df = vals_df.sort_values("date")

    # Helper: get player value on or before a given date
    def player_value_on_date(player_id, match_date):
        pv = vals_df[
            (vals_df["player_id"] == player_id) & (vals_df["date"] <= match_date)
        ]
        if pv.empty:
            return 0.0
        return float(pv.iloc[-1]["market_value_in_eur"])

    # Pre-index valuations by player_id for speed
    print("  Indexing player valuations...")
    player_val_idx = {}
    for pid, grp in vals_df.groupby("player_id"):
        player_val_idx[pid] = grp[["date", "market_value_in_eur", "current_club_id"]].values.tolist()
        # already sorted by date from above sort

    def fast_player_value(player_id, match_date):
        if player_id not in player_val_idx:
            return 0.0
        records = player_val_idx[player_id]
        val = 0.0
        for rec in records:
            if rec[0] <= match_date:
                val = float(rec[1]) if rec[1] else 0.0
            else:
                break
        return val

    # Build squad valuations per club per date (all players at club on that date)
    # We'll compute per-game squad values using only starting XI and total squad
    # from the lineup data (since that's what we have)

    # For squad_value fallback: use club-level total from clubs.csv total_market_value
    clubs_df = pd.read_csv(os.path.join(DATA_DIR, "clubs.csv"))
    squad_val_fallback = {}
    for _, row in clubs_df.iterrows():
        cid = row["club_id"]
        tmv = row.get("total_market_value", None)
        if pd.notna(tmv) and tmv:
            # Parse "€150.00m" or similar
            tmv_str = str(tmv).replace("€", "").replace(",", "").strip()
            try:
                if "m" in tmv_str.lower():
                    val = float(tmv_str.lower().replace("m", "")) * 1_000_000
                elif "k" in tmv_str.lower():
                    val = float(tmv_str.lower().replace("k", "")) * 1_000
                else:
                    val = float(tmv_str)
                squad_val_fallback[cid] = val
            except Exception:
                pass

    # Main AMV computation per game
    amv_data = {}  # {game_id: {club_id: {xi_value, squad_value, amv_ratio}}}

    games_processed = 0
    for _, game in tr1_games.iterrows():
        game_id = game["game_id"]
        match_date = game["date"]
        home_id = int(game["home_club_id"])
        away_id = int(game["away_club_id"])

        game_lineups = tr1_lineups[tr1_lineups["game_id"] == game_id]

        amv_data[game_id] = {}

        for club_id in [home_id, away_id]:
            club_lu = game_lineups[game_lineups["club_id"] == club_id]
            starters = club_lu[club_lu["type"] == "starting_lineup"]
            all_squad = club_lu  # all lineup entries (starters + subs)

            if len(starters) == 0:
                # No lineup data — use fallback squad value with ratio=1.0
                sv = squad_val_fallback.get(club_id, 0.0)
                amv_data[game_id][club_id] = {
                    "xi_value": 0.0,
                    "squad_value": sv,
                    "amv_ratio": 1.0,
                    "has_lineup": False,
                }
                continue

            xi_value = sum(
                fast_player_value(int(pid), match_date)
                for pid in starters["player_id"].tolist()
            )
            squad_value = sum(
                fast_player_value(int(pid), match_date)
                for pid in all_squad["player_id"].tolist()
            )

            if squad_value > 0:
                amv_ratio = xi_value / squad_value
            else:
                amv_ratio = 1.0

            amv_data[game_id][club_id] = {
                "xi_value": xi_value,
                "squad_value": squad_value,
                "amv_ratio": amv_ratio,
                "has_lineup": True,
            }

        games_processed += 1

    games_with_lineup = sum(
        1
        for gd in amv_data.values()
        if any(v.get("has_lineup", False) for v in gd.values())
    )
    print(f"  Processed {games_processed} games, {games_with_lineup} with lineup data")

    # Build gameday → {club_id: squad_value} for strength comparison
    gd_squad_values = defaultdict(dict)
    for _, game in tr1_games.iterrows():
        game_id = game["game_id"]
        md = game["matchday"]
        if game_id in amv_data:
            for cid, info in amv_data[game_id].items():
                sv = info["squad_value"]
                if sv > 0:
                    gd_squad_values[md][cid] = sv

    # Also build: (matchday, club_id) → xi_value for the actual game
    game_xi = {}  # (matchday, club_id) → xi_value
    for _, game in tr1_games.iterrows():
        game_id = game["game_id"]
        md = game["matchday"]
        if game_id in amv_data:
            for cid, info in amv_data[game_id].items():
                game_xi[(md, cid)] = info

    return amv_data, tr1_games, gd_squad_values, game_xi


# ─── Phase 3: Motivation scores ──────────────────────────────────────────────

def motivation_score(pts, pts_4th, pts_6th, pts_16th):
    """Compute motivation score (0–1) for a team given their points and threshold points."""
    base = 0.4

    # Title / CL fight (top 4)
    if pts >= pts_4th - 5:
        title_boost = min(1.0, (pts - (pts_4th - 5)) / 5) * 0.5
    else:
        title_boost = 0.0

    # Europe fight (5th/6th)
    if (pts_6th - 5) <= pts < (pts_4th - 5):
        europe_boost = min(1.0, (pts - (pts_6th - 5)) / 5) * 0.3
    else:
        europe_boost = 0.0

    # Relegation fight
    if pts <= pts_16th + 6:
        survival_boost = min(1.0, (pts_16th + 6 - pts) / 6) * 0.5
    else:
        survival_boost = 0.0

    return min(1.0, base + title_boost + europe_boost + survival_boost)


def build_motivation_scores(all_matches_df, dc_preds):
    """
    Build motivation scores {(gameday, team): float} using standing before each gameday.
    Uses match results from football-data.co.uk 2025/26 season data.
    """
    print("\n[Phase 3] Computing motivation scores...")

    # Get current season (2526) matches only
    current = all_matches_df[all_matches_df["season"] == "2526"].copy()
    print(f"  Current season (2526) matches: {len(current)}")

    # Map from our team names to ensure coverage
    dc_teams = set()
    for p in dc_preds:
        dc_teams.add(p["home"])
        dc_teams.add(p["away"])

    # We need to assign gameday numbers to the football-data matches.
    # Use the dc_predictions to get gameday ordering — matches in GD N in dc_preds
    # correspond to specific home/away combos. We'll order current season matches
    # by date and group into matchdays based on dc_preds.

    # Build gameday → set of (home, away) from dc_preds
    gd_matches = defaultdict(set)
    for p in dc_preds:
        gd_matches[p["gameday"]].add((p["home"], p["away"]))

    # Build standings incrementally
    # Standing entry: {team: {pts, gf, ga, gd, played}}
    standings = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0})

    # We need to process results in order of gameday
    # Map results from football-data to our team names (already normalized via FD_NAME_MAP)
    # Build a lookup: (home, away) → result for current season
    fd_result_lookup = {}
    for _, row in current.iterrows():
        key = (str(row["home"]).strip(), str(row["away"]).strip())
        fd_result_lookup[key] = {
            "hg": row["hg"],
            "ag": row["ag"],
            "result": row["result"],
        }

    motivation = {}  # {(gameday, team): score}
    max_gd = max(p["gameday"] for p in dc_preds)

    for gd in range(1, max_gd + 1):
        # Compute standings snapshot BEFORE this gameday
        # Sort standings
        teams_in_standings = list(standings.keys())

        if len(teams_in_standings) < 4:
            # Early gamedays — use neutral
            for p in dc_preds:
                if p["gameday"] == gd:
                    motivation[(gd, p["home"])] = 0.5
                    motivation[(gd, p["away"])] = 0.5
        else:
            # Build standings DataFrame
            rows = []
            for t, s in standings.items():
                rows.append({"team": t, "pts": s["pts"], "gd": s["gd"],
                             "gf": s["gf"], "played": s["played"]})
            std_df = pd.DataFrame(rows).sort_values(
                ["pts", "gd", "gf"], ascending=False
            ).reset_index(drop=True)

            n = len(std_df)
            pts_4th  = std_df.iloc[min(3, n-1)]["pts"]
            pts_6th  = std_df.iloc[min(5, n-1)]["pts"]
            pts_16th = std_df.iloc[min(15, n-1)]["pts"]

            # Compute motivation for all teams in this gameday
            for p in dc_preds:
                if p["gameday"] == gd:
                    for team in [p["home"], p["away"]]:
                        team_row = std_df[std_df["team"] == team]
                        if team_row.empty:
                            motivation[(gd, team)] = 0.5
                        else:
                            pts = team_row.iloc[0]["pts"]
                            motivation[(gd, team)] = motivation_score(
                                pts, pts_4th, pts_6th, pts_16th
                            )

        # Now update standings with results from this gameday
        gd_pairs = gd_matches[gd]
        for home, away in gd_pairs:
            res_info = fd_result_lookup.get((home, away))
            if res_info is None:
                # Try common alternative spellings
                continue
            hg = res_info["hg"]
            ag = res_info["ag"]
            result = res_info["result"]

            standings[home]["gf"] += hg
            standings[home]["ga"] += ag
            standings[home]["gd"] += hg - ag
            standings[home]["played"] += 1

            standings[away]["gf"] += ag
            standings[away]["ga"] += hg
            standings[away]["gd"] += ag - hg
            standings[away]["played"] += 1

            if result == "H":
                standings[home]["pts"] += 3
            elif result == "D":
                standings[home]["pts"] += 1
                standings[away]["pts"] += 1
            elif result == "A":
                standings[away]["pts"] += 3

    applied = sum(1 for p in dc_preds if (p["gameday"], p["home"]) in motivation)
    print(f"  Motivation scores computed for {applied} home-team entries")
    return motivation


# ─── Phase 3: Apply modifiers ─────────────────────────────────────────────────

def apply_modifiers(dc_preds, amv_data, tr1_games, game_xi, motivation):
    """Apply AMV and motivation modifiers to DC Phase 1 probabilities."""

    AMV_WEIGHT = 0.25
    MOTIVE_WEIGHT = 0.20

    # Build lookup: (matchday, home_club_id, away_club_id) → amv_data entry
    # And (matchday_num) → list of (home_club_id, away_club_id, game_id)
    gd_game_lookup = {}  # (matchday, home_club_id, away_club_id) → game_id

    def parse_md(r):
        try:
            return int(str(r).split(".")[0].strip())
        except Exception:
            return None

    for _, g in tr1_games.iterrows():
        md = parse_md(g["round"])
        if md is not None:
            gd_game_lookup[(md, int(g["home_club_id"]), int(g["away_club_id"]))] = int(g["game_id"])

    results = []
    amv_covered = 0
    motive_applied = 0

    # Pre-build squad value by gameday for relative strength comparison
    # Use median squad value across all teams per gameday as normalisation base
    gd_median_squad = {}
    for gd in range(1, 30):
        svs = [info["squad_value"]
               for (md, cid), info in game_xi.items()
               if md == gd and info["squad_value"] > 0]
        gd_median_squad[gd] = float(np.median(svs)) if svs else 1.0

    for p in dc_preds:
        gd = p["gameday"]
        home = p["home"]
        away = p["away"]
        prob_H = p["prob_H"]
        prob_D = p["prob_D"]
        prob_A = p["prob_A"]
        actual = p["actual"]

        home_tm_id = CSV_TO_TM_ID.get(home)
        away_tm_id = CSV_TO_TM_ID.get(away)

        # ── AMV ratio ──────────────────────────────────────────────────────────
        home_amv = 1.0
        away_amv = 1.0
        has_amv = False

        if home_tm_id and away_tm_id:
            game_id = gd_game_lookup.get((gd, home_tm_id, away_tm_id))
            if game_id and game_id in amv_data:
                gd_amv = amv_data[game_id]
                h_info = gd_amv.get(home_tm_id, {})
                a_info = gd_amv.get(away_tm_id, {})

                if h_info.get("has_lineup") or a_info.get("has_lineup"):
                    has_amv = True
                    amv_covered += 1

                    h_xi = h_info.get("xi_value", 0.0) or 0.0
                    a_xi = a_info.get("xi_value", 0.0) or 0.0
                    h_sq = h_info.get("squad_value", 0.0) or 0.0
                    a_sq = a_info.get("squad_value", 0.0) or 0.0

                    # Use squad value as overall strength proxy for comparison
                    med = gd_median_squad.get(gd, 1.0) or 1.0

                    # If we have actual XI values, use XI ratio relative to squad
                    # If both have lineup: amv = xi / squad for each
                    # If not, fall back to squad_value / median
                    if h_info.get("has_lineup") and h_sq > 0:
                        home_amv = h_xi / h_sq  # fractional XI quality
                    elif h_sq > 0:
                        home_amv = h_sq / med

                    if a_info.get("has_lineup") and a_sq > 0:
                        away_amv = a_xi / a_sq
                    elif a_sq > 0:
                        away_amv = a_sq / med

                    # Ensure no zero
                    if home_amv <= 0:
                        home_amv = 1.0
                    if away_amv <= 0:
                        away_amv = 1.0
            else:
                # No game found in TM data for this GD — use per-team squad value
                h_info = game_xi.get((gd, home_tm_id))
                a_info = game_xi.get((gd, away_tm_id))
                med = gd_median_squad.get(gd, 1.0) or 1.0
                if h_info and h_info["squad_value"] > 0:
                    home_amv = h_info["squad_value"] / med
                    has_amv = True
                if a_info and a_info["squad_value"] > 0:
                    away_amv = a_info["squad_value"] / med
                    if has_amv:
                        amv_covered += 1
                    else:
                        has_amv = True
                        amv_covered += 1

        # Normalize AMV so both sides are relative to their mean
        amv_mean = (home_amv + away_amv) / 2.0
        if amv_mean > 0:
            home_amv_norm = home_amv / amv_mean
            away_amv_norm = away_amv / amv_mean
        else:
            home_amv_norm = 1.0
            away_amv_norm = 1.0

        # ── Motivation ────────────────────────────────────────────────────────
        home_motive = motivation.get((gd, home), 0.5)
        away_motive = motivation.get((gd, away), 0.5)
        if (gd, home) in motivation:
            motive_applied += 1

        # ── Compute home_edge ─────────────────────────────────────────────────
        home_edge = (
            AMV_WEIGHT * (home_amv_norm - 1.0)
            + MOTIVE_WEIGHT * (home_motive - away_motive)
        )

        # ── Apply logit adjustment to H vs A ──────────────────────────────────
        adj_prob_H, adj_prob_A = prob_H, prob_A

        if prob_H > 0 and prob_A > 0:
            logit_ha = math.log(prob_H / prob_A) + home_edge * 2.0
            ratio = math.exp(logit_ha)
            ha_pool = prob_H + prob_A
            adj_prob_H = ha_pool * ratio / (1.0 + ratio)
            adj_prob_A = ha_pool - adj_prob_H
        elif prob_H == 0 and prob_A > 0:
            adj_prob_H = 0.0
            adj_prob_A = prob_A
        elif prob_A == 0 and prob_H > 0:
            adj_prob_H = prob_H
            adj_prob_A = 0.0

        # Renormalise (keep prob_D stable)
        total = adj_prob_H + prob_D + adj_prob_A
        if total > 0:
            adj_prob_H /= total
            adj_prob_D  = prob_D / total
            adj_prob_A /= total
        else:
            adj_prob_H, adj_prob_D, adj_prob_A = prob_H, prob_D, prob_A

        # Predicted result
        probs = {"H": adj_prob_H, "D": adj_prob_D, "A": adj_prob_A}
        predicted = max(probs, key=probs.get)
        correct = 1 if predicted == actual else 0

        results.append({
            "gameday": gd,
            "home": home,
            "away": away,
            "prob_H": round(adj_prob_H, 4),
            "prob_D": round(adj_prob_D, 4),
            "prob_A": round(adj_prob_A, 4),
            "dc_prob_H": round(prob_H, 4),
            "dc_prob_D": round(prob_D, 4),
            "dc_prob_A": round(prob_A, 4),
            "home_amv_norm": round(home_amv_norm, 4),
            "away_amv_norm": round(away_amv_norm, 4),
            "home_motive": round(home_motive, 4),
            "away_motive": round(away_motive, 4),
            "home_edge": round(home_edge, 4),
            "predicted": predicted,
            "dc_predicted": p["predicted"],
            "actual": actual,
            "correct": correct,
            "dc_correct": p["correct"],
            "has_amv": has_amv,
        })

    print(f"  AMV coverage: {amv_covered}/{len(dc_preds)} matches")
    print(f"  Motivation applied: {motive_applied}/{len(dc_preds)} matches")
    return results, amv_covered, motive_applied


# ─── Report ──────────────────────────────────────────────────────────────────

def print_report(results, dc_preds, amv_covered, motive_applied):
    """Print comparison table and summary."""

    # Load SPI weekly accuracy
    spi_weekly = {}
    spi_overall_pct = 46.4
    try:
        with open(DASHBOARD_FILE) as f:
            dash = json.load(f)
        spi_overall_pct = dash["accuracy"].get("overall_pct", 46.4)
        for entry in dash["accuracy"].get("weekly", []):
            label = entry["week_label"]
            if label.startswith("GD "):
                gd_num = int(label.replace("GD ", "").strip())
                spi_weekly[gd_num] = entry["weekly_pct"]
    except Exception as e:
        print(f"  [WARN] Could not load dashboard.json: {e}")

    # Compute per-GD stats
    # Group dc_preds by gameday
    dc_gd = defaultdict(list)
    for p in dc_preds:
        dc_gd[p["gameday"]].append(p)

    p2_gd = defaultdict(list)
    for r in results:
        p2_gd[r["gameday"]].append(r)

    max_gd = max(p2_gd.keys())

    print("\n" + "=" * 85)
    print(f"{'GD':>3}  {'N':>3}  {'DC%':>7}  {'Phase2%':>8}  {'SPI%':>7}  {'Δ(DC)':>8}  {'Δ(SPI)':>8}")
    print("-" * 85)

    for gd in range(1, max_gd + 1):
        dc_list = dc_gd[gd]
        p2_list = p2_gd[gd]
        n = len(p2_list)
        if n == 0:
            continue

        dc_pct  = 100 * sum(p["correct"] for p in dc_list) / n
        p2_pct  = 100 * sum(r["correct"] for r in p2_list) / n
        spi_pct = spi_weekly.get(gd, float("nan"))

        d_dc  = p2_pct - dc_pct
        d_spi = p2_pct - spi_pct if not math.isnan(spi_pct) else float("nan")

        spi_str = f"{spi_pct:7.1f}%" if not math.isnan(spi_pct) else f"{'N/A':>7}"
        d_spi_str = f"{d_spi:+8.1f}pp" if not math.isnan(d_spi) else f"{'N/A':>8}"

        print(
            f"{gd:>3}  {n:>3}  {dc_pct:7.1f}%  {p2_pct:7.1f}%   {spi_str}  {d_dc:+8.1f}pp  {d_spi_str}"
        )

    # Overall
    total = len(results)
    dc_total_correct = sum(p["dc_correct"] for p in results)
    p2_total_correct = sum(r["correct"] for r in results)
    dc_overall_pct  = 100 * dc_total_correct / total
    p2_overall_pct  = 100 * p2_total_correct / total

    # SPI total from dashboard
    spi_total_correct = 130  # from prompt
    spi_total_n = 261
    try:
        with open(DASHBOARD_FILE) as f:
            dash = json.load(f)
        acc = dash["accuracy"]
        spi_overall_pct = acc.get("overall_pct", 46.4)
    except Exception:
        pass

    print("=" * 85)
    print("\nOverall accuracy:")
    print(f"  SPI model:    {spi_overall_pct:.1f}%  ({spi_total_correct}/{spi_total_n} over {len(spi_weekly)} GDs)")
    print(f"  DC Phase 1:   {dc_overall_pct:.1f}%  ({dc_total_correct}/{total} over {max_gd} GDs)")
    print(f"  DC + Phase2:  {p2_overall_pct:.1f}%  ({p2_total_correct}/{total} over {max_gd} GDs)")
    print(f"  Δ vs DC:      {p2_overall_pct - dc_overall_pct:+.1f}pp")
    print(f"  Δ vs SPI:     {p2_overall_pct - spi_overall_pct:+.1f}pp")
    print()
    print(f"AMV coverage:       {amv_covered}/{total} matches ({100*amv_covered/total:.1f}%)")
    print(f"Motivation applied: {motive_applied}/{total} matches ({100*motive_applied/total:.1f}%)")

    # Draw prediction stats
    dc_draws_pred   = sum(1 for p in dc_preds  if p["predicted"] == "D")
    p2_draws_pred   = sum(1 for r in results   if r["predicted"] == "D")
    actual_draws    = sum(1 for r in results   if r["actual"] == "D")
    dc_draw_tp  = sum(1 for p in dc_preds  if p["predicted"] == "D" and p["actual"] == "D")
    p2_draw_tp  = sum(1 for r in results   if r["predicted"] == "D" and r["actual"] == "D")

    dc_draw_recall = 100 * dc_draw_tp / actual_draws if actual_draws else 0
    p2_draw_recall = 100 * p2_draw_tp / actual_draws if actual_draws else 0

    print()
    print("Draw prediction:")
    print(f"  Actual draws:  {actual_draws}")
    print(f"  Phase 1:  {dc_draws_pred} draws predicted  (recall {dc_draw_recall:.1f}%,  TP={dc_draw_tp})")
    print(f"  Phase 2:  {p2_draws_pred} draws predicted  (recall {p2_draw_recall:.1f}%,  TP={p2_draw_tp})")

    # Changed predictions breakdown
    changed = [r for r in results if r["predicted"] != r["dc_predicted"]]
    changed_better = [r for r in changed if r["correct"] == 1 and r["dc_correct"] == 0]
    changed_worse  = [r for r in changed if r["correct"] == 0 and r["dc_correct"] == 1]
    changed_same   = [r for r in changed if r["correct"] == r["dc_correct"]]

    print()
    print(f"Predictions changed: {len(changed)}/{total}")
    print(f"  Improved: {len(changed_better)}")
    print(f"  Worsened: {len(changed_worse)}")
    print(f"  Same correctness: {len(changed_same)}")

    return p2_overall_pct, dc_overall_pct, spi_overall_pct


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Dixon-Coles Phase 2 + Motivation Backtest")
    print("=" * 60)

    # Load DC predictions
    with open(DC_PREDS_FILE) as f:
        dc_preds = json.load(f)
    print(f"\nLoaded {len(dc_preds)} DC Phase 1 predictions (GD1–GD{max(p['gameday'] for p in dc_preds)})")

    # Load match data for motivation
    print("\n[Phase 3 prep] Loading football-data.co.uk match results...")
    all_matches = load_match_data()

    # Phase 2: AMV
    amv_data, tr1_games, gd_squad_values, game_xi = load_amv_data()

    # Phase 3: Motivation
    motivation = build_motivation_scores(all_matches, dc_preds)

    # Apply modifiers
    print("\n[Applying modifiers] Combining AMV + motivation with DC probabilities...")
    results, amv_covered, motive_applied = apply_modifiers(
        dc_preds, amv_data, tr1_games, game_xi, motivation
    )

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-match results to {OUTPUT_FILE}")

    # Print report
    print_report(results, dc_preds, amv_covered, motive_applied)


if __name__ == "__main__":
    main()
