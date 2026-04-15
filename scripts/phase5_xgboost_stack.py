#!/usr/bin/env python3
"""
Phase 5: Two-Stage XGBoost Stack
Stage 1: Dixon-Coles (walk-forward per season) → prob_H, prob_D, prob_A
Stage 2: XGBoost classifier trained on DC probs + AMV + motivation + form

Training: seasons 2122, 2223, 2324, 2425
Test    : season 2526 (walk-forward, no leakage)
"""

import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent.parent
DUCKDB_PATH = ROOT / "data" / "football.duckdb"
TM_DIR      = ROOT / "data" / "transfermarkt"
DASH_JSON   = ROOT / "docs" / "data" / "dashboard.json"
OUT_JSON    = ROOT / "scripts" / "phase5_predictions.json"

XI = 0.0018  # time-decay rate

LABEL_MAP  = {"H": 0, "D": 1, "A": 2}
LABEL_INV  = {0: "H", 1: "D", 2: "A"}

CSV_TO_TM_ID = {
    "Galatasaray": 141, "Fenerbahce": 36, "Besiktas": 114,
    "Trabzonspor": 449, "Buyuksehyr": 6890, "Kayserispor": 3205,
    "Konyaspor": 2293, "Kasimpasa": 10484, "Alanyaspor": 11282,
    "Antalyaspor": 589, "Rizespor": 126, "Samsunspor": 152,
    "Gaziantep": 2832, "Eyupspor": 7160, "Genclerbirligi": 820,
    "Goztep": 1467, "Karagumruk": 6646, "Kocaelispor": 120,
}

BIG3 = {"Galatasaray", "Fenerbahce", "Besiktas"}

SEASON_CODES = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MATCH RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def load_season(code: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{code}/T1.csv"
    df = pd.read_csv(url, encoding="latin1")
    df = df.rename(columns={"HomeTeam":"home","AwayTeam":"away","FTHG":"hg","FTAG":"ag","FTR":"result","Date":"date_str"})
    df["date"] = pd.to_datetime(df["date_str"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["result","hg","ag","date"])
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["season"] = code
    return df[["season","date","home","away","hg","ag","result"]].copy()

def load_all_seasons():
    frames = []
    for code in SEASON_CODES:
        print(f"  {code}...", end="", flush=True)
        df = load_season(code)
        print(f" {len(df)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

def load_schedule():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df  = con.execute("SELECT round_number, home_team, away_team FROM schedule_2526 ORDER BY round_number").fetchdf()
    con.close()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. AMV DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_amv_data():
    """Return:
      - squad_value: {(club_id, year_month_str) -> total_squad_value_eur}
      - game_xi: {game_id -> {club_id -> xi_value}}
    """
    print("  Loading Transfermarkt AMV data...")
    pv   = pd.read_csv(TM_DIR / "player_valuations.csv", low_memory=False)
    gl   = pd.read_csv(TM_DIR / "game_lineups.csv", low_memory=False)
    gm   = pd.read_csv(TM_DIR / "games.csv", low_memory=False)

    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    pv = pv.dropna(subset=["date","market_value_in_eur","current_club_id"])
    pv["current_club_id"] = pv["current_club_id"].astype(int)
    pv = pv.sort_values("date")

    # Build per-player latest valuation lookup: (player_id, date) -> value+club
    # For each game, we want: player's value as of game date
    # Efficient approach: group by player_id, store sorted (date, value, club_id) arrays
    print("  Indexing player valuations...", end="", flush=True)
    player_hist = {}
    for pid, grp in pv.groupby("player_id"):
        grp = grp.sort_values("date")
        player_hist[pid] = (grp["date"].values, grp["market_value_in_eur"].values, grp["current_club_id"].values)
    print(f" {len(player_hist)} players")

    def get_player_value_and_club(player_id, match_date):
        if player_id not in player_hist:
            return 0.0, None
        dates, values, clubs = player_hist[player_id]
        idx = np.searchsorted(dates, np.datetime64(match_date), side="right") - 1
        if idx < 0:
            return 0.0, None
        return float(values[idx]), int(clubs[idx])

    # Filter TR1 games
    tr1_gm = gm[gm["competition_id"] == "TR1"].copy()
    tr1_gm["date"] = pd.to_datetime(tr1_gm["date"], errors="coerce")

    # For each TR1 game, compute XI values
    print(f"  Computing XI values for {len(tr1_gm)} TR1 games...", end="", flush=True)
    gl_tr1 = gl[gl["game_id"].isin(tr1_gm["game_id"])]
    gl_tr1 = gl_tr1[gl_tr1["type"] == "starting_lineup"].copy()

    game_date_map = tr1_gm.set_index("game_id")["date"].to_dict()
    game_clubs    = tr1_gm.set_index("game_id")[["home_club_id","away_club_id"]].to_dict("index")

    game_xi = defaultdict(lambda: defaultdict(float))
    game_squad = defaultdict(lambda: defaultdict(float))  # squad values per club per game

    for _, row in gl_tr1.iterrows():
        gid      = row["game_id"]
        pid      = row["player_id"]
        club_id  = row.get("club_id")
        if pd.isna(gid) or pd.isna(pid): continue
        gdate = game_date_map.get(gid)
        if gdate is None or pd.isna(gdate): continue

        val, val_club = get_player_value_and_club(int(pid), gdate)
        if val > 0 and club_id is not None and not pd.isna(club_id):
            game_xi[int(gid)][int(club_id)] += val

    # Squad values: all players (starting + subs) per club per game
    gl_all_tr1 = gl[gl["game_id"].isin(tr1_gm["game_id"])].copy()
    for _, row in gl_all_tr1.iterrows():
        gid     = row["game_id"]
        pid     = row["player_id"]
        club_id = row.get("club_id")
        if pd.isna(gid) or pd.isna(pid): continue
        gdate = game_date_map.get(gid)
        if gdate is None or pd.isna(gdate): continue
        val, _ = get_player_value_and_club(int(pid), gdate)
        if val > 0 and club_id is not None and not pd.isna(club_id):
            game_squad[int(gid)][int(club_id)] += val

    print(f" done ({len(game_xi)} games with XI data)")

    # Build lookup: (home_csv, away_csv, date_str) -> (home_amv, away_amv, xi_ratio)
    # Map TR1 games to CSV team names via CSV_TO_TM_ID reverse
    TM_TO_CSV = {v: k for k, v in CSV_TO_TM_ID.items()}

    amv_lookup = {}
    for _, row in tr1_gm.iterrows():
        gid  = int(row["game_id"])
        hcid = int(row["home_club_id"]) if not pd.isna(row["home_club_id"]) else None
        acid = int(row["away_club_id"]) if not pd.isna(row["away_club_id"]) else None
        gdate = row["date"]
        if pd.isna(gdate) or hcid is None or acid is None: continue

        h_xi  = game_xi[gid].get(hcid, 0)
        a_xi  = game_xi[gid].get(acid, 0)
        h_sq  = game_squad[gid].get(hcid, 0)
        a_sq  = game_squad[gid].get(acid, 0)

        h_amv = h_xi / h_sq if h_sq > 0 else 0.85
        a_amv = a_xi / a_sq if a_sq > 0 else 0.85

        h_csv = TM_TO_CSV.get(hcid)
        a_csv = TM_TO_CSV.get(acid)
        if h_csv and a_csv:
            key = (h_csv, a_csv, str(gdate.date()))
            amv_lookup[key] = {
                "home_xi": h_xi, "away_xi": a_xi,
                "home_sq": h_sq, "away_sq": a_sq,
                "home_amv": h_amv, "away_amv": a_amv,
            }

    print(f"  AMV lookup: {len(amv_lookup)} matches keyed")
    return amv_lookup


# ─────────────────────────────────────────────────────────────────────────────
# 3. MOTIVATION + FORM + STANDINGS
# ─────────────────────────────────────────────────────────────────────────────

def motivation_score(pts, standings_sorted):
    n = len(standings_sorted)
    pts_4th  = standings_sorted[3]["pts"] if n > 3  else 0
    pts_6th  = standings_sorted[5]["pts"] if n > 5  else 0
    pts_16th = standings_sorted[15]["pts"] if n > 15 else 0
    base = 0.35
    title_boost = max(0, min(0.55, (pts - (pts_4th - 6)) / 6 * 0.55)) if pts >= pts_4th - 6 else 0
    euro_boost  = max(0, min(0.30, (pts - (pts_6th - 5)) / 5 * 0.30)) if pts_4th - 6 > pts >= pts_6th - 5 else 0
    surv_boost  = max(0, min(0.55, (pts_16th + 7 - pts) / 7 * 0.55)) if pts <= pts_16th + 7 else 0
    return min(1.0, base + title_boost + euro_boost + surv_boost)

def compute_season_features(season_df: pd.DataFrame, round_col="round_number", all_data: pd.DataFrame = None) -> list:
    """Given a season's matches (with round_number), compute all contextual features.
    all_data: full historical DataFrame used for cross-season h2h lookup (no leakage).
    """
    records = []
    teams = sorted(set(season_df["home"]) | set(season_df["away"]))
    standings = {t: {"pts": 0, "gf": 0, "ga": 0, "played": 0} for t in teams}
    results_by_team = defaultdict(list)  # team -> [(date, pts, gf, ga)]
    season_code = season_df["season"].iloc[0] if "season" in season_df.columns else None

    # Sort rounds
    for rn in sorted(season_df[round_col].unique()):
        round_matches = season_df[season_df[round_col] == rn]

        # Build sorted standings before this round
        sorted_standings = sorted(standings.values(), key=lambda x: (x["pts"], x.get("gf",0)-x.get("ga",0)), reverse=True)
        # Add rank info
        standing_list = []
        for t in sorted(standings.keys(), key=lambda t: (standings[t]["pts"], standings[t]["gf"]-standings[t]["ga"]), reverse=True):
            standing_list.append({"team": t, "pts": standings[t]["pts"]})

        for _, m in round_matches.iterrows():
            ht, at = m["home"], m["away"]

            # Standings
            h_pts = standings[ht]["pts"] if ht in standings else 0
            a_pts = standings[at]["pts"] if at in standings else 0
            h_played = standings[ht]["played"] if ht in standings else 0
            a_played = standings[at]["played"] if at in standings else 0
            h_rank = next((i+1 for i, s in enumerate(standing_list) if s["team"]==ht), 9)
            a_rank = next((i+1 for i, s in enumerate(standing_list) if s["team"]==at), 9)

            # Form: last 5 results for each team
            def form_stats(team):
                hist = results_by_team[team][-5:]
                if not hist: return 0.0, 0.0
                pts_pg = sum(h[1] for h in hist) / len(hist)
                gd_pg  = sum(h[2]-h[3] for h in hist) / len(hist)
                return pts_pg, gd_pg

            h_fpts, h_fgd = form_stats(ht)
            a_fpts, a_fgd = form_stats(at)

            # H2H: last 5 meetings BEFORE this match (no leakage)
            # Use prior seasons + current season matches before this date
            m_date = m["date"]
            cur_season_past = season_df[season_df["date"] < m_date]
            if all_data is not None and season_code is not None:
                prior_data = all_data[all_data["season"] < season_code]
                h2h_pool = pd.concat([prior_data, cur_season_past], ignore_index=True)
            else:
                h2h_pool = cur_season_past
            h2h_mask = (
                ((h2h_pool["home"]==ht) & (h2h_pool["away"]==at)) |
                ((h2h_pool["home"]==at) & (h2h_pool["away"]==ht))
            )
            h2h = [(r["result"], r["home"]) for _, r in h2h_pool[h2h_mask].sort_values("date").tail(5).iterrows()]
            h_h2h_wins = sum(1 for res, hm in h2h if (res=="H" and hm==ht) or (res=="A" and hm!=ht)) / max(len(h2h),1)

            # Motivation
            h_motive = motivation_score(h_pts, standing_list)
            a_motive = motivation_score(a_pts, standing_list)

            records.append({
                "round": int(rn),
                "home": ht, "away": at,
                "date": str(m["date"].date()) if hasattr(m["date"], "date") else str(m["date"])[:10],
                "actual": m["result"],
                "home_rank": h_rank, "away_rank": a_rank,
                "home_pts_pg": round(h_pts / max(h_played,1), 3),
                "away_pts_pg": round(a_pts / max(a_played,1), 3),
                "home_form_pts": round(h_fpts, 3), "away_form_pts": round(a_fpts, 3),
                "home_form_gd": round(h_fgd, 3),  "away_form_gd": round(a_fgd, 3),
                "home_motivation": round(h_motive, 3), "away_motivation": round(a_motive, 3),
                "h2h_home_rate": round(h_h2h_wins, 3),
                "derby": int(ht in BIG3 and at in BIG3),
                "gameday": int(rn),
            })

            # Update standings AFTER recording
            res = m["result"]
            h_pts_gained = 3 if res=="H" else (1 if res=="D" else 0)
            a_pts_gained = 3 if res=="A" else (1 if res=="D" else 0)
            standings[ht]["pts"] += h_pts_gained; standings[ht]["gf"] += m["hg"]; standings[ht]["ga"] += m["ag"]; standings[ht]["played"] += 1
            standings[at]["pts"] += a_pts_gained; standings[at]["gf"] += m["ag"]; standings[at]["ga"] += m["hg"]; standings[at]["played"] += 1
            results_by_team[ht].append((m["date"], h_pts_gained, m["hg"], m["ag"]))
            results_by_team[at].append((m["date"], a_pts_gained, m["ag"], m["hg"]))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4. DC WALK-FORWARD PER SEASON
# ─────────────────────────────────────────────────────────────────────────────

def dc_walk_forward(all_data: pd.DataFrame, season_code: str, schedule_df: pd.DataFrame = None) -> dict:
    """
    Run DC walk-forward for one season. Returns dict: (home, away) -> (prob_H, prob_D, prob_A).
    For 2526, uses schedule_df to get round numbers.
    For prior seasons, round number is inferred from match order.
    """
    df_season = all_data[all_data["season"] == season_code].copy()
    df_prior  = all_data[all_data["season"] <  season_code].copy()

    if schedule_df is not None:
        # Map to round numbers via schedule
        df_merged = df_season.merge(schedule_df, left_on=["home","away"], right_on=["home_team","away_team"], how="left")
        df_merged = df_merged.dropna(subset=["round_number"])
        df_merged["round_number"] = df_merged["round_number"].astype(int)
        df_season_rounds = df_merged
        get_round = lambda df: df["round_number"]
    else:
        # Sort by date, assign round numbers (9 games per round assumed)
        df_season = df_season.sort_values("date").reset_index(drop=True)
        df_season["round_number"] = (df_season.index // 9) + 1
        df_season_rounds = df_season
        get_round = lambda df: df["round_number"]

    results = {}
    max_round = int(df_season_rounds["round_number"].max()) if len(df_season_rounds) else 0

    for rn in range(1, max_round + 1):
        # Training: all prior seasons + current season rounds < rn
        prior_curr = df_season_rounds[df_season_rounds["round_number"] < rn][["date","home","away","hg","ag"]]
        train = pd.concat([df_prior[["date","home","away","hg","ag"]], prior_curr], ignore_index=True)
        train = train.dropna(subset=["date","hg","ag"]).sort_values("date")

        if len(train) < 20:
            # Not enough data — use all current season as fallback
            train = pd.concat([df_prior[["date","home","away","hg","ag"]], df_season[["date","home","away","hg","ag"]]], ignore_index=True)
            train = train.dropna(subset=["date","hg","ag"]).sort_values("date")

        weights = dixon_coles_weights(train["date"].tolist(), xi=XI)

        try:
            model = DixonColesGoalModel(
                goals_home=train["hg"].astype(int).tolist(),
                goals_away=train["ag"].astype(int).tolist(),
                teams_home=train["home"].tolist(),
                teams_away=train["away"].tolist(),
                weights=weights,
            )
            model.fit()
            model_ok = True
        except Exception:
            model_ok = False

        # Get matches in this round
        round_matches = df_season_rounds[df_season_rounds["round_number"] == rn]
        for _, row in round_matches.iterrows():
            ht, at = row["home"], row["away"]
            if model_ok:
                try:
                    pred = model.predict(ht, at)
                    ph, pd_, pa = float(pred.home_win), float(pred.draw), float(pred.away_win)
                except Exception:
                    ph, pd_, pa = 0.4, 0.25, 0.35
            else:
                ph, pd_, pa = 0.4, 0.25, 0.35
            results[(ht, at)] = (ph, pd_, pa)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD FEATURE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(all_data, schedule_df, amv_lookup):
    """Build full feature matrix for all 5 seasons."""
    all_records = []

    for i, code in enumerate(SEASON_CODES):
        print(f"\n  Season {code}:")
        df_season = all_data[all_data["season"] == code].copy()

        # Get round numbers
        sched = schedule_df if code == "2526" else None
        if sched is not None:
            df_season_r = df_season.merge(sched, left_on=["home","away"], right_on=["home_team","away_team"], how="left")
            df_season_r = df_season_r.dropna(subset=["round_number"])
            df_season_r["round_number"] = df_season_r["round_number"].astype(int)
        else:
            df_season = df_season.sort_values("date").reset_index(drop=True)
            df_season["round_number"] = (df_season.index // 9) + 1
            df_season_r = df_season

        # Context features
        print(f"    Computing context features...", end="", flush=True)
        ctx_records = compute_season_features(df_season_r.rename(columns={"round_number":"round_number"}), "round_number", all_data=all_data)
        ctx_map = {(r["home"], r["away"]): r for r in ctx_records}
        print(f" {len(ctx_records)} matches")

        # DC walk-forward
        print(f"    DC walk-forward...", end="", flush=True)
        dc_preds = dc_walk_forward(all_data, code, schedule_df if code == "2526" else None)
        print(f" {len(dc_preds)} predictions")

        # Assemble
        for (ht, at), (ph, pd_, pa) in dc_preds.items():
            ctx = ctx_map.get((ht, at), {})
            actual = ctx.get("actual")
            if actual not in ("H","D","A"): continue
            date_str = ctx.get("date","")

            # AMV
            amv = amv_lookup.get((ht, at, date_str), {})
            h_amv = amv.get("home_amv", 0.85)
            a_amv = amv.get("away_amv", 0.85)
            h_sq  = amv.get("home_sq", 0)
            a_sq  = amv.get("away_sq", 0)
            sq_total = h_sq + a_sq
            xi_ratio = math.log((amv.get("home_xi",1)+1) / (amv.get("away_xi",1)+1)) if sq_total > 0 else 0.0
            sq_ratio  = math.log((h_sq+1) / (a_sq+1)) if sq_total > 0 else 0.0

            season_progress = ctx.get("gameday", 17) / 34.0

            all_records.append({
                "season": code,
                "gameday": ctx.get("gameday", 0),
                "date": date_str,
                "home": ht, "away": at,
                # DC Stage 1
                "prob_H": round(ph, 4),
                "prob_D": round(pd_, 4),
                "prob_A": round(pa, 4),
                "dc_predicted": max({"H":ph,"D":pd_,"A":pa}, key={"H":ph,"D":pd_,"A":pa}.get),
                # Form
                "home_form_pts": ctx.get("home_form_pts", 1.2),
                "away_form_pts": ctx.get("away_form_pts", 1.2),
                "home_form_gd":  ctx.get("home_form_gd",  0.0),
                "away_form_gd":  ctx.get("away_form_gd",  0.0),
                # Standings
                "home_rank": ctx.get("home_rank", 9),
                "away_rank": ctx.get("away_rank", 9),
                "home_pts_pg": ctx.get("home_pts_pg", 1.2),
                "away_pts_pg": ctx.get("away_pts_pg", 1.2),
                # Motivation
                "home_motivation": ctx.get("home_motivation", 0.5),
                "away_motivation": ctx.get("away_motivation", 0.5),
                # AMV
                "home_amv_ratio": round(h_amv, 4),
                "away_amv_ratio": round(a_amv, 4),
                "squad_value_ratio": round(np.clip(sq_ratio, -2, 2), 4),
                "xi_value_ratio": round(np.clip(xi_ratio, -2, 2), 4),
                # Other
                "h2h_home_rate": ctx.get("h2h_home_rate", 0.4),
                "derby": ctx.get("derby", 0),
                "season_progress": round(season_progress, 3),
                # Target
                "actual": actual,
                "label": LABEL_MAP[actual],
            })

    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────────────────────
# 6. XGBOOST WALK-FORWARD ON 2526
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "prob_H","prob_D","prob_A",
    "home_form_pts","away_form_pts","home_form_gd","away_form_gd",
    "home_rank","away_rank","home_pts_pg","away_pts_pg",
    "home_motivation","away_motivation",
    "home_amv_ratio","away_amv_ratio","squad_value_ratio","xi_value_ratio",
    "h2h_home_rate","derby","gameday","season_progress",
]

def train_xgb(X_train, y_train):
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0,
        eval_metric="mlogloss", random_state=42, verbosity=0,
        num_class=3,
    )
    model.fit(X_train, y_train)
    return model

def xgb_walkforward(df: pd.DataFrame):
    """Walk-forward XGBoost on 2526: train on prior seasons + 2526 GDs < N."""
    df_prior  = df[df["season"] != "2526"].copy()
    df_test   = df[df["season"] == "2526"].copy()

    predictions = []
    rounds = sorted(df_test["gameday"].unique())

    for rn in rounds:
        df_curr_prev = df_test[df_test["gameday"] < rn]
        train_df = pd.concat([df_prior, df_curr_prev], ignore_index=True)

        X_train = train_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].median())
        y_train = train_df["label"]

        model = train_xgb(X_train.values, y_train.values)

        round_df = df_test[df_test["gameday"] == rn]
        X_pred = round_df[FEATURE_COLS].fillna(X_train.median())
        proba  = model.predict_proba(X_pred.values)
        preds  = model.predict(X_pred.values)

        for idx, (_, row) in enumerate(round_df.iterrows()):
            pred_label = LABEL_INV[int(preds[idx])]
            ph_xgb = float(proba[idx][0])
            pd_xgb = float(proba[idx][1])
            pa_xgb = float(proba[idx][2])
            # Ensemble blend: 60% DC + 40% XGB
            ALPHA = 0.6
            ph_ens = ALPHA * row["prob_H"] + (1 - ALPHA) * ph_xgb
            pd_ens = ALPHA * row["prob_D"] + (1 - ALPHA) * pd_xgb
            pa_ens = ALPHA * row["prob_A"] + (1 - ALPHA) * pa_xgb
            ens_predicted = max({"H": ph_ens, "D": pd_ens, "A": pa_ens}, key={"H": ph_ens, "D": pd_ens, "A": pa_ens}.get)
            predictions.append({
                "gameday": int(rn),
                "home": row["home"], "away": row["away"],
                "date": row["date"],
                "prob_H_xgb": round(ph_xgb, 4),
                "prob_D_xgb": round(pd_xgb, 4),
                "prob_A_xgb": round(pa_xgb, 4),
                "prob_H_ens": round(ph_ens, 4),
                "prob_D_ens": round(pd_ens, 4),
                "prob_A_ens": round(pa_ens, 4),
                "xgb_predicted": pred_label,
                "ens_predicted": ens_predicted,
                "dc_predicted": row["dc_predicted"],
                "actual": row["actual"],
                "xgb_correct": int(pred_label == row["actual"]),
                "ens_correct":  int(ens_predicted == row["actual"]),
                "dc_correct":  int(row["dc_predicted"] == row["actual"]),
                # pass through features for inspection
                "prob_H": row["prob_H"], "prob_D": row["prob_D"], "prob_A": row["prob_A"],
                "home_motivation": row["home_motivation"], "away_motivation": row["away_motivation"],
                "home_amv_ratio": row["home_amv_ratio"], "away_amv_ratio": row["away_amv_ratio"],
            })

    return predictions, model  # return last model for feature importance


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(predictions, spi_weekly, spi_overall):
    by_gd = defaultdict(list)
    for p in predictions:
        by_gd[p["gameday"]].append(p)

    dc_total = xgb_total = ens_total = 0

    print(f"\n{'═'*88}")
    print(f"{'GD':>4}  {'XGB':>6} {'XGB%':>6}  {'ENS%':>6}  {'DC%':>6}  {'SPI%':>6}  {'Δ(ENS-DC)':>9}")
    print(f"{'─'*88}")

    for gd in sorted(by_gd.keys()):
        ms = by_gd[gd]
        n = len(ms)
        xgb_c = sum(m["xgb_correct"] for m in ms)
        ens_c = sum(m["ens_correct"] for m in ms)
        dc_c  = sum(m["dc_correct"]  for m in ms)
        dc_total  += dc_c
        xgb_total += xgb_c
        ens_total += ens_c
        xgb_pct = xgb_c/n*100
        ens_pct = ens_c/n*100
        dc_pct  = dc_c/n*100
        spi_pct = spi_weekly.get(gd)
        d_ens_dc = f"{ens_pct-dc_pct:+.1f}pp"
        spi_s = f"{spi_pct:.1f}%" if spi_pct else "  n/a"
        print(f"GD{gd:>2}  {xgb_c}/{n:<4}  {xgb_pct:>5.1f}%  {ens_pct:>5.1f}%  {dc_pct:>5.1f}%  {spi_s:>6}  {d_ens_dc:>9}")

    n_total = len(predictions)
    xgb_ov = xgb_total/n_total*100
    ens_ov = ens_total/n_total*100
    dc_ov  = dc_total/n_total*100

    print(f"{'═'*88}")
    print(f"\n{'Overall accuracy':}")
    print(f"  SPI model:        {spi_overall:.1f}%")
    print(f"  DC Phase 1:       {dc_ov:.1f}%  ({dc_total}/{n_total})")
    print(f"  XGBoost alone:    {xgb_ov:.1f}%  ({xgb_total}/{n_total})")
    print(f"  DC+XGB ensemble:  {ens_ov:.1f}%  ({ens_total}/{n_total})  [60% DC + 40% XGB]")
    print(f"  Δ(Ensemble vs DC):  {ens_ov-dc_ov:+.1f}pp")
    print(f"  Δ(Ensemble vs SPI): {ens_ov-spi_overall:+.1f}pp")

    # Confusion matrix
    actuals   = [LABEL_MAP[p["actual"]]       for p in predictions]
    xgb_preds = [LABEL_MAP[p["xgb_predicted"]] for p in predictions]
    ens_preds = [LABEL_MAP[p["ens_predicted"]] for p in predictions]
    dc_preds  = [LABEL_MAP[p["dc_predicted"]]  for p in predictions]
    cm_xgb = confusion_matrix(actuals, xgb_preds, labels=[0,1,2])
    cm_ens = confusion_matrix(actuals, ens_preds, labels=[0,1,2])
    cm_dc  = confusion_matrix(actuals, dc_preds,  labels=[0,1,2])

    actual_draws = sum(1 for a in actuals if a==1)
    print(f"\n{'Draw prediction analysis':}")
    for name, preds, cm in [("DC", dc_preds, cm_dc), ("XGB", xgb_preds, cm_xgb), ("ENS", ens_preds, cm_ens)]:
        draws_pred = sum(1 for p in preds if p==1)
        draw_tp = cm[1][1]
        print(f"  {name}: predicted {draws_pred} draws, {draw_tp} correct  (recall {draw_tp/actual_draws*100:.1f}%, precision {draw_tp/draws_pred*100:.1f}%)" if draws_pred else f"  {name}: 0 draws predicted")

    print(f"\n{'Confusion matrix — Ensemble (60%DC + 40%XGB)':}")
    print(f"              Pred H   Pred D   Pred A")
    for i, lab in enumerate(["Actual H","Actual D","Actual A"]):
        print(f"  {lab}:   {cm_ens[i][0]:>6}   {cm_ens[i][1]:>6}   {cm_ens[i][2]:>6}")


def load_spi_accuracy():
    with open(DASH_JSON) as f:
        data = json.load(f)
    spi = {int(e["week_label"].split()[1]): e["weekly_pct"] for e in data["accuracy"]["weekly"]}
    return spi, data["accuracy"]["overall_pct"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Phase 5: Two-Stage XGBoost Stack")
    print("="*60)

    print("\n[1] Loading match results...")
    all_data = load_all_seasons()
    schedule = load_schedule()
    spi_weekly, spi_overall = load_spi_accuracy()
    print(f"    SPI overall: {spi_overall}%")

    print("\n[2] Loading AMV data...")
    amv_lookup = load_amv_data()

    print("\n[3] Building feature dataset (all 5 seasons)...")
    df = build_dataset(all_data, schedule, amv_lookup)
    print(f"\n    Total records: {len(df)}  ({df['season'].value_counts().to_dict()})")
    print(f"    Features: {len(FEATURE_COLS)}")
    print(f"    Label dist: {df['actual'].value_counts().to_dict()}")

    print("\n[4] XGBoost walk-forward on 2025-26...")
    predictions, final_model = xgb_walkforward(df)
    print(f"    Generated {len(predictions)} test predictions")

    print("\n[5] Feature importance (top 10):")
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp[:10]:
        bar = "█" * int(imp * 200)
        print(f"    {feat:<22} {imp:.4f}  {bar}")

    print_report(predictions, spi_weekly, spi_overall)

    with open(OUT_JSON, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")

    feat_csv = ROOT / "scripts" / "phase5_features.csv"
    df.to_csv(feat_csv, index=False)
    print(f"Feature dataset saved to {feat_csv}")


if __name__ == "__main__":
    main()
