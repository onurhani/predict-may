#!/usr/bin/env python3
"""
Weekly gameday prediction — Trendyol Süper Lig 2025-26.
DC + XGBoost Stack + Referee Bias + EV ranking.

Usage:
    python3 scripts/gd30_predict.py              # runs with GAMEDAY below, saves MD
    python3 scripts/gd30_predict.py --gameday 31 # override gameday
    python3 scripts/gd30_predict.py --no-md      # skip Obsidian export
"""

import argparse
import json
import math
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# ▶  WEEKLY CONFIG — update these three things before each gameday run
# ════════════════════════════════════════════════════════════════════════════
GAMEDAY = 30          # ← change to 31, 32 … each week

# Bookmaker odds from screenshots.
# Key: (home_csv_name, away_csv_name)
# DC odds: 1X (home+draw), X2 (draw+away), 12 (home+away)
MATCH_ODDS = {
    ("Antalyaspor",   "Konyaspor"):   {"o_H":3.10,"o_D":3.40,"o_A":2.15, "dc_1x":1.65,"dc_x2":1.34,"dc_12":1.30},
    ("Fenerbahce",    "Rizespor"):    {"o_H":1.34,"o_D":5.25,"o_A":7.50, "dc_1x":1.09,"dc_x2":3.10,"dc_12":1.15},
    ("Karagumruk",    "Eyupspor"):    {"o_H":1.95,"o_D":3.40,"o_A":3.75, "dc_1x":1.26,"dc_x2":1.80,"dc_12":1.30},
    ("Kocaelispor",   "Goztep"):      {"o_H":2.50,"o_D":3.10,"o_A":2.80, "dc_1x":1.40,"dc_x2":1.49,"dc_12":1.35},
    ("Genclerbirligi","Galatasaray"): {"o_H":6.50,"o_D":4.75,"o_A":1.42, "dc_1x":2.80,"dc_x2":1.11,"dc_12":1.18},
    ("Kasimpasa",     "Alanyaspor"):  {"o_H":2.65,"o_D":3.20,"o_A":2.55, "dc_1x":1.47,"dc_x2":1.44,"dc_12":1.32},
    ("Trabzonspor",   "Buyuksehyr"):  {"o_H":1.95,"o_D":3.60,"o_A":3.50, "dc_1x":1.29,"dc_x2":1.78,"dc_12":1.27},
    ("Gaziantep",     "Kayserispor"): {"o_H":2.15,"o_D":3.40,"o_A":3.10, "dc_1x":1.35,"dc_x2":1.65,"dc_12":1.29},
    ("Samsunspor",    "Besiktas"):    {"o_H":3.30,"o_D":3.45,"o_A":2.10, "dc_1x":1.75,"dc_x2":1.35,"dc_12":1.29},
}

# O/U odds from screenshots. Leave entry out if not available for a match.
MATCH_OU = {
    ("Antalyaspor",   "Konyaspor"):   {"line": 2.5, "over": 1.93, "under": 1.77},
    ("Fenerbahce",    "Rizespor"):    {"line": 3.5, "over": 2.20, "under": 1.57},
    ("Karagumruk",    "Eyupspor"):    {"line": 2.5, "over": 1.95, "under": 1.75},
    ("Kocaelispor",   "Goztep"):      {"line": 2.5, "over": 2.37, "under": 1.51},
    ("Genclerbirligi","Galatasaray"): {"line": 2.5, "over": 1.58, "under": 2.20},
    ("Kasimpasa",     "Alanyaspor"):  {"line": 2.5, "over": 1.98, "under": 1.72},
    ("Trabzonspor",   "Buyuksehyr"):  {"line": 2.5, "over": 1.61, "under": 2.15},
    ("Gaziantep",     "Kayserispor"): {"line": 2.5, "over": 1.75, "under": 1.93},
    # ("Samsunspor", "Besiktas"): O/U odds not available — add when obtained
}
# ════════════════════════════════════════════════════════════════════════════

ROOT          = Path(__file__).parent.parent
DUCKDB_PATH   = ROOT / "data" / "football.duckdb"
TM_DIR        = ROOT / "data" / "transfermarkt"
FEAT_CSV      = ROOT / "scripts" / "phase5_features.csv"
REF_STATS     = ROOT / "data" / "referee_stats.json"
REF_ASSIGN    = ROOT / "data" / "referee_assignments.json"
OBSIDIAN_DIR  = Path.home() / "Documents" / "Obsidian Vault" / "Predict May"
ML_PREDS_JSON = ROOT / "scripts" / "ml_predictions.json"

XI    = 0.0018
ALPHA = 0.60   # DC weight in ensemble
BIG3  = {"Galatasaray", "Fenerbahce", "Besiktas"}

SEASON_CODES = ["1617","1718","1819","1920","2021","2122","2223","2324","2425","2526"]

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_INV = {0: "H", 1: "D", 2: "A"}

FEATURE_COLS = [
    "prob_H","prob_D","prob_A",
    "home_form_pts","away_form_pts","home_form_gd","away_form_gd",
    "home_rank","away_rank","home_pts_pg","away_pts_pg",
    "home_motivation","away_motivation",
    "home_amv_ratio","away_amv_ratio","squad_value_ratio","xi_value_ratio",
    "h2h_home_rate","derby","gameday","season_progress",
]


CSV_TO_TM_ID = {
    "Galatasaray":141,"Fenerbahce":36,"Besiktas":114,"Trabzonspor":449,
    "Buyuksehyr":6890,"Kayserispor":3205,"Konyaspor":2293,"Kasimpasa":10484,
    "Alanyaspor":11282,"Antalyaspor":589,"Rizespor":126,"Samsunspor":152,
    "Gaziantep":2832,"Eyupspor":7160,"Genclerbirligi":820,"Goztep":1467,
    "Karagumruk":6646,"Kocaelispor":120,
}

# ─── Data loading ────────────────────────────────────────────────────────────

def load_season(code):
    url = f"https://www.football-data.co.uk/mmz4281/{code}/T1.csv"
    df = pd.read_csv(url, encoding="latin1")
    df = df.rename(columns={"HomeTeam":"home","AwayTeam":"away",
                             "FTHG":"hg","FTAG":"ag","FTR":"result","Date":"date_str"})
    df["date"] = pd.to_datetime(df["date_str"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["result","hg","ag","date"])
    df[["hg","ag"]] = df[["hg","ag"]].astype(int)
    df["season"] = code
    return df[["season","date","home","away","hg","ag","result"]].copy()


def load_schedule(gameday: int):
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df  = con.execute(f"SELECT home_team, away_team FROM schedule_2526 WHERE round_number={gameday} ORDER BY home_team").fetchdf()
    con.close()
    return df


# ─── DC model ────────────────────────────────────────────────────────────────

def fit_dc(train_df):
    t = train_df.dropna(subset=["date","hg","ag"]).sort_values("date")
    w = dixon_coles_weights(t["date"].tolist(), xi=XI)
    m = DixonColesGoalModel(
        goals_home=t["hg"].tolist(), goals_away=t["ag"].tolist(),
        teams_home=t["home"].tolist(), teams_away=t["away"].tolist(), weights=w,
    )
    m.fit()
    return m


def dc_pred(model, home, away):
    """Returns (prob_H, prob_D, prob_A, pred_obj). pred_obj is None on failure."""
    try:
        p = model.predict(home, away)
        return float(p.home_win), float(p.draw), float(p.away_win), p
    except Exception:
        return None, None, None, None


def ou_probs(pred_obj, line: float):
    """
    Compute P(over line), P(under line) from the DC score matrix.
    Uses pred_obj.totals(line) which returns (p_under, push, p_over).
    """
    try:
        p_under, _, p_over = pred_obj.totals(line)
        return float(p_over), float(p_under)
    except Exception:
        return None, None


# ─── Context (standings + form) ───────────────────────────────────────────────

def motivation_score(pts, slist):
    pts_4th  = slist[3]["pts"]  if len(slist) > 3  else 0
    pts_6th  = slist[5]["pts"]  if len(slist) > 5  else 0
    pts_16th = slist[15]["pts"] if len(slist) > 15 else 0
    base = 0.35
    title = max(0, min(0.55, (pts-(pts_4th-6))/6*0.55)) if pts >= pts_4th-6 else 0
    euro  = max(0, min(0.30, (pts-(pts_6th-5))/5*0.30)) if pts_4th-6 > pts >= pts_6th-5 else 0
    surv  = max(0, min(0.55, (pts_16th+7-pts)/7*0.55)) if pts <= pts_16th+7 else 0
    return min(1.0, base + title + euro + surv)


def compute_context(df_2526, all_data):
    """Compute standings + form from all completed 2526 matches."""
    teams = sorted(set(df_2526["home"]) | set(df_2526["away"]))
    std = {t: {"pts":0,"gf":0,"ga":0,"played":0} for t in teams}
    hist = defaultdict(list)

    for _, m in df_2526.sort_values("date").iterrows():
        ht, at, res = m["home"], m["away"], m["result"]
        hp = 3 if res=="H" else (1 if res=="D" else 0)
        ap = 3 if res=="A" else (1 if res=="D" else 0)
        std[ht]["pts"]+=hp; std[ht]["gf"]+=m["hg"]; std[ht]["ga"]+=m["ag"]; std[ht]["played"]+=1
        std[at]["pts"]+=ap; std[at]["gf"]+=m["ag"]; std[at]["ga"]+=m["hg"]; std[at]["played"]+=1
        hist[ht].append((m["date"],hp,m["hg"],m["ag"]))
        hist[at].append((m["date"],ap,m["ag"],m["hg"]))

    slist = sorted([{"team":t,"pts":std[t]["pts"]} for t in std],
                   key=lambda x:(x["pts"], std[x["team"]]["gf"]-std[x["team"]]["ga"]), reverse=True)

    ctx = {}
    for t in teams:
        s = std[t]; pl = max(s["played"],1)
        rank = next((i+1 for i,x in enumerate(slist) if x["team"]==t), 9)
        motive = motivation_score(s["pts"], slist)
        last5 = hist[t][-5:]
        fpts = sum(h[1] for h in last5)/len(last5) if last5 else 1.2
        fgd  = sum(h[2]-h[3] for h in last5)/len(last5) if last5 else 0.0
        ctx[t] = {"rank":rank, "pts_pg":round(s["pts"]/pl,3),
                  "form_pts":round(fpts,3), "form_gd":round(fgd,3),
                  "motivation":round(motive,3)}
    return ctx, slist


def h2h_rate(home, away, all_data):
    mask = (((all_data["home"]==home)&(all_data["away"]==away))|
            ((all_data["home"]==away)&(all_data["away"]==home)))
    df = all_data[mask].sort_values("date").tail(5)
    if len(df)==0: return 0.4
    wins = sum(1 for _,r in df.iterrows()
               if (r["result"]=="H" and r["home"]==home) or (r["result"]=="A" and r["home"]==away))
    return round(wins/len(df),3)


# ─── AMV ────────────────────────────────────────────────────────────────────

def load_amv(feat_df):
    """
    Extract latest per-team AMV ratios from phase5_features.csv (2526 season).
    Also approximate squad_value_ratio from per-match log ratios.
    Returns:
      team_amv: {team -> amv_ratio}
      match_sq_ratio: {(home,away) -> squad_value_ratio}  (log squad value ratio)
      match_xi_ratio: {(home,away) -> xi_value_ratio}
    """
    df26 = feat_df[feat_df["season"].astype(str) == "2526"].sort_values("gameday")

    team_amv = {}
    for _, row in df26.iterrows():
        team_amv[row["home"]] = float(row["home_amv_ratio"])
        team_amv[row["away"]] = float(row["away_amv_ratio"])

    # Build per-team squad value index from log ratio + last known match
    # squad_value_ratio = log(h_squad / a_squad); we need absolute values per team
    # Approximate: use ratio from last match as proxy for new matchup
    sq_ratio_map = {}
    xi_ratio_map = {}
    for _, row in df26.iterrows():
        sq_ratio_map[(row["home"], row["away"])] = float(row["squad_value_ratio"])
        xi_ratio_map[(row["home"], row["away"])] = float(row["xi_value_ratio"])

    return team_amv, sq_ratio_map, xi_ratio_map


def squad_ratios_for_match(home, away, team_amv, sq_ratio_map, xi_ratio_map):
    """Get squad_value_ratio and xi_value_ratio for a new (home, away) pair."""
    # Direct lookup
    if (home, away) in sq_ratio_map:
        return sq_ratio_map[(home,away)], xi_ratio_map.get((home,away), 0.0)
    # Inverse lookup (reverse sign)
    if (away, home) in sq_ratio_map:
        return -sq_ratio_map[(away,home)], -xi_ratio_map.get((away,home), 0.0)
    # Approximate from amv ratios
    h_amv = team_amv.get(home, 0.85)
    a_amv = team_amv.get(away, 0.85)
    sq_r = math.log((h_amv+0.01)/(a_amv+0.01))
    return round(np.clip(sq_r,-2,2),4), 0.0


# ─── XGBoost ─────────────────────────────────────────────────────────────────

def train_xgb(feat_df):
    train = feat_df[feat_df["season"].astype(str) != "2526"].copy()
    med   = train[FEATURE_COLS].median()
    X = train[FEATURE_COLS].fillna(med).values
    y = train["label"].values
    clf = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0,
        eval_metric="mlogloss", random_state=42, verbosity=0, num_class=3,
    )
    clf.fit(X, y)
    return clf, med


# ─── Referee bias ────────────────────────────────────────────────────────────

def load_referee_data(gameday: int = 30):
    """
    Load referee_stats.json and referee_assignments.json.
    Returns (ref_stats_by_id, gd_lookup) where:
      ref_stats_by_id : {referee_id -> stats dict}
      gd_lookup       : {(home_sofascore_name, away_sofascore_name) -> ref_stats or None}
    Returns ({}, {}) if files are missing.
    """
    _gameday = gameday
    if not REF_STATS.exists() or not REF_ASSIGN.exists():
        return {}, {}

    data = json.loads(REF_STATS.read_text())
    refs = {int(k): v for k, v in data.get("referees", {}).items()}

    assignments = json.loads(REF_ASSIGN.read_text())
    gd_entries = [a for a in assignments if a.get("round") == _gameday]

    gd30_lookup = {}
    for a in gd_entries:
        ref_id = a.get("referee_id")
        gd30_lookup[(a["home"], a["away"])] = refs.get(ref_id) if ref_id else None

    return refs, gd30_lookup


# Sofascore team names → CSV names (same as fetch_schedule.py SOFASCORE_TO_DB)
SOFASCORE_TO_CSV = {
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
    "Galatasaray": "Galatasaray",
    "Antalyaspor": "Antalyaspor",
    "Trabzonspor": "Trabzonspor",
    "Konyaspor": "Konyaspor",
    "Alanyaspor": "Alanyaspor",
    "Kayserispor": "Kayserispor",
    "Samsunspor": "Samsunspor",
    "Kocaelispor": "Kocaelispor",
}
CSV_TO_SOFASCORE = {v: k for k, v in SOFASCORE_TO_CSV.items()}


def ref_for_match(csv_home: str, csv_away: str, gd30_lookup: dict):
    """Look up referee stats for a match (CSV team names → Sofascore lookup)."""
    ss_home = CSV_TO_SOFASCORE.get(csv_home, csv_home)
    ss_away = CSV_TO_SOFASCORE.get(csv_away, csv_away)
    return gd30_lookup.get((ss_home, ss_away))  # None if not assigned yet


# Strength controls how aggressively referee bias shifts probabilities.
# 0.25 means a 2x draw-bias referee shifts draw prob by ~2^0.25 ≈ +19%.
REF_BIAS_STRENGTH = 0.25


def apply_referee_bias(ph: float, pd_: float, pa: float, ref_stats: dict | None):
    """
    Softly nudge H/D/A probabilities toward this referee's historical rates.
    Returns (ph, pd_, pa) unchanged if ref_stats is None (not yet assigned).
    """
    if ref_stats is None:
        return ph, pd_, pa

    draw_bias = ref_stats.get("draw_bias", 1.0)
    home_bias = ref_stats.get("home_bias", 1.0)
    away_bias = ref_stats.get("away_bias", 1.0)
    s = REF_BIAS_STRENGTH

    ph_adj  = ph  * (home_bias ** s)
    pd_adj  = pd_ * (draw_bias ** s)
    pa_adj  = pa  * (away_bias ** s)

    total = ph_adj + pd_adj + pa_adj
    if total == 0:
        return ph, pd_, pa
    return ph_adj / total, pd_adj / total, pa_adj / total


# ─── EV ──────────────────────────────────────────────────────────────────────

def best_ev(ph, pd_, pa, dc_1x, dc_x2, dc_12):
    opts = [
        ((ph+pd_)*dc_1x-1, "1X", dc_1x, ("H","D")),
        ((pd_+pa)*dc_x2-1, "X2", dc_x2, ("D","A")),
        ((ph+pa)*dc_12-1,  "12", dc_12, ("H","A")),
    ]
    return max(opts, key=lambda x: x[0])


# ─── Markdown export ─────────────────────────────────────────────────────────

def generate_markdown(gameday, results, slist, ctx, ref_assigned, run_date):
    """Return a fully-formatted Markdown string for the Obsidian vault."""
    ev_games  = [r for r in results if r["ev"] is not None]
    ev_ranked = sorted(ev_games, key=lambda x: x["ev"], reverse=True)
    top5_ev   = ev_ranked[:5]
    top5_agree = [r for r in ev_ranked if r["agree"]][:5]

    ou_games   = [r for r in results if r["ou_best_ev"] is not None]
    ou_ranked  = sorted(ou_games, key=lambda x: x["ou_best_ev"], reverse=True)
    ou_positive = [r for r in ou_ranked if r["ou_best_ev"] > 0]

    any_ref = any(r["referee"] for r in results)
    ref_note = "referee bias applied" if any_ref else "referees not yet assigned — re-run after Sofascore publishes"

    def accu(games):
        p = 1.0
        for g in games:
            p *= g["ev_odds"]
        return p

    lines = []
    lines += [
        f"# GD{gameday} Predictions — Trendyol Süper Lig 2025-26",
        f"*Generated: {run_date}  ·  {ref_note}*",
        "",
        "---",
        "",
        "## Standings (before GD{})".format(gameday),
        "",
        "| Rank | Team | Pts | Pts/G | Last 5 |",
        "|------|------|----:|------:|--------|",
    ]
    for i, s in enumerate(slist):
        t = s["team"]
        c = ctx.get(t, {})
        lines.append(f"| {i+1} | {t} | {s['pts']} | {c.get('pts_pg',0):.2f} | {c.get('form_pts',0):.1f} pts/g |")

    lines += [
        "",
        "---",
        "",
        "## Probability Table",
        "",
        "> Columns: DC = Dixon-Coles · XGB = XGBoost · ENS = 60/40 blend · REF = referee-adjusted",
        "",
        "| Match | DC H/D/A | XGB H/D/A | ENS H/D/A | REF H/D/A | Agree | Referee |",
        "|-------|----------|-----------|-----------|-----------|:-----:|---------|",
    ]
    for r in sorted(results, key=lambda x: x["ev"] if x["ev"] else -99, reverse=True):
        match = f"{r['home']} v {r['away']}"
        ag = "✓" if r["agree"] else "✗"
        if r["referee"]:
            ref_str = f"{r['referee']} (D×{r['ref_draw_bias']:.2f} H×{r['ref_home_bias']:.2f} n={r['ref_matches']})"
        elif any_ref:
            ref_str = "not assigned"
        else:
            ref_str = "—"
        lines.append(
            f"| {match} "
            f"| {r['dc_H']:.2f}/{r['dc_D']:.2f}/{r['dc_A']:.2f} "
            f"| {r['xgb_H']:.2f}/{r['xgb_D']:.2f}/{r['xgb_A']:.2f} "
            f"| {r['ens_H']:.2f}/{r['ens_D']:.2f}/{r['ens_A']:.2f} "
            f"| **{r['ref_H']:.2f}/{r['ref_D']:.2f}/{r['ref_A']:.2f}** "
            f"| {ag} | {ref_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## EV Rankings — Double Chance",
        "",
        "> EV = (referee-adjusted probability × bookmaker DC odds) − 1. Positive EV = value bet.",
        "",
        "| Rk | Match | Bet | Odds | EV | DC pred | Final pred | Agree | Referee |",
        "|----|-------|-----|-----:|---:|---------|------------|:-----:|---------|",
    ]
    for rk, r in enumerate(ev_ranked, 1):
        ev_flag = " ✦" if r["ev"] > 0 else ""
        ag = "✓" if r["agree"] else "✗"
        ref_str = f"{r['referee']} (D×{r['ref_draw_bias']:.2f})" if r["referee"] else "—"
        lines.append(
            f"| {rk} | {r['home']} v {r['away']} "
            f"| **{r['ev_bet']}** | {r['ev_odds']:.2f} | {r['ev']:+.4f}{ev_flag} "
            f"| {r['dc_pred']} | {r['ens_pred']} | {ag} | {ref_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Strategy Picks",
        "",
        "### EV Strategy — Top 5 by EV",
        "",
    ]
    for rk, r in enumerate(top5_ev, 1):
        ag = "consensus" if r["agree"] else "split signal"
        ev_str = f"EV={r['ev']:+.4f}" if r["ev"] else "—"
        lines.append(f"{rk}. **{r['home']} v {r['away']}** → `{r['ev_bet']} @ {r['ev_odds']:.2f}` · {ev_str} · {ag}")
    lines += [
        "",
        f"**Accumulator odds: {accu(top5_ev):.2f}×** *(win only if ALL 5 are covered)*",
        "",
        "### EV+Agree Strategy — Consensus Picks",
        "",
    ]
    if len(top5_agree) < 5:
        lines.append(f"> ⚠ Only {len(top5_agree)} games pass the DC=XGB agreement filter.")
        lines.append("")
    for rk, r in enumerate(top5_agree, 1):
        ev_str = f"EV={r['ev']:+.4f}" if r["ev"] else "—"
        lines.append(f"{rk}. **{r['home']} v {r['away']}** → `{r['ev_bet']} @ {r['ev_odds']:.2f}` · {ev_str}")
    if top5_agree:
        lines += ["", f"**Accumulator odds: {accu(top5_agree):.2f}×**", ""]

    lines += [
        "",
        "---",
        "",
        "## Over/Under Analysis",
        "",
        "| Rk | Match | Line | xG | P(Over) | P(Under) | Ov odds | Un odds | EV(Ov) | EV(Un) | Best |",
        "|----|-------|-----:|---:|--------:|---------:|--------:|--------:|-------:|-------:|------|",
    ]
    for rk, r in enumerate(ou_ranked, 1):
        xg = (r["dc_home_exp"] or 0) + (r["dc_away_exp"] or 0)
        ou_info = MATCH_OU.get((r["home"], r["away"]), {})
        ev_flag = " ✦" if r["ou_best_ev"] > 0 else ""
        lines.append(
            f"| {rk} | {r['home']} v {r['away']} "
            f"| {r['ou_line']:.1f} | {xg:.2f} "
            f"| {r['ou_p_over']:.4f} | {r['ou_p_under']:.4f} "
            f"| {ou_info.get('over',0):.2f} | {ou_info.get('under',0):.2f} "
            f"| {r['ou_ev_over']:+.4f} | {r['ou_ev_under']:+.4f} "
            f"| **{r['ou_best_bet']}**{ev_flag} |"
        )

    lines += ["", "### O/U Positive EV Picks", ""]
    if not ou_positive:
        lines.append("> No O/U bets with positive EV found this gameday.")
    else:
        for rk, r in enumerate(ou_positive, 1):
            xg = (r["dc_home_exp"] or 0) + (r["dc_away_exp"] or 0)
            ou_info = MATCH_OU.get((r["home"], r["away"]), {})
            odds_val = ou_info.get("over" if r["ou_best_bet"] == "OVER" else "under", 0)
            lines.append(
                f"{rk}. **{r['home']} v {r['away']}** → "
                f"`{r['ou_best_bet']} {r['ou_line']:.1f} @ {odds_val:.2f}` · "
                f"EV={r['ou_best_ev']:+.4f} · xG={xg:.2f}"
            )

    lines += [
        "",
        "---",
        "",
        "*Pipeline: Dixon-Coles × XGBoost Stack × Referee Bias | [predict-may](https://github.com/onurhani/predict-may)*",
    ]
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly Super Lig gameday prediction")
    parser.add_argument("--gameday", type=int, default=GAMEDAY,
                        help=f"Gameday to predict (default: {GAMEDAY})")
    parser.add_argument("--no-md", dest="write_md", action="store_false",
                        help="Skip writing Markdown to Obsidian vault")
    args = parser.parse_args()

    gd = args.gameday
    run_date = date.today().isoformat()

    print(f"GD{gd} Predictions — Trendyol Süper Lig 2025-26")
    print("=" * 60)

    # Load season data
    print("\n[1] Loading season data from football-data.co.uk...")
    frames = []
    for code in SEASON_CODES:
        print(f"  {code}...", end="", flush=True)
        df = load_season(code)
        print(f" {len(df)}")
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df_2526  = all_data[all_data["season"] == "2526"].copy()
    print(f"  Completed 2526 matches: {len(df_2526)}  (GD1–{len(df_2526)//9})")

    schedule = load_schedule(gd)
    print(f"  GD{gd} schedule: {len(schedule)} matches")

    # Fit DC model on all data
    print("\n[2] Fitting DC model on all completed data...")
    dc_model = fit_dc(all_data)
    print("  Done.")

    # Compute DC probs for all GD matches
    dc_results = {}   # (ht,at) -> (ph, pd_, pa, pred_obj)
    for _, row in schedule.iterrows():
        ht, at = row["home_team"], row["away_team"]
        dc_results[(ht,at)] = dc_pred(dc_model, ht, at)

    # Context
    print(f"[3] Computing current standings and form (after GD{gd-1})...")
    ctx, slist = compute_context(df_2526, all_data)

    # Print standings
    print(f"\n  Standings after GD{gd-1}:")
    print(f"  {'Rank':<5} {'Team':<22} {'Pts':>4} {'Pts/G':>6} {'Form':>6}")
    for i, s in enumerate(slist):
        t = s["team"]
        c = ctx.get(t, {})
        form_pts_str = f"{c.get('form_pts',0):.1f}"
        print(f"  {i+1:<5} {t:<22} {s['pts']:>4} {c.get('pts_pg',0):>6.2f} {form_pts_str:>6}")

    # Load referee data
    print("\n[4] Loading referee data...")
    _, gd_ref_lookup = load_referee_data(gd)
    ref_assigned = sum(1 for v in gd_ref_lookup.values() if v is not None)
    print(f"  GD{gd} referee assignments: {ref_assigned}/{len(gd_ref_lookup)} known")
    if ref_assigned == 0:
        print("  ℹ  No referees assigned yet — bias will be applied once Sofascore publishes them.")
        print("  ℹ  Re-run: python3 scripts/fetch_referee_data.py  to refresh.")

    # Load features and train XGBoost
    print("\n[5] Loading feature dataset and training XGBoost...")
    feat_df  = pd.read_csv(FEAT_CSV)
    feat_df["season"] = feat_df["season"].astype(str)
    team_amv, sq_ratio_map, xi_ratio_map = load_amv(feat_df)
    xgb_model, feat_med = train_xgb(feat_df)
    print(f"  XGBoost trained on {len(feat_df[feat_df['season']!='2526'])} matches")

    # Build feature vectors for this gameday
    gd_rows = []
    for _, row in schedule.iterrows():
        ht, at = row["home_team"], row["away_team"]
        ph, pd_, pa, pred_obj = dc_results.get((ht,at), (None,None,None,None))
        if ph is None:
            print(f"  Warning: no DC probs for {ht} v {at}")
            continue

        h_ctx = ctx.get(ht, {"rank":9,"pts_pg":1.2,"form_pts":1.2,"form_gd":0.0,"motivation":0.5})
        a_ctx = ctx.get(at, {"rank":9,"pts_pg":1.2,"form_pts":1.2,"form_gd":0.0,"motivation":0.5})
        sq_r, xi_r = squad_ratios_for_match(ht, at, team_amv, sq_ratio_map, xi_ratio_map)
        h2h = h2h_rate(ht, at, all_data)
        derby = int(ht in BIG3 and at in BIG3)

        feat = {
            "prob_H": ph, "prob_D": pd_, "prob_A": pa,
            "home_form_pts": h_ctx["form_pts"], "away_form_pts": a_ctx["form_pts"],
            "home_form_gd":  h_ctx["form_gd"],  "away_form_gd":  a_ctx["form_gd"],
            "home_rank": h_ctx["rank"],          "away_rank": a_ctx["rank"],
            "home_pts_pg": h_ctx["pts_pg"],      "away_pts_pg": a_ctx["pts_pg"],
            "home_motivation": h_ctx["motivation"], "away_motivation": a_ctx["motivation"],
            "home_amv_ratio": team_amv.get(ht, 0.85),
            "away_amv_ratio": team_amv.get(at, 0.85),
            "squad_value_ratio": float(np.clip(sq_r,-2,2)),
            "xi_value_ratio":    float(np.clip(xi_r,-2,2)),
            "h2h_home_rate": h2h, "derby": derby,
            "gameday": gd, "season_progress": round(gd/34, 3),
        }
        gd_rows.append((ht, at, ph, pd_, pa, feat, pred_obj))

    # Apply XGBoost
    feat_arr = np.array([[r[5].get(c, float(feat_med.get(c,0))) for c in FEATURE_COLS]
                         for r in gd_rows], dtype=float)  # r[5] is the feat dict
    feat_arr = np.where(np.isnan(feat_arr), feat_med.fillna(0).values, feat_arr)
    xgb_proba = xgb_model.predict_proba(feat_arr)

    # Assemble results
    results = []
    for i, (ht, at, ph, pd_, pa, feat, pred_obj) in enumerate(gd_rows):
        ph_x = float(xgb_proba[i][0])
        pd_x = float(xgb_proba[i][1])
        pa_x = float(xgb_proba[i][2])

        ph_e = ALPHA*ph  + (1-ALPHA)*ph_x
        pd_e = ALPHA*pd_ + (1-ALPHA)*pd_x
        pa_e = ALPHA*pa  + (1-ALPHA)*pa_x

        # Referee bias correction (applied on top of ensemble)
        ref_stats = ref_for_match(ht, at, gd_ref_lookup)
        ph_r, pd_r, pa_r = apply_referee_bias(ph_e, pd_e, pa_e, ref_stats)
        ref_name    = ref_stats["referee_name"] if ref_stats else None
        ref_draw_b  = ref_stats["draw_bias"]    if ref_stats else None
        ref_home_b  = ref_stats["home_bias"]    if ref_stats else None
        ref_matches = ref_stats["matches"]       if ref_stats else None

        dc_p  = max({"H":ph,  "D":pd_,  "A":pa},  key={"H":ph,"D":pd_,"A":pa}.get)
        xgb_p = max({"H":ph_x,"D":pd_x,"A":pa_x}, key={"H":ph_x,"D":pd_x,"A":pa_x}.get)
        ens_p = max({"H":ph_r,"D":pd_r,"A":pa_r}, key={"H":ph_r,"D":pd_r,"A":pa_r}.get)
        agree = dc_p == xgb_p

        odds = MATCH_ODDS.get((ht, at))
        ev = ev_bet = ev_odds = ev_covers = None
        if odds:
            # Use referee-adjusted probs for EV; if no referee yet, falls back to ensemble
            ev, ev_bet, ev_odds, ev_covers = best_ev(ph_r, pd_r, pa_r,
                                                      odds["dc_1x"], odds["dc_x2"], odds["dc_12"])

        # Over/Under
        ou = MATCH_OU.get((ht, at))
        ou_ev_over = ou_ev_under = ou_p_over = ou_p_under = None
        ou_best_ev = ou_best_bet = ou_best_odds = None
        if ou and pred_obj is not None:
            ou_p_over, ou_p_under = ou_probs(pred_obj, ou["line"])
            if ou_p_over is not None:
                ou_ev_over  = round(ou_p_over  * ou["over"]  - 1, 5)
                ou_ev_under = round(ou_p_under * ou["under"] - 1, 5)
                if ou_ev_over >= ou_ev_under:
                    ou_best_ev, ou_best_bet, ou_best_odds = ou_ev_over,  "OVER",  ou["over"]
                else:
                    ou_best_ev, ou_best_bet, ou_best_odds = ou_ev_under, "UNDER", ou["under"]

        results.append({
            "home":ht, "away":at,
            "dc_H":round(ph,3),   "dc_D":round(pd_,3),  "dc_A":round(pa,3),
            "xgb_H":round(ph_x,3),"xgb_D":round(pd_x,3),"xgb_A":round(pa_x,3),
            "ens_H":round(ph_e,3),"ens_D":round(pd_e,3),"ens_A":round(pa_e,3),
            "ref_H":round(ph_r,3),"ref_D":round(pd_r,3),"ref_A":round(pa_r,3),
            "referee":     ref_name,
            "ref_matches": ref_matches,
            "ref_draw_bias": ref_draw_b,
            "ref_home_bias": ref_home_b,
            "dc_pred":dc_p, "xgb_pred":xgb_p, "ens_pred":ens_p, "agree":agree,
            "ev":ev, "ev_bet":ev_bet, "ev_odds":ev_odds, "ev_covers":ev_covers,
            "has_odds": odds is not None,
            # O/U
            "ou_line": ou["line"] if ou else None,
            "ou_p_over": round(ou_p_over,4)  if ou_p_over  is not None else None,
            "ou_p_under": round(ou_p_under,4) if ou_p_under is not None else None,
            "ou_ev_over":  ou_ev_over,
            "ou_ev_under": ou_ev_under,
            "ou_best_ev":  ou_best_ev,
            "ou_best_bet": ou_best_bet,
            "ou_best_odds": ou_best_odds,
            "dc_home_exp": round(float(pred_obj.home_goal_expectation),3) if pred_obj else None,
            "dc_away_exp": round(float(pred_obj.away_goal_expectation),3) if pred_obj else None,
            "h_rank":feat["home_rank"],   "a_rank":feat["away_rank"],
            "h_form":feat["home_form_pts"],"a_form":feat["away_form_pts"],
            "h_motive":feat["home_motivation"],"a_motive":feat["away_motivation"],
        })

    # ─── Output ────────────────────────────────────────────────────────────────
    any_ref = any(r["referee"] for r in results)

    print("\n" + "="*110)
    print(f"GD{gd} FULL PROBABILITY TABLE")
    print("="*110)
    hdr = (f"{'Match':<35}  {'─── DC ───':^18} {'─── XGB ──':^18} {'─── ENS ──':^18} "
           f"{'── REF ADJ ─':^18}  Ag  Referee")
    print(hdr)
    print("─"*110)

    for r in sorted(results, key=lambda x: x["ev"] if x["ev"] else -99, reverse=True):
        match = f"{r['home']:<18} v {r['away']}"
        ag = "✓" if r["agree"] else "✗"
        ref_str = "—"
        if r["referee"]:
            db = r["ref_draw_bias"]; hb = r["ref_home_bias"]
            ref_str = f"{r['referee']} (D×{db:.2f} H×{hb:.2f} n={r['ref_matches']})"
        elif any_ref:
            ref_str = "not assigned"
        print(f"{match:<35}  "
              f"H{r['dc_H']:.2f} D{r['dc_D']:.2f} A{r['dc_A']:.2f}  "
              f"H{r['xgb_H']:.2f} D{r['xgb_D']:.2f} A{r['xgb_A']:.2f}  "
              f"H{r['ens_H']:.2f} D{r['ens_D']:.2f} A{r['ens_A']:.2f}  "
              f"H{r['ref_H']:.2f} D{r['ref_D']:.2f} A{r['ref_A']:.2f}  "
              f"{ag:^3}  {ref_str}")

    # EV Table
    ev_games  = [r for r in results if r["ev"] is not None]
    ev_ranked = sorted(ev_games, key=lambda x: x["ev"], reverse=True)

    print("\n" + "="*110)
    print("EV RANKINGS  (Referee-adjusted ensemble probs × Bookmaker DC odds − 1)  — sorted by EV")
    print(f"  Note: REF ADJ = ENS (60%DC+40%XGB) nudged by referee historical draw/home bias.")
    print(f"  {'— Referee unassigned' if not any_ref else '— All referee bias applied'}")
    print("="*110)
    print(f"{'Rk':<3} {'Match':<35} {'Bet':>3} {'BkOdds':>7} {'EV':>8}  {'DC':>4} {'ENS':>4} {'Ag':>3}  Referee")
    print("─"*110)
    for rk, r in enumerate(ev_ranked, 1):
        match = f"{r['home']} v {r['away']}"
        ev_flag = " ◄" if r["ev"] > 0 else "  "
        ag = "✓" if r["agree"] else "✗"
        ref_str = "—"
        if r["referee"]:
            ref_str = f"{r['referee']} (D×{r['ref_draw_bias']:.2f})"
        print(f"{rk:<3} {match:<35} {r['ev_bet']:>3} {r['ev_odds']:>7.2f} {r['ev']:>+8.4f}{ev_flag}  "
              f"{r['dc_pred']:>4} {r['ens_pred']:>4} {ag:>3}  {ref_str}")

    # ─── Strategy Selections ─────────────────────────────────────────────────
    print("\n" + "="*100)
    print(f"STRATEGY SELECTIONS FOR GD{gd}")
    print("="*100)

    top5_ev    = ev_ranked[:5]
    agree_pool = [r for r in ev_ranked if r["agree"]]
    top5_agree = agree_pool[:5]

    def accu_info(games):
        odds_prod = 1.0
        for g in games:
            odds_prod *= g["ev_odds"]
        return odds_prod

    print(f"\n  ┌─ EV Strategy (Top 5 by EV) ──────────────────────────────────────────┐")
    for rk, r in enumerate(top5_ev, 1):
        ev_tag = f"EV={r['ev']:+.4f}" if r["ev"] else "no odds"
        ag     = "✓agree" if r["agree"] else "✗disagree"
        print(f"  │  {rk}. {r['home']:18s} v {r['away']:18s}  →  {r['ev_bet']} @ {r['ev_odds']:.2f}   {ev_tag}  {ag}")
    print(f"  │")
    print(f"  │  Accumulator odds: {accu_info(top5_ev):.2f}x   (win if ALL 5 outcomes covered)")
    print(f"  └──────────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ EV+Agree Strategy (Top 5 by EV, DC==XGB consensus) ─────────────────┐")
    if len(top5_agree) < 5:
        print(f"  │  ⚠ Only {len(top5_agree)} games pass agreement filter (need 5)")
    for rk, r in enumerate(top5_agree, 1):
        ev_tag = f"EV={r['ev']:+.4f}" if r["ev"] else "no odds"
        print(f"  │  {rk}. {r['home']:18s} v {r['away']:18s}  →  {r['ev_bet']} @ {r['ev_odds']:.2f}   {ev_tag}")
    print(f"  │")
    if top5_agree:
        print(f"  │  Accumulator odds: {accu_info(top5_agree):.2f}x")
    print(f"  └──────────────────────────────────────────────────────────────────────┘")


    # ─── Over/Under Rankings ─────────────────────────────────────────────────
    ou_games  = [r for r in results if r["ou_best_ev"] is not None]
    ou_ranked = sorted(ou_games, key=lambda x: x["ou_best_ev"], reverse=True)

    print("\n" + "="*100)
    print("OVER/UNDER EV RANKINGS  (DC score matrix × bookmaker O/U odds − 1)  — sorted by EV")
    print("="*100)
    print(f"{'Rk':<3} {'Match':<35} {'Line':>5} {'xG H+A':>8} {'P(Over)':>8} {'P(Under)':>9} "
          f"{'OvOdds':>7} {'UnOdds':>7} {'EV(Ov)':>8} {'EV(Un)':>8} {'Best':>7}")
    print("─"*100)
    for rk, r in enumerate(ou_ranked, 1):
        match = f"{r['home']} v {r['away']}"
        ev_flag = " ◄" if r["ou_best_ev"] > 0 else "  "
        xg_total = (r["dc_home_exp"] or 0) + (r["dc_away_exp"] or 0)
        print(f"{rk:<3} {match:<35} "
              f"{r['ou_line']:>5.1f} "
              f"{xg_total:>7.2f}g "
              f"{r['ou_p_over']:>8.4f} "
              f"{r['ou_p_under']:>9.4f} "
              f"{MATCH_OU.get((r['home'],r['away']),{}).get('over',0):>7.2f} "
              f"{MATCH_OU.get((r['home'],r['away']),{}).get('under',0):>7.2f} "
              f"{r['ou_ev_over']:>+8.4f} "
              f"{r['ou_ev_under']:>+8.4f} "
              f"  {r['ou_best_bet']:>5}{ev_flag}")

    # O/U Strategy: top picks with positive EV
    ou_positive = [r for r in ou_ranked if r["ou_best_ev"] > 0]
    print(f"\n  ┌─ O/U Strategy (Positive EV picks) ────────────────────────────────────┐")
    if not ou_positive:
        print(f"  │  ⚠  No O/U bets with positive EV found this gameday")
    for rk, r in enumerate(ou_positive, 1):
        xg_total = (r["dc_home_exp"] or 0) + (r["dc_away_exp"] or 0)
        ou_info = MATCH_OU.get((r['home'], r['away']), {})
        odds_val = ou_info.get("over" if r["ou_best_bet"]=="OVER" else "under", 0)
        print(f"  │  {rk}. {r['home']:18s} v {r['away']:18s}  →  "
              f"{r['ou_best_bet']} {r['ou_line']:.1f} @ {odds_val:.2f}   "
              f"EV={r['ou_best_ev']:+.4f}   xG={xg_total:.2f}")
    print(f"  └──────────────────────────────────────────────────────────────────────┘")

    # ─── Save ML predictions JSON (consumed by export_dashboard + simulate_season) ──
    ml_out = {
        "gameday": gd,
        "generated_at": run_date,
        "matches": [
            {
                "home": r["home"],
                "away": r["away"],
                "prob_home": r["ref_H"],
                "prob_draw": r["ref_D"],
                "prob_away": r["ref_A"],
                "predicted": r["ens_pred"],
            }
            for r in results
        ],
    }
    ML_PREDS_JSON.write_text(json.dumps(ml_out, indent=2))
    print(f"\n💾  ML predictions saved → {ML_PREDS_JSON}")

    # ─── Markdown export ───────────────────────────────────────────────────────
    if args.write_md:
        OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True)
        md_path = OBSIDIAN_DIR / f"GD{gd}.md"
        md = generate_markdown(gd, results, slist, ctx, ref_assigned, run_date)
        md_path.write_text(md, encoding="utf-8")
        print(f"\n📓  Obsidian note written → {md_path}")
    print()


if __name__ == "__main__":
    main()
