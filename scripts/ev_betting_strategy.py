#!/usr/bin/env python3
"""
EV-Ranked Double-Chance Betting Strategy — Turkish Super Lig 2025-26

Implements the 5 improvements from the strategy roadmap:
  1. EV-ranked selection   (model_prob × DC_odds - 1)
  2. Referee draw rate     (placeholder; uses draw classifier instead)
  3. Extended training data (handled in phase5_xgboost_stack.py)
  4. Draw-risk binary classifier  (logistic regression on phase5 features)
  5. Agreement filter      (DC and XGB must agree on predicted outcome)

Strategies compared (all use DC double-chance accumulators, Top-5 per week):
  Naive:      Ranked by coverage confidence  (existing baseline)
  EV:         Ranked by EV (model prob × DC odds - 1)
  EV+Agree:   EV, only games where DC and XGB agree
  EV+Draw:    EV, exclude high draw-risk games
  EV+Both:    EV + agreement + draw-risk filter combined
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
PREDS_JSON = ROOT / "scripts" / "phase5_predictions.json"
FEAT_CSV   = ROOT / "scripts" / "phase5_features.csv"

STAKE          = 1.0   # units staked per gameday per strategy
TOP_N          = 5     # games per accumulator
DRAW_THRESHOLD = 0.42  # binary draw classifier cutoff (tuned to ~top 30% of draw_prob)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions() -> list:
    with open(PREDS_JSON) as f:
        return json.load(f)


def load_features() -> pd.DataFrame:
    """Full feature dataset across all seasons built by phase5_xgboost_stack.py."""
    df = pd.read_csv(FEAT_CSV, low_memory=False)
    df["season"] = df["season"].astype(str)  # normalise to strings
    return df


def load_odds_2526() -> pd.DataFrame:
    """B365 (or best available) odds for the 2025-26 season."""
    url = "https://www.football-data.co.uk/mmz4281/2526/T1.csv"
    df = pd.read_csv(url, encoding="latin1")
    df = df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTR": "result"})
    for prefix in ["B365", "Avg", "BW"]:
        cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
        if all(c in df.columns for c in cols):
            df["o_H"] = pd.to_numeric(df[cols[0]], errors="coerce")
            df["o_D"] = pd.to_numeric(df[cols[1]], errors="coerce")
            df["o_A"] = pd.to_numeric(df[cols[2]], errors="coerce")
            df["odds_src"] = prefix
            break
    return df[["home", "away", "o_H", "o_D", "o_A"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DC odds and EV
# ─────────────────────────────────────────────────────────────────────────────

def dc_odds_from_1x2(o_h, o_d, o_a):
    """Convert 1X2 to DC odds: 1X (home+draw), X2 (away+draw), 12 (home+away)."""
    return (
        1.0 / (1.0/o_h + 1.0/o_d),
        1.0 / (1.0/o_d + 1.0/o_a),
        1.0 / (1.0/o_h + 1.0/o_a),
    )


def best_ev_option(p_h, p_d, p_a, o_h, o_d, o_a):
    """
    Returns (ev, bet_type, dc_odds_val, covers) for highest-EV DC option.
    Uses ensemble probabilities for EV calculation.
    """
    try:
        o1x, ox2, o12 = dc_odds_from_1x2(o_h, o_d, o_a)
    except (ZeroDivisionError, TypeError, ValueError):
        return None, None, None, None
    opts = [
        ((p_h + p_d) * o1x - 1, "1X", o1x, ("H", "D")),
        ((p_d + p_a) * ox2 - 1, "X2", ox2, ("D", "A")),
        ((p_h + p_a) * o12 - 1, "12", o12, ("H", "A")),
    ]
    return max(opts, key=lambda x: x[0])


def naive_option(p_h, p_d, p_a, dc_pred, o_h, o_d, o_a):
    """
    Naive strategy: bet 1X if DC predicts H, X2 if A.
    Returns (confidence, bet_type, dc_odds_val, covers).
    """
    try:
        o1x, ox2, _ = dc_odds_from_1x2(o_h, o_d, o_a)
    except (ZeroDivisionError, TypeError, ValueError):
        return None, None, None, None
    if dc_pred == "H":
        return (p_h + p_d), "1X", o1x, ("H", "D")
    elif dc_pred == "A":
        return (p_d + p_a), "X2", ox2, ("D", "A")
    else:  # D (rare) — take whichever side is stronger
        if p_h >= p_a:
            return (p_h + p_d), "1X", o1x, ("H", "D")
        return (p_d + p_a), "X2", ox2, ("D", "A")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Draw-risk binary classifier
# ─────────────────────────────────────────────────────────────────────────────

DRAW_FEATS = [
    "prob_H", "prob_D", "prob_A",
    "home_motivation", "away_motivation",
    "home_amv_ratio", "away_amv_ratio",
    "squad_value_ratio", "xi_value_ratio",
    "home_rank", "away_rank",
    "home_pts_pg", "away_pts_pg",
    "home_form_pts", "away_form_pts",
    "gameday", "season_progress",
]


def train_draw_classifier(feat_df: pd.DataFrame):
    """
    Train binary draw/not-draw logistic regression on all non-2526 seasons.
    Returns (clf, scaler, train_draw_rate, train_accuracy).
    """
    train = feat_df[feat_df["season"] != "2526"].copy()
    med = train[DRAW_FEATS].median()
    X = train[DRAW_FEATS].fillna(med)
    y = (train["actual"] == "D").astype(int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Use mild draw weight (draws are ~25% of outcomes; don't over-flag)
    clf = LogisticRegression(C=0.5, class_weight={0: 1, 1: 2}, max_iter=1000, random_state=42)
    clf.fit(X_sc, y)

    draw_rate = y.mean()
    train_acc = clf.score(X_sc, y)
    return clf, scaler, med, float(draw_rate), float(train_acc)


def predict_draw_probs(clf, scaler, med, feat_df: pd.DataFrame) -> dict:
    """
    Apply draw classifier to 2526 matches.
    Returns {(home, away): draw_prob}.
    """
    test = feat_df[feat_df["season"] == "2526"].copy()
    X = test[DRAW_FEATS].fillna(med)
    X_sc = scaler.transform(X)
    probs = clf.predict_proba(X_sc)[:, 1]  # P(draw)
    return {(row["home"], row["away"]): float(probs[i]) for i, (_, row) in enumerate(test.iterrows())}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build per-match enriched records for 2526
# ─────────────────────────────────────────────────────────────────────────────

def build_match_records(preds: list, odds_df: pd.DataFrame, draw_probs: dict) -> dict:
    """
    Merge predictions + odds + draw probs for every 2526 match.
    Returns dict: gameday -> list of enriched match dicts.
    """
    odds_map = {(r["home"], r["away"]): r for _, r in odds_df.iterrows()}
    by_gd = defaultdict(list)

    for p in preds:
        ht, at = p["home"], p["away"]
        gd = p["gameday"]
        actual = p["actual"]

        # Probabilities
        p_h  = p["prob_H"];    p_d  = p["prob_D"];    p_a  = p["prob_A"]
        ph_e = p["prob_H_ens"]; pd_e = p["prob_D_ens"]; pa_e = p["prob_A_ens"]

        # Agreement filter: DC argmax == XGB argmax
        agreement = (p["dc_predicted"] == p["xgb_predicted"])

        # Draw risk
        draw_prob = draw_probs.get((ht, at), 0.3)
        draw_risk = draw_prob >= DRAW_THRESHOLD

        # Odds
        odds = odds_map.get((ht, at))
        o_h = o_d = o_a = None
        if odds is not None:
            o_h, o_d, o_a = odds["o_H"], odds["o_D"], odds["o_A"]
            # Sanity check: odds must be valid
            if any(v is None or np.isnan(v) or v < 1.0 for v in [o_h, o_d, o_a]):
                o_h = o_d = o_a = None

        # EV option (using ensemble probs)
        best_ev, best_bet, best_dc_odds, best_covers = (None, None, None, None)
        if o_h is not None:
            best_ev, best_bet, best_dc_odds, best_covers = best_ev_option(ph_e, pd_e, pa_e, o_h, o_d, o_a)

        # Naive option (using DC probs)
        naive_conf, naive_bet, naive_dc_odds, naive_covers = (None, None, None, None)
        if o_h is not None:
            naive_conf, naive_bet, naive_dc_odds, naive_covers = naive_option(p_h, p_d, p_a, p["dc_predicted"], o_h, o_d, o_a)

        by_gd[gd].append({
            "gameday": gd,
            "home": ht, "away": at,
            "actual": actual,
            "dc_predicted": p["dc_predicted"],
            "xgb_predicted": p["xgb_predicted"],
            "ens_predicted": p["ens_predicted"],
            "agreement": agreement,
            "draw_prob": round(draw_prob, 4),
            "draw_risk": draw_risk,
            "has_odds": (o_h is not None),
            # EV strategy
            "best_ev": best_ev,
            "best_bet": best_bet,
            "best_dc_odds": best_dc_odds,
            "best_covers": best_covers,
            # Naive strategy
            "naive_conf": naive_conf,
            "naive_bet": naive_bet,
            "naive_dc_odds": naive_dc_odds,
            "naive_covers": naive_covers,
        })

    return by_gd


# ─────────────────────────────────────────────────────────────────────────────
# 5. Strategy simulation
# ─────────────────────────────────────────────────────────────────────────────

def accumulator_result(selected: list, stake: float):
    """
    selected: list of dicts with keys: covers (tuple), dc_bet_odds (float), actual (str)
    Returns (net_return_float, won_bool).
    """
    if not selected:
        return -stake, False
    won = all(g["actual"] in g["covers"] for g in selected)
    if not won:
        return -stake, False
    accu_odds = 1.0
    for g in selected:
        accu_odds *= g["dc_bet_odds"]
    return round(stake * accu_odds - stake, 4), True


STRATEGY_DEFS = {
    "Naive":    {"ev_rank": False, "agree": False, "draw_filter": False},
    "EV":       {"ev_rank": True,  "agree": False, "draw_filter": False},
    "EV+Agree": {"ev_rank": True,  "agree": True,  "draw_filter": False},
    "EV+Draw":  {"ev_rank": True,  "agree": False, "draw_filter": True},
    "EV+Both":  {"ev_rank": True,  "agree": True,  "draw_filter": True},
}


def run_all_strategies(by_gd: dict, top_n: int = TOP_N, stake: float = STAKE) -> dict:
    """
    Returns dict: strategy_name -> list[{gameday, n_games, net, won, accu_odds, selected}]
    """
    results = {s: [] for s in STRATEGY_DEFS}

    for gd in sorted(by_gd.keys()):
        games = by_gd[gd]
        valid = [g for g in games if g["has_odds"]]

        for strat, cfg in STRATEGY_DEFS.items():
            pool = valid[:]

            if cfg["agree"]:
                pool = [g for g in pool if g["agreement"]]
            if cfg["draw_filter"]:
                pool = [g for g in pool if not g["draw_risk"]]

            if cfg["ev_rank"]:
                # Sort by EV descending; use only positive EV when available
                pool_sorted = sorted(
                    [g for g in pool if g["best_ev"] is not None],
                    key=lambda g: g["best_ev"], reverse=True
                )
                pos_ev = [g for g in pool_sorted if g["best_ev"] > 0]
                chosen = (pos_ev if pos_ev else pool_sorted)[:top_n]
                selected = [
                    {"covers": g["best_covers"], "dc_bet_odds": g["best_dc_odds"], "actual": g["actual"]}
                    for g in chosen
                ]
            else:
                # Naive: sort by confidence
                pool_sorted = sorted(
                    [g for g in pool if g["naive_conf"] is not None],
                    key=lambda g: g["naive_conf"], reverse=True
                )
                chosen = pool_sorted[:top_n]
                selected = [
                    {"covers": g["naive_covers"], "dc_bet_odds": g["naive_dc_odds"], "actual": g["actual"]}
                    for g in chosen
                ]

            net, won = accumulator_result(selected, stake)
            accu_odds = (net + stake) / stake if won and selected else 0.0

            results[strat].append({
                "gameday": gd,
                "n_games": len(selected),
                "net": net,
                "won": won,
                "accu_odds": round(accu_odds, 2),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_strategy_report(results: dict, stake: float = STAKE):
    strategies = list(STRATEGY_DEFS.keys())

    # Per-gameday table
    gds = sorted({r["gameday"] for gd_results in results.values() for r in gd_results})
    # Index results by gameday per strategy
    by_strat_gd = {s: {r["gameday"]: r for r in results[s]} for s in strategies}

    print(f"\n{'═'*105}")
    header = f"{'GD':>4}  " + "  ".join(f"{s:>12}" for s in strategies)
    print(header)
    print(f"{'─'*105}")

    running = {s: 0.0 for s in strategies}

    for gd in gds:
        row = f"GD{gd:>2}  "
        for s in strategies:
            r = by_strat_gd[s].get(gd, {})
            net  = r.get("net", 0.0)
            won  = r.get("won", False)
            n    = r.get("n_games", 0)
            odds = r.get("accu_odds", 0.0)
            running[s] += net
            if won:
                cell = f"  +{net:>6.2f}u ({odds:.1f}x)"
            else:
                cell = f"  -{stake:.2f}u      "
            row += f"{cell:>14}"
        print(row)

    print(f"{'═'*105}")

    # Summary
    print(f"\n{'Summary':}")
    print(f"{'Strategy':<12}  {'Wins':>5}  {'Win%':>6}  {'Total P&L':>10}  {'ROI':>7}  {'Avg accu odds':>14}")
    print(f"{'─'*65}")

    total_weeks = len(gds)
    for s in strategies:
        res_list = results[s]
        wins = sum(r["won"] for r in res_list)
        total_stake = total_weeks * stake
        total_pnl   = sum(r["net"] for r in res_list)
        roi = total_pnl / total_stake * 100
        avg_odds = np.mean([r["accu_odds"] for r in res_list if r["won"]]) if wins else 0.0
        print(f"{s:<12}  {wins:>5}  {wins/total_weeks*100:>5.1f}%  {total_pnl:>+10.2f}u  {roi:>+6.1f}%  {avg_odds:>14.1f}x")

    print()


def print_draw_classifier_report(clf, scaler, draw_probs: dict, preds: list, train_draw_rate: float, train_acc: float):
    """Show draw classifier performance on 2526 data."""
    pred_map = {(p["home"], p["away"]): p["actual"] for p in preds}
    draws_detected = sum(1 for (k, prob) in draw_probs.items() if prob >= DRAW_THRESHOLD)
    correct_draws  = sum(1 for (k, prob) in draw_probs.items() if prob >= DRAW_THRESHOLD and pred_map.get(k) == "D")
    actual_draws   = sum(1 for p in preds if p["actual"] == "D")
    tp_draws = sum(1 for (k, prob) in draw_probs.items() if prob >= DRAW_THRESHOLD and pred_map.get(k) == "D")
    fn_draws = sum(1 for (k, prob) in draw_probs.items() if prob < DRAW_THRESHOLD and pred_map.get(k) == "D")
    precision = correct_draws / draws_detected if draws_detected else 0
    recall    = tp_draws / actual_draws if actual_draws else 0

    print(f"\nDraw-risk classifier (threshold={DRAW_THRESHOLD}):")
    print(f"  Training draw rate:  {train_draw_rate:.1%}  |  Training accuracy: {train_acc:.1%}")
    print(f"  2526 actual draws:   {actual_draws}/{len(preds)}")
    print(f"  Flagged as draw-risk: {draws_detected}  ({correct_draws} were actual draws)")
    print(f"  Precision: {precision:.1%}  |  Recall: {recall:.1%}")


def print_ev_analysis(by_gd: dict):
    """Show EV distribution across games."""
    all_evs = [g["best_ev"] for games in by_gd.values() for g in games if g["best_ev"] is not None]
    pos_ev  = [e for e in all_evs if e > 0]
    print(f"\nEV distribution across all {len(all_evs)} games with odds:")
    print(f"  Positive EV: {len(pos_ev)}/{len(all_evs)}  ({len(pos_ev)/len(all_evs)*100:.1f}%)")
    if all_evs:
        print(f"  EV range: {min(all_evs):+.3f} to {max(all_evs):+.3f}")
        print(f"  Mean EV: {np.mean(all_evs):+.3f}  |  Median EV: {np.median(all_evs):+.3f}")

    # Per-gameday how many positive EV games
    print(f"\n  GD  |  +EV games  |  Top-5 avg EV")
    print(f"  {'─'*35}")
    for gd in sorted(by_gd.keys()):
        games = [g for g in by_gd[gd] if g["best_ev"] is not None]
        pos   = sorted([g for g in games if g["best_ev"] > 0], key=lambda g: -g["best_ev"])
        top5  = pos[:5]
        avg_ev = np.mean([g["best_ev"] for g in top5]) if top5 else 0.0
        print(f"  GD{gd:>2} | {len(pos):>3}/{len(games):<3}       | {avg_ev:>+.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("EV-Ranked Betting Strategy — Turkish Super Lig 2025-26")
    print("=" * 60)

    # Load
    print("\n[1] Loading predictions (phase5)...")
    preds = load_predictions()
    print(f"    {len(preds)} matches for 2025-26")

    print("[2] Loading feature dataset...")
    if not FEAT_CSV.exists():
        print(f"    ERROR: {FEAT_CSV} not found. Run phase5_xgboost_stack.py first.")
        return
    feat_df = load_features()
    seasons = feat_df["season"].unique()
    print(f"    {len(feat_df)} records across seasons: {sorted(seasons)}")

    print("[3] Loading B365 odds for 2025-26...")
    odds_df = load_odds_2526()
    print(f"    {len(odds_df)} matches with odds")

    # Draw classifier
    print("[4] Training draw-risk classifier...")
    clf, scaler, med, train_draw_rate, train_acc = train_draw_classifier(feat_df)
    draw_probs = predict_draw_probs(clf, scaler, med, feat_df)
    print_draw_classifier_report(clf, scaler, draw_probs, preds, train_draw_rate, train_acc)

    # Build enriched match records
    print("\n[5] Computing EV and building match records...")
    by_gd = build_match_records(preds, odds_df, draw_probs)
    total_with_odds = sum(1 for gs in by_gd.values() for g in gs if g["has_odds"])
    total_without   = sum(1 for gs in by_gd.values() for g in gs if not g["has_odds"])
    print(f"    {total_with_odds} matches with odds  |  {total_without} without odds")
    print_ev_analysis(by_gd)

    # Run strategies
    print("\n[6] Running strategy simulations...")
    results = run_all_strategies(by_gd, top_n=TOP_N, stake=STAKE)

    # Report
    print_strategy_report(results, stake=STAKE)

    # Agreement filter diagnostic
    agree_games = [g for gs in by_gd.values() for g in gs if g["agreement"] and g["has_odds"]]
    total_games  = [g for gs in by_gd.values() for g in gs if g["has_odds"]]
    print(f"Agreement filter: {len(agree_games)}/{len(total_games)} games pass ({len(agree_games)/len(total_games)*100:.1f}%)")
    if agree_games:
        agree_acc = sum(1 for g in agree_games if g["dc_predicted"] == g["actual"]) / len(agree_games)
        all_acc   = sum(1 for g in total_games if g["dc_predicted"] == g["actual"])  / len(total_games)
        print(f"  Accuracy on agreed games: {agree_acc:.1%}  vs overall: {all_acc:.1%}")


if __name__ == "__main__":
    main()
