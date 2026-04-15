#!/usr/bin/env python3
"""
Dixon-Coles walk-forward backtest for Turkish Super Lig 2025-26 season.

For each gameday N in 1..MAX_GD:
  - Training set: all prior seasons (2021-22 to 2024-25) + 2025-26 GDs < N
  - Fit DixonColesGoalModel with time-decay weights (xi=0.0018)
  - Predict all matches in GD N
  - Record and compare against actual results + SPI model accuracy
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
DASHBOARD_JSON = PROJECT_ROOT / "docs" / "data" / "dashboard.json"
OUTPUT_JSON = PROJECT_ROOT / "scripts" / "dc_predictions.json"

# ---------------------------------------------------------------------------
# Team name mapping: CSV/schedule ASCII name → display name (for output only)
# ---------------------------------------------------------------------------
NAME_MAP = {
    "Besiktas": "Beşiktaş",
    "Buyuksehyr": "Başakşehir",
    "Fenerbahce": "Fenerbahçe",
    "Galatasaray": "Galatasaray",
    "Genclerbirligi": "Gençlerbirliği",
    "Goztep": "Göztepe",
    "Karagumruk": "Karagümrük",
    "Kasimpasa": "Kasımpaşa",
    "Trabzonspor": "Trabzonspor",
    "Alanyaspor": "Alanyaspor",
    "Antalyaspor": "Antalyaspor",
    "Eyupspor": "Eyüpspor",
    "Gaziantep": "Gaziantep",
    "Kayserispor": "Kayserispor",
    "Kocaelispor": "Kocaelispor",
    "Konyaspor": "Konyaspor",
    "Rizespor": "Rizespor",
    "Samsunspor": "Samsunspor",
}

# Time-decay rate — half-life ~385 days (penaltyblog default)
XI = 0.0018


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season(season_code: str) -> pd.DataFrame:
    """Load one season from football-data.co.uk.

    season_code: e.g. '2122', '2223', '2324', '2425', '2526'
    Returns DataFrame with columns: date, home, away, hg, ag, result
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/T1.csv"
    df = pd.read_csv(url, encoding="latin1")
    # Rename columns to a consistent schema
    df = df.rename(
        columns={
            "HomeTeam": "home",
            "AwayTeam": "away",
            "FTHG": "hg",
            "FTAG": "ag",
            "FTR": "result",
            "Date": "date_str",
        }
    )
    df["date"] = pd.to_datetime(df["date_str"], dayfirst=True, errors="coerce")
    # Keep only completed matches (non-null result and goals)
    df = df.dropna(subset=["result", "hg", "ag", "date"])
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["season"] = season_code
    return df[["season", "date", "home", "away", "hg", "ag", "result"]].copy()


def load_all_seasons() -> pd.DataFrame:
    """Load 2021-22 through 2025-26 and return combined DataFrame."""
    seasons = ["2122", "2223", "2324", "2425", "2526"]
    frames = []
    for s in seasons:
        print(f"  Loading {s}...", end="", flush=True)
        df = load_season(s)
        print(f" {len(df)} matches")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_schedule() -> pd.DataFrame:
    """Load 2025-26 schedule with round numbers from DuckDB."""
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    sched = conn.execute(
        "SELECT round_number, home_team, away_team FROM schedule_2526 ORDER BY round_number"
    ).fetchdf()
    conn.close()
    return sched


def load_spi_accuracy() -> dict:
    """Load SPI model per-gameday accuracy from dashboard.json."""
    with open(DASHBOARD_JSON) as f:
        data = json.load(f)
    spi = {}
    for entry in data["accuracy"]["weekly"]:
        label = entry["week_label"]  # e.g. "GD 1"
        gd = int(label.split()[1])
        spi[gd] = entry["weekly_pct"]
    overall = data["accuracy"]["overall_pct"]
    return spi, overall


# ---------------------------------------------------------------------------
# Brier score helper
# ---------------------------------------------------------------------------

def brier_score(records: list) -> float:
    """Compute multi-class Brier score over prediction records."""
    total = 0.0
    n = 0
    for r in records:
        if r["actual"] is None:
            continue
        probs = [r["prob_H"], r["prob_D"], r["prob_A"]]
        outcomes = [
            1 if r["actual"] == "H" else 0,
            1 if r["actual"] == "D" else 0,
            1 if r["actual"] == "A" else 0,
        ]
        total += sum((p - o) ** 2 for p, o in zip(probs, outcomes))
        n += 1
    return total / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(all_data: pd.DataFrame, schedule: pd.DataFrame) -> list:
    """
    Run walk-forward backtest for each gameday.

    Returns list of per-match prediction dicts.
    """
    # Merge results into schedule to get actual outcomes
    # CSV team names == schedule team names (confirmed by exploration)
    df_2526 = all_data[all_data["season"] == "2526"].copy()
    df_prior = all_data[all_data["season"] != "2526"].copy()

    # Map CSV 2526 matches to round numbers
    df_2526_sched = df_2526.merge(
        schedule,
        left_on=["home", "away"],
        right_on=["home_team", "away_team"],
        how="left",
    )
    unmatched = df_2526_sched["round_number"].isna().sum()
    if unmatched > 0:
        print(f"WARNING: {unmatched} 2025-26 matches could not be mapped to a gameday!")

    max_gd = int(df_2526_sched["round_number"].max())
    print(f"\nGamedays with data: GD1 to GD{max_gd}\n")

    all_predictions = []

    for gd in range(1, max_gd + 1):
        # --- Build training set ---
        # All prior seasons
        train_frames = [df_prior]

        # 2025-26 matches from GDs strictly before current GD
        df_2526_prev = df_2526_sched[df_2526_sched["round_number"] < gd][
            ["date", "home", "away", "hg", "ag", "result"]
        ]
        if len(df_2526_prev) > 0:
            train_frames.append(df_2526_prev)

        train = pd.concat(train_frames, ignore_index=True)
        train = train.dropna(subset=["date", "hg", "ag"]).sort_values("date")

        # Check all GD-N teams appear in training; if not, log a warning
        gd_matches = schedule[schedule["round_number"] == gd]
        gd_teams = set(gd_matches["home_team"]) | set(gd_matches["away_team"])
        train_teams = set(train["home"]) | set(train["away"])
        unseen = gd_teams - train_teams
        if unseen:
            print(f"  GD{gd}: teams with no prior history: {unseen} — including all 2526 data")
            # Fallback: include all available 2025-26 data so far including this GD's earlier part
            extra = df_2526_sched[df_2526_sched["round_number"] <= gd][
                ["date", "home", "away", "hg", "ag", "result"]
            ]
            train = pd.concat([train, extra], ignore_index=True).drop_duplicates()
            train = train.dropna(subset=["date", "hg", "ag"]).sort_values("date")

        # --- Fit model ---
        dates = train["date"].tolist()
        weights = dixon_coles_weights(dates, xi=XI)

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
        except Exception as exc:
            print(f"  GD{gd}: model fit failed — {exc}")
            model_ok = False

        # --- Predict GD-N matches ---
        gd_results = df_2526_sched[df_2526_sched["round_number"] == gd]

        for _, row in gd_matches.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            # Find actual result
            actual_row = gd_results[
                (gd_results["home"] == home_team) & (gd_results["away"] == away_team)
            ]
            actual = actual_row["result"].values[0] if len(actual_row) > 0 else None

            prob_H, prob_D, prob_A = None, None, None
            predicted = None

            if model_ok:
                try:
                    pred = model.predict(home_team, away_team)
                    prob_H = float(pred.home_win)
                    prob_D = float(pred.draw)
                    prob_A = float(pred.away_win)
                    # Predicted result = argmax of probabilities
                    probs = {"H": prob_H, "D": prob_D, "A": prob_A}
                    predicted = max(probs, key=probs.get)
                except Exception as exc:
                    print(f"    GD{gd} {home_team} vs {away_team}: predict failed — {exc}")

            correct = int(predicted == actual) if (predicted is not None and actual is not None) else None

            all_predictions.append(
                {
                    "gameday": int(gd),
                    "home": home_team,
                    "away": away_team,
                    "home_display": NAME_MAP.get(home_team, home_team),
                    "away_display": NAME_MAP.get(away_team, away_team),
                    "prob_H": round(prob_H, 4) if prob_H is not None else None,
                    "prob_D": round(prob_D, 4) if prob_D is not None else None,
                    "prob_A": round(prob_A, 4) if prob_A is not None else None,
                    "predicted": predicted,
                    "actual": actual,
                    "correct": correct,
                }
            )

    return all_predictions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(predictions: list, spi_weekly: dict, spi_overall: float) -> None:
    """Print per-gameday comparison table and overall stats."""

    # Group by gameday
    by_gd: dict[int, list] = {}
    for p in predictions:
        gd = p["gameday"]
        by_gd.setdefault(gd, []).append(p)

    dc_total_correct = 0
    dc_total_evaluated = 0
    all_gds = sorted(by_gd.keys())

    print("=" * 90)
    print(
        f"{'GD':<4} {'DC':>6} {'DC%':>6} {'SPI%':>6} {'Δ':>7}   {'Draw analysis'}"
    )
    print("-" * 90)

    for gd in all_gds:
        matches = by_gd[gd]
        evaluated = [m for m in matches if m["correct"] is not None]
        correct = sum(m["correct"] for m in evaluated)
        n = len(evaluated)
        dc_pct = (correct / n * 100) if n > 0 else 0.0

        dc_total_correct += correct
        dc_total_evaluated += n

        spi_pct = spi_weekly.get(gd, None)
        if spi_pct is not None:
            delta = dc_pct - spi_pct
            delta_str = f"{delta:+.1f}%"
            spi_str = f"{spi_pct:.1f}%"
        else:
            delta_str = "  n/a "
            spi_str = "  n/a"

        # Draw analysis
        dc_draws_pred = sum(1 for m in matches if m["predicted"] == "D")
        actual_draws = sum(1 for m in evaluated if m["actual"] == "D")
        draw_str = f"DC pred {dc_draws_pred}D, actual {actual_draws}D"

        print(
            f"{gd:<4} {correct}/{n:<5} {dc_pct:>5.1f}% {spi_str:>6} {delta_str:>7}   {draw_str}"
        )

    print("=" * 90)

    # Overall
    dc_overall = (dc_total_correct / dc_total_evaluated * 100) if dc_total_evaluated > 0 else 0.0
    spi_gds_evaluated = sum(1 for gd in all_gds if gd in spi_weekly)
    delta_overall = dc_overall - spi_overall

    print(
        f"\nOverall: DC={dc_overall:.1f}% ({dc_total_correct}/{dc_total_evaluated})  "
        f"SPI={spi_overall:.1f}%  Δ={delta_overall:+.1f}%"
    )

    # Draw analysis totals
    all_evaluated = [p for p in predictions if p["correct"] is not None]
    total_dc_draws = sum(1 for p in predictions if p["predicted"] == "D")
    total_actual_draws = sum(1 for p in all_evaluated if p["actual"] == "D")
    dc_draw_correct = sum(
        1 for p in all_evaluated if p["predicted"] == "D" and p["actual"] == "D"
    )
    print(f"\nDraw analysis:")
    print(f"  DC predicted draws : {total_dc_draws} / {len(predictions)} matches")
    print(f"  Actual draws       : {total_actual_draws} / {len(all_evaluated)} matches")
    print(
        f"  DC draw precision  : {dc_draw_correct}/{total_dc_draws} "
        f"({dc_draw_correct/total_dc_draws*100:.1f}% of predicted draws were actual draws)"
        if total_dc_draws > 0
        else "  DC draw precision  : n/a"
    )
    print(
        f"  DC draw recall     : {dc_draw_correct}/{total_actual_draws} "
        f"({dc_draw_correct/total_actual_draws*100:.1f}% of actual draws predicted)"
        if total_actual_draws > 0
        else "  DC draw recall     : n/a"
    )

    # Brier score
    bs_dc = brier_score(all_evaluated)
    print(f"\nBrier score (DC): {bs_dc:.4f}  (lower is better; random baseline ~0.667)")

    # Note on SPI Brier — we don't have per-match probabilities for SPI
    print("Brier score (SPI): not available (no per-match probabilities in dashboard.json)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Dixon-Coles Walk-Forward Backtest — Turkish Super Lig 2025-26")
    print("=" * 60)

    # Load data
    print("\nLoading season data from football-data.co.uk...")
    all_data = load_all_seasons()
    print(f"Total matches loaded: {len(all_data)}")

    print("\nLoading 2025-26 schedule from DuckDB...")
    schedule = load_schedule()
    print(f"Schedule entries: {len(schedule)} ({schedule['round_number'].nunique()} rounds)")

    print("\nLoading SPI accuracy from dashboard.json...")
    spi_weekly, spi_overall = load_spi_accuracy()
    print(f"SPI overall accuracy: {spi_overall:.1f}%  ({len(spi_weekly)} gamedays)")

    # Run backtest
    print("\nRunning walk-forward backtest...")
    predictions = run_backtest(all_data, schedule)
    print(f"\nTotal predictions generated: {len(predictions)}")

    # Print report
    print("\n")
    print_report(predictions, spi_weekly, spi_overall)

    # Save JSON output
    output_path = OUTPUT_JSON
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()
