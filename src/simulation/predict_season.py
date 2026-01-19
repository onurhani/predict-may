import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

DB_PATH = "data/football.duckdb"
NUM_SIMULATIONS = 5000

def run_simulation():
    # 1. Connect to DuckDB
    con = duckdb.connect(DB_PATH)

    # 2. Fetch current standings (Actual Points)
    # This assumes you have populated marts.team_season_stats
    standings_df = con.execute("SELECT team, points FROM marts.team_season_stats").df()
    current_points = dict(zip(standings_df['team'], standings_df['points']))

    # 3. Fetch latest SPI ratings for each team
    # We take the most recent rating per team
    spi_df = con.execute("""
        SELECT team, spi_rating 
        FROM marts.team_spi 
        QUALIFY row_number() OVER (PARTITION BY team ORDER BY match_date DESC) = 1
    """).df()
    ratings = dict(zip(spi_df['team'], spi_df['spi_rating']))

    # 4. Fetch remaining fixtures (where goals are NULL or status is not 'FT')
    remaining_fixtures = con.execute("""
        SELECT home_team, away_team 
        FROM stg_fixtures 
        WHERE home_goals IS NULL
    """).df()

    print(f"Starting {NUM_SIMULATIONS} simulations for {len(remaining_fixtures)} matches...")

    # Storage for simulation results
    final_positions = []

    for i in tqdm(range(NUM_SIMULATIONS)):
        # Start each sim with current actual points
        sim_points = current_points.copy()

        for _, match in remaining_fixtures.iterrows():
            home, away = match['home_team'], match['away_team']
            
            # Get SPIs (defaulting to 50 if team not found)
            h_spi = ratings.get(home, 50.0)
            a_spi = ratings.get(away, 50.0)

            # Simple Lambda (λ) Calculation:
            # Base goals + SPI difference adjustment
            # Every 10 SPI points diff = ~0.2 goals advantage
            home_exp = 1.3 + (h_spi - a_spi) / 50.0
            away_exp = 1.1 + (a_spi - h_spi) / 50.0

            # Poisson roll for goals
            h_goals = np.random.poisson(max(0.1, home_exp))
            a_goals = np.random.poisson(max(0.1, away_exp))

            # Assign points
            if h_goals > a_goals:
                sim_points[home] = sim_points.get(home, 0) + 3
            elif h_goals < a_goals:
                sim_points[away] = sim_points.get(away, 0) + 3
            else:
                sim_points[home] = sim_points.get(home, 0) + 1
                sim_points[away] = sim_points.get(away, 0) + 1

        # Sort teams by points to find final ranks
        sorted_teams = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
        for rank, (team, pts) in enumerate(sorted_teams, 1):
            final_positions.append({'sim_id': i, 'team': team, 'rank': rank, 'final_pts': pts})

    # 5. Aggregate Results
    results_df = pd.DataFrame(final_positions)
    
    # Calculate Probabilities (e.g., % chance of finishing 1st)
    projections = results_df.groupby('team').agg(
        avg_finish_pts=('final_pts', 'mean'),
        win_league_pct=('rank', lambda x: (x == 1).mean() * 100),
        top_4_pct=('rank', lambda x: (x <= 4).mean() * 100),
        relegation_pct=('rank', lambda x: (x >= 16).mean() * 100) # Assuming bottom 4 relegate
    ).reset_index()

    # 6. Save to DuckDB for your Website
    con.execute("CREATE SCHEMA IF NOT EXISTS marts")
    con.register("projections_df", projections)
    con.execute("CREATE OR REPLACE TABLE marts.season_projections AS SELECT * FROM projections_df")
    
    print("✅ Simulation complete. Results saved to marts.season_projections.")
    con.close()

if __name__ == "__main__":
    run_simulation()



2423243200