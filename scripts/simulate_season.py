"""
Monte Carlo simulation for final league standings
Runs 10,000 simulations of remaining matches to project May standings
"""
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict

DB_PATH = "data/football.duckdb"
N_SIMULATIONS = 10000
CURRENT_SEASON = 2025

def get_current_standings(con):
    """Get current points for each team"""
    query = """
    SELECT 
        team,
        SUM(points) as current_points
    FROM main_intermediate.int_team_matches
    WHERE season = ?
    GROUP BY team
    """
    return con.execute(query, [CURRENT_SEASON]).fetchdf().set_index('team')['current_points'].to_dict()

def get_future_match_probabilities(con):
    """Get predictions for all remaining matches"""
    query = """
    SELECT 
        home_team,
        away_team,
        prob_home_win,
        prob_draw,
        prob_away_win
    FROM main_marts.match_predictions_future
    WHERE season = ?
      AND prob_home_win IS NOT NULL
      AND prob_draw IS NOT NULL
      AND prob_away_win IS NOT NULL
      AND prob_home_win >= 0
      AND prob_draw >= 0
      AND prob_away_win >= 0
    ORDER BY match_date
    """
    df = con.execute(query, [CURRENT_SEASON]).fetchdf()
    
    # Validate probabilities
    if len(df) > 0:
        print(f"\n🔍 Probability validation:")
        print(f"   Valid matches: {len(df)}")
        print(f"   Min prob_home_win: {df['prob_home_win'].min():.3f}")
        print(f"   Max prob_home_win: {df['prob_home_win'].max():.3f}")
        prob_sum = df['prob_home_win'] + df['prob_draw'] + df['prob_away_win']
        print(f"   Prob sum range: {prob_sum.min():.3f} - {prob_sum.max():.3f}")
        
        # Check for invalid sums
        invalid = df[abs(prob_sum - 1.0) > 0.01]
        if len(invalid) > 0:
            print(f"   ⚠️ {len(invalid)} matches have probabilities not summing to 1.0")
    
    return df

def simulate_match(prob_home, prob_draw, prob_away):
    """
    Simulate a single match outcome based on probabilities
    Returns: (home_points, away_points)
    """
    # Normalize probabilities to ensure they sum to 1.0
    total = prob_home + prob_draw + prob_away
    prob_home = prob_home / total
    prob_draw = prob_draw / total
    prob_away = prob_away / total
    
    outcome = np.random.choice(['H', 'D', 'A'], p=[prob_home, prob_draw, prob_away])
    
    if outcome == 'H':
        return 3, 0
    elif outcome == 'A':
        return 0, 3
    else:
        return 1, 1

def run_simulation(current_standings, future_matches):
    """
    Run one complete simulation of remaining matches
    Returns: dict of final points by team
    """
    # Start with current standings
    final_standings = current_standings.copy()
    
    # Simulate each remaining match
    for _, match in future_matches.iterrows():
        home_pts, away_pts = simulate_match(
            match['prob_home_win'],
            match['prob_draw'],
            match['prob_away_win']
        )
        
        home_team = match['home_team']
        away_team = match['away_team']
        
        final_standings[home_team] = final_standings.get(home_team, 0) + home_pts
        final_standings[away_team] = final_standings.get(away_team, 0) + away_pts
    
    return final_standings

def calculate_position_probabilities(simulation_results):
    """
    Calculate probability of each team finishing in each position
    """
    position_counts = defaultdict(lambda: defaultdict(int))
    
    for standings in simulation_results:
        # Sort teams by points (descending)
        sorted_teams = sorted(standings.items(), key=lambda x: x[1], reverse=True)
        
        for position, (team, points) in enumerate(sorted_teams, start=1):
            position_counts[team][position] += 1
    
    # Convert counts to probabilities
    n_sims = len(simulation_results)
    position_probs = {}
    
    for team, positions in position_counts.items():
        position_probs[team] = {
            pos: count / n_sims 
            for pos, count in positions.items()
        }
    
    return position_probs

def main():
    print(f"🎲 Running {N_SIMULATIONS:,} season simulations...\n")
    
    con = duckdb.connect(DB_PATH)
    
    # Get current state
    print("📊 Loading current standings...")
    current_standings = get_current_standings(con)
    print(f"   {len(current_standings)} teams")
    
    # Get future matches
    print("📅 Loading future match predictions...")
    future_matches = get_future_match_probabilities(con)
    print(f"   {len(future_matches)} matches remaining")
    
    con.close()
    
    if future_matches.empty:
        print("\n⚠️  No future matches found!")
        print("Run: python fetch_future_fixtures.py")
        print("Then: dbt run --target motherduck")
        return
    
    # Run simulations
    print(f"\n🔄 Simulating {N_SIMULATIONS:,} seasons...")
    simulation_results = []
    
    for i in range(N_SIMULATIONS):
        if (i + 1) % 1000 == 0:
            print(f"   {i + 1:,} / {N_SIMULATIONS:,}")
        
        final_standings = run_simulation(current_standings, future_matches)
        simulation_results.append(final_standings)
    
    # Calculate probabilities
    print("\n📈 Calculating position probabilities...")
    position_probs = calculate_position_probabilities(simulation_results)
    
    # Expected final points
    expected_points = {
        team: np.mean([sim[team] for sim in simulation_results])
        for team in current_standings.keys()
    }
    
    # Create results DataFrame
    results = []
    for team in sorted(expected_points.keys(), key=lambda t: expected_points[t], reverse=True):
        current_pts = current_standings[team]
        expected_pts = expected_points[team]
        
        # Most likely finish position
        most_likely_pos = max(position_probs[team].items(), key=lambda x: x[1])[0]
        
        # Probability of top 4 (Champions League)
        prob_top4 = sum(position_probs[team].get(i, 0) for i in range(1, 5))
        
        # Probability of relegation (bottom 3)
        num_teams = len(current_standings)
        prob_relegation = sum(position_probs[team].get(i, 0) for i in range(num_teams - 2, num_teams + 1))
        
        results.append({
            'Team': team,
            'Current Points': current_pts,
            'Expected Points': round(expected_pts, 1),
            'Most Likely Position': most_likely_pos,
            'Prob Top 4': f"{prob_top4 * 100:.1f}%",
            'Prob Relegation': f"{prob_relegation * 100:.1f}%"
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("🏆 PROJECTED MAY STANDINGS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    con = duckdb.connect(DB_PATH)
    con.register("results_df", results_df)
    con.execute("""
        CREATE OR REPLACE TABLE main_marts.season_projections AS 
        SELECT * FROM results_df
    """)
    con.close()
    
    print("\n✅ Results saved to: main_marts.season_projections")
    print(f"💾 Database: {DB_PATH}")

if __name__ == "__main__":
    main()