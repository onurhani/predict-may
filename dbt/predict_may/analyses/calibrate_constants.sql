-- Run this query to calculate proper constants based on your actual Turkish Süper Lig data
-- Copy the results into dbt_project.yml vars section

WITH league_stats AS (
    SELECT 
        -- Expected goals parameters
        AVG(home_goals) as avg_home_goals,
        AVG(away_goals) as avg_away_goals,
        AVG(home_goals) - AVG(away_goals) as home_advantage_goals,
        
        -- Outcome frequencies
        COUNT(*) as total_matches,
        SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) as home_wins,
        SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) as draws,
        SUM(CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END) as away_wins,
        
        -- Average goal difference
        AVG(ABS(home_goals - away_goals)) as avg_goal_diff
        
    FROM main_staging.stg_fixtures
    WHERE season >= 2021  -- Use last few seasons for calibration
),

calculated_constants AS (
    SELECT 
        -- Expected goals (use directly)
        ROUND(avg_home_goals, 2) as expected_goals_home_base,
        ROUND(avg_away_goals, 2) as expected_goals_away_base,
        ROUND(home_advantage_goals, 2) as home_advantage_goals,
        
        -- Draw probability (use actual frequency)
        ROUND(draws::float / total_matches, 3) as base_draw_prob,
        
        -- Outcome percentages for reference
        ROUND(100.0 * home_wins / total_matches, 1) as home_win_pct,
        ROUND(100.0 * draws / total_matches, 1) as draw_pct,
        ROUND(100.0 * away_wins / total_matches, 1) as away_win_pct,
        
        total_matches
        
    FROM league_stats
)

SELECT 
    '# Copy these values to dbt_project.yml vars section' as instruction,
    '' as blank_line_1,
    '## Expected Goals Parameters' as section_1,
    'expected_goals_home_base: ' || expected_goals_home_base as constant_1,
    'expected_goals_away_base: ' || expected_goals_away_base as constant_2,
    'home_advantage_goals: ' || home_advantage_goals as constant_3,
    '' as blank_line_2,
    '## Draw Probability Parameters' as section_2,
    'base_draw_prob: ' || base_draw_prob as constant_4,
    'bonus_draw_prob: 0.18  # Keep at 0.18 (allows max ~' || 
        ROUND((base_draw_prob + 0.18) * 100, 0) || '% draw prob for evenly matched teams)' as constant_5,
    'draw_prediction_threshold: ' || 
        ROUND(base_draw_prob + 0.05, 2) || '  # Predict draw when prob >= this' as constant_6,
    '' as blank_line_3,
    '## Actual League Statistics (for reference)' as section_3,
    'Home wins: ' || home_win_pct || '%' as stat_1,
    'Draws: ' || draw_pct || '%' as stat_2,
    'Away wins: ' || away_win_pct || '%' as stat_3,
    'Total matches analyzed: ' || total_matches as stat_4
    
FROM calculated_constants;