-- Complete season statistics by team
-- This is the main analytical table for understanding team performance

with team_matches as (
    select * from {{ ref('int_team_matches') }}
),

aggregated as (
    select
        season,
        team,
        
        -- Match counts
        count(*) as matches_played,
        sum(case when is_home = 1 then 1 else 0 end) as home_matches,
        sum(case when is_home = 0 then 1 else 0 end) as away_matches,
        
        -- Points
        sum(points) as total_points,
        sum(case when is_home = 1 then points else 0 end) as home_points,
        sum(case when is_home = 0 then points else 0 end) as away_points,
        
        -- Wins/Draws/Losses
        sum(case when points = 3 then 1 else 0 end) as wins,
        sum(case when points = 1 then 1 else 0 end) as draws,
        sum(case when points = 0 then 1 else 0 end) as losses,
        
        -- Goals
        sum(goals_for) as goals_for,
        sum(goals_against) as goals_against,
        sum(goal_diff) as goal_difference,
        
        -- Home/away splits
        sum(case when is_home = 1 then goals_for else 0 end) as home_goals_for,
        sum(case when is_home = 1 then goals_against else 0 end) as home_goals_against,
        sum(case when is_home = 0 then goals_for else 0 end) as away_goals_for,
        sum(case when is_home = 0 then goals_against else 0 end) as away_goals_against
        
    from team_matches
    group by season, team
),

with_rates as (
    select
        *,
        
        -- Points per game
        total_points::float / matches_played as ppg,
        home_points::float / nullif(home_matches, 0) as home_ppg,
        away_points::float / nullif(away_matches, 0) as away_ppg,
        
        -- Goals per game
        goals_for::float / matches_played as goals_for_pg,
        goals_against::float / matches_played as goals_against_pg,
        
        -- Win rate
        wins::float / matches_played as win_rate,
        
        -- Form rating (simple version)
        (total_points::float / matches_played) * 100 / 3 as form_pct
        
    from aggregated
)

select * from with_rates
order by season desc, total_points desc