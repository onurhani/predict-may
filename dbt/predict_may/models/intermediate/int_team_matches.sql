select
    season,
    match_date,
    home_team as team,
    away_team as opponent,
    home_goals as goals_for,
    away_goals as goals_against,
    home_goal_diff as goal_diff,
    home_points as points,
    'home' as venue
from {{ ref('int_fixtures_enriched') }}

union all

select
    season,
    match_date,
    away_team as team,
    home_team as opponent,
    away_goals as goals_for,
    home_goals as goals_against,
    away_goal_diff as goal_diff,
    away_points as points,
    'away' as venue
from {{ ref('int_fixtures_enriched') }}
