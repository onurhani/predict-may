select
    team,
    season,
    match_date,

    avg(goal_diff) over (
        partition by team, season
        order by match_date
        rows between 4 preceding and current row
    ) as avg_goal_diff_5,

    avg(goals_for) over (
        partition by team, season
        order by match_date
        rows between 4 preceding and current row
    ) as avg_goals_for_5,

    avg(goals_against) over (
        partition by team, season
        order by match_date
        rows between 4 preceding and current row
    ) as avg_goals_against_5

from {{ ref('int_team_matches') }}
