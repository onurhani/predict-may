select
    season,
    team,

    count(*) as matches_played,
    sum(points) as points,
    sum(points) * 1.0 / count(*) as ppm,

    sum(goals_for) as goals_for,
    sum(goals_against) as goals_against,
    sum(goal_diff) as goal_diff,
    avg(goal_diff) as avg_goal_diff

from {{ ref('int_team_matches') }}
group by 1, 2
