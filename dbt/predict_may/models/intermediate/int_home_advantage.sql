select
    avg(goals_for - goals_against) as home_goal_advantage
from {{ ref('int_team_matches') }}
where is_home = 1
