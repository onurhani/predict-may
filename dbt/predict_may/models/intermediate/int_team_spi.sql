with base as (

    select
        season,
        match_date,
        team,
        opponent,
        is_home,

        goals_for,
        goals_against,
        goal_diff,

        row_number() over (
            partition by season, team
            order by match_date
        ) as match_number

    from {{ ref('int_team_matches') }}

),

rolling as (

    select
        season,
        match_date,
        team,

        avg(goal_diff) over (
            partition by season, team
            order by match_number
            rows between 4 preceding and current row
        ) as rolling_avg_goal_diff,

        avg(goals_for) over (
            partition by season, team
            order by match_number
            rows between 4 preceding and current row
        ) as rolling_goals_for,

        avg(goals_against) over (
            partition by season, team
            order by match_number
            rows between 4 preceding and current row
        ) as rolling_goals_against

    from base
),

spi as (

    select
        season,
        match_date,
        team,

        rolling_avg_goal_diff,
        rolling_goals_for,
        rolling_goals_against,

        -- SPI v1 formula (intentionally simple & explainable)
        50
        + rolling_avg_goal_diff * 10
        + rolling_goals_for * 5
        - rolling_goals_against * 5
        as spi_rating

    from rolling
)

select *
from spi
