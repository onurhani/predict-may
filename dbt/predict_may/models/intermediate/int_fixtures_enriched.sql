with base as (

    select
        match_id,
        season,
        match_date,
        home_team,
        away_team,
        home_goals,
        away_goals,

        case
            when home_goals > away_goals then 'H'
            when home_goals < away_goals then 'A'
            else 'D'
        end as result,

        case
            when home_goals > away_goals then 3
            when home_goals = away_goals then 1
            else 0
        end as home_points,

        case
            when away_goals > home_goals then 3
            when away_goals = home_goals then 1
            else 0
        end as away_points,

        home_goals - away_goals as home_goal_diff,
        away_goals - home_goals as away_goal_diff

    from {{ ref('stg_fixtures') }}

)

select * from base
