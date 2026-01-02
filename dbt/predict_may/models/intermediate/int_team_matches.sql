with base as (

    select
        season,
        match_date,
        home_team,
        away_team,
        home_goals,
        away_goals
    from {{ ref('stg_fixtures') }}

),

home as (

    select
        season,
        match_date,
        home_team as team,
        away_team as opponent,
        1 as is_home,
        home_goals as goals_for,
        away_goals as goals_against
    from base

),

away as (

    select
        season,
        match_date,
        away_team as team,
        home_team as opponent,
        0 as is_home,
        away_goals as goals_for,
        home_goals as goals_against
    from base

)

select
    season,
    match_date,
    team,
    opponent,
    is_home,
    goals_for,
    goals_against,
    goals_for - goals_against as goal_diff,

    case
        when goals_for > goals_against then 3
        when goals_for = goals_against then 1
        else 0
    end as points

from home

union all

select
    season,
    match_date,
    team,
    opponent,
    is_home,
    goals_for,
    goals_against,
    goals_for - goals_against as goal_diff,

    case
        when goals_for > goals_against then 3
        when goals_for = goals_against then 1
        else 0
    end as points

from away
