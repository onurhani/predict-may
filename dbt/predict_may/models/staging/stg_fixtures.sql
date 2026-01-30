-- Staging model for COMPLETED fixtures only
-- Used for training models

with source_table as (
    select * from {{ source('football', 'fixtures') }}
    where status = 'FT'  -- Only finished matches
)

select
    {{ dbt_utils.generate_surrogate_key(['date', 'home_team', 'away_team']) }} as match_id,
    cast(season as integer) as season,
    cast(date as date) as match_date,

    trim(home_team) as home_team,
    trim(away_team) as away_team,

    cast(home_goals as integer) as home_goals,
    cast(away_goals as integer) as away_goals,

    -- points
    case
        when home_goals > away_goals then 3
        when home_goals = away_goals then 1
        else 0
    end as home_points,

    case
        when away_goals > home_goals then 3
        when home_goals = away_goals then 1
        else 0
    end as away_points

from source_table