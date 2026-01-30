-- Staging model for future fixtures (not yet played)

with source_table as (
    select * from {{ source('football', 'fixtures') }}
    where status IN ('NS', 'TBD', 'PST')  -- Only upcoming matches
)

select
    {{ dbt_utils.generate_surrogate_key(['date', 'home_team', 'away_team']) }} as match_id,
    cast(season as integer) as season,
    cast(date as date) as match_date,
    
    trim(home_team) as home_team,
    trim(away_team) as away_team,
    
    status,
    fixture_id

from source_table