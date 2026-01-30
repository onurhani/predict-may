-- Match-level prediction features
-- Combines team ratings and actual results

with fixtures as (
    select
        match_id,
        season,
        match_date,
        home_team,
        away_team,
        home_goals,
        away_goals,
        home_points,
        away_points,
        
        case
            when home_goals > away_goals then 'H'
            when home_goals < away_goals then 'A'
            else 'D'
        end as result
        
    from {{ ref('stg_fixtures') }}
),

home_spi as (
    select
        season,
        match_date,
        team,
        spi_rating as home_spi,
        prior_avg_goal_diff_5 as home_form_5,
        prior_season_goals_for as home_attack_strength,
        prior_season_goals_against as home_defense_strength
    from {{ ref('int_team_spi') }}
),

away_spi as (
    select
        season,
        match_date,
        team,
        spi_rating as away_spi,
        prior_avg_goal_diff_5 as away_form_5,
        prior_season_goals_for as away_attack_strength,
        prior_season_goals_against as away_defense_strength
    from {{ ref('int_team_spi') }}
),

combined as (
    select
        f.match_id,
        f.season,
        f.match_date,
        f.home_team,
        f.away_team,
        
        -- Actual result (for training)
        f.result,
        f.home_goals,
        f.away_goals,
        f.home_points,
        f.away_points,
        
        -- Home team features (as of match date)
        h.home_spi,
        h.home_form_5,
        h.home_attack_strength,
        h.home_defense_strength,
        
        -- Away team features (as of match date)
        a.away_spi,
        a.away_form_5,
        a.away_attack_strength,
        a.away_defense_strength,
        
        -- Derived features
        h.home_spi - a.away_spi as spi_difference,
        h.home_form_5 - a.away_form_5 as form_difference,
        
        -- Home advantage (can calculate dynamically or use constant)
        0.3 as assumed_home_advantage
        
    from fixtures f
    
    left join home_spi h
        on f.season = h.season
        and f.match_date = h.match_date
        and f.home_team = h.team
        
    left join away_spi a
        on f.season = a.season
        and f.match_date = a.match_date
        and f.away_team = a.team
)

select *
from combined
where home_spi is not null 
  and away_spi is not null
order by season desc, match_date desc