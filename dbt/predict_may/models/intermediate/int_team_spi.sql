-- Team strength ratings over time using SPI-style approach

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
        points,
        
        row_number() over (
            partition by season, team
            order by match_date
        ) as match_number
        
    from {{ ref('int_team_matches') }}
),

-- Calculate rolling metrics using prior matches
rolling_form as (
    select
        season,
        match_date,
        team,
        match_number,
        
        -- Prior 5 matches (excluding current match)
        avg(goal_diff) over (
            partition by season, team
            order by match_number
            rows between 5 preceding and 1 preceding
        ) as prior_avg_goal_diff_5,
        
        avg(goals_for) over (
            partition by season, team
            order by match_number
            rows between 5 preceding and 1 preceding
        ) as prior_avg_goals_for_5,
        
        avg(goals_against) over (
            partition by season, team
            order by match_number
            rows between 5 preceding and 1 preceding
        ) as prior_avg_goals_against_5,
        
        -- Season-to-date (excluding current match)
        avg(goal_diff) over (
            partition by season, team
            order by match_number
            rows between unbounded preceding and 1 preceding
        ) as prior_season_goal_diff,
        
        avg(goals_for) over (
            partition by season, team
            order by match_number
            rows between unbounded preceding and 1 preceding
        ) as prior_season_goals_for,
        
        avg(goals_against) over (
            partition by season, team
            order by match_number
            rows between unbounded preceding and 1 preceding
        ) as prior_season_goals_against
        
    from base
),

spi_calculated as (
    select
        season,
        match_date,
        team,
        match_number,
        
        prior_avg_goal_diff_5,
        prior_avg_goals_for_5,
        prior_avg_goals_against_5,
        prior_season_goal_diff,
        prior_season_goals_for,
        prior_season_goals_against,
        
        -- SPI Rating (v1 formula - simple & explainable)
        -- Base of 50, then adjust by recent form
        case
            when match_number <= 5 then
                -- Early season: use season-to-date
                50 
                + coalesce(prior_season_goal_diff * 10, 0)
                + coalesce(prior_season_goals_for * 5, 0)
                - coalesce(prior_season_goals_against * 5, 0)
            else
                -- After 5 matches: use rolling 5-match form
                50
                + coalesce(prior_avg_goal_diff_5 * 10, 0)
                + coalesce(prior_avg_goals_for_5 * 5, 0)
                - coalesce(prior_avg_goals_against_5 * 5, 0)
        end as spi_rating,
        
        -- Confidence indicator
        case
            when match_number <= 3 then 'Low'
            when match_number <= 10 then 'Medium'
            else 'High'
        end as rating_confidence
        
    from rolling_form
)

select *
from spi_calculated
order by season desc, match_date desc, team