-- Predictions for upcoming matches
-- Uses calibrated constants from dbt_project.yml
-- Uses latest SPI ratings to forecast results

with future_fixtures as (
    select * from {{ ref('stg_fixtures_future') }}
),

-- Get latest SPI rating for each team
latest_home_spi as (
    select
        team,
        season,
        spi_rating as home_spi,
        prior_avg_goal_diff_5 as home_form_5,
        prior_season_goals_for as home_attack_strength,
        prior_season_goals_against as home_defense_strength,
        
        row_number() over (
            partition by team, season
            order by match_date desc
        ) as recency_rank
        
    from {{ ref('int_team_spi') }}
),

latest_away_spi as (
    select
        team,
        season,
        spi_rating as away_spi,
        prior_avg_goal_diff_5 as away_form_5,
        prior_season_goals_for as away_attack_strength,
        prior_season_goals_against as away_defense_strength,
        
        row_number() over (
            partition by team, season
            order by match_date desc
        ) as recency_rank
        
    from {{ ref('int_team_spi') }}
),

predictions as (
    select
        f.match_id,
        f.season,
        f.match_date,
        f.home_team,
        f.away_team,
        f.fixture_id,
        
        -- Latest ratings for both teams
        h.home_spi,
        a.away_spi,
        h.home_form_5,
        a.away_form_5,
        
        -- Derived features
        h.home_spi - a.away_spi as spi_difference,
        h.home_form_5 - a.away_form_5 as form_difference,
        
        -- Home advantage from constants
        {{ get_constant('home_advantage') }}::float as assumed_home_advantage,
        
        -- Step 1: Calculate draw probability first (higher for evenly matched teams)
        {{ get_constant('base_draw_prob') }} + 
        ({{ get_constant('bonus_draw_prob') }} * exp(-abs(h.home_spi - a.away_spi) / {{ get_constant('draw_decay_rate') }}))
        as prob_draw,
        
        -- Step 2: Home win ratio from logistic function (used to split remaining probability)
        1.0 / (1 + exp(-{{ get_constant('spi_scaling_factor') }} * 
            (h.home_spi - a.away_spi + {{ get_constant('home_advantage') }} * {{ get_constant('home_advantage_multiplier') }}))
        ) as home_win_ratio
        
    from future_fixtures f
    
    left join latest_home_spi h
        on f.season = h.season
        and f.home_team = h.team
        and h.recency_rank = 1
        
    left join latest_away_spi a
        on f.season = a.season
        and f.away_team = a.team
        and a.recency_rank = 1
),

split as (
    select
        *,
        -- Step 3: Split remaining probability (1 - draw) between home and away
        (1.0 - prob_draw) * home_win_ratio as prob_home_win,
        (1.0 - prob_draw) * (1.0 - home_win_ratio) as prob_away_win,
        
        -- Expected goals using constants
        {{ get_constant('expected_goals_home_base') }} + 
        (spi_difference * {{ get_constant('goals_per_spi_point') }}) + 
        (assumed_home_advantage * {{ get_constant('home_advantage_goals') }}) as expected_home_goals,
        
        {{ get_constant('expected_goals_away_base') }} + 
        (-spi_difference * {{ get_constant('goals_per_spi_point') }}) as expected_away_goals
        
    from predictions
),

final as (
    select
        match_id,
        season,
        match_date,
        home_team,
        away_team,
        fixture_id,
        home_spi,
        away_spi,
        home_form_5,
        away_form_5,
        spi_difference,
        form_difference,
        assumed_home_advantage,
        prob_home_win,
        prob_draw,
        prob_away_win,
        expected_home_goals,
        expected_away_goals,
        
        -- Predicted result using improved logic with draw threshold
        case
            when prob_draw >= {{ get_constant('draw_prediction_threshold') }} then 'D'
            when prob_home_win > prob_away_win then 'H'
            else 'A'
        end as predicted_result,
        
        -- Expected points for each team
        (prob_home_win * 3) + (prob_draw * 1) as expected_home_points,
        (prob_away_win * 3) + (prob_draw * 1) as expected_away_points
        
    from split
)

select *
from final
where home_spi is not null 
  and away_spi is not null
order by match_date