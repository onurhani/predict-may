-- Historical match predictions with actual results
-- Uses calibrated constants from dbt_project.yml
-- Used for backtesting model accuracy

with match_features as (
    select * from {{ ref('int_match_features') }}
),

predictions as (
    select
        match_id,
        season,
        match_date,
        home_team,
        away_team,
        
        -- ACTUAL RESULTS
        result as actual_result,
        home_goals as actual_home_goals,
        away_goals as actual_away_goals,
        home_points as actual_home_points,
        away_points as actual_away_points,
        
        -- Features
        home_spi,
        away_spi,
        home_form_5,
        away_form_5,
        spi_difference,
        form_difference,
        assumed_home_advantage,
        
        -- Step 1: Calculate draw probability first (higher for evenly matched teams)
        {{ get_constant('base_draw_prob') }} + 
        ({{ get_constant('bonus_draw_prob') }} * exp(-abs(home_spi - away_spi) / {{ get_constant('draw_decay_rate') }}))
        as prob_draw,
        
        -- Step 2: Home win ratio from logistic function (used to split remaining probability)
        1.0 / (1 + exp(-{{ get_constant('spi_scaling_factor') }} * 
            (home_spi - away_spi + assumed_home_advantage * {{ get_constant('home_advantage_multiplier') }}))
        ) as home_win_ratio
        
    from match_features
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
        
        -- Actual results
        actual_result,
        actual_home_goals,
        actual_away_goals,
        actual_home_points,
        actual_away_points,
        
        -- Features
        home_spi,
        away_spi,
        spi_difference,
        
        -- Predictions
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
        
        -- Expected points
        (prob_home_win * 3) + (prob_draw * 1) as expected_home_points,
        (prob_away_win * 3) + (prob_draw * 1) as expected_away_points,
        
        -- Accuracy flag
        case
            when prob_draw >= {{ get_constant('draw_prediction_threshold') }} and actual_result = 'D' then 1
            when prob_draw < {{ get_constant('draw_prediction_threshold') }} and prob_home_win > prob_away_win and actual_result = 'H' then 1
            when prob_draw < {{ get_constant('draw_prediction_threshold') }} and prob_away_win > prob_home_win and actual_result = 'A' then 1
            else 0
        end as correct_prediction
        
    from split
)

select * from final
order by season desc, match_date desc