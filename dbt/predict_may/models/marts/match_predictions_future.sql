-- Predictions for upcoming matches
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
        
        -- Home advantage
        0.3 as assumed_home_advantage,
        
        -- Calculate raw win probabilities using logistic function
        1.0 / (1 + exp(-0.04 * (h.home_spi - a.away_spi + 0.3 * 10))) as raw_prob_home_win,
        1.0 / (1 + exp(-0.04 * (a.away_spi - h.home_spi))) as raw_prob_away_win,
        
        -- Draw probability based on match closeness (closer matches = higher draw probability)
        0.20 + (0.15 * exp(-abs(h.home_spi - a.away_spi) / 20.0)) as raw_prob_draw
        
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

normalized as (
    select
        *,
        raw_prob_home_win + raw_prob_away_win + raw_prob_draw as prob_sum,
        
        -- Normalize to ensure probabilities sum to 1.0
        raw_prob_home_win / (raw_prob_home_win + raw_prob_away_win + raw_prob_draw) as prob_home_win,
        raw_prob_draw / (raw_prob_home_win + raw_prob_away_win + raw_prob_draw) as prob_draw,
        raw_prob_away_win / (raw_prob_home_win + raw_prob_away_win + raw_prob_draw) as prob_away_win,
        
        -- Expected goals
        1.5 + (spi_difference * 0.02) + (assumed_home_advantage * 0.5) as expected_home_goals,
        1.3 + (-spi_difference * 0.02) as expected_away_goals
        
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
        
        -- Most likely result
        case
            when prob_home_win > prob_draw and prob_home_win > prob_away_win then 'H'
            when prob_away_win > prob_draw and prob_away_win > prob_home_win then 'A'
            else 'D'
        end as predicted_result,
        
        -- Expected points for each team
        (prob_home_win * 3) + (prob_draw * 1) as expected_home_points,
        (prob_away_win * 3) + (prob_draw * 1) as expected_away_points
        
    from normalized
)

select *
from final
where home_spi is not null 
  and away_spi is not null
order by match_date