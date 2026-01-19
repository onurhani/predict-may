with base as (
    select
        season,
        match_date,
        team,
        opponent,
        is_home,
        goals_for,
        goals_against,
        -- Calculate Goal Difference
        goals_for - goals_against as goal_diff,
        -- Apply a "cap" to goal difference to reduce noise from blowouts
        case 
            when (goals_for - goals_against) > 3 then 3
            when (goals_for - goals_against) < -3 then -3
            else (goals_for - goals_against)
        end as capped_goal_diff
    from {{ ref('int_team_matches') }}
),

rolling_metrics as (
    select
        *,
        -- Use a larger window (10 games) for mid-season stability
        avg(capped_goal_diff) over (
            partition by team 
            order by match_date 
            rows between 10 preceding and 1 preceding
        ) as rolling_gdiff_10,

        avg(goals_for) over (
            partition by team 
            order by match_date 
            rows between 10 preceding and 1 preceding
        ) as rolling_gf_10
    from base
),

spi_calc as (
    select
        season,
        match_date,
        team,
        rolling_gdiff_10,
        rolling_gf_10,

        -- THE FORMULA
        -- 50 is the league average baseline
        -- We multiply GDiff by 15 to create a spread
        -- We subtract a small "Home Neutralizer" if they were home, to see true strength
        50 
        + (rolling_gdiff_10 * 15) 
        + (case when is_home = 1 then -0.3 else 0.3 end) as adjusted_spi
    from rolling_metrics
)

select * from spi_calc