with form as (
    select *
    from {{ ref('int_team_form') }}
),

spi as (
    select
        team,
        season,
        match_date,

        -- Base SPI (arbitrary center)
        50

        + (avg_goal_diff_5 * 10)
        + (avg_goals_for_5 * 5)
        - (avg_goals_against_5 * 5)

        as spi_rating

    from form
)

select * from spi
