# Turkish Super Lig Projections
```sql projections
SELECT 
  ROW_NUMBER() OVER (ORDER BY "Expected Points" DESC) as Position,
  Team,
  "Current Points" as current_pts,
  ROUND("Expected Points", 1) as projected_pts,
  "Prob Top 4" as top4_prob,
  "Prob Relegation" as rel_prob
FROM main_marts.season_projections
ORDER BY "Expected Points" DESC
```

<DataTable data={projections} />

<BarChart 
  data={projections} 
  x=Team 
  y=projected_pts 
  title="Projected Final Points"
/>