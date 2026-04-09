# Turkish Super Lig Projections
```sql projections
SELECT 
  ROW_NUMBER() OVER (ORDER BY "Expected Points" DESC) as Position,
  Team,
  "Current Points" as current_pts,
  ROUND("Expected Points", 1) as projected_pts,
  "Prob Top 4" as top4_prob,
  -- Show relegation % only if > 90%, make negative for red coloring
  CASE WHEN CAST(REPLACE("Prob Relegation", '%', '') AS FLOAT) > 90
       THEN -1 * CAST(REPLACE("Prob Relegation", '%', '') AS FLOAT)
       ELSE null END as rel_highlight
FROM season_projections
ORDER BY "Expected Points" DESC
```

<DataTable data={projections}>
  <Column id=Position align=center />
  <Column id=Team align=left />
  <Column id=current_pts title="Current Pts" fmt='#,##0' align=right />
  <Column id=projected_pts title="Projected Pts" fmt='#,##0.0' align=right />
  <Column id=top4_prob title="Top 4 %" align=right />
  <Column id=rel_highlight title="Rel %" contentType=delta fmt='0.0"%"' align=right />
</DataTable>

<BarChart 
  data={projections} 
  x=Team 
  y=projected_pts 
  title="Projected Final Points"
/>