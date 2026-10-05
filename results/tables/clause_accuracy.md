### Clause-level accuracy (Spider dev, gold queries containing the clause; our measurements)

| Clause | n | full | vanilla | no_linking | simple | no_grammar |
|---|---|---|---|---|---|---|
| SELECT | 1034 | 85.0 | 73.5 | 80.2 | 52.0 | 25.2 |
| WHERE | 478 | 75.3 | 53.8 | 56.1 | 25.9 | 26.2 |
| JOIN | 378 | 73.8 | 29.6 | 55.3 | 10.8 | 13.8 |
| GROUP BY | 271 | 79.7 | 62.7 | 79.0 | 49.4 | 12.2 |
| HAVING | 75 | 78.7 | 77.3 | 77.3 | 72.0 | 2.7 |
| ORDER BY | 231 | 77.5 | 73.6 | 75.8 | 51.5 | 13.0 |
| NESTED | 159 | 44.7 | 23.3 | 34.6 | 8.8 | 8.2 |
| AGGREGATION | 551 | 71.7 | 64.6 | 67.2 | 40.5 | 20.7 |
