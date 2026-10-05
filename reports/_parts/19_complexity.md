## 19. Complexity analysis

{{table:hardness_results.md}}

{{table:clause_accuracy.md}}

{{table:complexity_results.md}}

**Where reasoning breaks down.**

* *Joins are the main difficulty axis.* The controlled FULL model's EM falls from 71.5 % on
  single-table queries to 54.2 % with two tables, 18.3 % with three and 0 / 6 with four or more;
  by number of joins 70.6 → 54.1 → 25.0 → 12.5 %. Without relations (vanilla) accuracy collapses at
  two tables (16.3 % EM, 13.5 % EX): relation-aware attention adds +24.5 EM on single-table and +37.9
  EM on two-table queries. At the clause level, JOIN accuracy rises from 29.6 % (vanilla) to 73.8 %
  (FULL).
* *Nesting and set operations* remain hard for every model: 41.0 % EM on queries with a sub-query
  and 35.0 % with INTERSECT / UNION / EXCEPT (FULL); NESTED clause accuracy 44.7 %.
* *Conditions:* one condition is as easy as none (65.2 vs 65.8 % EM), two conditions drop to 41.5 %,
  three or more (17 examples) are never exact.
* *Aggregation, GROUP BY and ORDER BY* cost only a few points each (e.g. 63.9 → 54.5 % EM with
  GROUP BY): clause-level operators are learned well; multi-table structure is not.
* *Difficulty levels* (final model, dev): 83.9 / 67.7 / 52.9 / 41.6 % EM on easy / medium / hard / extra.

Plots: `results/plots/complexity_em.png`, `results/plots/complexity_ex.png`.
