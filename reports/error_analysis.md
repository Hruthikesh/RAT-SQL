# Error analysis

All numbers are **our measurements** on Spider dev (1,034 examples) for the controlled models
(BERT-small encoder, 20 epochs, seed 42). The FULL model is correct (EM *and* EX) on 602 of 1,034
examples; the remaining **432 failures** are analysed below. Per-failure details (question, gold SQL,
predicted SQL, labels, automatic explanation): `results/tables/error_analysis_full.csv`.
Notebook: `notebooks/error_analysis.ipynb`.

## Method

A prediction is a *failure* unless it is **both** an exact match and an execution match. Every
failure of the controlled FULL model on Spider dev is labelled automatically by
`ratsql/analysis/errors.py` (structural comparison of the parsed gold and predicted SQL plus the
question's schema links); `scripts/run_error_analysis.py` writes one row per failure
(question, gold SQL, predicted SQL, all labels, primary label, automatic explanation) to
`results/tables/error_analysis_full.csv`. The same procedure is applied to the baselines for the
distribution table.

| # | Category | Rule (evaluated on the whole query tree) |
|---|---|---|
| 1 | schema-linking error | wrong schema element although the question names the right one **exactly** (EM name link or exact cell-value link) — or the model used an exactly-named element that gold does not use (misleading/ambiguous link) |
| 2 | wrong table | table multiset differs (no strong linking evidence) |
| 3 | wrong column | column multiset or SELECT list differs (missing, extra, duplicated items) |
| 4 | wrong join | same tables, different join conditions or number of table units |
| 5 | wrong value | structure is an exact match but the execution result differs (literal, LIMIT value, DISTINCT) |
| 6 | aggregation error | same columns, different aggregation functions |
| 7 | WHERE error | WHERE operators / connectors / number of conditions differ |
| 8 | GROUP BY error | GROUP BY / HAVING component differs |
| 9 | ORDER BY error | direction, keys or LIMIT presence differ |
| 10 | nested-query error | sub-query / set-operation structure differs |
| 11 | grammar error | no complete, well-formed AST (never happens with grammar constraints) |
| 12 | semantic-reasoning error | none of the structural rules explains the failure |

The *primary* label follows the priority grammar → nested → schema-linking → table → join →
column → aggregation → WHERE → GROUP BY → ORDER BY → value → semantic; all labels are kept in
`error_labels`. The "strong evidence" restriction for category 1 is deliberate: partial name matches
are attached to almost every column (precision 11.6 % on dev), so counting them would label nearly
every column error as a linking error.

Validation of the labeller: during development we inspected random samples per category
(fixed seed) and fixed two problems: SELECT-list multiplicity errors were initially falling into
"semantic", and partial-match links made category 1 uninformative. Samples in the notebook
`notebooks/error_analysis.ipynb` are drawn uniformly at random per category — nothing is
cherry-picked.


## Quantitative results

#### Primary error category (Spider dev failures, our measurements)

| Category | full | no_grammar | no_linking | simple | vanilla |
|---|---|---|---|---|---|
| grammar_error | 0 | 754 | 2 | 0 | 0 |
| nested_query_error | 56 | 9 | 86 | 94 | 78 |
| schema_linking_error | 107 | 22 | 229 | 680 | 366 |
| wrong_table | 52 | 12 | 68 | 108 | 129 |
| wrong_join | 43 | 4 | 28 | 30 | 69 |
| wrong_column | 85 | 8 | 91 | 32 | 63 |
| aggregation_error | 22 | 1 | 9 | 1 | 6 |
| where_error | 21 | 7 | 22 | 2 | 12 |
| group_by_error | 11 | 0 | 3 | 2 | 3 |
| order_by_error | 15 | 1 | 10 | 0 | 9 |
| wrong_value | 19 | 11 | 17 | 0 | 7 |
| semantic_reasoning_error | 1 | 0 | 0 | 0 | 0 |
| **total failures** | 432 | 829 | 565 | 949 | 742 |

Labels: `full` = FULL model, `no_linking` = Baseline C, `vanilla` = Baseline B, `simple` =
Baseline A, `no_grammar` = ablation H. `results/plots/error_distribution.png` shows the same data.

**Reading the distribution.**

* *FULL* — the largest primary categories are schema-linking errors (107, 25 % of failures),
  wrong columns (85), nested-query structure (56), wrong tables (52) and wrong joins (43). Errors in
  individual clauses (aggregation, WHERE, GROUP BY, ORDER BY) are comparatively rare (≤ 22 each).
* *Removing linking relations* (Baseline C) doubles schema-linking errors (107 → 229) while leaving
  clause-level categories roughly unchanged — the lost accuracy is almost entirely linking.
* *Removing relation-aware attention* (Baseline B) mostly adds schema-linking (366) and wrong-table /
  wrong-join errors (129 / 69): without schema-graph relations the model cannot tell which tables must
  be joined.
* *Without the grammar mask*, 754 of 1,034 outputs are not even well-formed trees (grammar errors);
  with the mask, grammar errors are impossible (the 2 grammar errors of Baseline C are decoding runs
  that hit the 250-step limit).

#### FULL failures by Spider difficulty (primary category)

| Category | easy | medium | hard | extra |
|---|---|---|---|---|
| schema_linking_error | 12 | 52 | 21 | 22 |
| wrong_column | 12 | 45 | 12 | 16 |
| nested_query_error | 5 | 5 | 22 | 24 |
| wrong_table | 3 | 15 | 17 | 17 |
| wrong_join | 0 | 12 | 11 | 20 |
| aggregation_error | 3 | 18 | 1 | 0 |
| where_error | 4 | 10 | 6 | 1 |
| wrong_value | 5 | 8 | 4 | 2 |
| order_by_error | 4 | 9 | 2 | 0 |
| group_by_error | 0 | 7 | 1 | 3 |
| semantic_reasoning_error | 0 | 0 | 0 | 1 |

Nested-query and join errors concentrate in *hard* / *extra* queries; linking and column errors are
spread over all levels.

#### Clause-level accuracy (Spider dev, gold queries containing the clause; our measurements)

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

JOIN (multi-table gold queries) and NESTED are the weakest clauses of every model; relation-aware
attention (full vs vanilla) improves JOIN accuracy from 29.6 % to 73.8 % and NESTED from 23.3 % to
44.7 %, the largest clause-level gains.

#### Complexity

Accuracy falls steeply with the number of tables (FULL EM: 71.5 % with one table, 54.2 % with two,
18.3 % with three, 0 / 6 with four or more) and with set operations (35.0 % EM) and nested
sub-queries (41.0 %). The vanilla model collapses already at two tables (16.3 % EM, 13.5 % EX).
Full table: `results/tables/complexity_results.md`; plots `results/plots/complexity_em.png`,
`complexity_ex.png`.

## Manual inspection (30 failures drawn at random)

We read all 30 failures returned by `d.sample(30, random_state=7)` on
`results/tables/error_analysis_full.csv` (the controlled FULL model). Counts are ours from reading
them; a failure can show more than one pattern, and the primary pattern is counted.

| Pattern | Count / 30 | Examples (abridged) |
|---|---|---|
| Missing table in FROM (often: a column used from a table that is not joined) | 11 | "*models produced after 1980*" → `FROM model_list WHERE cars_data.Year > 1980`; "*number of documents using each type*" → no `Documents` join; "*most frequent source airport*" → no `flights` join |
| Comparison / LIKE operator | 4 | "*letter 'w' in it*" → `= 'w'` instead of `LIKE '%w%'`; `>=` vs `>`; `<` instead of `>` in the second INTERSECT branch |
| Ambiguous similarly named elements | 3 | "*names of tournaments*" → `winner_name`; "*names and ids of makers*" → `Maker` instead of `FullName`, `Id` |
| Aggregation function | 3 | "*average and maximum age*" → `avg, min` (the same question paraphrased twice); missing `min(grade)` |
| Superlatives / world knowledge | 3 | "*3 most highly rated*" → `GROUP BY … ORDER BY count(*)`; "*largest area*" → `LifeExpectancy` instead of `SurfaceArea`; "*youngest dog*" |
| Value bound to the wrong column / numbers read as values | 2 | "*republics*" → `Continent = 'Republic'` although value linking linked it to `GovernmentForm`; "*line 1 and line 2*" → `WHERE line_1 = 1 AND line_1 = 2` |
| Literal value | 1 | `abandoned_yn = 0` although the question says "*1 stands for yes*" |
| Missing condition | 1 | "*has high definition TV*" → no WHERE clause |
| Equivalent formulation (execution-correct) | 1 | `GROUP BY T2.Singer_ID` instead of `GROUP BY T1.Name` |
| **Gold query wrong, prediction right** | 1 | "*student named Timmothy Ward*": the database stores `'Timmothy'`; the gold query's `'timmothy'` returns no rows (SQLite `=` is case-sensitive), our `'Timmothy'` returns the correct phone number — counted as an EX failure by the metric |

Three sampled gold queries are questionable: the Timmothy case above (verified against the
database), "*at least 10 flights*" annotated as `> 10`, and "*youngest dog*" annotated with
`max(age)`. Spider contains annotation noise, so single-example judgements need care and a small
part of the measured error is not model error.

Across **all** 432 failures (not just the sample): 77 predictions fail to execute with
"no such column" — a column from a table that never entered FROM; 33 predictions are exact matches
but fail execution (13 because of wrong join conditions, 19 because of literal values), and 57 are
execution-correct without being exact matches.

**Take-aways.**

1. *FROM / join construction is the largest error source* (11 of 30 sampled failures; 77
   non-executable predictions overall). The grammar guarantees a well-formed tree but not a
   schema-consistent FROM clause. RAT-SQL's unparser avoids this by deriving FROM and ON from the
   foreign-key graph; our join-inference variant does the same and raises EX from 63.7 to 66.6 and
   executable SQL from 92.4 % to 98.6 % without retraining (separately reported).
2. *Schema-linking evidence is usually present; disambiguation fails.* In the linking-type failures
   the correct element was linked (often exactly) but the model preferred a similarly named one
   (`Maker` vs `FullName`, `winner_name` vs `tourney_name`) or attached a linked value to the wrong
   column.
3. *The remainder needs reasoning*: operators implied by wording (*at least*, *contains*),
   superlatives, and question-specific conventions ("*1 stands for yes*"). These are the failures a
   larger pretrained encoder is most likely to reduce (cf. the paper's +7 EM from BERT-large).


## Fixing the dominant error class with the schema graph (no retraining)

Because FROM / join construction is the dominant execution failure, we re-serialised every
decoded query with foreign-key join inference (`ratsql/sql/join_inference.py`: FROM completed with
the tables of all referenced columns, ON clauses from shortest foreign-key paths — as RAT-SQL's
unparser does). This is a separate, clearly labelled variant; the main tables report the as-decoded
predictions.

#### Schema-graph join inference at serialisation time (Spider dev, our measurements)

| Model | Variant | EM (%) | EX (%) | SQL valid (%) | Rewritten queries |
|---|---|---|---|---|---|
| full | as decoded | 61.4 | 63.7 | 92.4 | 0 |
| full | + FK join inference | 62.8 | 66.6 | 98.6 | 156 |
| vanilla | as decoded | 33.1 | 31.4 | 54.7 | 0 |
| vanilla | + FK join inference | 37.5 | 48.9 | 94.0 | 530 |
| no_linking | as decoded | 48.7 | 53.0 | 88.0 | 0 |
| no_linking | + FK join inference | 50.9 | 57.8 | 97.1 | 200 |
| simple | as decoded | 9.6 | 10.0 | 31.7 | 0 |
| simple | + FK join inference | 10.4 | 20.6 | 90.9 | 709 |

The vanilla model gains the most (+17.5 EX), confirming that its main deficit is exactly the
relational schema structure that the relation-aware encoder provides.

## Schema-linking behaviour (model-independent)

From `results/tables/linking_analysis.json` (dev, gold SQL as reference): 93.1 % of gold columns
receive at least one name or value link, 54.2 % an exact name link and 24.3 % a value link; 80.2 % of
gold tables are linked. Precision is low for partial matches: of all linked (question token, column)
pairs, 63.5 % of exact-match pairs point to a gold column, 29.9 % of value exact-matches, 11.6 % of
partial matches, 4.5 % of partial value matches and 1.9 % of number links. 92.2 % of gold literals are
recoverable as value candidates. Linking recall is therefore high; the error analysis shows the
bottleneck is *choosing among* linked candidates.
