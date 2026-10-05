## 21. Error analysis

The full analysis (taxonomy definitions, per-category counts by difficulty, 30 randomly sampled
failures read by hand, cross-model comparison) is in [`error_analysis.md`](error_analysis.md). In
short, for the controlled FULL model (432 failures of 1,034):

* the largest primary categories are schema-linking errors (107; a strongly linked element was
  missed or a similarly named, also-linked element was chosen), wrong columns (85), nested-query
  structure (56), wrong tables (52) and wrong joins (43);
* 77 predictions do not execute because a column is used from a table that never entered FROM, and
  33 predictions are exact matches that still fail execution (13 wrong join conditions, 19 literal
  values) — errors that Exact Match cannot see;
* removing linking relations doubles schema-linking errors (107 → 229), removing relation-aware
  attention adds mostly linking and table / join errors (366 / 129 / 69), and removing the grammar
  mask produces 754 ill-formed outputs;
* in the hand-read sample, missing tables / columns outside FROM are the most frequent pattern
  (11 / 30), followed by operator mistakes (4), and three sampled gold queries were themselves wrong or
  questionable (one verified against the database: the gold literal's casing returns no rows).

{{table:error_distribution.md}}
