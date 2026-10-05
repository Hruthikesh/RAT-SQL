All tables below are generated from saved metric files (`scripts/collect_results.py`,
`scripts/build_report.py`). Full discussion: [`reports/RAT_SQL_REPORT.md`](reports/RAT_SQL_REPORT.md);
all tables: [`reports/experiment_results.md`](reports/experiment_results.md); per-run status and
provenance: [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

{{table:main_results.md}}

#### Agreement with the official Spider evaluator (`results/metrics/official_crosscheck_*.json`)

{{crosscheck}}

Our EX keeps `DISTINCT`; the official test-suite executor strips it, hence its slightly higher EX.

{{table:ablation_results.md}}

{{table:seed_variance.md}}

{{table:scaling_results.md}}

![Data scaling](results/plots/data_scaling.png)

#### Where the model fails

* Accuracy falls with the number of joined tables (`results/tables/complexity_results.md`); the
  vanilla transformer collapses already at two tables, the relation-aware model degrades gracefully.
* Dominant failure: FROM / join construction (columns used from tables that are not joined, wrong ON
  clauses), then disambiguation between similarly named, equally well-linked schema elements.
  Error taxonomy and 30 hand-read random failures: [`reports/error_analysis.md`](reports/error_analysis.md).
* Re-serialising the decoded trees with schema-graph (foreign-key) join inference is a separate
  post-processing variant, not part of the main numbers:

{{table:join_inference.md}}

#### Why relations help: attention and a synthetic test

* Relation-aware layers attend to a column's foreign-key partner with 5.75× uniform attention; the
  vanilla encoder shows no preference (0.73×). Question tokens attend 3.30× to exactly-matched
  columns vs 0.29× to unlinked ones (`results/tables/attention_relation_lift.md`).
* On a synthetic task solvable only through pairwise relations, relation-aware attention reaches
  100 % (one-hop) while vanilla attention stays at the best relation-free strategy (~29 %):

{{table:synthetic_relation_experiment.md}}

#### Compute, inference speed and beam search

{{table:compute_results.md}}

{{beam}}

#### Original paper results (for context only — *not our measurements*)

| RAT-SQL paper (Wang et al., 2020) | Dev EM | Test EM |
|---|---|---|
| RAT-SQL (GloVe) | 62.7 | 57.2 |
| RAT-SQL + BERT (BERT-large) | 69.7 | 65.6 |
