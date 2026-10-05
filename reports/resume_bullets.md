# Resume bullets

Every number below was measured by this repository (sources in the table at the end). Numbers
are from single training runs on one 6 GB laptop GPU; the Spider test set was evaluated once.

## Recommended (4 bullets)

- **Built a relation-aware Text-to-SQL system from scratch** (PyTorch re-implementation of RAT-SQL):
  relational schema graphs with 43 relation types, n-gram and database-value schema linking, BERT
  contextual encoding, a relation-aware transformer and a grammar-constrained LSTM AST decoder with
  column / table / value pointers that generates executable SQL.
- **Reached 64.9 % Exact Match / 66.9 % Execution Accuracy on Spider dev (59.3 % / 63.4 % on the
  Spider test set)** with a BERT-medium encoder (49.7 M parameters) trained in 101 minutes on a single
  6 GB laptop GPU; re-implemented Spider's exact-match metric and verified per-example agreement with
  the official evaluator on all 1,034 dev predictions.
- **Ran 16 controlled experiments** (3 baselines, 7 ablations, 5 data-scaling points) with
  paired-bootstrap confidence intervals: relation-aware attention added +28.3 EM points over an
  equal-size vanilla transformer, schema linking +12.7, fine-grained relation types +5.1, and
  grammar-constrained decoding raised well-formed SQL from 27 % to 100 %.
- **Analysed failures and attention**: relation-aware layers attend 5.8× more than uniform to
  foreign-key partners (vanilla: no preference); identified FROM/join construction as the dominant
  failure mode and showed schema-graph join inference lifts execution accuracy from 63.7 % to 66.6 %
  (executable SQL 92.4 % → 98.6 %) without retraining; diagnosed an fp16 underflow that silently
  skipped mixed-precision optimizer steps and re-ran all experiments after fixing it.

## Shorter variant (3 bullets)

- Developed a relation-aware Text-to-SQL parser (RAT-SQL re-implementation) combining relational
  schema graphs, n-gram and value-based schema linking, BERT, a relation-aware transformer and a
  grammar-constrained LSTM AST decoder that produces executable SQL.
- Achieved 64.9 % Exact Match and 66.9 % Execution Accuracy on Spider dev (59.3 % / 63.4 % on test)
  on a single 6 GB GPU, with an exact-match implementation verified against the official evaluator.
- Quantified each component through controlled ablations and error / attention analysis
  (relation-aware attention +28.3 EM, schema linking +12.7 EM, grammar constraints 27 % → 100 %
  well-formed SQL).

## Sources of every number

| Claim | Source file |
|---|---|
| 64.9 % EM / 66.9 % EX (dev); 59.3 % / 63.4 % (test) | `experiments/full_model/final/{dev,test}_metrics.json` |
| 49.7 M parameters, 101 min training | `experiments/full_model/final/train_summary.json` (`train_seconds` = 6,077.6) |
| per-example EM agreement with the official evaluator (1,034 / 1,034) | `results/metrics/official_crosscheck_full.json`, `official_crosscheck_final.json` |
| +28.3 EM (relation-aware vs vanilla), +12.7 EM (linking), +5.1 EM (fine vs coarse relations) | `results/tables/ablation_results.csv` (controlled BERT-small setting) |
| 27.1 % vs 100 % well-formed ASTs | `results/tables/ablation_results.csv` (ablation H vs FULL) |
| 5.8× attention lift on FK partners | `results/tables/attention_relation_lift.md` (`cc_fk_forward`, last layer: 5.75 vs 0.73) |
| 63.7 % → 66.6 % EX, 92.4 % → 98.6 % executable | `results/tables/join_inference.md` (controlled FULL model) |

Note: the controlled-setting numbers (ablations, join inference) come from the BERT-small FULL model
(61.4 % EM / 63.7 % EX on dev), not from the final BERT-medium model.
