# Results

Everything in this folder is generated from saved experiment outputs — no number is typed in by
hand. Regenerate with:

```bash
python scripts/collect_results.py                    # main / ablation / scaling / hardness tables, training curves
python scripts/run_error_analysis.py --primary full --predictions full=... vanilla=... no_linking=... simple=...
python scripts/run_join_inference.py --predictions full=... vanilla=... no_linking=... simple=...
python scripts/run_attention_analysis.py --models rat=... vanilla=...
python scripts/build_schema_graphs.py --config configs/small.yaml
python scripts/run_synthetic_relation_experiment.py --steps 3000
python scripts/official_eval_crosscheck.py --predictions experiments/full_model/full/dev_predictions.jsonl --exec
```

| Folder | Content |
|---|---|
| `tables/` | `main_results`, `ablation_results`, `scaling_results`, `hardness_results`, `component_results`, `clause_accuracy`, `complexity_results`, `error_distribution`, `error_analysis_full.csv` (every failure), `join_inference`, `attention_relation_lift`, `synthetic_relation_experiment`, schema-graph and relation statistics |
| `plots/` | training curves, data scaling, complexity, error distribution, attention lift / heatmaps, schema graphs, relation matrices, synthetic experiment curves |
| `metrics/` | official-evaluator cross-check, linking analysis, synthetic experiment raw curves, fp16 diagnostic |
| `predictions/` | per-example dev (and test) predictions of the main models (question, gold SQL, predicted SQL, per-example metrics) |

All accuracy numbers are **our measurements** on Spider dev (1,034 examples) unless a table says
otherwise; numbers quoted from the RAT-SQL paper appear only in `reports/`, labelled as such.
