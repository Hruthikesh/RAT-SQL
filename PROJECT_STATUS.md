# Project status — RAT-SQL: Schema-Aware Text-to-SQL

Snapshot saved on **2026-10-01 03:12 IST**, when work was paused at the user's request. No
experiment is running (GPU idle); every file, log, metric, report and checkpoint listed below is on
disk in this folder. All percentages are **our measurements**, read from the metric files named in
each table; numbers from the RAT-SQL paper are not used anywhere in this file.

---

## 1. Completed components

All code is in `src/ratsql/` and is exercised by the test suite (**61 passed**, 0 failed —
`results/metrics/test_results.txt`, run 2026-10-01).

| Area | Implemented | Code |
|---|---|---|
| Data | Spider download (official link) + validation, deterministic preprocessing, database-disjoint train / val / dev / test splits, synthetic Spider-format dataset | `data/`, `scripts/download_data.py`, `scripts/preprocess.py` |
| SQL | Spider-style parser (re-implementation of `process_sql` semantics), ASDL grammar (20 composite + 3 primitive types, 49 constructors), AST ↔ Spider dict, transition system with grammar masks, serializer, FK join inference | `sql/` |
| Schema | schema model (PK / FK), explicit `SchemaGraph`, `RelationVocabulary` (43 types; coarse 6), `RelationMatrixBuilder`, n-gram + value schema linking, DB value index, visualisation | `schema/` |
| Model | BERT input encoder (long-schema chunking), BiLSTM encoder, relation-aware transformer (efficient + reference, vanilla mode), grammar-constrained LSTM AST decoder (memory-aligned column/table pointers, value pointer, batched teacher forcing, greedy + beam search) | `models/` |
| Training | AdamW (2 LRs), warm-up + decay, fp16 AMP with float32 pointer mixture, clipping, accumulation, bucketing, checkpoint / resume, early stopping, non-finite-loss and skipped-step monitoring | `training/` |
| Evaluation | exact match (re-implementation, verified against the official evaluator), execution accuracy, SQL validity, components, hardness, complexity | `evaluation/` |
| Analysis | error taxonomy, clause accuracy, complexity, linking statistics, attention lift, synthetic relation task | `analysis/` |
| Inference | predictor + CLI | `inference/`, `scripts/predict.py` |
| Tooling | experiment registry + runner, results collector, analysis driver, report / README / status builders, notebooks (4, executed) | `scripts/`, `configs/`, `notebooks/` |

## 2. Completed experiments

| Experiment | Status | Directory |
|---|---|---|
| Spider preprocessing + oracle checks (parse agreement, AST round trip, oracle EX) | done | `data/processed/spider/stats.json` |
| DEBUG mode on synthetic data | done | `experiments/debug/rat_debug` |
| Calibration run (budget choice only, not reported) | done | `experiments/sanity/small_8ep` |
| Controlled FULL (BERT-small, 20 epochs, seed 42) | done | `experiments/full_model/full` |
| Baselines A (BiLSTM), B (vanilla transformer), C (no linking) | done | `experiments/baseline/*` |
| Ablations C (no n-gram), D (no value), E (no RAT), G (no FK), H (no grammar), I (BiLSTM encoder), K (coarse relations) | done | `experiments/ablations/*` |
| Data scaling 1 / 5 / 10 / 25 / 50 % (100 % = FULL) | done | `experiments/scaling/*` |
| Final model (BERT-medium, 40 epochs), dev + test evaluation | done | `experiments/full_model/final` |
| Second seed of FULL (seed 43) | done (resumed once from epoch 4) | `experiments/full_model/full_seed43` |
| Official Spider evaluator cross-checks (full dev, final dev, final test) | done | `results/metrics/official_crosscheck_*.json` |
| Error analysis, clause / complexity / hardness breakdowns, linking statistics | done | `results/tables/` |
| Attention analysis (FULL vs vanilla, 400 dev examples) | done | `results/tables/attention_relation_lift.*` |
| Synthetic relation experiment (1,500 and 3,000 steps, 3 seeds) | done | `results/tables/synthetic_relation_experiment*.md` |
| FK join-inference re-serialisation (no retraining) | done | `results/tables/join_inference.*` |
| fp16 numerical diagnosis + fix; all affected runs re-trained | done | `experiments/superseded_fp16_pointer/`, `results/metrics/fp16_pointer_check.json` |

## 3. Measured results available

### Spider dev / test (`experiments/**/{dev,test}_metrics.json`)

| Model | Split | EM | EX | Executable SQL | Well-formed AST |
|---|---|---|---|---|---|
| **Final** (BERT-medium, 40 ep.) | dev | **64.89 %** | **66.92 %** | 90.04 % | 100.0 % |
| **Final** (BERT-medium, 40 ep.) | test | **59.34 %** | **63.44 %** | 88.64 % | 99.91 % |
| FULL (BERT-small, 20 ep., seed 42) | dev | 61.41 % | 63.73 % | 92.36 % | 100.0 % |
| FULL, seed 43 | dev | 59.09 % | 61.12 % | 88.88 % | 99.81 % |
| Baseline A — BiLSTM, no RAT, no linking | dev | 9.57 % | 9.96 % | 31.72 % | 100.0 % |
| Baseline B — vanilla transformer (= ablation F) | dev | 33.08 % | 31.43 % | 54.74 % | 100.0 % |
| Baseline C — no schema linking (= ablation B) | dev | 48.74 % | 53.00 % | 88.01 % | 99.81 % |
| C — no n-gram linking | dev | 54.26 % | 55.71 % | 89.56 % | 100.0 % |
| D — no value linking | dev | 58.32 % | 60.54 % | 88.59 % | 100.0 % |
| E — no RAT layers | dev | 31.82 % | 32.21 % | 54.64 % | 100.0 % |
| G — no foreign-key relations | dev | 60.15 % | 59.96 % | 91.10 % | 100.0 % |
| H — no grammar constraints | dev | 20.89 % | 21.66 % | 26.21 % | 27.08 % |
| I — BiLSTM instead of BERT | dev | 49.90 % | 52.80 % | 85.59 % | 99.90 % |
| K — coarse relation vocabulary | dev | 56.29 % | 57.93 % | 86.65 % | 100.0 % |
| Scaling 1 % (83 ex.) | dev | 4.16 % | 7.16 % | 53.00 % | 100.0 % |
| Scaling 5 % (414 ex.) | dev | 24.66 % | 29.11 % | 77.27 % | 100.0 % |
| Scaling 10 % (828 ex.) | dev | 34.53 % | 39.65 % | 83.37 % | 99.90 % |
| Scaling 25 % (2,070 ex.) | dev | 47.10 % | 47.58 % | 86.56 % | 100.0 % |
| Scaling 50 % (4,140 ex.) | dev | 54.06 % | 58.51 % | 87.62 % | 100.0 % |
| DEBUG (synthetic, BERT-tiny) | synthetic dev (152) | 88.82 % | 86.84 % | 92.11 % | 100.0 % |

Paired-bootstrap CIs, parameter counts and training times: `results/tables/ablation_results.md`,
`main_results.md`, `compute_results.md`.

### Other measured results

* **Official evaluator agreement** (`results/metrics/official_crosscheck_{full,final,final_test}.json`):
  exact match identical on every example (FULL dev 61.41 %, final dev 64.89 %, final test 59.34 %);
  official test-suite EX 64.51 % / 68.09 % / 64.28 % vs our stricter 63.73 % / 66.92 % / 63.44 %
  (per-example EX agreement 98.5 % / 98.6 % / 98.9 %; 3 test predictions that hit our 5 s timeout were
  skipped and counted wrong).
* **Seed variance** (`results/tables/seed_variance.md`): seed 43 − seed 42 = −2.3 EM (paired-bootstrap
  95 % CI [−4.5, +0.0]), −2.6 EX.
* **FK join inference** (`results/tables/join_inference.md`): FULL EM 61.4 → 62.8, EX 63.7 → 66.6,
  executable 92.4 → 98.6 %; final EM 64.9 → 65.7, EX 66.9 → 69.9, executable 90.0 → 96.4 %; vanilla EX
  31.4 → 48.9.
* **Attention lift, last layer** (`results/tables/attention_relation_lift.md`): column → FK partner
  5.75 (RAT) vs 0.73 (vanilla); question → exact-match column 3.30 vs 2.51; question → unlinked column
  0.29 vs 0.83.
* **Synthetic relation task, 3,000 steps** (`results/tables/synthetic_relation_experiment.md`): one-hop
  1.000 ± 0.000 (relation-aware) vs 0.293 ± 0.002 (vanilla); two-hop 0.887 ± 0.107 vs 0.286 ± 0.003.
* **Dataset checks** (`data/processed/spider/stats.json`): parse agreement with Spider's `sql` field
  1,034/1,034 dev, 2,147/2,147 test; oracle EX 95.0 % dev / 96.4 % test; zero database overlap between
  splits.
* **Linker quality** (`results/tables/linking_analysis.json`): 93.1 % of gold columns linked; precision
  EM 63.5 %, VEM 29.9 %, PM 11.6 %.
* **fp16 diagnosis** (`results/metrics/fp16_pointer_check.json`): infinite teacher-forced losses 20/379
  (fp16 mixture) vs 0/379 (float32); identical greedy actions 378/379. All re-run experiments logged 0
  non-finite batches.
* **Error analysis** (`results/tables/error_analysis_full.csv`, `reports/error_analysis.md`): 432 FULL
  failures classified; 77 non-executable ("no such column"); 33 EM-correct but EX-wrong.

## 4. Pending experiments / tasks (not done — no results exist for these)

| Item | State | What remains |
|---|---|---|
| Ablation **H2-ref** (direct pointers, grammar on) | **interrupted**: 7 of 20 epochs; `last.pt` = end of epoch 7 (step 1,756); no dev metrics | resume training (command below), then dev evaluation |
| Ablation **H2** (direct pointers, grammar off) | **not started** | train + evaluate |
| **Beam-search timing** of the final model (k = 5, dev) | **not started** (a beam run exists only for the *superseded* FULL model and is not reported) | one evaluation run |
| Report sections §16b (H2 results) and §23 (compute / inference) | `reports/RAT_SQL_REPORT.md` currently contains the placeholders `[MISSING SECTION: 16b_h2]` and `[MISSING SECTION: 23_compute]` | write `reports/_parts/16b_h2.md` and `23_compute.md` after the runs above, then `python scripts/build_report.py` |
| README beam-search row | README shows "beam-search evaluation not available" | regenerated automatically by `build_report.py` after the beam run |
| `PROJECT_COMPLETION_REPORT.md`, final cleanup (`__pycache__`, `src/*.egg-info`), final ZIP `rat-sql-complete.zip` | not created | `python scripts/build_status.py` (note: it regenerates *this* file too), cleanup, zip (excluding `.venv`, `data/raw`, `data/processed`, `third_party`, `last.pt`) |

## 5. Checkpoint locations

| Checkpoint | Contents | Size |
|---|---|---|
| `experiments/full_model/final/best.pt` | final model (BERT-medium), epoch 40, val EM 0.6491 | 199 MB |
| `experiments/full_model/full/best.pt` | controlled FULL (seed 42), epoch 20, val EM 0.5884 | 148 MB |
| `experiments/full_model/full_seed43/best.pt` | FULL seed 43, epoch 20, val EM 0.5752 | 148 MB |
| `experiments/baseline/*/best.pt`, `experiments/ablations/*/best.pt`, `experiments/scaling/*/best.pt` | one per completed run | 89–148 MB each |
| `experiments/ablations/ablation_h2_ref_direct_pointer/last.pt` | **resume state** of the interrupted H2-ref run (model, optimizer, scheduler, scaler, RNG, history), epoch 7 | 442 MB |
| `experiments/**/last.pt` | resume states of all runs (optimizer included) | 267–596 MB each |
| `experiments/debug/rat_debug/best.pt` | DEBUG model (synthetic data) | 21 MB |
| `experiments/superseded_fp16_pointer/*/best.pt` | first-round runs affected by the fp16 bug (not reported) | 89–148 MB |

All four key checkpoints above were verified to load on 2026-10-01 (final, FULL, seed 43, H2-ref resume state).

## 6. Exact commands to resume

Run from the project root with the project virtual environment (`.venv\Scripts\python.exe` on this
machine; `python` below). Spider must be at `data/raw/spider_data` and preprocessed in
`data/processed/spider` (both present on this machine).

```bash
# 1. finish the interrupted H2-ref run (resumes from last.pt at epoch 7), then run H2
#    run_ablation.py adds --resume automatically when last.pt exists and skips runs that have dev metrics
python scripts/run_ablation.py --only ablation_h2_ref_direct_pointer ablation_h2_no_grammar_direct_pointer

# 2. beam-search timing of the final model (writes experiments/full_model/final/dev_beam5_{metrics.json,predictions.jsonl})
python scripts/evaluate.py --checkpoint experiments/full_model/final/best.pt --split dev --beam-size 5

# 3. refresh tables / analyses (no training)
python scripts/collect_results.py
python scripts/run_analyses.py --skip attention official      # errors, joins, predictions copy, notebooks

# 4. write reports/_parts/16b_h2.md and reports/_parts/23_compute.md from the new metrics, then
python scripts/build_report.py
python -m pytest tests -p no:cacheprovider > results/metrics/test_results.txt
python scripts/build_status.py                                 # regenerates PROJECT_STATUS.md + PROJECT_COMPLETION_REPORT.md

# 5. cleanup and package (exclude .venv, data/raw, data/processed, third_party, last.pt, caches)
```

Equivalent single-run resume for H2-ref:
`python scripts/train.py --config configs/small.yaml --set model.decoder.pointer=direct train.epochs=20 --output-dir experiments/ablations/ablation_h2_ref_direct_pointer --eval-splits dev --resume`

## 7. Files generated

* **Reports**: `README.md` (result regions generated), `reports/RAT_SQL_REPORT.md` (27 sections; two
  pending placeholders), `reports/methodology.md`, `reports/experiment_results.md`,
  `reports/error_analysis.md`, `reports/resume_bullets.md`, `reports/_parts/*.md` (authored sections),
  `data/README.md`, `experiments/README.md`, `experiments/superseded_fp16_pointer/README.md`,
  `results/README.md`, this file.
* **Result tables** (`results/tables/`): `main_results`, `ablation_results`, `scaling_results`,
  `seed_variance`, `hardness_results`, `component_results.csv`, `clause_accuracy`, `complexity_results`,
  `compute_results`, `error_distribution`, `error_analysis_full.csv`, `join_inference`,
  `attention_relation_lift`, `synthetic_relation_experiment` (+ `_1500steps`), `linking_analysis.json`,
  `schema_graph_stats.csv`, `relation_type_counts.csv` (+ synthetic variants).
* **Plots** (`results/plots/`): `training_curves`, `data_scaling`, `complexity_em`, `complexity_ex`,
  `error_distribution`, `attention_lift`, `attention_heatmap_{rat,vanilla}`, `schema_graph_*`,
  `relation_matrix_*`, `synthetic_relation_experiment` (+ `_1500steps`).
* **Metrics** (`results/metrics/`): official cross-checks (full, final, final_test), fp16 check,
  synthetic experiment raw curves, `test_results.txt`.
* **Predictions**: `experiments/**/{dev,test}_predictions.jsonl` (per example), `results/predictions/`.
* **Logs**: `experiments/queue.log`, `experiments/orchestrator.log`, every run's `train.log`,
  `stdout.log`, `history.json`, `run_info.json`, `config.json`, `train_summary.json`.
* **Notebooks** (executed): `schema_graph_visualization`, `schema_linking_analysis`,
  `attention_analysis`, `error_analysis`.

## 8. Known limitations

* One 6 GB laptop GPU: BERT-small (controlled) / BERT-medium (final), 4 RAT layers, 20 / 40 epochs —
  far below the paper's BERT-large / 8 layers / 90k updates; our numbers are not a reproduction of the
  paper's.
* Ablations are single-seed; a second FULL seed differs by 2.3 EM / 2.6 EX, so effects of ≈ 3 points or
  less (value linking, foreign keys on EM) are suggestive, not established.
* The grammar-ablation result (H) may be inflated by the pointer parameterisation; the H2 / H2-ref control
  that would quantify this is **not finished** (§4).
* Execution accuracy uses the original Spider databases (not test suites) and keeps `DISTINCT`
  (0.8–1.2 points stricter than the official executor); exact match ignores join conditions and values;
  Spider gold contains some annotation errors (one verified in a 30-example sample).
* Simplifications: rule-based singularisation instead of CoreNLP lemmas, bounded cell-value index,
  schema chunking beyond 512 word pieces, value prediction by pointing to deterministic candidates
  (oracle ceiling 95.0 % EX on dev).
* Grammar validity ≠ schema consistency: 7.6 % of FULL predictions (79 / 1,034) do not execute, 77 of
  them because a referenced column's table is not joined (handled only by the separately reported FK
  join-inference variant).
* Wall-clock timings are from a shared laptop and vary with machine load (e.g. identical-cost runs took
  112 s vs 161 s per epoch at different times); a beam-search timing for the final model is pending.
* The first round of experiments suffered from an fp16 underflow that silently skipped optimizer steps;
  it was fixed and every reported run was re-trained (superseded runs kept for transparency).
