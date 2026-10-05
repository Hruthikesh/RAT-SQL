# Experiment runs

Every run directory is produced by `scripts/train.py` (usually via `scripts/run_ablation.py`)
and contains:

| File | Content |
|---|---|
| `config.json` | the fully-resolved configuration (after inheritance and overrides) |
| `run_info.json` | parameter counts per module, device / library versions, seed, dataset sizes |
| `train.log` | step and epoch logs |
| `history.json` | per-epoch training loss / step accuracy, validation loss / EM / EX, learning rate, time |
| `train_summary.json` | best validation EM, best epoch, epochs run, training time |
| `dev_metrics.json` | Spider-dev metrics of the best checkpoint (EM, EX, validity, hardness, components, timing) |
| `dev_predictions.jsonl` | one row per dev example: question, gold SQL, predicted SQL, all per-example metrics |
| `best.pt`, `last.pt` | checkpoints (**not** included in the distributed archive; re-create with the commands below) |

Layout:

```
experiments/
├── full_model/full/                       controlled FULL model (ablation A)
├── full_model/final/                      final model, larger budget (dev + test)
├── baseline/baseline_a_simple/            BiLSTM encoder, no RAT, no linking
├── baseline/baseline_b_vanilla_transformer/   (= ablation F)
├── baseline/baseline_c_no_linking/        (= ablation B)
├── ablations/ablation_{c,d,e,g,h,i,k}_*/  controlled ablations
├── scaling/scaling_{01,05,10,25,50}pct/   data-scaling runs (ablation J)
├── debug/rat_debug/                       DEBUG mode on the synthetic dataset
├── sanity/small_8ep/                      8-epoch calibration run used to choose the budget
└── queue.log                              log of scripts/run_ablation.py
```

Re-create everything (≈ 15 GPU-hours on an RTX 3050 laptop GPU):

```bash
python scripts/run_ablation.py                                   # baselines, ablations, scaling
python scripts/train.py --config configs/full.yaml --eval-splits dev test   # final model
python scripts/collect_results.py                                # tables and plots in results/
```
