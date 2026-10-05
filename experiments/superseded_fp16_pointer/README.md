# Superseded runs (fp16 pointer underflow)

These runs were produced by the first round of the controlled experiments and are **not** used in
any result table. They are kept for transparency.

**What went wrong.** The memory-aligned column/table pointer computes a mixture probability
`p = λ · L`. Under fp16 autocast this product was computed in half precision, where probabilities
below ~6e-8 underflow to 0; `log(p)` then returns `-inf`. Whenever a gold column/table was that
unlikely, the batch loss became infinite, and PyTorch's AMP GradScaler **skipped that optimizer
step** (and halved the loss scale). Epoch-average training losses of `Infinity` in
`history.json` reveal it:

| Run | Epochs with ≥1 infinite-loss batch |
|---|---|
| `full` | 17 |
| `baseline_a_simple` | 2 |
| `baseline_b_vanilla_transformer_incomplete` (stopped at epoch 17) | 9, 11, 12, 15 |

Diagnosis (on the `full` checkpoint, validation split): 20 / 379 examples had an infinite
teacher-forced loss with the fp16 mixture and 0 / 379 with a float32 mixture; greedy predictions
were identical for 378 / 379 examples (`results/metrics/fp16_pointer_check.json`). The effect on
*inference* is therefore negligible, but *training* lost an unknown number of updates, unequally
across configurations — which would confound a controlled comparison.

**Fix.** `decoder.pointer_fp32: true` (now the default in `configs/base.yaml`) computes the
mixture in float32. The trainer now also logs `nonfinite_loss_batches` and `amp_skipped_steps`
per epoch. All controlled experiments were re-run from scratch with the fix.

Superseded measurements (for reference only): `full` reached 61.5 % EM / 63.0 % EX on Spider dev
(official evaluator: identical EM on all 1,034 examples); `baseline_a_simple` 10.8 % EM / 11.5 % EX.
