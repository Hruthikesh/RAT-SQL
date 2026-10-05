## 22. Numerical-stability investigation (fp16 underflow in the pointer)

**Symptom.** In the first round of controlled experiments the logged *validation* loss of the FULL
model became `inf` from epoch 9 on, while validation EM kept improving; the vanilla baseline then
showed `inf` epoch-average *training* losses (epochs 9, 11, 12, 15).

**Diagnosis** (forward hooks on the FULL checkpoint; `results/metrics/fp16_pointer_check.json`).
The non-finite values came from the column / table pointers. RAT-SQL's memory-aligned pointer
returns `log(λ · L)`; under fp16 autocast the product `λ · L` was computed in half precision, where
probabilities below ≈ 6 × 10⁻⁸ underflow to 0 (`clamp_min(1e-12)` does not help: 1e-12 is not
representable in fp16), so `log` returned −∞ whenever a gold column was very unlikely. Measured on the
379 validation examples: 20 teacher-forced losses were infinite with the fp16 mixture and 0 with a
float32 mixture, and greedy decoding produced identical action sequences for 378 / 379 examples — so
*predictions* were essentially unaffected, but *training* was: a batch with an infinite loss produces
non-finite gradients, and PyTorch's GradScaler then silently **skips that optimizer step**.

**Impact on the first round** (runs kept in `experiments/superseded_fp16_pointer/`):

| Superseded run | Epochs with an infinite-loss batch | Infinite validation losses | Dev EM / EX |
|---|---|---|---|
| FULL | 17 | 5 of 7 evaluations | 61.5 / 63.0 |
| Baseline A | 2 | 3 | 10.8 / 11.5 |
| Baseline B (stopped at epoch 17) | 9, 11, 12, 15 | 2 | – |

Because the number of skipped updates differed between configurations, a controlled comparison
built on these runs would have been confounded (the weaker vanilla model was hit hardest).

**Fix and verification.** The mixture is now computed in float32 inside an autocast-disabled region
(`decoder.pointer_fp32: true`, default in `configs/base.yaml`), and the trainer logs
`nonfinite_loss_batches` and `amp_skipped_steps` every epoch. **All** reported experiments were
re-trained from scratch after the fix. Across the 18 re-run training jobs (FULL, 3 baselines, 7
ablations, 5 scaling runs, final model, second seed) the logs show **0 batches with a non-finite loss
and 0 infinite validation losses**; the 4–8 skipped AMP steps per run are the normal dynamic
loss-scale calibration at the start of training and after scale growth. The re-trained FULL model
(61.4 % EM / 63.7 % EX) is within noise of the superseded one (61.5 / 63.0), consistent with the
superseded FULL run having lost updates in only one epoch; the fix also costs no measurable speed.
