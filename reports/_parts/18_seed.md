## 18. Seed and reproducibility results

The controlled FULL configuration was trained a second time with seed 43 (identical config, data and
code; `experiments/full_model/full_seed43`). The seed changes weight initialisation, dropout masks and
the order / composition of length-bucketed batches (5,034 vs 4,940 optimizer steps).

{{table:seed_variance.md}}

**Interpretation.** Two seeds give only a rough estimate of training variance, but the observed
spread — 2.3 EM and 2.6 EX points — is as large as the smaller ablation effects. Consequently:

* effects of ≥ 5 points (relation-aware attention −28.3, removing linking −12.7, removing name linking
  −7.2, coarse relations −5.1 EM, grammar constraints −40.5 EM / −73 points of well-formed trees) are
  much larger than both the seed spread and the paired-bootstrap intervals and are robust conclusions;
* the value-linking effect (−3.1 EM / −3.2 EX) and the foreign-key effect on EM (−1.3) are of the same
  order as the seed spread and should be read as *suggestive, not established*; the FK effect on EX
  (−3.8) is borderline;
* both seeds converge to their best validation EM in the last epoch, i.e. the 20-epoch budget, not
  over-fitting, bounds the controlled models.

Reproducibility measures that *are* in place: deterministic preprocessing (identical
`stats.json` on re-runs), fixed seeds for Python / NumPy / PyTorch, resolved configs, library versions
and parameter counts stored with every run, per-epoch histories, and resumable checkpoints (the seed-43
run was interrupted after epoch 4 and resumed from `last.pt`, restoring model, optimizer, scheduler,
scaler and RNG states). GPU training is not bitwise deterministic (CUDA atomics in scatter / index-put
backward), so a re-run reproduces the numbers only up to the seed-level noise shown above.
