### Ablations (our measurements)

All numbers are measured by this repository on the Spider **dev** set (1,034 examples, 20 unseen databases) unless a Split column says test (2,147 examples, 40 unseen databases). Controlled setting (all rows except Final): BERT-small encoder, identical training budget and data; checkpoints selected on held-out *training* databases. SQL validity = fraction of predictions that execute without error. ΔEM/ΔEX are differences to the FULL row; 95 % CIs from a paired bootstrap over dev examples (2,000 resamples) capture dev-set sampling noise but not training-seed variance.

| ID | Experiment | EM (%) | EX (%) | SQL valid (%) | Well-formed AST (%) | ΔEM (pts) | ΔEM 95% CI | ΔEX (pts) | ΔEX 95% CI | Params | Train (min) | Infer (ms/ex) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | FULL: BERT + RAT + schema linking + grammar decoding | 61.4 | 63.7 | 92.4 | 100.0 | 0.0 | – | 0.0 | – | 37,067,826 | 51.8 | 22.98 |
| B | Baseline C: schema-aware RAT without schema linking | 48.7 | 53.0 | 88.0 | 99.8 | -12.7 | [-15.5, -10.0] | -10.7 | [-13.2, -8.3] | 37,067,826 | 57.7 | 27.43 |
| C | No n-gram (name) linking | 54.3 | 55.7 | 89.6 | 100.0 | -7.2 | [-9.9, -4.4] | -8.0 | [-10.6, -5.4] | 37,067,826 | 40.9 | 10.26 |
| D | No value (cell) linking | 58.3 | 60.5 | 88.6 | 100.0 | -3.1 | [-5.6, -0.8] | -3.2 | [-5.6, -0.8] | 37,067,826 | 37.2 | 10.15 |
| E | No relation-aware transformer (BERT output fed to decoder) | 31.8 | 32.2 | 54.6 | 100.0 | -29.6 | [-32.8, -26.3] | -31.5 | [-34.7, -28.2] | 33,897,010 | 31.4 | 9.89 |
| F | Baseline B: vanilla transformer (no relation-aware attention) | 33.1 | 31.4 | 54.7 | 100.0 | -28.3 | [-31.3, -25.4] | -32.3 | [-35.4, -29.2] | 37,056,562 | 53.4 | 23.63 |
| G | No foreign-key relations | 60.2 | 60.0 | 91.1 | 100.0 | -1.3 | [-3.9, +1.5] | -3.8 | [-6.2, -1.2] | 37,067,826 | 36.6 | 10.66 |
| H | No grammar constraints (unmasked action softmax) | 20.9 | 21.7 | 26.2 | 27.1 | -40.5 | [-43.7, -37.4] | -42.1 | [-45.2, -38.9] | 37,067,826 | 36.6 | 7.26 |
| I | BiLSTM instead of BERT (with RAT + linking) | 49.9 | 52.8 | 85.6 | 99.9 | -11.5 | [-14.4, -8.7] | -10.9 | [-13.6, -8.1] | 25,377,586 | 27.8 | 11.18 |
| J | Reduced training data (10%) | 34.5 | 39.6 | 83.4 | 99.9 | -26.9 | [-29.8, -23.9] | -24.1 | [-27.0, -21.0] | 37,067,826 | 11.5 | 10.36 |
| K | Coarse relation vocabulary (6 undirected relation types) | 56.3 | 57.9 | 86.7 | 100.0 | -5.1 | [-7.6, -2.6] | -5.8 | [-8.4, -3.2] | 37,058,354 | 36.8 | 9.75 |
