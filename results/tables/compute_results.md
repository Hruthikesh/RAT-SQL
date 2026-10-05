### Computational cost (RTX 3050 Laptop 6 GB, our measurements)

Wall-clock numbers on a shared laptop (they vary by roughly ±30 % with machine load). Greedy decoding is batched (batch 64) over the 1,034 dev examples and includes the encoder; beam search runs one example at a time.

| Experiment | Params | Epochs | Optimizer steps | Train (min) | Median epoch (s) | Train ex/s | Greedy ms/ex | Beam-5 ms/ex |
|---|---|---|---|---|---|---|---|---|
| FULL: BERT + RAT + schema linking + grammar decoding | 37,067,826 | 20 | 4,940 | 51.8 | 161.2 | 51.4 | 22.98 | – |
| Baseline A: BiLSTM schema encoder + grammar decoder (no BERT, no RAT, no linking) | 22,206,770 | 20 | 4,940 | 41.5 | 132.0 | 62.7 | 31.93 | – |
| Baseline B: vanilla transformer (no relation-aware attention) | 37,056,562 | 20 | 4,940 | 53.4 | 170.3 | 48.6 | 23.63 | – |
| Baseline C: schema-aware RAT without schema linking | 37,067,826 | 20 | 4,940 | 57.7 | 164.5 | 50.3 | 27.43 | – |
| No n-gram (name) linking | 37,067,826 | 20 | 4,940 | 40.9 | 112.8 | 73.4 | 10.26 | – |
| No value (cell) linking | 37,067,826 | 20 | 4,940 | 37.2 | 113.2 | 73.2 | 10.15 | – |
| No relation-aware transformer (BERT output fed to decoder) | 33,897,010 | 20 | 4,940 | 31.4 | 95.5 | 86.7 | 9.89 | – |
| No foreign-key relations | 37,067,826 | 20 | 4,940 | 36.6 | 111.4 | 74.3 | 10.66 | – |
| No grammar constraints (unmasked action softmax) | 37,067,826 | 20 | 4,940 | 36.6 | 111.2 | 74.5 | 7.26 | – |
| BiLSTM instead of BERT (with RAT + linking) | 25,377,586 | 20 | 4,940 | 27.8 | 84.7 | 97.8 | 11.18 | – |
| Coarse relation vocabulary (6 undirected relation types) | 37,058,354 | 20 | 4,940 | 36.8 | 111.7 | 74.1 | 9.75 | – |
| Final: FULL with BERT-medium, 40 epochs | 49,677,362 | 40 | 11,520 | 101.3 | 153.6 | 53.9 | 12.39 | – |
