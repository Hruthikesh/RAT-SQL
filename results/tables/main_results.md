### Main results (our measurements)

All numbers are measured by this repository on the Spider **dev** set (1,034 examples, 20 unseen databases) unless a Split column says test (2,147 examples, 40 unseen databases). Controlled setting (all rows except Final): BERT-small encoder, identical training budget and data; checkpoints selected on held-out *training* databases. SQL validity = fraction of predictions that execute without error.

| Model | Split | Exact Match (%) | EM 95% CI | Execution Accuracy (%) | EX 95% CI | SQL Validity (%) | Params | Train (min) |
|---|---|---|---|---|---|---|---|---|
| Baseline A: BiLSTM schema encoder + grammar decoder (no BERT, no RAT, no linking) | dev | 9.6 | [7.8, 11.3] | 10.0 | [8.2, 11.7] | 31.7 | 22,206,770 | 41.5 |
| Baseline B: vanilla transformer (no relation-aware attention) | dev | 33.1 | [30.4, 36.0] | 31.4 | [28.7, 34.3] | 54.7 | 37,056,562 | 53.4 |
| Baseline C: schema-aware RAT without schema linking | dev | 48.7 | [45.6, 51.7] | 53.0 | [49.7, 56.0] | 88.0 | 37,067,826 | 57.7 |
| FULL: BERT + RAT + schema linking + grammar decoding | dev | 61.4 | [58.4, 64.3] | 63.7 | [60.6, 66.5] | 92.4 | 37,067,826 | 51.8 |
| Final: FULL with BERT-medium, 40 epochs | dev | 64.9 | [62.0, 67.7] | 66.9 | [64.1, 69.9] | 90.0 | 49,677,362 | 101.3 |
| Final: FULL with BERT-medium, 40 epochs | test | 59.3 | [57.3, 61.5] | 63.4 | [61.4, 65.5] | 88.6 | 49,677,362 | 101.3 |
