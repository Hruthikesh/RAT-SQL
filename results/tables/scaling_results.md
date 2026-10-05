### Data scaling (our measurements)

All numbers are measured by this repository on the Spider **dev** set (1,034 examples, 20 unseen databases) unless a Split column says test (2,147 examples, 40 unseen databases). Controlled setting (all rows except Final): BERT-small encoder, identical training budget and data; checkpoints selected on held-out *training* databases. SQL validity = fraction of predictions that execute without error.

| Training data | Examples | EM (%) | EX (%) | SQL valid (%) | Epochs | Train (min) |
|---|---|---|---|---|---|---|
| 1% of training data | 83 | 4.2 | 7.2 | 53.0 | 200 | 4.3 |
| 5% of training data | 414 | 24.7 | 29.1 | 77.3 | 89 | 7.1 |
| 10% of training data | 828 | 34.5 | 39.6 | 83.4 | 63 | 11.5 |
| 25% of training data | 2,070 | 47.1 | 47.6 | 86.6 | 40 | 19.4 |
| 50% of training data | 4,140 | 54.1 | 58.5 | 87.6 | 28 | 26.9 |
| 100% of training data | 8,280 | 61.4 | 63.7 | 92.4 | 20 | 51.8 |
