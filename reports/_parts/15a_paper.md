### 15.1 Original paper results (Wang et al., 2020) — *not our measurements*

Quoted from the RAT-SQL paper (arXiv:1911.04942, Tables 2–4) for context only. The paper used
GloVe + BiLSTM or BERT (the public implementation uses BERT-large), 8 RAT layers, batch 20–24,
40k (GloVe) / 90k (BERT) steps.

| Model (paper) | Dev EM | Test EM |
|---|---|---|
| RAT-SQL | 62.7 | 57.2 |
| RAT-SQL + BERT | 69.7 | 65.6 |

| Paper, dev EM by hardness | Easy | Medium | Hard | Extra | All |
|---|---|---|---|---|---|
| RAT-SQL | 80.4 | 63.9 | 55.7 | 40.6 | 62.7 |
| RAT-SQL + BERT | 86.4 | 73.6 | 62.1 | 42.9 | 69.7 |

| Paper ablations (GloVe model, mean ± 95 % CI over 5 seeds) | Dev EM |
|---|---|
| RAT-SQL + value-based linking | 60.54 ± 0.80 |
| RAT-SQL (no value-based linking) | 55.13 ± 0.84 |
| w/o schema linking relations | 40.37 ± 2.32 |
| w/o schema graph relations | 35.59 ± 0.85 |

The paper reports Exact Match only (it does not predict literal values), so there is no paper
Execution Accuracy to compare with.
