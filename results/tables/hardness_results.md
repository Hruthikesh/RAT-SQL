### EM / EX by Spider difficulty (dev, our measurements)

| Experiment | easy EM | medium EM | hard EM | extra EM | easy EX | medium EX | hard EX | extra EX |
|---|---|---|---|---|---|---|---|---|
| ablation_c_no_ngram_linking | 77.4 | 52.7 | 48.3 | 30.1 | 76.6 | 54.3 | 52.3 | 31.9 |
| ablation_d_no_value_linking | 76.2 | 60.3 | 44.8 | 40.4 | 79.4 | 62.8 | 49.4 | 38.0 |
| ablation_e_no_rat | 52.0 | 29.8 | 23.6 | 15.7 | 55.6 | 30.5 | 23.6 | 10.8 |
| ablation_g_no_foreign_keys | 83.5 | 61.4 | 48.9 | 33.7 | 82.7 | 62.3 | 45.4 | 34.9 |
| ablation_h_no_grammar | 42.3 | 18.8 | 10.9 | 4.8 | 40.3 | 20.8 | 12.6 | 5.4 |
| ablation_i_lstm_encoder | 75.8 | 46.4 | 39.1 | 31.9 | 77.4 | 50.0 | 43.7 | 33.1 |
| ablation_k_coarse_relations | 81.0 | 55.8 | 43.7 | 33.7 | 81.0 | 59.6 | 44.8 | 32.5 |
| baseline_a_simple | 21.4 | 8.1 | 5.2 | 0.6 | 24.2 | 7.0 | 5.2 | 1.8 |
| baseline_b_vanilla_transformer | 52.4 | 31.6 | 29.9 | 11.5 | 52.4 | 30.3 | 25.3 | 9.6 |
| baseline_c_no_linking | 68.5 | 49.5 | 39.7 | 26.5 | 73.4 | 53.8 | 43.1 | 30.7 |
| final | 83.9 | 67.7 | 52.9 | 41.6 | 82.3 | 72.0 | 56.9 | 41.0 |
| full | 82.7 | 62.6 | 48.9 | 39.8 | 82.7 | 65.9 | 51.7 | 42.2 |
| scaling_01pct | 11.7 | 1.8 | 3.5 | 0.0 | 17.3 | 4.5 | 4.0 | 2.4 |
| scaling_05pct | 50.4 | 18.8 | 19.0 | 7.8 | 54.4 | 24.2 | 21.8 | 12.0 |
| scaling_10pct | 58.5 | 31.8 | 27.0 | 13.9 | 61.7 | 38.8 | 30.5 | 18.7 |
| scaling_25pct | 76.2 | 45.1 | 33.3 | 23.5 | 74.6 | 47.8 | 36.2 | 18.7 |
| scaling_50pct | 76.2 | 56.5 | 39.1 | 30.1 | 78.2 | 62.8 | 41.4 | 35.5 |

Examples per level: easy=248, medium=446, hard=174, extra=166
