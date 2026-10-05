| Task | Attention | Test accuracy (mean ± std over seeds) | Chance | Params |
|---|---|---|---|---|
| one_hop | relation-aware | 1.000 ± 0.000 | 0.125 | 72,584 |
| one_hop | vanilla | 0.293 ± 0.002 | 0.125 | 72,200 |
| two_hop | relation-aware | 0.887 ± 0.107 | 0.125 | 72,584 |
| two_hop | vanilla | 0.286 ± 0.003 | 0.125 | 72,200 |

3,000 training steps, 3 seeds, 12 nodes, 8 labels, 1,024 fresh test graphs.
Relation-free reference (simulated on 4,000 random graphs): predicting the most frequent label
among the other nodes scores **0.295** (one-hop) and **0.285** (two-hop) — the level vanilla
attention converges to, i.e. vanilla attention learns the best strategy available without
relation information. A 1,500-step run of the same experiment is kept in
`synthetic_relation_experiment_1500steps.md` (two-hop relation-aware: 0.751 ± 0.022).
