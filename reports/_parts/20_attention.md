## 20. Attention analysis

### 20.1 Relation-aware vs vanilla encoder attention

`scripts/run_attention_analysis.py` runs the controlled FULL model and Baseline B (identical
architecture without relation embeddings) on 400 Spider-dev examples and groups the encoder's
self-attention by the (full-vocabulary) relation type of each query–key pair. *Lift* = mean
attention weight on pairs of that type × number of keys (1 = uniform). Head-averaged; full table in
`results/tables/attention_relation_lift.{csv,md}`, plot in `results/plots/attention_lift.png`.

| Relation (last layer) | relation-aware | vanilla |
|---|---|---|
| question → exactly-matched column (`qc_em`) | 3.30 | 2.51 |
| question → unlinked column (`qc_default`) | 0.29 | 0.83 |
| question → exactly-matched table (`qt_em`) | 4.03 | 1.27 |
| column → its table (`ct_belongs_to`) | 2.79 | 1.77 |
| column → another table (`ct_default`) | 1.05 | 1.46 |
| column → its foreign-key target (`cc_fk_forward`) | **5.75** | 0.73 |
| column → unrelated column (`cc_default`) | 0.65 | 0.74 |
| table → FK-linked table (`tt_fk_forward`) | 2.31 | 1.65 |

Observations (our measurements):

* **Schema linking is used.** The relation-aware encoder separates linked from unlinked schema
  elements sharply (exact-match columns 3.30 vs unlinked 0.29, an 11× ratio; tables 4.03 vs 1.63).
  The vanilla encoder, fed with the same BERT vectors, also prefers exact matches (2.51 vs 0.83,
  a 3× ratio) — BERT's lexical similarity recovers part of the linking signal on its own.
* **Foreign keys are *not* recovered without relations.** Columns of the relation-aware model attend
  to their foreign-key partner with lift 5.75; the vanilla model shows no preference at all (0.73,
  the same as for unrelated columns, 0.74). Column → own-table attention is also stronger with
  relations (2.79 vs 1.77) and does not prefer the own table in the vanilla model (1.77 vs 1.46 for
  other tables). This is the structural information the vanilla model lacks, consistent with its
  much lower JOIN accuracy.
* The example heat map (`results/plots/attention_heatmap_rat.png`, *"Find the number of pets
  whose weight is heavier than 10."*) shows question tokens concentrating on the `Pets` table and its
  columns, while the student tables receive almost no attention.

Caveat: attention weights are a descriptive, not causal, view of what the model uses; lifts are
averaged over heads and examples.

### 20.2 Controlled synthetic experiment: vanilla vs relation-aware attention

{{table:synthetic_relation_experiment.md}}

The task (`ratsql/analysis/synthetic_relations.py`) gives every node a random identity and a
visible label; a random derangement assigns each node one partner through a `link` relation, with
distractor relations elsewhere. The target is the partner's label (one-hop) or the partner's partner's
label (two-hop). Nothing in the node contents identifies the partner, so only relations can solve it.
The two encoders have the same architecture (2 layers, d = 64, 4 heads) and differ only in the relation
embeddings (384 extra parameters). Relation-aware attention solves one-hop perfectly after the first
100 steps and reaches 0.887 on two-hop (two seeds at ≥ 0.87, one seed at 0.79 and still improving;
with 1,500 steps two-hop was 0.751). Vanilla attention converges to 0.293 / 0.286 — exactly the
accuracy of the best relation-free strategy (predicting the most frequent label among the other
nodes: 0.295 / 0.285, simulated). Curves: `results/plots/synthetic_relation_experiment.png`.
