## 6. RAT-SQL architecture (as implemented)

```
question + schema ─► schema graph ─► schema linking ─► relation matrix R (N×N, 43 types)
        │                                                  │
        └─► BERT([CLS] q [SEP] c1 [SEP] … t_k [SEP]) ─► x ─► relation-aware transformer(x, R) ─► memory
                                                                                              │
                   grammar-masked LSTM tree decoder (rules | column / table / value pointers) ◄┘
                                          │
                                     AST ─► Spider SQL dict ─► SQL string ─► re-parse / execute
```

Measured parameter counts (`experiments/*/run_info.json`):

| Module | Controlled FULL (BERT-small) | Final (BERT-medium) |
|---|---|---|
| input encoder (BERT) | 28,500,992 | 41,110,528 |
| relation-aware transformer (4 layers) | 3,302,144 | 3,302,144 |
| AST decoder (LSTM, heads, pointers) | 5,132,082 | 5,132,082 |
| value-candidate encoder | 132,608 | 132,608 |
| **total (all trainable)** | **37,067,826** | **49,677,362** |

The relation embeddings themselves are only 11,264 parameters (4 layers × {K, V} × 44 ids × 32
dims): the vanilla baseline has 37,056,562 parameters. Every equation, design decision and
simplification is documented in [`methodology.md`](methodology.md); code references below are
relative to `src/ratsql/`.

## 7. Schema graph construction

`schema/schema.py` loads tables, columns (including `*`), types, primary and foreign keys from
`tables.json`. `schema/graph.py:SchemaGraph.from_schema` builds explicit labelled edges following
RAT-SQL Table 1 plus the implementation relations: column → table `ct_primary_key` / `ct_belongs_to`
/ `ct_foreign_key` / `ct_any_table` (for `*`) and their reverses; column ↔ column `cc_same_table`,
`cc_fk_forward` (x references y), `cc_fk_backward`; table ↔ table `tt_fk_forward`, `tt_fk_backward`,
`tt_fk_both`. `add_question` adds the linking edges. `schema/relations.py:RelationMatrixBuilder`
converts the graph into an N × N id matrix over `[question | columns | tables]` (pairs without an edge
get the default relation of their node-type pair; question–question pairs get the relative distance
clipped to ±2) and caches the schema block per database. Ablations switch relation families off by
mapping them to defaults, so all variants have the same vocabulary size; the coarse vocabulary maps the
43 types onto 6 (`coarse_of`). `schema/visualize.py` renders graphs and matrices
(`results/plots/schema_graph_*.png`, `relation_matrix_*.png`).

## 8. Schema linking

`schema/linking.py:SchemaLinker` implements (i) **n-gram matching** — normalised 1- to 5-grams
(lower-cased, accent-stripped, singularised; stop-word-only n-grams skipped) compared with column
and table names: exact (`EM`) and partial / contiguous-sub-sequence (`PM`) matches; and (ii)
**value-based matching** against a per-database index of up to 5,000 distinct values per column
(`schema/db_values.py`): exact n-gram = cell value (`VEM`), content word inside a short cell value
(`VPM`, ≤ 3 per token), numeric token ↔ number/time column (`NUM`). Per (token, column) pair the
strongest link wins (`EM > VEM > PM > VPM > NUM`). `describe_links` prints *phrase → table / column /
value*, e.g. for the running example *"Which employees work in Hyderabad?"*:
`'employees' → TABLE employee [EM]`, `'Hyderabad' → VALUE 'Hyderabad' in employee.city [VEM]`.

Measured linker quality on dev against gold SQL (`results/tables/linking_analysis.json`,
model-independent): 93.1 % of gold columns receive a name or value link (54.2 % an exact name link,
24.3 % a value link), 80.2 % of gold tables are linked; link precision is 63.5 % for `EM`, 29.9 % for
`VEM`, 11.6 % for `PM`, 4.5 % for `VPM`, 1.9 % for `NUM`. High recall with low precision for partial
matches is exactly why linking is encoded as *soft relations* rather than hard decisions.

## 9. BERT contextual representation

`models/encoders.py:BertInputEncoder` reads `[CLS] question [SEP] col_1 [SEP] … tab_k [SEP]`
(segment ids 0 / 1; columns written as `"<type> <name>"`), mean-pools word pieces into one vector per
question word, column and table, and splits schemas longer than 512 pieces into question-prefixed
segments (question vectors averaged across segments). All BERT weights are fine-tuned with their own
learning rate (5e-5). Encoders: `google/bert_uncased_L-4_H-512_A-8` (BERT-small, 4 layers, hidden
512) for the controlled experiments, `google/bert_uncased_L-8_H-512_A-8` (BERT-medium, 8 layers) for
the final model. The simpler encoder of ablation I / baseline A (`LSTMInputEncoder`) uses static
word-piece embeddings initialised from BERT-small plus BiLSTMs over the question and over each
schema-element name.

## 10. Relation-aware encoder

`models/rat.py` implements

```
e_ij^(h) = x_i W_Q^(h) (x_j W_K^(h) + r^K_ij)^T / sqrt(d_z/H)      α_ij^(h) = softmax_j(e_ij^(h))
z_i^(h)  = Σ_j α_ij^(h) (x_j W_V^(h) + r^V_ij)                     z_i = W_O [z_i^(1); …; z_i^(H)]
```

with residual connections, LayerNorm (pre-norm as in the official implementation; post-norm
available) and a ReLU feed-forward block; 4 layers, d_z = 256, H = 8, FFN 1024, dropout 0.1,
per-layer relation embeddings of size d_z / H shared by the heads. The relation terms are computed
without materialising N × N × d tensors: `q_i · r^K_ij = (q_i E_K^T)[r_ij]` (one N × R product and a
gather) and `Σ_j α_ij r^V_ij = Σ_r (Σ_{j: r_ij = r} α_ij) E_V[r]` (a scatter-add); the unit tests check
this against the literal formula (max. abs. difference < 1e-5), that zero relation embeddings reproduce
vanilla attention, and that changing one relation id changes only the affected query row.

## 11. Decoder and grammar-constrained SQL generation

**Grammar and AST.** `sql/grammar.py` defines an ASDL grammar of Spider SQL (20 composite types, 3
primitive types — `column`, `table`, `val_ref` —, 49 constructors + `Reduce`, 40 grammar masks); FROM is
decoded after SELECT / WHERE / GROUP BY / ORDER BY / LIMIT. `sql/ast.py` converts losslessly between
Spider SQL dicts and ASTs; `sql/transition.py` linearises ASTs pre-order into `ApplyRule`,
`SelectColumn`, `SelectTable`, `SelectValue` and `Reduce` actions, expands single-constructor nodes
automatically, and exposes at every step the *frontier* (legal action kind, legal rules, parent step,
parent rule). `DecodeState.apply` raises `InvalidActionError` for any illegal action.

**Decoder.** `models/decoder.py:ASTDecoder` is a TRANX / RAT-SQL LSTM
(`m_t, h_t = LSTM([a_{t-1}; z_t; h_{p_t}; a_{p_t}; n_{f_t}])`, hidden 512) with 8-head attention over
the encoder memory, a 2-layer MLP over rules, RAT-SQL's memory-aligned pointers for columns and tables
(`P(i) = Σ_j λ_j L_{j,i}`), and a bilinear pointer over value candidates (our addition: database values
from value linking, quoted spans, numbers, question n-grams — so predictions are executable).
Training is batched teacher forcing with the action-level NLL; inference is batched greedy decoding or
per-example beam search (max 250 actions).

**Constraint.** All action scores are concatenated and every action that is illegal at the frontier
is masked to −∞ *before* the softmax, in training and in decoding, so ill-formed trees have probability
zero; with `grammar_mask: false` (ablations H / H2) the softmax runs over all actions and the transition
system detects ill-formed outputs — nothing is repaired afterwards. Decoded ASTs are serialised by
`sql/serializer.py` (globally unique `T<k>` aliases, single-quoted literals) and re-parsed by our
Spider parser before scoring, exactly as the official evaluation does.

## 12. Training methodology

`training/trainer.py`: AdamW with two parameter groups (BERT 5e-5 with weight decay 0.01, other
parameters 1e-3), 5 % linear warm-up then linear decay to zero, gradient clipping at 1.0, fp16
autocast with dynamic loss scaling, float32 pointer mixture (§22), batches of ≤ 64 examples bucketed by
decoder length under a budget of 20,480 BERT word pieces (16,384 for BERT-medium), 20 epochs
(controlled) / 40 epochs (final), validation EM / EX on the held-out *training* databases every 3 (4)
epochs, best checkpoint by validation EM, full checkpoint / resume (model, optimizer, scheduler,
scaler, RNG states, history), fixed seeds (42; 43 for the variance run). Every epoch logs training
loss, step accuracy, the number of batches with a non-finite loss and the number of optimizer steps
skipped by the AMP scaler.
