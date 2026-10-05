# Methodology

This document describes exactly what is implemented in this repository, how it relates to
RAT-SQL (Wang et al., *RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL
Parsers*, ACL 2020), and every place where we deliberately simplified or deviated.
Code references are given as `module:function`.

---

## 1. Task

Given a natural-language question `Q = q_1 … q_n` and a relational database schema
`S = (C, T)` with columns `c_1 … c_m` (including the special `*`), tables `t_1 … t_k`,
primary keys and foreign keys, produce a SQL query `P` that answers `Q` on the database.
In Spider the databases at evaluation time are **unseen** during training, so the model must
generalise to new schemas; schema linking and schema structure are therefore essential.

## 2. Pipeline

```
question + schema
  → tokenisation                       ratsql/utils/text.py
  → schema model (PK / FK)             ratsql/schema/schema.py
  → relational schema graph            ratsql/schema/graph.py            (SchemaGraph)
  → schema linking (n-gram + values)   ratsql/schema/linking.py          (SchemaLinker)
  → relation matrix                    ratsql/schema/relations.py        (RelationVocabulary, RelationMatrixBuilder)
  → BERT contextual representations    ratsql/models/encoders.py         (BertInputEncoder)
  → relation-aware transformer         ratsql/models/rat.py              (RATEncoder)
  → LSTM AST decoder                   ratsql/models/decoder.py          (ASTDecoder)
  → grammar-constrained decoding       ratsql/sql/grammar.py, transition.py
  → AST → Spider SQL dict → SQL string ratsql/sql/ast.py, serializer.py
  → validation / execution / metrics   ratsql/evaluation/*
```

## 3. Data processing

* **Tokenisation** (`utils/text.py:tokenize`): a deterministic regular-expression word
  tokenizer with character offsets (numbers with decimals, words with inner apostrophes /
  hyphens, single symbols). No external NLP toolkit is used.
* **Normalisation** (`normalize_token`): lower-casing, accent stripping, possessive removal and
  a rule-based singulariser (`employees → employee`, `cities → city`, a short irregular list).
  *Simplification:* RAT-SQL used Stanford CoreNLP lemmas; our singulariser misses irregular
  forms outside its list and does not lemmatise verbs.
* **Gold SQL parsing** (`sql/parser.py`): an independent re-implementation of the parsing
  semantics of Spider's `process_sql.py` producing the nested Spider structure. On Spider it
  reproduces the `sql` field shipped with the dataset for **all** dev (1,034/1,034) and test
  (2,147/2,147) queries and 8,658 of 8,659 training queries (see
  `data/processed/spider/stats.json`). Like the original parser, trailing tokens it cannot
  consume are ignored when parsing gold SQL (5 Spider queries contain such malformed tails);
  the single failing training query references a table missing from `tables.json`, so its
  shipped `sql` field is used instead.
* **Splits:** Spider train (`train_spider` + `train_others`) minus 8 held-out *databases*
  (seeded hash, 6 % of the `train_spider` databases) = train; the held-out databases form the
  validation split used for checkpoint selection. Spider dev is used **only** for reporting;
  Spider test only for the final model. No database is shared between splits.

## 4. Relational schema graph and relation matrix

`SchemaGraph.from_schema` builds explicit labelled edges; `RelationMatrixBuilder.build`
converts graph edges to an `N × N` matrix over `X = [q_1..q_n, c_1..c_m, t_1..t_k]`
(pairs without an edge get the default relation of their node-type pair). The relation
vocabulary (`RelationVocabulary("full")`, 43 types + padding) follows Table 1 of the paper plus
the implementation relations of the official code:

| Pair | Relations |
|---|---|
| question–question | `qq_dist_d`, `d ∈ [-2, 2]` (relative position clipped at D = 2) |
| question–column | `qc_default`, `qc_em`, `qc_pm`, `qc_vem`, `qc_vpm`, `qc_num` (+ reverse `cq_*`) |
| question–table | `qt_default`, `qt_em`, `qt_pm` (+ reverse `tq_*`) |
| column–column | `cc_identity`, `cc_same_table`, `cc_fk_forward` (x references y), `cc_fk_backward`, `cc_default` |
| column–table | `ct_primary_key`, `ct_belongs_to`, `ct_foreign_key` (x references a column of y), `ct_any_table` (`*`), `ct_default` (+ reverse `tc_*`) |
| table–table | `tt_identity`, `tt_fk_forward`, `tt_fk_backward`, `tt_fk_both`, `tt_default` |

The **coarse** vocabulary (ablation K) collapses these into 6 undirected, type-agnostic
classes: identity, question–question, linked / unlinked question–schema, structural /
unrelated schema–schema.

## 5. Schema linking (`schema/linking.py`)

*Name-based (n-gram) linking.* For `n = 5 … 1` and every question n-gram that is not made only
of stop words, compare its normalised tokens with each normalised column / table name:
exact match (`EM`) if equal, partial match (`PM`) if the n-gram is a contiguous sub-sequence of
the name. All non-stop-word tokens of a matched n-gram are linked; `EM` overrides `PM`.
This is case-insensitive and plural-insensitive by construction.

*Value-based linking.* For each database we index up to 5,000 distinct values per column
(`schema/db_values.py`), keyed by the same normalisation. A question n-gram equal to a cell value
gives `VEM` (e.g. *“Which employees work in **Hyderabad**?”* → `employee.city` holds
`'Hyderabad'`); a content word occurring inside a short cell value gives `VPM`
(e.g. *Hobbit* → `'The Hobbit'`, at most 3 per token); numeric tokens get `NUM` links to
number/time columns (RAT-SQL's NUMBER/TIME relation).

Precedence per (token, column) pair: `EM > VEM > PM > VPM > NUM`.
`describe_links` prints *question phrase → table / column / value* for debugging.
*Deviation:* RAT-SQL queries the database with `LIKE` per word at preprocessing time; we
pre-index a bounded number of distinct values, which can miss values in very large columns.

## 6. Contextual representation (`models/encoders.py`)

The BERT input is `[CLS] q_1 … q_n [SEP] c_1 [SEP] c_2 [SEP] … t_k [SEP]` with segment id 0
for the question and 1 for the schema; columns are written as `"<type> <name>"`
(e.g. `number age`), `*` as `*`. Each question word and schema element is represented by the
mean of its word-piece vectors (`pooling: mean`; `first_last` reproduces RAT-SQL's average of the
first and last piece). When the sequence exceeds 512 pieces the schema is split into several
segments, each prefixed with the question, and question-word vectors are averaged over segments.
*Deviation:* RAT-SQL (BERT) dropped over-long training examples; we keep all examples.

Encoders used (all BERT-family, uncased, from the Hugging Face Hub):

| Name | Layers | Hidden | Heads | Parameters |
|---|---|---|---|---|
| `google/bert_uncased_L-2_H-128_A-2` (BERT-tiny, debug only) | 2 | 128 | 2 | 4.4 M |
| `google/bert_uncased_L-4_H-512_A-8` (BERT-small, controlled experiments) | 4 | 512 | 8 | 28.5 M |
| `google/bert_uncased_L-8_H-512_A-8` (BERT-medium) | 8 | 512 | 8 | 41.1 M |

All BERT parameters are fine-tuned (none frozen) with a separate learning rate.
RAT-SQL used BERT-large (24 layers, 1024 hidden, 335 M) — not feasible on a 6 GB GPU.

The *simpler contextual encoder* (ablation I, baseline A) replaces BERT by static word-piece
embeddings initialised from BERT-small's input embedding matrix, a BiLSTM over the question and
a separate BiLSTM over each schema element name (as in RAT-SQL's GloVe variant).

## 7. Relation-aware transformer (`models/rat.py`)

For layer input `x_i` and relation id `r_ij`, head `h` (of `H = 8`, `d_z = 256`):

```
e_ij^(h) = x_i W_Q^(h) (x_j W_K^(h) + r^K_ij)^T / sqrt(d_z / H)
α_ij^(h) = softmax_j(e_ij^(h))                       (padding keys masked)
z_i^(h)  = Σ_j α_ij^(h) (x_j W_V^(h) + r^V_ij)
z_i      = W_O [z_i^(1); …; z_i^(H)]
ỹ_i      = LayerNorm(x_i + z_i)
y_i      = LayerNorm(ỹ_i + W_2 ReLU(W_1 ỹ_i))
```

`r^K_ij, r^V_ij ∈ R^{d_z/H}` are learned embeddings of the relation type, shared across heads,
one table per layer (as in the official implementation). Instead of materialising
`N × N × d` relation tensors we compute `q_i·r^K_ij = (q_i E_K^T)[r_ij]` with one `N × R`
product plus a gather, and `Σ_j α_ij r^V_ij = Σ_r (Σ_{j:r_ij=r} α_ij) E_V[r]` with a
scatter-add into `R` buckets. `relation_aware_attention_reference` implements the literal
formula; `tests/test_rat.py` checks both agree to 1e-5. `norm: pre` (default) follows the
official implementation (pre-norm sublayers + final LayerNorm); `norm: post` follows the
equations above. With `use_relations: false` the identical architecture becomes a vanilla
transformer encoder (baseline B / ablation F); `num_layers: 0` removes it (ablation E).
Controlled setting: 4 layers, d_z = 256, 8 heads, FFN 1024, dropout 0.1
(RAT-SQL: 8 layers).

## 8. Grammar, AST and transition system (`sql/grammar.py`, `sql/ast.py`, `sql/transition.py`)

An ASDL-style grammar (20 composite types, 49 constructors + `Reduce`, 3 primitive types
`column`, `table`, `val_ref`) mirrors the Spider SQL structure:

```
sql      = Query(select select, cond_list? where, group_by? group_by, order_by? order_by,
                 limit? limit, from from, iue? iue)
select   = Select(distinct distinct, agg+ aggs)          agg = Agg(agg_op agg, val_unit val)
val_unit = Column(col_unit) | Minus | Plus | Times | Divide     col_unit = ColUnit(agg_op, column, distinct)
cond_list= Conds(cond first, conj* rest)                 conj = And(cond) | Or(cond)
cond     = Cmp(negation, cmp_op, val_unit, value) | Between(negation, val_unit, value, value)
value    = Literal(val_ref) | Subquery(sql) | ColumnValue(col_unit)
group_by = GroupBy(col_unit+ keys, cond_list? having)    order_by = OrderBy(order_dir, val_unit+ keys)
limit    = LimitOne | LimitValue(val_ref)                from = From(table_unit+ units, cond_list? conds)
table_unit = TableRef(table) | TableSubquery(sql)        iue = Intersect(sql) | Union(sql) | Except(sql)
```

FROM is decoded after SELECT/WHERE/GROUP/ORDER so tables are chosen after columns (as in
RAT-SQL). The conversion Spider-dict ↔ AST is lossless (100 % of Spider train/dev/test gold
queries convert; `oracle_em` = 100 % on dev and test, 8,278/8,280 on train — the two failures
are literal values inside FROM sub-queries that cannot be recovered from the question, which
the official evaluation does not mask).

The transition system linearises the AST pre-order into actions: `ApplyRule[r]`,
`SelectColumn[i]`, `SelectTable[j]`, `SelectValue[k]`; optional / repeated fields end with
`Reduce`. Nodes whose type has a single constructor in a non-optional position are expanded
automatically (probability 1 under the grammar; ~25 % shorter sequences). At every step the
*frontier* (first unfilled field) determines the action kind and the valid rules; `DecodeState.apply`
raises `InvalidActionError` for anything else. Mean action sequence length: 42.7 (train),
37.9 (dev); maximum 191.

**Value prediction (our addition).** RAT-SQL did not predict literal values (EM ignores them).
To make predictions executable, `val_ref` is a pointer over a per-example candidate list
(`data/values.py`): `<unk>`, database values from value linking (exact DB spelling), quoted spans,
numbers (incl. number words) and question n-grams up to length 4. On dev, 92.2 % of gold literals
are recoverable; executing the oracle SQL gives 95.0 % EX on dev and 96.4 % on test — the ceiling
of this design. LIKE patterns are re-wrapped as `%text%`.

## 9. LSTM AST decoder (`models/decoder.py`)

```
m_t, h_t = LSTM([a_{t-1}; z_t; h_{p_t}; a_{p_t}; n_{f_t}], m_{t-1}, h_{t-1})
```

* `a_{t-1}`: embedding of the previous action (rule embedding, or a projection of the encoder
  vector of the selected column / table / value);
* `z_t`: 8-head attention over the encoder memory with query `h_{t-1}`;
* `h_{p_t}`, `a_{p_t}`: LSTM state and constructor embedding of the parent node ("parent feeding");
* `n_{f_t}`: frontier node-type + field embedding.

Heads: `ApplyRule` = softmax of a 2-layer tanh MLP of `h_t`; `SelectColumn / SelectTable` =
RAT-SQL's memory-aligned pointer `P(i) = Σ_j λ_j L_{j,i}` with `λ = softmax_j(h_t W_Q (y_j W_K)^T)`
over all memory and memory–schema alignment `L = softmax_i(y_j W'_Q (c_i W'_K)^T)`;
`SelectValue` = bilinear pointer over value-candidate vectors (mean of the RAT outputs of the
candidate's question span + the column vector for database values + a source embedding).

*Grammar constraints.* Scores of all action kinds are concatenated into one vector; the grammar
mask keeps only actions valid for the frontier, so an invalid action has probability exactly 0 in
training *and* decoding. With `grammar_mask: false` (ablation H) the softmax runs over the whole
union and ill-formed trees are detected by the transition system — nothing is repaired afterwards.

*Training* uses teacher forcing with the summed action-level negative log-likelihood, batched over
examples (the parent state is gathered from a per-batch history buffer). *Inference*: batched
greedy decoding or per-example beam search (`beam_size`), max 250 steps. Controlled setting:
LSTM hidden 512, action embedding 128, node-type embedding 64, dropout 0.2.

## 10. Training (`training/trainer.py`)

AdamW with two parameter groups (BERT lr 5e-5, everything else 1e-3, no weight decay on non-BERT
parameters), linear warm-up (5 % of steps) then linear decay, gradient clipping at 1.0, fp16
autocast with dynamic loss scaling on CUDA, gradient accumulation (configurable), length-bucketed
batches with an input-token budget (`batch_size * max word pieces ≤ max_tokens`). Validation
(loss, EM, EX on the held-out training databases) every few epochs; the best checkpoint by
validation EM is kept; optional early stopping; full checkpoint / resume (model, optimizer,
scheduler, scaler, RNG states, history). Seeds are fixed (python, numpy, torch); CUDA
atomics (scatter/index-put backward) make runs non-bitwise-deterministic.

*Numerical precision.* The pointer mixture `p = λ·L` is computed in float32 even under fp16
autocast (`decoder.pointer_fp32: true`). In a first round of experiments it was computed in fp16,
where probabilities below ~6e-8 underflow to zero; an unlikely gold column then produced an
infinite batch loss and the AMP GradScaler silently skipped that optimizer step. All reported
experiments were re-run with the fix; the superseded runs are documented in
`experiments/superseded_fp16_pointer/`. Every epoch now logs `nonfinite_loss_batches` and
`amp_skipped_steps` (a few skipped steps at the very start of training are normal dynamic
loss-scale calibration).

## 11. Evaluation (`evaluation/*`)

* **Exact Match** (`exact_match.py`): an independent re-implementation of Spider's
  exact-set-match — values removed, DISTINCT ignored, FK-equivalent columns unified, ten
  components compared as sets plus identical FROM table units. Official quirks are replicated
  (greedy FK key-set grouping; no FK normalisation inside condition sub-queries; GROUP BY compared
  by column name; LIMIT presence only). Predictions are serialised to SQL and **re-parsed**, exactly
  like the official pipeline. Our hardness classifier reproduces the official dev distribution
  (248 / 446 / 174 / 166) exactly.
* **Execution Accuracy** (`execution.py`): gold and predicted SQL are executed read-only on the
  Spider SQLite database (5 s timeout); results are compared as bags of rows (lists if the gold
  query has ORDER BY) up to a column permutation, following the result comparison of the
  test-suite evaluation (Zhong et al., 2020). We use the single original database (not the
  distilled test suites), so false positives are possible.
* **SQL validity**: `grammar_validity` = a complete, well-formed AST was produced;
  `parse_validity` = the serialised SQL re-parses against the schema; `sql_validity` = the SQL
  executes without error (reported as “SQL Validity” in the tables).
* **Components**: official partial-match accuracy/recall/F1 for select, select (no AGG), where,
  where (no OP), group (no having), group, order, and/or, IUEN, keywords.
* **Hardness / complexity / errors**: `hardness`, `complexity.py`, `analysis/errors.py`.

## 12. Experimental protocol

All baselines and ablations share one controlled setting (`configs/small.yaml`): same data,
same encoder, same budget, same seed, checkpoint chosen on the validation databases, one
evaluation on Spider dev. The registry is `configs/experiments.yaml`; `scripts/run_ablation.py`
runs it and `scripts/collect_results.py` builds every table from the saved metric files.
Ablations are single-seed; a second seed of the FULL configuration differs by 2.3 EM / 2.6 EX points
(report §18), so differences of that size between configurations are treated as within noise.

## 13. Summary of simplifications

| What | Why | Possible effect |
|---|---|---|
| BERT-small / BERT-medium instead of BERT-large | 6 GB GPU, time budget | lower accuracy than the paper |
| 4 RAT layers (paper: 8) | compute | slightly less relational capacity |
| tens of epochs instead of ~90k steps | compute | under-trained relative to the paper |
| rule-based singulariser instead of CoreNLP lemmas | reproducibility, no Java | a few missed name matches |
| bounded value index instead of per-query DB `LIKE` | speed, determinism | may miss values in huge columns |
| schema chunking instead of dropping long examples | keep all data | none on short schemas |
| value pointer over candidates (not in RAT-SQL) | executable SQL | EX ceiling 95 % on dev |
| execution on original DBs (not test suites) | availability | EX may contain false positives |
| greedy decoding by default | speed | beam search available but slower |
