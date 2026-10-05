#### What the ablations show (controlled setting, Spider dev)

Intervals are paired-bootstrap 95 % CIs of the difference to FULL over dev examples. They capture
dev-set sampling noise only; training-seed noise, estimated in §18 from a second FULL seed, is about
2–3 points (−2.3 EM / −2.6 EX between seeds 42 and 43). Effects smaller than that are not interpreted
as established.

1. **Relation-aware attention is the single most important component.** Replacing it with vanilla
   attention of identical size (F / Baseline B) costs −28.3 EM [−31.3, −25.4] and −32.3 EX; removing
   the transformer layers altogether (E) is no worse than keeping vanilla layers (31.8 vs 33.1 EM).
   Extra *unstructured* capacity on top of BERT does not help; the relations do. With a BiLSTM encoder
   the same holds even more strongly: Baseline A (BiLSTM, no relations, no linking) reaches 9.6 EM,
   ablation I (BiLSTM + relations + linking) 49.9 EM.
2. **Schema linking matters, name linking most.** Removing all linking relations (B / Baseline C)
   costs −12.7 EM [−15.5, −10.0] and removing only n-gram name links (C) −7.2 [−9.9, −4.4] — both well
   beyond seed noise. Removing only value links (D) costs −3.1 EM [−5.6, −0.8] / −3.2 EX, the same order
   as the seed spread, so the value-linking effect is suggestive rather than established in our setting
   (the paper, averaging 5 seeds, reports +5.4 EM from value-based linking for its GloVe model). The
   ordering name > value and the size of the joint effect agree with the paper's ablations (w/o
   schema-linking relations −14.8 EM relative to its base), although the settings differ.
3. **Foreign keys mainly affect execution.** Without FK relations (G) EM changes by −1.3 [−3.9, +1.5]
   (indistinguishable from zero) while EX drops by −3.8 [−6.2, −1.2] — a borderline effect given seed
   noise, but consistent with two other measurements: the relation-aware model attends to FK partners
   with lift 5.75 whereas the vanilla model shows no preference (0.73), and FK-graph join inference
   repairs mostly join errors (§15). FK relations appear to help build correct joins, which Exact Match
   does not score.
4. **Fine-grained relation types help.** Collapsing the 43 directed relation types into 6 coarse,
   undirected classes (K) costs −5.1 EM [−7.6, −2.6] / −5.8 EX, so direction and type information
   (primary key vs belongs-to, exact vs partial match, FK direction) is used.
5. **Grammar constraints are essential for validity.** Without the grammar mask (H) only 27.1 % of
   decoded action sequences form a well-formed AST (100 % with the mask) and EM falls to 20.9. Almost
   all failures choose a pointer of the wrong kind (662 × a table where a column is required). Part of
   this gap may be specific to our pointer parameterisation (memory-aligned pointers are normalised
   within each action kind); H2-ref / H2 repeat the comparison with unnormalised direct pointers (below).
6. **BERT helps, but less than relations.** Replacing BERT by a BiLSTM over static word-piece
   embeddings (I) costs −11.5 EM, compared with −28 to −30 for removing relation-aware attention.
