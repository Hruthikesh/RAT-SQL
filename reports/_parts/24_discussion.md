## 24. Discussion

We return to the research questions, answering each only from the measurements above.

**Does schema linking improve Text-to-SQL?** Yes. Removing all question–schema link relations costs
12.7 EM / 10.7 EX (Baseline C vs FULL), roughly doubling schema-linking errors (107 → 229). Name
(n-gram) links carry most of the benefit (−7.2 EM without them); value links add a smaller amount
(−3.1 EM without them, within seed noise). The linker itself has high recall (93 % of gold columns
linked) but low precision for partial matches (12 %), and the model — not the linker — is the
bottleneck: most linking-type failures pick the wrong element among several linked candidates.

**Does relation-aware attention improve over vanilla attention?** Decisively: −28.3 EM with vanilla
attention of identical size, and vanilla layers are no better than no layers at all (33.1 vs 31.8 EM).
Relations cost 11,264 parameters (0.03 % of the model). The synthetic experiment isolates the reason —
without relations, attention cannot route information along pairwise structure and converges to the
best relation-free statistic — and the attention analysis shows it on Spider: foreign-key partners
receive 5.75× uniform attention with relations and none without.

**Which schema relations matter most?** In decreasing order of measured effect: question–schema link
relations (−12.7 EM), the relation-aware mechanism as a whole (−28.3), fine-grained relation *types*
versus 6 coarse classes (−5.1 EM), foreign-key relations (−1.3 EM / −3.8 EX, mostly on joins).

**How important is value linking?** As a *relation*, moderately (−3.1 EM / −3.2 EX, not beyond seed
noise). As a *value source* it is essential for execution: database values found by value linking give
the decoder correctly spelled literals, and 92.2 % of gold literals are recoverable as candidates (the
oracle ceiling of our value design is 95.0 % EX on dev).

**Complexity.** Accuracy degrades mainly with the number of joined tables (71.5 → 54.2 → 18.3 % EM for
1 → 2 → 3 tables) and with nesting / set operations (35–41 % EM); clause-level operators (aggregation,
GROUP BY, ORDER BY) are comparatively robust. Relation-aware attention helps most exactly where
structure matters: +37.9 EM on two-table queries versus +24.5 on single-table queries.

**Less training data.** EM grows roughly log-linearly from 4.2 % (1 %, 83 examples) to 61.4 % (100 %);
each of the last two doublings adds ≈ 7 points, so returns diminish per example but have not saturated.

**Where does the model fail?** FROM / join construction (columns from un-joined tables, wrong ON
clauses — 77 non-executable predictions, and 13 exact matches that fail execution), disambiguation among
equally well-linked elements, then compositional reasoning (nesting, set operations, superlatives,
operators implied by wording). Some measured errors are Spider annotation errors (verified in one case).

**Does grammar-constrained decoding improve SQL validity?** Yes — it guarantees well-formed trees
(100 % vs 27.1 % without the mask in ablation H). §16 separates how much of the unconstrained model's
failure is due to the pointer parameterisation (H2-ref / H2). Grammar validity is not schema
consistency: 7.6 % of the FULL model's well-formed predictions still fail to execute, which schema-graph
join inference reduces to ≈ 1.5 %.

**Complexity–performance trade-off.** The large gains come from cheap components: relation embeddings
(+28 EM for 0.03 % more parameters) and linking (no parameters). BERT-small instead of a BiLSTM adds
11.5 EM for 11.7 M more parameters and ≈ 30 % more training time; BERT-medium over BERT-small adds 3.5
EM (with twice the epochs) for 12.6 M more parameters. Greedy decoding costs ≈ 7–32 ms per example on
this GPU (batched); beam search is far more expensive for little or no gain (§23).
