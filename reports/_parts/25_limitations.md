## 25. Limitations

**Scale and budget.** All models were trained on one 6 GB laptop GPU. The controlled setting uses
BERT-small (28.5 M parameters, 4 layers) for 20 epochs (≈ 50 minutes, 4,940 updates); the final
model uses BERT-medium for 40 epochs (101 minutes, 11,520 updates). The paper used BERT-large (≈ 335 M), 8 RAT layers and
90k updates. Our absolute numbers are therefore not a reproduction of the paper's numbers and
should not be compared with them as if they were.

**Single seed for ablations.** Each ablation was trained once (seed 42). The paired-bootstrap
intervals in the ablation table only quantify dev-set sampling noise; a second FULL run (seed 43)
differs by −2.3 EM / −2.6 EX, so ablation effects of about 3 points or less are not interpreted as
established. Two seeds are themselves only a rough variance estimate.

**Evaluation.**
* Execution accuracy uses the single original Spider database per question (not the distilled
  test-suite databases of Zhong et al., 2020), so it can contain false positives; our EX keeps
  DISTINCT semantics and is therefore slightly stricter than the official test-suite executor
  (≈ 1 % of examples disagree; see the cross-check).
* Exact Match inherits the official metric's blind spots: join conditions and literal values are
  ignored, and FK normalisation can equate different columns. Our error analysis shows that several
  "exact matches" are execution failures because of wrong `ON` clauses.
* The Spider test set was evaluated once, for the final model only.

**Implementation simplifications** (details in `methodology.md` §13): rule-based singularisation
instead of CoreNLP lemmas; a bounded cell-value index instead of per-query database look-ups;
schema chunking for inputs over 512 word pieces; a value pointer over deterministic candidates
(ceiling: 95.0 % oracle EX on dev, 96.4 % on test) — RAT-SQL did not predict values at all.

**Engineering issue found and fixed during the study.** A first round of experiments computed
the pointer mixture in fp16; probability underflow produced infinite batch losses and AMP silently
skipped some optimizer steps (unequally across configurations). All reported runs were re-trained
after the fix; the superseded runs are kept and documented in
`experiments/superseded_fp16_pointer/`.

**Grammar ≠ schema consistency.** The grammar guarantees well-formed SQL but not that every
referenced column's table appears in FROM or that join conditions follow foreign keys; these are
the main causes of non-executable predictions of our models (addressed only by the separate
join-inference variant, not in the main numbers).
