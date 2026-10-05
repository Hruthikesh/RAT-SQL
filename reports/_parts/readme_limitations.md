* **Budget and scale.** One 6 GB laptop GPU: BERT-small (controlled) / BERT-medium (final), 4 RAT
  layers, 20 / 40 epochs — far below the paper's BERT-large, 8 layers, 90k updates. Our numbers are not
  a reproduction of the paper's.
* **Single seed per ablation.** Paired-bootstrap CIs cover dev-set sampling noise only; a second
  FULL seed differs by 2.3 EM / 2.6 EX points, so effects of ≈ 3 points or less (value linking, foreign
  keys on EM) are suggestive, not established.
* **Evaluation caveats.** EX is measured on the original Spider databases (not test suites) and keeps
  DISTINCT (slightly stricter than the official executor); EM ignores join conditions and values;
  Spider gold queries contain some annotation errors (we found one verified case in a 30-example sample).
* **Simplifications** (see `reports/methodology.md` §13): rule-based singularisation instead of
  CoreNLP lemmas, bounded cell-value index, schema chunking for > 512 word pieces, value prediction by
  pointing to deterministic candidates (ceiling 95.0 % oracle EX on dev).
* **Schema consistency is not enforced during decoding.** The grammar guarantees well-formed SQL, not
  that every referenced column's table is joined; this is the largest source of non-executable
  predictions (addressed only by the separately reported FK join-inference variant).
* **An fp16 underflow bug** in the pointer mixture silently skipped mixed-precision optimizer steps in
  a first round of experiments; it was fixed and every reported run was re-trained
  (`experiments/superseded_fp16_pointer/`).
* Timing columns (training minutes, ms / example) are wall-clock on a shared laptop and vary with load.
