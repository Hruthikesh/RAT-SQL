**Reading the main table.** EM / EX 95 % intervals are percentile bootstrap intervals over dev
(or test) examples.

* The controlled FULL model (61.4 % EM, 63.7 % EX) outperforms every baseline by a wide margin:
  Baseline C (relations without linking) by 12.7 EM, Baseline B (vanilla transformer) by 28.3 EM and
  Baseline A (BiLSTM, no relations, no linking) by 51.8 EM. The intervals do not overlap.
* Executable-SQL rates follow the same order (31.7 % → 54.7 % → 88.0 % → 92.4 %): schema structure
  in the encoder is what makes the decoder build consistent FROM clauses.
* The final model (BERT-medium, 40 epochs) adds 3.5 EM / 3.2 EX over the controlled model on dev.
  Its validation EM was still rising at epoch 40, so the budget — not the architecture — limits it.
* On the Spider test set the final model scores 59.3 % EM / 63.4 % EX, 5.6 EM below its dev score
  (the paper's gap between dev and test is 4.1 EM for RAT-SQL + BERT). The test set was evaluated once,
  after all development decisions had been made on the validation databases.

**Relation to the paper (not a reproduction).** The paper's RAT-SQL + BERT reports 69.7 / 65.6 EM
(dev / test) with BERT-large, 8 RAT layers and 90k updates; its GloVe model reports 62.7 / 57.2. Our
final model (BERT-medium, 4 RAT layers, 11.5k updates, 101 minutes) lies between the two on dev
(64.9) and between the two on test (59.3). By difficulty on dev, ours reaches 83.9 / 67.7 / 52.9 / 41.6
(easy / medium / hard / extra) vs the paper's 86.4 / 73.6 / 62.1 / 42.9 for RAT-SQL + BERT; the gap is
largest on medium and hard queries. Because encoder size, depth and training budget all differ, these
comparisons are indicative only.
