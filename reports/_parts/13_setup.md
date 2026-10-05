## 13. Experimental setup

| Item | Setting |
|---|---|
| Data | Spider 1.0 (official archive incl. the released test set); train = `train_spider` + `train_others` minus 8 held-out databases (8,280 ex.); val = held-out databases (379 ex.); dev = 1,034 ex. / 20 DBs (reported); test = 2,147 ex. / 40 DBs (final model only) |
| Hardware | 1 × NVIDIA RTX 3050 Laptop GPU (6 GB), Intel i5-13450HX, 16 GB RAM, Windows 11 |
| Software | Python 3.13.3, PyTorch 2.14.0+cu126, transformers 5.17.0 |
| Controlled setting (`configs/small.yaml`) | BERT-small encoder (fine-tuned, lr 5e-5), 4 RAT layers (d = 256, 8 heads, FFN 1024, dropout 0.1), LSTM decoder 512, lr 1e-3, AdamW, 20 epochs, batch ≤ 64 examples / 20,480 word pieces, fp16 autocast with float32 pointer mixture, grad-clip 1.0, 5 % warm-up + linear decay, seed 42 (43 for the variance run) |
| Final model (`configs/full.yaml`) | same, with BERT-medium (8 layers), 40 epochs and a 16,384-word-piece batch budget |
| Model selection | best validation EM (held-out training databases), evaluated every 3 epochs (4 for the final model) |
| Decoding | greedy, grammar-constrained, max 250 actions (beam search k = 5 evaluated separately, §23) |
| Metrics | Exact Match (official semantics; agreement with the official script verified per example), Execution Accuracy (strict: DISTINCT kept; 5 s timeout), SQL validity (executes without error), well-formed AST rate, hardness and component breakdowns |

Budget calibration: before the controlled runs, one 8-epoch run of the FULL configuration
(`experiments/sanity/small_8ep`) was used to choose batch size and epoch budget; its numbers are
reported only as calibration and are not part of any comparison.

## 14. Baselines

* **Baseline A — simple schema-aware model.** Static word-piece embeddings (initialised from
  BERT-small's input embeddings) + BiLSTM over the question and over each schema-element name, no
  relation-aware layers, no schema linking; same grammar-constrained decoder with column / table /
  value pointers. It is "schema-aware" only through the pointer over encoded schema names.
* **Baseline B — vanilla transformer.** The FULL architecture with relation embeddings removed
  (identical layers, heads and widths): the encoder sees question and schema jointly but no
  relations (= ablation F).
* **Baseline C — schema-aware without schema linking.** RAT with all schema-graph relations but no
  question–schema link relations (= ablation B).
* **FULL.** BERT + relation-aware transformer + n-gram and value linking + grammar-constrained
  decoding.

All four share data, seed, budget and selection protocol.
