## 1. Abstract

We re-implement RAT-SQL (Wang et al., 2020) from scratch and use it to study *how explicit relational
schema representation, schema linking, relation-aware attention and grammar-constrained decoding
affect Text-to-SQL generalisation to unseen databases*. The system builds a relational schema graph
(43 relation types), links question n-grams and database values to schema elements, encodes question
and schema jointly with BERT followed by a relation-aware transformer, and decodes an SQL abstract
syntax tree with a grammar-constrained LSTM decoder; unlike the original, it also predicts literal
values, so every prediction is executable. Evaluation uses an independent re-implementation of
Spider's exact-set-match metric that agrees with the official evaluator on every one of the 1,034 dev
and 2,147 test predictions we checked, plus execution accuracy and SQL validity.

All numbers are ours, from single runs on one 6 GB laptop GPU. The final model (BERT-medium, 40
epochs, 101 minutes of training) reaches **64.9 % Exact Match / 66.9 % Execution Accuracy on Spider dev
and 59.3 % / 63.4 % on the Spider test set** (the paper reports 69.7 % / 65.6 % EM with BERT-large and a
much larger budget). In a controlled setting (BERT-small, 20 epochs) the full model obtains 61.4 % EM;
replacing relation-aware attention with vanilla attention of the same size costs 28.3 EM points,
removing schema-linking relations 12.7, name linking alone 7.2, value linking alone 3.1, and collapsing
the relation vocabulary to 6 coarse types 5.1; removing foreign-key relations mainly lowers execution
accuracy (−3.8 EX, −1.3 EM). A second training seed differs by 2.3 EM points, so the effects of about
3 points or less (value linking, foreign keys) are suggestive rather than established. Grammar
constraints raise the share of well-formed SQL trees from 27 % to 100 %. Error, complexity and attention analyses show that the relation-aware encoder attends to
foreign-key partners (5.8× uniform, vs no preference without relations) and that the dominant
remaining failure is FROM/join construction — which schema-graph join inference at serialisation time
reduces (+2.9 EX) without retraining.
