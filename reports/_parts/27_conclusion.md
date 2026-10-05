## 27. Conclusion

We built a complete, tested re-implementation of RAT-SQL — relational schema graph, n-gram and
value-based schema linking, BERT contextual representations, relation-aware transformer,
grammar-constrained LSTM AST decoder with executable value prediction — together with an evaluation
stack whose exact-match metric agrees with the official Spider evaluator on every checked example.
On a single 6 GB laptop GPU the final model reaches 64.9 % EM / 66.9 % EX on Spider dev and
59.3 % / 63.4 % on the Spider test set.

The controlled experiments answer the research question clearly for this setting: *explicit relational
structure is what makes cross-database generalisation work.* Relation-aware attention is worth ≈ 28
EM points over an equal-size vanilla transformer, schema-linking relations ≈ 13 points, fine-grained
relation types ≈ 5 points, and foreign-key relations mainly improve the executability of joins. The
attention analysis shows the mechanism (attention concentrates on linked elements and foreign-key
partners, which a vanilla encoder does not discover), the synthetic experiment isolates it (relation
information turns an unsolvable task into a solved one), and grammar constraints guarantee
well-formed SQL. The remaining errors are dominated by FROM/join construction — which the schema graph
can repair at unparsing time — by disambiguating between equally well-linked schema elements, and by
compositional reasoning (nesting, set operations, superlatives).

Natural next steps within this code base: infer FROM/ON from the schema graph during decoding rather
than after it (as RAT-SQL does), constrain column choices to tables reachable in the FK graph, train
longer with a larger encoder, and evaluate on the distilled test-suite databases.
