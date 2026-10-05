### Headline results (our measurements; generated from `experiments/*/…_metrics.json`)

{{headline}}

* Our exact-match implementation agrees with the official Spider evaluator on **every** checked
  prediction (1,034 dev, 2,147 test) — see the cross-check table below.
* Controlled ablations (same data, budget and seed) — full table in [Results](#8-results):
  relation-aware → vanilla attention, no schema-linking relations, coarse relation vocabulary and no
  grammar constraints each cost a large, statistically clear amount of accuracy or validity.
