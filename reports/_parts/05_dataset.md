## 5. Dataset

**Source.** Spider 1.0 (Yu et al., 2018; CC BY-SA 4.0), downloaded from the link on the official
website by `scripts/download_data.py` (archive `spider_data.zip`, which includes the test set released
by the Spider authors) and validated for example counts, schema coverage and database files. The data
is not redistributed with this repository.

**Splits (strictly database-disjoint).** All numbers from `data/processed/spider/stats.json`.

| Split | Source | Examples | Databases | Easy / Medium / Hard / Extra | Role |
|---|---|---|---|---|---|
| train | `train_spider` + `train_others` minus validation DBs | 8,280 | 138 | 1,885 / 2,838 / 1,846 / 1,711 | training |
| val | 8 `train_spider` databases held out by a seeded hash | 379 | 8 | 98 / 161 / 76 / 44 | checkpoint selection |
| dev | `dev.json` | 1,034 | 20 | 248 / 446 / 174 / 166 | reported results |
| test | `test.json` | 2,147 | 40 | 470 / 857 / 463 / 357 | final model only, evaluated once |

The validation databases are `bike_1`, `county_public_safety`, `customers_card_transactions`,
`film_rank`, `insurance_and_eClaims`, `restaurant_1`, `ship_mission`, `solvency_ii`. No database
occurs in more than one split (all 6 pairwise overlaps are empty). The dev difficulty distribution
produced by our hardness classifier (248 / 446 / 174 / 166) equals the official one.

**Schemas.** The 166 train + dev databases have on average 5.3 tables (2–26), 27.1 columns (6–352)
and 4.8 foreign keys (0–25); their schema graphs have on average 284 labelled edges
(`results/tables/schema_graph_stats.csv`).

**Preprocessing checks** (deterministic; `scripts/preprocess.py --oracle-exec`):

| Check | train | val | dev | test |
|---|---|---|---|---|
| our SQL parse = `sql` field shipped with Spider | 8,279 / 8,280 | 379 / 379 | 1,034 / 1,034 | 2,147 / 2,147 |
| gold SQL → grammar AST | 8,280 / 8,280 | 379 / 379 | 1,034 / 1,034 | 2,147 / 2,147 |
| oracle round trip (AST → SQL → re-parse) is an exact match | 8,278 / 8,280 | 379 / 379 | 1,034 / 1,034 | 2,147 / 2,147 |
| gold literals recoverable as value candidates | 93.0 % | 94.5 % | 92.2 % | 96.9 % |
| oracle execution accuracy (upper bound of our value design) | 96.2 % | 96.8 % | 95.0 % | 96.4 % |
| mean question tokens / mean actions / max actions | 13.3 / 42.7 / 191 | 14.3 / 35.6 / 83 | 13.8 / 37.9 / 123 | 13.9 / 39.3 / 148 |
| mean value candidates per question | 32.4 | 35.4 | 34.0 | 33.5 |

The one training query our parser rejects references a table missing from `tables.json` (its shipped
`sql` field is used). The two training round-trip failures contain literals inside a FROM sub-query
that cannot be recovered from the question, and the official metric does not mask values there.

**Synthetic data.** For DEBUG mode and the tests, `ratsql/data/synthetic.py` generates a Spider-format
dataset of 5 toy SQLite databases (282 train / 152 dev questions from ~20 templates, disjoint
databases). It is never used for reported results.
