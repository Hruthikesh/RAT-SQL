# Data

This repository does **not** redistribute Spider. Spider (Yu et al., 2018) is released under
CC BY-SA 4.0 on the official website <https://yale-lily.github.io/spider>; the setup script
downloads the official archive from the link published there.

## Obtaining Spider

```bash
python scripts/download_data.py --dataset spider          # download + extract + validate
python scripts/download_data.py --dataset spider --zip /path/to/spider_data.zip   # offline / manual download
python scripts/download_data.py --dataset spider --validate-only
```

If the automatic Google-Drive download fails (quota), download `spider_data.zip` manually from the
Spider website and pass it with `--zip`.

## Expected layout

```
data/raw/spider_data/                  # (any directory under data/raw containing tables.json + dev.json is found)
├── tables.json                        # 166 schemas (train + dev databases)
├── train_spider.json                  # 7,000 examples
├── train_others.json                  # 1,659 examples (Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp)
├── dev.json                           # 1,034 examples, 20 databases
├── test.json                          # 2,147 examples, 40 databases (released by the Spider authors in 2023)
├── test_tables.json
├── database/<db_id>/<db_id>.sqlite    # SQLite databases for train/dev
└── test_database/<db_id>/<db_id>.sqlite
```

`scripts/download_data.py` validates the example counts above, that every `db_id` has a schema,
and that every database file exists.

## Splits used by this project

| Split | Source | Examples | Databases | Use |
|---|---|---|---|---|
| train | `train_spider.json` + `train_others.json` minus validation DBs | 8,280 | 138 | training |
| val | 8 databases held out from `train_spider.json` (seeded hash) | 379 | 8 | checkpoint selection / early stopping |
| dev | `dev.json` | 1,034 | 20 | **reported results** (never used for training or model selection) |
| test | `test.json` | 2,147 | 40 | evaluated once, for the final model only |

No database appears in more than one split (checked by `scripts/preprocess.py`, see
`data/processed/spider/stats.json`).

## Preprocessing

```bash
python scripts/preprocess.py --config configs/small.yaml --oracle-exec
```

writes `data/processed/spider/{train,val,dev,test}.jsonl` and `stats.json`. Each record contains
the tokenised question, schema links, value candidates, the gold SQL parsed into the Spider
structure, the oracle action sequence of the gold AST, Spider hardness and complexity features.
`--oracle-exec` executes the oracle SQL (gold → AST → SQL) to measure the reachable execution
accuracy (upper bound of the value-pointer design).

## Synthetic sample data

`data/samples/synthetic/` is a small, fully synthetic Spider-format dataset (5 toy SQLite
databases, 282 train / 152 dev template-generated questions; train and dev databases are
disjoint). It is generated deterministically by

```bash
python scripts/download_data.py --dataset synthetic
```

and is used by `configs/debug.yaml` and the unit / end-to-end tests. It is **not** a benchmark.
