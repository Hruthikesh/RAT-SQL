`scripts/official_eval_crosscheck.py` runs the official Spider evaluation code (test-suite version,
downloaded to `third_party/` only for this check) on the same predictions, example by example.
**Exact Match agrees on every example** (1,034 dev predictions of each model, 2,147 test predictions).
Execution accuracy agrees on 98.5–98.9 % of examples; the official executor removes `DISTINCT` from
both queries before comparing results while ours keeps it, so ours is slightly stricter (by 0.8–1.2
points). The official executor's `asyncio` timeout cannot interrupt a blocking SQLite call, so
predictions that hit our 5-second timeout (accidental cartesian products; 3 of the 2,147 test
predictions, none on dev) would hang it; they are skipped by the cross-check and counted as wrong in
the "official" column.
