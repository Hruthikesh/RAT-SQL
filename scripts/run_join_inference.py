"""Evaluate predictions with schema-graph (foreign-key) join inference at serialisation time.

    python scripts/run_join_inference.py --config configs/small.yaml \
        --predictions full=experiments/full_model/full/dev_predictions.jsonl vanilla=...

For each model, re-serialises every decoded query after ``ratsql.sql.join_inference.infer_joins``
(FROM tables completed from the referenced columns, ON clauses from shortest FK paths) and
re-computes EM / EX / validity. Writes results/tables/join_inference.{csv,md}. The as-decoded
numbers remain the primary results; this is a separate, clearly labelled variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ratsql.evaluation.evaluator import Evaluator, summarize  # noqa: E402
from ratsql.experiment import Experiment  # noqa: E402
from ratsql.sql.join_inference import infer_joins  # noqa: E402
from ratsql.sql.parser import canonical, to_tuples  # noqa: E402
from ratsql.sql.serializer import serialize_sql  # noqa: E402
from ratsql.utils.config import load_config  # noqa: E402
from ratsql.utils.io import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True, help="name=path")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    exp = Experiment(load_config(ROOT / args.config))
    recs = {r["id"]: r for r in read_jsonl(exp.processed_dir / f"{args.split}.jsonl")}
    schemas = exp.schemas(args.split)
    ev = Evaluator(schemas, exp.db_path_fn(args.split))
    out = []
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        rows = read_jsonl(ROOT / path)
        new_rows = []
        changed = 0
        for r in rows:
            if r.get("pred_parsed") is None:
                new_rows.append(ev.evaluate_one(recs[r["id"]], r["pred_sql"], r["grammar_valid"], r.get("decode_error")))
                continue
            schema = schemas[r["db_id"]]
            fixed = infer_joins(canonical(to_tuples(r["pred_parsed"])), schema)
            sql = serialize_sql(fixed, schema)
            changed += sql != r["pred_sql"]
            new_rows.append(ev.evaluate_one(recs[r["id"]], sql, True))
        base, joined = summarize(rows), summarize(new_rows)
        write_jsonl(new_rows, ROOT / "results" / "predictions" / f"{name}_{args.split}_join_inference.jsonl")
        for variant, m in (("as decoded", base), ("+ FK join inference", joined)):
            out.append({"model": name, "variant": variant, "exact_match": m["exact_match"], "execution_accuracy": m.get("execution_accuracy"), "sql_validity": m.get("sql_validity"), "queries_rewritten": changed if variant != "as decoded" else 0, "n": m["n"]})
        print(f"{name}: EM {base['exact_match']:.4f} -> {joined['exact_match']:.4f}; EX {base['execution_accuracy']:.4f} -> {joined['execution_accuracy']:.4f}; valid {base['sql_validity']:.4f} -> {joined['sql_validity']:.4f} ({changed} queries rewritten)")
    df = pd.DataFrame(out)
    tables = ROOT / "results" / "tables"
    df.to_csv(tables / "join_inference.csv", index=False)
    lines = ["### Schema-graph join inference at serialisation time (Spider dev, our measurements)", "",
             "| Model | Variant | EM (%) | EX (%) | SQL valid (%) | Rewritten queries |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.model} | {r.variant} | {100 * r.exact_match:.1f} | {100 * r.execution_accuracy:.1f} | {100 * r.sql_validity:.1f} | {r.queries_rewritten} |")
    (tables / "join_inference.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
