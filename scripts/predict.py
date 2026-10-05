"""Translate a natural-language question into SQL for a Spider-format database.

    python scripts/predict.py --checkpoint experiments/full_model/full/best.pt \
        --db-id concert_singer --question "How many singers do we have?"

    python scripts/predict.py --checkpoint experiments/debug/rat_debug/best.pt --config configs/debug.yaml \
        --db-id company --question "Which employees work in Hyderabad?"

Prints the schema links, the decoded AST, the SQL and (if the database file
exists) the first rows of its execution result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ratsql.inference.predictor import Predictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--db-id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--split", default="dev", help="which tables.json / database dir to use (dev|test)")
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--show-ast", action="store_true")
    args = ap.parse_args()
    p = Predictor(args.checkpoint)
    schema = p.exp.schemas(args.split)[args.db_id]
    db_path = p.exp.db_path_fn(args.split)(args.db_id)
    res = p.predict(args.question, schema, db_path if db_path.exists() else None, beam_size=args.beam_size)
    print(f"Question: {res['question']}")
    print("Schema links:")
    for line in res["schema_links"]:
        print(f"  {line}")
    if args.show_ast and res["ast"]:
        print("AST:\n" + res["ast"])
    print(f"SQL: {res['sql']}" if res["sql"] else f"No SQL produced: {res['error']}")
    if "execution" in res:
        print("Execution:", json.dumps(res["execution"], default=str)[:1000])


if __name__ == "__main__":
    main()
