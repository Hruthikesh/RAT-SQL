"""Cross-check our metrics against the official Spider evaluation scripts, example by example.

Requires the official scripts (not part of this repository):
    python scripts/download_data.py --dataset none --official-eval      # -> third_party/spider_eval
    pip install nltk                                                     # used by the official parser

    python scripts/official_eval_crosscheck.py --predictions experiments/full_model/full/dev_predictions.jsonl [--exec]

For each prediction the official ``get_sql`` / ``rebuild_sql_*`` / ``Evaluator.eval_exact_match``
pipeline (and optionally ``eval_exec_match``) is run exactly as in its ``evaluate`` function, and
compared with the exact_match / exec_match fields computed by our re-implementation.
Writes results/metrics/official_crosscheck_<name>.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "spider_eval"
sys.path.insert(0, str(ROOT / "src"))

from ratsql.data.spider import find_spider_root  # noqa: E402
from ratsql.utils.io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--exec", action="store_true", help="also run the official execution comparison")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if not (OFFICIAL / "evaluation.py").exists():
        sys.exit("official scripts missing: python scripts/download_data.py --dataset none --official-eval")
    nltk_dir = ROOT / "third_party" / "nltk_data"
    os.environ["NLTK_DATA"] = str(nltk_dir)
    import nltk

    nltk.data.path.insert(0, str(nltk_dir))
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=str(nltk_dir), quiet=True)
    sys.path.insert(0, str(OFFICIAL))
    import evaluation as off  # type: ignore
    from process_sql import Schema, get_schema, get_sql  # type: ignore

    spider = find_spider_root(ROOT / "data" / "raw")
    db_dir = spider / ("test_database" if args.split == "test" else "database")
    table_file = spider / ("test_tables.json" if args.split == "test" else "tables.json")
    kmaps = off.build_foreign_key_map_from_json(str(table_file))
    rows = read_jsonl(ROOT / args.predictions if not Path(args.predictions).is_absolute() else args.predictions)
    evaluator = off.Evaluator()
    agree_em = agree_ex = n_ex = 0
    official_em = official_ex = ours_em = ours_ex = 0
    disagreements = []
    skipped_timeouts: list[str] = []
    for r in rows:
        db = str(db_dir / r["db_id"] / f"{r['db_id']}.sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, r["gold_sql"])
        p_str = (r["pred_sql"] or "").replace("value", "1")
        try:
            p_sql = get_sql(schema, p_str)
        except Exception:
            p_sql = {"except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [], "intersect": None, "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []}
        kmap = kmaps[r["db_id"]]
        g_valid = off.build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = off.rebuild_sql_col(g_valid, off.rebuild_sql_val(g_sql), kmap)
        p_valid = off.build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = off.rebuild_sql_col(p_valid, off.rebuild_sql_val(p_sql), kmap)
        em = int(evaluator.eval_exact_match(p_sql, g_sql))
        official_em += em
        ours_em += int(bool(r["exact_match"]))
        if em == int(bool(r["exact_match"])):
            agree_em += 1
        else:
            disagreements.append({"id": r["id"], "kind": "em", "official": em, "ours": r["exact_match"], "gold": r["gold_sql"], "pred": r["pred_sql"]})
        if args.exec and "timeout" in (r.get("exec_error") or ""):
            # The official executor's asyncio timeout cannot interrupt a blocking sqlite call, so a
            # prediction that hit our 5 s timeout (e.g. an accidental cartesian product) would hang it.
            skipped_timeouts.append(r["id"])
        elif args.exec:
            with contextlib.redirect_stdout(io.StringIO()):
                ex = int(off.eval_exec_match(db=db, p_str=r["pred_sql"] or "SELECT", g_str=r["gold_sql"], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False))
            n_ex += 1
            official_ex += ex
            ours_ex += int(bool(r.get("exec_match")))
            if ex == int(bool(r.get("exec_match"))):
                agree_ex += 1
            else:
                disagreements.append({"id": r["id"], "kind": "exec", "official": ex, "ours": r.get("exec_match"), "gold": r["gold_sql"], "pred": r["pred_sql"]})
    n = len(rows)
    res = {
        "predictions": args.predictions,
        "n": n,
        "official_exact_match": official_em / n,
        "ours_exact_match": ours_em / n,
        "em_agreement": agree_em / n,
        "em_disagreements": sum(d["kind"] == "em" for d in disagreements),
    }
    if args.exec:
        res.update({
            "official_execution_accuracy_on_compared": official_ex / n_ex,
            "ours_execution_accuracy_on_compared": ours_ex / n_ex,
            "exec_compared": n_ex,
            "exec_skipped_timeouts": skipped_timeouts,
            "official_execution_accuracy_all": official_ex / n,  # skipped timeouts counted as wrong (they also are for us)
            "exec_agreement": agree_ex / n_ex,
            "exec_disagreements": sum(d["kind"] == "exec" for d in disagreements),
        })
    res["disagreement_examples"] = disagreements[:30]
    name = args.name or Path(args.predictions).parent.name
    write_json(res, ROOT / "results" / "metrics" / f"official_crosscheck_{name}.json")
    print(json.dumps({k: v for k, v in res.items() if k != "disagreement_examples"}, indent=1))


if __name__ == "__main__":
    main()
