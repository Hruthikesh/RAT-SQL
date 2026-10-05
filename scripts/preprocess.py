"""Deterministic preprocessing (schema linking, value candidates, ASTs, action sequences).

    python scripts/preprocess.py --config configs/full.yaml
    python scripts/preprocess.py --config configs/debug.yaml
    python scripts/preprocess.py --config configs/full.yaml --oracle-exec   # also execute oracle SQL

Outputs ``<processed_dir>/{train,val,dev[,test]}.jsonl`` and ``stats.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ratsql.data.preprocess import run_preprocessing  # noqa: E402
from ratsql.experiment import resolve_path  # noqa: E402
from ratsql.data.spider import make_spec  # noqa: E402
from ratsql.utils.config import load_config  # noqa: E402
from ratsql.utils.io import iter_jsonl, write_json  # noqa: E402


def oracle_execution(processed_dir: Path, spec, splits: list[str]) -> dict:
    """Execute the oracle SQL (gold -> AST -> SQL) and compare with the gold result."""
    from ratsql.evaluation.execution import execution_match

    out = {}
    for split in splits:
        p = processed_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        n = match = gold_fail = 0
        for r in iter_jsonl(p):
            if not r.get("oracle_sql"):
                continue
            db = spec.db_path("test" if split == "test" else "train", r["db_id"])
            res = execution_match(db, r["oracle_sql"], r["query"], timeout=10.0)
            n += 1
            match += res["exec_match"]
            gold_fail += not res["gold_exec_ok"]
        out[split] = {"n": n, "oracle_exec_match": match, "oracle_ex": round(match / max(1, n), 4), "gold_exec_failures": gold_fail}
        print(f"[oracle-exec] {split}: {out[split]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--oracle-exec", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    data_cfg["root"] = str(resolve_path(data_cfg["root"]))
    spec = make_spec(data_cfg)
    out_dir = resolve_path(cfg["data"]["processed_dir"])
    stats = run_preprocessing(spec, cfg["preprocess"], out_dir, out_dir.parent / "value_index" / spec.name, args.splits)
    for split, s in stats.items():
        if split == "database_overlap":
            continue
        keys = ["examples", "databases", "ast_ok", "oracle_em", "literal_coverage", "mean_actions"]
        print(split, {k: s.get(k) for k in keys})
    print("database overlap between splits:", {k: len(v) for k, v in stats["database_overlap"].items()})
    if args.oracle_exec:
        stats["oracle_execution"] = oracle_execution(out_dir, spec, [s for s in stats if s != "database_overlap"])
        write_json(stats, out_dir / "stats.json")


if __name__ == "__main__":
    main()
