"""Dataset validation: checks the expected Spider directory structure and contents."""

from __future__ import annotations

from pathlib import Path

from ratsql.data.spider import DatasetSpec, load_split
from ratsql.utils.io import read_json

EXPECTED_SPIDER_COUNTS = {"train_spider.json": 7000, "train_others.json": 1659, "dev.json": 1034}


def validate_dataset(spec: DatasetSpec, check_databases: bool = True) -> dict:
    report: dict = {"root": str(spec.root), "ok": True, "errors": [], "warnings": [], "splits": {}}

    def err(msg: str) -> None:
        report["ok"] = False
        report["errors"].append(msg)

    if not spec.root.exists():
        err(f"root {spec.root} does not exist")
        return report
    for split, files in spec.split_files.items():
        missing = [f for f in files if not (spec.root / f).exists()]
        if missing:
            (report["warnings"] if split == "test" else report["errors"]).append(f"split {split}: missing {missing}")
            if split != "test":
                report["ok"] = False
            continue
        tables_path = spec.tables_path(split)
        if not tables_path.exists():
            err(f"split {split}: missing {tables_path.name}")
            continue
        tables = {t["db_id"]: t for t in read_json(tables_path)}
        examples = load_split(spec, split)
        db_ids = sorted({e["db_id"] for e in examples})
        missing_schema = [d for d in db_ids if d not in tables]
        missing_db = [d for d in db_ids if check_databases and not spec.db_path(split, d).exists()]
        fields_missing = [i for i, e in enumerate(examples) if not {"db_id", "question", "query"} <= set(e)]
        if missing_schema:
            err(f"split {split}: {len(missing_schema)} db_ids without schema, e.g. {missing_schema[:3]}")
        if missing_db:
            err(f"split {split}: {len(missing_db)} databases missing, e.g. {missing_db[:3]}")
        if fields_missing:
            err(f"split {split}: {len(fields_missing)} examples missing required fields")
        per_file = {}
        for f in files:
            n = len(read_json(spec.root / f))
            per_file[f] = n
            exp = EXPECTED_SPIDER_COUNTS.get(f)
            if spec.name == "spider" and exp is not None and n != exp:
                report["warnings"].append(f"{f}: {n} examples (expected {exp} for Spider 1.0)")
        report["splits"][split] = {"examples": len(examples), "databases": len(db_ids), "files": per_file}
    return report


def format_report(report: dict) -> str:
    lines = [f"Dataset root: {report['root']}", f"Status: {'OK' if report['ok'] else 'FAILED'}"]
    for s, info in report["splits"].items():
        lines.append(f"  {s:6s} examples={info['examples']:6d} databases={info['databases']:4d} files={info['files']}")
    for w in report["warnings"]:
        lines.append(f"  WARNING: {w}")
    for e in report["errors"]:
        lines.append(f"  ERROR: {e}")
    return "\n".join(lines)


__all__ = ["validate_dataset", "format_report", "Path"]
