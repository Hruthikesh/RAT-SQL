"""Locating and loading Spider-format datasets.

Expected layout (the official ``spider_data.zip`` extracted under ``data/raw``)::

    data/raw/spider/
        tables.json
        train_spider.json
        train_others.json
        dev.json
        database/<db_id>/<db_id>.sqlite
        test.json                 (optional, released 2023)
        test_tables.json          (optional)
        test_database/<db_id>/<db_id>.sqlite   (optional)

The synthetic debug dataset (``data/samples/synthetic``) uses the same layout
with ``train.json`` / ``dev.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ratsql.schema.schema import Schema, load_schemas
from ratsql.utils.io import read_json


@dataclass
class DatasetSpec:
    name: str
    root: Path
    split_files: dict[str, list[str]]  # split -> json file names (relative to root)
    tables_file: str = "tables.json"
    database_dir: str = "database"
    split_tables: dict[str, str] = field(default_factory=dict)  # split-specific tables.json
    split_database_dirs: dict[str, str] = field(default_factory=dict)

    def tables_path(self, split: str) -> Path:
        return self.root / self.split_tables.get(split, self.tables_file)

    def database_dir_for(self, split: str) -> Path:
        return self.root / self.split_database_dirs.get(split, self.database_dir)

    def db_path(self, split: str, db_id: str) -> Path:
        return self.database_dir_for(split) / db_id / f"{db_id}.sqlite"

    def available_splits(self) -> list[str]:
        return [s for s, files in self.split_files.items() if all((self.root / f).exists() for f in files)]


def spider_spec(root: str | Path, include_train_others: bool = True) -> DatasetSpec:
    root = Path(root)
    train = ["train_spider.json"] + (["train_others.json"] if include_train_others else [])
    spec = DatasetSpec(
        name="spider",
        root=root,
        split_files={"train": train, "dev": ["dev.json"], "test": ["test.json"]},
        split_tables={"test": "test_tables.json"},
        split_database_dirs={"test": "test_database"},
    )
    return spec


def synthetic_spec(root: str | Path) -> DatasetSpec:
    return DatasetSpec(name="synthetic", root=Path(root), split_files={"train": ["train.json"], "dev": ["dev.json"]})


def find_spider_root(base: str | Path) -> Path | None:
    """Find the directory containing Spider's tables.json below ``base``."""
    base = Path(base)
    for cand in [base, base / "spider", base / "spider_data", base / "spider" / "spider_data"]:
        if (cand / "tables.json").exists() and (cand / "dev.json").exists():
            return cand
    for p in base.rglob("tables.json"):
        if (p.parent / "dev.json").exists():
            return p.parent
    return None


def make_spec(cfg_data: dict) -> DatasetSpec:
    kind = cfg_data.get("dataset", "spider")
    root = Path(cfg_data["root"])
    if kind == "spider":
        found = find_spider_root(root) if not (root / "tables.json").exists() else root
        if found is None:
            raise FileNotFoundError(
                f"Spider not found under {root}. Run `python scripts/download_data.py` (see data/README.md)."
            )
        return spider_spec(found, cfg_data.get("include_train_others", True))
    if kind == "synthetic":
        return synthetic_spec(root)
    raise ValueError(f"Unknown dataset kind {kind!r}")


def load_split(spec: DatasetSpec, split: str) -> list[dict]:
    rows: list[dict] = []
    for f in spec.split_files[split]:
        data = read_json(spec.root / f)
        for ex in data:
            ex = dict(ex)
            ex.setdefault("source_file", f)
            rows.append(ex)
    return rows


def load_split_schemas(spec: DatasetSpec, split: str) -> dict[str, Schema]:
    return load_schemas(spec.tables_path(split))
