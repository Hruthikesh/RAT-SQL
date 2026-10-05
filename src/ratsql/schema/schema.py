"""Database schema model built from Spider's ``tables.json`` format.

Spider schema entry::

    {
      "db_id": "concert_singer",
      "table_names": ["stadium", ...],             # natural-language names
      "table_names_original": ["stadium", ...],    # SQL identifiers
      "column_names": [[-1, "*"], [0, "stadium id"], ...],
      "column_names_original": [[-1, "*"], [0, "Stadium_ID"], ...],
      "column_types": ["text", "number", ...],
      "primary_keys": [1, 8, ...],                 # column indices (int or list for composite keys)
      "foreign_keys": [[src_col, dst_col], ...]    # src references dst
    }

Column index 0 is the special ``*`` column, which belongs to no table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ratsql.utils.io import read_json
from ratsql.utils.text import normalize_phrase


@dataclass
class Column:
    id: int
    name: str  # natural-language name, e.g. "singer id"
    orig_name: str  # SQL identifier, e.g. "Singer_ID"
    type: str  # text | number | time | boolean | others
    table_id: int | None  # None for "*"
    is_primary: bool = False
    fk_target: int | None = None  # the column this column references (if foreign key)
    name_tokens: list[str] = field(default_factory=list)  # normalised tokens used for linking

    @property
    def is_star(self) -> bool:
        return self.table_id is None


@dataclass
class Table:
    id: int
    name: str
    orig_name: str
    column_ids: list[int] = field(default_factory=list)
    primary_key_ids: list[int] = field(default_factory=list)
    name_tokens: list[str] = field(default_factory=list)


class Schema:
    """A relational database schema with primary / foreign key information."""

    def __init__(self, db_id: str, tables: list[Table], columns: list[Column], foreign_keys: list[tuple[int, int]], raw: dict | None = None):
        self.db_id = db_id
        self.tables = tables
        self.columns = columns
        self.foreign_keys = foreign_keys
        self.raw = raw or {}
        self._col_by_qualified = {
            (self.tables[c.table_id].orig_name.lower(), c.orig_name.lower()): c.id for c in columns if c.table_id is not None
        }
        self._table_by_name = {t.orig_name.lower(): t.id for t in tables}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_spider(cls, entry: dict) -> "Schema":
        tables = [
            Table(id=i, name=name, orig_name=orig, name_tokens=normalize_phrase(name))
            for i, (name, orig) in enumerate(zip(entry["table_names"], entry["table_names_original"]))
        ]
        pk_ids: set[int] = set()
        for pk in entry.get("primary_keys", []):
            if isinstance(pk, list):
                pk_ids.update(pk)
            else:
                pk_ids.add(pk)
        columns: list[Column] = []
        for i, ((tid, name), (_, orig), ctype) in enumerate(
            zip(entry["column_names"], entry["column_names_original"], entry["column_types"])
        ):
            table_id = None if tid < 0 else tid
            col = Column(
                id=i,
                name=name,
                orig_name=orig,
                type=ctype,
                table_id=table_id,
                is_primary=i in pk_ids,
                name_tokens=normalize_phrase(name) if table_id is not None else ["*"],
            )
            columns.append(col)
            if table_id is not None:
                tables[table_id].column_ids.append(i)
                if col.is_primary:
                    tables[table_id].primary_key_ids.append(i)
        fks = [(int(a), int(b)) for a, b in entry.get("foreign_keys", [])]
        for src, dst in fks:
            if columns[src].fk_target is None:
                columns[src].fk_target = dst
        return cls(entry["db_id"], tables, columns, fks, raw=entry)

    # ------------------------------------------------------------------ query
    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def num_tables(self) -> int:
        return len(self.tables)

    def column_id(self, table: str, column: str) -> int | None:
        return self._col_by_qualified.get((table.lower(), column.lower()))

    def table_id(self, table: str) -> int | None:
        return self._table_by_name.get(table.lower())

    def columns_of(self, table_id: int) -> list[Column]:
        return [self.columns[c] for c in self.tables[table_id].column_ids]

    def qualified_name(self, col_id: int) -> str:
        col = self.columns[col_id]
        if col.table_id is None:
            return "*"
        return f"{self.tables[col.table_id].orig_name}.{col.orig_name}"

    def fk_table_pairs(self) -> set[tuple[int, int]]:
        """Directed (src_table, dst_table) pairs induced by foreign keys."""
        pairs = set()
        for src, dst in self.foreign_keys:
            ts, td = self.columns[src].table_id, self.columns[dst].table_id
            if ts is not None and td is not None:
                pairs.add((ts, td))
        return pairs

    def to_spider(self) -> dict:
        return self.raw

    def summary(self) -> str:
        lines = [f"Database: {self.db_id}"]
        for t in self.tables:
            cols = []
            for c in self.columns_of(t.id):
                tag = []
                if c.is_primary:
                    tag.append("PK")
                if c.fk_target is not None:
                    tag.append(f"FK->{self.qualified_name(c.fk_target)}")
                cols.append(f"{c.orig_name}:{c.type}" + (f"[{','.join(tag)}]" if tag else ""))
            lines.append(f"  {t.orig_name}({', '.join(cols)})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Schema(db_id={self.db_id!r}, tables={self.num_tables}, columns={self.num_columns}, fks={len(self.foreign_keys)})"


def load_schemas(tables_json: str | Path | Iterable[dict]) -> dict[str, Schema]:
    """Load ``tables.json`` (path or already-parsed list) into ``{db_id: Schema}``."""
    entries = read_json(tables_json) if isinstance(tables_json, (str, Path)) else list(tables_json)
    return {e["db_id"]: Schema.from_spider(e) for e in entries}
