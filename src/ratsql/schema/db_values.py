"""Database cell-value index used for value-based schema linking.

For every (non-``*``) column we read up to ``max_values_per_column`` distinct
values from SQLite and build two inverted indices keyed by *normalised*
tokens (lower-cased, accent-stripped, singularised -- the same normalisation
as question tokens):

* ``exact``: full normalised value (token tuple) -> [(column_id, original value)]
* ``words``: single normalised word of a short text value -> [(column_id, original value)]

Indices are cached as JSON under ``data/processed/value_index`` so that
preprocessing is deterministic and fast on re-runs.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from ratsql.schema.schema import Schema
from ratsql.utils.text import format_number, is_stopword, normalize_phrase


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class DatabaseValueIndex:
    def __init__(self, db_id: str, exact: dict[tuple[str, ...], list[tuple[int, str]]], words: dict[str, list[tuple[int, str]]], stats: dict | None = None):
        self.db_id = db_id
        self.exact = exact
        self.words = words
        self.stats = stats or {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        schema: Schema,
        db_path: str | Path,
        max_values_per_column: int = 5000,
        max_value_chars: int = 80,
        max_words_per_value: int = 6,
    ) -> "DatabaseValueIndex":
        exact: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
        words: dict[str, list[tuple[int, str]]] = defaultdict(list)
        stats = {"columns": 0, "values": 0, "errors": 0}
        db_path = Path(db_path)
        if not db_path.exists():
            return cls(schema.db_id, {}, {}, {"missing_db": True})
        conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        try:
            for col in schema.columns:
                if col.table_id is None:
                    continue
                table = schema.tables[col.table_id].orig_name
                q = f"SELECT DISTINCT {_quote_ident(col.orig_name)} FROM {_quote_ident(table)} LIMIT {int(max_values_per_column)}"
                try:
                    rows = conn.execute(q).fetchall()
                except sqlite3.Error:
                    stats["errors"] += 1
                    continue
                stats["columns"] += 1
                seen: set[tuple[str, ...]] = set()
                for (v,) in rows:
                    if v is None:
                        continue
                    if isinstance(v, bytes):
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        text = format_number(float(v)) if float(v) == float(v) else str(v)
                    else:
                        text = str(v).strip()
                    if not text or len(text) > max_value_chars:
                        continue
                    key = tuple(normalize_phrase(text))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    stats["values"] += 1
                    exact[key].append((col.id, text))
                    if len(key) <= max_words_per_value and not isinstance(v, (int, float)):
                        for w in set(key):
                            if not is_stopword(w) and len(w) >= 3 and not w.isdigit():
                                words[w].append((col.id, text))
        finally:
            conn.close()
        return cls(schema.db_id, dict(exact), dict(words), stats)

    # ------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return {
            "db_id": self.db_id,
            "exact": [[list(k), v] for k, v in self.exact.items()],
            "words": self.words,
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "DatabaseValueIndex":
        exact = {tuple(k): [tuple(x) for x in v] for k, v in obj["exact"]}
        words = {k: [tuple(x) for x in v] for k, v in obj["words"].items()}
        return cls(obj["db_id"], exact, words, obj.get("stats"))

    @classmethod
    def load_or_build(cls, schema: Schema, db_path: str | Path, cache_dir: str | Path | None, **kwargs) -> "DatabaseValueIndex":
        if cache_dir is not None:
            cache = Path(cache_dir) / f"{schema.db_id}.json"
            if cache.exists():
                with open(cache, "r", encoding="utf-8") as f:
                    return cls.from_json(json.load(f))
        idx = cls.build(schema, db_path, **kwargs)
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(cache_dir) / f"{schema.db_id}.json", "w", encoding="utf-8") as f:
                json.dump(idx.to_json(), f, ensure_ascii=False)
        return idx

    def lookup_exact(self, norm_tokens: tuple[str, ...]) -> list[tuple[int, str]]:
        return self.exact.get(tuple(norm_tokens), [])

    def lookup_word(self, norm_word: str) -> list[tuple[int, str]]:
        return self.words.get(norm_word, [])
