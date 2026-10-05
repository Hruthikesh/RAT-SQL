"""Schema linking: n-gram name matching and database value matching.

Follows the schema-linking procedure of RAT-SQL (Sec. 3.2 of the paper):

Name-based linking (n-gram matching)
    For n = max_n .. 1 and every question n-gram (normalised: lower-case,
    accent-stripped, singularised; n-grams consisting only of stop words are
    skipped), compare against every column / table name:

    * ``EM`` (exact match)   the n-gram equals the full name;
    * ``PM`` (partial match) the n-gram is a contiguous sub-sequence of the name.

    Every question token inside a matched n-gram is linked to the element.
    Exact matches override partial matches.

Value-based linking
    * ``VEM`` a question n-gram equals a (normalised) cell value of a column;
    * ``VPM`` a single content word of the question occurs as a word inside a
      short cell value of a column (e.g. "Hobbit" in "The Hobbit");
    * ``NUM`` a numeric question token and a number/time typed column
      (RAT-SQL's NUMBER/TIME relation).

For each (question token, column) pair only the strongest link is kept, with
precedence EM > VEM > PM > VPM > NUM (name evidence first, as in RAT-SQL,
except that an exact cell-value hit outranks a partial name hit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ratsql.schema.db_values import DatabaseValueIndex
from ratsql.schema.schema import Schema
from ratsql.utils.text import is_number_token, is_stopword, normalize_token

LINK_PRIORITY = {"EM": 5, "VEM": 4, "PM": 3, "VPM": 2, "NUM": 1}
COLUMN_LINK_TYPES = ("EM", "PM", "VEM", "VPM", "NUM")
TABLE_LINK_TYPES = ("EM", "PM")


@dataclass
class ValueMatch:
    start: int  # question token span [start, end)
    end: int
    column: int
    value: str  # original database value
    kind: str  # VEM | VPM


@dataclass
class SchemaLinks:
    q_col: dict[tuple[int, int], str] = field(default_factory=dict)
    q_tab: dict[tuple[int, int], str] = field(default_factory=dict)
    value_matches: list[ValueMatch] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "q_col": [[q, c, t] for (q, c), t in sorted(self.q_col.items())],
            "q_tab": [[q, c, t] for (q, c), t in sorted(self.q_tab.items())],
            "value_matches": [[m.start, m.end, m.column, m.value, m.kind] for m in self.value_matches],
        }

    @classmethod
    def from_json(cls, obj: dict) -> "SchemaLinks":
        return cls(
            q_col={(q, c): t for q, c, t in obj["q_col"]},
            q_tab={(q, c): t for q, c, t in obj["q_tab"]},
            value_matches=[ValueMatch(*m) for m in obj.get("value_matches", [])],
        )

    def filtered(self, ngram: bool = True, value: bool = True, numbers: bool = True) -> "SchemaLinks":
        """Drop link types (used by the linking ablations)."""
        keep_col = set()
        if ngram:
            keep_col |= {"EM", "PM"}
        if value:
            keep_col |= {"VEM", "VPM"}
        if numbers:
            keep_col.add("NUM")
        return SchemaLinks(
            q_col={k: v for k, v in self.q_col.items() if v in keep_col},
            q_tab=dict(self.q_tab) if ngram else {},
            value_matches=list(self.value_matches),
        )


def _is_contiguous_sub(gram: tuple[str, ...], name: tuple[str, ...]) -> bool:
    n = len(gram)
    return any(name[i : i + n] == gram for i in range(len(name) - n + 1))


class SchemaLinker:
    def __init__(self, max_n: int = 5, max_vpm_per_token: int = 3, use_values: bool = True, use_numbers: bool = True):
        self.max_n = max_n
        self.max_vpm_per_token = max_vpm_per_token
        self.use_values = use_values
        self.use_numbers = use_numbers

    def link(self, question_tokens: list[str], schema: Schema, value_index: DatabaseValueIndex | None = None) -> SchemaLinks:
        norm = [normalize_token(t) for t in question_tokens]
        n_q = len(norm)
        links = SchemaLinks()

        def set_col(q: int, c: int, kind: str) -> None:
            old = links.q_col.get((q, c))
            if old is None or LINK_PRIORITY[kind] > LINK_PRIORITY[old]:
                links.q_col[(q, c)] = kind

        def set_tab(q: int, t: int, kind: str) -> None:
            old = links.q_tab.get((q, t))
            if old is None or LINK_PRIORITY[kind] > LINK_PRIORITY[old]:
                links.q_tab[(q, t)] = kind

        col_names = [(c.id, tuple(c.name_tokens)) for c in schema.columns if c.table_id is not None and c.name_tokens]
        tab_names = [(t.id, tuple(t.name_tokens)) for t in schema.tables if t.name_tokens]

        # ---------------------------------------------------- n-gram matching
        for n in range(min(self.max_n, n_q), 0, -1):
            for i in range(n_q - n + 1):
                gram = tuple(norm[i : i + n])
                if all(is_stopword(w) for w in gram) or any(not w for w in gram):
                    continue
                for cid, name in col_names:
                    if gram == name:
                        kind = "EM"
                    elif len(gram) < len(name) and _is_contiguous_sub(gram, name):
                        kind = "PM"
                    else:
                        continue
                    for q in range(i, i + n):
                        if not is_stopword(norm[q]):
                            set_col(q, cid, kind)
                for tid, name in tab_names:
                    if gram == name:
                        kind = "EM"
                    elif len(gram) < len(name) and _is_contiguous_sub(gram, name):
                        kind = "PM"
                    else:
                        continue
                    for q in range(i, i + n):
                        if not is_stopword(norm[q]):
                            set_tab(q, tid, kind)

        # ---------------------------------------------------- value matching
        if self.use_values and value_index is not None:
            covered: set[tuple[int, int]] = set()  # (token, column) with an exact value hit
            seen_matches: set[tuple[int, int, int, str]] = set()
            for n in range(min(self.max_n, n_q), 0, -1):
                for i in range(n_q - n + 1):
                    gram = tuple(norm[i : i + n])
                    if all(is_stopword(w) for w in gram) or any(not w for w in gram):
                        continue
                    for cid, value in value_index.lookup_exact(gram):
                        for q in range(i, i + n):
                            set_col(q, cid, "VEM")
                            covered.add((q, cid))
                        key = (i, i + n, cid, value)
                        if key not in seen_matches:
                            seen_matches.add(key)
                            links.value_matches.append(ValueMatch(i, i + n, cid, value, "VEM"))
            for q, w in enumerate(norm):
                if is_stopword(w) or len(w) < 3 or is_number_token(w):
                    continue
                hits = value_index.lookup_word(w)
                # prefer short values (more specific) and deterministic order
                hits = sorted(hits, key=lambda h: (len(h[1]), h[0], h[1]))
                used = 0
                for cid, value in hits:
                    if (q, cid) in covered:
                        continue
                    set_col(q, cid, "VPM")
                    key = (q, q + 1, cid, value)
                    if key not in seen_matches:
                        seen_matches.add(key)
                        links.value_matches.append(ValueMatch(q, q + 1, cid, value, "VPM"))
                    used += 1
                    if used >= self.max_vpm_per_token:
                        break

        # ---------------------------------------------------- number / time
        if self.use_numbers:
            num_cols = [c.id for c in schema.columns if c.table_id is not None and c.type in ("number", "time")]
            for q, tok in enumerate(question_tokens):
                if is_number_token(tok):
                    for cid in num_cols:
                        set_col(q, cid, "NUM")
        return links


def describe_links(question_tokens: list[str], schema: Schema, links: SchemaLinks) -> list[str]:
    """Human-readable 'question phrase -> linked element' lines (debug output)."""
    lines = []

    def spans(pairs: dict[tuple[int, int], str]) -> list[tuple[int, int, int, str]]:
        """Group linked tokens into contiguous spans: (start, end, element, kind)."""
        by_elem: dict[tuple[int, str], list[int]] = {}
        for (q, e), kind in sorted(pairs.items()):
            by_elem.setdefault((e, kind), []).append(q)
        out = []
        for (e, kind), qs in by_elem.items():
            start = prev = qs[0]
            for q in qs[1:] + [None]:
                if q is not None and q == prev + 1:
                    prev = q
                    continue
                out.append((start, prev + 1, e, kind))
                if q is not None:
                    start = prev = q
        return sorted(out)

    for s, e, tid, kind in spans(links.q_tab):
        lines.append(f"'{' '.join(question_tokens[s:e])}' -> TABLE {schema.tables[tid].orig_name} [{kind}]")
    for s, e, cid, kind in spans(links.q_col):
        if kind != "NUM":
            lines.append(f"'{' '.join(question_tokens[s:e])}' -> COLUMN {schema.qualified_name(cid)} [{kind}]")
    seen = set()
    for m in links.value_matches:
        key = (" ".join(question_tokens[m.start : m.end]).lower(), m.value, m.column)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"'{' '.join(question_tokens[m.start : m.end])}' -> VALUE {m.value!r} in {schema.qualified_name(m.column)} [{m.kind}]")
    return lines
