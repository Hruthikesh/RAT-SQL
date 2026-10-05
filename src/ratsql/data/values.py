"""Value candidates: the pointer vocabulary for SQL literals.

The original RAT-SQL does not predict literal values (it emits a placeholder
terminal), which is sufficient for Exact Match but makes queries
non-executable.  To measure Execution Accuracy we add a *value pointer*: the
decoder selects literals from a per-example candidate list built
deterministically from the question and the database (no gold information):

0. ``<unk>``          fallback (used when the gold value is not recoverable)
1. database values    cell values found by value linking (VEM before VPM),
                      in their exact database spelling / casing
2. quoted spans       text between quotes in the question
3. numbers            numeric tokens and number words ("three" -> 3)
4. question n-grams   all n-grams up to ``max_ngram`` that do not start/end
                      with punctuation and are not only stop words

Candidates are de-duplicated by surface text (first occurrence wins).  The
fraction of gold literals that can be resolved to a non-``<unk>`` candidate
is an upper bound on value correctness and is reported by preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ratsql.schema.linking import SchemaLinks
from ratsql.utils.text import Token, format_number, is_stopword, normalize_token, parse_number

QUOTE_CHARS = {'"', "'", "“", "”", "‘", "’", "`"}
SOURCES = ("unk", "db", "quote", "number", "ngram")


@dataclass
class ValueCandidate:
    text: str
    source: str
    start: int  # question token span [start, end)
    end: int
    column: int = -1  # database column for 'db' candidates
    is_number: bool = False

    def to_json(self) -> list:
        return [self.text, self.source, self.start, self.end, self.column, self.is_number]

    @classmethod
    def from_json(cls, obj: list) -> "ValueCandidate":
        return cls(*obj)


def build_value_candidates(
    question: str,
    tokens: list[Token],
    links: SchemaLinks | None,
    max_ngram: int = 4,
    max_db_values: int = 30,
    max_candidates: int = 128,
) -> list[ValueCandidate]:
    cands: list[ValueCandidate] = [ValueCandidate("<unk>", "unk", 0, 0)]
    seen: set[str] = set()

    def add(c: ValueCandidate) -> None:
        if len(cands) >= max_candidates:
            return
        key = c.text
        if not key or key in seen:
            return
        seen.add(key)
        cands.append(c)

    words = [t.text for t in tokens]
    norm = [normalize_token(w) for w in words]
    # 1. database values from value linking
    if links is not None:
        vms = sorted(links.value_matches, key=lambda m: (0 if m.kind == "VEM" else 1, m.start, -(m.end - m.start), m.column))
        for m in vms[:max_db_values]:
            add(ValueCandidate(m.value, "db", m.start, m.end, m.column, parse_number(m.value) is not None and _looks_numeric(m.value)))
    # 2. quoted spans
    open_idx = None
    for i, w in enumerate(words):
        if w in QUOTE_CHARS:
            if open_idx is None:
                open_idx = i
            else:
                if i > open_idx + 1:
                    text = question[tokens[open_idx + 1].start : tokens[i - 1].end]
                    add(ValueCandidate(text, "quote", open_idx + 1, i))
                open_idx = None
    # 3. numbers
    for i, w in enumerate(words):
        num = parse_number(w)
        if num is not None and (_looks_numeric(w) or w.lower() in _SMALL_NUMBER_WORDS):
            add(ValueCandidate(format_number(num), "number", i, i + 1, is_number=True))
    # 4. n-grams
    n_q = len(tokens)
    for n in range(1, max_ngram + 1):
        for i in range(n_q - n + 1):
            span = words[i : i + n]
            if not span[0][0].isalnum() or not span[-1][-1].isalnum():
                continue
            if all(is_stopword(x) for x in norm[i : i + n]):
                continue
            text = question[tokens[i].start : tokens[i + n - 1].end]
            add(ValueCandidate(text, "ngram", i, i + n, is_number=_looks_numeric(text)))
    return cands


_SMALL_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty"}


def _looks_numeric(text: str) -> bool:
    t = text.replace(",", "").strip()
    try:
        float(t)
        return True
    except ValueError:
        return False


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


class ValueResolver:
    """Maps gold literals to candidate indices (gold side) and back (prediction side)."""

    def __init__(self, candidates: list[ValueCandidate]):
        self.cands = candidates
        self.resolved = 0
        self.unresolved = 0
        self.unresolved_values: list[str] = []

    # gold literal -> candidate index
    def __call__(self, raw: Any, context: str) -> int | None:
        idx = self._find(raw, context)
        if idx is None:
            if context == "limit":
                return None
            self.unresolved += 1
            self.unresolved_values.append(str(raw))
            return 0
        self.resolved += 1
        return idx

    def _find(self, raw: Any, context: str) -> int | None:
        cands = self.cands
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            target = float(raw)
            for i, c in enumerate(cands[1:], 1):
                if c.source in ("number", "db", "ngram") and _looks_numeric(c.text):
                    try:
                        if float(c.text.replace(",", "")) == target:
                            return i
                    except ValueError:
                        continue
            return None
        text = _strip_quotes(str(raw))
        if context == "like":
            text = text.strip("%")
        if not text:
            return None
        for i, c in enumerate(cands[1:], 1):  # exact, case-sensitive (db values first by construction)
            if c.text == text:
                return i
        low = " ".join(text.lower().split())
        for i, c in enumerate(cands[1:], 1):
            if " ".join(c.text.lower().split()) == low:
                return i
        if _looks_numeric(text):
            return self._find(float(text.replace(",", "")), "number")
        return None

    # candidate index -> literal for the Spider SQL dict
    def lookup(self, ref: int, context: str) -> Any:
        if ref < 0 or ref >= len(self.cands):
            return '"value"'
        c = self.cands[ref]
        if c.source == "unk":
            return 1.0 if context == "limit" else '"value"'
        text = c.text
        if context == "like":
            if "%" not in text:
                text = f"%{text}%"
            return f'"{text}"'
        if context == "limit" or (c.is_number and _looks_numeric(text)):
            try:
                return float(text.replace(",", ""))
            except ValueError:
                pass
        return f'"{text}"'


def candidates_to_json(cands: list[ValueCandidate]) -> list[list]:
    return [c.to_json() for c in cands]


def candidates_from_json(obj: list[list]) -> list[ValueCandidate]:
    return [ValueCandidate.from_json(x) for x in obj]


__all__ = ["ValueCandidate", "ValueResolver", "build_value_candidates", "candidates_to_json", "candidates_from_json", "SOURCES", "asdict"]
