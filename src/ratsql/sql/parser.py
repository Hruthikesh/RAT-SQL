"""Spider-format SQL parser.

This is an independent re-implementation of the parsing semantics of Spider's
``process_sql.py`` (Yu et al., 2018), which the official Spider evaluation
uses to turn SQL strings into a nested structure::

    sql = {
      'select':  (is_distinct, [(agg_id, val_unit), ...]),
      'from':    {'table_units': [('table_unit', table_id) | ('sql', sql)], 'conds': [cond_unit, 'and', ...]},
      'where':   [cond_unit, 'and'|'or', cond_unit, ...],
      'groupBy': [col_unit, ...],
      'having':  [cond_unit, ...],
      'orderBy': ('asc'|'desc', [val_unit, ...]) | [],
      'limit':   int | None,
      'intersect'|'union'|'except': sql | None,
    }
    cond_unit = (not_op, op_id, val_unit, val1, val2)
    val_unit  = (unit_op, col_unit1, col_unit2)
    col_unit  = (agg_id, col_id, is_distinct)
    value     = float | '"string"' | sql | col_unit

Differences from the original, all intentional and documented:

* Tokenisation uses a regular expression instead of NLTK ``word_tokenize``
  (equivalent on SQL; negative numbers are merged like NLTK does).
* Both quote styles are accepted and escaped quotes ('') are supported.
* ``limit`` keeps its integer value (the original always returns 1 because of
  a type-check bug).  Exact-match only compares the *presence* of LIMIT, so
  this does not change evaluation, but it lets us execute parsed queries.
* Column / table identifiers are resolved against ``tables.json`` (lower-cased
  original names) instead of the sqlite catalogue.

The global alias map (aliases collected over the whole string) and the
"first FROM table containing the column" resolution rule of the original are
replicated because they influence which column id an unqualified name maps to.
"""

from __future__ import annotations

import re
from typing import Any

from ratsql.schema.schema import Schema
from ratsql.sql.constants import (
    AGG_OPS,
    CLAUSE_KEYWORDS,
    COND_OPS,
    JOIN_KEYWORDS,
    ORDER_OPS,
    SQL_OPS,
    TABLE_TYPE,
    UNIT_OPS,
    WHERE_OPS,
)


class SQLParseError(ValueError):
    pass


_SQL_TOKEN_RE = re.compile(
    r"""
      "(?:[^"]|"")*"            # double-quoted string
    | '(?:[^']|'')*'            # single-quoted string
    | `[^`]*`                   # back-quoted identifier
    | !=|>=|<=|<>
    | [A-Za-z_][\w$]*(?:\.(?:[A-Za-z_][\w$]*|\*))?   # identifier, optionally qualified
    | \d+(?:\.\d*)?(?:[eE][+-]?\d+)?                   # number
    | \.\d+
    | [(),;*=<>+\-/%.]
    """,
    re.VERBOSE,
)

_OPERATOR_PREV = {"=", ">", "<", ">=", "<=", "!=", "(", ",", "between", "and", "or", "in", "like", "select", "limit", "-", "+", "*", "/"}


def tokenize_sql(sql: str) -> list[str]:
    """Tokenise SQL: lower-case everything except string literals (re-quoted with ")."""
    sql = sql.strip()
    toks: list[str] = []
    pos = 0
    for m in _SQL_TOKEN_RE.finditer(sql):
        gap = sql[pos : m.start()]
        if gap.strip():
            raise SQLParseError(f"Unexpected characters {gap.strip()!r} in SQL: {sql}")
        pos = m.end()
        tok = m.group(0)
        if tok[0] in "\"'":
            quote = tok[0]
            inner = tok[1:-1].replace(quote * 2, quote)
            toks.append('"' + inner + '"')
        elif tok[0] == "`":
            toks.append(tok[1:-1].lower())
        else:
            toks.append(tok.lower())
    if sql[pos:].strip():
        raise SQLParseError(f"Unexpected trailing characters {sql[pos:]!r}")
    if toks and toks[-1] == ";":
        toks = toks[:-1]
    # merge '<>' into '!=' and negative numbers ("> -5")
    merged: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == "<>":
            tok = "!="
        if tok == "-" and i + 1 < len(toks) and _is_number(toks[i + 1]) and (not merged or merged[-1] in _OPERATOR_PREV):
            merged.append("-" + toks[i + 1])
            i += 2
            continue
        merged.append(tok)
        i += 1
    return merged


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


class _Parser:
    def __init__(self, schema: Schema, toks: list[str]):
        self.schema = schema
        self.toks = toks
        # table name -> list of lower-cased column names (the "schema" dict of the original)
        self.table_cols: dict[str, list[str]] = {}
        for t in schema.tables:
            self.table_cols[t.orig_name.lower()] = [schema.columns[c].orig_name.lower() for c in t.column_ids]
        self.aliases = self._tables_with_alias()

    # ---------------------------------------------------------------- aliases
    def _tables_with_alias(self) -> dict[str, str]:
        toks = self.toks
        alias: dict[str, str] = {}
        for idx, tok in enumerate(toks):
            if tok == "as" and 0 < idx < len(toks) - 1:
                alias[toks[idx + 1]] = toks[idx - 1]
        for key in self.table_cols:
            if key in alias:
                raise SQLParseError(f"Alias {key} has the same name as a table")
            alias[key] = key
        return alias

    def _err(self, msg: str, idx: int) -> SQLParseError:
        return SQLParseError(f"{msg} at token {idx}: {' '.join(self.toks)}")

    def _tok(self, idx: int) -> str:
        if idx >= len(self.toks):
            raise self._err("Unexpected end of SQL", idx)
        return self.toks[idx]

    # ---------------------------------------------------------------- columns
    def parse_col(self, toks: list[str], idx: int, default_tables: list[str] | None) -> tuple[int, int]:
        tok = toks[idx] if idx < len(toks) else None
        if tok is None:
            raise self._err("Expected column", idx)
        if tok == "*":
            return idx + 1, 0
        if "." in tok:
            alias, col = tok.split(".", 1)
            if alias not in self.aliases:
                raise self._err(f"Unknown alias {alias}", idx)
            table = self.aliases[alias]
            if col == "*":  # "T1.*" is treated as the global star column
                return idx + 1, 0
            cid = self.schema.column_id(table, col)
            if cid is None:
                raise self._err(f"Unknown column {table}.{col}", idx)
            return idx + 1, cid
        if not default_tables:
            raise self._err("Default tables should not be empty", idx)
        for alias in default_tables:
            table = self.aliases.get(alias, alias)
            if tok in self.table_cols.get(table, []):
                cid = self.schema.column_id(table, tok)
                assert cid is not None
                return idx + 1, cid
        raise self._err(f"Unknown column {tok}", idx)

    def parse_col_unit(self, toks: list[str], idx: int, default_tables) -> tuple[int, tuple]:
        n = len(toks)
        is_block = False
        is_distinct = False
        if idx < n and toks[idx] == "(":
            is_block = True
            idx += 1
        if idx < n and toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
            if not (idx < n and toks[idx] == "("):
                raise self._err("Expected '(' after aggregation", idx)
            idx += 1
            if idx < n and toks[idx] == "distinct":
                idx += 1
                is_distinct = True
            idx, col_id = self.parse_col(toks, idx, default_tables)
            if not (idx < n and toks[idx] == ")"):
                raise self._err("Expected ')' after aggregated column", idx)
            idx += 1
            return idx, (agg_id, col_id, is_distinct)
        if idx < n and toks[idx] == "distinct":
            idx += 1
            is_distinct = True
        agg_id = AGG_OPS.index("none")
        idx, col_id = self.parse_col(toks, idx, default_tables)
        if is_block:
            if not (idx < n and toks[idx] == ")"):
                raise self._err("Expected ')'", idx)
            idx += 1
        return idx, (agg_id, col_id, is_distinct)

    def parse_val_unit(self, toks: list[str], idx: int, default_tables) -> tuple[int, tuple]:
        n = len(toks)
        is_block = False
        if idx < n and toks[idx] == "(":
            is_block = True
            idx += 1
        col_unit2 = None
        unit_op = UNIT_OPS.index("none")
        idx, col_unit1 = self.parse_col_unit(toks, idx, default_tables)
        if idx < n and toks[idx] in UNIT_OPS:
            unit_op = UNIT_OPS.index(toks[idx])
            idx += 1
            idx, col_unit2 = self.parse_col_unit(toks, idx, default_tables)
        if is_block:
            if not (idx < n and toks[idx] == ")"):
                raise self._err("Expected ')' closing value unit", idx)
            idx += 1
        return idx, (unit_op, col_unit1, col_unit2)

    # ---------------------------------------------------------------- tables
    def parse_table_unit(self, idx: int) -> tuple[int, int, str]:
        toks = self.toks
        tok = self._tok(idx)
        if tok not in self.aliases:
            raise self._err(f"Unknown table {tok}", idx)
        key = self.aliases[tok]
        tid = self.schema.table_id(key)
        if tid is None:
            raise self._err(f"Unknown table {key}", idx)
        if idx + 1 < len(toks) and toks[idx + 1] == "as":
            idx += 3
        else:
            idx += 1
        return idx, tid, key

    # ---------------------------------------------------------------- values
    def parse_value(self, idx: int, default_tables) -> tuple[int, Any]:
        toks = self.toks
        n = len(toks)
        start_idx = idx
        is_block = False
        if self._tok(idx) == "(":
            is_block = True
            idx += 1
        tok = self._tok(idx)
        if tok == "select":
            idx, val = self.parse_sql(idx)
        elif '"' in tok:
            val = tok
            idx += 1
        else:
            try:
                val = float(tok)
                idx += 1
            except ValueError:
                end_idx = idx
                while (
                    end_idx < n
                    and toks[end_idx] not in (",", ")", "and")
                    and toks[end_idx] not in CLAUSE_KEYWORDS
                    and toks[end_idx] not in JOIN_KEYWORDS
                ):
                    end_idx += 1
                _, val = self.parse_col_unit(toks[start_idx:end_idx], 0, default_tables)
                idx = end_idx
        if is_block:
            if not (idx < n and toks[idx] == ")"):
                raise self._err("Expected ')' closing value", idx)
            idx += 1
        return idx, val

    def parse_condition(self, idx: int, default_tables) -> tuple[int, list]:
        toks = self.toks
        n = len(toks)
        conds: list = []
        while idx < n:
            idx, val_unit = self.parse_val_unit(toks, idx, default_tables)
            not_op = False
            if self._tok(idx) == "not":
                not_op = True
                idx += 1
            tok = self._tok(idx)
            if tok not in WHERE_OPS:
                raise self._err(f"Unknown condition operator {tok}", idx)
            op_id = WHERE_OPS.index(tok)
            idx += 1
            if op_id == WHERE_OPS.index("between"):
                idx, val1 = self.parse_value(idx, default_tables)
                if self._tok(idx) != "and":
                    raise self._err("Expected AND in BETWEEN", idx)
                idx += 1
                idx, val2 = self.parse_value(idx, default_tables)
            else:
                idx, val1 = self.parse_value(idx, default_tables)
                val2 = None
            conds.append((not_op, op_id, val_unit, val1, val2))
            if idx < n and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
                break
            if idx < n and toks[idx] in COND_OPS:
                conds.append(toks[idx])
                idx += 1
        return idx, conds

    # ---------------------------------------------------------------- clauses
    def parse_select(self, idx: int, default_tables) -> tuple[int, tuple]:
        toks = self.toks
        n = len(toks)
        if self._tok(idx) != "select":
            raise self._err("'select' not found", idx)
        idx += 1
        is_distinct = False
        if idx < n and toks[idx] == "distinct":
            idx += 1
            is_distinct = True
        val_units = []
        while idx < n and toks[idx] not in CLAUSE_KEYWORDS:
            agg_id = AGG_OPS.index("none")
            if toks[idx] in AGG_OPS:
                agg_id = AGG_OPS.index(toks[idx])
                idx += 1
            idx, val_unit = self.parse_val_unit(toks, idx, default_tables)
            val_units.append((agg_id, val_unit))
            if idx < n and toks[idx] == ",":
                idx += 1
        return idx, (is_distinct, val_units)

    def parse_from(self, start_idx: int) -> tuple[int, list, list, list[str]]:
        toks = self.toks
        n = len(toks)
        try:
            idx = toks.index("from", start_idx) + 1
        except ValueError as e:
            raise self._err("'from' not found", start_idx) from e
        default_tables: list[str] = []
        table_units: list = []
        conds: list = []
        while idx < n:
            is_block = False
            if toks[idx] == "(":
                is_block = True
                idx += 1
            if self._tok(idx) == "select":
                idx, sub = self.parse_sql(idx)
                table_units.append((TABLE_TYPE["sql"], sub))
            else:
                if idx < n and toks[idx] == "join":
                    idx += 1
                idx, tid, table_name = self.parse_table_unit(idx)
                table_units.append((TABLE_TYPE["table_unit"], tid))
                default_tables.append(table_name)
            if idx < n and toks[idx] == "on":
                idx += 1
                idx, this_conds = self.parse_condition(idx, default_tables)
                if conds:
                    conds.append("and")
                conds.extend(this_conds)
            if is_block:
                if not (idx < n and toks[idx] == ")"):
                    raise self._err("Expected ')' closing FROM block", idx)
                idx += 1
            if idx < n and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
                break
        return idx, table_units, conds, default_tables

    def parse_where(self, idx: int, default_tables) -> tuple[int, list]:
        if idx >= len(self.toks) or self.toks[idx] != "where":
            return idx, []
        return self.parse_condition(idx + 1, default_tables)

    def parse_group_by(self, idx: int, default_tables) -> tuple[int, list]:
        toks = self.toks
        n = len(toks)
        col_units: list = []
        if idx >= n or toks[idx] != "group":
            return idx, col_units
        idx += 1
        if self._tok(idx) != "by":
            raise self._err("Expected BY after GROUP", idx)
        idx += 1
        while idx < n and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, col_unit = self.parse_col_unit(toks, idx, default_tables)
            col_units.append(col_unit)
            if idx < n and toks[idx] == ",":
                idx += 1
            else:
                break
        return idx, col_units

    def parse_having(self, idx: int, default_tables) -> tuple[int, list]:
        if idx >= len(self.toks) or self.toks[idx] != "having":
            return idx, []
        return self.parse_condition(idx + 1, default_tables)

    def parse_order_by(self, idx: int, default_tables) -> tuple[int, Any]:
        toks = self.toks
        n = len(toks)
        val_units: list = []
        order_type = "asc"
        if idx >= n or toks[idx] != "order":
            return idx, val_units
        idx += 1
        if self._tok(idx) != "by":
            raise self._err("Expected BY after ORDER", idx)
        idx += 1
        while idx < n and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, val_unit = self.parse_val_unit(toks, idx, default_tables)
            val_units.append(val_unit)
            if idx < n and toks[idx] in ORDER_OPS:
                order_type = toks[idx]
                idx += 1
            if idx < n and toks[idx] == ",":
                idx += 1
            else:
                break
        return idx, (order_type, val_units)

    def parse_limit(self, idx: int) -> tuple[int, int | None]:
        toks = self.toks
        if idx < len(toks) and toks[idx] == "limit":
            val_tok = self._tok(idx + 1)
            try:
                value = int(float(val_tok))
            except ValueError:
                value = 1
            return idx + 2, value
        return idx, None

    def parse_sql(self, start_idx: int) -> tuple[int, dict]:
        toks = self.toks
        n = len(toks)
        idx = start_idx
        is_block = False
        sql: dict = {}
        if self._tok(idx) == "(":
            is_block = True
            idx += 1
        from_end_idx, table_units, conds, default_tables = self.parse_from(start_idx)
        sql["from"] = {"table_units": table_units, "conds": conds}
        _, select = self.parse_select(idx, default_tables)
        idx = from_end_idx
        sql["select"] = select
        idx, sql["where"] = self.parse_where(idx, default_tables)
        idx, sql["groupBy"] = self.parse_group_by(idx, default_tables)
        idx, sql["having"] = self.parse_having(idx, default_tables)
        idx, sql["orderBy"] = self.parse_order_by(idx, default_tables)
        idx, sql["limit"] = self.parse_limit(idx)
        while idx < n and toks[idx] == ";":
            idx += 1
        if is_block:
            if not (idx < n and toks[idx] == ")"):
                raise self._err("Expected ')' closing sub-query", idx)
            idx += 1
        while idx < n and toks[idx] == ";":
            idx += 1
        for op in SQL_OPS:
            sql[op] = None
        if idx < n and toks[idx] in SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, sql[sql_op] = self.parse_sql(idx)
        return idx, sql


def parse_sql(sql: str, schema: Schema, strict: bool = True) -> dict:
    """Parse a SQL string into the Spider nested structure.

    With ``strict=True`` any unconsumed trailing tokens raise ``SQLParseError``
    (the original silently ignores them; we are stricter for validation).
    """
    toks = tokenize_sql(sql)
    if not toks:
        raise SQLParseError("Empty SQL")
    parser = _Parser(schema, toks)
    try:
        idx, parsed = parser.parse_sql(0)
    except SQLParseError:
        raise
    except (IndexError, KeyError, AssertionError) as e:  # defensive: never leak internal errors
        raise SQLParseError(f"{type(e).__name__}: {e} in {sql}") from e
    if strict and idx < len(toks):
        raise SQLParseError(f"Unconsumed tokens {toks[idx:]} in: {sql}")
    return parsed


def to_tuples(obj: Any) -> Any:
    """Convert JSON-loaded Spider SQL (lists) into the canonical tuple form (dicts kept)."""
    if isinstance(obj, dict):
        return {k: to_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(to_tuples(v) for v in obj)
    return obj


def canonical(sql: dict) -> dict:
    """Canonical form used for comparisons: tuples everywhere except the clause lists.

    The clause containers ('where', 'having', 'groupBy', 'table_units', 'conds',
    select list) are lists in the parser output and tuples after ``to_tuples``;
    evaluation code only uses ``==`` / ``in`` on their *elements*, so we convert
    everything to tuples, then restore lists for the containers.
    """
    t = to_tuples(sql)
    return _restore_lists(t)


def _restore_lists(sql: Any) -> Any:
    if not isinstance(sql, dict):
        return sql
    out = dict(sql)
    out["from"] = {
        "table_units": [
            (tu[0], _restore_lists(tu[1]) if isinstance(tu[1], dict) else tu[1]) for tu in sql["from"]["table_units"]
        ],
        "conds": [_restore_cond(c) for c in sql["from"]["conds"]],
    }
    sel = sql["select"]
    out["select"] = (sel[0], list(sel[1]))
    out["where"] = [_restore_cond(c) for c in sql["where"]]
    out["having"] = [_restore_cond(c) for c in sql["having"]]
    out["groupBy"] = list(sql["groupBy"])
    ob = sql["orderBy"]
    out["orderBy"] = (ob[0], list(ob[1])) if ob else []
    for op in SQL_OPS:
        out[op] = _restore_lists(sql[op]) if sql.get(op) is not None else None
    return out


def _restore_cond(c: Any) -> Any:
    if isinstance(c, str):
        return c
    not_op, op_id, val_unit, v1, v2 = c
    v1 = _restore_lists(v1) if isinstance(v1, dict) else v1
    v2 = _restore_lists(v2) if isinstance(v2, dict) else v2
    return (not_op, op_id, val_unit, v1, v2)


def to_jsonable(sql: Any) -> Any:
    """Tuples -> lists for JSON storage."""
    if isinstance(sql, dict):
        return {k: to_jsonable(v) for k, v in sql.items()}
    if isinstance(sql, (list, tuple)):
        return [to_jsonable(v) for v in sql]
    return sql
