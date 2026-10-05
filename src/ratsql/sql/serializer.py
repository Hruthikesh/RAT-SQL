"""Serialise Spider SQL dicts into executable SQLite SQL.

Conventions (chosen so that the output re-parses with Spider's parser and
executes on SQLite):

* single-table queries use bare column names; multi-table FROM clauses use
  ``AS T<k>`` aliases that are unique across the *whole* query (the Spider
  parser keeps one global alias map, so re-using T1 in a sub-query would be
  ambiguous);
* all join conditions are emitted after the last JOIN (``a AS T1 JOIN b AS T2
  JOIN c AS T3 ON c1 AND c2``), which is equivalent for inner joins;
* ORDER BY repeats the (single) Spider direction after every key so that the
  SQL semantics matches the structure;
* string literals use single quotes with '' escaping.
* a column whose table is not in FROM is qualified with its real table name;
  such a query fails at execution (it is counted as invalid SQL, not repaired).
"""

from __future__ import annotations

from typing import Any

from ratsql.schema.schema import Schema
from ratsql.sql.constants import AGG_OPS, SQL_OPS, UNIT_OPS, WHERE_OPS
from ratsql.utils.text import format_number


class SQLSerializer:
    def __init__(self, schema: Schema):
        self.schema = schema
        self._alias_counter = 0

    def serialize(self, sql: dict) -> str:
        self._alias_counter = 0
        return self._sql(sql)

    # ----------------------------------------------------------------- query
    def _sql(self, sql: dict) -> str:
        units = sql["from"]["table_units"]
        use_alias = len(units) > 1
        alias_of: dict[int, str | None] = {}
        from_parts: list[str] = []
        for kind, val in units:
            if kind == "table_unit":
                tname = self.schema.tables[int(val)].orig_name
                if use_alias:
                    self._alias_counter += 1
                    alias = f"T{self._alias_counter}"
                    alias_of.setdefault(int(val), alias)
                    from_parts.append(f"{tname} AS {alias}")
                else:
                    alias_of.setdefault(int(val), None)
                    from_parts.append(tname)
            else:
                from_parts.append(f"({self._sql(val)})")
        ctx = alias_of
        select_distinct, select_items = sql["select"]
        sel = ", ".join(self._select_item(agg, vu, ctx) for agg, vu in select_items)
        out = f"SELECT {'DISTINCT ' if select_distinct else ''}{sel} FROM {' JOIN '.join(from_parts)}"
        if sql["from"]["conds"]:
            out += " ON " + self._conds(sql["from"]["conds"], ctx)
        if sql["where"]:
            out += " WHERE " + self._conds(sql["where"], ctx)
        if sql["groupBy"]:
            out += " GROUP BY " + ", ".join(self._col_unit(cu, ctx) for cu in sql["groupBy"])
        if sql["having"]:
            out += " HAVING " + self._conds(sql["having"], ctx)
        if sql["orderBy"]:
            direction, keys = sql["orderBy"]
            d = "DESC" if direction == "desc" else "ASC"
            out += " ORDER BY " + ", ".join(f"{self._val_unit(vu, ctx)} {d}" for vu in keys)
        if sql["limit"] is not None:
            out += f" LIMIT {int(sql['limit'])}"
        for op in SQL_OPS:
            if sql.get(op) is not None:
                out += f" {op.upper()} " + self._sql(sql[op])
                break
        return out

    # ----------------------------------------------------------------- units
    def _column(self, col_id: int, ctx: dict[int, str | None]) -> str:
        col = self.schema.columns[int(col_id)]
        if col.table_id is None:
            return "*"
        if col.table_id in ctx:
            alias = ctx[col.table_id]
            return f"{alias}.{col.orig_name}" if alias else col.orig_name
        return f"{self.schema.tables[col.table_id].orig_name}.{col.orig_name}"

    def _col_unit(self, cu: Any, ctx) -> str:
        agg, col_id, distinct = cu
        col = self._column(col_id, ctx)
        inner = f"DISTINCT {col}" if distinct else col
        if agg:
            return f"{AGG_OPS[agg]}({inner})"
        return inner

    def _val_unit(self, vu: Any, ctx) -> str:
        op, cu1, cu2 = vu
        if op == 0 or cu2 is None:
            return self._col_unit(cu1, ctx)
        return f"{self._col_unit(cu1, ctx)} {UNIT_OPS[op]} {self._col_unit(cu2, ctx)}"

    def _select_item(self, agg: int, vu: Any, ctx) -> str:
        inner = self._val_unit(vu, ctx)
        return f"{AGG_OPS[agg]}({inner})" if agg else inner

    def _value(self, v: Any, ctx) -> str:
        if isinstance(v, dict):
            return f"({self._sql(v)})"
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return self._col_unit(v, ctx)
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return format_number(float(v))
        s = str(v)
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        return "'" + s.replace("'", "''") + "'"

    def _cond_unit(self, cu: Any, ctx) -> str:
        not_op, op_id, vu, v1, v2 = cu
        left = self._val_unit(vu, ctx)
        neg = "NOT " if not_op else ""
        op = WHERE_OPS[op_id]
        if op == "between":
            return f"{left} {neg}BETWEEN {self._value(v1, ctx)} AND {self._value(v2, ctx)}"
        right = self._value(v1, ctx)
        if op == "in" and not right.startswith("("):
            right = f"({right})"
        return f"{left} {neg}{op.upper()} {right}"

    def _conds(self, conds: list, ctx) -> str:
        parts = []
        for c in conds:
            parts.append(c.upper() if isinstance(c, str) else self._cond_unit(c, ctx))
        return " ".join(parts)


def serialize_sql(sql: dict, schema: Schema) -> str:
    return SQLSerializer(schema).serialize(sql)
