"""Schema-graph join inference (evaluation-time variant, not used in training).

RAT-SQL does not decode join conditions: its unparser derives the FROM clause
from the predicted tables *and* the tables of all predicted columns, and connects
them through shortest foreign-key paths.  Our decoder predicts join conditions
explicitly (they are part of the grammar), and error analysis showed that many
execution failures come from wrong ``ON`` conditions or from columns whose table
never entered FROM.  This module rewrites every query level of a *predicted*
Spider SQL dict:

1. required tables = predicted table units ∪ tables of every column referenced at
   this level (SELECT, WHERE, GROUP BY, HAVING, ORDER BY);
2. the required tables are connected greedily by shortest paths in the undirected
   foreign-key graph (intermediate tables are added when needed; an unreachable
   table is kept without a join condition);
3. the ``ON`` conditions are replaced by the foreign-key column equalities along
   the chosen paths.

Sub-queries are processed recursively; FROM sub-queries are left untouched.
Because FROM table units may change, this can change Exact Match as well as
Execution Accuracy.  It is reported as a separate, clearly labelled variant.
"""

from __future__ import annotations

import copy
from collections import deque

from ratsql.schema.schema import Schema
from ratsql.sql.constants import SQL_OPS


def _fk_graph(schema: Schema) -> dict[int, list[tuple[int, int, int]]]:
    """table -> [(neighbour table, column in this table, column in neighbour)]."""
    g: dict[int, list[tuple[int, int, int]]] = {t.id: [] for t in schema.tables}
    for src, dst in schema.foreign_keys:
        ts, td = schema.columns[src].table_id, schema.columns[dst].table_id
        if ts is None or td is None or ts == td:
            continue
        g[ts].append((td, src, dst))
        g[td].append((ts, dst, src))
    return g


def _shortest_path(g, sources: set[int], target: int):
    """BFS from any source table to target; returns list of (t_from, t_to, col_from, col_to)."""
    prev: dict[int, tuple[int, int, int] | None] = {s: None for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        if u == target:
            break
        for v, cu, cv in sorted(g[u]):
            if v not in prev:
                prev[v] = (u, cu, cv)
                q.append(v)
    if target not in prev:
        return None
    path = []
    cur = target
    while prev[cur] is not None:
        u, cu, cv = prev[cur]
        path.append((u, cur, cu, cv))
        cur = u
    return list(reversed(path))


def _level_columns(sql: dict) -> set[int]:
    cols: set[int] = set()

    def cu(c):
        if c is not None:
            cols.add(int(c[1]))

    def vu(v):
        cu(v[1])
        cu(v[2])

    for _, v in sql["select"][1]:
        vu(v)
    for conds in (sql["where"], sql["having"]):
        for c in conds[::2]:
            vu(c[2])
            for val in (c[3], c[4]):
                if isinstance(val, (list, tuple)) and len(val) == 3 and isinstance(val[1], int):
                    cu(val)
    for c in sql["groupBy"]:
        cu(c)
    if sql["orderBy"]:
        for v in sql["orderBy"][1]:
            vu(v)
    return cols


def infer_joins(sql: dict, schema: Schema) -> dict:
    sql = copy.deepcopy(sql)
    g = _fk_graph(schema)

    def rewrite(q: dict) -> dict:
        # recurse into condition sub-queries and set operations first
        for conds in (q["where"], q["having"]):
            for i in range(0, len(conds), 2):
                c = list(conds[i])
                for k in (3, 4):
                    if isinstance(c[k], dict):
                        c[k] = rewrite(c[k])
                conds[i] = tuple(c)
        for op in SQL_OPS:
            if q.get(op) is not None:
                q[op] = rewrite(q[op])
        units = q["from"]["table_units"]
        if any(u[0] == "sql" for u in units):
            return q  # leave FROM sub-queries untouched
        pred_tables = [int(u[1]) for u in units]
        if len(set(pred_tables)) < len(pred_tables):
            return q  # self-joins: the join condition cannot be inferred from the FK graph
        needed = list(dict.fromkeys(pred_tables + sorted({schema.columns[c].table_id for c in _level_columns(q) if schema.columns[c].table_id is not None})))
        if not needed:
            return q
        if len(needed) == 1:
            q["from"] = {"table_units": [("table_unit", needed[0])], "conds": []}
            return q
        connected = [needed[0]]
        conds: list = []
        for t in needed[1:]:
            if t in connected:
                continue
            path = _shortest_path(g, set(connected), t)
            if path is None:
                connected.append(t)  # unreachable: cross join, no condition
                continue
            for u, v, cu_, cv in path:
                if v not in connected:
                    connected.append(v)
                if conds:
                    conds.append("and")
                conds.append((False, 2, (0, (0, cu_, False), None), (0, cv, False), None))
        q["from"] = {"table_units": [("table_unit", t) for t in connected], "conds": conds}
        return q

    return rewrite(sql)
