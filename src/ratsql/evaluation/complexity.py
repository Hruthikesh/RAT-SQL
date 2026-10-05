"""Query-complexity features of gold SQL (used for complexity-based analysis).

All counts are over the *whole* query tree (nested sub-queries and set
operation branches included) unless noted.
"""

from __future__ import annotations

from typing import Any, Iterator

from ratsql.sql.constants import AGG_OPS


def _subqueries(sql: dict) -> Iterator[dict]:
    """Direct child queries (FROM sub-queries, condition sub-queries, set-op branches)."""
    for kind, val in sql["from"]["table_units"]:
        if kind == "sql":
            yield val
    for conds in (sql["from"]["conds"], sql["where"], sql["having"]):
        for cu in conds[::2]:
            for v in (cu[3], cu[4]):
                if isinstance(v, dict):
                    yield v
    for op in ("intersect", "union", "except"):
        if sql.get(op) is not None:
            yield sql[op]


def _all_queries(sql: dict) -> Iterator[dict]:
    yield sql
    for sub in _subqueries(sql):
        yield from _all_queries(sub)


def _depth(sql: dict) -> int:
    subs = list(_subqueries(sql))
    return 1 + (max(_depth(s) for s in subs) if subs else 0)


def _has_agg(sql: dict) -> bool:
    if any(agg != 0 for agg, _ in sql["select"][1]):
        return True
    units: list[Any] = []
    for _, vu in sql["select"][1]:
        units += [vu[1], vu[2]]
    for conds in (sql["where"], sql["having"]):
        for cu in conds[::2]:
            units += [cu[2][1], cu[2][2]]
    if sql["orderBy"]:
        for vu in sql["orderBy"][1]:
            units += [vu[1], vu[2]]
    return any(u is not None and u[0] != AGG_OPS.index("none") for u in units)


def complexity_features(sql: dict) -> dict:
    queries = list(_all_queries(sql))
    tables = set()
    joins = 0
    conditions = 0
    for q in queries:
        tus = [tu for tu in q["from"]["table_units"] if tu[0] == "table_unit"]
        tables.update(int(tu[1]) for tu in tus)
        joins += max(0, len(q["from"]["table_units"]) - 1)
        conditions += len(q["where"][::2]) + len(q["having"][::2])
    nested = any(
        isinstance(v, dict)
        for q in queries
        for conds in (q["from"]["conds"], q["where"], q["having"])
        for cu in conds[::2]
        for v in (cu[3], cu[4])
    ) or any(tu[0] == "sql" for q in queries for tu in q["from"]["table_units"])
    set_op = any(q.get(op) is not None for q in queries for op in ("intersect", "union", "except"))
    return {
        "num_tables": len(tables),
        "num_joins": joins,
        "num_conditions": conditions,
        "num_select": len(sql["select"][1]),
        "has_aggregation": any(_has_agg(q) for q in queries),
        "has_group_by": any(bool(q["groupBy"]) for q in queries),
        "has_having": any(bool(q["having"]) for q in queries),
        "has_order_by": any(bool(q["orderBy"]) for q in queries),
        "has_limit": any(q["limit"] is not None for q in queries),
        "has_nested": nested,
        "has_set_op": set_op,
        "depth": _depth(sql),
    }


def bucket(value: int, edges: tuple[int, ...]) -> str:
    """Bucket an integer: edges=(0,1,2) -> '0','1','2','3+'."""
    for e in edges:
        if value <= e:
            return str(e)
    return f"{edges[-1] + 1}+"
