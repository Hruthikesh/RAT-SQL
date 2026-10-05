"""Clause-level accuracy: SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY, nested, aggregation.

Complements the official Spider components (which do not isolate JOIN, HAVING or nested
sub-queries).  Both queries are first normalised exactly as for Exact Match (values removed,
DISTINCT dropped, foreign-key-equivalent columns unified).  For every clause we report
accuracy over the examples whose *gold* query contains the clause:

* SELECT       multiset of (aggregation, value unit) of the top-level SELECT
* WHERE        multiset of condition units and set of AND/OR connectors (top level)
* JOIN         (multi-table gold only) FROM table multiset and the set of join column pairs
* GROUP BY     multiset of grouping columns
* HAVING       multiset of HAVING condition units
* ORDER BY     direction, key list and LIMIT presence
* NESTED       (gold with sub-query or set operation) nested queries match exactly
* AGGREGATION  (gold with any aggregation) multiset of (aggregation, column) pairs over the query
"""

from __future__ import annotations

from collections import Counter

from ratsql.analysis.errors import _aggs
from ratsql.evaluation.exact_match import ExactMatchEvaluator, normalize_for_em
from ratsql.schema.schema import Schema
from ratsql.sql.constants import empty_sql

CLAUSES = ("SELECT", "WHERE", "JOIN", "GROUP BY", "HAVING", "ORDER BY", "NESTED", "AGGREGATION")


def _bag(units) -> Counter:
    """Multiset of condition units; sub-query values are dicts, so count by a stable repr."""
    return Counter(repr(u) for u in units)


def _join_sig(sql: dict) -> tuple:
    tabs = Counter(tu[1] if tu[0] == "table_unit" else "sql" for tu in sql["from"]["table_units"])
    pairs = Counter()
    for cu in sql["from"]["conds"][::2]:
        v = cu[3]
        if isinstance(v, (list, tuple)) and len(v) == 3:
            pairs[frozenset((cu[2][1][1], v[1]))] += 1
    return tabs, pairs


def _nested(sql: dict) -> list:
    out = []
    for conds in (sql["where"], sql["having"], sql["from"]["conds"]):
        for cu in conds[::2]:
            for v in (cu[3], cu[4]):
                if isinstance(v, dict):
                    out.append(v)
    out += [tu[1] for tu in sql["from"]["table_units"] if tu[0] == "sql"]
    out += [sql[op] for op in ("intersect", "union", "except") if sql.get(op)]
    return out


def clause_correctness(gold: dict, pred: dict | None, schema: Schema) -> dict[str, bool | None]:
    """None = clause absent from gold (not scored)."""
    em = ExactMatchEvaluator(schema)
    g = normalize_for_em(gold, schema, em.kmap)
    p = normalize_for_em(pred if pred is not None else empty_sql(), schema, em.kmap)
    res: dict[str, bool | None] = {}
    res["SELECT"] = Counter(g["select"][1]) == Counter(p["select"][1])
    res["WHERE"] = (_bag(g["where"][::2]) == _bag(p["where"][::2]) and set(g["where"][1::2]) == set(p["where"][1::2])) if g["where"] else None
    multi = len(g["from"]["table_units"]) > 1
    res["JOIN"] = (_join_sig(g) == _join_sig(p)) if multi else None
    res["GROUP BY"] = (Counter(c[1] for c in g["groupBy"]) == Counter(c[1] for c in p["groupBy"])) if g["groupBy"] else None
    res["HAVING"] = (_bag(g["having"][::2]) == _bag(p["having"][::2])) if g["having"] else None
    res["ORDER BY"] = (g["orderBy"] == p["orderBy"] and (g["limit"] is None) == (p["limit"] is None)) if g["orderBy"] else None
    gn, pn = _nested(gold), _nested(pred if pred is not None else empty_sql())
    if gn:
        ok = len(gn) == len(pn) and all(em.evaluate(b, a)["exact"] for a, b in zip(gn, pn))
        ok = ok and all((gold.get(op) is None) == ((pred or {}).get(op) is None) for op in ("intersect", "union", "except"))
        res["NESTED"] = ok
    else:
        res["NESTED"] = None
    ga = _aggs(gold)
    has_agg = any(a != 0 for a, _ in ga)
    res["AGGREGATION"] = (_aggs(gold) == _aggs(pred) if pred is not None else False) if has_agg else None
    return res


def clause_accuracy(rows: list[dict], golds: dict, schemas: dict) -> dict[str, dict]:
    """rows: prediction rows (with pred_parsed); golds: id -> canonical gold dict."""
    from ratsql.sql.parser import canonical, to_tuples

    tot: dict[str, list[bool]] = {c: [] for c in CLAUSES}
    for r in rows:
        pred = canonical(to_tuples(r["pred_parsed"])) if r.get("pred_parsed") else None
        res = clause_correctness(golds[r["id"]], pred, schemas[r["db_id"]])
        for c, v in res.items():
            if v is not None:
                tot[c].append(bool(v))
    return {c: {"n": len(v), "accuracy": (sum(v) / len(v)) if v else None} for c, v in tot.items()}
