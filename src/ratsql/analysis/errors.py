"""Systematic error taxonomy for Text-to-SQL predictions.

Every prediction that is not both an exact match and an execution match gets
one or more error labels (and a primary label chosen by the priority below).
Labels are computed structurally from the parsed gold / predicted SQL and the
schema links of the question:

1. ``grammar_error``            decoding produced no complete, well-formed AST
2. ``nested_query_error``       sub-query / set-operation structure differs
3. ``schema_linking_error``     a wrong schema element where the linker offered
                                *strong* evidence (exact name match or exact cell-value
                                match): a *missed* gold column/table that the question
                                names exactly, or an *extra* predicted element that the
                                question names exactly although gold does not use it
                                ("misleading link")
4. ``wrong_table``              the set of tables differs (no linking evidence)
5. ``wrong_join``               same tables, different join structure / conditions
6. ``wrong_column``             same tables, different columns (no linking evidence)
7. ``aggregation_error``        same columns, different aggregation functions
8. ``where_error``              WHERE differs (operators, conjunctions, #conditions)
9. ``group_by_error``           GROUP BY / HAVING differs
10. ``order_by_error``          ORDER BY direction / keys / LIMIT presence differs
11. ``wrong_value``             structure is an exact match but execution differs
                                (literal values, LIKE patterns, LIMIT value)
12. ``semantic_reasoning_error`` none of the above structural differences explains
                                the error (e.g. correct pieces combined wrongly,
                                DISTINCT, or equivalent-looking but different logic)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterator

from ratsql.schema.schema import Schema

PRIORITY = [
    "grammar_error",
    "nested_query_error",
    "schema_linking_error",
    "wrong_table",
    "wrong_join",
    "wrong_column",
    "aggregation_error",
    "where_error",
    "group_by_error",
    "order_by_error",
    "wrong_value",
    "semantic_reasoning_error",
]


def _queries(sql: dict) -> Iterator[dict]:
    yield sql
    for kind, v in sql["from"]["table_units"]:
        if kind == "sql":
            yield from _queries(v)
    for conds in (sql["from"]["conds"], sql["where"], sql["having"]):
        for cu in conds[::2]:
            for v in (cu[3], cu[4]):
                if isinstance(v, dict):
                    yield from _queries(v)
    for op in ("intersect", "union", "except"):
        if sql.get(op):
            yield from _queries(sql[op])


def _col_units(vu) -> list:
    return [c for c in (vu[1], vu[2]) if c is not None]


def _tables(sql: dict) -> Counter:
    return Counter(int(tu[1]) for q in _queries(sql) for tu in q["from"]["table_units"] if tu[0] == "table_unit")


def _columns(sql: dict) -> Counter:
    cols: Counter = Counter()
    for q in _queries(sql):
        for _, vu in q["select"][1]:
            cols.update(cu[1] for cu in _col_units(vu))
        for conds in (q["where"], q["having"]):
            for cu in conds[::2]:
                cols.update(c[1] for c in _col_units(cu[2]))
        cols.update(cu[1] for cu in q["groupBy"])
        if q["orderBy"]:
            for vu in q["orderBy"][1]:
                cols.update(c[1] for c in _col_units(vu))
    return cols


def _aggs(sql: dict) -> Counter:
    out: Counter = Counter()
    for q in _queries(sql):
        for agg, vu in q["select"][1]:
            for cu in _col_units(vu):
                out[(max(agg, cu[0]), cu[1])] += 1
        for conds in (q["having"],):
            for cu in conds[::2]:
                for c in _col_units(cu[2]):
                    out[(c[0], c[1])] += 1
        if q["orderBy"]:
            for vu in q["orderBy"][1]:
                for c in _col_units(vu):
                    out[(c[0], c[1])] += 1
    return out


def _joins(sql: dict) -> Counter:
    out: Counter = Counter()
    for q in _queries(sql):
        for cu in q["from"]["conds"][::2]:
            v = cu[3]
            if isinstance(v, (list, tuple)) and len(v) == 3:
                out[frozenset((cu[2][1][1], v[1]))] += 1
        out[("n_units", len(q["from"]["table_units"]))] += 1
    return out


def _where_sig(sql: dict) -> list:
    sig = []
    for q in _queries(sql):
        sig.append((sorted((cu[0], cu[1]) for cu in q["where"][::2]), sorted(q["where"][1::2])))
    return sig


def _nesting_sig(sql: dict) -> tuple:
    subs = 0
    for q in _queries(sql):
        for conds in (q["where"], q["having"], q["from"]["conds"]):
            subs += sum(isinstance(cu[3], dict) or isinstance(cu[4], dict) for cu in conds[::2])
        subs += sum(tu[0] == "sql" for tu in q["from"]["table_units"])
    ops = tuple(op for op in ("intersect", "union", "except") if sql.get(op))
    return subs, ops


STRONG_COLUMN_LINKS = {"EM", "VEM"}
STRONG_TABLE_LINKS = {"EM"}


def linked_elements(links: dict, strong_only: bool = True) -> tuple[dict[int, set], dict[int, set]]:
    """Columns / tables with question links.  By default only *strong* evidence counts
    (exact name match or exact cell-value match): partial matches are attached to most
    schema elements (precision ~12 % on dev) and would make every error a "linking" error."""
    cols: dict[int, set] = {}
    tabs: dict[int, set] = {}
    for q, c, t in links.get("q_col", []):
        if t != "NUM" and (not strong_only or t in STRONG_COLUMN_LINKS):
            cols.setdefault(c, set()).add(t)
    for q, t_, t in links.get("q_tab", []):
        if not strong_only or t in STRONG_TABLE_LINKS:
            tabs.setdefault(t_, set()).add(t)
    return cols, tabs


def classify_error(row: dict, gold: dict, pred: dict | None, links: dict, schema: Schema) -> dict:
    """Return {'labels': [...], 'primary': str, 'analysis': str} for an incorrect prediction."""
    notes: list[str] = []
    labels: list[str] = []
    if not row.get("grammar_valid") or pred is None:
        reason = row.get("decode_error") or row.get("parse_error") or "no well-formed SQL"
        return {"labels": ["grammar_error"], "primary": "grammar_error", "analysis": f"no valid SQL produced ({reason})"}
    name = schema.qualified_name
    gt, pt = _tables(gold), _tables(pred)
    gc, pc = _columns(gold), _columns(pred)
    link_cols, link_tabs = linked_elements(links)
    if _nesting_sig(gold) != _nesting_sig(pred):
        labels.append("nested_query_error")
        g, p = _nesting_sig(gold), _nesting_sig(pred)
        notes.append(f"nesting differs (gold sub-queries={g[0]} set-ops={list(g[1])}; pred sub-queries={p[0]} set-ops={list(p[1])})")
    missing_t = set(gt) - set(pt)
    extra_t = set(pt) - set(gt)
    missing_c = {c for c in gc if c not in pc and c != 0}
    extra_c = {c for c in pc if c not in gc and c != 0}
    linked_missing = [c for c in missing_c if c in link_cols] + [f"T{t}" for t in missing_t if t in link_tabs]
    linked_extra = [c for c in extra_c if c in link_cols] + [f"T{t}" for t in extra_t if t in link_tabs]
    if missing_t or extra_t or missing_c or extra_c:
        if linked_missing or linked_extra:
            labels.append("schema_linking_error")
            if linked_missing:
                desc = [name(c) + f" (link {'/'.join(sorted(link_cols[c]))})" if isinstance(c, int) else schema.tables[int(c[1:])].orig_name + " (table link)" for c in linked_missing]
                notes.append("missed linked gold element(s): " + ", ".join(desc))
            if linked_extra:
                desc = [name(c) if isinstance(c, int) else schema.tables[int(c[1:])].orig_name for c in linked_extra]
                notes.append("used linked but wrong element(s): " + ", ".join(desc))
        if missing_t or extra_t:
            labels.append("wrong_table")
            notes.append(f"tables gold={sorted(schema.tables[t].orig_name for t in gt)} pred={sorted(schema.tables[t].orig_name for t in pt)}")
        if missing_c or extra_c:
            labels.append("wrong_column")
            if missing_c:
                notes.append("missing columns: " + ", ".join(name(c) for c in sorted(missing_c)))
            if extra_c:
                notes.append("extra columns: " + ", ".join(name(c) for c in sorted(extra_c)))
    if not missing_t and not extra_t and _joins(gold) != _joins(pred):
        labels.append("wrong_join")
        notes.append("same tables but different join conditions / number of table units")
    if gc == pc and _aggs(gold) != _aggs(pred):
        labels.append("aggregation_error")
        notes.append("aggregation functions differ")
    partial = row.get("partial") or {}

    def comp_wrong(k):
        p = partial.get(k)
        return p is not None and p["f1"] != 1

    if comp_wrong("select") and not {"wrong_column", "aggregation_error", "schema_linking_error"} & set(labels):
        labels.append("wrong_column")
        notes.append("SELECT list differs (missing / extra / duplicated items)")
    if comp_wrong("where") or comp_wrong("and/or") or _where_sig(gold) != _where_sig(pred):
        labels.append("where_error")
        notes.append("WHERE conditions differ (operators / conjunctions / count)")
    if comp_wrong("group(no Having)") or comp_wrong("group"):
        labels.append("group_by_error")
        notes.append("GROUP BY / HAVING differs")
    if comp_wrong("order"):
        labels.append("order_by_error")
        notes.append(f"ORDER BY differs (gold {gold['orderBy'] or 'none'} limit={gold['limit']}; pred {pred['orderBy'] or 'none'} limit={pred['limit']})")
    if row.get("exact_match") and not row.get("exec_match"):
        labels.append("wrong_value")
        notes.append("structure matches but execution result differs (literal values / LIMIT / DISTINCT)")
    if not labels:
        labels.append("semantic_reasoning_error")
        notes.append("no single structural component explains the difference")
    labels = sorted(set(labels), key=PRIORITY.index)
    return {"labels": labels, "primary": labels[0], "analysis": "; ".join(notes)}
