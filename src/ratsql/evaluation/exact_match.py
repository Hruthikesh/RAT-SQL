"""Spider Exact-Set-Match (EM) evaluation.

An independent re-implementation of the matching rules of the official Spider
``evaluation.py`` (Yu et al., 2018).  Both gold and predicted SQL are parsed
into the Spider structure, then:

1. values are removed (``DISABLE_VALUE``): non-subquery literals become None;
2. DISTINCT flags are dropped for columns / select (``DISABLE_DISTINCT``);
3. columns that are foreign-key equivalent are mapped to one representative
   (only for columns of tables that appear in the query's own FROM);
4. ten components are compared as *sets* (order-insensitive):
   select, select(no AGG), where, where(no OP), group(no Having), group,
   order, and/or, IUEN, keywords -- a query is an exact match iff all ten
   components match and the FROM table units are identical.

We deliberately replicate the official quirks so that our numbers are
comparable to published Spider EM numbers:

* the FK "union" is the greedy key-set assignment of ``build_foreign_key_map``
  (not a full transitive closure);
* FK normalisation and DISTINCT removal are *not* applied inside sub-queries
  that appear as condition values (the official code only rebuilds the
  top-level val_units), but values are removed recursively;
* GROUP BY columns are compared by column *name* only (table ignored);
* ORDER BY compares the (direction, keys) tuple and only the *presence* of LIMIT.

``tests/test_exact_match.py`` checks these behaviours on hand-made cases.
"""

from __future__ import annotations

import copy
from typing import Any

from ratsql.schema.schema import Schema
from ratsql.sql.constants import AGG_OPS, WHERE_OPS, empty_sql
from ratsql.sql.parser import SQLParseError, parse_sql

COMPONENTS = (
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
)


# ---------------------------------------------------------------------------
# foreign-key normalisation
def build_foreign_key_map(schema: Schema) -> dict[int, int]:
    key_sets: list[set[int]] = []
    for k1, k2 in schema.foreign_keys:
        target = None
        for ks in key_sets:
            if k1 in ks or k2 in ks:
                target = ks
                break
        if target is None:
            target = set()
            key_sets.append(target)
        target.add(k1)
        target.add(k2)
    fk_map: dict[int, int] = {}
    for ks in key_sets:
        ordered = sorted(ks)
        for idx in ordered:
            fk_map[idx] = ordered[0]
    return fk_map


def _valid_col_units(table_units: list, schema: Schema) -> set[int]:
    tables = {int(tu[1]) for tu in table_units if tu[0] == "table_unit"}
    return {c.id for c in schema.columns if c.table_id is not None and c.table_id in tables}


def _rebuild_col_unit(valid: set[int], cu: Any, kmap: dict[int, int]) -> Any:
    if cu is None:
        return cu
    agg, col, _distinct = cu
    if col in kmap and col in valid:
        col = kmap[col]
    return (agg, col, None)


def _rebuild_val_unit(valid, vu, kmap):
    if vu is None:
        return vu
    op, c1, c2 = vu
    return (op, _rebuild_col_unit(valid, c1, kmap), _rebuild_col_unit(valid, c2, kmap))


def _rebuild_cond_col(valid, conds, kmap):
    out = []
    for c in conds:
        if isinstance(c, str):
            out.append(c)
        else:
            not_op, op_id, vu, v1, v2 = c
            out.append((not_op, op_id, _rebuild_val_unit(valid, vu, kmap), v1, v2))
    return out


def _rebuild_sql_col(valid, sql, kmap):
    if sql is None:
        return sql
    distinct, items = sql["select"]
    sql["select"] = (None, [(agg, _rebuild_val_unit(valid, vu, kmap)) for agg, vu in items])
    sql["from"]["table_units"] = [
        (kind, _rebuild_col_unit(valid, val, kmap) if isinstance(val, tuple) else val) for kind, val in sql["from"]["table_units"]
    ]
    sql["from"]["conds"] = _rebuild_cond_col(valid, sql["from"]["conds"], kmap)
    sql["where"] = _rebuild_cond_col(valid, sql["where"], kmap)
    sql["groupBy"] = [_rebuild_col_unit(valid, cu, kmap) for cu in sql["groupBy"]]
    if sql["orderBy"]:
        direction, keys = sql["orderBy"]
        sql["orderBy"] = (direction, [_rebuild_val_unit(valid, vu, kmap) for vu in keys])
    sql["having"] = _rebuild_cond_col(valid, sql["having"], kmap)
    for op in ("intersect", "except", "union"):
        sql[op] = _rebuild_sql_col(valid, sql[op], kmap)
    return sql


def _rebuild_cond_val(conds):
    out = []
    for c in conds:
        if isinstance(c, str):
            out.append(c)
            continue
        not_op, op_id, vu, v1, v2 = c
        v1 = _rebuild_sql_val(v1) if isinstance(v1, dict) else None
        v2 = _rebuild_sql_val(v2) if isinstance(v2, dict) else None
        out.append((not_op, op_id, vu, v1, v2))
    return out


def _rebuild_sql_val(sql):
    if sql is None:
        return sql
    sql["from"]["conds"] = _rebuild_cond_val(sql["from"]["conds"])
    sql["having"] = _rebuild_cond_val(sql["having"])
    sql["where"] = _rebuild_cond_val(sql["where"])
    for op in ("intersect", "except", "union"):
        sql[op] = _rebuild_sql_val(sql[op])
    return sql


def normalize_for_em(sql: dict, schema: Schema, kmap: dict[int, int] | None = None) -> dict:
    kmap = build_foreign_key_map(schema) if kmap is None else kmap
    sql = copy.deepcopy(sql)
    valid = _valid_col_units(sql["from"]["table_units"], schema)
    sql = _rebuild_sql_val(sql)
    sql = _rebuild_sql_col(valid, sql, kmap)
    return sql


# ---------------------------------------------------------------------------
# component matching
def _scores(count: int, pred_total: int, label_total: int) -> tuple[int, int, int]:
    if pred_total != label_total:
        return 0, 0, 0
    if count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def _eval_sel(pred, label):
    pred_sel = list(pred["select"][1])
    label_sel = list(label["select"][1])
    label_wo_agg = [u[1] for u in label_sel]
    cnt = cnt_wo_agg = 0
    pred_total, label_total = len(pred_sel), len(label_sel)
    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])
    return label_total, pred_total, cnt, cnt_wo_agg


def _eval_where(pred, label):
    pred_conds = list(pred["where"][::2])
    label_conds = list(label["where"][::2])
    label_wo_agg = [u[2] for u in label_conds]
    cnt = cnt_wo_agg = 0
    pred_total, label_total = len(pred_conds), len(label_conds)
    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])
    return label_total, pred_total, cnt, cnt_wo_agg


class ExactMatchEvaluator:
    """Spider exact-set-match for one database schema."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.kmap = build_foreign_key_map(schema)
        self._col_name = {c.id: ("__all__" if c.table_id is None else c.orig_name.lower()) for c in schema.columns}

    # ------------------------------------------------------------ components
    def _eval_group(self, pred, label):
        pred_cols = [self._col_name.get(u[1], u[1]) for u in pred["groupBy"]]
        label_cols = [self._col_name.get(u[1], u[1]) for u in label["groupBy"]]
        cnt = 0
        pred_total, label_total = len(pred_cols), len(label_cols)
        for col in pred_cols:
            if col in label_cols:
                cnt += 1
                label_cols.remove(col)
        return label_total, pred_total, cnt

    @staticmethod
    def _eval_having(pred, label):
        pred_total = 1 if pred["groupBy"] else 0
        label_total = 1 if label["groupBy"] else 0
        cnt = 0
        if pred_total == label_total == 1 and [u[1] for u in pred["groupBy"]] == [u[1] for u in label["groupBy"]] and pred["having"] == label["having"]:
            cnt = 1
        return label_total, pred_total, cnt

    @staticmethod
    def _eval_order(pred, label):
        pred_total = 1 if pred["orderBy"] else 0
        label_total = 1 if label["orderBy"] else 0
        cnt = 0
        if label["orderBy"] and pred["orderBy"] == label["orderBy"] and ((pred["limit"] is None) == (label["limit"] is None)):
            cnt = 1
        return label_total, pred_total, cnt

    @staticmethod
    def _eval_and_or(pred, label):
        pred_ao = set(pred["where"][1::2])
        label_ao = set(label["where"][1::2])
        if pred_ao == label_ao:
            return 1, 1, 1
        return len(pred_ao), len(label_ao), 0

    def _eval_nested(self, pred, label):
        label_total = 1 if label is not None else 0
        pred_total = 1 if pred is not None else 0
        cnt = 0
        if pred is not None and label is not None:
            cnt = int(self._exact(pred, label))
        return label_total, pred_total, cnt

    def _eval_iuen(self, pred, label):
        lt = pt = cnt = 0
        for op in ("intersect", "except", "union"):
            a, b, c = self._eval_nested(pred[op], label[op])
            lt, pt, cnt = lt + a, pt + b, cnt + c
        return lt, pt, cnt

    @staticmethod
    def get_keywords(sql) -> set[str]:
        res = set()
        if sql["where"]:
            res.add("where")
        if sql["groupBy"]:
            res.add("group")
        if sql["having"]:
            res.add("having")
        if sql["orderBy"]:
            res.add(sql["orderBy"][0])
            res.add("order")
        if sql["limit"] is not None:
            res.add("limit")
        for op in ("except", "union", "intersect"):
            if sql[op] is not None:
                res.add(op)
        ao = list(sql["from"]["conds"][1::2]) + list(sql["where"][1::2]) + list(sql["having"][1::2])
        if any(t == "or" for t in ao):
            res.add("or")
        cond_units = list(sql["from"]["conds"][::2]) + list(sql["where"][::2]) + list(sql["having"][::2])
        if any(cu[0] for cu in cond_units):
            res.add("not")
        if any(cu[1] == WHERE_OPS.index("in") for cu in cond_units):
            res.add("in")
        if any(cu[1] == WHERE_OPS.index("like") for cu in cond_units):
            res.add("like")
        return res

    def _eval_keywords(self, pred, label):
        pk, lk = self.get_keywords(pred), self.get_keywords(label)
        return len(lk), len(pk), sum(1 for k in pk if k in lk)

    def partial_match(self, pred: dict, label: dict) -> dict[str, dict[str, float]]:
        res: dict[str, dict[str, float]] = {}

        def put(name, lt, pt, cnt):
            acc, rec, f1 = _scores(cnt, pt, lt)
            res[name] = {"acc": acc, "rec": rec, "f1": f1, "label_total": lt, "pred_total": pt}

        lt, pt, cnt, cnt_wo = _eval_sel(pred, label)
        put("select", lt, pt, cnt)
        put("select(no AGG)", lt, pt, cnt_wo)
        lt, pt, cnt, cnt_wo = _eval_where(pred, label)
        put("where", lt, pt, cnt)
        put("where(no OP)", lt, pt, cnt_wo)
        put("group(no Having)", *self._eval_group(pred, label))
        put("group", *self._eval_having(pred, label))
        put("order", *self._eval_order(pred, label))
        put("and/or", *self._eval_and_or(pred, label))
        put("IUEN", *self._eval_iuen(pred, label))
        put("keywords", *self._eval_keywords(pred, label))
        return res

    def _exact(self, pred: dict, label: dict) -> bool:
        partial = self.partial_match(pred, label)
        if any(s["f1"] != 1 for s in partial.values()):
            return False
        if label["from"]["table_units"]:
            return _sorted_units(label["from"]["table_units"]) == _sorted_units(pred["from"]["table_units"])
        return True

    # ------------------------------------------------------------------ API
    def evaluate(self, pred_sql: dict | None, gold_sql: dict) -> dict:
        """Compare two *parsed* SQL dicts; ``pred_sql=None`` means unparsable."""
        pred = empty_sql() if pred_sql is None else pred_sql
        g = normalize_for_em(gold_sql, self.schema, self.kmap)
        p = normalize_for_em(pred, self.schema, self.kmap)
        partial = self.partial_match(p, g)
        exact = all(s["f1"] == 1 for s in partial.values())
        if exact and g["from"]["table_units"]:
            exact = _sorted_units(g["from"]["table_units"]) == _sorted_units(p["from"]["table_units"])
        return {"exact": bool(exact), "partial": partial}

    def evaluate_strings(self, pred_str: str, gold_str: str) -> dict:
        gold = parse_sql(gold_str, self.schema, strict=False)
        try:
            pred = parse_sql(pred_str, self.schema, strict=False)
            parse_ok = True
        except SQLParseError:
            pred, parse_ok = None, False
        out = self.evaluate(pred, gold)
        out["pred_parse_ok"] = parse_ok
        return out


def _sorted_units(units: list) -> list:
    # sub-query units are dicts, which are not orderable; sort by a stable key
    return sorted(units, key=lambda u: (u[0], repr(u[1])))


# ---------------------------------------------------------------------------
# hardness (Spider difficulty levels)
def _count_agg(units) -> int:
    return len([u for u in units if u[0] != AGG_OPS.index("none")])


def _nested(sql) -> list:
    nested = []
    for cu in list(sql["from"]["conds"][::2]) + list(sql["where"][::2]) + list(sql["having"][::2]):
        if isinstance(cu[3], dict):
            nested.append(cu[3])
        if isinstance(cu[4], dict):
            nested.append(cu[4])
    for op in ("intersect", "except", "union"):
        if sql[op] is not None:
            nested.append(sql[op])
    return nested


def _component1(sql) -> int:
    count = 0
    count += 1 if sql["where"] else 0
    count += 1 if sql["groupBy"] else 0
    count += 1 if sql["orderBy"] else 0
    count += 1 if sql["limit"] is not None else 0
    if sql["from"]["table_units"]:
        count += len(sql["from"]["table_units"]) - 1
    ao = list(sql["from"]["conds"][1::2]) + list(sql["where"][1::2]) + list(sql["having"][1::2])
    count += len([t for t in ao if t == "or"])
    cus = list(sql["from"]["conds"][::2]) + list(sql["where"][::2]) + list(sql["having"][::2])
    count += len([cu for cu in cus if cu[1] == WHERE_OPS.index("like")])
    return count


def _others(sql) -> int:
    count = 0
    agg = _count_agg(sql["select"][1])
    agg += _count_agg(sql["where"][::2])  # official quirk: counts NOT flags
    agg += _count_agg(sql["groupBy"])
    if sql["orderBy"]:
        agg += _count_agg([u[1] for u in sql["orderBy"][1] if u[1]] + [u[2] for u in sql["orderBy"][1] if u[2]])
    agg += _count_agg(sql["having"])  # official quirk: connectors count as aggregations
    if agg > 1:
        count += 1
    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def hardness(sql: dict) -> str:
    """Spider difficulty level of a parsed SQL: easy | medium | hard | extra."""
    c1, c2, oth = _component1(sql), len(_nested(sql)), _others(sql)
    if c1 <= 1 and oth == 0 and c2 == 0:
        return "easy"
    if (oth <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and oth < 2 and c2 == 0):
        return "medium"
    if (oth > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and oth <= 2 and c2 == 0) or (c1 <= 1 and oth == 0 and c2 <= 1):
        return "hard"
    return "extra"


HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")
