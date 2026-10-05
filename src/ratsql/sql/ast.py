"""AST nodes and lossless conversion between Spider SQL dicts and grammar ASTs.

``sql_to_ast`` needs a *value resolver* that maps a literal (``'"text"'`` or a
float) to an index into the example's value-candidate list; ``ast_to_sql``
needs the inverse (candidate index -> literal).  This keeps the grammar
independent of how value candidates are produced.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ratsql.sql.constants import AGG_OPS, SQL_OPS, UNIT_OPS, WHERE_OPS
from ratsql.sql.grammar import Grammar, default_grammar


class UnsupportedSQLError(ValueError):
    """The SQL structure cannot be represented by the grammar."""


class Node:
    """A grammar AST node: a constructor name plus field values."""

    __slots__ = ("ctor", "fields")

    def __init__(self, ctor: str, **fields: Any):
        self.ctor = ctor
        self.fields = fields

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.ctor == other.ctor and self.fields == other.fields

    def __repr__(self) -> str:
        if not self.fields:
            return self.ctor
        inner = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"{self.ctor}({inner})"

    def to_json(self) -> Any:
        return {"_": self.ctor, **{k: _to_json(v) for k, v in self.fields.items()}}

    @classmethod
    def from_json(cls, obj: Any) -> "Node":
        fields = {k: _from_json(v) for k, v in obj.items() if k != "_"}
        return cls(obj["_"], **fields)

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        if not self.fields:
            return pad + self.ctor
        lines = [pad + self.ctor]
        for k, v in self.fields.items():
            if isinstance(v, Node):
                lines.append(f"{pad}  .{k}:")
                lines.append(v.pretty(indent + 2))
            elif isinstance(v, list):
                lines.append(f"{pad}  .{k}: [{len(v)}]")
                for x in v:
                    lines.append(x.pretty(indent + 2))
            else:
                lines.append(f"{pad}  .{k} = {v}")
        return "\n".join(lines)


def _to_json(v: Any) -> Any:
    if isinstance(v, Node):
        return v.to_json()
    if isinstance(v, list):
        return [_to_json(x) for x in v]
    return v


def _from_json(v: Any) -> Any:
    if isinstance(v, dict):
        return Node.from_json(v)
    if isinstance(v, list):
        return [_from_json(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# name tables
_AGG_CTORS = ("NoAgg", "Max", "Min", "Count", "Sum", "Avg")
_UNIT_CTORS = ("Column", "Minus", "Plus", "Times", "Divide")
_CMP_CTORS = {2: "Eq", 3: "Gt", 4: "Lt", 5: "Ge", 6: "Le", 7: "Ne", 8: "In", 9: "Like", 10: "Is", 11: "Exists"}
_CMP_IDS = {v: k for k, v in _CMP_CTORS.items()}
_IUE_CTORS = {"intersect": "Intersect", "union": "Union", "except": "Except"}
_IUE_KEYS = {v: k for k, v in _IUE_CTORS.items()}

ValueResolver = Callable[[Any, str], Optional[int]]  # (raw literal, context) -> candidate index
ValueLookup = Callable[[int, str], Any]  # (candidate index, context) -> raw literal


def _is_col_unit(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and isinstance(v[1], int) and not isinstance(v[1], bool)


# ---------------------------------------------------------------------------
# dict -> AST
def sql_to_ast(sql: dict, resolve_value: ValueResolver) -> Node:
    """Convert a Spider SQL dict into a grammar AST."""
    sel_distinct, sel_items = sql["select"]
    if not sel_items:
        raise UnsupportedSQLError("empty SELECT")
    select = Node(
        "Select",
        distinct=Node("Distinct" if sel_distinct else "NotDistinct"),
        aggs=[Node("Agg", agg=Node(_AGG_CTORS[agg]), val=_val_unit(vu)) for agg, vu in sel_items],
    )
    where = _cond_list(sql["where"], resolve_value) if sql["where"] else None
    group_by = None
    if sql["groupBy"]:
        group_by = Node(
            "GroupBy",
            keys=[_col_unit(cu) for cu in sql["groupBy"]],
            having=_cond_list(sql["having"], resolve_value) if sql["having"] else None,
        )
    elif sql["having"]:
        raise UnsupportedSQLError("HAVING without GROUP BY")
    order_by = None
    if sql["orderBy"]:
        direction, keys = sql["orderBy"]
        order_by = Node("OrderBy", dir=Node("Desc" if direction == "desc" else "Asc"), keys=[_val_unit(vu) for vu in keys])
    limit = None
    if sql["limit"] is not None:
        n = int(sql["limit"])
        ref = None if n == 1 else resolve_value(float(n), "limit")
        limit = Node("LimitOne") if ref is None else Node("LimitValue", ref=ref)
    units = []
    for kind, val in sql["from"]["table_units"]:
        if kind == "table_unit":
            units.append(Node("TableRef", t=int(val)))
        elif kind == "sql":
            units.append(Node("TableSubquery", query=sql_to_ast(val, resolve_value)))
        else:
            raise UnsupportedSQLError(f"unknown table unit {kind}")
    if not units:
        raise UnsupportedSQLError("empty FROM")
    from_ = Node(
        "From", units=units, conds=_cond_list(sql["from"]["conds"], resolve_value) if sql["from"]["conds"] else None
    )
    iue = None
    present = [op for op in SQL_OPS if sql.get(op) is not None]
    if len(present) > 1:
        raise UnsupportedSQLError("multiple set operations at one level")
    if present:
        op = present[0]
        iue = Node(_IUE_CTORS[op], query=sql_to_ast(sql[op], resolve_value))
    return Node("Query", select=select, where=where, group_by=group_by, order_by=order_by, limit=limit, **{"from": from_}, iue=iue)


def _col_unit(cu: Any) -> Node:
    agg, col, distinct = cu
    return Node("ColUnit", agg=Node(_AGG_CTORS[agg]), col=int(col), distinct=Node("Distinct" if distinct else "NotDistinct"))


def _val_unit(vu: Any) -> Node:
    op, cu1, cu2 = vu
    if op == 0:
        return Node("Column", c=_col_unit(cu1))
    return Node(_UNIT_CTORS[op], left=_col_unit(cu1), right=_col_unit(cu2))


def _value(v: Any, context: str, resolve_value: ValueResolver) -> Node:
    if isinstance(v, dict):
        return Node("Subquery", query=sql_to_ast(v, resolve_value))
    if _is_col_unit(v):
        return Node("ColumnValue", c=_col_unit(v))
    if v is None:
        raise UnsupportedSQLError("missing value")
    ref = resolve_value(v, context)
    if ref is None:
        raise UnsupportedSQLError(f"value {v!r} could not be resolved")
    return Node("Literal", ref=int(ref))


def _cond(cu: Any, resolve_value: ValueResolver) -> Node:
    not_op, op_id, vu, v1, v2 = cu
    neg = Node("Negated" if not_op else "Positive")
    if op_id == WHERE_OPS.index("between"):
        return Node(
            "Between", neg=neg, left=_val_unit(vu), low=_value(v1, "between", resolve_value), high=_value(v2, "between", resolve_value)
        )
    if op_id not in _CMP_CTORS:
        raise UnsupportedSQLError(f"unsupported operator id {op_id}")
    op_name = _CMP_CTORS[op_id]
    return Node("Cmp", neg=neg, op=Node(op_name), left=_val_unit(vu), right=_value(v1, op_name.lower(), resolve_value))


def _cond_list(conds: list, resolve_value: ValueResolver) -> Node:
    if len(conds) % 2 != 1:
        raise UnsupportedSQLError("malformed condition list")
    first = _cond(conds[0], resolve_value)
    rest = []
    for i in range(1, len(conds), 2):
        conj = conds[i]
        if conj not in ("and", "or"):
            raise UnsupportedSQLError(f"bad conjunction {conj}")
        rest.append(Node("And" if conj == "and" else "Or", c=_cond(conds[i + 1], resolve_value)))
    return Node("Conds", first=first, rest=rest)


# ---------------------------------------------------------------------------
# AST -> dict
def ast_to_sql(node: Node, lookup_value: ValueLookup) -> dict:
    """Convert a grammar AST back into a Spider SQL dict (tuples for units)."""
    if node.ctor != "Query":
        raise ValueError(f"expected Query, got {node.ctor}")
    select = node["select"]
    sql: dict = {
        "select": (select["distinct"].ctor == "Distinct", [(_AGG_CTORS.index(a["agg"].ctor), _dict_val_unit(a["val"])) for a in select["aggs"]]),
    }
    frm = node["from"]
    table_units = []
    for u in frm["units"]:
        if u.ctor == "TableRef":
            table_units.append(("table_unit", int(u["t"])))
        else:
            table_units.append(("sql", ast_to_sql(u["query"], lookup_value)))
    sql["from"] = {"table_units": table_units, "conds": _dict_conds(frm["conds"], lookup_value)}
    sql["where"] = _dict_conds(node["where"], lookup_value)
    gb = node["group_by"]
    sql["groupBy"] = [_dict_col_unit(c) for c in gb["keys"]] if gb is not None else []
    sql["having"] = _dict_conds(gb["having"], lookup_value) if gb is not None else []
    ob = node["order_by"]
    sql["orderBy"] = ("desc" if ob["dir"].ctor == "Desc" else "asc", [_dict_val_unit(v) for v in ob["keys"]]) if ob is not None else []
    lim = node["limit"]
    if lim is None:
        sql["limit"] = None
    elif lim.ctor == "LimitOne":
        sql["limit"] = 1
    else:
        raw = lookup_value(lim["ref"], "limit")
        try:
            sql["limit"] = max(1, int(float(str(raw).strip('"'))))
        except (TypeError, ValueError):
            sql["limit"] = 1
    for op in SQL_OPS:
        sql[op] = None
    iue = node["iue"]
    if iue is not None:
        sql[_IUE_KEYS[iue.ctor]] = ast_to_sql(iue["query"], lookup_value)
    return sql


def _dict_col_unit(n: Node) -> tuple:
    return (_AGG_CTORS.index(n["agg"].ctor), int(n["col"]), n["distinct"].ctor == "Distinct")


def _dict_val_unit(n: Node) -> tuple:
    if n.ctor == "Column":
        return (0, _dict_col_unit(n["c"]), None)
    return (_UNIT_CTORS.index(n.ctor), _dict_col_unit(n["left"]), _dict_col_unit(n["right"]))


def _dict_value(n: Node, context: str, lookup_value: ValueLookup) -> Any:
    if n.ctor == "Subquery":
        return ast_to_sql(n["query"], lookup_value)
    if n.ctor == "ColumnValue":
        return _dict_col_unit(n["c"])
    return lookup_value(n["ref"], context)


def _dict_cond(n: Node, lookup_value: ValueLookup) -> tuple:
    not_op = n["neg"].ctor == "Negated"
    if n.ctor == "Between":
        return (
            not_op,
            WHERE_OPS.index("between"),
            _dict_val_unit(n["left"]),
            _dict_value(n["low"], "between", lookup_value),
            _dict_value(n["high"], "between", lookup_value),
        )
    op_name = n["op"].ctor
    return (not_op, _CMP_IDS[op_name], _dict_val_unit(n["left"]), _dict_value(n["right"], op_name.lower(), lookup_value), None)


def _dict_conds(n: Node | None, lookup_value: ValueLookup) -> list:
    if n is None:
        return []
    out: list = [_dict_cond(n["first"], lookup_value)]
    for conj in n["rest"]:
        out.append("and" if conj.ctor == "And" else "or")
        out.append(_dict_cond(conj["c"], lookup_value))
    return out


# ---------------------------------------------------------------------------
def validate_ast(node: Node, grammar: Grammar | None = None, num_columns: int | None = None, num_tables: int | None = None, num_values: int | None = None) -> list[str]:
    """Structural validation of an AST against the grammar. Returns a list of problems."""
    g = grammar or default_grammar()
    problems: list[str] = []

    def check(n: Any, expected_type: str, path: str) -> None:
        if g.is_primitive(expected_type):
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                problems.append(f"{path}: expected {expected_type} index, got {n!r}")
                return
            limit = {"column": num_columns, "table": num_tables, "val_ref": num_values}[expected_type]
            if limit is not None and n >= limit:
                problems.append(f"{path}: {expected_type} index {n} out of range ({limit})")
            return
        if not isinstance(n, Node):
            problems.append(f"{path}: expected node of type {expected_type}, got {n!r}")
            return
        ctor = g.constructors.get(n.ctor)
        if ctor is None or ctor.type != expected_type:
            problems.append(f"{path}: constructor {n.ctor} is not a {expected_type}")
            return
        for f in ctor.fields:
            if f.name not in n.fields:
                problems.append(f"{path}.{f.name}: missing field")
                continue
            v = n.fields[f.name]
            if f.card == "optional":
                if v is not None:
                    check(v, f.type, f"{path}.{f.name}")
            elif f.card in ("seq", "seq1"):
                if not isinstance(v, list):
                    problems.append(f"{path}.{f.name}: expected list")
                    continue
                if f.card == "seq1" and not v:
                    problems.append(f"{path}.{f.name}: must have at least one element")
                for i, x in enumerate(v):
                    check(x, f.type, f"{path}.{f.name}[{i}]")
            else:
                check(v, f.type, f"{path}.{f.name}")

    check(node, "sql", "root")
    return problems
