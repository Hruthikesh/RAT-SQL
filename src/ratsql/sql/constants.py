"""Operator inventories of the Spider SQL representation (same indices as Spider's JSON)."""

CLAUSE_KEYWORDS = ("select", "from", "where", "group", "order", "limit", "intersect", "union", "except")
JOIN_KEYWORDS = ("join", "on", "as")
WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {"sql": "sql", "table_unit": "table_unit"}
COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")


def empty_sql() -> dict:
    """The empty SQL structure (used by Spider's evaluator for unparsable predictions)."""
    return {
        "except": None,
        "from": {"conds": [], "table_units": []},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": (False, []),
        "union": None,
        "where": [],
    }
