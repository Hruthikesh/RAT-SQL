"""An ASDL-style abstract grammar for Spider SQL.

The grammar mirrors the nested structure produced by the Spider parser (see
``parser.py``) so that every Spider query can be converted losslessly into an
abstract syntax tree (AST) and back.  It follows the design of RAT-SQL's
Spider grammar (itself derived from TRANX, Yin & Neubig 2018):

* composite types have one or more *constructors*; choosing a constructor is
  an ``ApplyRule`` action;
* fields have a cardinality: single, optional (``?``), zero-or-more (``*``)
  or one-or-more (``+``); optional and repeated fields are closed with a
  ``Reduce`` action;
* three primitive types are filled with pointer actions: ``column`` (pointer
  over schema columns), ``table`` (pointer over tables) and ``val_ref``
  (pointer over value candidates extracted from the question / database).

Field order inside ``Query`` puts FROM after SELECT/WHERE/GROUP/ORDER so that
the decoder has already chosen columns when it chooses tables (as in RAT-SQL).
Unlike the original RAT-SQL grammar we also decode literal values (via
``val_ref``) so that predictions are executable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

SPIDER_GRAMMAR = """
sql         = Query(select select, cond_list? where, group_by? group_by, order_by? order_by, limit? limit, from from, iue? iue)
select      = Select(distinct distinct, agg+ aggs)
agg         = Agg(agg_op agg, val_unit val)
val_unit    = Column(col_unit c) | Minus(col_unit left, col_unit right) | Plus(col_unit left, col_unit right)
            | Times(col_unit left, col_unit right) | Divide(col_unit left, col_unit right)
col_unit    = ColUnit(agg_op agg, column col, distinct distinct)
agg_op      = NoAgg | Max | Min | Count | Sum | Avg
distinct    = NotDistinct | Distinct
cond_list   = Conds(cond first, conj* rest)
conj        = And(cond c) | Or(cond c)
cond        = Cmp(negation neg, cmp_op op, val_unit left, value right)
            | Between(negation neg, val_unit left, value low, value high)
negation    = Positive | Negated
cmp_op      = Eq | Gt | Lt | Ge | Le | Ne | In | Like | Is | Exists
value       = Literal(val_ref ref) | Subquery(sql query) | ColumnValue(col_unit c)
group_by    = GroupBy(col_unit+ keys, cond_list? having)
order_by    = OrderBy(order_dir dir, val_unit+ keys)
order_dir   = Asc | Desc
limit       = LimitOne | LimitValue(val_ref ref)
from        = From(table_unit+ units, cond_list? conds)
table_unit  = TableRef(table t) | TableSubquery(sql query)
iue         = Intersect(sql query) | Union(sql query) | Except(sql query)
"""

PRIMITIVE_TYPES = ("column", "table", "val_ref")
ROOT_TYPE = "sql"

SINGLE, OPTIONAL, SEQ, SEQ1 = "single", "optional", "seq", "seq1"


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    card: str  # single | optional | seq | seq1

    def __str__(self) -> str:
        suffix = {SINGLE: "", OPTIONAL: "?", SEQ: "*", SEQ1: "+"}[self.card]
        return f"{self.type}{suffix} {self.name}"


@dataclass(frozen=True)
class Constructor:
    name: str
    type: str
    fields: tuple[Field, ...]

    def __str__(self) -> str:
        if not self.fields:
            return self.name
        return f"{self.name}({', '.join(str(f) for f in self.fields)})"


class Grammar:
    """Parsed grammar with integer ids for rules, types, fields and masks."""

    REDUCE = "Reduce"

    def __init__(self, text: str = SPIDER_GRAMMAR):
        self.text = text
        self.types: dict[str, list[Constructor]] = {}
        self.constructors: dict[str, Constructor] = {}
        self._parse(text)
        # ---- ids ---------------------------------------------------------
        self.rule_names: list[str] = [c for t in self.types for c in (x.name for x in self.types[t])] + [self.REDUCE]
        self.rule_id: dict[str, int] = {n: i for i, n in enumerate(self.rule_names)}
        self.reduce_id = self.rule_id[self.REDUCE]
        self.type_names: list[str] = list(self.types) + list(PRIMITIVE_TYPES)
        self.type_id: dict[str, int] = {n: i for i, n in enumerate(self.type_names)}
        # field ids: (constructor, field name); id 0 is the virtual root field
        self.field_keys: list[tuple[str, str]] = [("<root>", "<root>")]
        for c in self.constructors.values():
            for f in c.fields:
                self.field_keys.append((c.name, f.name))
        self.field_id: dict[tuple[str, str], int] = {k: i for i, k in enumerate(self.field_keys)}
        # masks: one per (type, allow_reduce)
        self.mask_keys: list[tuple[str, bool]] = []
        for t in self.types:
            self.mask_keys.append((t, False))
            self.mask_keys.append((t, True))
        self.mask_id: dict[tuple[str, bool], int] = {k: i for i, k in enumerate(self.mask_keys)}

    # ------------------------------------------------------------------ parse
    def _parse(self, text: str) -> None:
        # join continuation lines that start with '|'
        lines: list[str] = []
        for raw in text.strip().splitlines():
            line = raw.strip()
            if not line or line.startswith("--"):
                continue
            if line.startswith("|") and lines:
                lines[-1] += " " + line
            else:
                lines.append(line)
        for line in lines:
            lhs, rhs = [s.strip() for s in line.split("=", 1)]
            ctors = []
            for alt in _split_alternatives(rhs):
                m = re.fullmatch(r"(\w+)\s*(?:\((.*)\))?", alt.strip())
                if not m:
                    raise ValueError(f"Bad constructor: {alt}")
                name, args = m.group(1), m.group(2)
                fields = []
                if args:
                    for arg in args.split(","):
                        ftype, fname = arg.strip().split()
                        card = SINGLE
                        if ftype.endswith("?"):
                            card, ftype = OPTIONAL, ftype[:-1]
                        elif ftype.endswith("*"):
                            card, ftype = SEQ, ftype[:-1]
                        elif ftype.endswith("+"):
                            card, ftype = SEQ1, ftype[:-1]
                        fields.append(Field(fname, ftype, card))
                ctor = Constructor(name, lhs, tuple(fields))
                if name in self.constructors:
                    raise ValueError(f"Duplicate constructor {name}")
                self.constructors[name] = ctor
                ctors.append(ctor)
            self.types[lhs] = ctors
        for c in self.constructors.values():
            for f in c.fields:
                if f.type not in self.types and f.type not in PRIMITIVE_TYPES:
                    raise ValueError(f"Unknown type {f.type} in {c}")
                if f.type in PRIMITIVE_TYPES and f.card != SINGLE:
                    raise ValueError("Primitive fields must have single cardinality")

    # ---------------------------------------------------------------- queries
    @property
    def num_rules(self) -> int:
        return len(self.rule_names)

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    @property
    def num_fields(self) -> int:
        return len(self.field_keys)

    @property
    def num_masks(self) -> int:
        return len(self.mask_keys)

    def is_primitive(self, type_name: str) -> bool:
        return type_name in PRIMITIVE_TYPES

    def valid_rule_ids(self, type_name: str, allow_reduce: bool) -> list[int]:
        ids = [self.rule_id[c.name] for c in self.types[type_name]]
        if allow_reduce:
            ids.append(self.reduce_id)
        return ids

    @cached_property
    def mask_table(self) -> list[list[bool]]:
        """``mask_table[mask_id][rule_id]`` is True when the rule is allowed."""
        table = []
        for t, allow_reduce in self.mask_keys:
            valid = set(self.valid_rule_ids(t, allow_reduce))
            table.append([r in valid for r in range(self.num_rules)])
        return table

    def describe(self) -> str:
        out = []
        for t, ctors in self.types.items():
            out.append(f"{t} = " + " | ".join(str(c) for c in ctors))
        return "\n".join(out)


def _split_alternatives(rhs: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


_DEFAULT: Grammar | None = None


def default_grammar() -> Grammar:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Grammar()
    return _DEFAULT
