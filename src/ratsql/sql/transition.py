"""Transition system: AST <-> action sequence, with grammar masks.

The decoder builds the AST top-down, left-to-right (pre-order).  At every step
the *frontier* is the first unfilled field; it determines

* the action kind  (``RULE`` for composite types, ``COLUMN`` / ``TABLE`` /
  ``VALUE`` pointers for primitive types),
* the set of grammatically valid rules (constructors of the field type, plus
  ``Reduce`` for optional fields and for repeated fields after their minimum
  length) -> a *mask id*,
* the parent step (whose LSTM state is fed to the decoder, "parent feeding")
  and the parent rule (constructor of the node that owns the field).

Nodes whose type has a single constructor in a non-optional position are
expanded automatically (no action is emitted) because their probability is 1
under the grammar; this shortens action sequences by ~25 % without changing
the model distribution.

``DecodeState.apply`` rejects any action that is not valid for the current
frontier (``InvalidActionError``).  Grammar-constrained decoding masks the
decoder's distribution with ``valid_mask`` so an invalid action can never be
selected; the "no grammar" ablation decodes without the mask and lets
``apply`` detect the resulting ill-formed trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ratsql.sql.ast import Node
from ratsql.sql.grammar import OPTIONAL, SEQ, SEQ1, SINGLE, Field, Grammar, default_grammar

RULE, COLUMN, TABLE, VALUE = 0, 1, 2, 3
KIND_NAMES = ("rule", "column", "table", "value")
PRIMITIVE_KIND = {"column": COLUMN, "table": TABLE, "val_ref": VALUE}

Action = tuple[int, int]  # (kind, index)


class InvalidActionError(ValueError):
    pass


@dataclass(frozen=True)
class Frontier:
    kind: int
    type_name: str
    type_id: int
    field_id: int
    allow_reduce: bool
    mask_id: int  # -1 for pointer steps
    parent_step: int  # -1 if the parent node was produced without an emitted action at the root
    parent_rule: int  # rule id of the owner constructor, -1 for the virtual root field


@dataclass
class _Slot:
    owner: Any  # Node or the root holder dict
    field: Field
    field_id: int
    parent_step: int
    parent_rule: int
    count: int = 0

    def allow_reduce(self) -> bool:
        if self.field.card in (OPTIONAL, SEQ):
            return True
        if self.field.card == SEQ1:
            return self.count > 0
        return False


_ROOT_FIELD = Field("<root>", "sql", SINGLE)


class DecodeState:
    """Incremental AST construction driven by actions."""

    def __init__(self, grammar: Grammar | None = None, num_columns: int | None = None, num_tables: int | None = None, num_values: int | None = None, auto_expand: bool = True):
        self.g = grammar or default_grammar()
        self.num_columns = num_columns
        self.num_tables = num_tables
        self.num_values = num_values
        self.auto_expand = auto_expand
        self.root_holder: dict[str, Any] = {"<root>": None}
        self.stack: list[_Slot] = [_Slot(self.root_holder, _ROOT_FIELD, 0, -1, -1)]
        self.step = 0
        self.actions: list[Action] = []
        self._resolve_auto()

    # ------------------------------------------------------------------ core
    def _set_field(self, slot: _Slot, value: Any) -> None:
        name = slot.field.name
        if isinstance(slot.owner, Node):
            if slot.field.card in (SEQ, SEQ1):
                slot.owner.fields[name].append(value)
            else:
                slot.owner.fields[name] = value
        else:
            slot.owner[name] = value

    def _expand(self, slot: _Slot, ctor_name: str, step: int) -> None:
        ctor = self.g.constructors[ctor_name]
        node = Node(ctor_name)
        for f in ctor.fields:
            node.fields[f.name] = [] if f.card in (SEQ, SEQ1) else None
        self._set_field(slot, node)
        if slot.field.card in (SEQ, SEQ1):
            slot.count += 1
        else:
            self.stack.pop()
        rule = self.g.rule_id[ctor_name]
        for f in reversed(ctor.fields):
            self.stack.append(_Slot(node, f, self.g.field_id[(ctor_name, f.name)], step, rule))

    def _resolve_auto(self) -> None:
        if not self.auto_expand:
            return
        while self.stack:
            slot = self.stack[-1]
            ftype = slot.field.type
            if self.g.is_primitive(ftype):
                return
            ctors = self.g.types[ftype]
            if len(ctors) == 1 and not slot.allow_reduce():
                self._expand(slot, ctors[0].name, slot.parent_step)
            else:
                return

    # --------------------------------------------------------------- queries
    @property
    def is_done(self) -> bool:
        return not self.stack

    @property
    def ast(self) -> Node | None:
        return self.root_holder["<root>"] if self.is_done else None

    @property
    def partial_ast(self) -> Node | None:
        return self.root_holder["<root>"]

    def frontier(self) -> Frontier | None:
        if not self.stack:
            return None
        slot = self.stack[-1]
        ftype = slot.field.type
        if self.g.is_primitive(ftype):
            return Frontier(PRIMITIVE_KIND[ftype], ftype, self.g.type_id[ftype], slot.field_id, False, -1, slot.parent_step, slot.parent_rule)
        allow = slot.allow_reduce()
        return Frontier(RULE, ftype, self.g.type_id[ftype], slot.field_id, allow, self.g.mask_id[(ftype, allow)], slot.parent_step, slot.parent_rule)

    def valid_rules(self) -> list[int]:
        fr = self.frontier()
        if fr is None or fr.kind != RULE:
            return []
        return self.g.valid_rule_ids(fr.type_name, fr.allow_reduce)

    def pointer_size(self, kind: int) -> int | None:
        return {COLUMN: self.num_columns, TABLE: self.num_tables, VALUE: self.num_values}.get(kind)

    # ----------------------------------------------------------------- apply
    def apply(self, action: Action) -> Frontier:
        """Apply ``action`` and return the frontier it was applied at."""
        fr = self.frontier()
        if fr is None:
            raise InvalidActionError("decoding already finished")
        kind, idx = int(action[0]), int(action[1])
        if kind != fr.kind:
            raise InvalidActionError(f"expected a {KIND_NAMES[fr.kind]} action for {fr.type_name}, got {KIND_NAMES[kind]}")
        slot = self.stack[-1]
        step = self.step
        if kind == RULE:
            valid = self.g.valid_rule_ids(fr.type_name, fr.allow_reduce)
            if idx not in valid:
                raise InvalidActionError(f"rule {self.g.rule_names[idx] if 0 <= idx < self.g.num_rules else idx} invalid for type {fr.type_name}")
            if idx == self.g.reduce_id:
                self.stack.pop()
            else:
                self._expand(slot, self.g.rule_names[idx], step)
        else:
            size = self.pointer_size(kind)
            if idx < 0 or (size is not None and idx >= size):
                raise InvalidActionError(f"{KIND_NAMES[kind]} index {idx} out of range {size}")
            self._set_field(slot, idx)
            self.stack.pop()
        self.step += 1
        self.actions.append((kind, idx))
        self._resolve_auto()
        return fr

    def copy(self) -> "DecodeState":
        """Deep copy (used by beam search)."""
        import copy as _copy

        new = DecodeState.__new__(DecodeState)
        memo: dict = {}
        new.g = self.g
        new.num_columns, new.num_tables, new.num_values = self.num_columns, self.num_tables, self.num_values
        new.auto_expand = self.auto_expand
        new.root_holder = _copy.deepcopy(self.root_holder, memo)
        new.stack = [
            _Slot(_copy.deepcopy(s.owner, memo), s.field, s.field_id, s.parent_step, s.parent_rule, s.count) for s in self.stack
        ]
        new.step = self.step
        new.actions = list(self.actions)
        return new


# ---------------------------------------------------------------------------
def ast_to_actions(ast: Node, grammar: Grammar | None = None, auto_expand: bool = True) -> list[Action]:
    """Pre-order linearisation of an AST into actions (the oracle sequence)."""
    g = grammar or default_grammar()
    actions: list[Action] = []

    def visit_node(node: Node, type_name: str, allow_reduce: bool) -> None:
        ctors = g.types[type_name]
        if node.ctor not in {c.name for c in ctors}:
            raise ValueError(f"{node.ctor} is not a constructor of {type_name}")
        if not (auto_expand and len(ctors) == 1 and not allow_reduce):
            actions.append((RULE, g.rule_id[node.ctor]))
        for f in g.constructors[node.ctor].fields:
            visit_field(node.fields.get(f.name), f)

    def visit_field(value: Any, f: Field) -> None:
        if g.is_primitive(f.type):
            actions.append((PRIMITIVE_KIND[f.type], int(value)))
            return
        if f.card == SINGLE:
            visit_node(value, f.type, False)
        elif f.card == OPTIONAL:
            if value is None:
                actions.append((RULE, g.reduce_id))
            else:
                visit_node(value, f.type, True)
        else:
            for i, v in enumerate(value):
                visit_node(v, f.type, f.card == SEQ or i > 0)
            actions.append((RULE, g.reduce_id))

    visit_node(ast, "sql", False)
    return actions


@dataclass
class StepInfo:
    kind: int
    target: int
    type_id: int
    field_id: int
    mask_id: int
    parent_step: int
    parent_rule: int


def replay(actions: list[Action], grammar: Grammar | None = None, num_columns: int | None = None, num_tables: int | None = None, num_values: int | None = None, auto_expand: bool = True) -> tuple[Node, list[StepInfo]]:
    """Replay an action sequence; returns the AST and per-step decoder metadata."""
    state = DecodeState(grammar, num_columns, num_tables, num_values, auto_expand)
    infos: list[StepInfo] = []
    for a in actions:
        fr = state.apply(a)
        infos.append(StepInfo(a[0], a[1], fr.type_id, fr.field_id, fr.mask_id, fr.parent_step, fr.parent_rule))
    if not state.is_done:
        raise InvalidActionError(f"action sequence ended before the AST was complete ({len(state.stack)} open fields)")
    return state.ast, infos  # type: ignore[return-value]
