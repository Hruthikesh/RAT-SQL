"""Explicit relational schema graph.

Nodes are schema elements (``("column", i)``, ``("table", j)``) and, once a
question is attached, question tokens (``("question", k)``).  Directed edges
carry a relation label.  The graph is the *source of truth* for relations:
``RelationMatrixBuilder`` reads edge labels from it and fills every pair
without an edge with the default relation of its node-type pair.

Schema edge labels (RAT-SQL, Table 1, plus the implementation relations for
the ``*`` column and identity):

=====================  ===========================================================
cc_same_table          columns x, y belong to the same table
cc_fk_forward          column x is a foreign key referencing column y
cc_fk_backward         column y is a foreign key referencing column x
ct_primary_key         column x is the primary key of table y
ct_belongs_to          column x belongs to table y (not primary key)
ct_foreign_key         column x is a foreign key referencing a column of table y
ct_any_table           x is ``*`` (belongs to every table)
tc_*                   reverse of the ct_* relations
tt_fk_forward          table x has a foreign key referencing table y
tt_fk_backward         table y has a foreign key referencing table x
tt_fk_both             foreign keys in both directions
=====================  ===========================================================

Question edges come from schema linking (``qc_em``, ``qc_pm``, ``qc_vem``,
``qc_vpm``, ``qc_num``, ``qt_em``, ``qt_pm`` and their reverses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ratsql.schema.linking import SchemaLinks
from ratsql.schema.schema import Schema

Node = tuple[str, int]


@dataclass
class SchemaGraph:
    schema: Schema
    nodes: list[Node] = field(default_factory=list)
    edges: dict[tuple[Node, Node], str] = field(default_factory=dict)
    question_tokens: list[str] = field(default_factory=list)

    @classmethod
    def from_schema(cls, schema: Schema, use_foreign_keys: bool = True) -> "SchemaGraph":
        g = cls(schema)
        g.nodes = [("column", c.id) for c in schema.columns] + [("table", t.id) for t in schema.tables]
        cols = schema.columns
        # column -- column
        for t in schema.tables:
            for a in t.column_ids:
                for b in t.column_ids:
                    if a != b:
                        g.edges[(("column", a), ("column", b))] = "cc_same_table"
        if use_foreign_keys:
            for src, dst in schema.foreign_keys:
                g.edges[(("column", src), ("column", dst))] = "cc_fk_forward"
                if (("column", dst), ("column", src)) not in g.edges or g.edges[(("column", dst), ("column", src))] == "cc_same_table":
                    g.edges[(("column", dst), ("column", src))] = "cc_fk_backward"
        # column -- table
        for c in cols:
            for t in schema.tables:
                cn, tn = ("column", c.id), ("table", t.id)
                if c.table_id is None:
                    g.edges[(cn, tn)] = "ct_any_table"
                    g.edges[(tn, cn)] = "tc_any_table"
                elif c.table_id == t.id:
                    kind = "primary_key" if c.is_primary else "belongs_to"
                    g.edges[(cn, tn)] = f"ct_{kind}"
                    g.edges[(tn, cn)] = f"tc_{kind}"
                elif use_foreign_keys and c.fk_target is not None and cols[c.fk_target].table_id == t.id:
                    g.edges[(cn, tn)] = "ct_foreign_key"
                    g.edges[(tn, cn)] = "tc_foreign_key"
        # table -- table
        if use_foreign_keys:
            pairs = schema.fk_table_pairs()
            for a, b in pairs:
                if a == b:
                    continue
                if (b, a) in pairs:
                    g.edges[(("table", a), ("table", b))] = "tt_fk_both"
                else:
                    g.edges[(("table", a), ("table", b))] = "tt_fk_forward"
                    g.edges[(("table", b), ("table", a))] = "tt_fk_backward"
        return g

    def add_question(self, question_tokens: list[str], links: SchemaLinks) -> "SchemaGraph":
        self.question_tokens = list(question_tokens)
        self.nodes = [("question", i) for i in range(len(question_tokens))] + [n for n in self.nodes if n[0] != "question"]
        for (q, c), kind in links.q_col.items():
            k = kind.lower()
            self.edges[(("question", q), ("column", c))] = f"qc_{k}"
            self.edges[(("column", c), ("question", q))] = f"cq_{k}"
        for (q, t), kind in links.q_tab.items():
            k = kind.lower()
            self.edges[(("question", q), ("table", t))] = f"qt_{k}"
            self.edges[(("table", t), ("question", q))] = f"tq_{k}"
        return self

    # ---------------------------------------------------------------- helpers
    def node_label(self, node: Node) -> str:
        kind, i = node
        if kind == "question":
            return self.question_tokens[i]
        if kind == "table":
            return self.schema.tables[i].orig_name
        return self.schema.qualified_name(i)

    def to_networkx(self, include_same_table: bool = False):
        import networkx as nx

        G = nx.MultiDiGraph()
        for n in self.nodes:
            G.add_node(n, kind=n[0], label=self.node_label(n))
        for (a, b), lab in self.edges.items():
            if not include_same_table and lab == "cc_same_table":
                continue
            G.add_edge(a, b, relation=lab)
        return G

    def edge_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for lab in self.edges.values():
            out[lab] = out.get(lab, 0) + 1
        return dict(sorted(out.items()))
