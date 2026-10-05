"""Visualisation of schema graphs, relation matrices and schema links (matplotlib).

The functions do not select a matplotlib backend: scripts use ``Agg`` and notebooks
use ``%matplotlib inline``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ratsql.schema.graph import SchemaGraph
from ratsql.schema.relations import RelationMatrix

EDGE_COLORS = {
    "fk": "#c05621",
    "primary_key": "#2b6cb0",
    "belongs_to": "#a0aec0",
    "any_table": "#e2e8f0",
    "qc": "#2f855a",
    "qt": "#6b46c1",
}


def _edge_color(label: str) -> str:
    if "fk" in label or "foreign" in label:
        return EDGE_COLORS["fk"]
    if "primary" in label:
        return EDGE_COLORS["primary_key"]
    if "any_table" in label:
        return EDGE_COLORS["any_table"]
    if label.startswith(("qc_", "cq_")):
        return EDGE_COLORS["qc"]
    if label.startswith(("qt_", "tq_")):
        return EDGE_COLORS["qt"]
    return EDGE_COLORS["belongs_to"]


def plot_schema_graph(graph: SchemaGraph, path: str | Path | None = None, title: str | None = None, show_star: bool = False):
    """Tables as large nodes, columns around them; FK edges highlighted; question links if present."""
    import matplotlib.pyplot as plt
    import networkx as nx

    schema = graph.schema
    G = nx.DiGraph()
    for t in schema.tables:
        G.add_node(("table", t.id))
    for c in schema.columns:
        if c.table_id is None and not show_star:
            continue
        G.add_node(("column", c.id))
    for q in range(len(graph.question_tokens)):
        G.add_node(("question", q))
    drawn = []
    for (a, b), lab in graph.edges.items():
        if a not in G or b not in G:
            continue
        # draw each undirected pair once; skip dense same-table / reverse edges
        if lab in ("cc_same_table",) or lab.startswith(("tc_", "cq_", "tq_")) or lab.endswith("backward"):
            continue
        G.add_edge(a, b, label=lab)
        drawn.append((a, b, lab))
    # layout: tables on a circle, columns near their table, question tokens on a line below
    pos = {}
    n_t = max(1, schema.num_tables)
    for t in schema.tables:
        ang = 2 * np.pi * t.id / n_t
        pos[("table", t.id)] = np.array([np.cos(ang), np.sin(ang)]) * 3.0
    for t in schema.tables:
        cols = t.column_ids
        for k, cid in enumerate(cols):
            ang = 2 * np.pi * k / max(1, len(cols))
            pos[("column", cid)] = pos[("table", t.id)] + np.array([np.cos(ang), np.sin(ang)]) * 1.1
    if show_star:
        pos[("column", 0)] = np.array([0.0, 0.0])
    nq = len(graph.question_tokens)
    for q in range(nq):
        pos[("question", q)] = np.array([-4 + 8 * q / max(1, nq - 1), -5.2])
    fig, ax = plt.subplots(figsize=(11, 9 if nq else 8))
    for a, b, lab in drawn:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color=_edge_color(lab), lw=2.2 if "fk" in lab else 0.8, alpha=0.9 if "fk" in lab or lab.startswith("q") else 0.5, zorder=1)
    for n in G.nodes:
        x, y = pos[n]
        kind, i = n
        if kind == "table":
            ax.scatter([x], [y], s=500, color="#2c5282", zorder=3)
            ax.text(x, y + 0.32, schema.tables[i].orig_name, color="#1a365d", ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2c5282", lw=0.8))
        elif kind == "column":
            col = schema.columns[i]
            color = "#63b3ed" if col.is_primary else ("#f6ad55" if col.fk_target is not None else "#e2e8f0")
            ax.scatter([x], [y], s=160, color=color, edgecolor="#4a5568", lw=0.5, zorder=3)
            ax.text(x, y - 0.22, col.orig_name, ha="center", va="top", fontsize=6.5, zorder=4)
        else:
            ax.text(x, y, graph.question_tokens[i], ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="#f0fff4", ec="#2f855a"), zorder=4)
    handles = [
        plt.Line2D([], [], color=EDGE_COLORS["fk"], lw=2.2, label="foreign key"),
        plt.Line2D([], [], color=EDGE_COLORS["primary_key"], lw=1, label="primary key of"),
        plt.Line2D([], [], color=EDGE_COLORS["belongs_to"], lw=1, label="belongs to"),
    ]
    if nq:
        handles += [plt.Line2D([], [], color=EDGE_COLORS["qc"], lw=1, label="question-column link"), plt.Line2D([], [], color=EDGE_COLORS["qt"], lw=1, label="question-table link")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    ax.set_title(title or f"Relational schema graph: {schema.db_id}")
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_relation_matrix(m: RelationMatrix, labels: list[str], path: str | Path | None = None, title: str = "Relation matrix", max_items: int = 60):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = min(m.size, max_items)
    ids = m.ids[:n, :n]
    uniq = sorted(np.unique(ids).tolist())
    remap = {u: i for i, u in enumerate(uniq)}
    dense = np.vectorize(remap.get)(ids)
    cmap = ListedColormap(plt.get_cmap("tab20").colors[: len(uniq)] if len(uniq) <= 20 else plt.get_cmap("nipy_spectral")(np.linspace(0, 1, len(uniq))))
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(dense, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels[:n], rotation=90, fontsize=6)
    ax.set_yticklabels(labels[:n], fontsize=6)
    for boundary in (m.n_q, m.n_q + m.n_c):
        if boundary < n:
            ax.axhline(boundary - 0.5, color="black", lw=1)
            ax.axvline(boundary - 0.5, color="black", lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(remap[u])) for u in uniq]
    ax.legend(handles, [m.vocab.name(u) for u in uniq], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def element_labels(schema, question_tokens: list[str]) -> list[str]:
    return [f"Q:{w}" for w in question_tokens] + [f"C:{schema.qualified_name(c.id)}" for c in schema.columns] + [f"T:{t.orig_name}" for t in schema.tables]
