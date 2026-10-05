"""Build relational schema graphs for every database, report statistics and draw examples.

    python scripts/build_schema_graphs.py --config configs/small.yaml
    python scripts/build_schema_graphs.py --config configs/debug.yaml --example "Which employees work in Hyderabad?"

Outputs:
  results/tables/schema_graph_stats.csv       per-database node / edge / relation counts
  results/tables/relation_type_counts.csv     relation-type frequencies over the dev set
  results/plots/schema_graph_<db>.png          schema graphs (with question links for the example)
  results/plots/relation_matrix_<db>.png       relation matrix of one example
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from ratsql.experiment import Experiment  # noqa: E402
from ratsql.schema.graph import SchemaGraph  # noqa: E402
from ratsql.schema.linking import SchemaLinks, describe_links  # noqa: E402
from ratsql.schema.visualize import element_labels, plot_relation_matrix, plot_schema_graph  # noqa: E402
from ratsql.utils.config import load_config  # noqa: E402
from ratsql.utils.io import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--example", default=None, help="question text of an example to visualise")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Experiment(cfg)
    schemas = exp.schemas(args.split)
    rows = []
    for db_id, s in sorted(schemas.items()):
        g = SchemaGraph.from_schema(s)
        counts = g.edge_counts()
        rows.append(
            {
                "db_id": db_id,
                "tables": s.num_tables,
                "columns": s.num_columns - 1,
                "primary_keys": sum(c.is_primary for c in s.columns),
                "foreign_keys": len(s.foreign_keys),
                "schema_nodes": len(g.nodes),
                "schema_edges": len(g.edges),
                **{f"edges_{k}": v for k, v in counts.items()},
            }
        )
    out_tables = ROOT / "results" / "tables"
    out_plots = ROOT / "results" / "plots"
    out_tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(out_tables / f"{args.out_prefix}schema_graph_stats.csv", index=False)
    print(df[["tables", "columns", "foreign_keys", "schema_edges"]].describe().round(2).to_string())

    recs = read_jsonl(exp.processed_dir / f"{args.split}.jsonl")
    rel_counter: Counter = Counter()
    for r in recs:
        m = exp.relation_builder.build(schemas[r["db_id"]], len(r["question_tokens"]), SchemaLinks.from_json(r["links"]))
        rel_counter.update(m.counts())
    pd.DataFrame(sorted(rel_counter.items(), key=lambda x: -x[1]), columns=["relation", "count"]).to_csv(out_tables / f"{args.out_prefix}relation_type_counts.csv", index=False)

    ex = next((r for r in recs if args.example and r["question"] == args.example), recs[0])
    schema = schemas[ex["db_id"]]
    links = SchemaLinks.from_json(ex["links"])
    g = SchemaGraph.from_schema(schema).add_question(ex["question_tokens"], links)
    print(f"\nExample: {ex['question']}\nGold: {ex['query']}")
    for line in describe_links(ex["question_tokens"], schema, links):
        print("  " + line)
    plot_schema_graph(g, out_plots / f"{args.out_prefix}schema_graph_{schema.db_id}.png", title=f"{schema.db_id}: \"{ex['question']}\"")
    m = exp.relation_builder.build(schema, len(ex["question_tokens"]), links)
    plot_relation_matrix(m, element_labels(schema, ex["question_tokens"]), out_plots / f"{args.out_prefix}relation_matrix_{schema.db_id}.png", title=f"Relation matrix ({schema.db_id})")
    print(f"plots written to {out_plots}")


if __name__ == "__main__":
    main()
