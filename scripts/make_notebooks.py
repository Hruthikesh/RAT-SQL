"""Generate the analysis notebooks (then execute them with nbconvert).

    python scripts/make_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """%matplotlib inline
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
pd.set_option("display.max_colwidth", 200)"""


def notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return nb


def schema_graph_nb():
    return notebook([
        ("md", "# Relational schema graphs\n\nBuilds the explicit schema graph (tables, columns, primary / foreign keys) and the relation matrix fed to the relation-aware encoder, for a synthetic example and a Spider dev example."),
        ("code", SETUP),
        ("code", """from ratsql.schema.schema import load_schemas
from ratsql.schema.graph import SchemaGraph
from ratsql.schema.relations import RelationMatrixBuilder
from ratsql.schema.linking import SchemaLinks, describe_links
from ratsql.schema.visualize import plot_schema_graph, plot_relation_matrix, element_labels
from ratsql.utils.io import read_jsonl"""),
        ("md", "## Synthetic running example: *Which employees work in Hyderabad?*"),
        ("code", """schemas = load_schemas(ROOT / "data/samples/synthetic/tables.json")
recs = read_jsonl(ROOT / "data/processed/synthetic/train.jsonl")
ex = next(r for r in recs if r["question"] == "Which employees work in Hyderabad?")
schema = schemas[ex["db_id"]]
print(schema.summary())
links = SchemaLinks.from_json(ex["links"])
print("\\n".join(describe_links(ex["question_tokens"], schema, links)))
g = SchemaGraph.from_schema(schema).add_question(ex["question_tokens"], links)
print(g.edge_counts())
plot_schema_graph(g, title=ex["question"]);"""),
        ("code", """rb = RelationMatrixBuilder({})
m = rb.build(schema, len(ex["question_tokens"]), links)
print("matrix", m.ids.shape, "relation types used:", len(m.counts()))
plot_relation_matrix(m, element_labels(schema, ex["question_tokens"]), title="Relation matrix (synthetic)");"""),
        ("md", "## Spider dev example"),
        ("code", """spider_schemas = load_schemas(next((ROOT / "data/raw").rglob("tables.json")))
dev = read_jsonl(ROOT / "data/processed/spider/dev.jsonl")
ex = next(r for r in dev if r["db_id"] == "concert_singer" and "stadium" in r["question"].lower() and "JOIN" in r["query"])
schema = spider_schemas[ex["db_id"]]
print(ex["question"], "\\n", ex["query"])
links = SchemaLinks.from_json(ex["links"])
print("\\n".join(describe_links(ex["question_tokens"], schema, links)))
g = SchemaGraph.from_schema(schema).add_question(ex["question_tokens"], links)
plot_schema_graph(g, title=ex["question"]);"""),
        ("code", """m = rb.build(schema, len(ex["question_tokens"]), links)
plot_relation_matrix(m, element_labels(schema, ex["question_tokens"]), title="Relation matrix (concert_singer)");
pd.Series(m.counts()).sort_values(ascending=False).head(20)"""),
        ("md", "## Schema-graph statistics over all Spider databases"),
        ("code", """stats = ROOT / "results/tables/schema_graph_stats.csv"
df = pd.read_csv(stats) if stats.exists() else None
df[["tables", "columns", "primary_keys", "foreign_keys", "schema_edges"]].describe().round(1) if df is not None else "run scripts/build_schema_graphs.py first\""""),
        ("code", """rel = ROOT / "results/tables/relation_type_counts.csv"
pd.read_csv(rel).head(25) if rel.exists() else None"""),
    ])


def linking_nb():
    return notebook([
        ("md", "# Schema-linking analysis\n\nHow well does the n-gram / value linker connect question phrases to the tables, columns and values that the *gold* SQL uses? (Model-independent.)"),
        ("code", SETUP),
        ("code", """from ratsql.analysis.linking_analysis import analyse_linking, gold_elements
from ratsql.schema.schema import load_schemas
from ratsql.schema.linking import SchemaLinks, describe_links
from ratsql.utils.io import read_jsonl
schemas = load_schemas(next((ROOT / "data/raw").rglob("tables.json")))
dev = read_jsonl(ROOT / "data/processed/spider/dev.jsonl")
res = analyse_linking(dev)
display(pd.Series({k: v for k, v in res.items() if not isinstance(v, dict)}).round(3))
pd.DataFrame(res["per_link_type"]).T.round(3)"""),
        ("md", "Precision per link type = fraction of (question-token, column) links of that type whose column is used by the gold query. Exact name matches are the most reliable; partial matches are frequent but noisy — which is why RAT-SQL encodes them as *relations* the model can weigh rather than hard decisions."),
        ("code", """def show(r):
    schema = schemas[r["db_id"]]
    gcols, gtabs = gold_elements(r)
    print("Q:", r["question"]); print("SQL:", r["query"])
    for line in describe_links(r["question_tokens"], schema, SchemaLinks.from_json(r["links"])):
        print("   ", line)
    print("   gold tables:", sorted(schema.tables[t].orig_name for t in gtabs))
    print("   gold columns:", sorted(schema.qualified_name(c) for c in gcols))
    print()

# successes: every gold column is linked
succ = [r for r in dev if (lambda g: g[0] and all(any(c == cc for _, cc, t in r["links"]["q_col"] if t != "NUM") for c in g[0]))(gold_elements(r))]
print(len(succ), "examples where every gold column is linked"); [show(r) for r in succ[:3]];"""),
        ("code", """# failures: gold columns with no link at all (require inference, synonyms or world knowledge)
fail = [r for r in dev if (lambda g: any(not any(c == cc for _, cc, t in r["links"]["q_col"] if t != "NUM") for c in g[0]))(gold_elements(r))]
print(len(fail), "examples with at least one unlinked gold column"); [show(r) for r in fail[:4]];"""),
        ("code", """# value linking: gold literals recovered from the database
vals = [(r["question"], r["value_stats"]["unresolved_values"]) for r in dev if r["value_stats"]["unresolved"]]
print(f"{len(vals)} dev questions have gold literals that are not recoverable as candidates, e.g.:")
for q, v in vals[:10]: print("  ", v, "<-", q)"""),
    ])


def attention_nb():
    return notebook([
        ("md", "# Attention analysis: relation-aware vs vanilla self-attention\n\nAttention *lift* of a relation type = mean attention weight on pairs of that type × number of keys (1 = uniform). Computed by `scripts/run_attention_analysis.py` over Spider dev examples for the FULL (relation-aware) model and baseline B (vanilla transformer, identical architecture without relation embeddings)."),
        ("code", SETUP),
        ("code", """lift = ROOT / "results/tables/attention_relation_lift.md"
display(Markdown(lift.read_text(encoding="utf-8")) if lift.exists() else Markdown("_run scripts/run_attention_analysis.py first_"))"""),
        ("code", """p = ROOT / "results/plots/attention_lift.png"
display(Image(filename=str(p))) if p.exists() else None"""),
        ("code", """df = pd.read_csv(ROOT / "results/tables/attention_relation_lift.csv")
sel = ["qc_em", "qc_pm", "qc_vem", "qc_default", "qt_em", "qt_default", "ct_primary_key", "ct_belongs_to", "cc_fk_forward", "cc_default", "tt_fk_forward", "tt_default"]
piv = df[df.relation.isin(sel)].pivot_table(index=["relation"], columns=["model", "layer"], values="lift").round(2)
piv"""),
        ("md", "## Per-layer trend of question → linked-column attention"),
        ("code", """fig, ax = plt.subplots(figsize=(6, 3.5))
for (m, r), g in df[df.relation.isin(["qc_em", "qc_default", "qt_em"])].groupby(["model", "relation"]):
    ax.plot(g.layer, g.lift, "o-", label=f"{m}: {r}")
ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_ylabel("lift"); ax.axhline(1, color="k", lw=0.6); ax.legend(fontsize=7, frameon=False)"""),
        ("md", "## Example heatmaps (last layer, head-averaged, question tokens × schema elements)"),
        ("code", """for name in ("rat", "vanilla"):
    p = ROOT / f"results/plots/attention_heatmap_{name}.png"
    if p.exists(): display(Image(filename=str(p)))"""),
    ])


def error_nb():
    return notebook([
        ("md", "# Error analysis\n\nEvery Spider-dev prediction of the FULL model that is not both an exact match and an execution match is labelled with the taxonomy in `ratsql/analysis/errors.py` (primary label = first in priority order). Nothing is cherry-picked: samples below are drawn uniformly at random per category with a fixed seed."),
        ("code", SETUP),
        ("code", """err = pd.read_csv(ROOT / "results/tables/error_analysis_full.csv")
print(len(err), "failures")
err.primary_error.value_counts()"""),
        ("code", """display(Markdown((ROOT / "results/tables/error_distribution.md").read_text(encoding="utf-8")))
display(Image(filename=str(ROOT / "results/plots/error_distribution.png")))"""),
        ("code", """pd.crosstab(err.primary_error, err.hardness).reindex(columns=["easy", "medium", "hard", "extra"])"""),
        ("code", """labels = err.error_labels.str.split("|").explode()
labels.value_counts().rename("any-label count")"""),
        ("md", "## Random samples per category"),
        ("code", """for cat, g in err.groupby("primary_error"):
    print("=" * 100); print(cat, f"({len(g)})")
    for _, r in g.sample(min(3, len(g)), random_state=0).iterrows():
        print("Q   :", r.question); print("GOLD:", r.gold_sql); print("PRED:", r.pred_sql); print("WHY :", r.analysis); print()"""),
        ("md", "## Accuracy by query complexity"),
        ("code", """display(Image(filename=str(ROOT / "results/plots/complexity_em.png")))
pd.read_csv(ROOT / "results/tables/complexity_results.csv").pivot_table(index=["feature", "bucket"], columns="model", values="exact_match").round(3)"""),
    ])


def main() -> None:
    NB.mkdir(exist_ok=True)
    for name, nb in [
        ("schema_graph_visualization", schema_graph_nb()),
        ("schema_linking_analysis", linking_nb()),
        ("attention_analysis", attention_nb()),
        ("error_analysis", error_nb()),
    ]:
        nbf.write(nb, NB / f"{name}.ipynb")
        print("wrote", NB / f"{name}.ipynb")


if __name__ == "__main__":
    main()
