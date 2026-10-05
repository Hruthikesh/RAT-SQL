"""Compare relation-aware vs vanilla encoder attention by relation type.

    python scripts/run_attention_analysis.py \
        --models rat=experiments/full_model/full/best.pt vanilla=experiments/baseline/baseline_b_vanilla_transformer/best.pt \
        --split dev --limit 300

Outputs:
  results/tables/attention_relation_lift.csv      lift per (model, layer, relation)
  results/tables/attention_relation_lift.md       selected relations, last layer
  results/plots/attention_lift.png                bar chart, RAT vs vanilla
  results/plots/attention_heatmap_<model>.png     question x schema attention for one example
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ratsql.analysis.attention import example_heatmap, relation_lift  # noqa: E402
from ratsql.schema.relations import RelationMatrixBuilder  # noqa: E402
from ratsql.training.run import load_model_from_checkpoint  # noqa: E402

SELECTED = [
    ("qq_dist_0", "question -> itself"),
    ("qq_dist_1", "question -> next token"),
    ("qc_em", "question -> exactly-matched column"),
    ("qc_pm", "question -> partially-matched column"),
    ("qc_vem", "question -> value-matched column"),
    ("qc_default", "question -> unlinked column"),
    ("qt_em", "question -> exactly-matched table"),
    ("qt_default", "question -> unlinked table"),
    ("ct_primary_key", "column -> its table (PK)"),
    ("ct_belongs_to", "column -> its table"),
    ("ct_default", "column -> other table"),
    ("cc_fk_forward", "column -> FK target column"),
    ("cc_same_table", "column -> same-table column"),
    ("cc_default", "column -> unrelated column"),
    ("tt_fk_forward", "table -> FK-linked table"),
    ("tt_default", "table -> unrelated table"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--example", type=int, default=None, help="index of the example for heatmaps")
    args = ap.parse_args()
    rows = []
    heat = {}
    vocab = RelationMatrixBuilder({}).vocab  # full vocabulary used for the analysis
    for spec in args.models:
        name, ckpt = spec.split("=", 1)
        model, exp, cfg, device = load_model_from_checkpoint(ROOT / ckpt if not Path(ckpt).is_absolute() else ckpt)
        feats = exp.features(args.split, limit=args.limit)  # the model's own relation matrices
        full_builder = RelationMatrixBuilder({})  # grouping always uses the full vocabulary
        schemas = exp.schemas(args.split)

        def analysis_relations(f, _b=full_builder, _s=schemas):
            from ratsql.schema.linking import SchemaLinks

            return _b.build(_s[f.rec["db_id"]], f.n_q, SchemaLinks.from_json(f.rec["links"])).ids

        res = relation_lift(model, feats, device, len(vocab), analysis_relations=analysis_relations)
        for layer in range(res["lift"].shape[0]):
            for rid, rname in enumerate(vocab.names):
                if res["counts"][rid] > 0:
                    rows.append({"model": name, "layer": layer + 1, "relation": rname, "pairs": int(res["counts"][rid]), "lift": float(res["lift"][layer, rid])})
        idx = args.example
        if idx is None:  # a medium-sized example with an exact column match
            idx = next((i for i, f in enumerate(feats) if 8 <= f.n_q <= 14 and f.n_c <= 20 and any(t == "EM" for _, _, t in f.rec["links"]["q_col"])), 0)
        f = feats[idx]
        heat[name] = (example_heatmap(model, f, device), f)
        print(f"{name}: analysed {len(feats)} examples, {res['lift'].shape[0]} layers")
    out_t = ROOT / "results" / "tables"
    out_p = ROOT / "results" / "plots"
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_t / "attention_relation_lift.csv", index=False)
    if df.empty:
        return
    last = df[df["layer"] == df.groupby("model")["layer"].transform("max")]
    first = df[df["layer"] == 1]
    models = list(dict.fromkeys(df["model"]))
    lines = ["### Attention lift by relation type (1 = uniform attention; head-averaged; our measurements)", "", "| Relation | Meaning | " + " | ".join(f"{m} L1 | {m} last" for m in models) + " |", "|---|---|" + "---|---|" * len(models)]
    for rname, meaning in SELECTED:
        cells = []
        for m in models:
            for d in (first, last):
                v = d[(d["model"] == m) & (d["relation"] == rname)]["lift"]
                cells.append(f"{v.iloc[0]:.2f}" if len(v) else "–")
        lines.append(f"| {rname} | {meaning} | " + " | ".join(cells) + " |")
    (out_t / "attention_relation_lift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(SELECTED))
    w = 0.8 / len(models)
    colors = ["#2b6cb0", "#8a8f98", "#2f855a", "#c05621"]
    for i, m in enumerate(models):
        vals = []
        for rname, _ in SELECTED:
            v = last[(last["model"] == m) & (last["relation"] == rname)]["lift"]
            vals.append(v.iloc[0] if len(v) else 0)
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"{m} (last layer)", color=colors[i % len(colors)])
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r for r, _ in SELECTED], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("attention lift (log)")
    ax.set_title("Encoder attention by relation type: relation-aware vs vanilla")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_p / "attention_lift.png", dpi=150)
    plt.close(fig)
    for name, (h, f) in heat.items():
        schema = exp.schemas(args.split)[f.rec["db_id"]]
        labels = [schema.qualified_name(c.id) for c in schema.columns] + [f"[T] {t.orig_name}" for t in schema.tables]
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(labels)), max(3, 0.35 * f.n_q)))
        ax.imshow(h, aspect="auto", cmap="Blues")
        ax.set_yticks(range(f.n_q))
        ax.set_yticklabels(f.rec["question_tokens"], fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(f"{name}: last-layer question->schema attention\n{f.rec['question']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_p / f"attention_heatmap_{name}.png", dpi=150)
        plt.close(fig)
    print(open(out_t / "attention_relation_lift.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
