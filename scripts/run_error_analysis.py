"""Error taxonomy, complexity analysis and schema-linking analysis of predictions.

    python scripts/run_error_analysis.py \
        --predictions full=experiments/full_model/full/dev_predictions.jsonl \
                      vanilla=experiments/baseline/baseline_b_vanilla_transformer/dev_predictions.jsonl \
        --primary full --config configs/small.yaml

Outputs (results/):
  tables/error_analysis_<primary>.csv        every failure: question, gold, predicted, labels, analysis
  tables/error_distribution.{csv,md}         primary error category counts per model
  tables/complexity_results.{csv,md}         EM / EX by complexity bucket per model
  tables/clause_accuracy.{csv,md}            per-clause accuracy (SELECT/WHERE/JOIN/GROUP/HAVING/ORDER/nested/agg)
  tables/linking_analysis.json               linker precision / recall vs gold SQL
  plots/error_distribution.png, plots/complexity_em.png, plots/complexity_ex.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ratsql.analysis.clauses import CLAUSES, clause_accuracy  # noqa: E402
from ratsql.analysis.complexity_analysis import complexity_table, plot_complexity  # noqa: E402
from ratsql.analysis.errors import PRIORITY, classify_error  # noqa: E402
from ratsql.analysis.linking_analysis import analyse_linking  # noqa: E402
from ratsql.experiment import Experiment  # noqa: E402
from ratsql.sql.parser import canonical, to_tuples  # noqa: E402
from ratsql.utils.config import load_config  # noqa: E402
from ratsql.utils.io import read_jsonl, write_json  # noqa: E402

TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"


def analyse_model(rows: list[dict], recs: dict, schemas: dict) -> list[dict]:
    out = []
    for r in rows:
        if r.get("exact_match") and r.get("exec_match"):
            continue
        rec = recs[r["id"]]
        gold = canonical(to_tuples(rec["sql"]))
        pred = canonical(to_tuples(r["pred_parsed"])) if r.get("pred_parsed") else None
        res = classify_error(r, gold, pred, rec["links"], schemas[r["db_id"]])
        out.append(
            {
                "id": r["id"],
                "db_id": r["db_id"],
                "hardness": r.get("hardness"),
                "question": r["question"],
                "gold_sql": r["gold_sql"],
                "pred_sql": r["pred_sql"],
                "exact_match": r.get("exact_match"),
                "exec_match": r.get("exec_match"),
                "exec_error": r.get("exec_error"),
                "primary_error": res["primary"],
                "error_labels": "|".join(res["labels"]),
                "analysis": res["analysis"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True, help="name=path/to/dev_predictions.jsonl")
    ap.add_argument("--primary", default=None, help="model whose failures are exported in full")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    exp = Experiment(cfg)
    recs_list = read_jsonl(exp.processed_dir / f"{args.split}.jsonl")
    recs = {r["id"]: r for r in recs_list}
    schemas = exp.schemas(args.split)
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    preds = dict(p.split("=", 1) for p in args.predictions)
    primary = args.primary or next(iter(preds))
    dist_rows, comp_frames, clause_rows = [], [], []
    golds = {rid: canonical(to_tuples(r["sql"])) for rid, r in recs.items() if r.get("sql")}
    for name, path in preds.items():
        rows = read_jsonl(ROOT / path if not Path(path).is_absolute() else path)
        errs = analyse_model(rows, recs, schemas)
        for clause, v in clause_accuracy(rows, golds, schemas).items():
            clause_rows.append({"model": name, "clause": clause, "n": v["n"], "accuracy": v["accuracy"]})
        n = len(rows)
        if name == primary:
            pd.DataFrame(errs).to_csv(TABLES / f"error_analysis_{name}.csv", index=False)
        c = Counter(e["primary_error"] for e in errs)
        all_labels = Counter(lab for e in errs for lab in e["error_labels"].split("|"))
        for cat in PRIORITY:
            dist_rows.append({"model": name, "category": cat, "primary_count": c.get(cat, 0), "primary_share_of_failures": c.get(cat, 0) / max(1, len(errs)), "any_label_count": all_labels.get(cat, 0), "failures": len(errs), "examples": n})
        comp_frames.append(complexity_table(rows, name))
        print(f"{name}: {len(errs)} failures / {n}; primary categories: {dict(c.most_common())}")
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(TABLES / "error_distribution.csv", index=False)
    piv = dist.pivot(index="category", columns="model", values="primary_count").reindex(PRIORITY)
    lines = ["### Primary error category (Spider dev failures, our measurements)", "", "| Category | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
    for cat, r in piv.iterrows():
        lines.append(f"| {cat} | " + " | ".join(str(int(v)) for v in r.values) + " |")
    fails = dist.groupby("model")["failures"].first()
    lines.append("| **total failures** | " + " | ".join(str(int(fails[m])) for m in piv.columns) + " |")
    (TABLES / "error_distribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cl = pd.DataFrame(clause_rows)
    cl.to_csv(TABLES / "clause_accuracy.csv", index=False)
    lines = ["### Clause-level accuracy (Spider dev, gold queries containing the clause; our measurements)", "", "| Clause | n | " + " | ".join(preds) + " |", "|---|---|" + "---|" * len(preds)]
    for clause in CLAUSES:
        g = cl[cl["clause"] == clause]
        cells = [f"{100 * g[g['model'] == m]['accuracy'].iloc[0]:.1f}" if len(g[g['model'] == m]) and g[g['model'] == m]['accuracy'].iloc[0] is not None else "–" for m in preds]
        lines.append(f"| {clause} | {int(g['n'].iloc[0])} | " + " | ".join(cells) + " |")
    (TABLES / "clause_accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    comp = pd.concat(comp_frames)
    comp.to_csv(TABLES / "complexity_results.csv", index=False)
    lines = ["### Accuracy by query complexity (Spider dev, our measurements)", "", "| Feature | Bucket | n | " + " | ".join(f"{m} EM | {m} EX" for m in preds) + " |", "|---|---|---|" + "---|---|" * len(preds)]
    for (feat, b), g in comp.groupby(["feature", "bucket"], sort=False):
        cells = []
        for m in preds:
            s = g[g["model"] == m]
            cells += [f"{100 * s['exact_match'].iloc[0]:.1f}", f"{100 * s['execution_accuracy'].iloc[0]:.1f}"] if len(s) else ["–", "–"]
        lines.append(f"| {feat} | {b} | {int(g['n'].iloc[0])} | " + " | ".join(cells) + " |")
    (TABLES / "complexity_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_complexity(comp, PLOTS / "complexity_em.png", "exact_match")
    plot_complexity(comp, PLOTS / "complexity_ex.png", "execution_accuracy")
    _plot_errors(dist)
    link = analyse_linking(recs_list)
    write_json(link, TABLES / "linking_analysis.json")
    print("linking:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in link.items() if k not in ("counts", "per_link_type")}))


def _plot_errors(dist: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(dict.fromkeys(dist["model"]))
    cats = PRIORITY
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(cats))
    w = 0.8 / max(1, len(models))
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#718096"]
    for i, m in enumerate(models):
        s = dist[dist["model"] == m].set_index("category").reindex(cats)
        ax.bar(x + i * w - 0.4 + w / 2, s["primary_count"], w, label=m, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("failures (primary category)")
    ax.set_title("Error taxonomy on Spider dev")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "error_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
