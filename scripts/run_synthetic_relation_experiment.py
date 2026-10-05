"""Synthetic experiment: vanilla vs relation-aware attention (see ratsql/analysis/synthetic_relations.py).

    python scripts/run_synthetic_relation_experiment.py            # 3 seeds x 2 tasks x 2 attention types (CPU, a few minutes)

Writes results/tables/synthetic_relation_experiment.{csv,md}, results/metrics/synthetic_relation_experiment.json
and results/plots/synthetic_relation_experiment.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ratsql.analysis.synthetic_relations import run  # noqa: E402
from ratsql.utils.io import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    results = []
    for task in ("one_hop", "two_hop"):
        for use_rel in (False, True):
            for seed in args.seeds:
                r = run(task=task, use_relations=use_rel, seed=seed, steps=args.steps, device=args.device)
                print(f"{task:8s} {r['attention']:15s} seed={seed} acc={r['final_test_accuracy']:.3f} ({r['seconds']}s)")
                results.append(r)
    write_json(results, ROOT / "results" / "metrics" / "synthetic_relation_experiment.json")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in results])
    summary = df.groupby(["task", "attention"]).agg(mean_test_accuracy=("final_test_accuracy", "mean"), std_test_accuracy=("final_test_accuracy", "std"), seeds=("seed", "count"), chance=("chance", "first"), parameters=("parameters", "first")).reset_index()
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROOT / "results" / "tables" / "synthetic_relation_experiment.csv", index=False)
    with open(ROOT / "results" / "tables" / "synthetic_relation_experiment.md", "w", encoding="utf-8") as f:
        f.write("| Task | Attention | Test accuracy (mean ± std over seeds) | Chance | Params |\n|---|---|---|---|---|\n")
        for _, r in summary.iterrows():
            f.write(f"| {r.task} | {r.attention} | {r.mean_test_accuracy:.3f} ± {r.std_test_accuracy:.3f} | {r.chance:.3f} | {r.parameters:,} |\n")
    print(summary.to_string(index=False))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colors = {"vanilla": "#8a8f98", "relation-aware": "#2b6cb0"}
    for ax, task in zip(axes, ("one_hop", "two_hop")):
        for r in results:
            if r["task"] != task:
                continue
            xs = [c["step"] for c in r["curve"]]
            ys = [c["test_accuracy"] for c in r["curve"]]
            ax.plot(xs, ys, color=colors[r["attention"]], alpha=0.8, lw=1.5, label=r["attention"] if r["seed"] == args.seeds[0] else None)
        ax.axhline(results[0]["chance"], color="#c05621", ls="--", lw=1, label="chance")
        ax.set_title(f"{task.replace('_', '-')} relational lookup")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (fresh graphs)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    (ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "plots" / "synthetic_relation_experiment.png", dpi=150)


if __name__ == "__main__":
    main()
