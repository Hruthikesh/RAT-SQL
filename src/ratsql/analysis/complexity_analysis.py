"""Accuracy as a function of gold-query complexity."""

from __future__ import annotations

import pandas as pd

FEATURES = [
    ("num_tables", "Tables", (1, 2, 3)),
    ("num_joins", "Joins", (0, 1, 2)),
    ("num_conditions", "Conditions (WHERE+HAVING)", (0, 1, 2)),
    ("has_aggregation", "Aggregation", None),
    ("has_group_by", "GROUP BY", None),
    ("has_order_by", "ORDER BY", None),
    ("has_nested", "Nested sub-query", None),
    ("has_set_op", "Set operation", None),
]


def _bucket(v, edges):
    if edges is None:
        return "yes" if v else "no"
    for e in edges:
        if v <= e:
            return str(e)
    return f"{edges[-1] + 1}+"


def complexity_table(rows: list[dict], model: str) -> pd.DataFrame:
    out = []
    for key, label, edges in FEATURES:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            c = r.get("complexity") or {}
            groups.setdefault(_bucket(c.get(key, 0), edges), []).append(r)
        order = sorted(groups, key=lambda b: (b not in ("no", "yes"), b)) if edges is None else sorted(groups, key=lambda b: int(b.rstrip("+")))
        for b in order:
            g = groups[b]
            out.append(
                {
                    "model": model,
                    "feature": label,
                    "bucket": b,
                    "n": len(g),
                    "exact_match": sum(bool(r.get("exact_match")) for r in g) / len(g),
                    "execution_accuracy": sum(bool(r.get("exec_match")) for r in g) / len(g),
                }
            )
    return pd.DataFrame(out)


def plot_complexity(df: pd.DataFrame, path, metric: str = "exact_match") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    feats = list(dict.fromkeys(df["feature"]))
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#718096", "#b7791f"]
    for ax, feat in zip(axes.ravel(), feats):
        sub = df[df["feature"] == feat]
        buckets = list(dict.fromkeys(sub["bucket"]))
        x = np.arange(len(buckets))
        w = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            s = sub[sub["model"] == m].set_index("bucket").reindex(buckets)
            ax.bar(x + i * w - 0.4 + w / 2, 100 * s[metric].fillna(0), w, label=m, color=palette[i % len(palette)])
        ns = sub.groupby("bucket")["n"].first().reindex(buckets)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(buckets, ns)], fontsize=8)
        ax.set_title(feat, fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel(f"{metric.replace('_', ' ')} (%)")
    axes[1, 0].set_ylabel(f"{metric.replace('_', ' ')} (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Spider dev {metric.replace('_', ' ')} by query complexity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
