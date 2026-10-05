"""Aggregate measured metrics from experiments/ into result tables and plots.

Only values found in experiment output files (dev_metrics.json, history.json,
train_summary.json) are used -- nothing is typed in by hand.

Outputs (results/):
  tables/main_results.{csv,md}        baselines vs full model
  tables/ablation_results.{csv,md}    controlled ablations A-K
  tables/scaling_results.{csv,md}     data-scaling experiment
  tables/hardness_results.{csv,md}    EM / EX by Spider difficulty
  tables/component_results.csv        component-level F1 (official aggregation)
  plots/training_curves.png           validation EM / training loss per epoch
  plots/data_scaling.png              training-set size vs EM / EX
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_ablation import experiment_dir  # noqa: E402

TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"
FINAL_DIR = ROOT / "experiments" / "full_model" / "final"


def _load(p: Path):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def row_for(name: str, label: str, d: Path, split: str = "dev") -> dict | None:
    m = _load(d / f"{split}_metrics.json")
    if m is None:
        return None
    s = _load(d / "train_summary.json") or {}
    info = _load(d / "run_info.json") or {}
    params = m.get("parameters", {})
    return {
        "experiment": name,
        "description": label,
        "split": split,
        "n": m["n"],
        "exact_match": m["exact_match"],
        "execution_accuracy": m.get("execution_accuracy"),
        "sql_validity": m.get("sql_validity"),
        "grammar_validity": m.get("grammar_validity"),
        "parameters": params.get("total"),
        "trainable_parameters": params.get("trainable"),
        "train_examples": s.get("train_examples", info.get("train_examples")),
        "epochs_run": s.get("epochs_run"),
        "best_epoch": s.get("best_epoch"),
        "val_em_best": s.get("best_metric"),
        "train_minutes": round(s.get("train_seconds", 0) / 60, 1) if s.get("train_seconds") else None,
        "inference_ms_per_example": m.get("ms_per_example"),
    }


def _per_example(d: Path, key: str, split: str = "dev") -> dict[str, bool] | None:
    p = d / f"{split}_predictions.jsonl"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return {json.loads(l)["id"]: bool(json.loads(l).get(key)) for l in f if l.strip()}


def paired_bootstrap(a: dict[str, bool], b: dict[str, bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference (b - a) and its 95 % percentile CI over examples (paired resampling)."""
    import numpy as np

    ids = sorted(set(a) & set(b))
    diff = np.array([float(b[i]) - float(a[i]) for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boots = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_ci(a: dict[str, bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    import numpy as np

    v = np.array([float(x) for x in a.values()])
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{100 * x:.1f}"


def write_md(df: pd.DataFrame, path: Path, cols: list[tuple[str, str]], title: str, note: str = "") -> None:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(h for _, h in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if key in ("exact_match", "execution_accuracy", "sql_validity", "grammar_validity", "val_em_best"):
                cells.append(pct(v))
            elif key in ("parameters", "trainable_parameters", "train_examples") and v is not None and not pd.isna(v):
                cells.append(f"{int(v):,}")
            else:
                cells.append("–" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "configs" / "experiments.yaml", "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    exps = {e["name"]: e for e in reg["experiments"]}
    rows = {}
    for name, e in exps.items():
        r = row_for(name, e["label"], experiment_dir(e))
        if r is not None:
            r["group"] = e["group"]
            r["ablation_id"] = e.get("ablation_id")
            rows[name] = r
    common_note = (
        "All numbers are measured by this repository on the Spider **dev** set (1,034 examples, 20 unseen databases) unless a Split column says test (2,147 examples, 40 unseen databases). "
        "Controlled setting (all rows except Final): BERT-small encoder, identical training budget and data; checkpoints selected on "
        "held-out *training* databases. SQL validity = fraction of predictions that execute without error."
    )
    # ---------------------------------------------------------------- main
    main_order = ["baseline_a_simple", "baseline_b_vanilla_transformer", "baseline_c_no_linking", "full"]
    main_rows = [rows[n] for n in main_order if n in rows]
    final = row_for("final", "Final: FULL with BERT-medium, 40 epochs", FINAL_DIR)
    if final:
        final["group"] = "final"
        main_rows.append(final)
        t = row_for("final", "Final: FULL with BERT-medium, 40 epochs", FINAL_DIR, split="test")
        if t:
            t["group"] = "final"
            main_rows.append(t)
    if main_rows:
        for r in main_rows:
            d = FINAL_DIR if r["experiment"] == "final" else experiment_dir(exps[r["experiment"]])
            for key, col in (("exact_match", "em_ci95"), ("exec_match", "ex_ci95")):
                pe = _per_example(d, key, r["split"])
                if pe:
                    lo, hi = bootstrap_ci(pe)
                    r[col] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        df = pd.DataFrame(main_rows)
        df.to_csv(TABLES / "main_results.csv", index=False)
        write_md(
            df,
            TABLES / "main_results.md",
            [("description", "Model"), ("split", "Split"), ("exact_match", "Exact Match (%)"), ("em_ci95", "EM 95% CI"), ("execution_accuracy", "Execution Accuracy (%)"), ("ex_ci95", "EX 95% CI"), ("sql_validity", "SQL Validity (%)"), ("parameters", "Params"), ("train_minutes", "Train (min)")],
            "Main results (our measurements)",
            common_note,
        )
    # ---------------------------------------------------------------- ablations
    abl = [r for r in rows.values() if r.get("ablation_id") and r["group"] != "scaling"]
    scale10 = rows.get("scaling_10pct")
    if scale10:
        abl.append(dict(scale10, ablation_id="J", description="Reduced training data (10%)"))
    if abl:
        df = pd.DataFrame(sorted(abl, key=lambda r: r["ablation_id"]))
        full_em = rows["full"]["exact_match"] if "full" in rows else None
        full_ex = rows["full"]["execution_accuracy"] if "full" in rows else None
        df["delta_em"] = df["exact_match"].apply(lambda x: None if full_em is None else round(100 * (x - full_em), 1))
        df["delta_ex"] = df["execution_accuracy"].apply(lambda x: None if full_ex is None or x is None else round(100 * (x - full_ex), 1))
        # paired bootstrap 95 % CIs of the differences to FULL (dev-set sampling noise only)
        full_dir = experiment_dir(exps["full"]) if "full" in exps else None
        ci_em, ci_ex = [], []
        for _, r in df.iterrows():
            d = experiment_dir(exps[r["experiment"]])
            if full_dir is None or r["experiment"] == "full":
                ci_em.append("–")
                ci_ex.append("–")
                continue
            for key, store in (("exact_match", ci_em), ("exec_match", ci_ex)):
                a, b = _per_example(full_dir, key), _per_example(d, key)
                if a and b:
                    _, lo, hi = paired_bootstrap(a, b)
                    store.append(f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")
                else:
                    store.append("–")
        df["delta_em_ci95"] = ci_em
        df["delta_ex_ci95"] = ci_ex
        cols = ["ablation_id", "experiment", "description", "exact_match", "execution_accuracy", "sql_validity", "grammar_validity", "delta_em", "delta_em_ci95", "delta_ex", "delta_ex_ci95", "parameters", "train_minutes", "inference_ms_per_example", "best_epoch", "epochs_run"]
        df[cols].to_csv(TABLES / "ablation_results.csv", index=False)
        write_md(
            df,
            TABLES / "ablation_results.md",
            [("ablation_id", "ID"), ("description", "Experiment"), ("exact_match", "EM (%)"), ("execution_accuracy", "EX (%)"), ("sql_validity", "SQL valid (%)"), ("grammar_validity", "Well-formed AST (%)"), ("delta_em", "ΔEM (pts)"), ("delta_em_ci95", "ΔEM 95% CI"), ("delta_ex", "ΔEX (pts)"), ("delta_ex_ci95", "ΔEX 95% CI"), ("parameters", "Params"), ("train_minutes", "Train (min)"), ("inference_ms_per_example", "Infer (ms/ex)")],
            "Ablations (our measurements)",
            common_note + " ΔEM/ΔEX are differences to the FULL row; 95 % CIs from a paired bootstrap over dev examples (2,000 resamples) capture dev-set sampling noise but not training-seed variance.",
        )
    # ---------------------------------------------------------------- scaling
    sc = [r for r in rows.values() if r["group"] == "scaling"]
    if "full" in rows:
        sc.append(dict(rows["full"], description="100% of training data"))
    if sc:
        df = pd.DataFrame(sc).sort_values("train_examples")
        df[["experiment", "description", "train_examples", "exact_match", "execution_accuracy", "sql_validity", "epochs_run", "best_epoch", "train_minutes"]].to_csv(TABLES / "scaling_results.csv", index=False)
        write_md(df, TABLES / "scaling_results.md", [("description", "Training data"), ("train_examples", "Examples"), ("exact_match", "EM (%)"), ("execution_accuracy", "EX (%)"), ("sql_validity", "SQL valid (%)"), ("epochs_run", "Epochs"), ("train_minutes", "Train (min)")], "Data scaling (our measurements)", common_note)
        _plot_scaling(df)
    # ---------------------------------------------------------------- hardness / components
    hrows, crows = [], []
    for name, r in list(rows.items()) + ([("final", final)] if final else []):
        d = FINAL_DIR if name == "final" else experiment_dir(exps[name])
        m = _load(d / "dev_metrics.json")
        for h, v in m.get("by_hardness", {}).items():
            hrows.append({"experiment": name, "hardness": h, "n": v["n"], "exact_match": v["exact_match"], "execution_accuracy": v.get("execution_accuracy")})
        for c, v in m.get("components", {}).items():
            crows.append({"experiment": name, "component": c, **v})
    if hrows:
        hd = pd.DataFrame(hrows)
        hd.to_csv(TABLES / "hardness_results.csv", index=False)
        piv = hd.pivot_table(index="experiment", columns="hardness", values=["exact_match", "execution_accuracy"])
        lines = ["### EM / EX by Spider difficulty (dev, our measurements)", "", "| Experiment | " + " | ".join(f"{h} EM" for h in ("easy", "medium", "hard", "extra")) + " | " + " | ".join(f"{h} EX" for h in ("easy", "medium", "hard", "extra")) + " |", "|" + "---|" * 9]
        for exp_name, r in piv.iterrows():
            ems = [pct(r.get(("exact_match", h))) for h in ("easy", "medium", "hard", "extra")]
            exs = [pct(r.get(("execution_accuracy", h))) for h in ("easy", "medium", "hard", "extra")]
            lines.append(f"| {exp_name} | " + " | ".join(ems) + " | " + " | ".join(exs) + " |")
        n_by = hd.groupby("hardness")["n"].first().to_dict()
        lines += ["", "Examples per level: " + ", ".join(f"{h}={n_by.get(h)}" for h in ("easy", "medium", "hard", "extra"))]
        (TABLES / "hardness_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if crows:
        pd.DataFrame(crows).to_csv(TABLES / "component_results.csv", index=False)
    # ---------------------------------------------------------------- compute / inference cost
    comp_rows = []
    runs = [(n, exps[n]["label"], experiment_dir(exps[n])) for n in rows if exps[n]["group"] != "scaling"]
    if (FINAL_DIR / "dev_metrics.json").exists():
        runs.append(("final", "Final: FULL with BERT-medium, 40 epochs", FINAL_DIR))
    for name, label, d in runs:
        m, s, h = _load(d / "dev_metrics.json"), _load(d / "train_summary.json") or {}, _load(d / "history.json") or []
        if not m or not h:
            continue
        ep = [r["epoch_seconds"] for r in h if r.get("epoch_seconds")]
        n_train = s.get("train_examples") or 0
        beam = _load(d / "dev_beam5_metrics.json")
        comp_rows.append(
            {
                "experiment": name,
                "description": label,
                "parameters": (m.get("parameters") or {}).get("total"),
                "epochs": s.get("epochs_run"),
                "optimizer_steps": s.get("steps"),
                "train_minutes": round(s.get("train_seconds", 0) / 60, 1),
                "median_epoch_seconds": round(float(sorted(ep)[len(ep) // 2]), 1) if ep else None,
                "train_examples_per_second": round(n_train / (sorted(ep)[len(ep) // 2]), 1) if ep and n_train else None,
                "greedy_ms_per_example": m.get("ms_per_example"),
                "beam5_ms_per_example": beam.get("ms_per_example") if beam else None,
            }
        )
    if comp_rows:
        cdf = pd.DataFrame(comp_rows)
        cdf.to_csv(TABLES / "compute_results.csv", index=False)
        lines = [
            "### Computational cost (RTX 3050 Laptop 6 GB, our measurements)", "",
            "Wall-clock numbers on a shared laptop (they vary by roughly ±30 % with machine load). Greedy decoding is batched "
            "(batch 64) over the 1,034 dev examples and includes the encoder; beam search runs one example at a time.", "",
            "| Experiment | Params | Epochs | Optimizer steps | Train (min) | Median epoch (s) | Train ex/s | Greedy ms/ex | Beam-5 ms/ex |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in cdf.iterrows():
            def f(v):
                return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{int(v):,}" if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer() and v > 1000) else str(v))
            lines.append(f"| {r['description']} | {f(r['parameters'])} | {f(r['epochs'])} | {f(r['optimizer_steps'])} | {f(r['train_minutes'])} | {f(r['median_epoch_seconds'])} | {f(r['train_examples_per_second'])} | {f(r['greedy_ms_per_example'])} | {f(r['beam5_ms_per_example'])} |")
        (TABLES / "compute_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # ---------------------------------------------------------------- seed variance
    seed_dir = ROOT / "experiments" / "full_model" / "full_seed43"
    if "full" in rows and (seed_dir / "dev_metrics.json").exists():
        s43 = row_for("full_seed43", "FULL, seed 43", seed_dir)
        s42 = rows["full"]
        a, b = _per_example(experiment_dir(exps["full"]), "exact_match"), _per_example(seed_dir, "exact_match")
        _, lo, hi = paired_bootstrap(a, b)
        lines = [
            "### Training-seed variance of the controlled FULL model (dev, our measurements)", "",
            "| Seed | EM (%) | EX (%) | SQL valid (%) | Best epoch |", "|---|---|---|---|---|",
            f"| 42 | {pct(s42['exact_match'])} | {pct(s42['execution_accuracy'])} | {pct(s42['sql_validity'])} | {s42['best_epoch']} |",
            f"| 43 | {pct(s43['exact_match'])} | {pct(s43['execution_accuracy'])} | {pct(s43['sql_validity'])} | {s43['best_epoch']} |",
            "",
            f"Difference (43 − 42): EM {100 * (s43['exact_match'] - s42['exact_match']):+.1f} pts "
            f"(paired-bootstrap 95 % CI over dev examples [{100 * lo:+.1f}, {100 * hi:+.1f}]), "
            f"EX {100 * (s43['execution_accuracy'] - s42['execution_accuracy']):+.1f} pts.",
        ]
        (TABLES / "seed_variance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        pd.DataFrame([dict(s42, seed=42), dict(s43, seed=43)]).to_csv(TABLES / "seed_variance.csv", index=False)
    _plot_curves(rows, exps, final)
    print(f"collected {len(rows)} experiments -> {TABLES}")


def _plot_scaling(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(df["train_examples"], 100 * df["exact_match"], "o-", color="#2b6cb0", label="Exact Match")
    ax.plot(df["train_examples"], 100 * df["execution_accuracy"], "s--", color="#c05621", label="Execution Accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("Spider dev accuracy (%)")
    ax.set_title("Data scaling (controlled setting)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "data_scaling.png", dpi=150)
    plt.close(fig)


def _plot_curves(rows: dict, exps: dict, final: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("baseline_a_simple", "baseline_b_vanilla_transformer", "baseline_c_no_linking", "full") if n in rows]
    series = [(n, experiment_dir(exps[n])) for n in names] + ([("final", FINAL_DIR)] if final else [])
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, d in series:
        h = _load(d / "history.json") or []
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], label=n)
        ev = [(r["epoch"], r["val_em"]) for r in h if r.get("val_em") is not None]
        if ev:
            axes[1].plot([e for e, _ in ev], [100 * v for _, v in ev], "o-", label=n)
    axes[0].set_title("training loss (per example)")
    axes[0].set_yscale("log")
    axes[1].set_title("validation EM (held-out training DBs)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
