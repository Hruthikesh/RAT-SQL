"""Assemble reports/RAT_SQL_REPORT.md and reports/experiment_results.md.

Authored text lives in reports/_parts/*.md; every table is read from results/tables (generated
from saved metrics), so the reports never contain hand-typed result tables.

    python scripts/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "reports" / "_parts"
T = ROOT / "results" / "tables"
M = ROOT / "results" / "metrics"
E = ROOT / "experiments"


MISSING: list[str] = []


def part(name: str) -> str:
    p = PARTS / f"{name}.md"
    if not p.exists():
        MISSING.append(name)
        return f"**[MISSING SECTION: {name}]**\n"
    return p.read_text(encoding="utf-8").strip() + "\n"


def table(name: str, demote: bool = True) -> str:
    p = T / name
    if not p.exists():
        return f"_({name} not available)_\n"
    txt = p.read_text(encoding="utf-8").strip() + "\n"
    return txt.replace("### ", "#### ") if demote else txt


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def crosscheck_summary() -> str:
    rows = ["| Predictions | Examples | EM (ours) | EM (official) | EM agreement | EX (ours, strict) | EX (official test-suite exec.) | EX agreement |", "|---|---|---|---|---|---|---|---|"]
    for name, label in (("full", "controlled FULL, dev"), ("final", "final model, dev"), ("final_test", "final model, test")):
        c = load(M / f"official_crosscheck_{name}.json")
        if not c:
            continue
        ex_o = c.get("official_execution_accuracy_all")
        ex_u = c.get("ours_execution_accuracy_on_compared")
        if ex_u is not None and c.get("exec_compared"):
            # skipped (timed-out) predictions are execution failures for us as well -> rescale to all examples
            ex_u = ex_u * c["exec_compared"] / c["n"]
        agree = c.get("exec_agreement")
        rows.append(
            f"| {label} | {c['n']:,} | {100 * c['ours_exact_match']:.2f} | {100 * c['official_exact_match']:.2f} | {100 * c['em_agreement']:.1f} % | "
            + (f"{100 * ex_u:.2f} | {100 * ex_o:.2f} | {100 * agree:.1f} % |" if ex_o is not None else "– | – | – |")
        )
    return "\n".join(rows) + "\n"


def beam_summary() -> str:
    lines = ["| Model | Decoding | EM (%) | EX (%) | SQL valid (%) | ms / example |", "|---|---|---|---|---|---|"]
    found = False
    for label, d in (("controlled FULL", E / "full_model" / "full"), ("final model", E / "full_model" / "final")):
        g, b = load(d / "dev_metrics.json"), load(d / "dev_beam5_metrics.json")
        if g and b:
            found = True
            for dec, m in (("greedy", g), ("beam 5", b)):
                lines.append(f"| {label} | {dec} | {100 * m['exact_match']:.1f} | {100 * m['execution_accuracy']:.1f} | {100 * m['sql_validity']:.1f} | {m['ms_per_example']:.1f} |")
    return ("\n".join(lines) + "\n") if found else "_(beam-search evaluation not available)_\n"


def headline_table() -> str:
    rows = ["| Model | Spider split | Exact Match | Execution Accuracy | Executable SQL | Training |", "|---|---|---|---|---|---|"]
    final_dir, full_dir = E / "full_model" / "final", E / "full_model" / "full"
    fs = load(final_dir / "train_summary.json") or {}
    for split, n in (("dev", "1,034"), ("test", "2,147")):
        m = load(final_dir / f"{split}_metrics.json")
        if m:
            rows.append(f"| Final model — BERT-medium + RAT + linking + grammar | {split} ({n}) | **{100 * m['exact_match']:.1f} %** | **{100 * m['execution_accuracy']:.1f} %** | {100 * m['sql_validity']:.1f} % | {fs.get('epochs_run')} epochs, {fs.get('train_seconds', 0) / 60:.0f} min |")
    m, s = load(full_dir / "dev_metrics.json"), load(full_dir / "train_summary.json") or {}
    if m:
        rows.append(f"| Controlled FULL — BERT-small + RAT + linking + grammar | dev (1,034) | {100 * m['exact_match']:.1f} % | {100 * m['execution_accuracy']:.1f} % | {100 * m['sql_validity']:.1f} % | {s.get('epochs_run')} epochs, {s.get('train_seconds', 0) / 60:.0f} min |")
    rows.append("| RAT-SQL + BERT-large (*paper*, not ours) | dev / test | 69.7 % / 65.6 % | – | – | 90k updates |")
    return "\n".join(rows)


def expand(text: str) -> str:
    """Replace {{table:name.md}}, {{headline}}, {{crosscheck}} and {{beam}} placeholders with generated content."""
    import re

    text = re.sub(r"\{\{table:([\w.\-]+)\}\}", lambda m: table(m.group(1)).strip(), text)
    text = text.replace("{{headline}}", headline_table())
    return text.replace("{{crosscheck}}", crosscheck_summary().strip()).replace("{{beam}}", beam_summary().strip())


def fill_readme() -> None:
    """Fill <!-- RESULTS:x:start --> ... <!-- RESULTS:x:end --> regions of README.md from _parts/readme_x.md."""
    import re

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for key in ("headline", "body", "limitations"):
        src = PARTS / f"readme_{key}.md"
        if not src.exists():
            continue
        content = expand(src.read_text(encoding="utf-8").strip())
        pattern = re.compile(rf"<!-- RESULTS:{key}:start -->.*?<!-- RESULTS:{key}:end -->", re.S)
        repl = f"<!-- RESULTS:{key}:start -->\n{content}\n<!-- RESULTS:{key}:end -->"
        if pattern.search(text):
            text = pattern.sub(lambda _m: repl, text)
        else:
            text = text.replace(f"<!-- RESULTS:{key} -->", repl)
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    fill_readme()
    report = [
        "# RAT-SQL: Schema-Aware Text-to-SQL — Research Report",
        "",
        part("00_abstract"),
        "## Contents",
        "",
        "1 Abstract · 2 Introduction · 3 Problem formulation · 4 Background · 5 Dataset · 6 RAT-SQL architecture · "
        "7 Schema graph construction · 8 Schema linking · 9 BERT contextual representation · 10 Relation-aware encoder · "
        "11 Decoder and grammar-constrained SQL generation · 12 Training methodology · 13 Experimental setup · "
        "14 Baselines · 15 Main results · 16 Ablation results · 17 Data-scaling results · 18 Seed and reproducibility "
        "results · 19 Complexity analysis · 20 Attention analysis · 21 Error analysis · 22 Numerical-stability "
        "investigation · 23 Computational and inference analysis · 24 Discussion · 25 Limitations · 26 Reproducibility · "
        "27 Conclusion",
        "",
        "> All result tables in this report are generated from saved metric files (`scripts/build_report.py`); "
        "numbers quoted from the RAT-SQL paper appear only in §15.1 and are labelled as such.",
        "",
        part("02_intro"),
        part("05_dataset"),
        part("06_architecture"),
        part("13_setup"),
        "## 15. Main results",
        "",
        part("15a_paper"),
        "### 15.2 Our results",
        "",
        table("main_results.md"),
        part("15b_main_discussion"),
        "#### Agreement with the official Spider evaluator",
        "",
        crosscheck_summary(),
        part("15c_crosscheck_note"),
        table("join_inference.md"),
        part("15d_join_note"),
        "## 16. Ablation results",
        "",
        table("ablation_results.md"),
        part("16a_ablation_discussion"),
        part("16b_h2"),
        part("17_scaling"),
        part("18_seed"),
        part("19_complexity"),
        part("20_attention"),
        part("21_errors"),
        part("22_numerics"),
        part("23_compute"),
        part("24_discussion"),
        part("25_limitations"),
        part("26_reproducibility"),
        part("27_conclusion"),
    ]
    (ROOT / "reports" / "RAT_SQL_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    results = [
        "# Experiment results",
        "",
        "Every table below is generated from saved metric files by `scripts/collect_results.py`, "
        "`scripts/run_error_analysis.py`, `scripts/run_join_inference.py`, "
        "`scripts/run_attention_analysis.py` and `scripts/official_eval_crosscheck.py`. "
        "All numbers are **our measurements**; the RAT-SQL paper's numbers are only in the research report, "
        "section 15.1.",
        "",
        "## Main results",
        "",
        table("main_results.md"),
        "## Official evaluator cross-check",
        "",
        crosscheck_summary(),
        "## Greedy vs beam search",
        "",
        beam_summary(),
        "## Ablations",
        "",
        table("ablation_results.md"),
        table("seed_variance.md"),
        "## Data scaling",
        "",
        table("scaling_results.md"),
        "![data scaling](../results/plots/data_scaling.png)",
        "",
        "## Difficulty, clauses and complexity",
        "",
        table("hardness_results.md"),
        table("clause_accuracy.md"),
        table("complexity_results.md"),
        "## Schema-graph join inference",
        "",
        table("join_inference.md"),
        "## Attention analysis",
        "",
        table("attention_relation_lift.md"),
        "## Synthetic relation experiment",
        "",
        table("synthetic_relation_experiment.md"),
        "## Error distribution",
        "",
        table("error_distribution.md"),
        "## Plots",
        "",
        "![training curves](../results/plots/training_curves.png)",
        "![complexity EM](../results/plots/complexity_em.png)",
        "![errors](../results/plots/error_distribution.png)",
        "![attention](../results/plots/attention_lift.png)",
        "![synthetic](../results/plots/synthetic_relation_experiment.png)",
        "",
    ]
    (ROOT / "reports" / "experiment_results.md").write_text("\n".join(results), encoding="utf-8")
    print("wrote reports/RAT_SQL_REPORT.md and reports/experiment_results.md")
    if MISSING:
        print("WARNING: missing report sections:", MISSING)


if __name__ == "__main__":
    main()
