"""Run every post-training analysis in order (after run_ablation.py and the final model).

    python scripts/run_analyses.py            # all steps
    python scripts/run_analyses.py --skip official notebooks

Steps: collect (tables/plots) -> errors (taxonomy, clauses, complexity, linking) ->
joins (FK join inference) -> attention (RAT vs vanilla) -> official (evaluator cross-check,
needs third_party/) -> predictions (copy per-example predictions to results/predictions) ->
notebooks (regenerate + execute).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
E = ROOT / "experiments"
MODELS = {
    "full": E / "full_model" / "full",
    "vanilla": E / "baseline" / "baseline_b_vanilla_transformer",
    "no_linking": E / "baseline" / "baseline_c_no_linking",
    "simple": E / "baseline" / "baseline_a_simple",
    "no_grammar": E / "ablations" / "ablation_h_no_grammar",
}


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=False)


def preds(names) -> list[str]:
    return [f"{n}={(MODELS[n] / 'dev_predictions.jsonl').relative_to(ROOT).as_posix()}" for n in names if (MODELS[n] / "dev_predictions.jsonl").exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    steps = ["collect", "errors", "joins", "attention", "official", "predictions", "notebooks"]
    for step in steps:
        if step in args.skip:
            continue
        if step == "collect":
            run([PY, "scripts/collect_results.py"])
        elif step == "errors":
            run([PY, "scripts/run_error_analysis.py", "--primary", "full", "--predictions", *preds(["full", "vanilla", "no_linking", "simple", "no_grammar"])])
        elif step == "joins":
            extra = []
            final = E / "full_model" / "final" / "dev_predictions.jsonl"
            if final.exists():
                extra.append(f"final={final.relative_to(ROOT).as_posix()}")
            run([PY, "scripts/run_join_inference.py", "--predictions", *preds(["full", "vanilla", "no_linking", "simple"]), *extra])
        elif step == "attention":
            models = [f"{n}={(MODELS[n] / 'best.pt').relative_to(ROOT).as_posix()}" for n in ("full", "vanilla") if (MODELS[n] / "best.pt").exists()]
            if models:
                run([PY, "scripts/run_attention_analysis.py", "--models", *[m.replace("full=", "rat=") for m in models], "--limit", "400"])
        elif step == "official":
            if (ROOT / "third_party" / "spider_eval" / "evaluation.py").exists():
                for name, d, split in (("full", MODELS["full"], "dev"), ("final", E / "full_model" / "final", "dev"), ("final_test", E / "full_model" / "final", "test")):
                    if (d / f"{split}_predictions.jsonl").exists():
                        run([PY, "scripts/official_eval_crosscheck.py", "--predictions", (d / f"{split}_predictions.jsonl").relative_to(ROOT).as_posix(), "--split", split, "--exec", "--name", name])
        elif step == "predictions":
            out = ROOT / "results" / "predictions"
            out.mkdir(parents=True, exist_ok=True)
            for name, d in list(MODELS.items()) + [("final", E / "full_model" / "final")]:
                for split in ("dev", "test"):
                    src = d / f"{split}_predictions.jsonl"
                    if src.exists():
                        shutil.copy(src, out / f"{name}_{split}_predictions.jsonl")
        elif step == "notebooks":
            run([PY, "scripts/make_notebooks.py"])
            run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=1200", *sorted(str(p) for p in (ROOT / "notebooks").glob("*.ipynb"))])


if __name__ == "__main__":
    main()
