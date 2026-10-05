"""Evaluate a trained checkpoint: Exact Match, Execution Accuracy, SQL validity, components.

    python scripts/evaluate.py --checkpoint experiments/full_model/rat_full/best.pt --split dev
    python scripts/evaluate.py --checkpoint ... --split test --beam-size 5
    python scripts/evaluate.py --predictions-only preds.sql --split dev --config configs/full.yaml
        (evaluate an external file with one SQL per line, aligned with the split)

Writes ``<out>_metrics.json`` and ``<out>_predictions.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ratsql.evaluation.evaluator import Evaluator, summarize  # noqa: E402
from ratsql.experiment import Experiment  # noqa: E402
from ratsql.training.run import evaluate_checkpoint  # noqa: E402
from ratsql.utils.config import load_config  # noqa: E402
from ratsql.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None, help="output prefix (default: next to the checkpoint)")
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--predictions-only", default=None, help="text file with one predicted SQL per line")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.predictions_only:
        cfg = load_config(args.config)
        exp = Experiment(cfg)
        recs = read_jsonl(exp.processed_dir / f"{args.split}.jsonl")
        preds = Path(args.predictions_only).read_text(encoding="utf-8").splitlines()
        assert len(preds) == len(recs), f"{len(preds)} predictions for {len(recs)} examples"
        ev = Evaluator(exp.schemas(args.split), exp.db_path_fn(args.split), do_exec=not args.no_exec)
        rows = [ev.evaluate_one(r, p, bool(p.strip())) for r, p in zip(recs, preds)]
        metrics = summarize(rows)
        out = Path(args.out or f"results/predictions/external_{args.split}")
        write_json(metrics, str(out) + "_metrics.json")
        write_jsonl(rows, str(out) + "_predictions.jsonl")
        print({k: metrics[k] for k in ("n", "exact_match", "execution_accuracy", "sql_validity") if k in metrics})
        return
    ckpt = Path(args.checkpoint)
    out = args.out or str(ckpt.parent / f"{args.split}{'_beam' + str(args.beam_size) if args.beam_size else ''}")
    m = evaluate_checkpoint(ckpt, args.split, out, beam_size=args.beam_size, limit=args.limit, do_exec=not args.no_exec)
    print({k: m.get(k) for k in ("n", "exact_match", "execution_accuracy", "sql_validity", "grammar_validity", "ms_per_example")})
    print("by hardness:", m["by_hardness"])


if __name__ == "__main__":
    main()
