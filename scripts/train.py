"""Train a model from a YAML config.

    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/debug.yaml --set train.epochs=5 model.rat.num_layers=2
    python scripts/train.py --config configs/full.yaml --resume
    python scripts/train.py --config configs/full.yaml --eval-splits dev test   # evaluate best checkpoint afterwards

Artifacts in ``output_dir``: config.json, run_info.json (parameter counts,
device, seed), train.log, history.json, best.pt, last.pt, train_summary.json
and, with --eval-splits, <split>_metrics.json / <split>_predictions.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ratsql.experiment import resolve_path  # noqa: E402
from ratsql.training.run import evaluate_checkpoint, train_from_config  # noqa: E402
from ratsql.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=3")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-splits", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    summary = train_from_config(cfg, resume=args.resume)
    print("train summary:", summary)
    out_dir = resolve_path(cfg["output_dir"])
    for split in args.eval_splits:
        evaluate_checkpoint(out_dir / "best.pt", split, out_dir / split)


if __name__ == "__main__":
    main()
