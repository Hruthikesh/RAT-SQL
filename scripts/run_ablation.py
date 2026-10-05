"""Run the controlled experiment registry (baselines, ablations, data scaling).

    python scripts/run_ablation.py                              # everything in configs/experiments.yaml
    python scripts/run_ablation.py --groups main baseline       # a subset of groups
    python scripts/run_ablation.py --only full ablation_h_no_grammar
    python scripts/run_ablation.py --dry-run                    # print the commands only
    python scripts/run_ablation.py --set train.epochs=20        # override the shared budget

Each experiment trains with scripts/train.py (resumable), evaluates its best
checkpoint on Spider dev, and writes into experiments/<group>/<name>/.  The
runner skips experiments whose dev metrics already exist (use --force to
re-run).  Afterwards scripts/collect_results.py builds the result tables.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GROUP_DIRS = {"main": "full_model", "baseline": "baseline", "ablation": "ablations", "scaling": "scaling"}


def experiment_dir(exp: dict) -> Path:
    return ROOT / "experiments" / GROUP_DIRS[exp["group"]] / exp["name"]


def load_registry(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_command(reg: dict, exp: dict, extra_sets: list[str], base_epochs: int | None) -> list[str]:
    out = experiment_dir(exp)
    sets = [f"{k}={_fmt(v)}" for k, v in (exp.get("overrides") or {}).items()]
    if base_epochs is not None and exp.get("epoch_multiplier"):
        # smaller training sets get proportionally more (cheaper) epochs; validate at a similar cadence
        sets.append(f"train.epochs={int(round(base_epochs * exp['epoch_multiplier']))}")
        sets.append(f"train.eval_every={max(3, int(round(3 * exp['epoch_multiplier'])))}")
    sets += extra_sets
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(ROOT / reg["base_config"]), "--output-dir", str(out), "--eval-splits", "dev"]
    if sets:
        cmd += ["--set", *sets]
    if (out / "last.pt").exists():
        cmd.append("--resume")
    return cmd


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every experiment")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    reg = load_registry(args.registry)
    with open(ROOT / reg["base_config"], "r", encoding="utf-8") as f:
        base_epochs = yaml.safe_load(f).get("train", {}).get("epochs")
    for s in args.set:
        if s.startswith("train.epochs="):
            base_epochs = int(s.split("=", 1)[1])
    extra = [s for s in args.set if not s.startswith("train.epochs=")]
    if base_epochs is not None:
        extra_epochs = [f"train.epochs={base_epochs}"]
    else:
        extra_epochs = []
    todo = [e for e in reg["experiments"] if (args.groups is None or e["group"] in args.groups) and (args.only is None or e["name"] in args.only)]
    for exp in todo:
        out = experiment_dir(exp)
        if (out / "dev_metrics.json").exists() and not args.force:
            print(f"[skip] {exp['name']} (dev metrics exist)")
            continue
        sets = extra + ([] if exp.get("epoch_multiplier") else extra_epochs)
        cmd = build_command(reg, exp, sets, base_epochs)
        print(f"[run] {exp['name']}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out / "stdout.log", "a", encoding="utf-8") as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f"[done] {exp['name']} exit={ret} in {(time.time() - t0) / 60:.1f} min")
        if ret != 0:
            print(f"[error] {exp['name']} failed; see {out / 'stdout.log'}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_results.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
