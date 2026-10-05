"""High-level train / evaluate entry points shared by the scripts."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ratsql.evaluation.evaluator import Evaluator, predict_features
from ratsql.experiment import Experiment, resolve_path
from ratsql.training.trainer import Trainer
from ratsql.utils.device import device_info, resolve_device
from ratsql.utils.io import get_logger, write_json, write_jsonl
from ratsql.utils.seed import set_seed

log = get_logger("ratsql.run")


def train_from_config(cfg: dict, resume: bool = False) -> dict:
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "config.json")
    exp = Experiment(cfg)
    dcfg = cfg["data"]
    train_feats = exp.features("train", require_actions=True, fraction=dcfg.get("train_fraction", 1.0), seed=cfg.get("seed", 42), limit=dcfg.get("train_limit"))
    val_feats = exp.features("val" if (exp.processed_dir / "val.jsonl").exists() and _nonempty(exp.processed_dir / "val.jsonl") else "dev", limit=dcfg.get("val_limit"))
    model = exp.build_model()
    counts = model.parameter_counts()
    log.info(f"parameters: {counts}")
    evaluator = Evaluator(exp.schemas("val"), exp.db_path_fn("val"), cfg.get("eval", {}).get("exec_timeout", 5.0))
    trainer = Trainer(cfg, model, train_feats, val_feats, evaluator, out_dir, device)
    write_json({"parameters": counts, "device": device_info(), "train_examples": len(train_feats), "val_examples": len(val_feats), "seed": cfg.get("seed", 42)}, out_dir / "run_info.json")
    summary = trainer.fit(resume=resume)
    summary["parameters"] = counts
    summary["train_examples"] = len(train_feats)
    write_json(summary, out_dir / "train_summary.json")
    return summary


def _nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def load_model_from_checkpoint(ckpt_path: str | Path, device=None, cfg_override: dict | None = None):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_override or state["config"]
    exp = Experiment(cfg)
    model = exp.build_model(pretrained=False)
    model.load_state_dict(state["model"])
    device = device or resolve_device(cfg.get("device", "auto"))
    model.to(device).eval()
    return model, exp, cfg, device


def evaluate_checkpoint(ckpt_path: str | Path, split: str, out_prefix: str | Path, beam_size: int | None = None, limit: int | None = None, do_exec: bool = True, cfg_override: dict | None = None) -> dict:
    model, exp, cfg, device = load_model_from_checkpoint(ckpt_path, cfg_override=cfg_override)
    ecfg = cfg.get("eval", {})
    beam = ecfg.get("beam_size", 1) if beam_size is None else beam_size
    feats = exp.features(split, limit=limit)
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    outs, secs = predict_features(model, feats, device, cfg["train"].get("eval_batch_size", 32), beam, ecfg.get("max_steps", 250), amp)
    evaluator = Evaluator(exp.schemas(split), exp.db_path_fn(split), ecfg.get("exec_timeout", 5.0), do_exec=do_exec)
    t0 = time.time()
    metrics, rows = evaluator.evaluate(feats, outs)
    metrics.update(
        {
            "split": split,
            "checkpoint": str(ckpt_path),
            "beam_size": beam,
            "decode_seconds": round(secs, 2),
            "ms_per_example": round(1000 * secs / max(1, len(feats)), 2),
            "eval_seconds": round(time.time() - t0, 2),
            "parameters": model.parameter_counts(),
        }
    )
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics, str(out_prefix) + "_metrics.json")
    write_jsonl(rows, str(out_prefix) + "_predictions.jsonl")
    log.info(f"[{split}] EM={metrics['exact_match']:.4f} EX={metrics.get('execution_accuracy')} valid={metrics.get('sql_validity')} ({len(feats)} examples)")
    return metrics
