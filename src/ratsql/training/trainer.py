"""Training loop.

* teacher forcing with an action-level negative log-likelihood (sum over the
  gold action sequence, averaged over examples);
* AdamW with separate learning rates for the pretrained encoder and the rest;
* linear warm-up followed by polynomial decay (power 1 = linear; RAT-SQL
  used power 0.5);
* automatic mixed precision (fp16 autocast + GradScaler) on CUDA;
* gradient accumulation, gradient clipping;
* periodic validation (loss, exact match, execution accuracy) on held-out
  *training databases*; the best checkpoint by validation EM is kept;
* early stopping (patience in evaluations), full checkpoint/resume
  (model, optimizer, scheduler, scaler, RNG states, history).
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ratsql.data.dataset import BucketBatchSampler, Features, TextToSQLDataset, collate
from ratsql.evaluation.evaluator import Evaluator, predict_features
from ratsql.utils.io import get_logger, write_json


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, power: float = 1.0, min_ratio: float = 0.0):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return max(min_ratio, (1.0 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, cfg: dict, model, train_feats: list[Features], val_feats: list[Features], evaluator: Evaluator | None, out_dir: str | Path, device):
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.model = model.to(device)
        self.device = device
        self.train_feats = train_feats
        self.val_feats = val_feats
        self.evaluator = evaluator
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"ratsql.train.{self.out_dir.name}", self.out_dir / "train.log")
        t = self.tcfg
        self.sampler = BucketBatchSampler(train_feats, t["batch_size"], shuffle=True, seed=cfg.get("seed", 0), max_tokens=t.get("max_tokens"))
        self.loader = torch.utils.data.DataLoader(TextToSQLDataset(train_feats), batch_sampler=self.sampler, collate_fn=collate, num_workers=0)
        self.accum = max(1, t.get("grad_accum", 1))
        self.steps_per_epoch = math.ceil(len(self.sampler) / self.accum)
        self.epochs = t["epochs"]
        self.total_steps = t.get("max_steps") or self.steps_per_epoch * self.epochs
        groups = [{"params": [p for p in model.other_parameters() if p.requires_grad], "lr": t["lr"], "weight_decay": t.get("weight_decay", 0.0)}]
        pre = [p for p in model.pretrained_parameters() if p.requires_grad]
        if pre:
            groups.append({"params": pre, "lr": t.get("bert_lr", 2e-5), "weight_decay": t.get("bert_weight_decay", 0.01)})
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        warm = int(t.get("warmup_ratio", 0.05) * self.total_steps) if "warmup_steps" not in t else t["warmup_steps"]
        self.scheduler = make_scheduler(self.optimizer, warm, self.total_steps, t.get("lr_power", 1.0))
        self.use_amp = bool(t.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.step = 0
        self.epoch = 0
        self.best_metric = -1.0
        self.best_epoch = -1
        self.evals_without_improvement = 0
        self.history: list[dict] = []
        self.train_seconds = 0.0

    # ----------------------------------------------------------- checkpoint
    def save(self, name: str, full: bool = True) -> None:
        state = {"model": self.model.state_dict(), "config": self.cfg, "epoch": self.epoch, "step": self.step, "best_metric": self.best_metric, "best_epoch": self.best_epoch}
        if full:
            state.update(
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
                history=self.history,
                evals_without_improvement=self.evals_without_improvement,
                train_seconds=self.train_seconds,
                rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            )
        torch.save(state, self.out_dir / name)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch, self.step = state["epoch"], state["step"]
        self.best_metric, self.best_epoch = state["best_metric"], state["best_epoch"]
        self.history = state.get("history", [])
        self.evals_without_improvement = state.get("evals_without_improvement", 0)
        self.train_seconds = state.get("train_seconds", 0.0)
        rng = state.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        self.log.info(f"resumed from {path} at epoch {self.epoch}, step {self.step}")

    # ------------------------------------------------------------- training
    def train_epoch(self) -> dict:
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        ctx = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
        tot_loss = tot_steps = tot_correct = tot_ex = 0.0
        nonfinite_batches = skipped_steps = 0
        t0 = time.time()
        clip = self.tcfg.get("clip_grad", 1.0)
        self.optimizer.zero_grad(set_to_none=True)
        n_batches = len(self.sampler)
        for i, batch in enumerate(self.loader):
            batch = batch.to(self.device)
            with ctx:
                out = self.model(batch)
            loss = out["loss"] / self.accum
            self.scaler.scale(loss).backward()
            loss_value = out["loss"].item()
            if math.isfinite(loss_value):
                tot_loss += loss_value * len(batch.feats)
                tot_ex += len(batch.feats)
            else:
                nonfinite_batches += 1
            tot_steps += out["num_steps"].item()
            tot_correct += out["step_correct"].item()
            if (i + 1) % self.accum == 0 or (i + 1) == n_batches:
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if not self.use_amp or self.scaler.get_scale() >= scale_before:  # skip if AMP skipped the step (inf grads)
                    self.scheduler.step()
                else:
                    skipped_steps += 1
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % self.tcfg.get("log_every", 50) == 0:
                    self.log.info(
                        f"epoch {self.epoch} step {self.step}/{self.total_steps} loss {tot_loss / max(1, tot_ex):.3f} "
                        f"step_acc {tot_correct / max(1, tot_steps):.3f} gnorm {float(gnorm):.2f} lr {self.scheduler.get_last_lr()[0]:.2e}"
                    )
                if self.step >= self.total_steps:
                    break
        secs = time.time() - t0
        return {
            "train_loss": tot_loss / max(1, tot_ex),  # mean over batches with a finite loss
            "train_step_acc": tot_correct / max(1, tot_steps),
            "epoch_seconds": secs,
            "nonfinite_loss_batches": nonfinite_batches,
            "amp_skipped_steps": skipped_steps,
        }

    @torch.no_grad()
    def validation_loss(self) -> float:
        self.model.eval()
        feats = [f for f in self.val_feats if f.steps is not None]
        ctx = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
        tot, n = 0.0, 0
        bs = self.tcfg.get("eval_batch_size", 32)
        for s in range(0, len(feats), bs):
            batch = collate(feats[s : s + bs]).to(self.device)
            with ctx:
                out = self.model(batch)
            tot += out["loss"].item() * len(batch.feats)
            n += len(batch.feats)
        return tot / max(1, n)

    def validate(self) -> dict:
        res = {"val_loss": self.validation_loss()}
        if self.evaluator is not None and self.val_feats:
            outs, secs = predict_features(self.model, self.val_feats, self.device, self.tcfg.get("eval_batch_size", 32), 1, self.cfg.get("eval", {}).get("max_steps", 250), self.use_amp)
            metrics, _ = self.evaluator.evaluate(self.val_feats, outs)
            res.update({"val_em": metrics["exact_match"], "val_ex": metrics.get("execution_accuracy"), "val_grammar_valid": metrics["grammar_validity"], "val_exec_valid": metrics.get("sql_validity"), "val_decode_seconds": secs})
        return res

    def fit(self, resume: bool = False) -> dict:
        if resume and (self.out_dir / "last.pt").exists():
            self.load(self.out_dir / "last.pt")
        eval_every = self.tcfg.get("eval_every", 1)
        patience = self.tcfg.get("patience", 0)
        metric_name = self.tcfg.get("select_metric", "val_em")
        self.log.info(f"training {len(self.train_feats)} examples, {self.steps_per_epoch} optimizer steps/epoch, {self.total_steps} total steps, amp={self.use_amp}")
        while self.epoch < self.epochs and self.step < self.total_steps:
            t0 = time.time()
            rec = {"epoch": self.epoch + 1, **self.train_epoch()}
            self.train_seconds += time.time() - t0
            self.epoch += 1
            if self.epoch % eval_every == 0 or self.epoch == self.epochs or self.step >= self.total_steps:
                rec.update(self.validate())
                metric = rec.get(metric_name, -rec["val_loss"])
                if metric is not None and metric > self.best_metric:
                    self.best_metric, self.best_epoch = metric, self.epoch
                    self.evals_without_improvement = 0
                    self.save("best.pt", full=False)
                else:
                    self.evals_without_improvement += 1
            rec["lr"] = self.scheduler.get_last_lr()[0]
            rec["train_seconds_cum"] = round(self.train_seconds, 1)
            self.history.append(rec)
            self.log.info("epoch summary " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            self.save("last.pt", full=True)
            write_json(self.history, self.out_dir / "history.json")
            if patience and self.evals_without_improvement >= patience:
                self.log.info(f"early stopping at epoch {self.epoch} (best {metric_name}={self.best_metric:.4f} at epoch {self.best_epoch})")
                break
        summary = {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "epochs_run": self.epoch, "steps": self.step, "train_seconds": round(self.train_seconds, 1)}
        write_json(summary, self.out_dir / "train_summary.json")
        return summary
