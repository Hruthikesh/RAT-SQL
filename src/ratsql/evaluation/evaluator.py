"""Prediction and evaluation driver: decode -> AST -> SQL -> EM / EX / validity."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext

import torch

from ratsql.data.dataset import Features, collate
from ratsql.data.values import ValueResolver, candidates_from_json
from ratsql.evaluation.exact_match import COMPONENTS, HARDNESS_LEVELS, ExactMatchEvaluator
from ratsql.evaluation.execution import ExecutionResult, execute_sql, order_matters, results_equal
from ratsql.schema.schema import Schema
from ratsql.sql.ast import ast_to_sql
from ratsql.sql.parser import SQLParseError, canonical, parse_sql, to_jsonable, to_tuples
from ratsql.sql.serializer import serialize_sql


@torch.no_grad()
def predict_features(model, feats: list[Features], device, batch_size: int = 32, beam_size: int = 1, max_steps: int = 250, amp: bool = False) -> tuple[list[dict], float]:
    """Decode every example; returns raw decoder outputs aligned with ``feats`` and seconds spent."""
    model.eval()
    order = sorted(range(len(feats)), key=lambda i: feats[i].size)
    outs: list[dict | None] = [None] * len(feats)
    ctx = torch.autocast("cuda", dtype=torch.float16) if (amp and device.type == "cuda") else nullcontext()
    t0 = time.perf_counter()
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        batch = collate([feats[i] for i in idx]).to(device)
        with ctx:
            res = model.predict(batch, beam_size=beam_size, max_steps=max_steps)
        for i, r in zip(idx, res):
            outs[i] = r
    if device.type == "cuda":
        torch.cuda.synchronize()
    return outs, time.perf_counter() - t0  # type: ignore[return-value]


class Evaluator:
    def __init__(self, schemas: dict[str, Schema], db_path_fn, exec_timeout: float = 5.0, do_exec: bool = True):
        self.schemas = schemas
        self.db_path_fn = db_path_fn
        self.exec_timeout = exec_timeout
        self.do_exec = do_exec
        self._em: dict[str, ExactMatchEvaluator] = {}
        self._gold_cache: dict[tuple[str, str], ExecutionResult] = {}

    def em_eval(self, db_id: str) -> ExactMatchEvaluator:
        if db_id not in self._em:
            self._em[db_id] = ExactMatchEvaluator(self.schemas[db_id])
        return self._em[db_id]

    def gold_result(self, db_id: str, query: str) -> ExecutionResult:
        key = (db_id, query)
        if key not in self._gold_cache:
            self._gold_cache[key] = execute_sql(self.db_path_fn(db_id), query, self.exec_timeout)
        return self._gold_cache[key]

    def prediction_to_sql(self, rec: dict, out: dict) -> tuple[str, dict | None]:
        schema = self.schemas[rec["db_id"]]
        if out.get("ast") is None:
            return "", None
        resolver = ValueResolver(candidates_from_json(rec["candidates"]))
        sql_dict = ast_to_sql(out["ast"], resolver.lookup)
        return serialize_sql(sql_dict, schema), sql_dict

    def evaluate_one(self, rec: dict, pred_sql: str, grammar_valid: bool, decode_error: str | None = None) -> dict:
        db_id = rec["db_id"]
        schema = self.schemas[db_id]
        row = {
            "id": rec["id"],
            "db_id": db_id,
            "question": rec["question"],
            "gold_sql": rec["query"],
            "pred_sql": pred_sql,
            "hardness": rec.get("hardness"),
            "complexity": rec.get("complexity"),
            "grammar_valid": grammar_valid,
            "decode_error": decode_error,
        }
        parsed = None
        if grammar_valid:
            try:
                parsed = parse_sql(pred_sql, schema, strict=True)
                row["parse_valid"] = True
            except SQLParseError as e:
                row["parse_valid"] = False
                row["parse_error"] = str(e)[:300]
        else:
            row["parse_valid"] = False
        gold = canonical(to_tuples(rec["sql"])) if rec.get("sql") is not None else None
        if gold is not None:
            res = self.em_eval(db_id).evaluate(parsed, gold)
            row["exact_match"] = res["exact"]
            row["partial"] = {k: {"acc": v["acc"], "rec": v["rec"], "f1": v["f1"], "pred_total": v["pred_total"], "label_total": v["label_total"]} for k, v in res["partial"].items()}
        else:
            row["exact_match"] = False
            row["partial"] = None
        row["pred_parsed"] = to_jsonable(parsed) if parsed is not None else None
        if self.do_exec:
            g = self.gold_result(db_id, rec["query"])
            row["gold_exec_ok"] = g.ok
            if grammar_valid and pred_sql:
                p = execute_sql(self.db_path_fn(db_id), pred_sql, self.exec_timeout)
                row["exec_valid"] = p.ok
                row["exec_error"] = p.error
                row["exec_match"] = bool(g.ok and p.ok and results_equal(g.rows or [], p.rows or [], order_matters(rec["query"])))
            else:
                row["exec_valid"] = False
                row["exec_error"] = "no SQL (decoding failed)"
                row["exec_match"] = False
        return row

    def evaluate(self, feats: list[Features], outs: list[dict]) -> tuple[dict, list[dict]]:
        rows = []
        for f, out in zip(feats, outs):
            try:
                pred_sql, _ = self.prediction_to_sql(f.rec, out)
                grammar_valid = out.get("ast") is not None
            except Exception as e:  # serialization of a malformed tree
                pred_sql, grammar_valid = "", False
                out = dict(out, error=f"serialize: {e}")
            rows.append(self.evaluate_one(f.rec, pred_sql, grammar_valid, out.get("error")))
        return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def mean(key, subset=None):
        subset = rows if subset is None else subset
        vals = [bool(r.get(key)) for r in subset]
        return round(sum(vals) / max(1, len(vals)), 4)

    m = {
        "n": n,
        "exact_match": mean("exact_match"),
        "grammar_validity": mean("grammar_valid"),
        "parse_validity": mean("parse_valid"),
    }
    if "exec_match" in rows[0]:
        m["execution_accuracy"] = mean("exec_match")
        m["sql_validity"] = mean("exec_valid")  # executes without error
        m["gold_exec_ok"] = mean("gold_exec_ok")
    by_h = {}
    for h in HARDNESS_LEVELS:
        sub = [r for r in rows if r.get("hardness") == h]
        if sub:
            by_h[h] = {"n": len(sub), "exact_match": mean("exact_match", sub)}
            if "exec_match" in rows[0]:
                by_h[h]["execution_accuracy"] = mean("exec_match", sub)
    m["by_hardness"] = by_h
    # component scores, aggregated like the official Spider script
    comp = {}
    for c in COMPONENTS:
        acc_sum = acc_n = rec_sum = rec_n = 0
        for r in rows:
            p = (r.get("partial") or {}).get(c)
            if p is None:
                continue
            if p["pred_total"] > 0:
                acc_sum += p["acc"]
                acc_n += 1
            if p["label_total"] > 0:
                rec_sum += p["rec"]
                rec_n += 1
        acc = acc_sum / acc_n if acc_n else 0.0
        rec = rec_sum / rec_n if rec_n else 0.0
        f1 = 2 * acc * rec / (acc + rec) if (acc + rec) > 0 else 0.0
        comp[c] = {"acc": round(acc, 4), "rec": round(rec, 4), "f1": round(f1, 4)}
    m["components"] = comp
    errs = defaultdict(int)
    for r in rows:
        if r.get("decode_error"):
            errs[r["decode_error"].split(":")[-1].strip()[:60]] += 1
    m["decode_errors"] = dict(sorted(errs.items(), key=lambda x: -x[1])[:10])
    return m
