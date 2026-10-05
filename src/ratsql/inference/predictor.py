"""Single-question inference: question + database -> SQL (with an optional execution)."""

from __future__ import annotations

from pathlib import Path

import torch

from ratsql.data.dataset import collate
from ratsql.data.values import ValueResolver, build_value_candidates, candidates_from_json, candidates_to_json
from ratsql.evaluation.execution import execute_sql
from ratsql.schema.db_values import DatabaseValueIndex
from ratsql.schema.linking import SchemaLinker, SchemaLinks, describe_links
from ratsql.schema.schema import Schema
from ratsql.sql.ast import ast_to_sql
from ratsql.sql.serializer import serialize_sql
from ratsql.training.run import load_model_from_checkpoint
from ratsql.utils.text import tokenize


class Predictor:
    def __init__(self, checkpoint: str | Path, device=None):
        self.model, self.exp, self.cfg, self.device = load_model_from_checkpoint(checkpoint, device)
        pcfg = self.cfg.get("preprocess", {})
        lcfg = pcfg.get("linking", {})
        vcfg = pcfg.get("values", {})
        self.linker = SchemaLinker(max_n=lcfg.get("max_ngram", 5), max_vpm_per_token=lcfg.get("max_vpm_per_token", 3), use_values=lcfg.get("value_linking", True))
        self.vcfg = vcfg
        self._vindex: dict[str, DatabaseValueIndex] = {}

    def _value_index(self, schema: Schema, db_path: Path | None) -> DatabaseValueIndex | None:
        if db_path is None:
            return None
        if schema.db_id not in self._vindex:
            self._vindex[schema.db_id] = DatabaseValueIndex.build(schema, db_path)
        return self._vindex[schema.db_id]

    def make_record(self, question: str, schema: Schema, db_path: Path | None = None) -> dict:
        question = " ".join(question.split())
        toks = tokenize(question)
        words = [t.text for t in toks]
        links = self.linker.link(words, schema, self._value_index(schema, db_path))
        cands = build_value_candidates(question, toks, links, self.vcfg.get("max_ngram", 4), self.vcfg.get("max_db_values", 30), self.vcfg.get("max_candidates", 128))
        return {"id": "query", "db_id": schema.db_id, "question": question, "question_tokens": words, "links": links.to_json(), "candidates": candidates_to_json(cands), "actions": None}

    @torch.no_grad()
    def predict(self, question: str, schema: Schema, db_path: str | Path | None = None, beam_size: int = 1, execute: bool = True) -> dict:
        db_path = Path(db_path) if db_path else None
        rec = self.make_record(question, schema, db_path)
        fb = self.exp.feature_builder("dev")
        fb.schemas = dict(fb.schemas, **{schema.db_id: schema})
        feat = fb.build(rec, 0)
        out = self.model.predict(collate([feat]).to(self.device), beam_size=beam_size)[0]
        result = {
            "question": question,
            "schema_links": describe_links(rec["question_tokens"], schema, SchemaLinks.from_json(rec["links"])),
            "ast": out["ast"].pretty() if out["ast"] is not None else None,
            "actions": len(out["actions"]),
            "error": out["error"],
            "sql": None,
        }
        if out["ast"] is not None:
            resolver = ValueResolver(candidates_from_json(rec["candidates"]))
            result["sql"] = serialize_sql(ast_to_sql(out["ast"], resolver.lookup), schema)
            if execute and db_path is not None:
                res = execute_sql(db_path, result["sql"])
                result["execution"] = {"ok": res.ok, "rows": (res.rows or [])[:20], "error": res.error}
        return result
