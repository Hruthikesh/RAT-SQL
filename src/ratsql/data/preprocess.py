"""Deterministic preprocessing of Spider-format data.

For every example we store (JSONL, one record per line):

* ``question``, ``question_tokens`` (+ char offsets), ``db_id``, gold ``query``
* ``sql``: gold SQL parsed by our Spider parser (JSON form), plus agreement
  with the ``sql`` field shipped in the dataset
* ``links``: schema links (n-gram name matches, value matches, numbers)
* ``candidates``: value candidates for the value pointer
* ``actions``: the oracle action sequence of the gold AST (grammar
  transition system), ``ast_ok`` / ``unsupported`` flags
* ``value_stats``: how many gold literals were resolved to candidates
* ``hardness`` (Spider difficulty) and ``complexity`` features
* ``oracle_em``: whether gold -> AST -> SQL -> re-parse is an exact match
  (validates the grammar, serializer and parser end to end)

Train / validation separation: the validation split consists of whole
*databases* held out from the Spider training set (chosen by a seeded hash),
so model selection happens on unseen schemas and the official dev set is
never used for training or checkpoint selection.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path

from ratsql.data.spider import DatasetSpec, load_split, load_split_schemas
from ratsql.data.values import ValueResolver, build_value_candidates, candidates_to_json
from ratsql.evaluation.complexity import complexity_features
from ratsql.evaluation.exact_match import ExactMatchEvaluator, hardness
from ratsql.schema.db_values import DatabaseValueIndex
from ratsql.schema.linking import SchemaLinker
from ratsql.schema.schema import Schema
from ratsql.sql.ast import UnsupportedSQLError, ast_to_sql, sql_to_ast
from ratsql.sql.parser import SQLParseError, canonical, parse_sql, to_jsonable, to_tuples
from ratsql.sql.serializer import serialize_sql
from ratsql.sql.transition import ast_to_actions, replay
from ratsql.utils.io import get_logger, write_json, write_jsonl
from ratsql.utils.text import tokenize

log = get_logger("ratsql.preprocess")


def _same_sql(a: dict, b: dict) -> bool:
    """Structural equality ignoring the LIMIT value (the original parser always stores 1)."""
    a, b = canonical(a), canonical(b)

    def strip_limit(s):
        if s is None:
            return s
        s = dict(s)
        s["limit"] = None if s["limit"] is None else 1
        for op in ("intersect", "union", "except"):
            s[op] = strip_limit(s[op])
        return s

    return repr(strip_limit(a)) == repr(strip_limit(b))


def choose_validation_dbs(db_ids: list[str], fraction: float, seed: int) -> set[str]:
    ranked = sorted(set(db_ids), key=lambda d: hashlib.sha1(f"{seed}:{d}".encode()).hexdigest())
    k = max(1, round(len(ranked) * fraction))
    return set(ranked[:k])


class Preprocessor:
    def __init__(self, spec: DatasetSpec, cfg: dict, value_cache_dir: str | Path | None):
        self.spec = spec
        self.cfg = cfg
        self.value_cache_dir = Path(value_cache_dir) if value_cache_dir else None
        lcfg = cfg.get("linking", {})
        self.linker = SchemaLinker(
            max_n=lcfg.get("max_ngram", 5),
            max_vpm_per_token=lcfg.get("max_vpm_per_token", 3),
            use_values=lcfg.get("value_linking", True),
            use_numbers=True,
        )
        vcfg = cfg.get("values", {})
        self.max_value_ngram = vcfg.get("max_ngram", 4)
        self.max_db_values = vcfg.get("max_db_values", 30)
        self.max_candidates = vcfg.get("max_candidates", 128)
        self.max_values_per_column = lcfg.get("max_values_per_column", 5000)
        self._value_index: dict[tuple[str, str], DatabaseValueIndex] = {}

    def value_index(self, split: str, schema: Schema) -> DatabaseValueIndex:
        key = (split if split == "test" else "main", schema.db_id)
        if key not in self._value_index:
            cache = None
            if self.value_cache_dir is not None:
                cache = self.value_cache_dir / ("test" if split == "test" else "main")
            self._value_index[key] = DatabaseValueIndex.load_or_build(
                schema, self.spec.db_path(split, schema.db_id), cache, max_values_per_column=self.max_values_per_column
            )
        return self._value_index[key]

    # ---------------------------------------------------------------- single
    def process_example(self, ex: dict, schema: Schema, split: str, ex_id: str) -> dict:
        question = " ".join(ex["question"].split())
        toks = tokenize(question)
        words = [t.text for t in toks]
        vindex = self.value_index(split, schema)
        links = self.linker.link(words, schema, vindex)
        cands = build_value_candidates(question, toks, links, self.max_value_ngram, self.max_db_values, self.max_candidates)
        rec: dict = {
            "id": ex_id,
            "split": split,
            "db_id": ex["db_id"],
            "question": question,
            "question_tokens": words,
            "token_offsets": [[t.start, t.end] for t in toks],
            "query": ex["query"],
            "source_file": ex.get("source_file"),
            "links": links.to_json(),
            "candidates": candidates_to_json(cands),
        }
        # ---- gold SQL
        parse_error = None
        try:
            sql = parse_sql(ex["query"], schema, strict=False)  # official parser ignores trailing tokens
            rec["parse_ok"] = True
        except SQLParseError as e:
            parse_error = str(e)
            rec["parse_ok"] = False
            sql = to_tuples(ex["sql"]) if "sql" in ex else None
        rec["parse_error"] = parse_error
        rec["parse_agrees_with_dataset"] = bool("sql" in ex and sql is not None and rec["parse_ok"] and _same_sql(sql, ex["sql"]))
        if sql is None:
            rec.update(ast_ok=False, unsupported="no gold sql", actions=None, sql=None)
            return rec
        sql = canonical(sql)
        rec["sql"] = to_jsonable(sql)
        rec["hardness"] = hardness(sql)
        rec["complexity"] = complexity_features(sql)
        # ---- AST and actions
        resolver = ValueResolver(cands)
        try:
            ast = sql_to_ast(sql, resolver)
            actions = ast_to_actions(ast)
            rebuilt, _ = replay(actions, num_columns=schema.num_columns, num_tables=schema.num_tables, num_values=len(cands))
            assert rebuilt == ast, "replay mismatch"
            rec["ast_ok"] = True
            rec["unsupported"] = None
            rec["actions"] = [list(a) for a in actions]
            # oracle round trip: AST -> SQL dict -> string -> parse -> EM against gold
            oracle_sql = serialize_sql(ast_to_sql(ast, resolver.lookup), schema)
            rec["oracle_sql"] = oracle_sql
            em = ExactMatchEvaluator(schema)
            try:
                rec["oracle_em"] = em.evaluate(parse_sql(oracle_sql, schema), sql)["exact"]
            except SQLParseError as e:
                rec["oracle_em"] = False
                rec["oracle_error"] = str(e)
        except UnsupportedSQLError as e:
            rec.update(ast_ok=False, unsupported=str(e), actions=None, oracle_em=False)
        rec["value_stats"] = {
            "resolved": resolver.resolved,
            "unresolved": resolver.unresolved,
            "unresolved_values": resolver.unresolved_values,
        }
        return rec

    # ----------------------------------------------------------------- split
    def process_split(self, split: str) -> list[dict]:
        examples = load_split(self.spec, split)
        schemas = load_split_schemas(self.spec, split)
        out = []
        t0 = time.time()
        for i, ex in enumerate(examples):
            out.append(self.process_example(ex, schemas[ex["db_id"]], split, f"{split}_{i:05d}"))
            if (i + 1) % 1000 == 0:
                log.info(f"[{split}] {i + 1}/{len(examples)} examples ({time.time() - t0:.1f}s)")
        return out


def split_stats(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"examples": 0}
    vs_res = sum(r.get("value_stats", {}).get("resolved", 0) for r in records)
    vs_unres = sum(r.get("value_stats", {}).get("unresolved", 0) for r in records)
    ast_ok = [r for r in records if r.get("ast_ok")]
    return {
        "examples": n,
        "databases": len({r["db_id"] for r in records}),
        "parse_ok": sum(r.get("parse_ok", False) for r in records),
        "parse_agrees_with_dataset": sum(r.get("parse_agrees_with_dataset", False) for r in records),
        "ast_ok": len(ast_ok),
        "unsupported": dict(Counter(r["unsupported"].split(" ")[0] for r in records if r.get("unsupported"))),
        "oracle_em": sum(bool(r.get("oracle_em")) for r in records),
        "gold_literals_resolved": vs_res,
        "gold_literals_unresolved": vs_unres,
        "literal_coverage": round(vs_res / max(1, vs_res + vs_unres), 4),
        "hardness": dict(Counter(r.get("hardness") for r in records)),
        "mean_question_tokens": round(sum(len(r["question_tokens"]) for r in records) / n, 2),
        "mean_actions": round(sum(len(r["actions"]) for r in ast_ok) / max(1, len(ast_ok)), 2),
        "max_actions": max((len(r["actions"]) for r in ast_ok), default=0),
        "mean_candidates": round(sum(len(r["candidates"]) for r in records) / n, 2),
        "link_types": dict(Counter(t for r in records for _, _, t in r["links"]["q_col"])),
    }


def run_preprocessing(spec: DatasetSpec, cfg: dict, out_dir: str | Path, value_cache_dir: str | Path | None, splits: list[str] | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = Preprocessor(spec, cfg, value_cache_dir)
    splits = splits or spec.available_splits()
    stats: dict = {}
    val_cfg = cfg.get("validation", {})
    for split in splits:
        records = pre.process_split(split)
        if split == "train" and val_cfg.get("fraction", 0) > 0:
            pool = [r["db_id"] for r in records if r.get("source_file") in (None, "train_spider.json", "train.json")]
            val_dbs = choose_validation_dbs(pool, val_cfg["fraction"], val_cfg.get("seed", 0))
            val = [dict(r, split="val") for r in records if r["db_id"] in val_dbs]
            train = [r for r in records if r["db_id"] not in val_dbs]
            write_jsonl(train, out_dir / "train.jsonl")
            write_jsonl(val, out_dir / "val.jsonl")
            stats["train"] = split_stats(train)
            stats["val"] = split_stats(val)
            stats["val"]["db_ids"] = sorted(val_dbs)
            log.info(f"train: {len(train)} examples; val: {len(val)} examples from {len(val_dbs)} held-out databases")
        else:
            write_jsonl(records, out_dir / f"{split}.jsonl")
            stats[split] = split_stats(records)
            log.info(f"{split}: {len(records)} examples")
    # strict separation checks
    dbs = {s: set() for s in stats}
    for s in stats:
        p = out_dir / f"{s}.jsonl"
        if p.exists():
            from ratsql.utils.io import iter_jsonl

            dbs[s] = {r["db_id"] for r in iter_jsonl(p)}
    overlap = {}
    names = list(dbs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = dbs[a] & dbs[b]
            overlap[f"{a}&{b}"] = sorted(inter)
    stats["database_overlap"] = overlap
    write_json(stats, out_dir / "stats.json")
    return stats
