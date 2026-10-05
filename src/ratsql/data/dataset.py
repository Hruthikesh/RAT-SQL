"""Model features, datasets, bucketing sampler and collation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ratsql.data.values import SOURCES, candidates_from_json
from ratsql.schema.linking import SchemaLinks
from ratsql.schema.relations import RelationMatrixBuilder
from ratsql.schema.schema import Schema
from ratsql.sql.grammar import Grammar, default_grammar
from ratsql.sql.transition import replay
from ratsql.utils.io import read_jsonl

STEP_FIELDS = ("kind", "target", "type_id", "field_id", "mask_id", "parent_step", "parent_rule")


@dataclass
class Features:
    idx: int
    rec: dict
    q_pieces: list[list[int]]
    col_pieces: list[list[int]]
    tab_pieces: list[list[int]]
    relations: np.ndarray  # [N, N] uint8
    cand_spans: list[tuple[int, int]]
    cand_source: list[int]
    cand_column: list[int]
    steps: np.ndarray | None  # [S, 7]

    @property
    def n_q(self) -> int:
        return len(self.q_pieces)

    @property
    def n_c(self) -> int:
        return len(self.col_pieces)

    @property
    def n_t(self) -> int:
        return len(self.tab_pieces)

    @property
    def n_v(self) -> int:
        return len(self.cand_spans)

    @property
    def size(self) -> int:
        return self.n_q + self.n_c + self.n_t

    @property
    def num_pieces(self) -> int:
        """Approximate BERT input length ([CLS] q [SEP] elem [SEP] ...), used for memory budgeting."""
        return 2 + sum(len(p) for p in self.q_pieces) + sum(len(p) + 1 for p in self.col_pieces + self.tab_pieces)


class FeatureBuilder:
    """Turns processed records into model features for one tokenizer / relation config."""

    def __init__(self, tokenizer, relation_builder: RelationMatrixBuilder, schemas: dict[str, Schema], column_type_prefix: bool = True, max_word_pieces: int = 8, max_element_pieces: int = 16, grammar: Grammar | None = None):
        self.tok = tokenizer
        self.rel = relation_builder
        self.schemas = schemas
        self.column_type_prefix = column_type_prefix
        self.max_word_pieces = max_word_pieces
        self.max_element_pieces = max_element_pieces
        self.g = grammar or default_grammar()
        self._schema_cache: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
        self._word_cache: dict[str, list[int]] = {}
        self.unk_id = tokenizer.unk_token_id

    def _pieces(self, text: str, cap: int) -> list[int]:
        key = f"{cap}\x00{text}"
        if key not in self._word_cache:
            ids = self.tok.convert_tokens_to_ids(self.tok.tokenize(text))[:cap]
            self._word_cache[key] = ids or [self.unk_id]
        return self._word_cache[key]

    def schema_pieces(self, schema: Schema):
        if schema.db_id not in self._schema_cache:
            cols = []
            for c in schema.columns:
                if c.table_id is None:
                    text = "*"
                else:
                    text = f"{c.type} {c.name}" if self.column_type_prefix else c.name
                cols.append(self._pieces(text, self.max_element_pieces))
            tabs = [self._pieces(t.name, self.max_element_pieces) for t in schema.tables]
            self._schema_cache[schema.db_id] = (cols, tabs)
        return self._schema_cache[schema.db_id]

    def build(self, rec: dict, idx: int) -> Features:
        schema = self.schemas[rec["db_id"]]
        q_pieces = [self._pieces(w, self.max_word_pieces) for w in rec["question_tokens"]]
        if not q_pieces:  # degenerate empty question
            q_pieces = [[self.unk_id]]
        cols, tabs = self.schema_pieces(schema)
        links = SchemaLinks.from_json(rec["links"])
        rel = self.rel.build(schema, len(q_pieces), links).ids.astype(np.uint8)
        cands = candidates_from_json(rec["candidates"])
        steps = None
        if rec.get("actions"):
            _, infos = replay([tuple(a) for a in rec["actions"]], self.g, schema.num_columns, schema.num_tables, len(cands))
            steps = np.array([[i.kind, i.target, i.type_id, i.field_id, i.mask_id, i.parent_step, i.parent_rule] for i in infos], dtype=np.int64)
        return Features(
            idx=idx,
            rec=rec,
            q_pieces=q_pieces,
            col_pieces=cols,
            tab_pieces=tabs,
            relations=rel,
            cand_spans=[(c.start, c.end) for c in cands],
            cand_source=[SOURCES.index(c.source) for c in cands],
            cand_column=[c.column for c in cands],
            steps=steps,
        )


class TextToSQLDataset(torch.utils.data.Dataset):
    def __init__(self, features: list[Features]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> Features:
        return self.features[i]


def load_records(path: str | Path, require_actions: bool = False, fraction: float = 1.0, seed: int = 0, limit: int | None = None) -> list[dict]:
    recs = read_jsonl(path)
    if require_actions:
        recs = [r for r in recs if r.get("actions")]
    if fraction < 1.0:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(recs)), max(1, int(round(len(recs) * fraction)))))
        recs = [recs[i] for i in idx]
    if limit is not None:
        recs = recs[:limit]
    return recs


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batches of similar input/output length (less padding), shuffled per epoch."""

    def __init__(self, features: list[Features], batch_size: int, shuffle: bool = True, seed: int = 0, max_tokens: int | None = None):
        self.features = features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.max_tokens = max_tokens

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        # Sort by decoder length (the sequential decoder dominates run time) plus a little
        # input length and noise; cap each batch by an input-token budget (B * max pieces).
        rng = random.Random(self.seed + self.epoch)
        keys = []
        for i, f in enumerate(self.features):
            s = 0 if f.steps is None else len(f.steps)
            noise = rng.random() if self.shuffle else 0.0
            keys.append((s + 0.02 * min(f.num_pieces, 512) + 4 * noise, i))
        order = [i for _, i in sorted(keys)]
        batches, cur, cur_max = [], [], 0
        for i in order:
            f = self.features[i]
            cost = min(f.num_pieces, 512) * max(1, -(-f.num_pieces // 512))  # pieces incl. extra segments
            new_max = max(cur_max, cost)
            if cur and (len(cur) >= self.batch_size or (self.max_tokens and new_max * (len(cur) + 1) > self.max_tokens)):
                batches.append(cur)
                cur, new_max = [], cost
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


@dataclass
class Batch:
    feats: list[Features]
    relations: torch.Tensor  # [B, N, N]
    steps: dict[str, torch.Tensor] | None

    def to(self, device) -> "Batch":
        return Batch(
            self.feats,
            self.relations.to(device, non_blocking=True),
            None if self.steps is None else {k: v.to(device, non_blocking=True) for k, v in self.steps.items()},
        )


def collate(feats: list[Features]) -> Batch:
    n_max = max(f.size for f in feats)
    rel = torch.zeros(len(feats), n_max, n_max, dtype=torch.long)
    for b, f in enumerate(feats):
        n = f.size
        rel[b, :n, :n] = torch.from_numpy(f.relations.astype(np.int64))
    steps = None
    if all(f.steps is not None for f in feats):
        s_max = max(len(f.steps) for f in feats)
        arr = np.zeros((len(feats), s_max, len(STEP_FIELDS)), dtype=np.int64)
        arr[:, :, 5] = -1  # parent_step padding
        arr[:, :, 6] = -1
        arr[:, :, 4] = 0
        mask = np.zeros((len(feats), s_max), dtype=bool)
        for b, f in enumerate(feats):
            arr[b, : len(f.steps)] = f.steps
            mask[b, : len(f.steps)] = True
        t = torch.from_numpy(arr)
        steps = {name: t[:, :, i] for i, name in enumerate(STEP_FIELDS)}
        steps["step_mask"] = torch.from_numpy(mask)
    return Batch(feats, rel, steps)
