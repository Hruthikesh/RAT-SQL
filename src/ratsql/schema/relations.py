"""Relation vocabulary and relation matrices for the relation-aware encoder.

The encoder input is the sequence ``X = [q_1..q_n, c_1..c_m, t_1..t_k]``
(question tokens, columns incl. ``*``, tables).  For every ordered pair
``(x_i, x_j)`` we assign one relation id ``r_ij``; the relation-aware
self-attention (``models/rat.py``) adds learned embeddings of ``r_ij`` to the
keys and values.

Two vocabularies are provided:

``full`` (default, RAT-SQL)
    question-question relative distance clipped to [-D, D]; question-column /
    question-table link types (EM, PM, VEM, VPM, NUM, default) in both
    directions; the schema relations of ``SchemaGraph``; identity and default
    relations for each node-type pair.

``coarse`` (ablation K)
    direction- and type-agnostic: identity, question-question, linked /
    unlinked question-schema, structural / unrelated schema-schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ratsql.schema.graph import SchemaGraph
from ratsql.schema.linking import COLUMN_LINK_TYPES, TABLE_LINK_TYPES, SchemaLinks
from ratsql.schema.schema import Schema

PAD = "<pad>"


class RelationVocabulary:
    def __init__(self, kind: str = "full", qq_max_dist: int = 2):
        self.kind = kind
        self.qq_max_dist = qq_max_dist
        if kind == "full":
            names = [PAD]
            names += [f"qq_dist_{d}" for d in range(-qq_max_dist, qq_max_dist + 1)]
            names += ["qc_default"] + [f"qc_{t.lower()}" for t in COLUMN_LINK_TYPES]
            names += ["cq_default"] + [f"cq_{t.lower()}" for t in COLUMN_LINK_TYPES]
            names += ["qt_default"] + [f"qt_{t.lower()}" for t in TABLE_LINK_TYPES]
            names += ["tq_default"] + [f"tq_{t.lower()}" for t in TABLE_LINK_TYPES]
            names += ["cc_identity", "cc_default", "cc_same_table", "cc_fk_forward", "cc_fk_backward"]
            names += ["ct_default", "ct_primary_key", "ct_belongs_to", "ct_foreign_key", "ct_any_table"]
            names += ["tc_default", "tc_primary_key", "tc_belongs_to", "tc_foreign_key", "tc_any_table"]
            names += ["tt_identity", "tt_default", "tt_fk_forward", "tt_fk_backward", "tt_fk_both"]
        elif kind == "coarse":
            names = [PAD, "identity", "qq", "qs_linked", "qs_none", "ss_struct", "ss_none"]
        else:
            raise ValueError(f"Unknown relation vocabulary {kind!r}")
        self.names = names
        self.index = {n: i for i, n in enumerate(names)}

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> int:
        return self.index[name]

    def name(self, idx: int) -> str:
        return self.names[idx]

    def coarse_of(self, full_name: str) -> str:
        """Map a full relation name to its coarse class."""
        if full_name in ("cc_identity", "tt_identity") or full_name == "qq_dist_0":
            return "identity"
        if full_name.startswith("qq_"):
            return "qq"
        if full_name[:2] in ("qc", "cq", "qt", "tq"):
            return "qs_none" if full_name.endswith("default") else "qs_linked"
        return "ss_none" if full_name.endswith("default") else "ss_struct"


@dataclass
class RelationConfig:
    vocab: str = "full"
    qq_max_dist: int = 2
    link_ngram: bool = True
    link_value: bool = True
    link_numbers: bool = True
    use_foreign_keys: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "RelationConfig":
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class RelationMatrix:
    """An N x N matrix of relation ids with block structure (q | c | t)."""

    def __init__(self, ids: np.ndarray, n_q: int, n_c: int, n_t: int, vocab: RelationVocabulary):
        self.ids = ids
        self.n_q, self.n_c, self.n_t = n_q, n_c, n_t
        self.vocab = vocab

    @property
    def size(self) -> int:
        return self.ids.shape[0]

    def names(self) -> np.ndarray:
        return np.vectorize(self.vocab.name)(self.ids)

    def block(self, row: str, col: str) -> np.ndarray:
        s = {"q": (0, self.n_q), "c": (self.n_q, self.n_q + self.n_c), "t": (self.n_q + self.n_c, self.size)}
        (r0, r1), (c0, c1) = s[row], s[col]
        return self.ids[r0:r1, c0:c1]

    def counts(self) -> dict[str, int]:
        uniq, cnt = np.unique(self.ids, return_counts=True)
        return {self.vocab.name(int(u)): int(c) for u, c in zip(uniq, cnt)}


class RelationMatrixBuilder:
    """Builds relation matrices from schemas and schema links (with caching)."""

    def __init__(self, config: RelationConfig | dict | None = None):
        self.config = config if isinstance(config, RelationConfig) else RelationConfig.from_dict(config)
        self.vocab = RelationVocabulary(self.config.vocab, self.config.qq_max_dist)
        self._full_vocab = RelationVocabulary("full", self.config.qq_max_dist)
        self._schema_cache: dict[str, np.ndarray] = {}
        self._qq_cache: dict[int, np.ndarray] = {}
        if self.config.vocab == "coarse":
            self._to_coarse = np.array(
                [self.vocab[self._full_vocab.coarse_of(n)] if n != PAD else 0 for n in self._full_vocab.names], dtype=np.int64
            )
        else:
            self._to_coarse = None

    @property
    def num_relations(self) -> int:
        return len(self.vocab)

    # --------------------------------------------------------------- blocks
    def _schema_block(self, schema: Schema) -> np.ndarray:
        key = schema.db_id
        if key in self._schema_cache:
            return self._schema_cache[key]
        v = self._full_vocab
        n_c, n_t = schema.num_columns, schema.num_tables
        m = n_c + n_t
        graph = SchemaGraph.from_schema(schema, use_foreign_keys=self.config.use_foreign_keys)
        block = np.empty((m, m), dtype=np.int64)
        block[:n_c, :n_c] = v["cc_default"]
        block[:n_c, n_c:] = v["ct_default"]
        block[n_c:, :n_c] = v["tc_default"]
        block[n_c:, n_c:] = v["tt_default"]
        pos = {("column", i): i for i in range(n_c)}
        pos.update({("table", j): n_c + j for j in range(n_t)})
        for (a, b), lab in graph.edges.items():
            block[pos[a], pos[b]] = v[lab]
        for i in range(n_c):
            block[i, i] = v["cc_identity"]
        for j in range(n_t):
            block[n_c + j, n_c + j] = v["tt_identity"]
        self._schema_cache[key] = block
        return block

    def _qq_block(self, n_q: int) -> np.ndarray:
        if n_q not in self._qq_cache:
            d = self.config.qq_max_dist
            idx = np.arange(n_q)
            dist = np.clip(idx[None, :] - idx[:, None], -d, d)
            base = self._full_vocab[f"qq_dist_{-d}"]
            self._qq_cache[n_q] = base + (dist + d)
        return self._qq_cache[n_q]

    # ---------------------------------------------------------------- build
    def build(self, schema: Schema, n_q: int, links: SchemaLinks | None) -> RelationMatrix:
        v = self._full_vocab
        n_c, n_t = schema.num_columns, schema.num_tables
        n = n_q + n_c + n_t
        ids = np.empty((n, n), dtype=np.int64)
        ids[:n_q, :n_q] = self._qq_block(n_q)
        ids[n_q:, n_q:] = self._schema_block(schema)
        ids[:n_q, n_q : n_q + n_c] = v["qc_default"]
        ids[n_q : n_q + n_c, :n_q] = v["cq_default"]
        ids[:n_q, n_q + n_c :] = v["qt_default"]
        ids[n_q + n_c :, :n_q] = v["tq_default"]
        if links is not None:
            cfg = self.config
            links = links.filtered(ngram=cfg.link_ngram, value=cfg.link_value, numbers=cfg.link_numbers)
            for (q, c), kind in links.q_col.items():
                if q < n_q and c < n_c:
                    k = kind.lower()
                    ids[q, n_q + c] = v[f"qc_{k}"]
                    ids[n_q + c, q] = v[f"cq_{k}"]
            for (q, t), kind in links.q_tab.items():
                if q < n_q and t < n_t:
                    k = kind.lower()
                    ids[q, n_q + n_c + t] = v[f"qt_{k}"]
                    ids[n_q + n_c + t, q] = v[f"tq_{k}"]
        if self._to_coarse is not None:
            ids = self._to_coarse[ids]
        return RelationMatrix(ids, n_q, n_c, n_t, self.vocab)
