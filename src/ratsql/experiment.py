"""Experiment context: builds tokenizer, schemas, features and models from a config."""

from __future__ import annotations

import time
from pathlib import Path

from ratsql.data.dataset import FeatureBuilder, Features, load_records
from ratsql.data.spider import make_spec
from ratsql.models.model import Text2SQLModel, TokenizerInfo
from ratsql.schema.relations import RelationMatrixBuilder
from ratsql.schema.schema import load_schemas
from ratsql.utils.io import PROJECT_ROOT, get_logger

log = get_logger("ratsql.experiment")


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Experiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        data_cfg = dict(cfg["data"])
        data_cfg["root"] = str(resolve_path(data_cfg["root"]))
        self.spec = make_spec(data_cfg)
        self.processed_dir = resolve_path(cfg["data"]["processed_dir"])
        self._tokenizer = None
        self.relation_builder = RelationMatrixBuilder(cfg.get("relations", {}))
        self._schemas: dict[str, dict] = {}
        self._builders: dict[str, FeatureBuilder] = {}

    # ------------------------------------------------------------ resources
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            ecfg = self.cfg["model"]["encoder"]
            name = ecfg.get("tokenizer") or ecfg["pretrained_model"]
            self._tokenizer = AutoTokenizer.from_pretrained(name)
        return self._tokenizer

    def set_tokenizer(self, tok) -> None:
        self._tokenizer = tok

    def tokenizer_info(self) -> TokenizerInfo:
        t = self.tokenizer
        return TokenizerInfo(len(t), t.cls_token_id, t.sep_token_id, t.pad_token_id)

    def schemas(self, split: str) -> dict:
        key = "test" if split == "test" else "main"
        if key not in self._schemas:
            self._schemas[key] = load_schemas(self.spec.tables_path("test" if split == "test" else "train"))
        return self._schemas[key]

    def db_path_fn(self, split: str):
        s = "test" if split == "test" else "train"
        return lambda db_id: self.spec.db_path(s, db_id)

    def feature_builder(self, split: str) -> FeatureBuilder:
        key = "test" if split == "test" else "main"
        if key not in self._builders:
            dcfg = self.cfg["data"]
            self._builders[key] = FeatureBuilder(
                self.tokenizer,
                self.relation_builder,
                self.schemas(split),
                column_type_prefix=self.cfg["model"]["encoder"].get("column_type_prefix", True),
                max_word_pieces=dcfg.get("max_word_pieces", 8),
                max_element_pieces=dcfg.get("max_element_pieces", 16),
            )
        return self._builders[key]

    def features(self, split: str, require_actions: bool = False, fraction: float = 1.0, seed: int = 0, limit: int | None = None) -> list[Features]:
        path = self.processed_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/preprocess.py --config ... first")
        t0 = time.time()
        recs = load_records(path, require_actions=require_actions, fraction=fraction, seed=seed, limit=limit)
        fb = self.feature_builder(split)
        feats = [fb.build(r, i) for i, r in enumerate(recs)]
        log.info(f"built {len(feats)} {split} features in {time.time() - t0:.1f}s")
        return feats

    def build_model(self, pretrained: bool = True) -> Text2SQLModel:
        return Text2SQLModel(self.cfg["model"], self.tokenizer_info(), self.relation_builder.num_relations, pretrained=pretrained)
