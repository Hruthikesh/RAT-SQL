"""The full Text-to-SQL model: input encoder -> relation-aware transformer -> AST decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ratsql.data.dataset import Batch
from ratsql.data.values import SOURCES
from ratsql.models.decoder import ASTDecoder, EncoderOutputs
from ratsql.models.encoders import BertInputEncoder, LSTMInputEncoder
from ratsql.models.rat import RATEncoder
from ratsql.sql.grammar import Grammar, default_grammar


@dataclass
class TokenizerInfo:
    vocab_size: int
    cls_id: int
    sep_id: int
    pad_id: int


def build_input_encoder(cfg: dict, tok: TokenizerInfo, pretrained: bool = True) -> nn.Module:
    ecfg = cfg["encoder"]
    if ecfg["type"] == "bert":
        enc = BertInputEncoder(
            ecfg["pretrained_model"],
            max_length=ecfg.get("max_length", 512),
            pooling=ecfg.get("pooling", "mean"),
            freeze=ecfg.get("freeze", False),
            gradient_checkpointing=ecfg.get("gradient_checkpointing", False),
            pretrained=pretrained,
            config_overrides=ecfg.get("bert_config"),
        )
    elif ecfg["type"] == "lstm":
        init = None
        emb_dim = ecfg.get("emb_dim")
        if ecfg.get("init_from_pretrained", False):
            if pretrained:
                from transformers import AutoModel

                m = AutoModel.from_pretrained(ecfg["pretrained_model"])
                init = m.get_input_embeddings().weight.detach().clone()
                del m
                emb_dim = emb_dim or init.shape[1]
            else:  # rebuilding from a checkpoint: same width as the pretrained embedding matrix
                from transformers import AutoConfig

                emb_dim = emb_dim or AutoConfig.from_pretrained(ecfg["pretrained_model"]).hidden_size
        enc = LSTMInputEncoder(
            tok.vocab_size,
            emb_dim=emb_dim or 300,
            hidden=ecfg.get("hidden", 256),
            dropout=ecfg.get("dropout", 0.2),
            pad_id=tok.pad_id,
            init_embeddings=init,
            freeze_embeddings=ecfg.get("freeze_embeddings", False),
        )
    else:
        raise ValueError(f"unknown encoder type {ecfg['type']}")
    enc.set_special_ids(tok.cls_id, tok.sep_id, tok.pad_id)
    return enc


class Text2SQLModel(nn.Module):
    def __init__(self, model_cfg: dict, tok: TokenizerInfo, num_relations: int, grammar: Grammar | None = None, pretrained: bool = True):
        super().__init__()
        self.cfg = model_cfg
        self.grammar = grammar or default_grammar()
        self.encoder = build_input_encoder(model_cfg, tok, pretrained)
        rcfg = model_cfg["rat"]
        self.rat = RATEncoder(
            input_dim=self.encoder.output_dim,
            d_model=rcfg.get("d_model", 256),
            num_layers=rcfg.get("num_layers", 4),
            num_heads=rcfg.get("num_heads", 8),
            ff_dim=rcfg.get("ff_dim", 1024),
            num_relations=num_relations,
            dropout=rcfg.get("dropout", 0.1),
            use_relations=rcfg.get("use_relations", True),
            norm=rcfg.get("norm", "pre"),
            share_relation_embeddings=rcfg.get("share_relation_embeddings", False),
        )
        D = self.rat.output_dim
        self.val_proj = nn.Linear(2 * D, D)
        self.val_source = nn.Embedding(len(SOURCES), D)
        dcfg = model_cfg["decoder"]
        self.decoder = ASTDecoder(
            self.grammar,
            enc_dim=D,
            hidden=dcfg.get("hidden", 512),
            action_dim=dcfg.get("action_dim", 128),
            type_dim=dcfg.get("type_dim", 64),
            att_heads=dcfg.get("att_heads", 8),
            dropout=dcfg.get("dropout", 0.2),
            pointer=dcfg.get("pointer", "memory_aligned"),
            pointer_dim=dcfg.get("pointer_dim", 256),
            use_grammar_mask=dcfg.get("grammar_mask", True),
            pointer_fp32=dcfg.get("pointer_fp32", False),
        )

    # --------------------------------------------------------------- params
    def pretrained_parameters(self):
        if isinstance(self.encoder, BertInputEncoder):
            return list(self.encoder.bert.parameters())
        return []

    def other_parameters(self):
        pre = {id(p) for p in self.pretrained_parameters()}
        return [p for p in self.parameters() if id(p) not in pre]

    def parameter_counts(self) -> dict[str, int]:
        def count(mod, trainable_only=False):
            return sum(p.numel() for p in mod.parameters() if (p.requires_grad or not trainable_only))

        out = {
            "input_encoder": count(self.encoder),
            "rat_encoder": count(self.rat),
            "decoder": count(self.decoder),
            "value_encoder": count(self.val_proj) + count(self.val_source),
        }
        out["total"] = sum(p.numel() for p in self.parameters())
        out["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return out

    # --------------------------------------------------------------- encode
    def encode(self, batch: Batch) -> EncoderOutputs:
        device = batch.relations.device
        feats = batch.feats
        x, layout = self.encoder(feats, device)
        mem = self.rat(x, batch.relations, layout.mask)  # [B, N, D]
        B, N, D = mem.shape
        C = max(f.n_c for f in feats)
        T = max(f.n_t for f in feats)
        V = max(f.n_v for f in feats)
        col_idx = torch.zeros(B, C, dtype=torch.long)
        col_mask = torch.zeros(B, C, dtype=torch.bool)
        tab_idx = torch.zeros(B, T, dtype=torch.long)
        tab_mask = torch.zeros(B, T, dtype=torch.bool)
        for b, f in enumerate(feats):
            col_idx[b, : f.n_c] = torch.arange(f.n_q, f.n_q + f.n_c)
            col_mask[b, : f.n_c] = True
            tab_idx[b, : f.n_t] = torch.arange(f.n_q + f.n_c, f.size)
            tab_mask[b, : f.n_t] = True
        col_idx, col_mask, tab_idx, tab_mask = (t.to(device) for t in (col_idx, col_mask, tab_idx, tab_mask))
        col = torch.gather(mem, 1, col_idx.unsqueeze(-1).expand(B, C, D))
        tab = torch.gather(mem, 1, tab_idx.unsqueeze(-1).expand(B, T, D))
        # value candidates: mean of question-token encodings in the span (+ column encoding for DB values)
        item_idx, pos_idx, weights = [], [], []
        colref = torch.zeros(B, V, dtype=torch.long)
        has_col = torch.zeros(B, V, dtype=torch.bool)
        source = torch.zeros(B, V, dtype=torch.long)
        val_mask = torch.zeros(B, V, dtype=torch.bool)
        for b, f in enumerate(feats):
            val_mask[b, : f.n_v] = True
            for k, ((s, e), src, colid) in enumerate(zip(f.cand_spans, f.cand_source, f.cand_column)):
                source[b, k] = src
                e = min(e, f.n_q)
                if e > s:
                    for p in range(s, e):
                        item_idx.append(b * V + k)
                        pos_idx.append(b * N + p)
                        weights.append(1.0 / (e - s))
                if colid >= 0:
                    colref[b, k] = f.n_q + colid
                    has_col[b, k] = True
        flat = mem.reshape(B * N, D)
        span = torch.zeros(B * V, D, dtype=mem.dtype, device=device)
        if item_idx:
            w = torch.tensor(weights, dtype=mem.dtype, device=device).unsqueeze(-1)
            span.index_add_(0, torch.tensor(item_idx, device=device), flat[torch.tensor(pos_idx, device=device)] * w)
        span = span.view(B, V, D)
        colref, has_col, source, val_mask = (t.to(device) for t in (colref, has_col, source, val_mask))
        colvec = torch.gather(mem, 1, colref.unsqueeze(-1).expand(B, V, D)) * has_col.unsqueeze(-1).to(mem.dtype)
        val = self.val_proj(torch.cat([span, colvec], -1)) + self.val_source(source).to(mem.dtype)
        return EncoderOutputs(mem, layout.mask, col, col_mask, tab, tab_mask, val, val_mask)

    # -------------------------------------------------------------- forward
    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        enc = self.encode(batch)
        return self.decoder(enc, batch.steps)

    @torch.no_grad()
    def predict(self, batch: Batch, beam_size: int = 1, max_steps: int = 250) -> list[dict]:
        enc = self.encode(batch)
        sizes = [(f.n_c, f.n_t, f.n_v) for f in batch.feats]
        if beam_size <= 1:
            return self.decoder.greedy(enc, sizes, max_steps)
        out = []
        for b in range(len(batch.feats)):
            e = enc.select(torch.tensor([b], device=enc.memory.device))
            out.append(self.decoder.beam_search(e, sizes[b], beam_size, max_steps))
        return out
