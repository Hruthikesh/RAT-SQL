"""Input encoders producing initial representations of question words and schema elements.

Both encoders consume the same per-example features (word-piece ids of every
question word, column and table) and return a padded tensor laid out as the
relation-aware encoder expects::

    X[b] = [q_1 .. q_n, c_1 .. c_m, t_1 .. t_k, <pad> ...]      shape [B, N_max, H]

``BertInputEncoder`` (RAT-SQL + BERT, Sec. 4 of the paper)
    One BERT pass over ``[CLS] question [SEP] col_1 [SEP] col_2 [SEP] ... tab_k [SEP]``
    (question tokens get segment id 0, schema tokens segment id 1).  Each
    question word / schema element is pooled from its word pieces (mean, or
    the average of the first and last piece as in RAT-SQL).  When the input
    exceeds ``max_length`` the schema is split into several segments, each
    prefixed with the question; question-word vectors are averaged over the
    segments.  RAT-SQL instead dropped over-long training examples; chunking
    keeps every example (documented deviation).

``LSTMInputEncoder`` (ablation I / baseline A: "simpler contextual encoder")
    Static word-piece embeddings (optionally initialised from BERT's input
    embedding matrix) followed by a BiLSTM over the question and a separate
    BiLSTM over every schema element's name, as in the GloVe version of
    RAT-SQL.  There is no joint question-schema contextualisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderBatchLayout:
    """Index tensors describing where each item (word / column / table) lives."""

    pad_index: torch.Tensor  # [B, N_max] item index or -1
    mask: torch.Tensor  # [B, N_max] bool
    n_q: list[int]
    n_c: list[int]
    n_t: list[int]


def _layout(n_q: list[int], n_c: list[int], n_t: list[int], device) -> EncoderBatchLayout:
    sizes = [a + b + c for a, b, c in zip(n_q, n_c, n_t)]
    n_max = max(sizes)
    pad_index = torch.full((len(sizes), n_max), -1, dtype=torch.long)
    offset = 0
    for b, n in enumerate(sizes):
        pad_index[b, :n] = torch.arange(offset, offset + n)
        offset += n
    pad_index = pad_index.to(device)
    return EncoderBatchLayout(pad_index, pad_index >= 0, n_q, n_c, n_t)


def _scatter_items(items: torch.Tensor, layout: EncoderBatchLayout) -> torch.Tensor:
    x = items[layout.pad_index.clamp(min=0)]
    return x * layout.mask.unsqueeze(-1).to(x.dtype)


class BertInputEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        pooling: str = "mean",
        freeze: bool = False,
        gradient_checkpointing: bool = False,
        pretrained: bool = True,
        config_overrides: dict | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel, BertConfig, BertModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        elif config_overrides is not None:  # random init from an explicit config (unit tests / offline)
            self.bert = BertModel(BertConfig(**config_overrides), add_pooling_layer=False)
        else:  # random init with the architecture of ``model_name``
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(model_name), add_pooling_layer=False)
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
        self.max_length = max_length
        self.pooling = pooling
        self.output_dim = self.bert.config.hidden_size
        cfg = self.bert.config
        self.cls_id = getattr(cfg, "cls_token_id", None)
        self.freeze = freeze
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)

    def set_special_ids(self, cls_id: int, sep_id: int, pad_id: int) -> None:
        self.cls_id, self.sep_id, self.pad_id = cls_id, sep_id, pad_id

    # --------------------------------------------------------------- inputs
    def _segments(self, q_pieces: list[list[int]], elem_pieces: list[list[int]]):
        q_flat, q_spans = [], []
        for w in q_pieces:
            q_spans.append((len(q_flat), len(q_flat) + len(w)))
            q_flat.extend(w)
        budget = self.max_length - len(q_flat) - 2
        if budget < 16:
            raise ValueError("question too long for max_length; lower data.max_question_pieces")
        segs: list[list[int]] = []
        cur: list[int] = []
        cur_len = 0
        for e, pieces in enumerate(elem_pieces):
            need = len(pieces) + 1
            if cur and cur_len + need > budget:
                segs.append(cur)
                cur, cur_len = [], 0
            cur.append(e)
            cur_len += need
        segs.append(cur)
        out = []
        for seg in segs:
            ids = [self.cls_id] + q_flat + [self.sep_id]
            types = [0] * len(ids)
            elem_spans = {}
            for e in seg:
                start = len(ids)
                ids.extend(elem_pieces[e])
                elem_spans[e] = (start, len(ids))
                ids.append(self.sep_id)
            types += [1] * (len(ids) - len(types))
            out.append((ids, types, [(s + 1, t + 1) for s, t in q_spans], elem_spans))
        return out

    def _pool_positions(self, start: int, end: int) -> list[int]:
        if end <= start:
            return []
        if self.pooling == "first_last":
            return [start, end - 1] if end - 1 > start else [start]
        if self.pooling == "first":
            return [start]
        return list(range(start, end))

    def forward(self, feats: list, device) -> tuple[torch.Tensor, EncoderBatchLayout]:
        all_ids, all_types = [], []
        item_idx, pos_idx, weights = [], [], []
        n_q, n_c, n_t = [], [], []
        item_offset = 0
        for f in feats:
            elems = f.col_pieces + f.tab_pieces
            segs = self._segments(f.q_pieces, elems)
            nq = len(f.q_pieces)
            n_q.append(nq)
            n_c.append(len(f.col_pieces))
            n_t.append(len(f.tab_pieces))
            seg_base = len(all_ids)
            for s, (ids, types, q_spans, elem_spans) in enumerate(segs):
                all_ids.append(ids)
                all_types.append(types)
                for w, (a, b) in enumerate(q_spans):
                    ps = self._pool_positions(a, b)
                    for p in ps:
                        item_idx.append(item_offset + w)
                        pos_idx.append((seg_base + s, p))
                        weights.append(1.0 / (len(ps) * len(segs)))
                for e, (a, b) in elem_spans.items():
                    ps = self._pool_positions(a, b)
                    for p in ps:
                        item_idx.append(item_offset + nq + e)
                        pos_idx.append((seg_base + s, p))
                        weights.append(1.0 / len(ps))
            item_offset += nq + len(elems)
        L = max(len(x) for x in all_ids)
        S = len(all_ids)
        ids_t = torch.full((S, L), self.pad_id, dtype=torch.long)
        types_t = torch.zeros((S, L), dtype=torch.long)
        att_t = torch.zeros((S, L), dtype=torch.long)
        for i, (ids, types) in enumerate(zip(all_ids, all_types)):
            ids_t[i, : len(ids)] = torch.tensor(ids)
            types_t[i, : len(types)] = torch.tensor(types)
            att_t[i, : len(ids)] = 1
        ids_t, types_t, att_t = ids_t.to(device), types_t.to(device), att_t.to(device)
        if self.freeze:
            with torch.no_grad():
                hidden = self.bert(input_ids=ids_t, attention_mask=att_t, token_type_ids=types_t).last_hidden_state
        else:
            hidden = self.bert(input_ids=ids_t, attention_mask=att_t, token_type_ids=types_t).last_hidden_state
        flat = hidden.reshape(S * L, -1)
        flat_pos = torch.tensor([s * L + p for s, p in pos_idx], dtype=torch.long, device=device)
        w = torch.tensor(weights, dtype=flat.dtype, device=device).unsqueeze(-1)
        items = torch.zeros(item_offset, flat.shape[-1], dtype=flat.dtype, device=device)
        items.index_add_(0, torch.tensor(item_idx, dtype=torch.long, device=device), flat[flat_pos] * w)
        layout = _layout(n_q, n_c, n_t, device)
        return _scatter_items(items, layout), layout


class LSTMInputEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 300, hidden: int = 256, dropout: float = 0.2, pad_id: int = 0, init_embeddings: torch.Tensor | None = None, freeze_embeddings: bool = False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        if init_embeddings is not None:
            with torch.no_grad():
                self.emb.weight.copy_(init_embeddings[:, :emb_dim] if init_embeddings.shape[1] >= emb_dim else nn.functional.pad(init_embeddings, (0, emb_dim - init_embeddings.shape[1])))
        self.emb.weight.requires_grad_(not freeze_embeddings)
        self.q_lstm = nn.LSTM(emb_dim, hidden // 2, batch_first=True, bidirectional=True)
        self.s_lstm = nn.LSTM(emb_dim, hidden // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden
        self.pad_id = pad_id

    def set_special_ids(self, cls_id: int, sep_id: int, pad_id: int) -> None:
        self.pad_id = pad_id

    def _run(self, lstm: nn.LSTM, seqs: list[list[int]], device) -> torch.Tensor:
        lens = [max(1, len(s)) for s in seqs]
        L = max(lens)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            if s:
                ids[i, : len(s)] = torch.tensor(s)
        ids = ids.to(device)
        x = self.dropout(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lens), batch_first=True, enforce_sorted=False)
        out, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        return out  # [n, L, hidden]

    def forward(self, feats: list, device) -> tuple[torch.Tensor, EncoderBatchLayout]:
        n_q, n_c, n_t = [], [], []
        q_seqs, q_word_spans = [], []
        elem_seqs = []
        for f in feats:
            flat, spans = [], []
            for w in f.q_pieces:
                spans.append((len(flat), len(flat) + len(w)))
                flat.extend(w)
            q_seqs.append(flat)
            q_word_spans.append(spans)
            elem_seqs.extend(f.col_pieces + f.tab_pieces)
            n_q.append(len(f.q_pieces))
            n_c.append(len(f.col_pieces))
            n_t.append(len(f.tab_pieces))
        q_out = self._run(self.q_lstm, q_seqs, device)  # [B, Lq, H]
        e_out = self._run(self.s_lstm, elem_seqs, device)  # [E, Le, H]
        H = q_out.shape[-1]
        items = []
        e_ptr = 0
        # mean-pool pieces -> words / elements (loop over examples; sizes are small)
        e_lens = torch.tensor([max(1, len(s)) for s in elem_seqs], device=device, dtype=q_out.dtype)
        e_mask = torch.arange(e_out.shape[1], device=device)[None, :] < e_lens[:, None]
        e_mean = (e_out * e_mask.unsqueeze(-1)).sum(1) / e_lens.unsqueeze(-1)
        for b, f in enumerate(feats):
            spans = q_word_spans[b]
            if spans:
                starts = torch.tensor([s for s, _ in spans], device=device)
                ends = torch.tensor([max(e, s + 1) for s, e in spans], device=device)
                pos = torch.arange(q_out.shape[1], device=device)
                m = (pos[None, :] >= starts[:, None]) & (pos[None, :] < ends[:, None])
                words = (m.to(q_out.dtype) @ q_out[b]) / m.sum(1, keepdim=True).clamp(min=1).to(q_out.dtype)
            else:
                words = q_out.new_zeros((0, H))
            n_e = len(f.col_pieces) + len(f.tab_pieces)
            items.append(words)
            items.append(e_mean[e_ptr : e_ptr + n_e])
            e_ptr += n_e
        items_t = torch.cat(items, 0)
        layout = _layout(n_q, n_c, n_t, device)
        return _scatter_items(self.dropout(items_t), layout), layout
