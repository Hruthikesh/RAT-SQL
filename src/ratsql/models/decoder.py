r"""LSTM tree decoder over grammar actions (TRANX / RAT-SQL style).

At step t the decoder LSTM is updated with

.. math::

    m_t, h_t = f_{LSTM}([a_{t-1}; z_t; h_{p_t}; a_{p_t}; n_{f_t}],\ m_{t-1}, h_{t-1})

* ``a_{t-1}``  embedding of the previous action (rule embedding, or a
  projection of the encoding of the selected column / table / value),
* ``z_t``      multi-head attention over the encoder memory with query h_{t-1},
* ``h_{p_t}``  LSTM state at the step that created the parent node
  ("parent feeding"),
* ``a_{p_t}``  embedding of the parent node's constructor (rule),
* ``n_{f_t}``  embedding of the frontier node type (+ field).

Action distributions (all computed from h_t):

* ``ApplyRule[R]``:      softmax over rules of ``MLP(h_t)``;
* ``SelectColumn[i]``,   ``SelectTable[j]``: schema pointers.  The default
  ``memory_aligned`` pointer follows RAT-SQL Sec. 3.4:
  ``P(i) = sum_j lambda_j L_{j,i}`` with ``lambda = softmax_j(h_t W_Q (y_j W_K)^T)``
  over the whole memory and a memory-schema alignment matrix
  ``L = softmax_i(y_j W'_Q (c_i W'_K)^T)``; ``direct`` uses a bilinear pointer;
* ``SelectValue[k]``:    bilinear pointer over value candidates (our addition
  so that predicted SQL is executable).

Grammar constraints: the scores of all four action kinds are concatenated
into one "union" vector; the grammar mask keeps only the actions that are
valid for the current frontier (right kind; for rules, the constructors of the
frontier type plus Reduce where allowed).  With ``use_grammar_mask=False``
(ablation H) the softmax runs over the whole union at every step, during both
training and inference, and ill-formed trees are detected by the transition
system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ratsql.sql.grammar import Grammar
from ratsql.sql.transition import COLUMN, RULE, TABLE, VALUE, DecodeState, InvalidActionError

NEG_INF = -1e9


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.float().masked_fill(~mask, float("-inf"))
    return torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)


class DirectPointer(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, proj_dim: int):
        super().__init__()
        self.q = nn.Linear(query_dim, proj_dim)
        self.k = nn.Linear(key_dim, proj_dim)
        self.scale = 1.0 / math.sqrt(proj_dim)

    def prepare(self, keys, key_mask, memory, mem_mask):
        return {"k": self.k(keys)}

    def forward(self, query: torch.Tensor, state: dict) -> torch.Tensor:
        # query [B, S, Dq] -> logits [B, S, K]
        return torch.matmul(self.q(query), state["k"].transpose(-1, -2)).float() * self.scale


class MemoryAlignedPointer(nn.Module):
    """RAT-SQL pointer: P(i) = sum_j lambda_j L_{j,i} (returns log-probabilities).

    ``fp32=True`` (the default in ``configs/base.yaml``) computes the mixture ``lambda @ L``
    in float32 even under fp16 autocast.  In fp16, probabilities below ~6e-8 underflow to 0
    and ``log`` returns -inf, so the teacher-forced loss of a batch becomes infinite whenever
    a gold column is very unlikely; with AMP the GradScaler then *skips that optimizer step*.
    The first round of experiments ran with ``fp32=False`` and did skip steps; they were
    re-run with the fix (the superseded runs are kept in experiments/superseded_fp16_pointer).
    The code default stays ``False`` so that those old checkpoints still reload unchanged.
    """

    def __init__(self, query_dim: int, mem_dim: int, proj_dim: int, fp32: bool = False):
        super().__init__()
        self.lam_q = nn.Linear(query_dim, proj_dim)
        self.lam_k = nn.Linear(mem_dim, proj_dim)
        self.align_q = nn.Linear(mem_dim, proj_dim)
        self.align_k = nn.Linear(mem_dim, proj_dim)
        self.scale = 1.0 / math.sqrt(proj_dim)
        self.fp32 = fp32

    def prepare(self, keys, key_mask, memory, mem_mask):
        align = torch.matmul(self.align_q(memory), self.align_k(keys).transpose(-1, -2)) * self.scale  # [B,N,K]
        align = _masked_softmax(align, key_mask[:, None, :])
        return {"mem_k": self.lam_k(memory), "mem_mask": mem_mask, "align": align}

    def forward(self, query: torch.Tensor, state: dict) -> torch.Tensor:
        lam = torch.matmul(self.lam_q(query), state["mem_k"].transpose(-1, -2)) * self.scale  # [B,S,N]
        lam = _masked_softmax(lam, state["mem_mask"][:, None, :])
        if self.fp32:
            with torch.autocast(device_type=lam.device.type, enabled=False):
                p = torch.matmul(lam.float(), state["align"].float())  # [B,S,K]
                return torch.log(p.clamp_min(1e-12))
        p = torch.matmul(lam, state["align"])  # [B,S,K]
        return torch.log(p.clamp_min(1e-12))

    def alignment(self, state: dict) -> torch.Tensor:
        return state["align"]


@dataclass
class EncoderOutputs:
    memory: torch.Tensor  # [B, N, D]
    mem_mask: torch.Tensor  # [B, N]
    col: torch.Tensor  # [B, C, D]
    col_mask: torch.Tensor
    tab: torch.Tensor  # [B, T, D]
    tab_mask: torch.Tensor
    val: torch.Tensor  # [B, V, D]
    val_mask: torch.Tensor

    def select(self, idx: torch.Tensor) -> "EncoderOutputs":
        return EncoderOutputs(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


class ASTDecoder(nn.Module):
    def __init__(
        self,
        grammar: Grammar,
        enc_dim: int,
        hidden: int = 512,
        action_dim: int = 128,
        type_dim: int = 64,
        att_heads: int = 8,
        dropout: float = 0.2,
        pointer: str = "memory_aligned",
        pointer_dim: int = 256,
        use_grammar_mask: bool = True,
        pointer_fp32: bool = False,
    ):
        super().__init__()
        self.g = grammar
        R = grammar.num_rules
        self.num_rules = R
        self.hidden = hidden
        self.use_grammar_mask = use_grammar_mask
        self.rule_emb = nn.Embedding(R, action_dim)
        self.parent_rule_emb = nn.Embedding(R + 1, action_dim)  # 0 = root
        self.type_emb = nn.Embedding(grammar.num_types, type_dim)
        self.field_emb = nn.Embedding(grammar.num_fields, type_dim)
        self.start_emb = nn.Parameter(torch.zeros(action_dim))
        self.col_act = nn.Linear(enc_dim, action_dim)
        self.tab_act = nn.Linear(enc_dim, action_dim)
        self.val_act = nn.Linear(enc_dim, action_dim)
        # multi-head context attention with query h_{t-1}
        assert enc_dim % att_heads == 0
        self.att_heads = att_heads
        self.att_q = nn.Linear(hidden, enc_dim)
        self.att_k = nn.Linear(enc_dim, enc_dim)
        self.att_v = nn.Linear(enc_dim, enc_dim)
        self.att_o = nn.Linear(enc_dim, enc_dim)
        in_dim = action_dim + enc_dim + hidden + action_dim + type_dim
        self.cell = nn.LSTMCell(in_dim, hidden)
        self.init_h = nn.Linear(enc_dim, hidden)
        self.init_c = nn.Linear(enc_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.rule_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, R))
        if pointer == "memory_aligned":
            self.col_ptr = MemoryAlignedPointer(hidden, enc_dim, pointer_dim, fp32=pointer_fp32)
            self.tab_ptr = MemoryAlignedPointer(hidden, enc_dim, pointer_dim, fp32=pointer_fp32)
        else:
            self.col_ptr = DirectPointer(hidden, enc_dim, pointer_dim)
            self.tab_ptr = DirectPointer(hidden, enc_dim, pointer_dim)
        self.val_ptr = DirectPointer(hidden, enc_dim, pointer_dim)
        self.register_buffer("mask_table", torch.tensor(grammar.mask_table, dtype=torch.bool), persistent=False)

    # ---------------------------------------------------------------- shared
    def _init_state(self, enc: EncoderOutputs):
        m = enc.mem_mask.unsqueeze(-1).to(enc.memory.dtype)
        pooled = (enc.memory * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.tanh(self.init_h(pooled)), torch.tanh(self.init_c(pooled))

    def _prepare_attention(self, enc: EncoderOutputs):
        B, N, D = enc.memory.shape
        hh = self.att_heads
        k = self.att_k(enc.memory).view(B, N, hh, D // hh).transpose(1, 2)  # [B,h,N,d]
        v = self.att_v(enc.memory).view(B, N, hh, D // hh).transpose(1, 2)
        return k, v

    @staticmethod
    def _attention_bias(mem_mask: torch.Tensor) -> torch.Tensor:
        """Additive mask [B,1,1,N] (0 for memory, -inf for padding); every example has memory."""
        return torch.zeros(mem_mask.shape, dtype=torch.float32, device=mem_mask.device).masked_fill(~mem_mask, float("-inf"))[:, None, None, :]

    def _context(self, h, k, v, mem_mask, bias=None):
        B = h.shape[0]
        hh, d = k.shape[1], k.shape[3]
        q = self.att_q(h).view(B, hh, 1, d) * (1.0 / math.sqrt(d))
        bias = self._attention_bias(mem_mask) if bias is None else bias
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)).float() + bias, dim=-1).to(v.dtype)  # [B,h,1,N]
        ctx = torch.matmul(attn, v).reshape(B, hh * d)
        return self.att_o(ctx)

    def _prepare_pointers(self, enc: EncoderOutputs):
        return (
            self.col_ptr.prepare(enc.col, enc.col_mask, enc.memory, enc.mem_mask),
            self.tab_ptr.prepare(enc.tab, enc.tab_mask, enc.memory, enc.mem_mask),
            self.val_ptr.prepare(enc.val, enc.val_mask, enc.memory, enc.mem_mask),
        )

    def _embed_actions(self, kind: torch.Tensor, target: torch.Tensor, enc: EncoderOutputs) -> torch.Tensor:
        """Embedding of actions ``(kind, target)`` of shape [B, S] -> [B, S, A]."""
        B, S = kind.shape
        tgt = target.clamp(min=0)
        rule = self.rule_emb(torch.where(kind == RULE, tgt, torch.zeros_like(tgt)).clamp(max=self.num_rules - 1))

        def gather(x, t):
            idx = t.clamp(max=x.shape[1] - 1).unsqueeze(-1).expand(B, S, x.shape[-1])
            return torch.gather(x, 1, idx)

        col = self.col_act(gather(enc.col, torch.where(kind == COLUMN, tgt, torch.zeros_like(tgt))))
        tab = self.tab_act(gather(enc.tab, torch.where(kind == TABLE, tgt, torch.zeros_like(tgt))))
        val = self.val_act(gather(enc.val, torch.where(kind == VALUE, tgt, torch.zeros_like(tgt))))
        k = kind.unsqueeze(-1)
        out = torch.where(k == RULE, rule.to(col.dtype), col)
        out = torch.where(k == TABLE, tab, out)
        out = torch.where(k == VALUE, val, out)
        return out

    def _union_logits(self, H: torch.Tensor, ptr_states) -> torch.Tensor:
        rule = self.rule_head(H).float()
        col = self.col_ptr(H, ptr_states[0])
        tab = self.tab_ptr(H, ptr_states[1])
        val = self.val_ptr(H, ptr_states[2])
        return torch.cat([rule, col, tab, val], dim=-1)

    def _union_mask(self, kind, mask_id, enc: EncoderOutputs, grammar: bool) -> torch.Tensor:
        """[B, S, U] boolean mask of allowed actions."""
        B, S = kind.shape
        C, T, V = enc.col_mask.shape[1], enc.tab_mask.shape[1], enc.val_mask.shape[1]
        col = enc.col_mask[:, None, :].expand(B, S, C)
        tab = enc.tab_mask[:, None, :].expand(B, S, T)
        val = enc.val_mask[:, None, :].expand(B, S, V)
        if not grammar:
            rules = torch.ones(B, S, self.num_rules, dtype=torch.bool, device=kind.device)
            return torch.cat([rules, col, tab, val], -1)
        rules = self.mask_table[mask_id.clamp(min=0)] & (kind == RULE).unsqueeze(-1)
        col = col & (kind == COLUMN).unsqueeze(-1)
        tab = tab & (kind == TABLE).unsqueeze(-1)
        val = val & (kind == VALUE).unsqueeze(-1)
        return torch.cat([rules, col, tab, val], -1)

    def _union_target(self, kind, target, enc: EncoderOutputs) -> torch.Tensor:
        R = self.num_rules
        C, T = enc.col_mask.shape[1], enc.tab_mask.shape[1]
        off = torch.zeros_like(target)
        off = torch.where(kind == COLUMN, torch.full_like(target, R), off)
        off = torch.where(kind == TABLE, torch.full_like(target, R + C), off)
        off = torch.where(kind == VALUE, torch.full_like(target, R + C + T), off)
        return target.clamp(min=0) + off

    def _decode_union(self, u: int, C: int, T: int) -> tuple[int, int]:
        R = self.num_rules
        if u < R:
            return RULE, u
        if u < R + C:
            return COLUMN, u - R
        if u < R + C + T:
            return TABLE, u - R - C
        return VALUE, u - R - C - T

    # ------------------------------------------------------- teacher forcing
    def forward(self, enc: EncoderOutputs, steps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        kind, target = steps["kind"], steps["target"]
        B, S = kind.shape
        step_mask = steps["step_mask"]
        A = self.start_emb.shape[0]
        act = self._embed_actions(kind, target, enc)
        prev = torch.cat([self.start_emb.to(act.dtype).expand(B, 1, A), act[:, :-1]], 1)
        parent_rule = self.parent_rule_emb(steps["parent_rule"] + 1)
        frontier = self.type_emb(steps["type_id"]) + self.field_emb(steps["field_id"])
        att_k, att_v = self._prepare_attention(enc)
        ptr_states = self._prepare_pointers(enc)
        h, c = self._init_state(enc)
        # buffer of past hidden states; slot 0 = zeros (root parent)
        buf = h.new_zeros(B, S + 1, self.hidden)
        bidx = torch.arange(B, device=kind.device)
        parent_slot = steps["parent_step"] + 1  # -1 -> 0
        outs = []
        att_bias = self._attention_bias(enc.mem_mask)
        for t in range(S):
            ctx = self._context(h, att_k, att_v, enc.mem_mask, att_bias)
            ph = buf[bidx, parent_slot[:, t]]
            x = torch.cat([prev[:, t], ctx.to(prev.dtype), ph.to(prev.dtype), parent_rule[:, t].to(prev.dtype), frontier[:, t].to(prev.dtype)], -1)
            h, c = self.cell(self.drop(x), (h, c))
            outs.append(h)
            buf = buf.index_put((bidx, torch.full_like(bidx, t + 1)), h.to(buf.dtype))
        H = self.drop(torch.stack(outs, 1))
        logits = self._union_logits(H, ptr_states)  # [B,S,U] float32
        mask = self._union_mask(kind, steps["mask_id"], enc, self.use_grammar_mask)
        tgt_u = self._union_target(kind, target, enc)
        logits = logits.masked_fill(~mask, NEG_INF)
        logp = torch.log_softmax(logits, -1)
        tok_logp = logp.gather(-1, tgt_u.unsqueeze(-1)).squeeze(-1)
        m = step_mask.to(tok_logp.dtype)
        loss = -(tok_logp * m).sum() / B
        with torch.no_grad():
            correct = ((logits.argmax(-1) == tgt_u) & step_mask).sum()
            n_steps = step_mask.sum()
        return {"loss": loss, "step_correct": correct, "num_steps": n_steps, "nll_sum": -(tok_logp * m).sum().detach()}

    # ----------------------------------------------------------- inference
    @torch.no_grad()
    def greedy(self, enc: EncoderOutputs, sizes: list[tuple[int, int, int]], max_steps: int = 250, use_grammar_mask: bool | None = None) -> list[dict]:
        """Batched greedy decoding. ``sizes[b] = (n_columns, n_tables, n_values)``."""
        grammar_mask = self.use_grammar_mask if use_grammar_mask is None else use_grammar_mask
        B = enc.memory.shape[0]
        dev = enc.memory.device
        C, T = enc.col_mask.shape[1], enc.tab_mask.shape[1]
        states = [DecodeState(self.g, *sizes[b]) for b in range(B)]
        failed = [None] * B
        scores = [0.0] * B
        att_k, att_v = self._prepare_attention(enc)
        ptr_states = self._prepare_pointers(enc)
        h, c = self._init_state(enc)
        buf = h.new_zeros(B, max_steps + 1, self.hidden)
        bidx = torch.arange(B, device=dev)
        prev = self.start_emb.expand(B, -1).to(h.dtype)
        for t in range(max_steps):
            active = [b for b in range(B) if failed[b] is None and not states[b].is_done]
            if not active:
                break
            fr = [states[b].frontier() if b in active else None for b in range(B)]
            type_id = torch.tensor([f.type_id if f else 0 for f in fr], device=dev)
            field_id = torch.tensor([f.field_id if f else 0 for f in fr], device=dev)
            parent_rule = torch.tensor([f.parent_rule + 1 if f else 0 for f in fr], device=dev)
            parent_slot = torch.tensor([f.parent_step + 1 if f else 0 for f in fr], device=dev)
            kind = torch.tensor([f.kind if f else RULE for f in fr], device=dev)
            mask_id = torch.tensor([f.mask_id if f else 0 for f in fr], device=dev)
            ctx = self._context(h, att_k, att_v, enc.mem_mask)
            ph = buf[bidx, parent_slot]
            x = torch.cat([prev, ctx.to(prev.dtype), ph.to(prev.dtype), self.parent_rule_emb(parent_rule).to(prev.dtype), (self.type_emb(type_id) + self.field_emb(field_id)).to(prev.dtype)], -1)
            h, c = self.cell(x, (h, c))
            buf[:, t + 1] = h.to(buf.dtype)
            logits = self._union_logits(h.unsqueeze(1), ptr_states)[:, 0]  # [B,U]
            mask = self._union_mask(kind.unsqueeze(1), mask_id.unsqueeze(1), enc, grammar_mask)[:, 0]
            logits = logits.masked_fill(~mask, NEG_INF)
            logp = torch.log_softmax(logits, -1)
            best = logp.argmax(-1)
            best_lp = logp.gather(-1, best.unsqueeze(-1)).squeeze(-1)
            chosen_kind = torch.zeros(B, dtype=torch.long, device=dev)
            chosen_tgt = torch.zeros(B, dtype=torch.long, device=dev)
            best_l = best.tolist()
            best_lp_l = best_lp.tolist()
            for b in active:
                k_, i_ = self._decode_union(best_l[b], C, T)
                try:
                    states[b].apply((k_, i_))
                    scores[b] += best_lp_l[b]
                except InvalidActionError as e:
                    failed[b] = f"step {t}: {e}"
                chosen_kind[b], chosen_tgt[b] = k_, i_
            prev = self._embed_actions(chosen_kind.unsqueeze(1), chosen_tgt.unsqueeze(1), enc)[:, 0].to(h.dtype)
        results = []
        for b in range(B):
            st = states[b]
            if failed[b] is None and not st.is_done:
                failed[b] = f"exceeded max_steps={max_steps}"
            results.append({"ast": st.ast if failed[b] is None else None, "actions": list(st.actions), "error": failed[b], "score": scores[b]})
        return results

    @torch.no_grad()
    def beam_search(self, enc: EncoderOutputs, size: tuple[int, int, int], beam_size: int = 5, max_steps: int = 250) -> dict:
        """Beam search for a single example (``enc`` has batch size 1)."""
        dev = enc.memory.device
        C, T = enc.col_mask.shape[1], enc.tab_mask.shape[1]
        att_k, att_v = self._prepare_attention(enc)
        ptr_states = self._prepare_pointers(enc)
        h, c = self._init_state(enc)
        hyps = [{"state": DecodeState(self.g, *size), "score": 0.0}]
        buf = h.new_zeros(1, max_steps + 1, self.hidden)
        prev = self.start_emb.unsqueeze(0).to(h.dtype)
        finished: list[dict] = []

        def expand(x, n):
            return x.expand(n, *x.shape[1:]) if x.shape[0] == 1 else x

        for t in range(max_steps):
            n = len(hyps)
            e = enc.select(torch.zeros(n, dtype=torch.long, device=dev))
            ak, av = expand(att_k, n), expand(att_v, n)
            ps = tuple({k: expand(v, n) for k, v in s.items()} for s in ptr_states)
            frs = [hp["state"].frontier() for hp in hyps]
            type_id = torch.tensor([f.type_id for f in frs], device=dev)
            field_id = torch.tensor([f.field_id for f in frs], device=dev)
            parent_rule = torch.tensor([f.parent_rule + 1 for f in frs], device=dev)
            parent_slot = torch.tensor([f.parent_step + 1 for f in frs], device=dev)
            kind = torch.tensor([f.kind for f in frs], device=dev)
            mask_id = torch.tensor([f.mask_id for f in frs], device=dev)
            ctx = self._context(h, ak, av, e.mem_mask)
            ph = buf[torch.arange(n, device=dev), parent_slot]
            x = torch.cat([prev, ctx.to(prev.dtype), ph.to(prev.dtype), self.parent_rule_emb(parent_rule).to(prev.dtype), (self.type_emb(type_id) + self.field_emb(field_id)).to(prev.dtype)], -1)
            h, c = self.cell(x, (h, c))
            buf = buf.clone()
            buf[:, t + 1] = h.to(buf.dtype)
            logits = self._union_logits(h.unsqueeze(1), ps)[:, 0]
            mask = self._union_mask(kind.unsqueeze(1), mask_id.unsqueeze(1), e, self.use_grammar_mask)[:, 0]
            logp = torch.log_softmax(logits.masked_fill(~mask, NEG_INF), -1)
            k = min(beam_size, logp.shape[-1])
            top_lp, top_u = logp.topk(k, -1)
            cands = []
            for i in range(n):
                for lp, u in zip(top_lp[i].tolist(), top_u[i].tolist()):
                    if lp <= NEG_INF / 2:
                        continue
                    cands.append((hyps[i]["score"] + lp, i, u))
            cands.sort(key=lambda x: -x[0])
            new_hyps, keep_idx = [], []
            for score, i, u in cands:
                if len(new_hyps) + len(finished) >= beam_size * 2 or len(new_hyps) >= beam_size:
                    break
                st = hyps[i]["state"].copy()
                try:
                    st.apply(self._decode_union(u, C, T))
                except InvalidActionError:
                    continue
                if st.is_done:
                    finished.append({"state": st, "score": score})
                else:
                    new_hyps.append({"state": st, "score": score, "action": self._decode_union(u, C, T)})
                    keep_idx.append(i)
            if not new_hyps or len(finished) >= beam_size:
                break
            idx = torch.tensor(keep_idx, device=dev)
            h, c, buf = h[idx], c[idx], buf[idx]
            kinds = torch.tensor([hp["action"][0] for hp in new_hyps], device=dev).unsqueeze(1)
            tgts = torch.tensor([hp["action"][1] for hp in new_hyps], device=dev).unsqueeze(1)
            e2 = enc.select(torch.zeros(len(new_hyps), dtype=torch.long, device=dev))
            prev = self._embed_actions(kinds, tgts, e2)[:, 0].to(h.dtype)
            hyps = new_hyps
        if not finished:
            return {"ast": None, "actions": list(hyps[0]["state"].actions) if hyps else [], "error": "beam search found no complete tree", "score": float("-inf")}
        best = max(finished, key=lambda x: x["score"])
        return {"ast": best["state"].ast, "actions": list(best["state"].actions), "error": None, "score": best["score"]}
