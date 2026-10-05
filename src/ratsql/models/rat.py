r"""Relation-Aware Transformer (RAT) encoder.

Relation-aware self-attention (Shaw et al., 2018; RAT-SQL Eq. 2).  For input
elements ``x_1..x_N`` and a relation id ``r_ij`` for every ordered pair, head
``h`` computes

.. math::

    e^{(h)}_{ij} = \frac{x_i W_Q^{(h)} \left(x_j W_K^{(h)} + \mathbf{r}^K_{ij}\right)^\top}{\sqrt{d_z / H}}
    \qquad
    \alpha^{(h)}_{ij} = \operatorname{softmax}_j\left(e^{(h)}_{ij}\right)

.. math::

    z^{(h)}_i = \sum_j \alpha^{(h)}_{ij}\left(x_j W_V^{(h)} + \mathbf{r}^V_{ij}\right),
    \qquad z_i = W_O\,[z^{(1)}_i; \dots; z^{(H)}_i]

followed by residual connections, layer normalisation and a position-wise
feed-forward network:

.. math::

    \tilde y_i = \mathrm{LayerNorm}(x_i + z_i), \qquad
    y_i = \mathrm{LayerNorm}(\tilde y_i + \mathrm{FC}(\mathrm{ReLU}(\mathrm{FC}(\tilde y_i))))

``r^K_ij, r^V_ij`` are learned embeddings of the relation *type* (dimension
``d_z / H``, shared by all heads, as in RAT-SQL).

Efficient computation.  Materialising ``r^K_ij`` for every pair costs
O(N^2 d).  Because relations come from a small vocabulary of size R we use

* ``q_i . r^K_ij = (q_i E_K^T)[r_ij]``: one [N x R] product per head followed
  by a gather;
* ``sum_j a_ij r^V_ij = sum_r (sum_{j: r_ij = r} a_ij) E_V[r]``: a scatter-add
  of attention weights into R buckets followed by an [R x d] product.

``relation_aware_attention_reference`` implements the literal (materialised)
formula; ``tests/test_rat.py`` checks that both agree.

Setting ``use_relations=False`` gives standard (vanilla) multi-head
self-attention with the identical architecture (baseline B / ablation F).
``norm="pre"`` (default) follows the official RAT-SQL implementation
(pre-norm sublayers + final LayerNorm); ``norm="post"`` follows the equations
in the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relation_aware_attention(
    q: torch.Tensor,  # [B, H, N, d]
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor | None,  # [B, N, N] relation ids
    rel_k: torch.Tensor | None,  # [R, d]
    rel_v: torch.Tensor | None,  # [R, d]
    key_mask: torch.Tensor | None,  # [B, N] True = valid
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N, d = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2))  # [B,H,N,N]
    if rel is not None:
        R = rel_k.shape[0]
        qr = torch.matmul(q, rel_k.t())  # [B,H,N,R]
        idx = rel.unsqueeze(1).expand(B, H, N, N)
        scores = scores + torch.gather(qr, -1, idx)
    scores = scores / math.sqrt(d)
    scores = scores.float()
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # fully-masked rows (padding queries)
    attn_d = dropout(attn) if dropout is not None else attn
    attn_d = attn_d.to(v.dtype)
    out = torch.matmul(attn_d, v)  # [B,H,N,d]
    if rel is not None:
        bucket = torch.zeros(B, H, N, R, dtype=attn_d.dtype, device=attn_d.device)
        bucket.scatter_add_(-1, idx, attn_d)
        out = out + torch.matmul(bucket, rel_v.to(attn_d.dtype))
    return out, attn


def relation_aware_attention_reference(q, k, v, rel, rel_k, rel_v, key_mask=None):
    """Literal implementation with materialised per-pair relation vectors (for tests)."""
    B, H, N, d = q.shape
    rk = rel_k[rel]  # [B,N,N,d]
    rv = rel_v[rel]
    scores = torch.einsum("bhid,bhjd->bhij", q, k) + torch.einsum("bhid,bijd->bhij", q, rk)
    scores = scores / math.sqrt(d)
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    attn = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
    out = torch.einsum("bhij,bhjd->bhid", attn, v) + torch.einsum("bhij,bijd->bhid", attn, rv)
    return out, attn


class RelationAwareMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_relations: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.h, self.d_k = d_model, num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_relations = use_relations

    def forward(self, x, rel, rel_k, rel_v, mask):
        B, N, _ = x.shape

        def split(t):
            return t.view(B, N, self.h, self.d_k).transpose(1, 2)

        q, k, v = split(self.w_q(x)), split(self.w_k(x)), split(self.w_v(x))
        if self.use_relations:
            out, attn = relation_aware_attention(q, k, v, rel, rel_k, rel_v, mask, self.dropout)
        else:
            out, attn = relation_aware_attention(q, k, v, None, None, None, mask, self.dropout)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.w_o(out), attn


class RATLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, num_relations: int, dropout: float = 0.1, use_relations: bool = True, norm: str = "pre", relation_embeddings: tuple[nn.Embedding, nn.Embedding] | None = None):
        super().__init__()
        self.attn = RelationAwareMultiHeadAttention(d_model, num_heads, dropout, use_relations)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_style = norm
        self.use_relations = use_relations
        if use_relations:
            if relation_embeddings is None:
                d_k = d_model // num_heads
                self.rel_k = nn.Embedding(num_relations, d_k)
                self.rel_v = nn.Embedding(num_relations, d_k)
            else:
                self.rel_k, self.rel_v = relation_embeddings
        else:
            self.rel_k = self.rel_v = None

    def forward(self, x, rel, mask):
        rk = self.rel_k.weight if self.use_relations else None
        rv = self.rel_v.weight if self.use_relations else None
        if self.norm_style == "pre":
            h, attn = self.attn(self.norm1(x), rel, rk, rv, mask)
            x = x + self.drop(h)
            x = x + self.drop(self.ff(self.norm2(x)))
        else:
            h, attn = self.attn(x, rel, rk, rv, mask)
            x = self.norm1(x + self.drop(h))
            x = self.norm2(x + self.drop(self.ff(x)))
        return x, attn


class RATEncoder(nn.Module):
    """Stack of relation-aware (or vanilla) transformer layers."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        num_relations: int = 64,
        dropout: float = 0.1,
        use_relations: bool = True,
        norm: str = "pre",
        share_relation_embeddings: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.use_relations = use_relations
        shared = None
        if use_relations and share_relation_embeddings and num_layers > 0:
            d_k = d_model // num_heads
            shared = (nn.Embedding(num_relations, d_k), nn.Embedding(num_relations, d_k))
            self.shared_rel_k, self.shared_rel_v = shared
        self.layers = nn.ModuleList(
            [RATLayer(d_model, num_heads, ff_dim, num_relations, dropout, use_relations, norm, shared) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model) if (norm == "pre" and num_layers > 0) else nn.Identity()
        self.output_dim = d_model
        self.capture_attention = False
        self.attention_maps: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, rel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        self.attention_maps = []
        for layer in self.layers:
            x, attn = layer(x, rel, mask)
            if self.capture_attention:
                self.attention_maps.append(attn.detach().float().cpu())
        return self.final_norm(x) * mask.unsqueeze(-1).to(x.dtype)
