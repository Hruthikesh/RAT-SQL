"""Relation-aware attention and the RAT encoder."""

import torch

from ratsql.models.rat import RATEncoder, RATLayer, relation_aware_attention, relation_aware_attention_reference


def _inputs(B=2, H=4, N=7, d=8, R=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, N, d, generator=g) for _ in range(3))
    rel = torch.randint(0, R, (B, N, N), generator=g)
    rk, rv = torch.randn(R, d, generator=g), torch.randn(R, d, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 5:] = False
    return q, k, v, rel, rk, rv, mask


def test_efficient_equals_reference():
    q, k, v, rel, rk, rv, mask = _inputs()
    out, attn = relation_aware_attention(q, k, v, rel, rk, rv, mask)
    ref_out, ref_attn = relation_aware_attention_reference(q, k, v, rel, rk, rv, mask)
    assert torch.allclose(attn, ref_attn, atol=1e-5)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_masking():
    q, k, v, rel, rk, rv, mask = _inputs()
    _, attn = relation_aware_attention(q, k, v, rel, rk, rv, mask)
    assert torch.all(attn[1, :, :, 5:] == 0)  # padded keys get no attention
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))


def test_zero_relation_embeddings_equal_vanilla():
    q, k, v, rel, rk, rv, mask = _inputs()
    zero = torch.zeros_like(rk)
    out_rel, _ = relation_aware_attention(q, k, v, rel, zero, zero, mask)
    out_van, _ = relation_aware_attention(q, k, v, None, None, None, mask)
    assert torch.allclose(out_rel, out_van, atol=1e-6)


def test_relations_change_output():
    torch.manual_seed(0)
    layer = RATLayer(16, 4, 32, num_relations=6, dropout=0.0)
    x = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    r1 = torch.zeros(1, 5, 5, dtype=torch.long)
    r2 = r1.clone()
    r2[0, 0, 3] = 4
    y1, _ = layer(x, r1, mask)
    y2, _ = layer(x, r2, mask)
    assert not torch.allclose(y1[0, 0], y2[0, 0])  # the changed pair affects row 0
    assert torch.allclose(y1[0, 1:], y2[0, 1:], atol=1e-6)  # ... and only row 0 (single layer)


def test_vanilla_encoder_ignores_relations():
    torch.manual_seed(0)
    enc = RATEncoder(16, 16, num_layers=2, num_heads=4, ff_dim=32, num_relations=6, dropout=0.0, use_relations=False)
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    a = enc(x, torch.zeros(2, 5, 5, dtype=torch.long), mask)
    b = enc(x, torch.randint(0, 6, (2, 5, 5)), mask)
    assert torch.allclose(a, b)
    assert not any("rel_k" in n for n, _ in enc.named_parameters())


def test_encoder_shapes_gradients_and_capture():
    torch.manual_seed(0)
    for norm in ("pre", "post"):
        enc = RATEncoder(24, 16, num_layers=3, num_heads=4, ff_dim=32, num_relations=6, dropout=0.1, norm=norm)
        x = torch.randn(2, 6, 24, requires_grad=True)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[0, 4:] = False
        enc.capture_attention = True
        y = enc(x, torch.randint(0, 6, (2, 6, 6)), mask)
        assert y.shape == (2, 6, 16)
        assert torch.all(y[0, 4:] == 0)  # padded positions zeroed
        y.sum().backward()
        assert x.grad is not None and enc.layers[0].rel_k.weight.grad is not None
        assert len(enc.attention_maps) == 3 and enc.attention_maps[0].shape == (2, 4, 6, 6)


def test_shared_relation_embeddings():
    enc = RATEncoder(16, 16, num_layers=3, num_heads=4, ff_dim=32, num_relations=6, share_relation_embeddings=True)
    assert enc.layers[0].rel_k is enc.layers[2].rel_k
    per_layer = RATEncoder(16, 16, num_layers=3, num_heads=4, ff_dim=32, num_relations=6)
    n_shared = sum(p.numel() for p in enc.parameters())
    n_per = sum(p.numel() for p in per_layer.parameters())
    assert n_per - n_shared == 2 * 2 * 6 * 4  # two extra layers x (K,V) x R x d_k
