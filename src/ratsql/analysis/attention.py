"""Inspecting encoder self-attention by relation type.

For every example, layer and head we take the attention matrix ``alpha``
(queries x keys) of the relation-aware (or vanilla) encoder and the example's
relation matrix ``R``.  For each relation type r we report the *attention
lift*:

    lift(r) = mean over pairs (i, j) with R_ij = r of  alpha_ij * n

where n is the number of keys, so lift = 1 means "as much attention as a
uniform distribution would give", lift > 1 means pairs of this type are
preferentially attended.  The vanilla encoder never sees R, so its lifts
show which relations it discovers on its own; the RAT encoder receives R.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from ratsql.data.dataset import collate


@torch.no_grad()
def attention_maps(model, feats, device, batch_size: int = 16, amp: bool = True):
    """Yield (feature, [layer][head, n, n] numpy attention) for each example."""
    model.eval()
    model.rat.capture_attention = True
    ctx = torch.autocast("cuda", dtype=torch.float16) if (amp and device.type == "cuda") else nullcontext()
    try:
        for s in range(0, len(feats), batch_size):
            chunk = feats[s : s + batch_size]
            batch = collate(chunk).to(device)
            with ctx:
                model.encode(batch)
            maps = model.rat.attention_maps  # list over layers of [B, H, N, N]
            for b, f in enumerate(chunk):
                n = f.size
                yield f, [m[b, :, :n, :n].numpy() for m in maps]
    finally:
        model.rat.capture_attention = False


def relation_lift(model, feats, device, num_relations: int, batch_size: int = 16, analysis_relations=None) -> dict:
    """Return lift[layer][relation] (averaged over heads) and pair counts.

    The model runs on its own features (the relation matrices it was trained
    with); ``analysis_relations(feature) -> [n, n] ids`` optionally supplies the
    matrix used to *group* attention (e.g. the full vocabulary for a model
    trained without foreign keys or with the vanilla encoder).
    """
    sums = None
    counts = np.zeros(num_relations)
    for f, maps in attention_maps(model, feats, device, batch_size):
        if sums is None:
            sums = np.zeros((len(maps), num_relations))
        rel_m = f.relations if analysis_relations is None else analysis_relations(f)
        rel = rel_m.astype(np.int64).ravel()
        n = f.size
        counts += np.bincount(rel, minlength=num_relations)
        for layer, a in enumerate(maps):
            lift = (a.mean(0) * n).ravel()  # head-averaged
            sums[layer] += np.bincount(rel, weights=lift, minlength=num_relations)
    if sums is None:
        return {"lift": np.zeros((0, num_relations)), "counts": counts}
    return {"lift": sums / np.maximum(counts, 1)[None, :], "counts": counts}


def example_heatmap(model, feat, device, layer: int = -1) -> np.ndarray:
    """Head-averaged attention of question tokens over (columns + tables) for one example."""
    for _, maps in attention_maps(model, [feat], device, 1):
        a = maps[layer].mean(0)
        return a[: feat.n_q, feat.n_q :]
    raise ValueError("no attention captured")
