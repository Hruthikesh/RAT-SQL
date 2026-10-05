"""A controlled synthetic task isolating the value of relation-aware attention.

Each example is a set of N nodes.  Node i carries a random identity token and
a label ``y_i`` in {0..K-1}, both visible in its input embedding.  A random
graph assigns every node exactly one *partner* via a ``link`` relation (the
analogue of a foreign key / schema link); all other pairs get distractor
relations (``same_group`` for a random partition, otherwise ``default``).

* one-hop task: predict the label of your partner, ``y_{p(i)}``;
* two-hop task: predict the label of your partner's partner, ``y_{p(p(i))}``,
  with a partner function that is a random derangement (a permutation without
  fixed points), so every node links to exactly one *other* node (needs 2 layers).

Nothing in the node contents identifies the partner, so a vanilla
transformer (no positions, no relations) cannot do better than guessing
(~1/K), while relation-aware attention can route information along the
``link`` relation.  Train/test graphs are freshly sampled, so the model must
use the relation *type*, not memorise instances.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from ratsql.models.rat import RATEncoder

REL_PAD, REL_SELF, REL_DEFAULT, REL_GROUP, REL_LINK, REL_LINK_REV = range(6)


def _derangement(n: int, gen: torch.Generator) -> torch.Tensor:
    """Random permutation without fixed points, so every node's partner is another node."""
    while True:
        p = torch.randperm(n, generator=gen)
        if not bool((p == torch.arange(n)).any()):
            return p


def sample_batch(batch: int, n: int, k: int, n_ids: int, task: str, gen: torch.Generator):
    labels = torch.randint(0, k, (batch, n), generator=gen)
    ids = torch.randint(0, n_ids, (batch, n), generator=gen)
    rel = torch.full((batch, n, n), REL_DEFAULT, dtype=torch.long)
    groups = torch.randint(0, 3, (batch, n), generator=gen)
    rel[groups[:, :, None] == groups[:, None, :]] = REL_GROUP
    partner = torch.stack([_derangement(n, gen) for _ in range(batch)])
    b = torch.arange(batch)[:, None].expand(batch, n)
    i = torch.arange(n)[None, :].expand(batch, n)
    rel[b, i, partner] = REL_LINK
    rel[b, partner, i] = torch.where(rel[b, partner, i] == REL_LINK, REL_LINK, torch.full_like(partner, REL_LINK_REV))
    idx = torch.arange(n)
    rel[:, idx, idx] = REL_SELF
    if task == "one_hop":
        target = torch.gather(labels, 1, partner)
    else:
        target = torch.gather(labels, 1, torch.gather(partner, 1, partner))
    return ids, labels, rel, target


class NodeClassifier(nn.Module):
    def __init__(self, n_ids: int, k: int, d: int, layers: int, use_relations: bool):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, d)
        self.label_emb = nn.Embedding(k, d)
        self.enc = RATEncoder(d, d, num_layers=layers, num_heads=4, ff_dim=2 * d, num_relations=6, dropout=0.0, use_relations=use_relations)
        self.out = nn.Linear(d, k)

    def forward(self, ids, labels, rel):
        x = self.id_emb(ids) + self.label_emb(labels)
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        return self.out(self.enc(x, rel, mask))


def run(task: str = "one_hop", use_relations: bool = True, n: int = 12, k: int = 8, n_ids: int = 64, d: int = 64, layers: int = 2, steps: int = 1500, batch: int = 64, lr: float = 2e-3, seed: int = 0, eval_every: int = 100, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    test_gen = torch.Generator().manual_seed(10_000 + seed)
    test = sample_batch(1024, n, k, n_ids, task, test_gen)
    model = NodeClassifier(n_ids, k, d, layers, use_relations).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        ids, labels, rel, target = (t.to(device) for t in sample_batch(batch, n, k, n_ids, task, gen))
        loss = nn.functional.cross_entropy(model(ids, labels, rel).reshape(-1, k), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                ids, labels, rel, target = (t.to(device) for t in test)
                acc = (model(ids, labels, rel).argmax(-1) == target).float().mean().item()
            model.train()
            curve.append({"step": step, "test_accuracy": acc, "train_loss": loss.item()})
    return {
        "task": task,
        "attention": "relation-aware" if use_relations else "vanilla",
        "seed": seed,
        "layers": layers,
        "final_test_accuracy": curve[-1]["test_accuracy"],
        "chance": 1.0 / k,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - t0, 1),
        "curve": curve,
    }
