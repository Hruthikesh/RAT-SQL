"""BERT / LSTM input encoders, the AST decoder and the full model (tiny random configs)."""

import pytest
import torch

from conftest import tiny_model_cfg
from ratsql.data.dataset import FeatureBuilder, collate, load_records
from ratsql.models.model import Text2SQLModel, TokenizerInfo
from ratsql.schema.relations import RelationMatrixBuilder
from ratsql.sql.grammar import default_grammar
from ratsql.sql.transition import replay


@pytest.fixture(scope="module")
def feats(processed_synthetic, synthetic_schemas, tiny_tokenizer):
    fb = FeatureBuilder(tiny_tokenizer, RelationMatrixBuilder({}), synthetic_schemas)
    recs = load_records(processed_synthetic / "train.jsonl", require_actions=True, limit=12)
    return [fb.build(r, i) for i, r in enumerate(recs)]


def _model(tok, **kw):
    torch.manual_seed(0)
    info = TokenizerInfo(len(tok), tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    return Text2SQLModel(tiny_model_cfg(vocab_size=len(tok), **kw), info, RelationMatrixBuilder({}).num_relations, pretrained=False)


def test_features(feats, tiny_tokenizer):
    f = feats[0]
    assert f.relations.shape == (f.size, f.size)
    assert f.steps is not None and f.steps.shape[1] == 7
    assert all(len(p) >= 1 for p in f.q_pieces + f.col_pieces + f.tab_pieces)
    assert f.cand_source[0] == 0  # <unk> first
    b = collate(feats[:4])
    assert b.relations.shape[0] == 4 and b.steps["step_mask"].sum() == sum(len(x.steps) for x in feats[:4])


@pytest.mark.parametrize("encoder", ["bert", "lstm"])
def test_input_encoder_layout(feats, tiny_tokenizer, encoder):
    m = _model(tiny_tokenizer, encoder=encoder)
    batch = collate(feats[:3])
    x, layout = m.encoder(batch.feats, torch.device("cpu"))
    assert x.shape[:2] == (3, max(f.size for f in feats[:3]))
    for b, f in enumerate(feats[:3]):
        assert layout.mask[b].sum() == f.size
        assert torch.all(x[b, f.size :] == 0)


def test_bert_long_schema_chunking(feats, tiny_tokenizer):
    """With a small max_length the schema is split into several segments; outputs keep their layout."""
    m = _model(tiny_tokenizer)
    m.encoder.max_length = 64
    f = feats[0]
    segs = m.encoder._segments(f.q_pieces, f.col_pieces + f.tab_pieces)
    assert len(segs) > 1
    assert all(len(ids) <= 64 for ids, _, _, _ in segs)
    covered = sorted(e for _, _, _, spans in segs for e in spans)
    assert covered == list(range(len(f.col_pieces) + len(f.tab_pieces)))
    x, layout = m.encoder([f], torch.device("cpu"))
    assert x.shape == (1, f.size, m.encoder.output_dim)


def test_parameter_counts(tiny_tokenizer):
    m = _model(tiny_tokenizer)
    c = m.parameter_counts()
    assert c["total"] == c["input_encoder"] + c["rat_encoder"] + c["decoder"] + c["value_encoder"]
    assert len(m.pretrained_parameters()) > 0


def test_teacher_forcing_loss_and_overfit(feats, tiny_tokenizer):
    m = _model(tiny_tokenizer)
    batch = collate(feats[:4])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        out = m(batch)
        assert torch.isfinite(out["loss"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        first = first if first is not None else out["loss"].item()
    assert out["loss"].item() < 0.2 * first
    assert out["step_correct"].item() / out["num_steps"].item() > 0.9


def test_greedy_decoding_produces_valid_trees(feats, tiny_tokenizer):
    m = _model(tiny_tokenizer).eval()
    batch = collate(feats[:5])
    results = m.predict(batch, max_steps=150)
    for f, r in zip(feats[:5], results):
        if r["ast"] is not None:  # untrained model may run out of steps, but never builds an invalid tree
            ast, _ = replay(r["actions"], default_grammar(), f.n_c, f.n_t, f.n_v)
            assert ast == r["ast"]
        else:
            assert "max_steps" in r["error"]


def test_no_grammar_mask_can_produce_invalid_actions(feats, tiny_tokenizer):
    m = _model(tiny_tokenizer, grammar_mask=False).eval()
    res = m.predict(collate(feats[:6]), max_steps=60)
    assert any(r["error"] is not None for r in res)  # untrained + unconstrained -> ill-formed trees detected
    m2 = _model(tiny_tokenizer, grammar_mask=True).eval()
    res2 = m2.predict(collate(feats[:6]), max_steps=60)
    assert all(r["error"] is None or "max_steps" in r["error"] for r in res2)


def test_beam_search_matches_or_beats_greedy_score(feats, tiny_tokenizer):
    m = _model(tiny_tokenizer)
    batch = collate(feats[:2])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):  # train until the two examples are memorised
        loss = m(batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.02:
            break
    m.eval()
    greedy = m.predict(batch, beam_size=1)
    beam = m.predict(batch, beam_size=3)
    for g, b, f in zip(greedy, beam, feats[:2]):
        assert b["ast"] is not None
        if g["ast"] is not None:
            assert b["score"] >= g["score"] - 1e-4
        gold, _ = replay([tuple(a) for a in f.rec["actions"]], default_grammar(), f.n_c, f.n_t, f.n_v)
        assert g["ast"] == gold  # the overfit model reproduces the training trees
