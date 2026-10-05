"""Shared fixtures. Tests run offline: the tokenizer is built from a local vocab and
BERT models use small random configurations."""

from __future__ import annotations

import string
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ratsql.data.synthetic import generate_synthetic_dataset  # noqa: E402
from ratsql.schema.schema import Schema, load_schemas  # noqa: E402


TOY_SCHEMA = {
    "db_id": "toy",
    "table_names": ["employee", "department"],
    "table_names_original": ["Employee", "Department"],
    "column_names": [
        [-1, "*"],
        [0, "employee id"],
        [0, "name"],
        [0, "city"],
        [0, "age"],
        [0, "department id"],
        [1, "department id"],
        [1, "department name"],
        [1, "budget"],
    ],
    "column_names_original": [
        [-1, "*"],
        [0, "Emp_ID"],
        [0, "Name"],
        [0, "City"],
        [0, "Age"],
        [0, "Dept_ID"],
        [1, "Dept_ID"],
        [1, "Dept_Name"],
        [1, "Budget"],
    ],
    "column_types": [
        "text",
        "number",
        "text",
        "text",
        "number",
        "number",
        "number",
        "text",
        "number",
    ],
    "primary_keys": [1, 6],
    "foreign_keys": [[5, 6]],
}


@pytest.fixture(scope="session")
def toy_schema() -> Schema:
    return Schema.from_spider(TOY_SCHEMA)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthetic")
    generate_synthetic_dataset(d)
    return d


@pytest.fixture(scope="session")
def synthetic_schemas(synthetic_dir) -> dict[str, Schema]:
    return load_schemas(synthetic_dir / "tables.json")


@pytest.fixture(scope="session")
def processed_synthetic(synthetic_dir, tmp_path_factory) -> Path:
    from ratsql.data.preprocess import run_preprocessing
    from ratsql.data.spider import synthetic_spec

    out = tmp_path_factory.mktemp("processed")
    cfg = {
        "linking": {"max_ngram": 5},
        "values": {"max_ngram": 4},
        "validation": {"fraction": 0.0},
    }
    run_preprocessing(
        synthetic_spec(synthetic_dir),
        cfg,
        out,
        out / "value_index",
    )
    return out


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """A WordPiece tokenizer whose vocab covers every word via single-character pieces."""
    from transformers import BertTokenizerFast

    words = """
    how many what which who list show find the of all and are is in with for each
    number count average name city age employee department budget salary id project
    title student course major text star * work hyderabad
    """.split()

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    chars = list(string.ascii_lowercase + string.digits + string.punctuation)
    vocab += chars + [
        "##" + c
        for c in string.ascii_lowercase + string.digits
    ]
    vocab += [w for w in dict.fromkeys(words) if w not in vocab]

    d = tmp_path_factory.mktemp("tok")
    (d / "vocab.txt").write_text(
        "\n".join(vocab),
        encoding="utf-8",
    )

    return BertTokenizerFast(
        vocab=str(d / "vocab.txt"),
        do_lower_case=True,
    )


TINY_BERT = {
    "vocab_size": 256,
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 512,
}


def tiny_model_cfg(
    encoder: str = "bert",
    use_relations: bool = True,
    grammar_mask: bool = True,
    vocab_size: int = 256,
) -> dict:
    enc = {
        "type": encoder,
        "pretrained_model": "none",
        "max_length": 512,
        "pooling": "mean",
        "column_type_prefix": True,
    }

    if encoder == "bert":
        enc["bert_config"] = dict(
            TINY_BERT,
            vocab_size=vocab_size,
        )
    else:
        enc.update({
            "emb_dim": 32,
            "hidden": 32,
        })

    return {
        "encoder": enc,
        "rat": {
            "d_model": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "use_relations": use_relations,
        },
        "decoder": {
            "hidden": 48,
            "action_dim": 16,
            "type_dim": 8,
            "att_heads": 4,
            "dropout": 0.0,
            "pointer": "memory_aligned",
            "pointer_dim": 32,
            "grammar_mask": grammar_mask,
        },
    }