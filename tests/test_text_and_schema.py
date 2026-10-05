"""Tokenizer, normalisation, schema parsing and dataset loading."""

from pathlib import Path

from ratsql.data.spider import load_split, synthetic_spec
from ratsql.data.validate import validate_dataset
from ratsql.utils.text import is_number_token, normalize_phrase, normalize_token, parse_number, tokenize, tokenize_words


def test_word_tokenizer_offsets():
    q = 'Which employees work in "New York" with salary > 3.5?'
    toks = tokenize(q)
    assert [t.text for t in toks] == ["Which", "employees", "work", "in", '"', "New", "York", '"', "with", "salary", ">", "3.5", "?"]
    for t in toks:
        assert q[t.start : t.end] == t.text


def test_normalisation_singularises_and_lowercases():
    assert normalize_token("Employees") == "employee"
    assert normalize_token("cities") == "city"
    assert normalize_token("addresses") == "address"
    assert normalize_token("singer's") == "singer"
    assert normalize_token("People") == "person"
    assert normalize_phrase("Dept_Name") == ["dept", "name"]
    assert tokenize_words("O'Brien's car") == ["O'Brien's", "car"]


def test_numbers():
    assert parse_number("three") == 3.0
    assert parse_number("1,000") == 1000.0
    assert parse_number("abc") is None
    assert is_number_token("3.5") and not is_number_token("x3")


def test_schema_from_spider(toy_schema):
    s = toy_schema
    assert s.num_columns == 9 and s.num_tables == 2
    assert s.columns[0].is_star and s.columns[0].table_id is None
    assert s.columns[1].is_primary and s.columns[6].is_primary
    assert s.columns[5].fk_target == 6
    assert s.qualified_name(7) == "Department.Dept_Name"
    assert s.column_id("employee", "city") == 3  # case-insensitive lookup
    assert s.table_id("DEPARTMENT") == 1
    assert s.fk_table_pairs() == {(0, 1)}
    assert s.tables[0].column_ids == [1, 2, 3, 4, 5]
    assert "FK->Department.Dept_ID" in s.summary()


def test_synthetic_dataset_loads_and_validates(synthetic_dir):
    spec = synthetic_spec(synthetic_dir)
    report = validate_dataset(spec)
    assert report["ok"], report
    train = load_split(spec, "train")
    dev = load_split(spec, "dev")
    assert len(train) > 100 and len(dev) > 50
    assert {e["db_id"] for e in train}.isdisjoint({e["db_id"] for e in dev})  # cross-database split
    for e in train[:5]:
        assert {"db_id", "question", "query", "sql"} <= set(e)


def test_real_spider_if_available():
    import pytest

    from ratsql.data.spider import find_spider_root, spider_spec

    root = find_spider_root(Path(__file__).resolve().parents[1] / "data" / "raw")
    if root is None:
        pytest.skip("Spider not downloaded")
    report = validate_dataset(spider_spec(root), check_databases=True)
    assert report["ok"], report["errors"]
    assert report["splits"]["train"]["examples"] == 8659
    assert report["splits"]["dev"]["examples"] == 1034
