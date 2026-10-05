"""N-gram (name-based) and value-based schema linking."""

import sqlite3

from ratsql.schema.db_values import DatabaseValueIndex
from ratsql.schema.linking import SchemaLinker, describe_links


def _toy_db(tmp_path, toy_schema):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE Employee (Emp_ID INTEGER, Name TEXT, City TEXT, Age INTEGER, Dept_ID INTEGER)")
    conn.execute("CREATE TABLE Department (Dept_ID INTEGER, Dept_Name TEXT, Budget INTEGER)")
    conn.executemany("INSERT INTO Employee VALUES (?,?,?,?,?)", [(1, "Asha Rao", "Hyderabad", 30, 1), (2, "Tom Lee", "New York City", 41, 2)])
    conn.executemany("INSERT INTO Department VALUES (?,?,?)", [(1, "Research", 100), (2, "Human Resources", 50)])
    conn.commit()
    conn.close()
    return DatabaseValueIndex.build(toy_schema, path)


def test_unigram_exact_and_partial(toy_schema):
    links = SchemaLinker().link("What is the name and city of each employee ?".split(), toy_schema)
    assert links.q_col[(3, 2)] == "EM"  # name -> Employee.Name
    assert links.q_col[(5, 3)] == "EM"  # city -> Employee.City
    assert links.q_tab[(8, 0)] == "EM"  # employee -> Employee
    assert links.q_col[(3, 7)] == "PM"  # name -> Department."department name" (partial)


def test_bigram_and_case_insensitive(toy_schema):
    links = SchemaLinker().link("List the DEPARTMENT NAME of all departments".split(), toy_schema)
    assert links.q_col[(2, 7)] == "EM" and links.q_col[(3, 7)] == "EM"  # bigram exact match wins over partial
    assert links.q_tab[(6, 1)] == "EM"  # plural normalised


def test_trigram_match(toy_schema):
    import copy

    from ratsql.schema.schema import Schema

    from conftest import TOY_SCHEMA

    entry = copy.deepcopy(TOY_SCHEMA)
    entry["column_names"][8] = [1, "annual budget amount"]
    s = Schema.from_spider(entry)
    links = SchemaLinker().link("show the annual budget amount".split(), s)
    assert all(links.q_col[(i, 8)] == "EM" for i in (2, 3, 4))
    links2 = SchemaLinker(max_n=2).link("show the annual budget amount".split(), s)
    assert links2.q_col[(2, 8)] == "PM"  # trigram disabled -> only partial matches


def test_stopwords_do_not_link(toy_schema):
    links = SchemaLinker().link("what is the of".split(), toy_schema)
    assert not links.q_col and not links.q_tab


def test_value_linking_hyderabad(tmp_path, toy_schema):
    vindex = _toy_db(tmp_path, toy_schema)
    q = "Which employees work in Hyderabad ?".split()
    links = SchemaLinker().link(q, toy_schema, vindex)
    assert links.q_col[(4, 3)] == "VEM"  # Hyderabad -> Employee.City (cell value)
    assert links.q_tab[(1, 0)] == "EM"  # employees -> Employee
    vm = [m for m in links.value_matches if m.column == 3]
    assert vm and vm[0].value == "Hyderabad" and (vm[0].start, vm[0].end) == (4, 5)
    lines = describe_links(q, toy_schema, links)
    assert any("VALUE 'Hyderabad'" in line and "Employee.City" in line for line in lines)


def test_value_linking_multiword_and_partial(tmp_path, toy_schema):
    vindex = _toy_db(tmp_path, toy_schema)
    links = SchemaLinker().link("employees from new york city".split(), toy_schema, vindex)
    assert links.q_col[(2, 3)] == "VEM" and links.q_col[(3, 3)] == "VEM"  # trigram cell value "New York City"
    assert links.q_col[(4, 3)] == "EM"  # "city" is also the column name: name match has precedence
    assert any((m.start, m.end, m.value) == (2, 5, "New York City") for m in links.value_matches)
    links = SchemaLinker().link("staff of the resources department".split(), toy_schema, vindex)
    assert links.q_col[(3, 7)] == "VPM"  # word inside "Human Resources"
    assert any(m.value == "Human Resources" and m.kind == "VPM" for m in links.value_matches)


def test_number_links(toy_schema):
    links = SchemaLinker().link("employees older than 30".split(), toy_schema)
    assert links.q_col[(3, 4)] == "NUM"  # numeric token <-> numeric column
    assert (3, 2) not in links.q_col


def test_link_filtering(toy_schema):
    links = SchemaLinker().link("name of employees older than 30".split(), toy_schema)
    only_values = links.filtered(ngram=False, value=True, numbers=False)
    assert not only_values.q_tab and all(t in ("VEM", "VPM") for t in only_values.q_col.values())
