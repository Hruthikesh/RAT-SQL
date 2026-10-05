"""Exact Match, hardness, Execution Accuracy and SQL validity."""

import sqlite3

from ratsql.evaluation.exact_match import ExactMatchEvaluator, build_foreign_key_map, hardness
from ratsql.evaluation.execution import execute_sql, execution_match, results_equal
from ratsql.evaluation.evaluator import Evaluator
from ratsql.sql.parser import parse_sql


def em(schema, pred, gold):
    return ExactMatchEvaluator(schema).evaluate_strings(pred, gold)["exact"]


def test_exact_match_set_semantics_and_values(toy_schema):
    g = "SELECT Name, Age FROM Employee WHERE City = 'Pune' AND Age > 30"
    assert em(toy_schema, g, g)
    assert em(toy_schema, "SELECT Age, Name FROM Employee WHERE Age > 99 AND City = 'Delhi'", g)  # order + values ignored
    assert not em(toy_schema, "SELECT Name FROM Employee WHERE City = 'Pune' AND Age > 30", g)
    assert not em(toy_schema, "SELECT Name, Age FROM Employee WHERE City = 'Pune' OR Age > 30", g)  # and/or
    assert not em(toy_schema, "SELECT Name, Age FROM Employee WHERE City = 'Pune' AND Age < 30", g)  # operator
    assert not em(toy_schema, "SELECT Name, max(Age) FROM Employee WHERE City = 'Pune' AND Age > 30", g)  # aggregation


def test_exact_match_distinct_ignored_and_limit_presence(toy_schema):
    assert em(toy_schema, "SELECT DISTINCT City FROM Employee", "SELECT City FROM Employee")
    g = "SELECT Name FROM Employee ORDER BY Age DESC LIMIT 1"
    assert em(toy_schema, "SELECT Name FROM Employee ORDER BY Age DESC LIMIT 5", g)
    assert not em(toy_schema, "SELECT Name FROM Employee ORDER BY Age DESC", g)
    assert not em(toy_schema, "SELECT Name FROM Employee ORDER BY Age ASC LIMIT 1", g)


def test_exact_match_foreign_key_equivalence(toy_schema):
    assert build_foreign_key_map(toy_schema) == {5: 5, 6: 5}
    g = "SELECT count(*) FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID GROUP BY T1.Dept_ID"
    p = "SELECT count(*) FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID GROUP BY T2.Dept_ID"
    assert em(toy_schema, p, g)
    assert not em(toy_schema, "SELECT count(*) FROM Employee GROUP BY Dept_ID", g)  # FROM differs


def test_exact_match_nested_and_unparsable(toy_schema):
    g = "SELECT Name FROM Employee WHERE Age > (SELECT avg(Age) FROM Employee)"
    assert em(toy_schema, "SELECT Name FROM Employee WHERE Age > (SELECT avg(Age) FROM Employee)", g)
    assert not em(toy_schema, "SELECT Name FROM Employee WHERE Age > (SELECT max(Age) FROM Employee)", g)
    res = ExactMatchEvaluator(toy_schema).evaluate_strings("SELECT FROM WHERE", g)
    assert res["exact"] is False and res["pred_parse_ok"] is False


def test_hardness_levels(toy_schema):
    assert hardness(parse_sql("SELECT count(*) FROM Employee", toy_schema)) == "easy"
    assert hardness(parse_sql("SELECT Name, Age FROM Employee WHERE City = 'x'", toy_schema)) == "medium"
    assert hardness(parse_sql("SELECT Name FROM Employee WHERE Age > (SELECT avg(Age) FROM Employee)", toy_schema)) == "hard"
    q = "SELECT T1.Name, count(*) FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID WHERE T2.Budget > 5 GROUP BY T1.Name ORDER BY count(*) DESC LIMIT 1"
    assert hardness(parse_sql(q, toy_schema)) == "extra"


def test_results_equal_semantics():
    a = [(1, "x"), (2, "y")]
    assert results_equal(a, [(2, "y"), (1, "x")], order_matters=False)
    assert not results_equal(a, [(2, "y"), (1, "x")], order_matters=True)
    assert results_equal(a, [("x", 1), ("y", 2)], order_matters=True)  # column permutation
    assert not results_equal(a, [(1, "x"), (1, "x")], order_matters=False)  # bag semantics
    assert results_equal([], [], order_matters=False)
    assert not results_equal(a, [(1,), (2,)], order_matters=False)


def _db(tmp_path):
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE Employee (Emp_ID INTEGER, Name TEXT, City TEXT, Age INTEGER, Dept_ID INTEGER)")
    c.execute("CREATE TABLE Department (Dept_ID INTEGER, Dept_Name TEXT, Budget INTEGER)")
    c.executemany("INSERT INTO Employee VALUES (?,?,?,?,?)", [(1, "A", "Pune", 30, 1), (2, "B", "Delhi", 40, 1), (3, "C", "Pune", 50, 2)])
    c.executemany("INSERT INTO Department VALUES (?,?,?)", [(1, "R", 10), (2, "S", 20)])
    c.commit()
    c.close()
    return p


def test_execution_accuracy(tmp_path):
    db = _db(tmp_path)
    gold = "SELECT Name FROM Employee WHERE City = 'Pune'"
    assert execution_match(db, "SELECT Name FROM Employee WHERE City = 'Pune' ORDER BY Age DESC", gold)["exec_match"]
    assert not execution_match(db, "SELECT Name FROM Employee", gold)["exec_match"]
    r = execution_match(db, "SELECT Nope FROM Employee", gold)
    assert not r["pred_exec_ok"] and "no such column" in r["pred_error"]
    assert execute_sql(db, "SELECT count(*) FROM Employee").rows == [(3,)]
    # read-only connection
    assert not execute_sql(db, "DELETE FROM Employee").ok


def test_sql_validity_in_evaluator(tmp_path, toy_schema):
    db = _db(tmp_path)
    ev = Evaluator({"toy": toy_schema}, lambda _: db)
    gold_sql = parse_sql("SELECT Name FROM Employee", toy_schema)
    from ratsql.sql.parser import to_jsonable

    rec = {"id": "x", "db_id": "toy", "question": "q", "query": "SELECT Name FROM Employee", "sql": to_jsonable(gold_sql), "hardness": "easy"}
    ok = ev.evaluate_one(rec, "SELECT Name FROM Employee", True)
    assert ok["exact_match"] and ok["exec_match"] and ok["exec_valid"] and ok["parse_valid"]
    bad = ev.evaluate_one(rec, "SELECT Department.Dept_Name FROM Employee", True)  # column's table not in FROM
    assert bad["parse_valid"] and not bad["exec_valid"] and not bad["exec_match"]
    none = ev.evaluate_one(rec, "", False)
    assert not none["grammar_valid"] and not none["exec_valid"] and not none["exact_match"]
