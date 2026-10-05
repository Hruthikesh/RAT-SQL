"""Error taxonomy, complexity features, linking analysis and the synthetic relation task."""

from ratsql.analysis.errors import classify_error
from ratsql.analysis.linking_analysis import analyse_linking
from ratsql.analysis.synthetic_relations import REL_LINK, sample_batch
from ratsql.evaluation.complexity import complexity_features
from ratsql.evaluation.exact_match import ExactMatchEvaluator
from ratsql.sql.parser import parse_sql
from ratsql.utils.io import read_jsonl


def _row(schema, pred_sql, gold_sql, exec_match=False):
    em = ExactMatchEvaluator(schema).evaluate_strings(pred_sql, gold_sql)
    return {"grammar_valid": True, "exact_match": em["exact"], "exec_match": exec_match, "partial": em["partial"]}


def test_error_categories(toy_schema):
    gold = "SELECT Name FROM Employee WHERE City = 'Pune'"
    g = parse_sql(gold, toy_schema)
    links = {"q_col": [[3, 3, "VEM"], [0, 2, "EM"]], "q_tab": [[1, 0, "EM"]]}

    def cls(pred, **kw):
        return classify_error(_row(toy_schema, pred, gold, **kw), g, parse_sql(pred, toy_schema), links, toy_schema)

    assert cls("SELECT Name FROM Employee WHERE Age = 'Pune'")["primary"] == "schema_linking_error"  # missed linked City
    assert cls("SELECT Name FROM Employee WHERE City != 'Pune'")["primary"] == "where_error"
    assert cls("SELECT count(Name) FROM Employee WHERE City = 'Pune'")["primary"] == "aggregation_error"
    r = cls("SELECT Name FROM Employee WHERE City = 'Delhi'")
    assert r["primary"] == "wrong_value"  # exact match (values ignored) but execution differs
    assert cls("SELECT Name FROM Employee WHERE City IN (SELECT City FROM Employee)")["primary"] == "nested_query_error"
    t = classify_error({"grammar_valid": False, "decode_error": "step 3: rule invalid"}, g, None, links, toy_schema)
    assert t["primary"] == "grammar_error"
    j = classify_error(
        _row(toy_schema, "SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2", gold),
        g,
        parse_sql("SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2", toy_schema),
        {"q_col": [], "q_tab": []},
        toy_schema,
    )
    assert "wrong_table" in j["labels"]


def test_complexity_features(toy_schema):
    f = complexity_features(parse_sql("SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID WHERE T2.Budget > (SELECT avg(Budget) FROM Department) GROUP BY T1.Name ORDER BY count(*) DESC LIMIT 1", toy_schema))
    assert f["num_tables"] == 2 and f["num_joins"] == 1 and f["num_conditions"] == 1
    assert f["has_nested"] and f["has_group_by"] and f["has_order_by"] and f["has_aggregation"] and not f["has_set_op"]
    assert f["depth"] == 2


def test_linking_analysis(processed_synthetic):
    res = analyse_linking(read_jsonl(processed_synthetic / "dev.jsonl"))
    assert 0 < res["column_recall"] <= 1 and 0 < res["column_precision"] <= 1
    assert res["gold_literal_coverage"] == 1.0
    assert res["per_link_type"]["EM"]["precision"] > res["per_link_type"]["PM"]["precision"]


def test_synthetic_relation_task_structure():
    import torch

    ids, labels, rel, target = sample_batch(4, 6, 3, 10, "one_hop", torch.Generator().manual_seed(0))
    assert rel.shape == (4, 6, 6)
    for b in range(4):
        for i in range(6):
            partners = (rel[b, i] == REL_LINK).nonzero().flatten().tolist()
            assert len(partners) >= 1
            assert target[b, i] in labels[b, partners]


def test_clause_correctness(toy_schema):
    from ratsql.analysis.clauses import clause_correctness

    gold = parse_sql("SELECT T1.Name, count(*) FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID WHERE T2.Budget > 5 GROUP BY T1.Name HAVING count(*) > 1 ORDER BY T1.Age DESC", toy_schema)
    same = clause_correctness(gold, gold, toy_schema)
    assert all(v is True or v is None for v in same.values())
    assert same["NESTED"] is None and same["JOIN"] is True and same["HAVING"] is True
    pred = parse_sql("SELECT T1.Name, count(*) FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID WHERE T2.Budget < 5 GROUP BY T1.Name ORDER BY T1.Age ASC", toy_schema)
    r = clause_correctness(gold, pred, toy_schema)
    assert r["SELECT"] and r["JOIN"] and r["GROUP BY"]
    assert r["WHERE"] is False and r["HAVING"] is False and r["ORDER BY"] is False
    assert clause_correctness(gold, None, toy_schema)["SELECT"] is False
