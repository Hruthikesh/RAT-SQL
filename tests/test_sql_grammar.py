"""SQL parser, AST conversion, transition system (grammar constraints) and serializer."""

import pytest

from ratsql.sql.ast import Node, ast_to_sql, sql_to_ast, validate_ast
from ratsql.sql.grammar import default_grammar
from ratsql.sql.parser import SQLParseError, canonical, parse_sql, to_jsonable, to_tuples, tokenize_sql
from ratsql.sql.serializer import serialize_sql
from ratsql.sql.transition import COLUMN, RULE, TABLE, VALUE, DecodeState, InvalidActionError, ast_to_actions, replay
from ratsql.utils.io import read_json


class _Values:
    """Trivial value resolver: stores literals in a list."""

    def __init__(self):
        self.vals = ["<unk>"]

    def __call__(self, raw, ctx):
        if raw not in self.vals:
            self.vals.append(raw)
        return self.vals.index(raw)

    def lookup(self, ref, ctx):
        return self.vals[ref]


def test_tokenize_sql():
    toks = tokenize_sql("SELECT T1.Name FROM Employee AS T1 WHERE T1.age >= 30 AND name != 'O''Brien';")
    assert toks == ["select", "t1.name", "from", "employee", "as", "t1", "where", "t1.age", ">=", "30", "and", "name", "!=", '"O\'Brien"']
    assert tokenize_sql("x > -5")[-1] == "-5"


def test_parse_basic_structure(toy_schema):
    s = parse_sql("SELECT count(*), max(Age) FROM Employee WHERE City = 'Hyderabad' AND Age > 30", toy_schema)
    assert s["select"] == (False, [(3, (0, (0, 0, False), None)), (1, (0, (0, 4, False), None))])
    assert s["from"]["table_units"] == [("table_unit", 0)]
    assert s["where"][0] == (False, 2, (0, (0, 3, False), None), '"Hyderabad"', None)
    assert s["where"][1] == "and" and s["where"][2][3] == 30.0


def test_parse_join_nested_setop(toy_schema):
    q = (
        "SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID "
        "WHERE T2.Budget > (SELECT avg(Budget) FROM Department) GROUP BY T1.Name HAVING count(*) > 1 "
        "ORDER BY T1.Age DESC LIMIT 3 UNION SELECT Name FROM Employee"
    )
    s = parse_sql(q, toy_schema)
    assert s["from"]["table_units"] == [("table_unit", 0), ("table_unit", 1)]
    assert s["from"]["conds"][0][3] == (0, 6, False)  # join condition value is a column unit
    assert isinstance(s["where"][0][3], dict)
    assert s["groupBy"] == [(0, 2, False)] and s["having"][0][2][1][0] == 3
    assert s["orderBy"] == ("desc", [(0, (0, 4, False), None)]) and s["limit"] == 3
    assert s["union"] is not None and s["union"]["from"]["table_units"] == [("table_unit", 0)]


def test_parse_errors(toy_schema):
    with pytest.raises(SQLParseError):
        parse_sql("SELECT nonexistent FROM Employee", toy_schema)
    with pytest.raises(SQLParseError):
        parse_sql("SELECT Name FROM Nowhere", toy_schema)
    with pytest.raises(SQLParseError):
        parse_sql("SELECT Name FROM Employee ORDER BY Age > 3", toy_schema)  # strict: trailing tokens
    parse_sql("SELECT Name FROM Employee ORDER BY Age > 3", toy_schema, strict=False)  # official leniency


def _roundtrip(sql_dict, schema):
    vals = _Values()
    ast = sql_to_ast(sql_dict, vals)
    assert validate_ast(ast, num_columns=schema.num_columns, num_tables=schema.num_tables) == []
    actions = ast_to_actions(ast)
    rebuilt, infos = replay(actions, num_columns=schema.num_columns, num_tables=schema.num_tables, num_values=len(vals.vals))
    assert rebuilt == ast
    back = ast_to_sql(rebuilt, vals.lookup)
    return ast, actions, infos, back


def test_ast_roundtrip_all_synthetic(synthetic_dir, synthetic_schemas):
    """dict -> AST -> actions -> AST -> dict -> SQL -> dict is the identity on every synthetic query."""
    for split in ("train", "dev"):
        for ex in read_json(synthetic_dir / f"{split}.json"):
            schema = synthetic_schemas[ex["db_id"]]
            gold = canonical(to_tuples(ex["sql"]))
            _, actions, infos, back = _roundtrip(gold, schema)
            assert canonical(back) == gold
            sql = serialize_sql(back, schema)
            assert canonical(parse_sql(sql, schema)) == gold, (ex["query"], sql)
            assert len(infos) == len(actions)


def test_transition_masks_and_parents(toy_schema):
    g = default_grammar()
    s = parse_sql("SELECT Name FROM Employee WHERE Age > 30", toy_schema)
    _, actions, infos, _ = _roundtrip(s, toy_schema)
    # first action chooses the select-level distinct flag (Select/Agg/Query are auto-expanded)
    assert infos[0].kind == RULE and g.rule_names[actions[0][1]] == "NotDistinct"
    for a, info in zip(actions, infos):
        if info.kind == RULE:
            assert g.mask_table[info.mask_id][a[1]], "gold action must be allowed by its mask"
        assert info.parent_step < len(infos)
    kinds = [a[0] for a in actions]
    assert COLUMN in kinds and TABLE in kinds and VALUE in kinds


def test_grammar_rejects_invalid_actions(toy_schema):
    g = default_grammar()
    st = DecodeState(g, toy_schema.num_columns, toy_schema.num_tables, 3)
    fr = st.frontier()
    assert fr.kind == RULE and fr.type_name == "distinct"
    with pytest.raises(InvalidActionError):
        st.apply((COLUMN, 1))  # wrong action kind
    with pytest.raises(InvalidActionError):
        st.apply((RULE, g.rule_id["Count"]))  # rule of the wrong type
    with pytest.raises(InvalidActionError):
        st.apply((RULE, g.reduce_id))  # Reduce not allowed for a single field
    st.apply((RULE, g.rule_id["NotDistinct"]))
    assert st.frontier().type_name == "agg_op"
    st.apply((RULE, g.rule_id["NoAgg"]))
    assert st.frontier().type_name == "val_unit"
    st.apply((RULE, g.rule_id["Column"]))
    st.apply((RULE, g.rule_id["NoAgg"]))
    with pytest.raises(InvalidActionError):
        st.apply((COLUMN, 99))  # pointer out of range
    st.apply((COLUMN, 2))
    assert set(st.valid_rules()) == {g.rule_id["NotDistinct"], g.rule_id["Distinct"]}


def test_valid_mask_only_contains_legal_rules():
    g = default_grammar()
    for (tname, allow_reduce), mid in g.mask_id.items():
        allowed = {g.rule_names[i] for i, ok in enumerate(g.mask_table[mid]) if ok}
        expected = {c.name for c in g.types[tname]} | ({"Reduce"} if allow_reduce else set())
        assert allowed == expected


def test_unconstrained_sequence_detected():
    g = default_grammar()
    with pytest.raises(InvalidActionError):
        replay([(RULE, g.rule_id["Count"])], g, 5, 2, 2)
    with pytest.raises(InvalidActionError):
        replay([(RULE, g.rule_id["NotDistinct"])], g, 5, 2, 2)  # incomplete tree


def test_serializer_aliases_unique_and_executable(toy_schema):
    s = parse_sql(
        "SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID WHERE T1.Age > (SELECT avg(T3.Age) FROM Employee AS T3 JOIN Department AS T4 ON T3.Dept_ID = T4.Dept_ID)",
        toy_schema,
    )
    sql = serialize_sql(s, toy_schema)
    assert "T1" in sql and "T3" in sql and sql.count("AS T1") == 1
    assert canonical(parse_sql(sql, toy_schema)) == canonical(s)
    # a column whose table is not in FROM is qualified with the real table name (and would fail at execution)
    bad = parse_sql("SELECT Name FROM Employee", toy_schema)
    bad["select"] = (False, [(0, (0, (0, 7, False), None))])
    assert serialize_sql(bad, toy_schema) == "SELECT Department.Dept_Name FROM Employee"


def test_json_forms_roundtrip(toy_schema):
    s = parse_sql("SELECT Name FROM Employee WHERE City LIKE '%bad%'", toy_schema)
    assert canonical(to_tuples(to_jsonable(s))) == canonical(s)
    ast = sql_to_ast(s, _Values())
    assert Node.from_json(ast.to_json()) == ast


def test_schema_graph_join_inference(toy_schema):
    from ratsql.sql.join_inference import infer_joins

    # column of a table missing from FROM -> table added, ON clause from the foreign key
    pred = parse_sql("SELECT Name FROM Employee WHERE Department.Budget > 5", toy_schema)
    fixed = infer_joins(pred, toy_schema)
    assert [u[1] for u in fixed["from"]["table_units"]] == [0, 1]
    assert fixed["from"]["conds"] == [(False, 2, (0, (0, 5, False), None), (0, 6, False), None)]
    sql = serialize_sql(fixed, toy_schema)
    assert sql == "SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2 ON T1.Dept_ID = T2.Dept_ID WHERE T2.Budget > 5"
    # a wrong predicted ON condition is replaced by the FK condition
    bad = parse_sql("SELECT T1.Name FROM Employee AS T1 JOIN Department AS T2 ON T1.Age = T2.Budget", toy_schema)
    assert infer_joins(bad, toy_schema)["from"]["conds"][0][2][1][1] == 5
    # single-table queries are unchanged
    one = parse_sql("SELECT Name FROM Employee WHERE Age > 3", toy_schema)
    assert canonical(infer_joins(one, toy_schema)) == canonical(one)
