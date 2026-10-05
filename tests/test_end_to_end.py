"""End-to-end: question + schema -> graph -> linking -> BERT -> RAT -> LSTM decoder ->
grammar-constrained AST -> SQL -> execution, on a synthetic example."""

import torch

from conftest import tiny_model_cfg
from ratsql.data.dataset import FeatureBuilder, collate, load_records
from ratsql.data.values import ValueResolver, candidates_from_json
from ratsql.evaluation.execution import execution_match
from ratsql.models.model import Text2SQLModel, TokenizerInfo
from ratsql.schema.graph import SchemaGraph
from ratsql.schema.linking import SchemaLinks, describe_links
from ratsql.schema.relations import RelationMatrixBuilder
from ratsql.sql.ast import ast_to_sql, validate_ast
from ratsql.sql.serializer import serialize_sql


def test_end_to_end_hyderabad(processed_synthetic, synthetic_dir, synthetic_schemas, tiny_tokenizer):
    recs = load_records(processed_synthetic / "train.jsonl", require_actions=True)
    target = next(r for r in recs if r["question"] == "Which employees work in Hyderabad?")
    others = [r for r in recs if r is not target][:7]
    schema = synthetic_schemas[target["db_id"]]

    # 1. schema graph + 2. schema linking (computed during preprocessing)
    graph = SchemaGraph.from_schema(schema)
    links = SchemaLinks.from_json(target["links"])
    graph.add_question(target["question_tokens"], links)
    lines = describe_links(target["question_tokens"], schema, links)
    assert any("Hyderabad" in l and "employee.city" in l for l in lines), lines
    assert any(lab == "qc_vem" for lab in graph.edges.values())

    # 3. features: relation matrix + word pieces
    rb = RelationMatrixBuilder({})
    fb = FeatureBuilder(tiny_tokenizer, rb, synthetic_schemas)
    feats = [fb.build(r, i) for i, r in enumerate([target] + others)]

    # 4. BERT + RAT + LSTM decoder (tiny random model), overfit the 8 examples
    torch.manual_seed(0)
    info = TokenizerInfo(len(tiny_tokenizer), tiny_tokenizer.cls_token_id, tiny_tokenizer.sep_token_id, tiny_tokenizer.pad_token_id)
    model = Text2SQLModel(tiny_model_cfg(vocab_size=len(tiny_tokenizer)), info, rb.num_relations, pretrained=False)
    batch = collate(feats)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = model(batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    # 5. grammar-constrained decoding -> AST
    result = model.predict(collate(feats[:1]))[0]
    assert result["error"] is None
    ast = result["ast"]
    assert validate_ast(ast, num_columns=schema.num_columns, num_tables=schema.num_tables, num_values=feats[0].n_v) == []

    # 6. AST -> SQL -> execution
    resolver = ValueResolver(candidates_from_json(target["candidates"]))
    sql = serialize_sql(ast_to_sql(ast, resolver.lookup), schema)
    assert sql == "SELECT name FROM employee WHERE city = 'Hyderabad'", sql
    db = synthetic_dir / "database" / schema.db_id / f"{schema.db_id}.sqlite"
    res = execution_match(db, sql, target["query"])
    assert res["exec_match"] and res["pred_exec_ok"]
