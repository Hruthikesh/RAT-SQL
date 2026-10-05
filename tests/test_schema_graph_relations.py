"""Schema graph, relation vocabulary and relation matrix."""

import numpy as np

from ratsql.schema.graph import SchemaGraph
from ratsql.schema.linking import SchemaLinker, SchemaLinks
from ratsql.schema.relations import RelationMatrixBuilder, RelationVocabulary


def test_schema_graph_edges(toy_schema):
    g = SchemaGraph.from_schema(toy_schema)
    e = g.edges
    assert e[(("column", 2), ("column", 3))] == "cc_same_table"
    assert e[(("column", 5), ("column", 6))] == "cc_fk_forward"  # Employee.Dept_ID -> Department.Dept_ID
    assert e[(("column", 6), ("column", 5))] == "cc_fk_backward"
    assert e[(("column", 1), ("table", 0))] == "ct_primary_key"
    assert e[(("column", 2), ("table", 0))] == "ct_belongs_to"
    assert e[(("table", 0), ("column", 2))] == "tc_belongs_to"
    assert e[(("column", 5), ("table", 1))] == "ct_foreign_key"
    assert e[(("column", 0), ("table", 1))] == "ct_any_table"
    assert e[(("table", 0), ("table", 1))] == "tt_fk_forward"
    assert e[(("table", 1), ("table", 0))] == "tt_fk_backward"
    no_fk = SchemaGraph.from_schema(toy_schema, use_foreign_keys=False)
    assert not any("fk" in lab or "foreign" in lab for lab in no_fk.edges.values())


def test_relation_vocabulary_sizes():
    full = RelationVocabulary("full")
    assert full.names[0] == "<pad>" and len(set(full.names)) == len(full.names)
    assert "qc_em" in full.index and "cq_vem" in full.index and "tt_fk_both" in full.index
    assert len(RelationVocabulary("coarse")) == 7
    assert full.coarse_of("qc_pm") == "qs_linked" and full.coarse_of("cc_default") == "ss_none"


def test_relation_matrix_blocks_and_links(toy_schema):
    q = "Which employees work in Hyderabad ?".split()
    links = SchemaLinks(q_col={(1, 2): "PM", (4, 3): "VEM"}, q_tab={(1, 0): "EM"})
    b = RelationMatrixBuilder({"vocab": "full"})
    m = b.build(toy_schema, len(q), links)
    v = b.vocab
    n_q, n_c = len(q), toy_schema.num_columns
    assert m.ids.shape == (n_q + 9 + 2, n_q + 9 + 2)
    # question-question relative distance, clipped at +-2
    assert m.ids[0, 1] == v["qq_dist_1"] and m.ids[3, 0] == v["qq_dist_-2"] and m.ids[2, 2] == v["qq_dist_0"]
    # linking relations in both directions
    assert m.ids[1, n_q + 2] == v["qc_pm"] and m.ids[n_q + 2, 1] == v["cq_pm"]
    assert m.ids[4, n_q + 3] == v["qc_vem"] and m.ids[n_q + 3, 4] == v["cq_vem"]
    assert m.ids[1, n_q + n_c + 0] == v["qt_em"] and m.ids[n_q + n_c + 0, 1] == v["tq_em"]
    assert m.ids[0, n_q + 2] == v["qc_default"]
    # schema block
    assert m.ids[n_q + 5, n_q + 6] == v["cc_fk_forward"]
    assert m.ids[n_q + 2, n_q + 2] == v["cc_identity"]
    assert m.ids[n_q + n_c, n_q + n_c + 1] == v["tt_fk_forward"]
    assert (m.ids > 0).all()  # no padding id inside a real matrix


def test_relation_ablations(toy_schema):
    q = "employees in Hyderabad".split()
    links = SchemaLinks(q_col={(0, 2): "PM", (2, 3): "VEM"}, q_tab={(0, 0): "EM"})
    base = RelationMatrixBuilder({}).build(toy_schema, 3, links).ids
    no_val = RelationMatrixBuilder({"link_value": False}).build(toy_schema, 3, links)
    no_ng = RelationMatrixBuilder({"link_ngram": False}).build(toy_schema, 3, links)
    no_fk = RelationMatrixBuilder({"use_foreign_keys": False}).build(toy_schema, 3, links)
    v = no_val.vocab
    assert no_val.ids[2, 3 + 3] == v["qc_default"] and no_val.ids[0, 3 + 2] == v["qc_pm"]
    assert no_ng.ids[0, 3 + 2] == v["qc_default"] and no_ng.ids[2, 3 + 3] == v["qc_vem"]
    assert no_ng.ids[0, 3 + 9] == v["qt_default"]
    assert no_fk.ids[3 + 5, 3 + 6] == v["cc_same_table"] or no_fk.ids[3 + 5, 3 + 6] == v["cc_default"]
    assert (base != no_fk.ids).any()
    coarse = RelationMatrixBuilder({"vocab": "coarse"}).build(toy_schema, 3, links)
    assert coarse.ids.max() < 7 and set(np.unique(coarse.ids)) <= set(range(1, 7))


def test_linker_output_feeds_relations(toy_schema):
    q = "What is the name of each department ?".split()
    links = SchemaLinker().link(q, toy_schema)
    m = RelationMatrixBuilder({}).build(toy_schema, len(q), links)
    counts = m.counts()
    assert counts.get("qt_em", 0) >= 1  # "department" -> Department table
    assert counts.get("qc_em", 0) + counts.get("qc_pm", 0) >= 1
