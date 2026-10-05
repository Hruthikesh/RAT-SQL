"""Schema-linking analysis against gold SQL.

For each example we compare the schema elements *used by the gold query*
(columns and tables at any nesting level, excluding ``*``) with the elements
the linker connected to the question.  This measures the linker itself,
independently of any model:

* recall     fraction of gold columns / tables with at least one link
* precision  fraction of linked columns / tables that the gold query uses
* per link type (EM, PM, VEM, VPM, NUM): how often a linked column is used
* value coverage: gold string literals that appear among database-value
  candidates (value linking) or anywhere in the candidate list
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ratsql.analysis.errors import _columns, _tables
from ratsql.sql.parser import canonical, to_tuples


def gold_elements(rec: dict) -> tuple[set[int], set[int]]:
    sql = canonical(to_tuples(rec["sql"]))
    return {c for c in _columns(sql) if c != 0}, set(_tables(sql))


def analyse_linking(records: list[dict]) -> dict:
    tot = Counter()
    by_type_used = Counter()
    by_type_all = Counter()
    for r in records:
        if not r.get("sql"):
            continue
        gcols, gtabs = gold_elements(r)
        col_types: dict[int, set[str]] = defaultdict(set)
        for _, c, t in r["links"]["q_col"]:
            col_types[c].add(t)
        tab_types: dict[int, set[str]] = defaultdict(set)
        for _, t_, t in r["links"]["q_tab"]:
            tab_types[t_].add(t)
        name_linked_cols = {c for c, ts in col_types.items() if ts & {"EM", "PM", "VEM", "VPM"}}
        tot["gold_cols"] += len(gcols)
        tot["gold_cols_linked"] += len(gcols & name_linked_cols)
        tot["gold_cols_exact"] += len({c for c in gcols if "EM" in col_types.get(c, set())})
        tot["gold_cols_value"] += len({c for c in gcols if col_types.get(c, set()) & {"VEM", "VPM"}})
        tot["linked_cols"] += len(name_linked_cols)
        tot["linked_cols_used"] += len(name_linked_cols & gcols)
        tot["gold_tabs"] += len(gtabs)
        tot["gold_tabs_linked"] += len(gtabs & set(tab_types))
        tot["linked_tabs"] += len(tab_types)
        tot["linked_tabs_used"] += len(set(tab_types) & gtabs)
        for c, ts in col_types.items():
            best = max(ts, key=lambda t: {"EM": 5, "VEM": 4, "PM": 3, "VPM": 2, "NUM": 1}[t])
            by_type_all[best] += 1
            by_type_used[best] += c in gcols
        # values
        cands = r["candidates"]
        vs = r.get("value_stats") or {}
        tot["gold_literals"] += vs.get("resolved", 0) + vs.get("unresolved", 0)
        tot["gold_literals_resolved"] += vs.get("resolved", 0)
        tot["db_candidates"] += sum(1 for c in cands if c[1] == "db")
        tot["examples"] += 1
    res = {
        "examples": tot["examples"],
        "column_recall": tot["gold_cols_linked"] / max(1, tot["gold_cols"]),
        "column_recall_exact": tot["gold_cols_exact"] / max(1, tot["gold_cols"]),
        "column_recall_value": tot["gold_cols_value"] / max(1, tot["gold_cols"]),
        "column_precision": tot["linked_cols_used"] / max(1, tot["linked_cols"]),
        "table_recall": tot["gold_tabs_linked"] / max(1, tot["gold_tabs"]),
        "table_precision": tot["linked_tabs_used"] / max(1, tot["linked_tabs"]),
        "gold_literal_coverage": tot["gold_literals_resolved"] / max(1, tot["gold_literals"]),
        "mean_db_value_candidates": tot["db_candidates"] / max(1, tot["examples"]),
        "per_link_type": {t: {"linked_pairs": by_type_all[t], "precision": by_type_used[t] / max(1, by_type_all[t])} for t in ("EM", "VEM", "PM", "VPM", "NUM")},
        "counts": dict(tot),
    }
    return res
