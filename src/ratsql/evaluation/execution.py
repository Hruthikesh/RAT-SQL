"""Execution accuracy (EX) on SQLite databases.

A prediction is execution-correct when its result on the database equals the
gold query's result.  Result comparison follows the semantics of the
test-suite evaluation of Zhong et al. (2020), re-implemented here:

* rows are compared as a *bag* (multiset) unless the gold query contains
  ORDER BY, in which case row order matters;
* columns may be permuted (the SELECT order of two equivalent queries can
  differ); candidate permutations are pruned using per-column value sets.

We evaluate on the single database shipped with Spider (not the distilled
test-suite databases), so our EX is the classic "execution accuracy", which is
known to have some false positives (e.g. different queries that happen to
return the same rows).  Queries are executed read-only with a wall-clock
timeout enforced by an SQLite progress handler.
"""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from itertools import permutations, product
from pathlib import Path
from typing import Any

MAX_ROWS = 100_000


class ExecutionResult:
    __slots__ = ("ok", "rows", "error", "seconds")

    def __init__(self, ok: bool, rows: list[tuple] | None = None, error: str | None = None, seconds: float = 0.0):
        self.ok = ok
        self.rows = rows
        self.error = error
        self.seconds = seconds

    def __repr__(self) -> str:
        return f"ExecutionResult(ok={self.ok}, rows={None if self.rows is None else len(self.rows)}, error={self.error!r})"


_CONN_CACHE: dict[str, sqlite3.Connection] = {}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    key = str(Path(db_path).resolve())
    conn = _CONN_CACHE.get(key)
    if conn is None:
        uri = Path(key).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        _CONN_CACHE[key] = conn
    return conn


def close_connections() -> None:
    for c in _CONN_CACHE.values():
        try:
            c.close()
        except sqlite3.Error:
            pass
    _CONN_CACHE.clear()


def execute_sql(db_path: str | Path, sql: str, timeout: float = 10.0, max_rows: int = MAX_ROWS) -> ExecutionResult:
    """Execute ``sql`` read-only; never raises (errors are returned)."""
    start = time.perf_counter()
    try:
        conn = _connect(db_path)
    except sqlite3.Error as e:
        return ExecutionResult(False, error=f"connect: {e}")
    deadline = start + timeout

    def _handler() -> int:
        return 1 if time.perf_counter() > deadline else 0

    conn.set_progress_handler(_handler, 10_000)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return ExecutionResult(False, error=f"too many rows (>{max_rows})", seconds=time.perf_counter() - start)
        return ExecutionResult(True, rows=[tuple(r) for r in rows], seconds=time.perf_counter() - start)
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg:
            msg = f"timeout after {timeout}s"
        return ExecutionResult(False, error=msg, seconds=time.perf_counter() - start)
    except (sqlite3.Error, ValueError, OverflowError, MemoryError) as e:
        return ExecutionResult(False, error=f"{type(e).__name__}: {e}", seconds=time.perf_counter() - start)
    finally:
        conn.set_progress_handler(None, 0)


# ---------------------------------------------------------------------------
def _hashable(v: Any) -> Any:
    if isinstance(v, float) and v != v:  # NaN
        return "__nan__"
    return v


def _candidate_permutations(gold_cols: list[set], pred_rows: list[tuple], num_cols: int):
    """Yield column permutations of the prediction consistent with gold column value sets."""
    if num_cols <= 1:
        yield tuple(range(num_cols))
        return
    # for each gold column i, which predicted columns j could correspond to it
    sample = pred_rows[: min(len(pred_rows), 20)]
    options = []
    for i in range(num_cols):
        opts = [j for j in range(num_cols) if all(_hashable(r[j]) in gold_cols[i] for r in sample)]
        if not opts:
            return
        options.append(opts)
    if num_cols <= 6:
        for perm in product(*options):
            if len(set(perm)) == num_cols:
                yield perm
    else:  # very wide results: try identity and a bounded number of permutations
        yield tuple(range(num_cols))
        for k, perm in enumerate(permutations(range(num_cols))):
            if k > 5000:
                break
            yield perm


def results_equal(gold: list[tuple], pred: list[tuple], order_matters: bool) -> bool:
    if not gold and not pred:
        return True
    if len(gold) != len(pred):
        return False
    num_cols = len(gold[0])
    if len(pred[0]) != num_cols:
        return False
    gold_h = [tuple(_hashable(v) for v in r) for r in gold]
    pred_h = [tuple(_hashable(v) for v in r) for r in pred]
    # quick rejection: multiset of all cell values must be equal
    if Counter(v for r in gold_h for v in r) != Counter(v for r in pred_h for v in r):
        return False
    gold_cols = [{r[i] for r in gold_h} for i in range(num_cols)]
    gold_counter = None if order_matters else Counter(gold_h)
    for perm in _candidate_permutations(gold_cols, pred_h, num_cols):
        permuted = [tuple(r[j] for j in perm) for r in pred_h]
        if order_matters:
            if permuted == gold_h:
                return True
        elif Counter(permuted) == gold_counter:
            return True
    return False


def order_matters(gold_sql: str) -> bool:
    return "order by" in " ".join(gold_sql.lower().split())


def execution_match(db_path: str | Path, pred_sql: str, gold_sql: str, timeout: float = 10.0, gold_result: ExecutionResult | None = None) -> dict:
    """Execute gold and prediction and compare results."""
    g = gold_result or execute_sql(db_path, gold_sql, timeout)
    p = execute_sql(db_path, pred_sql, timeout)
    match = bool(g.ok and p.ok and results_equal(g.rows or [], p.rows or [], order_matters(gold_sql)))
    return {
        "exec_match": match,
        "pred_exec_ok": p.ok,
        "pred_error": p.error,
        "gold_exec_ok": g.ok,
        "gold_error": g.error,
        "pred_seconds": p.seconds,
    }
