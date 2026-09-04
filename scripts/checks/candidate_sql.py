"""Locks the SQL `rag_sec.candidates` emits to what the seven hand-copied versions emitted
before `RETR-36` consolidated them.

The expectations below are the fully-bound queries, transcribed from the pre-refactor call
sites in `retrieve.py`, `arm1_dense.py`, `arm2_hybrid.py`, `company_filter_ab.py` and
`rerank_prepare.py`. They are the point of the check: a shared helper is only a safe
refactor if it is behaviour-preserving, and "behaviour" here is the exact query text plus
the exact parameter order, since a reordered `%s` binds the wrong value silently.

Complements `variant_predicates.py`: that one proves a `variant` predicate is *present*,
this one proves the whole query is *unchanged*.

Usage:
    python scripts/checks/candidate_sql.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rag_sec import candidates as C  # noqa: E402

EMB = [0.1, 0.2]
TICKERS = ["AAPL", "MSFT"]
PAIRS = [("AAPL_2019_320193", 3), ("AAPL_2019_320193", 7)]

EXPECTED = {
    "dense, unfiltered": (
        lambda conn: C.dense(conn, EMB, 50, "A"),
        "SELECT filing_stem, chunk_index FROM chunks WHERE variant = 'A' "
        "ORDER BY embedding <=> [0.1, 0.2] LIMIT 50",
    ),
    "dense, company-filtered": (
        lambda conn: C.dense(conn, EMB, 50, "A", TICKERS),
        "SELECT filing_stem, chunk_index FROM chunks WHERE variant = 'A' "
        "AND split_part(filing_stem, '_', 1) = ANY(['AAPL', 'MSFT']) "
        "ORDER BY embedding <=> [0.1, 0.2] LIMIT 50",
    ),
    "bm25, unfiltered": (
        lambda conn: C.bm25(conn, "q", 50, "A"),
        "SELECT filing_stem, chunk_index, paradedb.score(id) AS s FROM chunks "
        "WHERE id @@@ paradedb.match('text', 'q') AND variant = 'A' ORDER BY s DESC LIMIT 50",
    ),
    "bm25, company-filtered": (
        lambda conn: C.bm25(conn, "q", 50, "A", TICKERS),
        "SELECT filing_stem, chunk_index, paradedb.score(id) AS s FROM chunks "
        "WHERE id @@@ paradedb.match('text', 'q') AND variant = 'A' "
        "AND split_part(filing_stem, '_', 1) = ANY(['AAPL', 'MSFT']) ORDER BY s DESC LIMIT 50",
    ),
    "chunk text fetch": (
        lambda conn: C.chunk_texts(conn, PAIRS, "A"),
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = 'A' "
        "AND filing_stem = ANY(['AAPL_2019_320193'])",
    ),
}


class _RecordingConn:
    """Captures (sql, args) instead of executing. No database is needed or wanted here --
    the claim under test is about the query text, not about what Postgres returns."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql, args=None):
        self.calls.append((sql, args))
        return self

    def fetchall(self):
        return []


def _bound(sql: str, args) -> str:
    """`%s` placeholders replaced in order, whitespace normalized. Substituting rather than
    comparing (sql, args) separately is what catches a reordered parameter list."""
    rest = list(args or ())
    out = re.sub(r"%s", lambda _m: repr(rest.pop(0)), sql)
    assert not rest, f"{len(rest)} unused parameter(s) for: {sql}"
    return " ".join(out.split())


def main() -> int:
    failures = []
    for name, (call, expected) in EXPECTED.items():
        conn = _RecordingConn()
        call(conn)
        if len(conn.calls) != 1:
            failures.append(f"{name}: expected 1 query, got {len(conn.calls)}")
            continue
        got = _bound(*conn.calls[0])
        if got != expected:
            failures.append(f"{name}:\n    expected: {expected}\n    got:      {got}")

    # RRF is pure, but its tie-break is `sorted`'s stability over dict insertion order, so
    # the fusion loop order is load-bearing and worth pinning too.
    a, b, c = ("X_2019_1", 0), ("Y_2019_2", 1), ("Z_2019_3", 2)
    # a and c score identically (1/61 + 1/63 each) and outscore b (2/62); the a-before-c
    # tie-break is insertion order, i.e. the order the lists were fused in.
    fused = C.rrf_fuse([[a, b, c], [c, b, a]])
    if fused != [a, c, b]:
        failures.append(f"rrf_fuse ordering changed: {fused}")

    if failures:
        print("candidate SQL changed (DECISIONS.md RETR-36):\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}\n", file=sys.stderr)
        return 1
    print(f"ok: {len(EXPECTED)} candidate queries unchanged, RRF ordering unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
