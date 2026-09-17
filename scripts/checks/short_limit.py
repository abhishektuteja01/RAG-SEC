"""Fails when a retriever hands back fewer rows than its `LIMIT` asked for.

The instance: `chunks_embedding_hnsw` indexes `embedding` only, so `WHERE variant = 'A'`
POST-filters whatever the index already returned. At pgvector's default `hnsw.ef_search`
of 40 the unfiltered `dense()` path returned a median of 35 rows for `LIMIT 50`, silently
shrinking the candidate pool on the questions where company resolution abstains (RETR-50).
`store.HNSW_EF_SEARCH` fixes it; this locks the general property instead of that one number,
because the same bug returns whenever an index's read depth drifts below a caller's `LIMIT`.

Same class as DECISIONS.md RETR-24 / AGENT-16: an assumption about an on-disk artifact --
here the index -- that its producer never promised.

Checked against READ_DEPTH_MAX, the deepest read any caller may ask for, not the pool size:
the two were one constant until P10 split them, and a pool-derived ef_search equals the read
depth at depth 200, which is the same bug wearing the fix's clothes.

Two legs. The static one needs no database, so a `HNSW_EF_SEARCH` edited below READ_DEPTH_MAX
fails in CI where Postgres is absent. The live one needs one, and is skipped (not failed)
when the connection is refused. Only `dense()` is probed: BM25 legitimately returns fewer
than `k` when fewer than `k` documents match, so a short list there is not evidence.

Usage:
    python scripts/checks/short_limit.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.candidates import LIVE_VARIANT, READ_DEPTH_MAX, dense  # noqa: E402
from rag_sec.store import EMBEDDING_DIM, HNSW_EF_SEARCH, get_conn  # noqa: E402

PROBES = 8  # random unit vectors: no embedding model, so this never loads one


def _probe_vectors() -> list:
    import numpy as np

    rng = np.random.default_rng(0)
    v = rng.standard_normal((PROBES, EMBEDDING_DIM))
    return list(v / np.linalg.norm(v, axis=1, keepdims=True))


def main() -> int:
    failures = []
    if HNSW_EF_SEARCH < READ_DEPTH_MAX:
        failures.append(
            f"store.HNSW_EF_SEARCH={HNSW_EF_SEARCH} is below READ_DEPTH_MAX={READ_DEPTH_MAX}: "
            "the index cannot return the depth the retriever asks for"
        )

    try:
        conn = get_conn()
    except Exception as exc:  # noqa: BLE001 -- any connection failure, not one driver's
        print(f"note: no database ({type(exc).__name__}), live leg skipped")
        conn = None

    if conn is not None:
        with conn:
            short = [n for n in (len(dense(conn, v, READ_DEPTH_MAX, LIVE_VARIANT))
                                 for v in _probe_vectors()) if n < READ_DEPTH_MAX]
            if short:
                failures.append(
                    f"dense() unfiltered returned {sorted(short)} rows for LIMIT "
                    f"{READ_DEPTH_MAX} on {len(short)}/{PROBES} probes (hnsw.ef_search="
                    f"{conn.execute('SHOW hnsw.ef_search').fetchone()[0]})"
                )
            # Negative control: the check has to be able to see the bug it was written for.
            conn.execute("SET hnsw.ef_search = 40")
            seen = sum(1 for v in _probe_vectors()
                       if len(dense(conn, v, READ_DEPTH_MAX, LIVE_VARIANT)) < READ_DEPTH_MAX)
            print(f"negative control at the pgvector default of 40: {seen}/{PROBES} probes short")

    if failures:
        print("a retriever returned fewer rows than its LIMIT (DECISIONS.md RETR-50):\n",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}\n", file=sys.stderr)
        return 1
    print(f"ok: HNSW_EF_SEARCH={HNSW_EF_SEARCH} >= READ_DEPTH_MAX={READ_DEPTH_MAX}"
          + ("; dense() filled LIMIT on every probe" if conn is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
