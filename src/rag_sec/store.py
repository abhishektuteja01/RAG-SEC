"""Postgres connection and chunk schema: pgvector for dense search, pg_search (ParadeDB)
for real BM25 in the same table -- not tsvector/ts_rank, which has neither length
normalization nor term saturation.
"""

import os

import psycopg
from pgvector.psycopg import register_vector

from rag_sec.candidates import READ_DEPTH_MAX
from rag_sec.preflight import assert_variant_predicates

EMBEDDING_DIM = 1024  # BGE-M3's dense output size, not an independent choice

# The HNSW index carries no `variant` column, so `WHERE variant = 'A'` POST-filters the rows
# the index already returned: a `LIMIT k` scan that visits ef_search candidates can hand back
# fewer than k. pgvector's default 40 returned a median of 35 rows for a LIMIT 50.
# Derived from READ_DEPTH_MAX -- the DEEPEST read any caller may ask for -- so it cannot
# drift below it; 4x is where dev fused-pool recall stops moving.
HNSW_EF_SEARCH = 4 * READ_DEPTH_MAX

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    filing_stem TEXT NOT NULL,
    chunk_index INT NOT NULL,
    heading TEXT,
    n_tokens INT,
    text TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIM}),
    -- Table-layout variant. 'A' (whole tables) is the only live one; the column
    -- stays so every read can constrain it (see preflight.py).
    variant TEXT NOT NULL DEFAULT 'A',
    excluded_by_variant TEXT[] NOT NULL DEFAULT '{{}}',
    UNIQUE (filing_stem, chunk_index, variant)
);
"""

HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);
"""

BM25_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_bm25_idx
ON chunks USING bm25 (id, text)
WITH (key_field='id');
"""


# Pinned per-variant, not as a total: rows of another variant entering the table would move
# only the total. Any row outside variant 'A' fails the check below.
EXPECTED_CHUNK_COUNTS = {"A": 99654}

_preflight_done = False


def preflight(conn) -> None:
    """Fail if the corpus isn't the one every published number was measured on.

    Once per process, not per connection -- retrieve.py opens one per search. ~10ms.
    The other half (a read missing its `variant` predicate) is checked in get_conn.
    """
    global _preflight_done
    if _preflight_done:
        return
    rows = conn.execute(
        "SELECT variant, count(*), count(embedding) FROM chunks GROUP BY variant"
    ).fetchall()
    actual = {v: n for v, n, _ in rows}
    unembedded = {v: n - e for v, n, e in rows if n != e}
    problems = []
    if actual != EXPECTED_CHUNK_COUNTS:
        for variant in sorted(set(actual) | set(EXPECTED_CHUNK_COUNTS)):
            want = EXPECTED_CHUNK_COUNTS.get(variant, 0)
            got = actual.get(variant, 0)
            if want != got:
                problems.append(f"  variant {variant!r}: expected {want}, found {got} ({got - want:+d})")
    if unembedded:
        problems.append(f"  rows with NULL embedding: {unembedded}")
    if problems:
        raise RuntimeError(
            "chunks table does not match EXPECTED_CHUNK_COUNTS:\n"
            + "\n".join(problems)
            + "\n\nIf this change was intentional, update EXPECTED_CHUNK_COUNTS in store.py."
            "\nNew non-A rows are only safe if every read constrains `variant`. Writers that"
            "\nlegitimately move these counts should call get_conn(check=False)."
        )
    _preflight_done = True


def get_conn(check: bool = True) -> psycopg.Connection:
    if check:
        # No DB needed, so it runs before connecting -- a forgotten `variant` predicate
        # is caught even when Postgres is unreachable.
        assert_variant_predicates()
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {int(HNSW_EF_SEARCH)}")  # not parameterizable
    if check:
        preflight(conn)
    return conn


def init_schema() -> None:
    # Table may not exist or be empty yet; nothing to assert about the corpus.
    with get_conn(check=False) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
