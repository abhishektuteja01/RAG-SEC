"""Postgres connection and chunk schema: pgvector for dense search, and pg_search (ParadeDB)
for real BM25 in the same table (Postgres's built-in ts_rank is not BM25).
"""

import os

import psycopg
from pgvector.psycopg import register_vector

from rag_sec.candidates import READ_DEPTH, VARIANT

EMBEDDING_DIM = 1024  # BGE-M3's dense output size

# How many index entries an HNSW search visits. It must be well above READ_DEPTH: the
# `variant` condition is applied after the index scan, so too small a value returns fewer
# rows than asked for, with no error. 4x is where recall stops moving.
HNSW_EF_SEARCH = 4 * READ_DEPTH

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
    variant TEXT NOT NULL DEFAULT 'A',
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

EXPECTED_CHUNK_COUNTS = {VARIANT: 99654}

_preflight_done = False


def preflight(conn) -> None:
    """Fail unless the table holds exactly the corpus every number was measured on: 99,654
    variant-A chunks, each with an embedding, and nothing else. Once per process, ~10 ms."""
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
            "\nWriters that legitimately move these counts should call get_conn(check=False)."
        )
    _preflight_done = True


def get_conn(check: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {int(HNSW_EF_SEARCH)}")  # SET takes no parameters
    if check:
        preflight(conn)
    return conn


def init_schema() -> None:
    # The table may not exist or be empty yet, so there is nothing to check.
    with get_conn(check=False) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
