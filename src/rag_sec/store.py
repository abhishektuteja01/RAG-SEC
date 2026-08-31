"""Postgres + pgvector(+pg_search) connection and schema for Arms 1-2.

DECISIONS.md INFRA-1: pgvector chosen over a dedicated vector DB. Embedding dim (1024) is
BGE-M3's dense output size, not chosen independently — see DECISIONS.md ARM1-1.
DECISIONS.md INFRA-4: pg_search (ParadeDB) adds a real BM25 index alongside pgvector, in
the same table, for Arm 2 hybrid search — not Postgres's tsvector/ts_rank, which lacks
BM25's document-length normalization and term saturation (spec.md's explicit trap).
"""

import os

import psycopg
from pgvector.psycopg import register_vector

EMBEDDING_DIM = 1024

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
    -- Arm 4 (DECISIONS.md ARM4-*): 'A' = whole-table chunk (all Arm 1-3 data,
    -- and the default for every filing). 'B'/'C' only exist for a handful of
    -- gold-table-adjacent chunks in the 324 dev-relevant filings, alongside
    -- 'A' rows. A given 'A' row is only ever superseded (excluded from that
    -- run's retrieval) for the variants listed here -- e.g. an 'A' chunk that
    -- IS a gold table gets ['B','C'] once replacements exist, so it doesn't
    -- also appear as a duplicate answer in those runs.
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


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    register_vector(conn)
    return conn


def init_schema() -> None:
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
