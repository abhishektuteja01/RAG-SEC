"""Postgres + pgvector connection and schema for Arm 1 (dense retrieval).

DECISIONS.md #3: pgvector chosen over a dedicated vector DB. Embedding dim (1024) is
BGE-M3's dense output size, not chosen independently — see DECISIONS.md #15.
"""

import os

import psycopg
from pgvector.psycopg import register_vector

EMBEDDING_DIM = 1024

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    filing_stem TEXT NOT NULL,
    chunk_index INT NOT NULL,
    heading TEXT,
    n_tokens INT,
    text TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIM}),
    UNIQUE (filing_stem, chunk_index)
);
"""

HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(
        host="localhost",
        port=5432,
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
