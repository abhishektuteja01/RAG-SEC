"""Postgres connection and chunk schema: pgvector for dense search, pg_search (ParadeDB)
for real BM25 in the same table -- not tsvector/ts_rank, which has neither length
normalization nor term saturation. DECISIONS.md INFRA-1/INFRA-4.
"""

import os
from urllib.parse import quote

import psycopg
from pgvector.psycopg import register_vector

from rag_sec.preflight import assert_variant_predicates

EMBEDDING_DIM = 1024  # BGE-M3's dense output size, not an independent choice (ARM1-1)

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


# Pinned per-variant, not as a total: RETR-24's contamination (Arm 4's B/C rows entering
# A's candidate pools) moved only the total, which nothing was comparing. Updating these
# numbers is the moment to re-audit every read for a `variant` predicate
# (scripts/checks/variant_predicates.py does that mechanically).
EXPECTED_CHUNK_COUNTS = {"A": 99654, "B": 4708, "C": 1665}

_preflight_done = False


def preflight(conn) -> None:
    """Fail if the corpus isn't the one every published number was measured on.

    Once per process, not per connection -- retrieve.py opens one per search. ~10ms.
    RETR-24's other half (a read missing its `variant` predicate) is checked in get_conn.
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
            "chunks table does not match EXPECTED_CHUNK_COUNTS (DECISIONS.md RETR-24):\n"
            + "\n".join(problems)
            + "\n\nIf this change was intentional, update EXPECTED_CHUNK_COUNTS in store.py"
            "\nAND re-run scripts/checks/variant_predicates.py -- new non-A rows"
            "\nare only safe if every read constrains `variant`. Writers that legitimately"
            "\nmove these counts should call get_conn(check=False)."
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
    if check:
        preflight(conn)
    return conn


def get_conn_string() -> str:
    """DSN form of get_conn()'s params, for libraries that want one string instead of
    kwargs (e.g. LangGraph's PostgresSaver) -- built from the same env vars so the two
    never drift apart.
    """
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", 5432))
    user = quote(os.environ["POSTGRES_USER"], safe="")
    password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    dbname = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def init_schema() -> None:
    # Table may not exist or be empty yet; nothing to assert about the corpus.
    with get_conn(check=False) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
