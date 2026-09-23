"""OPTIONAL AND SLOW: embed data/chunks/ into Postgres (pgvector) and build the BM25 index.

You do not need this if you restore the database dump (scripts/setup_db.sh). It is how a
machine without the dump gets a corpus: ~1.4 chunks/s on an Apple M3 (MPS), so roughly
9-10 hours for all 99,654 chunks; a CUDA GPU is much faster.

Writes variant 'A' rows. Both legs are safe to re-run.

    local   embed every filing not yet loaded with BGE-M3, INSERT, then build HNSW
    bm25    build the pg_search BM25 index over the text already loaded (no GPU)

TRAPS
  * `local` is INSERT-only and skips any filing_stem already present, so an interrupted run
    resumes safely -- but a re-chunk is invisible to it. Delete that filing's rows first.
  * Gold labels are chunk indices; the rows must come from the same data/chunks/ the labels
    were built on (scripts/rebuild/gold_cache.py check).

Usage:
    docker compose up -d                       # Postgres with pgvector + pg_search
    uv run --env-file .env --extra rebuild scripts/rebuild/index.py local
    uv run --env-file .env --extra rebuild scripts/rebuild/index.py bm25
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.candidates import VARIANT  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, EMBED_MODEL_REVISION, pick_device  # noqa: E402
from rag_sec.store import (  # noqa: E402
    BM25_INDEX_SQL,
    HNSW_INDEX_SQL,
    get_conn,
    init_schema,
)

CHUNKS_DIR = _ROOT / "data" / "chunks"

# 32 fits a 16 GiB Mac that also holds Postgres; raise it on a real GPU.
EMBED_BATCH_SIZE = 32

INSERT_CHUNK_SQL = (
    "INSERT INTO chunks (filing_stem, chunk_index, heading, n_tokens, text, embedding, variant)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s)"
)
DONE_STEMS_SQL = "SELECT DISTINCT filing_stem FROM chunks WHERE variant = %s"
COUNT_SQL = "SELECT count(*) FROM chunks WHERE variant = %s"


def run_local() -> None:
    from sentence_transformers import SentenceTransformer

    init_schema()
    model = SentenceTransformer(EMBED_MODEL_NAME, revision=EMBED_MODEL_REVISION,
                                device=pick_device())

    paths = sorted(CHUNKS_DIR.glob("*.json"))
    if not paths:
        sys.exit(f"no chunk files in {CHUNKS_DIR} -- run scripts/rebuild/corpus.py first")
    print(f"Embedding chunks from {len(paths)} filings")

    # check=False: the corpus assertion compares row counts, and this script moves them.
    with get_conn(check=False) as conn:
        done = {r[0] for r in conn.execute(DONE_STEMS_SQL, (VARIANT,)).fetchall()}
        print(f"{len(done)}/{len(paths)} filings already indexed, skipping those")

        for i, path in enumerate(paths, 1):
            stem = path.stem
            if stem in done:
                continue
            chunks = json.loads(path.read_text())
            if not chunks:
                continue
            embeddings = model.encode(
                [c["text"] for c in chunks], batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False, normalize_embeddings=True,
            )
            rows = [
                (stem, idx, c.get("heading"), c.get("n_tokens"), c["text"], emb, VARIANT)
                for idx, (c, emb) in enumerate(zip(chunks, embeddings))
            ]
            with conn.cursor() as cur:
                cur.executemany(INSERT_CHUNK_SQL, rows)
            conn.commit()
            print(f"[{i}/{len(paths)}] {stem}: {len(rows)} chunks embedded")

        print("Building HNSW index...")
        conn.execute(HNSW_INDEX_SQL)
        conn.commit()
    print("Done.")


def run_bm25() -> None:
    init_schema()
    with get_conn(check=False) as conn:
        print("Building BM25 index (pg_search)...")
        conn.execute(BM25_INDEX_SQL)
        conn.commit()
        n = conn.execute(COUNT_SQL, (VARIANT,)).fetchone()[0]
    print(f"Done. BM25 index covers {n} variant-{VARIANT} chunks.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("leg", choices=("local", "bm25"))
    args = ap.parse_args()
    run_local() if args.leg == "local" else run_bm25()


if __name__ == "__main__":
    main()
