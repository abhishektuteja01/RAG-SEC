"""Arm 4: builds Strategy B (per-row) and C (raw + LLM summary) chunk variants for
the gold tables identified by arm4_identify_gold_tables.py, embeds them, and loads them
into `chunks` tagged variant='B'/'C'. Every non-target table in the same filing stays
Strategy A (identical text to the existing variant='A' rows) -- see DECISIONS.md ARM4-3.

Only the DELTA is embedded/inserted, not the whole filing: `chunk_blocks()` is cheap to
re-run (no ML calls), but re-embedding a filing's ~100-250 largely-unchanged chunks per
variant is not (BGE-M3 forward-pass cost scales with sequence length -- a filing's own
prose/table chunks run up to MAX_CHUNK_TOKENS=1500, ~100x the cost-per-item of the tiny
row-chunks this job actually needs to add). Chunks whose text is byte-identical to an
existing 'A' chunk are skipped; only new/changed chunks are embedded and inserted. The
superseded 'A' chunk(s) -- the ones containing the target table's old whole-table
representation -- get their `excluded_by_variant` flag updated so a retrieval run for
that variant doesn't see the same answer twice (see store.py SCHEMA_SQL).

Resumable: skips (filing_stem, variant) pairs already loaded, and caches Groq summaries to
disk per (filing_stem, table_index) so a crash mid-filing doesn't lose already-fetched
summaries or re-spend Groq quota re-fetching them.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from rag_sec.chunking import Chunk, chunk_blocks_with_variant
from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.parsing import Block, TableBlock, TextBlock
from rag_sec.store import get_conn, init_schema
from rag_sec.summarize import summarize_table

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PARSED_DIR = DATA_DIR / "parsed"
GOLD_TABLES_PATH = DATA_DIR / "day6_gold_tables.json"
SUMMARY_CACHE_PATH = DATA_DIR / "day6_table_summaries.json"
EMBED_BATCH_SIZE = 32


def load_targets() -> dict[str, set[int]]:
    records = json.loads(GOLD_TABLES_PATH.read_text())
    targets: dict[str, set[int]] = defaultdict(set)
    for r in records:
        targets[r["filing_stem"]].add(r["table_index"])
    return targets


def load_blocks(stem: str) -> list[Block]:
    data = json.loads((PARSED_DIR / f"{stem}.json").read_text())
    blocks: list[Block] = []
    for d in data:
        if "rows" in d:
            blocks.append(TableBlock(rows=d["rows"]))
        else:
            blocks.append(TextBlock(text=d["text"], is_title=d["is_title"]))
    return blocks


def load_summary_cache() -> dict[str, str]:
    if SUMMARY_CACHE_PATH.exists():
        return json.loads(SUMMARY_CACHE_PATH.read_text())
    return {}


def make_cached_summarizer(stem: str, target_idxs: list[int], cache: dict[str, str]) -> Callable:
    """chunk_blocks calls the summarizer once per target table, in ascending table_index
    order (blocks are iterated in document order) -- so a plain iterator over the sorted
    target indices tells us which table each call corresponds to, without changing the
    summarize_table(rows) callback signature."""
    idx_iter = iter(sorted(target_idxs))

    def wrapped(rows: list[list[str]]) -> str:
        idx = next(idx_iter)
        key = f"{stem}:{idx}"
        if key in cache:
            return cache[key]
        summary = summarize_table(rows)
        cache[key] = summary
        SUMMARY_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        return summary

    return wrapped


def fetch_baseline_chunks(conn, stem: str) -> dict[str, int]:
    """text -> chunk_index for this filing's existing variant='A' rows."""
    rows = conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE filing_stem = %s AND variant = 'A'", (stem,)
    ).fetchall()
    return {text: idx for idx, text in rows}


def load_variant_delta(conn, model: SentenceTransformer, stem: str, variant: str, chunks: list[Chunk]) -> None:
    """Embeds/inserts only chunks whose text doesn't already exist as an 'A' row for this
    filing, and flags the 'A' rows that got superseded (their text disappeared from the
    new chunk set) so a run for `variant` doesn't retrieve both the old and new copy."""
    baseline = fetch_baseline_chunks(conn, stem)
    new_texts = {c.text for c in chunks}

    superseded_idxs = [idx for text, idx in baseline.items() if text not in new_texts]
    if superseded_idxs:
        conn.execute(
            """UPDATE chunks SET excluded_by_variant = array_append(excluded_by_variant, %s)
               WHERE filing_stem = %s AND variant = 'A' AND chunk_index = ANY(%s)
                     AND NOT (%s = ANY(excluded_by_variant))""",
            (variant, stem, superseded_idxs, variant),
        )

    to_insert = [c for c in chunks if c.text not in baseline]
    if to_insert:
        texts = [c.text for c in to_insert]
        embeddings = model.encode(
            texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True
        )
        # new chunk_index namespace per (filing_stem, variant) -- independent of 'A''s numbering
        rows = [
            (stem, idx, c.heading, c.n_tokens, c.text, variant, emb)
            for idx, (c, emb) in enumerate(zip(to_insert, embeddings))
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (filing_stem, chunk_index, heading, n_tokens, text, variant, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (filing_stem, chunk_index, variant) DO NOTHING""",
                rows,
            )
    conn.commit()
    print(f"  {variant}: {len(to_insert)} new chunks inserted, {len(superseded_idxs)} 'A' chunks superseded")


def main() -> None:
    init_schema()
    targets = load_targets()
    print(f"{len(targets)} filings with gold tables to convert, {sum(len(v) for v in targets.values())} tables total")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    summary_cache = load_summary_cache()

    with get_conn(check=False) as conn:  # build-time: this script moves the counts
        done = {
            (r[0], r[1])
            for r in conn.execute("SELECT DISTINCT filing_stem, variant FROM chunks WHERE variant != 'A'").fetchall()
        }
        print(f"{len(done)} (filing, variant) pairs already built, skipping those")

        for i, (stem, table_idxs) in enumerate(sorted(targets.items()), 1):
            blocks = load_blocks(stem)
            idxs = sorted(table_idxs)
            print(f"[{i}/{len(targets)}] {stem}: {len(idxs)} gold table(s)")

            if (stem, "B") not in done:
                variant_map_b = {idx: "B" for idx in idxs}
                chunks_b = chunk_blocks_with_variant(blocks, table_variant_map=variant_map_b)
                load_variant_delta(conn, model, stem, "B", chunks_b)

            if (stem, "C") not in done:
                variant_map_c = {idx: "C" for idx in idxs}
                summarizer = make_cached_summarizer(stem, idxs, summary_cache)
                chunks_c = chunk_blocks_with_variant(blocks, table_variant_map=variant_map_c, summarize_table=summarizer)
                load_variant_delta(conn, model, stem, "C", chunks_c)

    print("Done.")


if __name__ == "__main__":
    main()
