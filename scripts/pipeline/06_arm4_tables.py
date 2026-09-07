"""Pipeline phase 06 — Arm 4: the A/B/C table-indexing branch. A DEAD END. READ THIS.

**THIS IS NOT THE MAIN LINE.** Arm 4 asked one question: does changing how a TABLE is
turned into chunks help retrieval? Three strategies over the 498 tables that dev questions
actually cite as gold evidence:

    A  the whole table as one chunk (pipe-delimited)   -- the CONTROL, and what ships
    B  one chunk per table row
    C  the raw table PLUS a separate LLM-written summary chunk, indexed alongside it

**A won on every metric and B/C were dropped (`ARM4-10`).** Worse, B and C were
DELIBERATELY NEVER RE-EMBEDDED after the RETR-7/RETR-8 re-index of 2026-09-04, so their
rows in Postgres still hold pre-re-index headings and vectors. They are therefore **no
longer text-comparable with variant A at all** — any A-vs-B or A-vs-C number produced today
is comparing two different corpora and means nothing. Variant A is the live control and the
only variant `rag_sec.candidates.LIVE_VARIANT` reads.

So why does this phase still exist?
  1. `score --variant A` computes a LIVE number, not a historical one: the `unfiltered_raw`
     control that every Day 8 retrieval gain is measured against (`RETR-22`, `RETR-29`,
     `RETR-31`). "day6" in its output filenames records when it was written, not what it
     computes.
  2. It is the only code that can score B or C at all, and `ARM4-10` is a published result
     that has to stay reproducible from something.

PRODUCES
    variants  Postgres `chunks` rows tagged variant='B'/'C' (+ `excluded_by_variant`
              flags on the superseded 'A' rows), and data/day6_table_summaries.json
    prepare   data/day6_arm4_{A,B,C}_rerank_payload.json   (the cluster's input)
    score     data/day6_arm4_{A,B,C}_dev_results.json + _dev_failures.md

READS
    data/day6_gold_tables.json    which (filing, table_index) answers which question,
                                  from scripts/archive/arm4_identify_gold_tables.py
    data/parsed/*.json            the block lists phase 01 wrote
    data/day6_table_summaries.json  cached variant-C summaries — ALL 498 ALREADY CACHED
    Postgres `chunks`, and the matched question set via rag_sec.eval
    score also reads a cluster scores file (default day6_arm4_{variant}_rerank_scores.jsonl)

DECISIONS.md ROWS THIS BACKS
    ARM4-2    Strategy C indexes the summary ALONGSIDE the raw table, not instead of it,
              so a bad summary cannot regress below A for that table.
    ARM4-3    B/C scoped to the 498 gold-cited tables (of 38,959 across 324 filings), and
              every non-target table in the same filing stays byte-identical to its A text.
    ARM4-5    variant-aware relevance labeling: gold labels for B/C come from the
              variant-filtered candidate pool in `chunks`, NOT from data/chunks/*.json,
              which only ever held Strategy A. This is why `score` needs Postgres.
    ARM4-6    the summarizer is Claude Haiku 4.5, NOT Groq — a 50-table spot-check of the
              Groq summaries found a 22% comprehension error rate. (The merged
              build_table_variants.py docstring still said "Groq summaries"; that wording
              was stale and is corrected here.)
    ARM4-7    dash/em-dash "zero" cells are normalized in `rag_sec.summarize` before the
              call, because a prompt-only instruction did not hold.
    ARM4-8    delta-only embedding: only chunks whose text is not already an 'A' row get
              embedded, ~100x cheaper than re-embedding a filing per variant.
    ARM4-9    224/1235 dev questions have no matched gold table and are excluded.
    ARM4-10   the verdict: A beats B and C on every metric.
    GOLD-5    B/C were rescored offline under the new labeler from the saved orderings.
    RETR-22 / RETR-29 / RETR-31   consume `score --variant A`'s output as the control.

WHEN THIS ACTUALLY RAN (calendar dates, not "Day N")
    2026-08-29   gold tables identified; variant-C summaries fetched (data/
                 day6_table_summaries.json, 498 entries)
    2026-08-30   the whole A/B/C run — variants built, three cluster passes scored, and
                 every day6_*/day7_* artifact written. The published
                 data/day6_arm4_{A,B,C}_dev_results.json are 08-30 19:09-19:12.
    NOT 2026-09-04. The RETR-7/RETR-8 re-index touched variant A only.

TRAPS
  * `variants` WRITES TO THE LIVE INDEX and `score --variant A` OVERWRITES a published
    results file. Neither is gated, because both merged scripts behaved that way and this
    consolidation is behaviour-preserving -- but `variants` skips (filing, variant) pairs
    already present, so on a built index it inserts nothing. Money is the one thing that
    IS gated; see the next bullet.
  * `variants` COSTS MONEY. Strategy C calls the Anthropic API (claude-haiku-4-5,
    498 calls) once per gold table. All 498 are already cached on disk, so a normal run
    spends nothing — but a lost or partial cache would silently re-spend. An uncached
    summary therefore raises unless --allow-paid-summaries is passed. Do not pass it.
  * B and C use their OWN chunk_index namespace per (filing_stem, variant), numbered from
    0 independently of A. A bare (stem, index) pair COLLIDES across variants, which is why
    Arm 4 carries (stem, index, variant) triples everywhere and Arm 3 does not.
  * GPU PROVENANCE IS UNVERIFIED AND CONTRADICTORY. `score` stamps its results file
    "Tesla V100-SXM2-32GB", while the Arm 3 finalizer
    (scripts/archive/arm3_rerank_score.py:100) stamps "Tesla V100-PCIE-32GB" for what was
    the same cluster and the same week. Only one can be right; no log survives to settle
    it. The string is preserved verbatim rather than corrected or deleted — see
    RERANK_DEVICE_UNVERIFIED below.
  * The archived cluster leg for this arm is scripts/archive/arm4_rerank_hpc.py. It is NOT
    moved into scripts/pipeline/hpc/: it differs from the Arm 3 twin by three lines and
    exists only as the record of three finished GPU passes.
  * data/day6_arm4_{A,B,C}_cpu_dev_* are 5-question warm-ups. Three of them report ~0.9
    and are dangerous if mistaken for results.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.candidates import rrf_fuse  # noqa: E402
from rag_sec.chunking import Chunk, chunk_blocks_with_variant  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    gold_relevant_chunk_ids_db,
    load_matched_questions,
    mean_and_stderr,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from rag_sec.parsing import Block, TableBlock, TextBlock  # noqa: E402
from rag_sec.store import get_conn, init_schema  # noqa: E402

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"

# Which (filing, table_index) pairs are gold evidence for a dev question. 498 tables across
# 281 filings (ARM4-3). Written by scripts/archive/arm4_identify_gold_tables.py; also read
# by rag_sec.eval, so it is NOT an Arm-4-only artifact.
GOLD_TABLES_PATH = DATA_DIR / "day6_gold_tables.json"

# Variant-C summaries, keyed "stem:table_index". Written incrementally so a crash mid-run
# cannot lose already-purchased summaries. ALL 498 are present; see EXPECTED_SUMMARIES.
SUMMARY_CACHE_PATH = DATA_DIR / "day6_table_summaries.json"
EXPECTED_SUMMARIES = 498  # ARM4-3's table count == the number of paid calls C ever needed

# Embedding batch for the delta rows only (ARM4-8). Small on purpose: B's row-chunks are
# tiny and the batch never dominates, so a bigger number buys nothing and risks memory.
EMBED_BATCH_SIZE = 32

# Arm 4's pool sizes, kept at Arm 3's values so the two arms' numbers sit on one scale.
# TOP_K is the fused pool handed to the cluster, CANDIDATE_K each leg's own depth.
TOP_K = 50
CANDIDATE_K = 50

VARIANTS = ("A", "B", "C")

# UNVERIFIED, PRESERVED VERBATIM. This exact string is what the published
# data/day6_arm4_{A,B,C}_dev_results.json carry, so it stays byte-identical here or those
# files stop round-tripping. But it CONTRADICTS scripts/archive/arm3_rerank_score.py:100,
# which stamps "Tesla V100-PCIE-32GB (Northeastern Explorer HPC)" for the same cluster in
# the same week — and rerank_hpc.sbatch requests `gpu:v100-sxm2:1`, which is suggestive but
# not evidence, since the Arm 3/Arm 4 passes were interactive `srun` bookings whose actual
# node was never logged. Only one card can be right. Nothing on disk settles it, so neither
# string is treated as fact and neither is deleted.
RERANK_DEVICE_UNVERIFIED = "Tesla V100-SXM2-32GB (Northeastern Explorer HPC)"

# Worst-failure listing length, matching Arm 3's so the two failure files read the same.
N_WORST = 20


# ─── STEP 1: build the B and C chunk variants (COSTS MONEY, WRITES THE INDEX) ───
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


def make_cached_summarizer(
    stem: str, target_idxs: list[int], cache: dict[str, str], allow_paid: bool
) -> Callable:
    """chunk_blocks calls the summarizer once per target table, in ascending table_index
    order (blocks are iterated in document order) -- so a plain iterator over the sorted
    target indices tells us which table each call corresponds to, without changing the
    summarize_table(rows) callback signature.

    `allow_paid` is the money gate. A cache hit is free; a MISS is a live
    claude-haiku-4-5 call, and 498 of them is the whole of Arm 4's API bill. Since all 498
    are already on disk, a miss means the cache moved or was truncated -- which is exactly
    the situation where silently re-spending is worst. So a miss raises unless the operator
    said so on the command line.
    """
    idx_iter = iter(sorted(target_idxs))

    def wrapped(rows: list[list[str]]) -> str:
        idx = next(idx_iter)
        key = f"{stem}:{idx}"
        if key in cache:
            return cache[key]
        if not allow_paid:
            raise SystemExit(
                f"REFUSING TO SPEND: no cached summary for {key} in {SUMMARY_CACHE_PATH}\n"
                f"  All {EXPECTED_SUMMARIES} variant-C summaries are supposed to be cached "
                "on disk (ARM4-3).\n"
                "  A miss means the cache is missing or truncated, and continuing would "
                "make a paid\n"
                "  Anthropic call (claude-haiku-4-5) for a variant that LOST (ARM4-10). "
                "Restore the\n"
                "  cache from git, or pass --allow-paid-summaries if you really mean to buy "
                "it again."
            )
        # Imported HERE, not at module scope: importing it constructs nothing, but keeping
        # the only money-spending import inside the only gated path makes the blast radius
        # of this script readable at a glance.
        from rag_sec.summarize import summarize_table

        summary = summarize_table(rows)
        cache[key] = summary
        SUMMARY_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        return summary

    return wrapped


def fetch_baseline_chunks(conn, stem: str) -> dict[str, int]:
    """text -> chunk_index for this filing's existing variant='A' rows."""
    rows = conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE filing_stem = %s AND variant = 'A'",
        (stem,),
    ).fetchall()
    return {text: idx for idx, text in rows}


def load_variant_delta(conn, model, stem: str, variant: str, chunks: list[Chunk]) -> None:
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
            texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=True,
        )
        # new chunk_index namespace per (filing_stem, variant) -- independent of 'A''s
        # numbering, which is why every Arm 4 id is a (stem, index, variant) triple
        rows = [
            (stem, idx, c.heading, c.n_tokens, c.text, variant, emb)
            for idx, (c, emb) in enumerate(zip(to_insert, embeddings))
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks
                       (filing_stem, chunk_index, heading, n_tokens, text, variant, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (filing_stem, chunk_index, variant) DO NOTHING""",
                rows,
            )
    conn.commit()
    print(f"  {variant}: {len(to_insert)} new chunks inserted, "
          f"{len(superseded_idxs)} 'A' chunks superseded")


def cmd_variants(args: argparse.Namespace) -> None:
    """Build B and C for every gold table. Resumable per (filing_stem, variant)."""
    targets = load_targets()
    summary_cache = load_summary_cache()
    n_tables = sum(len(v) for v in targets.values())
    print(f"{len(targets)} filings with gold tables, {n_tables} tables total (ARM4-3)")
    print(f"{len(summary_cache)}/{EXPECTED_SUMMARIES} variant-C summaries cached")

    # Loud, but NOT a gate: the (filing_stem, variant) skip below means a re-run on a
    # fully-built index inserts nothing, so the original script's behaviour is preserved.
    # The only thing gated is SPEND -- see make_cached_summarizer.
    print("\nNOTE: this INSERTS variant B/C rows into the live `chunks` table and sets")
    print("`excluded_by_variant` on existing 'A' rows. B and C LOST (ARM4-10) and were")
    print("deliberately left un-re-embedded after RETR-7/RETR-8, so anything rebuilt now")
    print("is not text-comparable with the live variant A.\n")

    from sentence_transformers import SentenceTransformer

    init_schema()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    with get_conn(check=False) as conn:  # build-time: this step moves the counts
        done = {
            (r[0], r[1])
            for r in conn.execute(
                "SELECT DISTINCT filing_stem, variant FROM chunks WHERE variant != 'A'"
            ).fetchall()
        }
        print(f"{len(done)} (filing, variant) pairs already built, skipping those")

        for i, (stem, table_idxs) in enumerate(sorted(targets.items()), 1):
            blocks = load_blocks(stem)
            idxs = sorted(table_idxs)
            print(f"[{i}/{len(targets)}] {stem}: {len(idxs)} gold table(s)")

            if (stem, "B") not in done:
                chunks_b = chunk_blocks_with_variant(
                    blocks, table_variant_map={idx: "B" for idx in idxs})
                load_variant_delta(conn, model, stem, "B", chunks_b)

            if (stem, "C") not in done:
                summarizer = make_cached_summarizer(
                    stem, idxs, summary_cache, args.allow_paid_summaries)
                chunks_c = chunk_blocks_with_variant(
                    blocks, table_variant_map={idx: "C" for idx in idxs},
                    summarize_table=summarizer)
                load_variant_delta(conn, model, stem, "C", chunks_c)

    print("Done.")


# ─── STEP 2: build one variant's cluster payload (laptop, needs Postgres) ───────
ChunkId = tuple[str, int, str]  # (filing_stem, chunk_index, variant)


def retrieve_dense(conn, embedding, variant: str, k: int) -> list[ChunkId]:
    """Arm 4's pool: this variant's own rows, plus the 'A' rows it did not supersede.

    Not `rag_sec.candidates.dense`: that one takes a single `variant` and is correct for
    every arm EXCEPT this one, whose pool is a union across two variants (ARM4-3). The
    predicate is still explicit about `variant`, which is what RETR-24's guard requires.
    """
    rows = conn.execute(
        """SELECT filing_stem, chunk_index, variant FROM chunks
           WHERE (variant = 'A' AND NOT (%s = ANY(excluded_by_variant))) OR variant = %s
           ORDER BY embedding <=> %s LIMIT %s""",
        (variant, variant, embedding, k),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def retrieve_bm25(conn, query_text: str, variant: str, k: int) -> list[ChunkId]:
    rows = conn.execute(
        """SELECT filing_stem, chunk_index, variant, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s)
             AND ((variant = 'A' AND NOT (%s = ANY(excluded_by_variant))) OR variant = %s)
           ORDER BY s DESC LIMIT %s""",
        (query_text, variant, variant, k),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def fetch_texts(conn, triples: list[ChunkId]) -> dict[ChunkId, str]:
    """Text for a pool of triples, in the caller's order.

    The SQL below is byte-identical to the string registered in
    `rag_sec.preflight.ALLOWED` -- it returns every variant on purpose, because the caller
    keys by the full (stem, index, variant) triple. Editing it revokes that exemption and
    `scripts/checks/variant_predicates.py` will fail, by design.
    """
    if not triples:
        return {}
    stems = list({t[0] for t in triples})
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, variant, text FROM chunks WHERE filing_stem = ANY(%s)",
        (stems,),
    ).fetchall()
    lookup = {(r[0], r[1], r[2]): r[3] for r in rows}
    return {t: lookup[t] for t in triples if t in lookup}


def cmd_prepare(args: argparse.Namespace) -> None:
    """Dump this variant's fused top-50 with text INLINED, one payload per variant."""
    from sentence_transformers import SentenceTransformer

    variant = args.variant
    payload_path = DATA_DIR / f"day6_arm4_{variant}_rerank_payload.json"

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    print(f"Preparing Arm 4 variant={variant} rerank payload for {len(dev)} dev questions")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    payload = []

    with get_conn() as conn:
        for _, row in tqdm(dev.iterrows(), total=len(dev)):
            query_emb = embed_model.encode(row["question"], normalize_embeddings=True)
            dense = retrieve_dense(conn, query_emb, variant, CANDIDATE_K)
            bm25 = retrieve_bm25(conn, row["question"], variant, CANDIDATE_K)
            fused = rrf_fuse([dense, bm25])[:TOP_K]
            texts = fetch_texts(conn, fused)

            payload.append({
                "id": row["id"],
                "question": row["question"],
                "filing_stem": (f"{row['company_symbol']}_{int(row['report_year'])}"
                                f"_{int(row['company_cik'])}"),
                "candidates": [
                    [stem, idx, v, texts[(stem, idx, v)]]
                    for stem, idx, v in fused if (stem, idx, v) in texts
                ],
            })

    payload_path.write_text(json.dumps(payload))
    print(f"Wrote {len(payload)} questions' candidates to {payload_path}")
    print("Copy this file to the HPC node via the transfer node (DECISIONS.md ARM3-2 -- the")
    print("login node throttles/kills large transfers), e.g.:")
    print(f"  scp {payload_path} <user>@xfer.discovery.neu.edu:~/rerank_payload_{variant}.json")


# ─── STEP 3 (cluster): scripts/archive/arm4_rerank_hpc.py ──────────────────────
# Deliberately not moved here. It is the record of three finished GPU passes and differs
# from the Arm 3 twin by three lines; the live cluster leg is scripts/pipeline/hpc/.


# ─── STEP 4: score one variant against variant-aware gold labels ───────────────
def load_scores(path: Path) -> dict[str, dict]:
    scores = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                scores[row["id"]] = row
    return scores


def cmd_score(args: argparse.Namespace) -> None:
    """Join the cluster's orderings with variant-aware gold labels (ARM4-5). Needs Postgres.

    For --variant A this is a LIVE number, the `unfiltered_raw` control every Day 8 gain is
    measured against. It overwrites a published file, unconditionally, as it always has.
    """
    variant = args.variant
    scores_path = args.scores or DATA_DIR / f"day6_arm4_{variant}_rerank_scores.jsonl"
    results_path = DATA_DIR / f"day6_arm4_{variant}_dev_results.json"
    failures_path = DATA_DIR / f"day6_arm4_{variant}_dev_failures.md"

    # No gate here: the merged script wrote unconditionally and this is behaviour-
    # preserving. It IS destructive, so it says so before doing it.
    print(f"variant={variant}: will overwrite {results_path.name} and {failures_path.name}")
    if variant == "A":
        print("  Variant A is the LIVE unfiltered_raw control (RETR-22/29/31) -- a different")
        print("  scores file here changes a published baseline.")
    else:
        print(f"  Variant {variant} was never re-embedded after RETR-7/RETR-8 (ARM4-10), so")
        print("  this number compares two different corpora.")

    scores = load_scores(scores_path)

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    dev = dev[dev["id"].isin(scores)].reset_index(drop=True)
    print(f"Scoring variant={variant}: {len(dev)}/{len(scores)} scored questions (dev split)")

    per_question = []
    with get_conn() as conn:
        for _, row in dev.iterrows():
            s = scores[row["id"]]
            retrieved = [tuple(c) for c in s["reranked"]]

            filing_stem = (f"{row['company_symbol']}_{int(row['report_year'])}"
                           f"_{int(row['company_cik'])}")
            relevant = [(filing_stem, ci, v)
                        for ci, v in gold_relevant_chunk_ids_db(row, conn, variant)]

            per_question.append({
                "id": row["id"],
                "question": row["question"],
                "filing_stem": filing_stem,
                "n_relevant": len(relevant),
                "recall_10": recall_at_k(retrieved, relevant, 10),
                "recall_50": recall_at_k(retrieved, relevant, 50),
                "ndcg_10": ndcg_at_k(retrieved, relevant, 10),
                "mrr": mrr(retrieved, relevant),
                "rerank_latency_s": s["latency_s"],
                "top_5_retrieved": retrieved[:5],
            })

    metrics = {}
    for key in ("recall_10", "recall_50", "ndcg_10", "mrr"):
        mean, stderr = mean_and_stderr([q[key] for q in per_question])
        metrics[key] = {"mean": mean, "stderr": stderr}
        print(f"{key}: {mean:.3f} +/- {stderr:.3f}")

    latencies = sorted(q["rerank_latency_s"] for q in per_question)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"rerank latency (top-50 candidates, V100): p50={p50*1000:.0f}ms, "
          f"p95={p95*1000:.0f}ms")

    results_path.write_text(json.dumps({
        "variant": variant,
        "rerank_model": RERANK_MODEL_NAME,
        "rerank_device": RERANK_DEVICE_UNVERIFIED,
        "top_k": TOP_K,
        "n": len(dev),
        "metrics": metrics,
        "rerank_latency_ms": {"p50": p50 * 1000, "p95": p95 * 1000},
        "per_question": per_question,
    }, indent=2))
    print(f"Results written to {results_path}")

    worst = sorted(
        per_question,
        key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0),
    )[:N_WORST]
    with open(failures_path, "w") as f:
        f.write(f"# Arm 4 variant={variant} (Arm 2 hybrid + HPC cross-encoder rerank) — "
                f"{N_WORST} worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, "
                    f"filing={w['filing_stem']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {failures_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "variants", help="build B/C chunk variants. WRITES THE INDEX. CAN COST MONEY.",
        description=(
            "Step 1: build Strategy B (per-row) and C (raw + LLM summary) chunks for the "
            f"{EXPECTED_SUMMARIES} gold tables, embed the delta only (ARM4-8), and load "
            "them as variant='B'/'C'.\n\n"
            "MONEY: Strategy C needs one claude-haiku-4-5 call per gold table "
            f"({EXPECTED_SUMMARIES} calls). All of them are already cached in "
            f"{SUMMARY_CACHE_PATH.name}, so a normal run spends NOTHING and an uncached "
            "summary is a hard error unless --allow-paid-summaries is passed.\n\n"
            "DEAD END: B and C lost (ARM4-10) and were never re-embedded after the "
            "RETR-7/RETR-8 re-index, so rebuilding them now yields rows that are not "
            "text-comparable with the live variant A. It skips (filing, variant) pairs "
            "already in the index, so a re-run on a built index inserts nothing."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-paid-summaries", action="store_true",
                   help="permit live Anthropic calls for uncached variant-C summaries. "
                        "Do not use: all summaries are already on disk")
    p.set_defaults(fn=cmd_variants)

    p = sub.add_parser(
        "prepare", help="build one variant's cluster payload (needs Postgres)",
        description=("Step 2: run variant `--variant`'s retrieval over the dev split and "
                     "dump the fused top-50 with text inlined, as (stem, index, variant) "
                     "triples.\n\nRe-running --variant A overwrites the 230 MB payload "
                     "COST-14 reused as the slice arms' candidate pool; the content should "
                     "come out identical, but there is no reason to find out."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("-n", type=int, default=None,
                   help="limit to the first N dev questions (sanity/timing check)")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser(
        "score", help="score one variant. Variant A is a LIVE number. Needs Postgres.",
        description=("Step 4: join the cluster's reranked orderings with variant-aware gold "
                     "labels (ARM4-5, which is why this needs Postgres) and write "
                     "recall@10/50, nDCG@10 and MRR.\n\n"
                     "--variant A computes the LIVE `unfiltered_raw` control that RETR-22 / "
                     "RETR-29 / RETR-31 are measured against; B and C are a dead end "
                     "(ARM4-10) and are no longer text-comparable with A. Either way this "
                     "OVERWRITES data/day6_arm4_{variant}_dev_results.json and _failures.md "
                     "in place.\n\nFree: reads only, no GPU and no API."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--scores", type=Path, default=None,
                   help="default data/day6_arm4_{variant}_rerank_scores.jsonl")
    p.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
