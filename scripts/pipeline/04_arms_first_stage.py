"""Pipeline phase 04 — first-stage retrieval arms, scored on the dev split.

TWO ARMS, one argument apart. They are deliberately one change apart (spec.md 2.2: change
one thing per arm), which is why they belong in one file:

    arm1   dense only. BGE-M3 query embedding -> pgvector cosine, FIRST_STAGE_K back.
    arm2   hybrid. The same dense list PLUS a pg_search BM25 list, fused with Reciprocal
           Rank Fusion, then truncated to the same FIRST_STAGE_K. Nothing else differs, so the two
           results are directly comparable.

PRODUCES  (suffixes stack: _companyfilter from --company-filter, _limitN from --limit)
    arm1   data/day3_arm1_dev_results.json   data/day3_arm1_dev_failures.md
    arm2   data/day4_arm2_dev_results.json   data/day4_arm2_dev_failures.md
    The results JSON carries the run's config, the aggregate metrics with standard errors,
    and one row per question including its top 5 retrieved (stem, chunk_index) pairs.
    The failures markdown is the 20 worst questions by recall@10.

READS
    Postgres (`chunks`, variant 'A' only), the T2-RAGBench dev split via
    rag_sec.eval.load_matched_questions, and the frozen gold labels from phase 02.

DECISIONS.md ROWS THIS BACKS
    ARM1-2    no cik+year metadata pre-filter by default: pure semantic search across the
              whole corpus, so recall reflects real retrieval difficulty rather than
              "did we already know which filing to look in".
    RETR-5    --company-filter scopes candidates to the company named in the question. Off
              by default, and filtered runs write to a *_companyfilter sidecar rather than
              overwriting the unfiltered numbers.
    INFRA-4 / ARM2-1   real BM25 via pg_search, not tsvector/ts_rank — spec.md's explicit
              trap: no length normalization, no term saturation, degrades on long docs.
    RETR-36   dense/bm25/rrf_fuse live in src/rag_sec/candidates.py and `variant` is a
              REQUIRED argument there. These arms pass LIVE_VARIANT explicitly. The
              helpers used to exist as 11 hand-copied definitions across 5 files, which is
              what let RETR-24 — a MISSING variant predicate — need the same fix 7 times.
    RETR-35   gold labels are coverage-based since this row; nDCG/MRR under older labels
              are not comparable.
    RETR-7/8  the corpus was re-indexed under these, so anything measured before
              2026-09-04 is against a different corpus.

WHEN THIS ACTUALLY RAN
    2026-08-27         Arm 1's first pass, with the eval harness.
    2026-08-28 -> 29   Arm 2.
    2026-09-01         the --company-filter sidecars (RETR-5): data/day3_arm1_dev_*
                       _companyfilter 11:27, data/day4_arm2_dev_*_companyfilter 11:36.
    2026-09-04         BOTH arms re-run after the RETR-7/RETR-8 re-index, OVERWRITING the
                       08-27/08-29 files: arm1 results 03:33, arm2 results 03:37. The
                       pre-re-index copies survive only as *.bak_old_labeler (2026-08-30).
                       So the numbers currently in data/day3_* and data/day4_* are from
                       2026-09-04, not from the dates the "day3_"/"day4_" prefixes suggest
                       — those prefixes are plan numbers and carry no date at all.

TRAPS
  * These arms are DEV-ONLY. spec.md 2.2 holds test back to one touch per arm at the end;
    there is no --split here because that hold-back is the point.
  * Quote levels, not deltas, against anything from before 2026-09-04: corpus (RETR-7/8)
    and labels (RETR-35) both moved, so a pre/post difference mixes two changes.
  * recall@10 can be NaN for a question with no gold labels; the worst-failures sort maps
    NaN to 0 rather than dropping it, so a label-less question can appear in that list.
  * --limit is for smoke tests only and renames the outputs (_limitN) so it can never
    overwrite a published artifact. A limited run's metrics are not comparable to anything.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from rag_sec.candidates import LIVE_VARIANT, RRF_K, bm25, dense, rrf_fuse  # noqa: E402
from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, pick_device  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    mrr,
    ndcg_at_k,
    recall_at_k,
    write_worst_failures,
)
# How many each of dense/BM25 contributes to arm2's fused pool. Imported, not re-declared,
# so the offline arms and the shipping path cannot drift. It equals FIRST_STAGE_K below, so
# the fused list is drawn from two lists of the same depth — an asymmetric pair would bias
# RRF toward whichever list was allowed to be longer.
from rag_sec.retrieve import CANDIDATE_K  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"

# 50 for both arms, and it must stay equal across them: it is the ranked-list depth the
# metrics are computed over, so a different depth would make arm1 and arm2 incomparable.
# It is also the pool the later reranker arms consume. Named FIRST_STAGE_K, not TOP_K:
# rag_sec.retrieve.TOP_K is the shipped path's final depth of 10, and one identifier
# meaning both 10 and 50 across files that import from each other is a trap.
FIRST_STAGE_K = 50

# Metric columns, in the order they are printed and stored. recall@10 is the headline;
# recall@50 is the ceiling the reranker arms can reach from this pool.
METRIC_KEYS = ["recall_10", "recall_50", "ndcg_10", "mrr"]

# Per-arm output stems and titles. The dayN_ prefixes are plan numbers kept because live
# consumers and DECISIONS rows reference these exact filenames.
ARMS = {
    "arm1": {
        "results": DATA_DIR / "day3_arm1_dev_results.json",
        "failures": DATA_DIR / "day3_arm1_dev_failures.md",
        "title": "Arm 1 (dense, BGE-M3)",
        "banner": "Running Arm 1 on {n} dev-split questions",
    },
    "arm2": {
        "results": DATA_DIR / "day4_arm2_dev_results.json",
        "failures": DATA_DIR / "day4_arm2_dev_failures.md",
        "title": "Arm 2 (hybrid: dense + BM25, RRF)",
        "banner": "Running Arm 2 (hybrid: dense + BM25, RRF fusion) on {n} dev-split questions",
    },
}


def _suffixed(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def _retrieve(arm: str, conn, model, question: str, tickers: list[str]):
    """The one line of difference between the two arms."""
    query_emb = model.encode(question, normalize_embeddings=True)
    if arm == "arm1":
        return dense(conn, query_emb, FIRST_STAGE_K, LIVE_VARIANT, tickers)
    dense_hits = dense(conn, query_emb, CANDIDATE_K, LIVE_VARIANT, tickers)
    bm25_hits = bm25(conn, question, CANDIDATE_K, LIVE_VARIANT, tickers)
    return rrf_fuse([dense_hits, bm25_hits])[:FIRST_STAGE_K]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("arm", choices=sorted(ARMS),
                    help="arm1 = dense only; arm2 = dense + BM25 fused with RRF")
    # Off by default so the already-published numbers stay reproducible from this script.
    ap.add_argument("--company-filter", action="store_true",
                    help="scope candidates to the company named in the question (RETR-5)")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="first N dev questions only, smoke test. Renames the outputs to "
                         "*_limitN so a published artifact can never be overwritten")
    opts = ap.parse_args()

    arm = opts.arm
    cfg = ARMS[arm]
    suffix = "_companyfilter" if opts.company_filter else ""
    if opts.limit:
        suffix += f"_limit{opts.limit}"
    results_path = _suffixed(cfg["results"], suffix)
    failures_path = _suffixed(cfg["failures"], suffix)

    # ─── STEP 1: the dev split ─────────────────────────────────────────────────
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if opts.limit:
        dev = dev.head(opts.limit)
    print(cfg["banner"].format(n=len(dev)))

    # ─── STEP 2: retrieve and score, one question at a time ───────────────────
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL_NAME, device=pick_device())
    per_question = []

    with get_conn() as conn:
        for _, row in dev.iterrows():
            tickers = resolve_companies(row["question"]) if opts.company_filter else []
            retrieved = _retrieve(arm, conn, model, row["question"], tickers)
            filing_stem = Path(row["chunk_file"]).stem
            relevant = [(filing_stem, i) for i in gold_relevant_chunk_ids(row)]

            per_question.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "chunk_file": row["chunk_file"],
                    "n_relevant": len(relevant),
                    "recall_10": recall_at_k(retrieved, relevant, 10),
                    "recall_50": recall_at_k(retrieved, relevant, 50),
                    "ndcg_10": ndcg_at_k(retrieved, relevant, 10),
                    "mrr": mrr(retrieved, relevant),
                    "top_5_retrieved": retrieved[:5],
                }
            )

    # ─── STEP 3: aggregate with standard errors (spec.md 2.3) ─────────────────
    metrics = {}
    for key in METRIC_KEYS:
        mean, stderr = mean_and_stderr([q[key] for q in per_question])
        metrics[key] = {"mean": mean, "stderr": stderr}
        print(f"{key}: {mean:.3f} +/- {stderr:.3f}")

    # ─── STEP 4: write the results JSON ───────────────────────────────────────
    payload = {"model": EMBED_MODEL_NAME}
    if arm == "arm2":  # arm1 has no fusion config to record
        payload |= {"candidate_k": CANDIDATE_K, "rrf_k": RRF_K}
    payload |= {"top_k": FIRST_STAGE_K, "n": len(dev), "metrics": metrics,
                "per_question": per_question}
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {results_path}")

    # ─── STEP 5: write the 20 worst failures ──────────────────────────────────
    write_worst_failures(failures_path, cfg["title"], per_question, "chunk_file")
    print(f"Worst failures written to {failures_path}")


if __name__ == "__main__":
    main()
