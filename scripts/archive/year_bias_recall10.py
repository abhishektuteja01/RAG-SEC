"""Does the year-bias RRF nudge (year_bias_sweep.py: +3.4pt recall@50 on dev at alpha=0.01)
also move recall@10, i.e. does the reranker actually surface the recovered gold chunk into
the top 10, not just the top 50?

Reranks the UNION of the baseline pool and the year-biased pool ONCE per question (the two
pools overlap heavily), then reads each condition's own top-10 back out of that shared score
dict -- one reranker pass per question instead of two, same trick a live A/B would need to
stay affordable on a laptop. Same baseline as year_bias_sweep.py and the shipped arm:
company_filter ON, and here also strip_query ON (rerank_query), since recall@10 is a
property of the actual shipped rerank step, not just candidate generation.

Usage:
    python scripts/archive/year_bias_recall10.py [--n N] [--split dev] [--alpha 0.01]
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.candidates import LIVE_VARIANT, bm25, chunk_texts, dense, rrf_fuse_scores  # noqa: E402
from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.company import strip_entity_framing  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    recall_at_k,
)
from rag_sec.fiscal_year import extract_years, year_distance_bonus  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

CANDIDATE_K = 50
TOP_K = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = pick_device()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device=device)

    def drain():
        # Without this, MPS's caching allocator never returns freed blocks and memory
        # climbs unboundedly on a 16GB unified-memory machine -- documented and measured
        # in rag_sec.retrieve._drain_mps (AGENT-24). Two models held at once here makes
        # it worse than the single-model shipped path, so drain every iteration, not
        # just at the end.
        if device.startswith("mps"):
            import torch
            torch.mps.empty_cache()

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)
    if args.n:
        rows = rows.head(args.n)

    base_hits, biased_hits = [], []
    n_scored = 0

    with get_conn() as conn:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"recall@10 [{args.split}]"):
            gold_idx = gold_relevant_chunk_ids(row)
            if not gold_idx:
                continue
            stem = _filing_stem(row)
            gold = [(stem, gi) for gi in gold_idx]
            q = row["question"]
            emb = embed_model.encode(q, normalize_embeddings=True)
            tickers = resolve_companies(q)
            query_years = extract_years(q)
            rerank_query = strip_entity_framing(q)

            lists = [
                dense(conn, emb, CANDIDATE_K, LIVE_VARIANT, tickers),
                bm25(conn, q, CANDIDATE_K, LIVE_VARIANT, tickers),
            ] if tickers else [
                dense(conn, emb, CANDIDATE_K, LIVE_VARIANT),
                bm25(conn, q, CANDIDATE_K, LIVE_VARIANT),
            ]
            scores = rrf_fuse_scores(lists)
            base_pool = sorted(scores, key=lambda d: scores[d], reverse=True)[:CANDIDATE_K]

            if query_years:
                adjusted = {p: s + year_distance_bonus(p[0], query_years, args.alpha)
                            for p, s in scores.items()}
                biased_pool = sorted(adjusted, key=lambda d: adjusted[d], reverse=True)[:CANDIDATE_K]
            else:
                biased_pool = base_pool

            union = list({*base_pool, *biased_pool})
            texts = chunk_texts(conn, union, LIVE_VARIANT)
            union = [p for p in union if p in texts]
            pairs = [(rerank_query, texts[p]) for p in union]
            rerank_scores = cross_encoder.predict(pairs, batch_size=32) if pairs else []
            score_of = dict(zip(union, (float(s) for s in rerank_scores)))
            drain()

            base_top10 = sorted(base_pool, key=lambda p: score_of.get(p, -1e9), reverse=True)[:TOP_K]
            biased_top10 = sorted(biased_pool, key=lambda p: score_of.get(p, -1e9), reverse=True)[:TOP_K]

            n_scored += 1
            base_hits.append(recall_at_k(base_top10, gold, TOP_K))
            biased_hits.append(recall_at_k(biased_top10, gold, TOP_K))

    bm, bse = mean_and_stderr(base_hits)
    ym, yse = mean_and_stderr(biased_hits)
    print(f"\nquestions scored: {n_scored}   alpha={args.alpha}\n")
    print(f"  recall@10, baseline (filtered_stripped)     : {bm:.3f} ± {bse:.3f}")
    print(f"  recall@10, + year_bias                      : {ym:.3f} ± {yse:.3f}"
          f"   ({ym - bm:+.3f})")


if __name__ == "__main__":
    main()
