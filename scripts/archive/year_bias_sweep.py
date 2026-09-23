"""Does an additive year-proximity nudge in RRF fusion recover gold chunks lost to
sibling-year confusion (RETR-3: 50% of hand-read misses are the right company's wrong year)?

Same shape as `company_filter_ab.py` (RETR-5's own measurement): recall@50 on the
CANDIDATE POOL only, no cross-encoder. That recall is a BINARY any-gold-in-pool hit, NOT
the coverage-based `recall_at_k` every published table cell uses -- see `RETR-49`. Deltas
between alphas here are self-consistent; the levels are not comparable to a table number. The diagnosis is "the gold never reaches the
reranker", so reranking is not part of the question and running it would cost GPU time
without changing the answer. `strip_query` is irrelevant here for the same reason -- it
only changes what the reranker sees.

Baseline = the shipped candidate-generation side of `filtered_stripped` (company filter
ON, tickers resolved from the question text). year_bias is layered on top of that, one
variable (alpha) at a time -- not a new filter/strip grid.

Usage:
    python scripts/archive/year_bias_sweep.py [--n N] [--split dev] [--alphas 0,0.002,0.005,0.01,0.02,0.05,0.1]
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.candidates import LIVE_VARIANT, bm25, dense, rrf_fuse_scores  # noqa: E402
from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
)
from rag_sec.fiscal_year import extract_years, year_distance_bonus  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

CANDIDATE_K = 50
DEFAULT_ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def _biased_top_k(scores: dict, query_years: list, alpha: float, k: int) -> set:
    if alpha == 0.0 or not query_years:
        return set(sorted(scores, key=lambda d: scores[d], reverse=True)[:k])
    adjusted = {
        pair: s + year_distance_bonus(pair[0], query_years, alpha)
        for pair, s in scores.items()
    }
    return set(sorted(adjusted, key=lambda d: adjusted[d], reverse=True)[:k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    from sentence_transformers import SentenceTransformer

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)
    if args.n:
        rows = rows.head(args.n)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    per_alpha_hits = {a: 0 for a in alphas}
    n_scored = 0
    n_with_query_year = 0

    with get_conn() as conn:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"year-bias sweep [{args.split}]"):
            gold_idx = gold_relevant_chunk_ids(row)
            if not gold_idx:
                continue
            stem = _filing_stem(row)
            gold = {(stem, gi) for gi in gold_idx}
            q = row["question"]
            emb = model.encode(q, normalize_embeddings=True)
            tickers = resolve_companies(q)
            query_years = extract_years(q)
            n_scored += 1
            n_with_query_year += bool(query_years)

            lists = [
                dense(conn, emb, CANDIDATE_K, LIVE_VARIANT, tickers),
                bm25(conn, q, CANDIDATE_K, LIVE_VARIANT, tickers),
            ] if tickers else [
                dense(conn, emb, CANDIDATE_K, LIVE_VARIANT),
                bm25(conn, q, CANDIDATE_K, LIVE_VARIANT),
            ]
            scores = rrf_fuse_scores(lists)

            for a in alphas:
                top = _biased_top_k(scores, query_years, a, CANDIDATE_K)
                per_alpha_hits[a] += bool(gold & top)

    print(f"\nquestions scored: {n_scored}   query mentions a year: "
          f"{n_with_query_year} ({100*n_with_query_year/n_scored:.1f}%)\n")
    # BINARY any-gold-in-pool, not coverage `recall_at_k` -- the two are not
    # interchangeable and mixing them cost 4.6pt of phantom headroom (`RETR-49`).
    print(f"{'alpha':>10}{'recall@50 (binary)':>22}{'delta vs alpha=0':>20}")
    base = per_alpha_hits.get(0.0, per_alpha_hits[alphas[0]]) / n_scored
    for a in alphas:
        r = per_alpha_hits[a] / n_scored
        print(f"{a:>10}{100*r:>13.2f}%{100*(r-base):>+19.2f}pt")


if __name__ == "__main__":
    main()
