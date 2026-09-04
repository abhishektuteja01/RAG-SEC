"""Does scoping candidate generation to the question's company recover the gold?

The failure triage found 19.6% of dev questions never get the gold chunk into the 50
RRF candidates, and a 100-case qualitative pass found ~74% of retrieved chunks come from
the wrong document (half of them the right company's wrong year). This A/Bs the fix.

Measures recall@50 -- candidate generation only, no cross-encoder. That is deliberate:
the diagnosis is "the gold never reaches the reranker", so the reranker is not part of
the question and running it would cost GPU time without changing the answer.

The company is resolved from the QUESTION TEXT ONLY, against a lexicon built from the
799-filing corpus. It never reads which filing the question came from -- that would be
the inflation DATA-4 explicitly refused. Questions where no company name is found fall
back to unfiltered search rather than being dropped.

Usage:
    python scripts/analysis/company_filter_ab.py [--n N]
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
)
from rag_sec.retrieve import _rrf_fuse  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

CANDIDATE_K = 50
def _dense(conn, emb, k, tickers):
    sql = "SELECT filing_stem, chunk_index FROM chunks WHERE variant = 'A' {} ORDER BY embedding <=> %s LIMIT %s"
    where = "AND split_part(filing_stem, '_', 1) = ANY(%s)" if tickers else ""
    args = (tickers, emb, k) if tickers else (emb, k)
    return [(r[0], r[1]) for r in conn.execute(sql.format(where), args).fetchall()]


def _bm25(conn, q, k, tickers):
    if tickers:
        sql = """SELECT filing_stem, chunk_index, paradedb.score(id) AS s FROM chunks
                 WHERE id @@@ paradedb.match('text', %s) AND variant = 'A'
                   AND split_part(filing_stem, '_', 1) = ANY(%s)
                 ORDER BY s DESC LIMIT %s"""
        args = (q, tickers, k)
    else:
        sql = """SELECT filing_stem, chunk_index, paradedb.score(id) AS s FROM chunks
                 WHERE id @@@ paradedb.match('text', %s) AND variant = 'A'
                 ORDER BY s DESC LIMIT %s"""
        args = (q, k)
    return [(r[0], r[1]) for r in conn.execute(sql, args).fetchall()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    # test is a pristine holdout -- no arm, diagnostic or eval in this repo has ever
    # filtered to it (verified 2026-09-01). Kept behind a flag so it stays that way except
    # when a number is deliberately being confirmed -- DECISIONS.md RETR-18.
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    df = load_matched_questions()
    dev = df[df["split"] == args.split].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    stats = {"base_hit": 0, "filt_hit": 0, "n": 0, "resolved": 0, "recovered": 0, "lost": 0}
    with get_conn() as conn:
        for _, row in tqdm(dev.iterrows(), total=len(dev), desc=f"A/B recall@50 [{args.split}]"):
            gold_idx = gold_relevant_chunk_ids(row)
            if not gold_idx:
                continue
            stem = _filing_stem(row)
            gold = {(stem, gi) for gi in gold_idx}
            q = row["question"]
            emb = model.encode(q, normalize_embeddings=True)
            tickers = resolve_companies(q)
            stats["n"] += 1
            stats["resolved"] += bool(tickers)

            base = set(_rrf_fuse([_dense(conn, emb, CANDIDATE_K, None),
                                  _bm25(conn, q, CANDIDATE_K, None)])[:CANDIDATE_K])
            filt = set(_rrf_fuse([_dense(conn, emb, CANDIDATE_K, tickers),
                                  _bm25(conn, q, CANDIDATE_K, tickers)])[:CANDIDATE_K]) if tickers else base

            b, f = bool(gold & base), bool(gold & filt)
            stats["base_hit"] += b
            stats["filt_hit"] += f
            stats["recovered"] += (f and not b)
            stats["lost"] += (b and not f)

    n = stats["n"]
    print(f"\nquestions scored: {n}   company resolved from question text: "
          f"{stats['resolved']} ({100*stats['resolved']/n:.1f}%)\n")
    print(f"  recall@50, no filter        : {100*stats['base_hit']/n:5.1f}%")
    print(f"  recall@50, company-filtered : {100*stats['filt_hit']/n:5.1f}%"
          f"   ({100*(stats['filt_hit']-stats['base_hit'])/n:+.1f} pts)")
    print(f"\n  questions recovered (miss -> hit): {stats['recovered']}")
    print(f"  questions lost     (hit -> miss) : {stats['lost']}")


if __name__ == "__main__":
    main()
