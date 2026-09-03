"""Day 8, stage 1 (laptop): payload for the RETR-16 2x2 rerank job.

Two independent changes both need a cross-encoder pass over the same candidates, differing
only in the query text, so they share one GPU booking:

    candidates: unfiltered | company-filtered   (RETR-5)
    query:      raw        | entity-framing stripped   (RETR-6)

Unfiltered+raw is already scored (`data/day6_arm4_A_rerank_scores.jsonl`) and is not
re-run; the other three cells come out of this payload. The fourth cell is what makes the
result attributable -- without it a better number can't be assigned to the filter or to the
query cleanup.

The stripped query is computed HERE, on the laptop, and travels as a field. The HPC stage
stays self-contained (no `rag_sec` import, no DB) exactly as `day5_hpc_rerank.py` and
`day8_hpc_slice_rerank.py` do, so it can't be broken by a dropped connection or a missing
package on the compute node.

Chunk text is stored ONCE in a shared `texts` map keyed "stem|chunk_index", not inlined per
candidate. The filtered and unfiltered pools overlap heavily and chunks repeat across
questions, so inlining would write the same text many times over -- the older payloads did
that and paid ~240MB for one pool.

Usage:
    python scripts/arms/day8_prepare_filtered_rerank_payload.py [--n N]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.company import strip_entity_framing  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME  # noqa: E402
from rag_sec.eval import load_matched_questions  # noqa: E402
from rag_sec.retrieve import _retrieve_bm25, _retrieve_dense, _rrf_fuse  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

CANDIDATE_K = 50
OUT_PATH = Path("data/day8_retr16v2_rerank_payload.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="limit questions (smoke test)")
    # dev already has a clean unfiltered_raw baseline (day6_arm4_A_rerank_scores.jsonl),
    # so it only needs the three new cells. test has no baseline at all, so it needs the
    # full 2x2 -- see DECISIONS.md RETR-16/RETR-18.
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    df = load_matched_questions()
    wanted = [x.strip() for x in args.splits.split(",") if x.strip()]
    dev = df[df["split"].isin(wanted)].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    questions: list[dict] = []
    texts: dict[str, str] = {}
    n_resolved = n_stripped = 0

    with get_conn() as conn:
        for _, row in tqdm(dev.iterrows(), total=len(dev), desc="building RETR-16 payload"):
            q = row["question"]
            emb = model.encode(q, normalize_embeddings=True)
            tickers = resolve_companies(q)
            n_resolved += bool(tickers)

            unfiltered = _rrf_fuse(
                [
                    _retrieve_dense(conn, emb, CANDIDATE_K),
                    _retrieve_bm25(conn, q, CANDIDATE_K),
                ]
            )[:CANDIDATE_K]
            # No ticker resolved -> the filtered pool IS the unfiltered pool, which is the
            # same fallback `retrieve()` uses. Recorded rather than skipped so the filtered
            # arm is scored over all 1235 questions, not just the resolvable ones.
            filtered = (
                _rrf_fuse(
                    [
                        _retrieve_dense(conn, emb, CANDIDATE_K, tickers),
                        _retrieve_bm25(conn, q, CANDIDATE_K, tickers),
                    ]
                )[:CANDIDATE_K]
                if tickers
                else unfiltered
            )

            need = {p for p in unfiltered + filtered if f"{p[0]}|{p[1]}" not in texts}
            if need:
                stems = list({p[0] for p in need})
                rows = conn.execute(
                    "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = 'A' AND filing_stem = ANY(%s)",
                    (stems,),
                ).fetchall()
                for stem, idx, text in rows:
                    if (stem, idx) in need:
                        texts[f"{stem}|{idx}"] = text

            stripped = strip_entity_framing(q)
            n_stripped += stripped != q
            questions.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "cells": (
                        ["filtered_raw", "filtered_stripped", "unfiltered_stripped"]
                        if row["split"] == "dev"
                        else ["unfiltered_raw", "filtered_raw", "unfiltered_stripped", "filtered_stripped"]
                    ),
                    "question": q,
                    "question_stripped": stripped,
                    "tickers": tickers,
                    "cands_unfiltered": [list(p) for p in unfiltered],
                    "cands_filtered": [list(p) for p in filtered],
                }
            )

    args.out.write_text(json.dumps({"questions": questions, "texts": texts}))
    n = len(questions)
    pairs = sum(
        len(q["cands_filtered" if c.startswith("filtered") else "cands_unfiltered"])
        for q in questions
        for c in q["cells"]
    )
    from collections import Counter

    print("by split:", dict(Counter(q["split"] for q in questions)))
    print(f"\nquestions {n}   company resolved {n_resolved} ({100 * n_resolved / n:.1f}%)"
          f"   query stripped {n_stripped} ({100 * n_stripped / n:.1f}%)")
    print(f"unique chunks carried: {len(texts)}")
    n_cells = sorted({len(q["cells"]) for q in questions})
    print(f"pairs to score across {'/'.join(map(str, n_cells))} cells per question: {pairs:,}")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.0f} MB)")
    print("\nCopy via the transfer node, not the login node (DECISIONS.md ARM3-2):")
    print(f"  scp {args.out} tuteja.a@xfer.discovery.neu.edu:~/{args.out.name}")
    print("  scp scripts/arms/day8_hpc_retr16_rerank.py tuteja.a@xfer.discovery.neu.edu:~/")
    print("\nOn the GPU node (inside tmux -- srun --pty dies with the SSH session):")
    print("  srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 \\")
    print("       --mem=48G --time=08:00:00 --pty /bin/bash")
    print("  module load python/3.13.5 && source ~/rerank-env/bin/activate")
    # Derived from the payload name, not hardcoded: the dev run's scores file is already on
    # disk, and a second job writing day8_retr16_scores.jsonl would clobber it on copy-back.
    scores_name = args.out.name.replace("_payload", "_scores").replace(".json", ".jsonl")
    print(f"  python day8_hpc_retr16_rerank.py {args.out.name} {scores_name}")


if __name__ == "__main__":
    main()
