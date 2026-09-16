"""Laptop leg 1/3 of the year-bias recall@10 check (needs Postgres) -- cluster route,
same shape as `05_arm3_rerank.py prepare` / `hpc/rerank_hpc.py` (ARM3-2): laptop builds a
self-contained payload, the HPC GPU node reranks with no DB/rag_sec install, laptop scores.

WHY A CLUSTER ROUTE FOR A DIAGNOSTIC: year_bias_sweep.py already showed +3.4pt recall@50
on dev at alpha=0.01 (candidate pool only, no reranker). The open question is whether the
reranker actually surfaces the recovered gold chunk into the top 10 -- that needs the
cross-encoder to run, and a laptop M3 measured ~50s/question for this (union-pool, one
reranker pass covering both conditions), so a few thousand questions is hours, not the
~20-25 min a V100 does it in (measured HPC rate from ARM3-2/RETR-18: 309,160 pairs in
~7.3 GPU-h => ~11.76 pairs/s, vs ~1.2 pairs/s on this M3's MPS).

TWO CONDITIONS, NOT THE 2x2: `filtered_stripped` (company filter + query strip) is fixed
as given -- this is a new technique layered on top of that shipped winner, one variable
(alpha) at a time, not a re-derivation of the filter/strip choice. The two conditions are
"base" (plain RRF) and "biased" (+ year_distance_bonus at ALPHA, chosen on dev by
year_bias_sweep.py).

ONE RERANK PASS PER QUESTION, NOT TWO: base_pool and biased_pool overlap heavily (same
RRF scores, just reordered by a small additive term), so `candidates` carries their UNION
and the HPC leg reranks it once. `base_pool`/`biased_pool` are membership lists into that
union, read back on the laptop `score` leg to reconstruct each condition's own top-10 --
same pattern `rag_sec.eval.load_ranking` already uses for a cell's own reranked order.

PRODUCES
    data/year_bias_recall10_{split}_payload.json

READS
    Postgres `chunks` (variant 'A'), `rag_sec.eval.load_matched_questions` for the
    question set + gold labels (gold labels only checked at `score` time, not needed here).

Usage:
    python scripts/archive/year_bias_recall10_prepare.py [--n N] [--splits dev,test] [--alpha 0.01]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.candidates import LIVE_VARIANT, bm25, chunk_texts, dense, rrf_fuse_scores  # noqa: E402
from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.company import strip_entity_framing  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, pick_device  # noqa: E402
from rag_sec.eval import load_matched_questions  # noqa: E402
from rag_sec.fiscal_year import extract_years, year_distance_bonus  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data"
CANDIDATE_K = 50
ALPHA_DEFAULT = 0.01  # picked on dev by year_bias_sweep.py: peak recall@50, +3.4pt

XFER_HOST = "<user>@xfer.discovery.neu.edu"
SRUN_RECIPE = (
    "  srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 \\\n"
    "       --mem=48G --time=02:00:00 --pty /bin/bash\n"
    "  module load python/3.13.5 && source ~/rerank-env/bin/activate"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="limit questions (smoke test)")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    df = load_matched_questions()
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    model = SentenceTransformer(EMBED_MODEL_NAME, device=pick_device())

    for split in wanted:
        rows = df[df["split"] == split].reset_index(drop=True)
        if args.n:
            rows = rows.head(args.n)

        questions: list[dict] = []
        texts: dict[str, str] = {}
        n_resolved = n_with_year = 0

        with get_conn() as conn:
            for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"preparing [{split}]"):
                q = row["question"]
                emb = model.encode(q, normalize_embeddings=True)
                tickers = resolve_companies(q)
                query_years = extract_years(q)
                n_resolved += bool(tickers)
                n_with_year += bool(query_years)

                lists = (
                    [dense(conn, emb, CANDIDATE_K, LIVE_VARIANT, tickers),
                     bm25(conn, q, CANDIDATE_K, LIVE_VARIANT, tickers)]
                    if tickers else
                    [dense(conn, emb, CANDIDATE_K, LIVE_VARIANT),
                     bm25(conn, q, CANDIDATE_K, LIVE_VARIANT)]
                )
                scores = rrf_fuse_scores(lists)
                base_pool = sorted(scores, key=lambda d: scores[d], reverse=True)[:CANDIDATE_K]
                if query_years:
                    adjusted = {p: s + year_distance_bonus(p[0], query_years, args.alpha)
                                for p, s in scores.items()}
                    biased_pool = sorted(adjusted, key=lambda d: adjusted[d], reverse=True)[:CANDIDATE_K]
                else:
                    biased_pool = base_pool

                union = list({*base_pool, *biased_pool})
                need = [p for p in union if f"{p[0]}|{p[1]}" not in texts]
                for (stem, idx), text in chunk_texts(conn, need, LIVE_VARIANT).items():
                    texts[f"{stem}|{idx}"] = text
                # Drop any candidate whose text never came back (should not happen for a
                # live-variant pair, but a membership list pointing at a missing text would
                # silently break the score leg's lookup rather than fail loudly here).
                has_text = lambda p: f"{p[0]}|{p[1]}" in texts  # noqa: E731
                union = [p for p in union if has_text(p)]

                questions.append({
                    "id": row["id"],
                    "split": split,
                    "rerank_query": strip_entity_framing(q),
                    "candidates": [list(p) for p in union],
                    "base_pool": [list(p) for p in base_pool if has_text(p)],
                    "biased_pool": [list(p) for p in biased_pool if has_text(p)],
                    "alpha": args.alpha,
                })

        n = len(questions)
        out_path = args.out_dir / f"year_bias_recall10_{split}_payload.json"
        out_path.write_text(json.dumps({"questions": questions, "texts": texts}))
        pairs = sum(len(q["candidates"]) for q in questions)
        print(f"\n[{split}] questions {n}   company resolved {n_resolved} ({100*n_resolved/n:.1f}%)"
              f"   query has a year {n_with_year} ({100*n_with_year/n:.1f}%)")
        print(f"[{split}] unique chunks carried: {len(texts)}   union pairs to rerank: {pairs:,}"
              f"   (~{pairs/11.76/3600:.2f} GPU-h at the measured V100 rate, ARM3-2/RETR-18)")
        print(f"[{split}] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\nCopy via the transfer node, not the login node (DECISIONS.md ARM3-2):")
    for split in wanted:
        print(f"  scp {args.out_dir}/year_bias_recall10_{split}_payload.json {XFER_HOST}:~/")
    print("  scp scripts/archive/year_bias_recall10_hpc.py "
          "scripts/archive/year_bias_recall10.sbatch "
          f"{XFER_HOST}:~/")
    print("\nOn the GPU node (inside tmux, or just sbatch the batch script):")
    print(SRUN_RECIPE)
    print("  python -u year_bias_recall10_hpc.py year_bias_recall10_dev_payload.json "
          "year_bias_recall10_dev_scores.jsonl")
    print("  # or: sbatch year_bias_recall10.sbatch")
    print("\nCopy results back, then score locally (no GPU, no DB):")
    for split in wanted:
        print(f"  scp {XFER_HOST}:~/year_bias_recall10_{split}_scores.jsonl data/")
    print("  uv run scripts/archive/year_bias_recall10_score.py --split dev")


if __name__ == "__main__":
    main()
