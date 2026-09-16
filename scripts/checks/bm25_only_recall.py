"""Standing probe for research.md sec3a: BM25 alone (no company filter, no fusion, no
reranker) recovers most of the task for free -- recall@10 0.497 vs the shipped hybrid's
0.514. That gap says a chunk of "difficulty" is lexical match, not retrieval quality.

This is a floor, not a target: it should stay roughly where it was measured. A large move
means the corpus, split, or labels changed under the eval, which is worth knowing about on
its own (same shape as the RETR-24/RETR-30 on-disk-shape bugs) -- not something a one-off
number would have caught.

No GPU, no paid API -- BM25 is Postgres full-text (`03_index.py bm25` must have run once).

Usage:
    python scripts/checks/bm25_only_recall.py [--split test] [--tolerance 0.03]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.candidates import LIVE_VARIANT, bm25  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    recall_at_k,
)
from rag_sec.store import get_conn  # noqa: E402

# research.md sec3a, full test split n=1545, measured 2026-09-07.
EXPECTED = {"dev": None, "test": 0.497}
CANDIDATE_K = 50


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--tolerance", type=float, default=0.03)
    args = ap.parse_args()

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)

    recalls = []
    skipped_gold = 0
    with get_conn() as conn:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc="bm25-only"):
            stem = _filing_stem(row)
            rel = [(stem, i) for i in gold_relevant_chunk_ids(row)]
            if not rel:
                skipped_gold += 1
                continue
            got = bm25(conn, row["question"], CANDIDATE_K, LIVE_VARIANT)
            recalls.append(recall_at_k(got, rel, 10))

    mean, stderr = mean_and_stderr(recalls)
    print(f"BM25-only recall@10 ({args.split}, n={len(recalls)}, "
          f"skipped {skipped_gold} no-gold): {mean:.4f} ± {stderr:.4f}")

    expected = EXPECTED.get(args.split)
    if expected is None:
        print(f"no recorded baseline for split={args.split!r} -- reporting only")
        return 0
    delta = mean - expected
    if abs(delta) > args.tolerance:
        print(f"FAIL: {mean:.4f} is {delta:+.4f} from the recorded {expected:.4f} "
              f"(tolerance {args.tolerance})")
        return 1
    print(f"OK: within {args.tolerance} of the recorded {expected:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
