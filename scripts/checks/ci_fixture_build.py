"""Builds the committed inputs the CI retrieval gate scores against. Run rarely, by hand.

WHY A FIXTURE EXISTS AT ALL. GitHub Actions has no GPU and no Postgres holding the 99,654
embedded chunks, so CI cannot re-run retrieval. It replays: `data/retr7_rr_dev_scores.jsonl`
already holds the reranked `filtered_stripped` list for every dev question on the post-RETR-7
corpus -- the exact arm RETR-39 published. Both that file and `data/chunks/` are gitignored
(11MB and the whole corpus), so this script distills them into one small committed file
carrying only what a scorer needs: gold ids and the ranked ids.

WHAT THE GATE CAN AND CANNOT CATCH, stated here because it is the honest limit (COST-39).
Replay vets the SCORER and the LABELS: a change to eval.py's metrics, to the gold matcher, or
to the ranking-order handling fails the build. It CANNOT vet a change to retrieval itself --
a new embedder or fusion rule produces different candidates, and the fixture still holds the
old ones. That case needs a GPU run, which is a scheduled job, not a per-push gate.

DEV, NEVER TEST. The gate runs on every push; gating on the held-out split would consult the
holdout continuously and destroy the one thing making 0.747 credible.

Usage:
    uv run scripts/checks/ci_fixture_build.py            # writes fixture + baseline
    uv run scripts/checks/ci_fixture_build.py -n 400
"""

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    ndcg_at_k,
    recall_at_k,
)

SCORES = _ROOT / "data" / "retr7_rr_dev_scores.jsonl"
CELL = "filtered_stripped"  # the shipped arm; the gate protects what ships, not the ablations
FIXTURE = _ROOT / "data" / "ci_retrieval_fixture.jsonl"
BASELINE = _ROOT / "data" / "ci_retrieval_baseline.json"
# Fixed so the slice is identical on every rebuild. A slice that resampled would move the
# baseline every time it was regenerated, and the gate would be measuring the sampler.
SEED = 17


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=400,
                    help="questions in the slice; 400 keeps the committed file ~1MB")
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    if not args.scores.exists():
        print(f"error: {args.scores} absent -- it is gitignored and must be rebuilt first "
              f"(scripts/retrieval/rerank_hpc.py)", file=sys.stderr)
        return 1

    ranked = {}
    with open(args.scores) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                # Sorted on load, not read as stored. rerank_hpc.py zips scores onto the
                # FIRST-STAGE order, so 0/1235 cells are in rank order on disk -- AGENT-16,
                # and scripts/checks/static_ranking_order.py locks the same property.
                ranked[r["id"]] = [(s, i) for s, i, _ in sorted(r["cells"][CELL], key=lambda x: -x[2])]

    df = load_matched_questions()
    split = df[df["split"] == args.split].reset_index(drop=True)

    rows = []
    for _, row in tqdm(split.iterrows(), total=len(split), desc="resolving gold"):
        qid = row["id"]
        if qid not in ranked:
            continue
        stem = _filing_stem(row)
        gold = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not gold:  # unscoreable, and rerank_score.py drops these too -- stay identical
            continue
        rows.append({"id": qid, "gold": gold, "ranked": ranked[qid][:50]})

    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = sorted(rows[: args.n], key=lambda r: r["id"])  # sorted so diffs stay readable

    FIXTURE.write_text("".join(json.dumps(r) + "\n" for r in rows))

    metrics = {"recall_10": [], "recall_50": [], "ndcg_10": []}
    for r in rows:
        got = [tuple(c) for c in r["ranked"]]
        rel = [tuple(g) for g in r["gold"]]
        metrics["recall_10"].append(recall_at_k(got, rel, 10))
        metrics["recall_50"].append(recall_at_k(got, rel, 50))
        metrics["ndcg_10"].append(ndcg_at_k(got, rel, 10))

    # The baseline is measured on the fixture itself, not copied from DECISIONS.md. A 400-
    # question sample does not land on the full-split number, and hardcoding the published
    # figure would make the gate fire on the sampling difference rather than on a regression.
    baseline = {
        "split": args.split,
        "cell": CELL,
        "n": len(rows),
        "seed": SEED,
        "source": args.scores.name,
        "metrics": {k: round(mean_and_stderr(v)[0], 6) for k, v in metrics.items()},
        "stderr": {k: round(mean_and_stderr(v)[1], 6) for k, v in metrics.items()},
    }
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")

    print(f"wrote {FIXTURE.name} ({len(rows)} questions, {FIXTURE.stat().st_size/1e6:.2f} MB)")
    print(f"wrote {BASELINE.name}: {baseline['metrics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
