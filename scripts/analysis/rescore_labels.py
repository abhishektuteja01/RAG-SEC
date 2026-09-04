"""RETR-35 follow-up: re-score every ordering still on disk under the corrected labels.

Labels are used at scoring time only, so this needs no GPU and no API -- it replays orderings
already computed and re-grades them. Reports legacy and coverage side by side so the size of
the correction is attributable per arm rather than assumed uniform.

NOT covered, and it is a data-availability problem rather than a choice: Arm 1 and Arm 2's
results files persist only `top_5_retrieved` per question, so recall@10/@50 cannot be
recomputed from them. Those two need a genuine retrieval re-run against Postgres (still no
GPU) and are reported here as blocked, not silently skipped.

Usage:
    python scripts/analysis/rescore_labels.py
"""

import argparse
import json
import statistics as st
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec import eval as E  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("ab", Path(__file__).with_name("label_matcher_ab.py"))
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)
legacy = _ab.legacy_table_row_relevant_chunks

# (label, split, path, cell) -- cell=None means the file holds one already-ranked list
TARGETS = [
    ("Arm3 test baseline",      "test", "data/day8_retr18_test_scores.jsonl", "unfiltered_raw"),
    ("Arm3 test filter-only",   "test", "data/day8_retr18_test_scores.jsonl", "filtered_raw"),
    ("Arm3 test strip-only",    "test", "data/day8_retr18_test_scores.jsonl", "unfiltered_stripped"),
    ("Arm3 test filter+strip",  "test", "data/day8_retr18_test_scores.jsonl", "filtered_stripped"),
    ("Arm3 dev filter-only",    "dev",  "data/day8_retr16v2_dev_scores.jsonl", "filtered_raw"),
    ("Arm3 dev strip-only",     "dev",  "data/day8_retr16v2_dev_scores.jsonl", "unfiltered_stripped"),
    ("Arm3 dev filter+strip",   "dev",  "data/day8_retr16v2_dev_scores.jsonl", "filtered_stripped"),
    ("Arm4-A dev",              "dev",  "data/day6_arm4_A_rerank_scores.jsonl", None),
    ("Arm4-B dev",              "dev",  "data/day6_arm4_B_rerank_scores.jsonl", None),
    ("Arm4-C dev",              "dev",  "data/day6_arm4_C_rerank_scores.jsonl", None),
]


def load_order(path: str, cell: str | None) -> dict:
    order = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if cell is None:
            order[r["id"]] = [(s, i) for s, i, _v in r["reranked"]]
        elif cell in r.get("cells", {}):
            order[r["id"]] = [(s, i) for s, i, _ in sorted(r["cells"][cell], key=lambda x: -x[2])]
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/day8_rescore_labels.json"))
    args = ap.parse_args()

    df = E.load_matched_questions()
    new_fn = E._table_row_relevant_chunks
    # Gold sets depend only on the question, not the ordering, so resolve each labeling once
    # for the whole split and reuse it across every arm that scores on that split.
    gold = {}
    for name, fn in (("legacy", legacy), ("coverage", new_fn)):
        E._table_row_relevant_chunks = fn
        E._load_chunks.cache_clear()
        for split in ("dev", "test"):
            sub = df[df["split"] == split]
            g = {}
            for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"gold {name}/{split}"):
                stem = E._filing_stem(row)
                g[row["id"]] = [(stem, gi) for gi in E.gold_relevant_chunk_evidence(row)]
            gold[(name, split)] = g
    E._table_row_relevant_chunks = new_fn

    rows = []
    for label, split, path, cell in TARGETS:
        if not Path(path).exists():
            rows.append({"arm": label, "error": "missing file"})
            continue
        order = load_order(path, cell)
        rec = {"arm": label, "split": split}
        for name in ("legacy", "coverage"):
            g = gold[(name, split)]
            r10, r50, nd, mrr, n = [], [], [], [], 0
            for qid, cand in order.items():
                if qid not in g or not g[qid]:
                    continue
                n += 1
                r10.append(E.recall_at_k(cand, g[qid], 10))
                r50.append(E.recall_at_k(cand, g[qid], 50))
                nd.append(E.ndcg_at_k(cand, g[qid], 10))
                mrr.append(E.mrr(cand, g[qid]))
            rec[name] = {"n": n, "recall@10": st.mean(r10), "recall@50": st.mean(r50),
                         "nDCG@10": st.mean(nd), "MRR": st.mean(mrr)}
        rows.append(rec)

    print(f"\n{'arm':26}{'n':>6}{'recall@10':>22}{'nDCG@10':>20}")
    print(f"{'':26}{'':>6}{'legacy':>10}{'coverage':>12}{'legacy':>10}{'coverage':>10}")
    for r in rows:
        if "error" in r:
            print(f"{r['arm']:26}  {r['error']}")
            continue
        a, b = r["legacy"], r["coverage"]
        print(f"{r['arm']:26}{a['n']:>6}{a['recall@10']:>10.3f}{b['recall@10']:>12.3f}"
              f"{a['nDCG@10']:>10.3f}{b['nDCG@10']:>10.3f}")
    print("\nBLOCKED, needs a retrieval re-run (no GPU, but needs Postgres): Arm 1, Arm 2 --")
    print("their results files persist only top_5_retrieved, so recall@10/@50 is not recoverable.")
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
