"""Real per-query paired variance for the arm-to-arm recall@10 deltas we publish, replacing
the assumed-correlation power bound in research.md sec7 item 3 with a measured number.

Reads only already-stored per-query scores -- no retrieval, no GPU, no spend:
  - data/day3_arm1_dev_results.json, data/day4_arm2_dev_results.json (per-question recall_10,
    post-RETR-7 rerun per their 2026-09-04 mtimes)
  - data/retr7_rr_{dev,test}_scores.jsonl (Arm 3's 2x2 rerank cells)

Three sequential deltas, each a paired same-query comparison on the same corpus/labels:
  1. Arm 1 -> Arm 2               (dense -> +BM25/RRF, dev)
  2. Arm 2 -> Arm 3 unfiltered    (+reranker, dev, unfiltered_raw cell)
  3. Arm 3 unfiltered -> Arm 3 filtered+stripped   (+company filter +query strip, dev and test)

Arm1/Arm2 vs Arm3 join on question id; the join is inner and any mismatch is reported, not
silently dropped (INFRA-22 shape).

Usage:
    python scripts/archive/paired_delta_variance.py
"""

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    load_ranking,
    recall_at_k,
)

DATA_DIR = _ROOT / "data"


def paired_stats(a: list[float], b: list[float]) -> tuple[float, float, tuple[float, float]]:
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    return mean, se, ci


def report(name: str, a: list[float], b: list[float]) -> None:
    mean, se, (lo, hi) = paired_stats(a, b)
    print(f"{name:<45} n={len(a):<6} delta={mean:+.4f}  stderr={se:.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]")


def arm3_cell_recalls(scores_path: Path, split: str) -> dict[str, dict[str, float]]:
    """qid -> {cell: recall_10}, scored against the same gold labels cmd_score uses."""
    from rag_sec.eval import ALL_CELLS

    ranked = load_ranking(scores_path, ALL_CELLS)
    df = load_matched_questions()
    rows = df[df["split"] == split].reset_index(drop=True)

    out: dict[str, dict[str, float]] = {}
    for _, row in rows.iterrows():
        qid = row["id"]
        cells = ("unfiltered_raw", "filtered_stripped")
        if qid not in ranked or any(c not in ranked[qid] for c in cells):
            continue
        stem = _filing_stem(row)
        rel = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not rel:
            continue
        out[qid] = {c: recall_at_k(ranked[qid][c], rel, 10) for c in cells}
    return out


def main() -> None:
    arm1 = {r["id"]: r["recall_10"] for r in json.loads(
        (DATA_DIR / "day3_arm1_dev_results.json").read_text())["per_question"]}
    arm2 = {r["id"]: r["recall_10"] for r in json.loads(
        (DATA_DIR / "day4_arm2_dev_results.json").read_text())["per_question"]}

    common_12 = sorted(set(arm1) & set(arm2))
    if len(common_12) != len(arm1) or len(common_12) != len(arm2):
        print(f"WARNING: arm1 n={len(arm1)}, arm2 n={len(arm2)}, "
              f"joined n={len(common_12)} -- id mismatch, not silently ignored")
    report("Arm 1 -> Arm 2 (dev)",
           [arm1[q] for q in common_12], [arm2[q] for q in common_12])

    arm3_dev = arm3_cell_recalls(DATA_DIR / "retr7_rr_dev_scores.jsonl", "dev")
    common_23 = sorted(set(arm2) & set(arm3_dev))
    if len(common_23) != len(arm3_dev):
        print(f"WARNING: arm2 n={len(arm2)}, arm3 dev n={len(arm3_dev)}, "
              f"joined n={len(common_23)} -- id mismatch, not silently ignored")
    report("Arm 2 -> Arm 3 unfiltered (dev)",
           [arm2[q] for q in common_23],
           [arm3_dev[q]["unfiltered_raw"] for q in common_23])

    qids_dev = sorted(arm3_dev)
    report("Arm 3 unfiltered -> +filter+strip (dev)",
           [arm3_dev[q]["unfiltered_raw"] for q in qids_dev],
           [arm3_dev[q]["filtered_stripped"] for q in qids_dev])

    arm3_test = arm3_cell_recalls(DATA_DIR / "retr7_rr_test_scores.jsonl", "test")
    qids_test = sorted(arm3_test)
    report("Arm 3 unfiltered -> +filter+strip (test)",
           [arm3_test[q]["unfiltered_raw"] for q in qids_test],
           [arm3_test[q]["filtered_stripped"] for q in qids_test])


if __name__ == "__main__":
    main()
