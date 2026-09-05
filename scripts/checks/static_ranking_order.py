"""Locks the one property `agent_run.static_baseline` assumes about its input: the static
rankings it slices `[:TOP_K]` from are in DESCENDING rerank-score order at the point of use.

Nothing in the pipeline guarantees this. `rerank_hpc.py:132-135` writes each cell by zipping
scores onto the FIRST-STAGE RRF candidate order, so the score is merely attached; the
published scorer sorts on load (`rerank_score.py:62`) and `agent_run.py` did not. Measured
before the fix: 0/1235 dev cells were in score order, and Arm 6's own baseline scored
recall@10 0.552 instead of 0.739 on the 197-question sample.

Same bug class as DECISIONS.md RETR-24 -- an assumption about an on-disk artifact the
producer does not promise -- which is why it gets a check rather than a comment.

Usage:
    python scripts/checks/static_ranking_order.py
"""

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

from agent_run import STATIC_CELL, load_static_rankings  # noqa: E402


def _descending(cell) -> bool:
    scores = [c[2] for c in cell]
    return all(a >= b for a, b in zip(scores, scores[1:]))


def _synthetic() -> list[str]:
    """A file whose stored order is deliberately wrong. This is the load-bearing case: it
    fails if the sort is ever removed, with no dependency on the shipped artifact's contents.
    """
    rows = [{"id": "synth-1", "split": "dev", "latency_s": 1.0,
             "cells": {STATIC_CELL: [["A_2019_1", 0, 0.1], ["A_2019_1", 1, 0.9],
                                     ["A_2019_1", 2, 0.5]]}}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "scores.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ranked, latency = load_static_rankings(str(p))["synth-1"]
    if not _descending(ranked):
        return [f"load_static_rankings did not sort an out-of-order cell: {ranked}"]
    if [c[1] for c in ranked] != [1, 2, 0]:
        return [f"sort is not by score descending: {ranked}"]
    if latency != 1.0:
        return [f"latency_s was dropped or altered: {latency!r}"]
    return []


def main() -> int:
    failures = _synthetic()

    # The shipped artifact, if present. Skipped rather than failed when absent so the check
    # still runs in a checkout without the data files.
    path = _ROOT / "data" / "retr7_rr_dev_scores.jsonl"
    if not path.exists():
        print(f"note: {path.name} absent, real-artifact half skipped")
        n = 0
    else:
        rankings = load_static_rankings(str(path))
        n = len(rankings)
        bad = [qid for qid, (ranked, _) in rankings.items() if not _descending(ranked)]
        if bad:
            failures.append(f"{len(bad)}/{n} loaded cells not in descending score order: "
                            f"{bad[:5]}")

    if failures:
        print("static rankings are not rank-ordered at the point of use:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}\n", file=sys.stderr)
        return 1
    print(f"ok: load_static_rankings sorts by rerank score; {n} shipped cells descending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
