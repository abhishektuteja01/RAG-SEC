"""Locks the one property `07_arm6_loop.static_baseline` assumes about its input: the static
rankings it slices `[:TOP_K]` from are in DESCENDING rerank-score order at the point of use.

Checks `rag_sec.eval.load_ranking`, the single loader every caller now shares, rather than
importing a script by filename -- so this gate no longer breaks when scripts move.

Nothing in the pipeline guarantees this. `hpc/rerank_hpc.py`'s `main()` builds each cell in
its `cells_by_q` loop, zipping scores onto the FIRST-STAGE RRF candidate order, so the score
is merely attached (named, not line-cited: the range moved once already). The published
scorer sorts on load (`05_arm3_rerank.py score`) and `07_arm6_loop.py` did not. Measured
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

# Imported from the library, not from scripts/pipeline/07_arm6_loop.py. A CI gate must not depend
# on a pipeline script's filename or its symbol names -- moving that file used to break CI.
from rag_sec.eval import SHIPPED_CELL as STATIC_CELL  # noqa: E402
from rag_sec.eval import load_ranking  # noqa: E402

# The exact call 07_arm6_loop.py makes. Locking the flags, not a wrapper, keeps this the only
# implementation -- a wrapper here would just be a tenth copy of the thing being checked.
def _load(path: str) -> dict:
    return load_ranking(path, STATIC_CELL, with_score=True, with_latency=True)


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
        ranked, latency = _load(str(p))["synth-1"]
    if not _descending(ranked):
        return [f"load_ranking did not sort an out-of-order cell: {ranked}"]
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
        rankings = _load(str(path))
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
    print(f"ok: load_ranking sorts by rerank score; {n} shipped cells descending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
