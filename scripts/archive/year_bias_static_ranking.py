"""Derive a `year_bias`-enabled static ranking in the shape `07_arm6_loop.py` replays.

WHY THIS EXISTS
    `07_arm6_loop.py`'s paired baseline does not re-run retrieval -- it replays
    `retr7_rr_dev_scores.jsonl:filtered_stripped`, the published RETR-39 ranking. That was
    one arm's worth of drift removed when both arms shared a retrieval config. `RETR-43`
    then made `year_bias` the DEFAULT for `retrieve()`, so the loop arm started getting a
    retrieval improvement the replayed baseline could not, and the printed "LIKE-FOR-LIKE"
    pair stopped being like for like. Measured on the same 200 dev questions: replayed
    static 0.736, static WITH year_bias 0.763, loop iteration-1 0.753 -- i.e. the loop's
    apparent retrieval win is entirely the stale baseline, and it is still a loss.

    Nothing errored, which is the point. Same class as RETR-24/AGENT-16/INFRA-22: a
    consumer assuming a provenance the producer never guaranteed -- except here the false
    assumption was created by shipping a fix.

PRODUCES
    data/retr7_rr_dev_scores_year_bias.jsonl
        {"id":.., "split":.., "cells": {"filtered_stripped_year_bias": [[stem, idx, score]]},
         "latency_s":..} -- the SAME shape `load_ranking` already dispatches on, so the
        replay path, its sort-on-load, and `checks/static_ranking_order.py` are unchanged.

READS
    data/year_bias_recall10_dev_pools.json     `biased_pool` membership + alpha
    data/year_bias_recall10_dev_scores.jsonl   per-candidate rerank scores over the union

PARITY GATE, and it is the reason this is trustworthy
    The year-bias artifacts came from a separate pipeline (its own alpha, its own
    candidate-k), so "it should match" is exactly the assumption that caused the bug this
    file exists to fix. This script therefore REFUSES TO EMIT unless the payload's
    *unbiased* leg reproduces the published `filtered_stripped` top-10 byte-for-byte on
    every question. Verified at 1235/1235 when written; anything less and the biased leg is
    not the same pipeline and must not be substituted for it.

Usage:
    uv run scripts/archive/year_bias_static_ranking.py [--split dev] [--out PATH]
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from rag_sec.eval import ALL_CELLS, SHIPPED_CELL, load_ranking  # noqa: E402

DATA_DIR = _ROOT / "data"
TOP_K = 10
# Named, not spelled inline: `07_arm6_loop.py` must import this rather than re-type it, the
# same rule STATIC_CELL already follows.
YEAR_BIAS_CELL = f"{SHIPPED_CELL}_year_bias"


def build(split: str, out_path: Path) -> int:
    payload = json.loads((DATA_DIR / f"year_bias_recall10_{split}_pools.json").read_text())
    by_id = {q["id"]: q for q in payload["questions"]}

    score_of: dict[str, dict] = {}
    latency: dict[str, float] = {}
    with open(DATA_DIR / f"year_bias_recall10_{split}_scores.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            score_of[rec["id"]] = {(s[0], s[1]): s[2] for s in rec["scores"]}
            latency[rec["id"]] = rec.get("latency_s")

    published = load_ranking(str(DATA_DIR / f"retr7_rr_{split}_scores.jsonl"), ALL_CELLS)

    # -- parity gate ------------------------------------------------------------------
    checked = mismatched = 0
    first_mismatch = None
    for qid, q in by_id.items():
        if qid not in score_of or qid not in published or SHIPPED_CELL not in published[qid]:
            continue
        so = score_of[qid]
        base_top = sorted(
            (tuple(p) for p in q["base_pool"]), key=lambda p: so.get(p, -1e9), reverse=True
        )[:TOP_K]
        checked += 1
        if base_top != published[qid][SHIPPED_CELL][:TOP_K]:
            mismatched += 1
            if first_mismatch is None:
                first_mismatch = (qid, published[qid][SHIPPED_CELL][:3], base_top[:3])
    if mismatched:
        qid, pub, base = first_mismatch
        print(
            f"REFUSING TO EMIT: the payload's unbiased leg does not reproduce "
            f"{SHIPPED_CELL} -- {mismatched} of {checked} questions differ, so the biased "
            f"leg is NOT the same pipeline as the published ranking and cannot stand in for "
            f"it.\n  first: {qid}\n    published {pub}\n    base      {base}",
            file=sys.stderr,
        )
        return 1
    print(f"parity gate: {checked}/{checked} unbiased top-{TOP_K} identical to {SHIPPED_CELL}")

    # -- emit -------------------------------------------------------------------------
    written = 0
    with open(out_path, "w") as out:
        for qid, q in by_id.items():
            if qid not in score_of:
                continue
            so = score_of[qid]
            # Every biased candidate with its rerank score attached, NOT pre-sorted: the
            # replay path sorts on load, and a file that arrives pre-sorted would hide a
            # regression in that sort (AGENT-16 is exactly that failure).
            entries = [[p[0], p[1], so[tuple(p)]] for p in map(tuple, q["biased_pool"]) if tuple(p) in so]
            out.write(json.dumps({
                "id": qid,
                "split": split,
                "cells": {YEAR_BIAS_CELL: entries},
                "latency_s": latency.get(qid),
            }) + "\n")
            written += 1
    print(f"wrote {out_path} -- {written} questions, cell {YEAR_BIAS_CELL!r}, alpha="
          f"{payload['questions'][0].get('alpha') if payload['questions'] else '?'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or DATA_DIR / f"retr7_rr_{args.split}_scores_year_bias.jsonl"
    return build(args.split, out)


if __name__ == "__main__":
    raise SystemExit(main())
