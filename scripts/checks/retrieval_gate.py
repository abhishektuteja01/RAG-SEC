"""The CI quality gate's retrieval leg: fails the build when replayed scores drop below the
recorded baseline (spec.md 2.4).

Scores `data/ci_retrieval_fixture.jsonl` with eval.py's OWN metric functions -- imported,
never re-implemented. That is the whole point: a second copy of the scorer would be the
project's recurring bug class (RETR-24, AGENT-16), and the gate exists partly to catch a
change to those very functions.

Thresholds, from spec.md 2.4:
    recall@10   may not drop more than 2.0 points below baseline
    nDCG@10     may not drop more than 2.0 points below baseline
    recall@50   same tolerance; it is the candidate pool, so it moves for different reasons
    numeric-match accuracy   may not drop AT ALL -- skipped until a baseline exists

Latency and cost per query are the other two legs spec.md 2.4 names. Neither is gated here:
both need a measurement in the serving container (DEPLOY-1), not a replay.

Usage:
    uv run scripts/checks/retrieval_gate.py
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.eval import mean_and_stderr, ndcg_at_k, recall_at_k  # noqa: E402

FIXTURE = _ROOT / "data" / "ci_retrieval_fixture.jsonl"
BASELINE = _ROOT / "data" / "ci_retrieval_baseline.json"
TOLERANCE = 0.02  # 2 points, spec.md 2.4


def score(rows: list[dict]) -> dict[str, float]:
    metrics = {"recall_10": [], "recall_50": [], "ndcg_10": []}
    for r in rows:
        got = [tuple(c) for c in r["ranked"]]
        rel = [tuple(g) for g in r["gold"]]
        metrics["recall_10"].append(recall_at_k(got, rel, 10))
        metrics["recall_50"].append(recall_at_k(got, rel, 50))
        metrics["ndcg_10"].append(ndcg_at_k(got, rel, 10))
    return {k: mean_and_stderr(v)[0] for k, v in metrics.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    for p in (args.fixture, args.baseline):
        if not p.exists():
            print(f"error: {p} missing -- build it with "
                  f"scripts/checks/ci_fixture_build.py", file=sys.stderr)
            return 1

    rows = [json.loads(ln) for ln in args.fixture.read_text().splitlines() if ln.strip()]
    baseline = json.loads(args.baseline.read_text())

    # A fixture that lost or gained questions makes every comparison meaningless, and it is
    # the likeliest way a "passing" gate goes quietly wrong.
    if len(rows) != baseline["n"]:
        print(f"error: fixture has {len(rows)} questions, baseline was measured on "
              f"{baseline['n']} -- rebuild both together", file=sys.stderr)
        return 1

    got = score(rows)
    want = baseline["metrics"]

    failures = []
    print(f"{'metric':<12} {'baseline':>9} {'now':>9} {'delta':>8}")
    for k in sorted(want):
        delta = got[k] - want[k]
        flag = ""
        if delta < -args.tolerance:
            failures.append(f"{k}: {got[k]:.4f} vs baseline {want[k]:.4f} "
                            f"({delta:+.4f}, tolerance -{args.tolerance:.2f})")
            flag = "  FAIL"
        print(f"{k:<12} {want[k]:>9.4f} {got[k]:>9.4f} {delta:>+8.4f}{flag}")

    if "numeric_match" not in want:
        print("\nnote: answer-accuracy leg not gated -- no baseline yet (pending Day 9)")

    if failures:
        print(f"\nGATE FAILED on {len(failures)} metric(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"\nok: {len(rows)} questions, all metrics within {args.tolerance:.2f} of baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
