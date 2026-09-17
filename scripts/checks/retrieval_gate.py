"""The CI quality gate's retrieval leg: fails the build when replayed scores drop below the
recorded baseline.

Scores `data/ci_retrieval_fixture.jsonl` with eval.py's OWN metric functions -- imported,
never re-implemented. That is the whole point: a second copy of the scorer would be the
project's recurring bug class (RETR-24, AGENT-16), and the gate exists partly to catch a
change to those very functions.

Thresholds:
    recall@10   may not drop more than 2.0 points below baseline
    nDCG@10     may not drop more than 2.0 points below baseline
    recall@50   same tolerance; it is the candidate pool, so it moves for different reasons
    numeric-match accuracy   may not drop AT ALL, against AGENT-29's post-fix static-arm
                             figure. Dev, n=200 -- it was never re-run on test (that needs
                             the paid `07_arm6_loop.py run`), so this leg guards the scorer,
                             not a published headline.

p95 latency and cost per query are the gate's other two legs. Neither is here: both need a
measurement in the serving container (DEPLOY-1), not a replay.

Usage:
    uv run scripts/checks/retrieval_gate.py
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.answer_eval import (  # noqa: E402
    gold_is_scoreable,
    gold_values,
    is_correct,
    parse_reason,
)
from rag_sec.eval import mean_and_stderr, ndcg_at_k, recall_at_k  # noqa: E402

FIXTURE = _ROOT / "data" / "ci_retrieval_fixture.jsonl"
BASELINE = _ROOT / "data" / "ci_retrieval_baseline.json"
TOLERANCE = 0.02  # a metric may fall 2 points below baseline before the build fails

# Answers are a different artifact and a different population from the retrieval fixture --
# 200 generated answers, not 400 stored rankings -- so they cannot share that file's baseline
# block, whose `n` the guard below asserts. Hence a constant here.
ANSWERS = _ROOT / "data" / "day9_arm6_dev_results.jsonl"
ANSWER_N = 200
# AGENT-29, as corrected 2026-09-16. Its logged 66.0% used a 200 denominator; COST-21 excludes
# the 27 unscoreable-gold questions, so the base is 173 and the replay gives 115/173 = 0.6647,
# matching what `07_arm6_loop.py analyze` prints.
ANSWER_BASELINE = 115 / 173  # 0.6647; DECISIONS.md's "66.5%" is this to one decimal. Written
# as the fraction because the leg has no tolerance, so a rounded 0.665 fails on rounding alone.


def score(rows: list[dict]) -> dict[str, float]:
    metrics = {"recall_10": [], "recall_50": [], "ndcg_10": []}
    for r in rows:
        got = [tuple(c) for c in r["ranked"]]
        rel = [tuple(g) for g in r["gold"]]
        metrics["recall_10"].append(recall_at_k(got, rel, 10))
        metrics["recall_50"].append(recall_at_k(got, rel, 50))
        metrics["ndcg_10"].append(ndcg_at_k(got, rel, 10))
    return {k: mean_and_stderr(v)[0] for k, v in metrics.items()}


def score_answers(rows: list[dict]) -> tuple[float, int]:
    """Static-arm numeric-match accuracy, scored the way 07_arm6_loop.py's paired leg scores
    its static half: answer_eval's own functions, and no yes/no mapping -- AGENT-27 keeps that
    a labelled sensitivity, not the metric. Unscoreable gold is excluded, not counted wrong
    (COST-21), so the denominator is the scoreable subset and is reported with the result."""
    hits = scored = 0
    for r in rows:
        if not gold_is_scoreable(r["program_answer"], r["original_answer"]):
            continue
        scored += 1
        pred, _ = parse_reason(r["static_baseline"]["final_answer"])
        hits += is_correct(pred, gold_values(r["program_answer"], r["original_answer"]))
    return (hits / scored if scored else 0.0), scored


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
        # A baselined metric this replay cannot compute from stored rankings. Reported as
        # ungated rather than raising KeyError.
        if k not in got:
            print(f"{k:<12} {want[k]:>9.4f} {'n/a':>9} {'n/a':>8}  not scored here")
            continue
        delta = got[k] - want[k]
        flag = ""
        if delta < -args.tolerance:
            failures.append(f"{k}: {got[k]:.4f} vs baseline {want[k]:.4f} "
                            f"({delta:+.4f}, tolerance -{args.tolerance:.2f})")
            flag = "  FAIL"
        print(f"{k:<12} {want[k]:>9.4f} {got[k]:>9.4f} {delta:>+8.4f}{flag}")

    # The answers file is tracked, so absence means a pruned checkout rather than a
    # regression; the leg drops out the way an uncomputable metric does above, and only the
    # gate's two required inputs exit non-zero when missing.
    if not ANSWERS.exists():
        print(f"\nnote: answer-accuracy leg not gated -- {ANSWERS.name} absent")
    else:
        answers = [json.loads(ln) for ln in ANSWERS.read_text().splitlines() if ln.strip()]
        answers = [r for r in answers if "error" not in r]  # as cmd_analyze drops them
        if len(answers) != ANSWER_N:
            print(f"error: {ANSWERS.name} has {len(answers)} answered questions, baseline "
                  f"was measured on {ANSWER_N}", file=sys.stderr)
            return 1
        acc, n_scored = score_answers(answers)
        delta = acc - ANSWER_BASELINE
        flag = ""
        if acc < ANSWER_BASELINE:  # may not drop at all
            failures.append(f"numeric_match: {acc:.4f} vs baseline {ANSWER_BASELINE:.4f} "
                            f"({delta:+.4f}, no tolerance)")
            flag = "  FAIL"
        print(f"{'numeric_match':<12} {ANSWER_BASELINE:>9.4f} {acc:>9.4f} {delta:>+8.4f}"
              f"{flag}   ({n_scored}/{len(answers)} scoreable, dev)")

    if failures:
        print(f"\nGATE FAILED on {len(failures)} metric(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"\nok: {len(rows)} questions, all metrics within {args.tolerance:.2f} of baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
