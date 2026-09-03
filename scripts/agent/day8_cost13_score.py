"""COST-13, stage 3 (laptop, no API): score the responses and report the paired result.

Three things, in order of what they answer:

1. **Per-stratum paired accuracy.** Stratum A is where compression removed the gold
   evidence; stratum B is where it did not. A drop confined to A is compression costing
   answers *through the mechanism we think*; a drop in B as well means shorter context hurts
   for some other reason, which the original unstratified design could not have separated.

2. **McNemar, exact.** Only discordant pairs carry information: a question both arms get
   right, or both get wrong, says nothing about the difference. Exact binomial rather than
   the chi-square approximation because discordant counts here are small (COST-20 predicts
   ~89 in A, few in B) and chi-square is unreliable below ~25.

3. **Post-stratified population estimate.** Sampling was deliberately enriched toward
   stratum A, so raw pooled accuracy would overstate the damage. Reweighting uses the exact
   stratum sizes from all 1235 survival flags -- known, not estimated. Stratum C (gold
   absent from *both* arms) was not sampled; its difference is assumed 0, which is stated as
   an assumption rather than hidden, and it is reported both ways.

Usage:
    python scripts/agent/day8_cost13_score.py
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from rag_sec.answer_eval import is_correct, parse_reason

PAYLOAD = Path("data/day8_cost13_payload.json")
RESPONSES = Path("data/day8_cost13_responses.jsonl")
ARMS = ("uncompressed", "slices_1500")


def exact_mcnemar(n10: int, n01: int) -> float:
    """Two-sided exact binomial p on the discordant pairs (H0: p = 0.5)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_delta(pairs: list[tuple[bool, bool]]) -> tuple[float, float, int, int]:
    """(delta, stderr, n10, n01) for arm2 - arm1 on paired booleans."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0, 0
    n10 = sum(1 for a, b in pairs if a and not b)   # arm1 right, arm2 wrong
    n01 = sum(1 for a, b in pairs if b and not a)   # arm2 right, arm1 wrong
    delta = (n01 - n10) / n
    var = (n01 + n10 - (n01 - n10) ** 2 / n) / n**2
    return delta, math.sqrt(max(var, 0.0)), n10, n01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=Path, default=PAYLOAD)
    ap.add_argument("--responses", type=Path, default=RESPONSES)
    args = ap.parse_args()

    payload = json.loads(args.payload.read_text())
    golds = {q["id"]: q["golds"] for q in payload["questions"]}
    strata = {q["id"]: q["stratum"] for q in payload["questions"]}
    population = payload["population"]

    # Priced from the tier recorded per response, not a constant: the first version
    # hardcoded flex rates and reported $0.76 for a $1.53 standard-tier run.
    tiers: set[str] = set()
    correct: dict[tuple[str, str], bool] = {}
    unparsed = defaultdict(int)
    usage_in = usage_out = usage_vis = 0
    for line in open(args.responses):
        if not line.strip():
            continue
        r = json.loads(line)
        pred, reason = parse_reason(r["response"])
        if reason != "ok":
            unparsed[(r["arm"], reason)] += 1
        correct[(r["id"], r["arm"])] = is_correct(pred, golds.get(r["id"], []))
        usage_in += (r.get("usage") or {}).get("prompt_tokens") or 0
        u = r.get("usage") or {}
        # billed output = visible + thinking; fall back to deriving it from total for
        # responses recorded before COST-24.
        billed = u.get("billed_output_tokens")
        if billed is None:
            billed = max((u.get("total_tokens") or 0) - ((u.get("prompt_tokens") or 0) + (u.get("output_tokens") or 0)), 0) + (u.get("output_tokens") or 0)
        usage_out += billed
        usage_vis += u.get("output_tokens") or 0
        tiers.add(r.get("tier", "unknown"))

    complete = [q for q in golds if all((q, a) in correct for a in ARMS)]
    by_stratum: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for qid in complete:
        by_stratum[strata[qid]].append((correct[(qid, ARMS[0])], correct[(qid, ARMS[1])]))

    print(f"scored {len(complete)} questions with both arms present")
    if unparsed:
        # Reported per arm, not pooled: a compliance/refusal gap between arms is itself the
        # effect (the compressed arm should refuse more often when the gold was removed),
        # and pooling it would hide that.
        print("non-numeric predictions (scored wrong), by arm and reason:")
        for (arm, reason), k in sorted(unparsed.items()):
            print(f"  {arm:16} {reason:16} {k}")
    mult = 0.5 if tiers == {"flex"} else 1.0
    tier_label = "/".join(sorted(tiers)) if tiers != {"unknown"} else "unrecorded, assuming standard"
    print(f"spend: {usage_in:,} in + {usage_out:,} billed out "
          f"({usage_vis:,} visible + {usage_out - usage_vis:,} thinking) = "
          f"${(usage_in/1e6*0.75 + usage_out/1e6*3.75) * mult:.2f} ({tier_label})\n")

    hdr = f"{'stratum':16}{'n':>5}{'uncompressed':>14}{'slices@1500':>13}{'delta':>9}{'n10':>5}{'n01':>5}{'p (exact)':>11}"
    print(hdr)
    print("-" * len(hdr))
    deltas = {}
    for s in sorted(by_stratum):
        pairs = by_stratum[s]
        a1 = sum(1 for a, _ in pairs) and sum(a for a, _ in pairs) / len(pairs)
        a2 = sum(b for _, b in pairs) / len(pairs)
        d, se, n10, n01 = paired_delta(pairs)
        deltas[s] = (d, se, len(pairs))
        print(f"{s:16}{len(pairs):>5}{a1:>13.1%}{a2:>13.1%}{d:>+9.1%}{n10:>5}{n01:>5}{exact_mcnemar(n10, n01):>11.4f}")

    # Post-stratified: stratum C unsampled, difference assumed 0.
    tot = sum(population.values())
    est = se_sq = 0.0
    for s, (d, se, _n) in deltas.items():
        w = population.get(s, 0) / tot
        est += w * d
        se_sq += (w * se) ** 2
    covered = sum(population.get(s, 0) for s in deltas) / tot
    print(f"\npopulation weights: " + ", ".join(f"{k} {v} ({100*v/tot:.1f}%)" for k, v in population.items()))
    print(f"post-stratified delta (slices@1500 - uncompressed), stratum C assumed 0:")
    print(f"  {est:+.2%}  +/- {math.sqrt(se_sq):.2%} (1 se)   [{100*covered:.1f}% of population measured]")
    ab = sum(population.get(s, 0) for s in deltas)
    est_ab = sum(population.get(s, 0) / ab * d for s, (d, _se, _n) in deltas.items())
    print(f"restricted to the measured strata only (A+B, reweighted): {est_ab:+.2%}")


if __name__ == "__main__":
    main()
