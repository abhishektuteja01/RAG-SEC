"""RETR-9: aggregate the blinded label audit into rates by stratum and mechanism.

Joins the auditors' verdicts to the key they never saw. The decision-relevant number is not
the overall mislabel rate but the DIFFERENCE between the rate on questions retrieval failed
and the rate on questions it got right: only an excess in the failures would mean part of
what we call a retrieval problem is a labelling problem.

Usage:
    python scripts/archive/label_audit_score.py
"""

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

AUDIT = Path("data/label_audit")

# Auditors invented their own tags, so fold them into mechanisms. Kept as an explicit
# mapping rather than fuzzy clustering: the families are what get quoted, so they should be
# auditable, and `other:` surfaces anything the mapping fails to cover instead of hiding it.
FAMILIES = [
    (r"extra|irrelevant|distractor|unrelated-section|padding|different-table", "over-collection"),
    (r"boilerplate|wrong-topic|different-topic", "page-match-wrong-topic"),
    (r"wrong-year", "wrong-year-same-table"),
    (r"coincidental|number-format|unrelated-table-match|duplicate-table", "coincidental-figure"),
    (r"split-across|boundary", "evidence-split-across-chunks"),
    (r"caption|missing-breakdown|no-figures|incomplete", "partial-evidence-only"),
    (r"no-gold|empty-gold|no-benchmark", "benchmark-gave-no-evidence"),
    (r"not-in-filing|unrelated-to-question", "benchmark-evidence-wrong"),
    (r"prose-labeled", "wrong-modality"),
]


def family(cat: str | None) -> str:
    c = (cat or "").lower()
    if not c:
        return "uncategorised"
    for pat, name in FAMILIES:
        if re.search(pat, c):
            return name
    return "other:" + c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=AUDIT)
    args = ap.parse_args()

    key = json.loads((args.audit / "KEY_do_not_give_to_auditors.json").read_text())
    verdicts = {}
    for f in sorted((args.audit / "verdicts").glob("shard_*.json")):
        for r in json.loads(f.read_text()):
            verdicts[r["case_id"]] = r
    cases = {p.stem: json.loads(p.read_text()) for p in (args.audit / "blind").glob("*.json")}
    missing = set(key) - set(verdicts)
    print(f"{len(verdicts)}/{len(key)} cases verdicted" + (f"  MISSING {len(missing)}" if missing else ""))

    strata: dict = defaultdict(Counter)
    fams, assigned = Counter(), defaultdict(list)
    for cid, r in verdicts.items():
        k = key[cid]
        strata[f"{k['split']}/{k['bucket']}"][r["verdict"]] += 1
        assigned[r["verdict"]].append(len(cases[cid]["gold_chunk_indices"]))
        if r["verdict"] != "correct":
            fams[family(r.get("category"))] += 1

    cols = ("correct", "partially_correct", "wrong_chunk", "evidence_absent", "unjudgeable")
    print(f"\n{'stratum':22}{'n':>5}{'correct':>9}{'partial':>9}{'wrong':>7}{'absent':>8}{'unjudg':>8}{'not-correct':>13}")
    for s in sorted(strata):
        c = strata[s]
        n = sum(c.values())
        print(f"{s:22}{n:>5}" + "".join(f"{c[x]:>9}" if x in cols[:2] else f"{c[x]:>7}" if x == cols[2]
              else f"{c[x]:>8}" for x in cols) + f"{100*(n-c['correct'])/n:>12.1f}%")

    def rate(split, bucket):
        ids = [c for c in verdicts if key[c]["split"] == split and key[c]["bucket"] == bucket]
        return sum(1 for c in ids if verdicts[c]["verdict"] != "correct"), len(ids)

    bf, nf = rate("dev", "candidate_miss")
    bh, nh = rate("dev", "hit@10")
    pf, ph = bf / nf, bh / nh
    pooled = (bf + bh) / (nf + nh)
    se = (pooled * (1 - pooled) * (1 / nf + 1 / nh)) ** 0.5
    print(f"\nTHE TEST -- are failures mislabelled more often than successes?")
    print(f"  dev failures  {bf}/{nf} = {pf:.1%}")
    print(f"  dev successes {bh}/{nh} = {ph:.1%}")
    print(f"  difference {100*(pf-ph):+.1f} pt, se {100*se:.1f} pt, z = {(pf-ph)/se:+.2f}")

    print("\nmechanism families (non-correct only):")
    for k2, v in fams.most_common():
        print(f"  {k2:32}{v:>4}")

    print("\ngold chunks assigned, by verdict -- over-collection is visible here:")
    for v, lens in sorted(assigned.items(), key=lambda x: -len(x[1])):
        print(f"  {v:20} n={len(lens):>4}  mean={st.mean(lens):.2f}  median={st.median(lens):.0f}  max={max(lens)}")
    part = assigned["partially_correct"]
    print(f"\n  a partially_correct question caps at recall {st.mean([1/m for m in part]):.2f} "
          f"even with perfect retrieval, if exactly one assigned chunk is real")


if __name__ == "__main__":
    main()
