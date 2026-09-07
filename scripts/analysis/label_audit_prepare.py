"""RETR-9: build a blinded payload for auditing the relevance labels themselves.

Every retrieval number in this project is scored against *inferred* labels: T2-RAGBench
names a source row or sentence, never a chunk id in our own corpus, so `eval.py` infers
which chunk holds the evidence (GOLD-1's three layers). If that inference is wrong, a
question counted as a retrieval failure may not be one -- `RETR-33` left 142 of 160
remaining dev failures in the candidate-miss bucket, and ~15% of it looked mislabelled.

Design, and the two rules that keep the answer it produces usable:

  BLIND. Auditors are not told a case's bucket or split. Auditing only failures can only
  ever return "the score should be higher", because that is the only kind of error the
  search was pointed at. Controls drawn from questions retrieval got RIGHT catch the
  opposite error -- credit taken for a gold label that is itself wrong -- and only the two
  rates together give a net correction. COST-20 made exactly this mistake by enriching one
  stratum, so the estimate ended up resting on the stratum it had not enriched.

  TEST IS MEASURED, NEVER EDITED. Test cases are included so the mislabel *rate* is known
  out of sample, since RETR-31's headline is scored against these labels too. Hand-fixing
  an individual test label would fit the answer key to the split that certifies the
  headline. Any fix is derived as a rule on dev and applied mechanically.

Each case is one file, so an auditor reads them one at a time instead of loading a shard.
`figs_in_chunk` precomputes which gold figures literally appear in each retrieved chunk --
a deterministic signal that tells the auditor where to read closely rather than skimming
ten 900-token chunks per question.

Writes data/label_audit/cases/<case_id>.json, one manifest per shard, and a key file that
maps case ids back to questions. THE KEY IS NOT FOR THE AUDITORS.

Usage:
    python scripts/analysis/label_audit_prepare.py --shards 15
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.eval import (  # noqa: E402
    _YEAR_TOKEN_RE,
    _filing_stem,
    _gold_evidence_resolved,
    _load_chunks,
    gold_relevant_chunk_evidence,
    load_matched_questions,
    load_ranking,
)

DEV_SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")
TEST_SCORES = Path("data/day8_retr18_test_scores.jsonl")
CELL = "filtered_stripped"
OUT_DIR = Path("data/label_audit")
TOP_K = 10
SEED = 17

# Census of dev failures, plus controls large enough that the audit covers >=20% of the
# 2,781 scored questions (dev 1,235 + test 1,546).
N_DEV_CONTROL = 300
N_TEST = 200

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def figures(text: str) -> set[str]:
    """3+ digit non-year numbers -- `eval.MIN_ROW_NUMBER_DIGITS`'s rule. A 1-2 digit number
    or a year recurs in every chunk of a filing and would mark everything as a match."""
    out = set()
    for tok in _NUM.findall(text or ""):
        whole = tok.replace(",", "").split(".")[0]
        if len(whole) >= 3 and not _YEAR_TOKEN_RE.match(whole):
            out.add(tok.replace(",", ""))
    return out


def load_order(path: Path) -> dict[str, list[tuple[str, int]]]:
    return load_ranking(path, CELL)


def bucket_of(gold: set, cand: list) -> str:
    if not gold:
        return "no_gold_chunk"
    if gold & set(cand[:TOP_K]):
        return "hit@10"
    return "rerank_miss" if gold & set(cand) else "candidate_miss"


def build_case(row, cand, resolved, case_id: str) -> dict:
    stem = _filing_stem(row)
    evidence = gold_relevant_chunk_evidence(row)
    gold = {(stem, gi) for gi in evidence}
    chunks = _load_chunks(row["chunk_file"])
    r = resolved.get(row["id"]) or {}
    gold_rows = r.get("table_rows", [])
    want = set().union(*[figures(" ".join(x)) for x in gold_rows]) if gold_rows else set()

    retrieved = []
    for s, i in cand[:TOP_K]:
        text = _load_chunks(f"{s}.json")[i]["text"]
        retrieved.append({
            "stem": s, "chunk_index": i, "text": text,
            "figs_in_chunk": sorted(want & figures(text)),
        })
    return {
        "case_id": case_id,
        "question": row["question"],
        "answer": {"program": row["program_answer"], "original": row["original_answer"]},
        "filing_named_by_dataset": stem,
        "gold_table_rows": gold_rows,
        "gold_sentences": r.get("sentences", []),
        "gold_matcher_layers": sorted({e["layer"] for e in evidence.values()}),
        "gold_chunk_indices": sorted(gi for _s, gi in gold),
        "gold_chunk_texts": [chunks[gi]["text"] for _s, gi in sorted(gold)],
        "gold_distinctive_figures": sorted(want),
        "retrieved_top10": retrieved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=15)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    df = load_matched_questions()
    resolved = _gold_evidence_resolved()
    rng = random.Random(SEED)
    orders = {"dev": load_order(DEV_SCORES), "test": load_order(TEST_SCORES)}

    chosen: list[tuple] = []   # (row, split, bucket)
    by_split = {}
    for split in ("dev", "test"):
        sub = df[df["split"] == split]
        order = orders[split]
        rows = []
        for _, row in sub.iterrows():
            if row["id"] not in order:
                continue
            stem = _filing_stem(row)
            gold = {(stem, gi) for gi in gold_relevant_chunk_evidence(row)}
            rows.append((row, bucket_of(gold, order[row["id"]])))
        by_split[split] = rows

    dev_fail = [(r, "dev", b) for r, b in by_split["dev"] if b != "hit@10"]
    dev_hits = [(r, "dev", b) for r, b in by_split["dev"] if b == "hit@10"]
    chosen += dev_fail                                        # census
    chosen += rng.sample(dev_hits, min(N_DEV_CONTROL, len(dev_hits)))
    chosen += rng.sample([(r, "test", b) for r, b in by_split["test"]], N_TEST)

    rng.shuffle(chosen)   # blinding: shard membership must not encode the stratum
    cases_dir = args.out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    key, manifests = {}, [[] for _ in range(args.shards)]
    for n, (row, split, bucket) in enumerate(tqdm(chosen, desc="building cases")):
        case_id = "c" + hashlib.sha1(f"{SEED}:{row['id']}".encode()).hexdigest()[:10]
        case = build_case(row, orders[split][row["id"]], resolved, case_id)
        (cases_dir / f"{case_id}.json").write_text(json.dumps(case, indent=1))
        key[case_id] = {"id": row["id"], "split": split, "bucket": bucket}
        manifests[n % args.shards].append(case_id)

    for i, m in enumerate(manifests, 1):
        (args.out / f"shard_{i:02d}.json").write_text(json.dumps({"case_ids": m}, indent=1))
    (args.out / "KEY_do_not_give_to_auditors.json").write_text(json.dumps(key, indent=1))

    c = Counter((s, b) for _r, s, b in chosen)
    print(f"\n{len(chosen)} cases -> {args.shards} shards of ~{len(chosen)//args.shards}")
    print(f"coverage: {100*len(chosen)/(len(by_split['dev'])+len(by_split['test'])):.1f}% of scored questions")
    for k in sorted(c):
        print(f"  {k[0]:5} {k[1]:15} {c[k]:>4}")


if __name__ == "__main__":
    main()
