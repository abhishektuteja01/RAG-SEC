"""COST-25/COST-26: why did compression cost 8 points where the gold *survived*?

`COST-23` split compression's damage into stratum A (slicing removed the gold) and
stratum B (it did not). B should have been ~flat and was -8.0, which either means short
context hurts for its own reasons or means B is mislabelled. This separates the two
offline -- no API calls, no GPU -- from files already on disk.

The test is a second, stricter survival criterion. `eval.py`'s `_relevance_evidence`
(what the survival flags use) fires on a prose `gold_ind` sentence or a partial cluster
share, so a prompt holding a table's *caption* and none of its rows counts as surviving.
For FinQA the answer is arithmetic over table cells, so the criterion that actually
matters is whether the gold rows' distinctive figures -- 3+ digits, years excluded, the
same rule as `eval.MIN_ROW_NUMBER_DIGITS`/`_YEAR_TOKEN_RE` -- are literally in the prompt.
Literal, not fuzzy: an operand the reader cannot read is absent regardless of overlap.

Modes:
  --pairs      the 50 sampled stratum-B pairs: who flipped, and gold-figure coverage per arm
  --trace ID.. one question: where the gold-bearing slice ranked and what outranked it
  --survival N N dev questions: matcher survival vs figure survival, and B's mislabel rate

Usage:
    python scripts/analysis/stratum_b_channel.py --pairs
    python scripts/analysis/stratum_b_channel.py --trace finqa_dev_447
    python scripts/analysis/stratum_b_channel.py --survival 300
"""

import argparse
import json
import random
import re
import statistics as st
from collections import Counter
from pathlib import Path

from rag_sec.answer_eval import is_correct, parse_reason
from rag_sec.chunking import count_tokens
from rag_sec.compress import chunk_atoms, pack_by_score, slice_atom
from rag_sec.eval import _YEAR_TOKEN_RE, _gold_evidence_resolved, load_ranking

PAYLOAD = Path("data/day8_cost13_payload.json")
RESPONSES = Path("data/day8_cost13_responses.jsonl")
FLAGS = Path("data/day8_survival_flags.json")
CHUNK_SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")
SLICE_SCORES = Path("data/day8_slice_scores_t150_filtered_stripped.jsonl")
CELL = "filtered_stripped"
BUDGET, TARGET, TOP_K = 1500, 150, 10

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def figures(text: str) -> set[str]:
    """Distinctive figures only. A 1-2 digit number or a year recurs in every chunk of a
    filing, so counting it as coverage would make any prompt look complete (`eval.py`
    draws the same line, for the same reason)."""
    out = set()
    for tok in _NUM.findall(text):
        whole = tok.replace(",", "").split(".")[0]
        if len(whole) >= 3 and not _YEAR_TOKEN_RE.match(whole):
            out.add(tok.replace(",", ""))
    return out


def gold_figures(qid: str, gold: dict) -> set[str]:
    rows = gold.get(qid, {}).get("table_rows") or []
    return set().union(*[figures(" ".join(r)) for r in rows]) if rows else set()


def chunk_order() -> dict[str, list[tuple[str, int]]]:
    return load_ranking(CHUNK_SCORES, CELL)


def slice_scores(keep: set[str] | None = None) -> dict[str, list]:
    out = {}
    for line in open(SLICE_SCORES):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerated only for a partial tail, as in slice_budget_sweep
        if keep is None or r["id"] in keep:
            out[r["id"]] = r["scores"]
    return out


def slice_units(qid: str, top: list[tuple[str, int]], scores: list) -> list[dict]:
    """The exact units `answer_ab_prepare.py` packs, in score order, with provenance."""
    texts, full = {}, []
    for stem, idx in top:
        try:
            chunk, atoms = chunk_atoms(stem, idx)
        except (FileNotFoundError, IndexError):
            continue
        full.append(chunk.text)
        texts[(stem, idx)] = {
            (ai, pi): p for ai, a in enumerate(atoms) for pi, p in enumerate(slice_atom(a, TARGET))
        }
    rank = {c: i for i, c in enumerate(top)}
    units = []
    for stem, idx, atom_i, piece_i, score in sorted(scores, key=lambda s: s[4], reverse=True):
        if (stem, idx) not in rank:
            continue
        piece = texts.get((stem, idx), {}).get((atom_i, piece_i))
        if piece is None:
            continue
        units.append(
            {"text": piece, "tokens": count_tokens(piece), "score": score,
             "chunk": (stem, idx), "chunk_rank": rank[(stem, idx)]}
        )
    packed = set()
    remaining = BUDGET
    for i, u in enumerate(units):
        if u["tokens"] <= remaining:  # skip-don't-break, same as pack_by_score
            packed.add(i)
            remaining -= u["tokens"]
    for i, u in enumerate(units):
        u["packed"] = i in packed
    return units, "\n\n".join(full)


def _load_pairs():
    payload = json.loads(PAYLOAD.read_text())
    res = {}
    for line in open(RESPONSES):
        if line.strip():
            r = json.loads(line)
            res[(r["id"], r["arm"])] = r
    return payload, res


def cmd_pairs() -> None:
    payload, res = _load_pairs()
    gold = _gold_evidence_resolved()
    groups: dict[str, list] = {k: [] for k in ("B-LOSS", "B-GAIN", "both right", "both wrong")}
    for q in payload["questions"]:
        qid = q["id"]
        if q["stratum"] != "B_gold_kept" or any((qid, a) not in res for a in ("uncompressed", "slices_1500")):
            continue
        want = gold_figures(qid, gold)
        got = {}
        for arm in ("uncompressed", "slices_1500"):
            pred, reason = parse_reason(res[(qid, arm)]["response"])
            got[arm] = (is_correct(pred, q["golds"]), reason,
                        len(want & figures(q["prompts"][arm])), len(want))
        cu, cs = got["uncompressed"][0], got["slices_1500"][0]
        key = "B-LOSS" if cu and not cs else "B-GAIN" if cs and not cu else "both right" if cu else "both wrong"
        groups[key].append((qid, want, got))
    print(f"{'qid':18}{'goldfigs':>9}{'unc%':>7}{'slice%':>8}  slice outcome")
    for label, rows in groups.items():
        print(f"\n-- {label} (n={len(rows)})")
        cov = []
        for qid, want, got in rows:
            n, tot = got["slices_1500"][2], got["slices_1500"][3]
            frac = 100 * n / tot if tot else float("nan")
            if tot:
                cov.append(frac)
            u = 100 * got["uncompressed"][2] / tot if tot else float("nan")
            print(f"{qid:18}{tot:>9}{u:>7.0f}{frac:>8.0f}  {got['slices_1500'][1]}")
        if cov:
            print(f"   mean gold-figure coverage, slices@1500: {st.mean(cov):.0f}%  ({len(cov)}/{len(rows)} have numeric gold rows)")


def cmd_trace(ids: list[str]) -> None:
    gold = _gold_evidence_resolved()
    order, scores = chunk_order(), slice_scores(set(ids))
    for qid in ids:
        want = gold_figures(qid, gold)
        units, _ = slice_units(qid, order[qid][:TOP_K], scores[qid])
        hits = [i for i, u in enumerate(units) if want & figures(u["text"])]
        n_packed = sum(1 for u in units if u["packed"])
        print(f"\n===== {qid}  gold figures {sorted(want)}  ({len(units)} slices, {n_packed} packed)")
        for i in hits[:3]:
            u = units[i]
            print(f"  gold slice rank {i:>3} score {u['score']:+.4f} tok {u['tokens']:>3} "
                  f"chunk_rank {u['chunk_rank']} packed={u['packed']}\n      {u['text'][:130]!r}")
        if not hits:
            print("  no slice carries a gold figure")
        for i, u in enumerate(units):
            if u["packed"]:
                print(f"  packed rank {i:>3} score {u['score']:+.4f} chunk_rank {u['chunk_rank']}  {u['text'][:90]!r}")
                if i > 4:
                    break


def cmd_survival(n: int, seed: int = 7) -> None:
    gold = _gold_evidence_resolved()
    flags = json.loads(FLAGS.read_text())
    order, scores = chunk_order(), slice_scores()
    pool = [q for q in flags if q in order and q in scores and gold_figures(q, gold)]
    random.Random(seed).shuffle(pool)
    c: Counter = Counter()
    for qid in pool[:n]:
        want = gold_figures(qid, gold)
        units, full = slice_units(qid, order[qid][:TOP_K], scores[qid])
        packed = " ".join(pack_by_score([(u["text"], u["tokens"]) for u in units], BUDGET))
        in_full, in_slices = want <= figures(full), want <= figures(packed)
        c["n"] += 1
        c["flag_full"] += bool(flags[qid]["full"])
        c["flag_slices"] += bool(flags[qid]["slices_1500"])
        c["fig_full"] += in_full
        c["fig_slices"] += in_slices
        if flags[qid]["full"] and flags[qid]["slices_1500"]:
            c["B"] += 1
            c["B_all"] += in_slices
            c["B_any"] += bool(want & figures(packed))
    n = c["n"]
    print(f"n={n} dev questions with numeric gold table rows\n")
    print(f"  matcher says gold in full:          {c['flag_full']/n:6.1%}")
    print(f"  every gold figure in full:          {c['fig_full']/n:6.1%}")
    print(f"  matcher says gold in slices@1500:   {c['flag_slices']/n:6.1%}")
    print(f"  every gold figure in slices@1500:   {c['fig_slices']/n:6.1%}")
    print(f"\nstratum B (matcher: kept in both arms), n={c['B']}")
    print(f"  every gold figure present: {c['B_all']/max(c['B'],1):.1%}   at least one: {c['B_any']/max(c['B'],1):.1%}")
    print(f"  mislabelled (not one gold figure in the prompt): {c['B']-c['B_any']}/{c['B']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--trace", nargs="+", metavar="ID")
    ap.add_argument("--survival", type=int, metavar="N")
    args = ap.parse_args()
    if args.pairs:
        cmd_pairs()
    if args.trace:
        cmd_trace(args.trace)
    if args.survival:
        cmd_survival(args.survival)
    if not (args.pairs or args.trace or args.survival):
        ap.error("pick at least one of --pairs / --trace / --survival")


if __name__ == "__main__":
    main()
