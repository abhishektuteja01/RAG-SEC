"""COST-13, stage 1 (laptop, no API): pick the strata and build both arms' prompts.

Design is COST-20's, not the original COST-13 sketch. McNemar's power comes from
*discordant* pairs, and at n=100 random only ~8 of 1235 questions would differ between
uncompressed and slices@1500 -- 85% survive in both arms and are identical by construction.
So sample on the variable that matters instead:

  stratum A  gold-lost   uncompressed survives, slices@1500 does not   -> ALL of them
  stratum B  gold-kept   both survive                                  -> sample N_KEPT

and post-stratify back to the population using the exact stratum sizes, which are known
from all 1235 survival flags rather than estimated. Questions whose two gold answers are
irreconcilable are dropped first (`answer_eval.gold_is_scoreable`): they would score wrong
in both arms, land in the concordant cell, and consume budget while informing nothing.

Both prompts reuse `agent._ANSWER_PROMPT` plus one output-format line. The compressed arm
now packs with `compress.pack_grouped` (COST-27): labelled, chunk-grouped, document-ordered,
so it differs from its control in *how much text* and nothing else. `--legacy-packing`
restores the score-ordered unlabelled prompt COST-13 actually sent. Writing a payload rather than calling the API keeps this stage free
and inspectable -- `--show` prints a full pair for eyeballing before any spend.

Usage:
    python scripts/compression/answer_ab_prepare.py --show 1
"""

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.agent import _ANSWER_PROMPT  # noqa: E402
from rag_sec.answer_eval import gold_is_scoreable, gold_values  # noqa: E402
from rag_sec.chunking import count_tokens  # noqa: E402
from rag_sec.compress import Slice, chunk_atoms, pack_by_score, pack_grouped, slice_atom  # noqa: E402
from rag_sec.eval import load_matched_questions  # noqa: E402

FLAGS = Path("data/day8_survival_flags.json")
SLICE_SCORES = Path("data/day8_slice_scores_t150_filtered_stripped.jsonl")
CHUNK_SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")
CELL = "filtered_stripped"
OUT = Path("data/day8_cost13_payload.json")
BUDGET = 1500
SLICE_TARGET = 150
TOP_K = 10
N_KEPT = 50
SEED = 20260902

# The one addition to agent.py's prompt, and every clause is load-bearing (all three were
# added after a 6-call wiring check, COST-23):
#   * the units clause -- gold is the table figure and tables carry "(in thousands)", so a
#     model answering in dollars was scored wrong (convfinqa_1431, gold 567048)
#   * INSUFFICIENT -- an explicit refusal token, because free-text refusals were parsed as
#     spurious numbers lifted from the reasoning ("Insufficient information" -> 2007.0)
#   * the single-line requirement -- parsing is strict, so non-compliance is measured per
#     arm rather than silently absorbed
FORMAT_LINE = (
    "\n\nGive the numeric value in the same units as the evidence -- do not expand "
    "thousands or millions. End your response with a single line:\n"
    "ANSWER: <number>\n"
    "or, if the evidence does not contain the answer:\n"
    "ANSWER: INSUFFICIENT"
)


def _load_chunk_order() -> dict[str, list[tuple[str, int]]]:
    order = {}
    for line in open(CHUNK_SCORES):
        if line.strip():
            r = json.loads(line)
            if CELL in r["cells"]:
                order[r["id"]] = [(s, i) for s, i, _sc in sorted(r["cells"][CELL], key=lambda x: -x[2])]
    return order


def _load_slice_scores() -> dict[str, list]:
    scores, targets = {}, set()
    for line in open(SLICE_SCORES):
        if line.strip():
            r = json.loads(line)
            scores[r["id"]] = r["scores"]
            targets.add(r.get("target"))
    if targets != {SLICE_TARGET}:
        raise SystemExit(f"slice scores must carry target {SLICE_TARGET}, got {targets}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kept", type=int, default=N_KEPT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--show", type=int, default=0, help="print this many full prompt pairs")
    ap.add_argument(
        "--legacy-packing",
        action="store_true",
        help="score-ordered, unlabelled slices -- reproduces COST-13's payload byte-for-byte",
    )
    args = ap.parse_args()

    flags = json.load(open(FLAGS))
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].set_index("id")
    chunk_order = _load_chunk_order()
    slice_scores = _load_slice_scores()

    def scoreable(qid: str) -> bool:
        if qid not in dev.index:
            return False
        r = dev.loc[qid]
        return gold_is_scoreable(r["program_answer"], r["original_answer"])

    lost = [q for q, v in flags.items() if v["full"] and not v["slices_1500"] and scoreable(q)]
    kept = [q for q, v in flags.items() if v["full"] and v["slices_1500"] and scoreable(q)]
    neither = [q for q, v in flags.items() if not v["full"] and scoreable(q)]

    rng = random.Random(SEED)
    chosen = [(q, "A_gold_lost") for q in sorted(lost)]
    chosen += [(q, "B_gold_kept") for q in sorted(rng.sample(sorted(kept), min(args.n_kept, len(kept))))]

    questions = []
    for qid, stratum in tqdm(chosen, desc="building prompts"):
        row = dev.loc[qid]
        question = row["question"]
        top = chunk_order[qid][:TOP_K]

        chunk_units, texts = [], {}
        for stem, idx in top:
            try:
                chunk, atoms = chunk_atoms(stem, idx)
            except (FileNotFoundError, IndexError):
                continue
            chunk_units.append((f"[{stem} chunk {idx}] {chunk.text}", chunk.n_tokens))
            texts[(stem, idx)] = {
                (ai, pi): piece
                for ai, atom in enumerate(atoms)
                for pi, piece in enumerate(slice_atom(atom, SLICE_TARGET))
            }

        top_set = set(top)
        rank_of = {c: i for i, c in enumerate(top)}
        slice_units = []
        for stem, idx, atom_i, piece_i, _score in sorted(slice_scores[qid], key=lambda s: s[4], reverse=True):
            if (stem, idx) not in top_set:
                continue
            piece = texts.get((stem, idx), {}).get((atom_i, piece_i))
            if piece is None:
                continue
            slice_units.append(Slice(piece, count_tokens(piece), stem, idx, atom_i, piece_i, rank_of[(stem, idx)]))

        uncompressed = "\n\n".join(t for t, _ in chunk_units)
        compressed = (
            "\n\n".join(pack_by_score([(u.text, u.tokens) for u in slice_units], BUDGET))
            if args.legacy_packing
            else pack_grouped(slice_units, BUDGET)
        )
        arms = {
            "uncompressed": _ANSWER_PROMPT.format(question=question, evidence=uncompressed) + FORMAT_LINE,
            f"slices_{BUDGET}": _ANSWER_PROMPT.format(question=question, evidence=compressed) + FORMAT_LINE,
        }
        questions.append(
            {
                "id": qid,
                "stratum": stratum,
                "question": question,
                "golds": gold_values(row["program_answer"], row["original_answer"]),
                "prompts": arms,
                "prompt_tokens": {k: count_tokens(v) for k, v in arms.items()},
            }
        )

    weights = {"A_gold_lost": len(lost), "B_gold_kept": len(kept), "C_no_gold_either": len(neither)}
    payload = {"budget": BUDGET, "population": weights, "questions": questions}
    args.out.write_text(json.dumps(payload))

    n_a = sum(1 for q in questions if q["stratum"] == "A_gold_lost")
    n_b = len(questions) - n_a
    tot = sum(weights.values())
    print(f"\nstratum A (gold lost)  sampled {n_a}  of population {weights['A_gold_lost']}  ({100*weights['A_gold_lost']/tot:.1f}% of scoreable)")
    print(f"stratum B (gold kept)  sampled {n_b}  of population {weights['B_gold_kept']}  ({100*weights['B_gold_kept']/tot:.1f}%)")
    print(f"stratum C (no gold in either arm, NOT sampled): {weights['C_no_gold_either']}  ({100*weights['C_no_gold_either']/tot:.1f}%)")
    for arm in ("uncompressed", f"slices_{BUDGET}"):
        t = sum(q["prompt_tokens"][arm] for q in questions)
        print(f"  {arm:16} {t:>9,} prompt tokens  (mean {t/len(questions):,.0f})")
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")

    for q in questions[: args.show]:
        print("\n" + "=" * 78)
        print(f"{q['id']}  [{q['stratum']}]  golds={q['golds']}")
        for arm, text in q["prompts"].items():
            print(f"\n--- {arm} ({q['prompt_tokens'][arm]} tok) ---\n{text}")


if __name__ == "__main__":
    main()
