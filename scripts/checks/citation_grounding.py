"""Standing probe: can the answer's number be traced to the evidence it was given? Free,
deterministic, replayed from STORED answers -- nothing is sent (DECISIONS.md EVAL-4).

What the stored data allows. The answer prompt (`agent._ANSWER_PROMPT`) labels each chunk
`[stem chunk N]` but never asks the model to cite, so an answer names a supporting chunk only
when it volunteers one (~9% in the Arm 6 file). `/ask`'s `citations` are every delivered chunk. So the main
check is set-level -- is the number supported by ANY delivered chunk -- and per-chunk
citation accuracy is scored only on the answers that volunteer one.

Tiers, per answer:
  grounded       the number is in the evidence at the precision the answer quotes or finer
                 (`15.2` matches `15.18`, not `15`), sign-blind, same units
  computed       not grounded, but derivable from numbers the response writes down that are
                 themselves in the evidence or the question, one written step at a time:
                 two-operand + - x / % %-change mean, a unit restatement between its own
                 numbers, or a run of 3+ consecutive stated numbers summed. Each link must be
                 written; a formula done in one breath (a - b - c + d) is not followed
  ungrounded     neither. "Not traced by this check", not "hallucinated" -- read examples
  not_scoreable  ANSWER line refused, missing, or yes/no (no digit to trace)

It checks faithfulness, not correctness: a right figure read from the wrong row is grounded.

"The number" is `answer_eval`'s: `parse_reason` decides scoreable, the first `_NUMBER` token
on `answer_line` is the value. Matching does NOT reuse `is_correct`'s 1% tolerance and seven
scale factors: against a 10-chunk block of thousands of numbers that matches almost anything,
and the `loose` line prints the proof (real and shuffled evidence score the same).

Controls; the first two exit 1 unless the control rate is under half the real one:
  evidence shuffle  each answer against another question's evidence (seeded derangement)
  answer swap       each response's traced numbers against another question's answer --
                    the chance floor of the `computed` tier
  no evidence       the question's own numbers only; printed, not asserted

Needs data/chunks/ for the Arm 6 sets (the cost13 sets carry their exact prompts), so it is
not in CI. data/chunks/ was byte-identical to Postgres `variant='A'` on 2026-09-23.

Usage:
    python scripts/checks/citation_grounding.py [--chunks-dir DIR] [--examples 5]
        [--arm6 data/day9_arm6_dev_results.jsonl] [--cost13-responses data/cost31_thinking_low.jsonl]
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rag_sec.answer_eval import (  # noqa: E402
    _NUMBER,
    SCALES,
    _close,
    answer_line,
    gold_is_scoreable,
    gold_values,
    is_correct,
    parse_reason,
)

ARM6 = Path("data/day9_arm6_dev_results_fair.jsonl")  # AGENT-30's quotable file
COST13_PAYLOAD = Path("data/day8_cost13_payload.json")
COST13_RESPONSES = Path("data/cost31_thinking_medium.jsonl")  # medium = the default (COST-34)

_EVIDENCE = re.compile(r"Evidence gathered:\n(.*)\n\nAnswer the question using only this evidence\.",
                       re.DOTALL)
_HEADER = re.compile(r"(?:^|\n\n)\[([A-Z]+_\d{4}_\d+) chunk (\d+)\] ")
# A volunteered citation: "chunk 76", optionally prefixed by its stem.
_CITE = re.compile(r"(?:([A-Z]+_\d{4}_\d+) )?chunk (\d+)", re.IGNORECASE)
# Years are operands only by accident ("2017 - 2012 = 5"), so never count one (GOLD-3 rule).
_YEAR = re.compile(r"(?:19|20)\d\d")
# `_NUMBER` needs a leading digit, so it reads 10-K dividend tables' `$.455` as 455. Evidence
# and reasoning text get the zero put back; the ANSWER line is left to `answer_eval` exactly
# (no stored ANSWER line has a leading-dot number, 0 of 2,092 checked).
_LEADING_DOT = re.compile(r"(?<![\d.])\.(?=\d)")
TIERS = ("grounded", "computed", "ungrounded", "not_scoreable")
MAX_DP = 6


# Relative slack on a COMPUTED match only: absorbs the model rounding an intermediate step
# (100,000 x 17.0203% vs 17,020.31). Arbitrary; the answer-swap control prices it.
REL_SLACK = 1e-4
# Unit restatement between the response's OWN numbers only. Applied to the whole evidence
# block instead, x1e3/x1e6 matched shuffled evidence as often as the real one (chance).
UNIT_EXPS = (3, -3, 6, -6)


def _dec(token: str) -> Decimal | None:
    """|value| of one `_NUMBER` token, exact -- floats would round 2.675 down and turn a
    real match into a miss. Sign-blind like `is_correct`: 10-K tables print negatives as
    `(1,234)`, which `_NUMBER` reads as positive."""
    t = token.replace(",", "").replace("$", "").replace("%", "").lstrip("-").rstrip(".")
    try:
        return abs(Decimal(t))
    except InvalidOperation:
        return None


def _places(v: Decimal) -> int:
    """Decimal places the value actually carries: `855,000` claims units, not thousandths."""
    return max(0, -v.normalize().as_tuple().exponent)


def _key(v: Decimal, dp: int) -> Decimal:
    return v.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)


class Evidence:
    """Every number in a block of text, indexed by the precision it can vouch for: a token
    with 2 decimals supports a figure quoted to 0, 1 or 2 decimals, never to 3."""

    def __init__(self, text: str):
        self.by_dp = [set() for _ in range(MAX_DP + 1)]
        self.floats = set()
        for tok in _NUMBER.findall(_LEADING_DOT.sub("0.", text)):
            v = _dec(tok)
            if v is None:
                continue
            self.floats.add(float(v))
            dp = max(0, -v.as_tuple().exponent)
            for p in range(min(dp, MAX_DP) + 1):
                self.by_dp[p].add(_key(v, p))

    def has(self, v: Decimal) -> bool:
        dp = _places(v)
        return dp <= MAX_DP and _key(v, dp) in self.by_dp[dp]

    def has_loose(self, x: float) -> bool:
        """`is_correct`'s notion of the same number. Printed, never used for a tier."""
        return any(_close(x, e * s) for e in self.floats for s in SCALES)


def answer_number(text: str) -> Decimal | None:
    """The scored number, or None when `answer_eval` would not score one. A yes/no answer
    parses `ok` with no digit on the line; there is nothing to ground, so it is excluded."""
    _, reason = parse_reason(text)
    if reason != "ok":
        return None
    nums = _NUMBER.findall(answer_line(text))
    return _dec(nums[0]) if nums else None


def stated(text: str) -> list[Decimal]:
    """Numbers the response's reasoning states above its ANSWER line, in the order written."""
    body = text[: text.rfind("ANSWER")] if "ANSWER" in text else text
    out = [_dec(t) for t in _NUMBER.findall(_LEADING_DOT.sub("0.", body)) if not _YEAR.fullmatch(t)]
    return [v for v in out if v]


def _steps(a: float, b: float) -> list[float]:
    out = [a + b, a - b, a * b, a * b / 100, (a + b) / 2]  # "b% of a", "mean of a and b"
    if b:
        out += [a / b, 100 * a / b, (a - b) / b, 100 * (a - b) / b]
    return out


def _derivable(x: Decimal, known: list[float], seq: list[float]) -> bool:
    """x follows from traced numbers by one written step: a two-operand op, a unit
    restatement (`$2.4 billion ($2,400 million)`), or a run of 3+ consecutive stated
    numbers summed (`306 + 44 + 60 + 100 + 246 = 756`). Tolerance is half a unit in x's
    last digit, so `15.18` accepts 15.178 and rejects 15.17."""
    xf = float(x)
    tol = max(0.5 * 10 ** -_places(x), REL_SLACK * xf) + 1e-12
    near = lambda c: abs(abs(c) - xf) <= tol  # noqa: E731
    if any(near(k * 10 ** e) for k in known for e in UNIT_EXPS):
        return True
    for i, a in enumerate(known):
        for j, b in enumerate(known):
            if i != j and any(near(c) for c in _steps(a, b)):
                return True
    ks = set(known)
    for i in range(len(seq)):
        total, n = 0.0, 0
        for v in seq[i:]:
            if v not in ks:
                break
            total, n = total + v, n + 1
            if n >= 3 and near(total):
                return True
    return False


def traced(response: str, ev: Evidence, qev: Evidence) -> tuple[list[float], list[float]]:
    """(traced numbers, stated sequence). Traced starts as the stated numbers the evidence or
    the question contains, then grows to a fixpoint by `_derivable` -- a chain is followed
    only as far as each link is itself written down."""
    seq_d = stated(response)
    seq = [float(v) for v in seq_d]
    pending = list(dict.fromkeys(seq_d))
    known = [float(v) for v in pending if ev.has(v) or qev.has(v)]
    pending = [v for v in pending if float(v) not in known]
    grew = True
    while grew and pending:
        grew = False
        for v in list(pending):
            if _derivable(v, known, seq):
                known.append(float(v))
                pending.remove(v)
                grew = True
    return known, seq


def tier(item: dict, ev: Evidence, target=None) -> str:
    target = target if target is not None else answer_number(item["response"])
    if target is None:
        return "not_scoreable"
    if ev.has(target):
        return "grounded"
    known, seq = traced(item["response"], ev, item["qev"])
    if known and _derivable(target, known, seq):
        return "computed"
    return "ungrounded"


def derangement(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != j for i, j in enumerate(p)):
            return p


# ─── loaders: one item per answer, with the exact evidence it was shown ─────────
def _chunk_text(chunks_dir: Path, cache: dict, stem: str, idx: int) -> str:
    # chunk_index is the list position in data/chunks/{stem}.json (03_index.py enumerates it)
    if stem not in cache:
        cache[stem] = [c["text"] for c in json.loads((chunks_dir / f"{stem}.json").read_text())]
    return cache[stem][idx]


def load_arm6(path: Path, chunks_dir: Path) -> dict[str, list[dict]]:
    cache: dict = {}
    loop, static = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        gold = (r["program_answer"], r["original_answer"])
        # the loop answers over every iteration's top_k, deduped (agent.answer_node)
        lk = list(dict.fromkeys((s, i) for t in r["trajectory"] for s, i, _ in t["top_k"]))
        sk = [(s, i) for s, i, _ in r["static_baseline"]["top_k"]]
        for out, keys, resp in ((loop, lk, r["final_answer"]),
                                (static, sk, r["static_baseline"]["final_answer"])):
            out.append({"id": r["id"], "question": r["question"], "response": resp, "gold": gold,
                        "chunks": {k: _chunk_text(chunks_dir, cache, *k) for k in keys}})
    return {"arm6_loop": loop, "arm6_static": static}


def load_cost13(responses: Path) -> dict[str, list[dict]]:
    qs = {q["id"]: q for q in json.loads(COST13_PAYLOAD.read_text())["questions"]}
    sets: dict[str, list[dict]] = {}
    seen = set()
    for line in responses.read_text().splitlines():
        r = json.loads(line)
        if (r["id"], r["arm"]) in seen:
            raise SystemExit(f"duplicate {(r['id'], r['arm'])} in {responses}")
        seen.add((r["id"], r["arm"]))
        q = qs[r["id"]]
        m = _EVIDENCE.search(q["prompts"][r["arm"]])
        if m is None:
            raise SystemExit(f"no evidence block in {r['id']}/{r['arm']}'s stored prompt")
        ev = m.group(1)
        heads = list(_HEADER.finditer(ev))
        chunks = {(h[1], int(h[2])): ev[h.end(): heads[j + 1].start() if j + 1 < len(heads) else None]
                  for j, h in enumerate(heads)} if heads else {("", -1): ev}
        g = q["golds"]
        sets.setdefault(f"cost13_{r['arm']}", []).append(
            {"id": r["id"], "question": q["question"], "response": r["response"],
             "gold": (g[0], g[1] if len(g) > 1 else None), "chunks": chunks})
    return sets


# ─── scoring ────────────────────────────────────────────────────────────────────
def cited_chunks(item: dict) -> list[tuple[str, int]] | None:
    """Delivered chunks the response names, or None if it names none or one is unresolvable
    (a bare "chunk 76" that matches no delivered chunk, or more than one)."""
    keys = list(item["chunks"])
    hits = []
    for stem, idx in _CITE.findall(item["response"]):
        idx = int(idx)
        m = [k for k in keys if k[1] == idx and (not stem or k[0] == stem)]
        if len(m) != 1:
            return None
        hits.append(m[0])
    return list(dict.fromkeys(hits)) or None


def score(items: list[dict], seed: int) -> dict:
    for it in items:
        it["qev"] = Evidence(it["question"])
    evs = [Evidence("\n".join(it["chunks"].values())) for it in items]
    real = [tier(it, ev) for it, ev in zip(items, evs)]
    perm = derangement(len(items), seed)
    shuf = [tier(items[i], evs[perm[i]]) for i in range(len(items))]
    no_ev = [tier(it, Evidence("")) for it in items]
    targets = [answer_number(it["response"]) for it in items]
    swap = [tier(items[i], evs[i], targets[perm[i]])
            for i in range(len(items)) if targets[i] and targets[perm[i]]]
    loose_real = [ev.has_loose(float(t)) for t, ev in zip(targets, evs) if t]
    loose_shuf = [evs[perm[i]].has_loose(float(targets[i])) for i in range(len(items)) if targets[i]]

    correct = {}
    for it, t in zip(items, real):
        if t == "not_scoreable" or not gold_is_scoreable(*it["gold"]):
            continue
        ok = is_correct(float(answer_number(it["response"])), gold_values(*it["gold"]))
        correct.setdefault(ok, Counter())[t] += 1

    cites = Counter()
    for it, t in zip(items, real):
        if t == "not_scoreable" or not _CITE.search(it["response"]):
            continue
        cites["named"] += 1
        ck = cited_chunks(it)
        if ck is None:
            continue
        cites[tier(it, Evidence("\n".join(it["chunks"][k] for k in ck)))] += 1
    return {"real": real, "shuf": shuf, "no_ev": no_ev, "swap": swap, "loose": (loose_real, loose_shuf),
            "correct": correct, "cites": cites}


def _rate(tiers: list[str], names=("grounded", "computed")) -> tuple[int, int]:
    n = sum(t != "not_scoreable" for t in tiers)
    return sum(t in names for t in tiers), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm6", type=Path, default=ARM6)
    ap.add_argument("--cost13-responses", type=Path, default=COST13_RESPONSES)
    ap.add_argument("--examples", type=int, default=0, help="print N ungrounded answers per set")
    args = ap.parse_args()
    if not args.chunks_dir.is_dir():
        raise SystemExit(f"{args.chunks_dir} missing; run 01_corpus.py or pass --chunks-dir")

    sets = {**load_arm6(args.arm6, args.chunks_dir), **load_cost13(args.cost13_responses)}
    failed = False
    for name, items in sets.items():
        s = score(items, args.seed)
        c = Counter(s["real"])
        n = len(items) - c["not_scoreable"]
        print(f"\n{name}: {len(items)} answers, {n} scoreable ({c['not_scoreable']} not)")
        for t in TIERS[:3]:
            print(f"  {t:<11} {c[t]:>4}  {c[t] / n:6.1%}")
        tr, _ = _rate(s["real"])
        sh, sn = _rate(s["shuf"])
        sw, swn = _rate(s["swap"], ("computed",))
        ok_shuf = sh / sn < 0.5 * tr / n
        ok_swap = sw / swn < 0.5 * c["computed"] / n
        failed |= not (ok_shuf and ok_swap)
        verdict = lambda ok: "ok" if ok else "FAIL: control not under half the real rate"  # noqa: E731
        print(f"  traced (grounded+computed) {tr / n:.1%}  vs evidence-shuffled {sh / sn:.1%}"
              f"  -> {verdict(ok_shuf)}")
        print(f"  computed {c['computed'] / n:.1%}  vs answer-swap chance floor {sw}/{swn} = "
              f"{sw / swn:.1%}  -> {verdict(ok_swap)}")
        ne, _ = _rate(s["no_ev"])
        print(f"  traced with no evidence at all (question's own numbers): {ne / n:.1%}")
        lr, ls = s["loose"]
        print(f"  loose (answer_eval tolerance) in-evidence: real {sum(lr) / len(lr):.1%}"
              f"  shuffled {sum(ls) / len(ls):.1%}  <- why it is not the definition")
        for ok in (True, False):
            cc = s["correct"].get(ok, Counter())
            m = sum(cc.values())
            if m:
                print(f"  {'correct' if ok else 'wrong':<8} n={m:<4} traced "
                      f"{(cc['grounded'] + cc['computed']) / m:.1%}  ungrounded {cc['ungrounded'] / m:.1%}")
        ct = s["cites"]
        m = ct["grounded"] + ct["computed"] + ct["ungrounded"]
        if ct["named"]:
            print(f"  names a chunk: {ct['named']}/{n}; resolved to delivered chunks: {m}; cited "
                  f"chunks alone grounded {ct['grounded']}, computed {ct['computed']}, "
                  f"ungrounded {ct['ungrounded']}")
        shown = 0
        for it, t in zip(items, s["real"]):
            if t == "ungrounded" and shown < args.examples:
                shown += 1
                print(f"    [{it['id']}] ANSWER: {answer_line(it['response'])}  "
                      f"gold {it['gold'][0]!r}/{it['gold'][1]!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
