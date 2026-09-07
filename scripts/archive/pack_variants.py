"""COST-26: packing variants scored offline against gold-figure survival.

Free -- every input is on disk (chunk scores, slice scores, parsed filings). No GPU, no API.

The metric is `COST-25`'s stricter one: are *all* of a question's gold table-row figures
literally in the prompt. Not the 84.9% matcher, which fires on a caption alone and so
flatters the compressed arm twice as hard as the control.

Cells, each differing from the one before in exactly one rule:
  base      today's `compress.pack_by_score` -- score order, no labels
  A         group packed slices by chunk, label each group, document order within it.
            Headings are charged to the budget, so the arm stays honestly 1500 tokens.
  A+fig     A, plus: if a chunk got slices packed but none carrying a distinctive figure,
            promote its best figure-bearing slice (COST-26 fix 1)
  A+cap     A, plus: a table caption is admitted only if a figure-bearing piece of its own
            table is admitted too -- otherwise the caption's tokens buy nothing (fix 2)
  A+lim     A, plus: no single chunk may take more than MAX_CHUNK_SHARE of the budget (fix 3)
  A+all     all three

Usage:
    python scripts/archive/pack_variants.py --n 300
    python scripts/archive/pack_variants.py --n 300 --cells base A A+all
"""

import argparse
import importlib.util
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from rag_sec.chunking import count_tokens
from rag_sec.compress import chunk_atoms, pack_by_score, slice_atom
from rag_sec.eval import _gold_evidence_resolved

# Reuse the validated loaders rather than copying them: duplicated retrieval/scoring code is
# exactly what let RETR-24 hide in seven files at once. By file path because
# scripts/ is not a package, so a sibling script is not importable by name.
_spec = importlib.util.spec_from_file_location("sbc", Path(__file__).with_name("stratum_b_channel.py"))
sbc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sbc)
figures, gold_figures, chunk_order, slice_scores = sbc.figures, sbc.gold_figures, sbc.chunk_order, sbc.slice_scores

BUDGET, TARGET, TOP_K = sbc.BUDGET, sbc.TARGET, sbc.TOP_K
FLAGS = sbc.FLAGS

# One chunk may take at most a fifth of the budget, so at least five of the ten retrieved
# chunks can be represented. Structural, not fitted -- but it is still a chosen constant;
# --lim-share sweeps it and any winner has to survive the untouched test split.
MAX_CHUNK_SHARE = 0.2


def build_units(qid: str, top: list[tuple[str, int]], scores: list) -> list[dict]:
    """`answer_ab_prepare.py`'s slice units, score-descending, plus the provenance the
    variants need: which atom and piece a slice is, and whether its atom is a table."""
    texts: dict = {}
    for stem, idx in top:
        try:
            chunk, atoms = chunk_atoms(stem, idx)
        except (FileNotFoundError, IndexError):
            continue
        for ai, atom in enumerate(atoms):
            for pi, piece in enumerate(slice_atom(atom, TARGET)):
                texts[(stem, idx, ai, pi)] = (piece, atom.is_table)
    rank = {c: i for i, c in enumerate(top)}
    units = []
    for stem, idx, ai, pi, score in sorted(scores, key=lambda s: s[4], reverse=True):
        if (stem, idx) not in rank or (stem, idx, ai, pi) not in texts:
            continue
        piece, is_table = texts[(stem, idx, ai, pi)]
        units.append({
            "text": piece, "tokens": count_tokens(piece), "score": score,
            "chunk": (stem, idx), "chunk_rank": rank[(stem, idx)],
            "atom": ai, "piece": pi, "is_table": is_table,
            "figs": bool(figures(piece)),
        })
    return units


# --- cells ----------------------------------------------------------------------------
# Each returns the indices it keeps, into `units` (score order). Rendering is separate so
# that survival differences come from selection, not from how the text was joined.

def cell_base(units: list[dict]) -> list[int]:
    kept, remaining = [], BUDGET
    for i, u in enumerate(units):
        if u["tokens"] <= remaining:  # skip-don't-break, as pack_by_score
            kept.append(i)
            remaining -= u["tokens"]
    return kept


def _heading(chunk: tuple[str, int]) -> str:
    return f"[{chunk[0]} chunk {chunk[1]}]"


def _head_cost(chunk: tuple[str, int]) -> int:
    return count_tokens(_heading(chunk) + "\n")


def cell_grouped(units, *, figure_guard=False, caption_bind=False, chunk_limit=False,
                 fig_top: int | None = None, cap_share: float = MAX_CHUNK_SHARE):
    """A, plus whichever COST-26 rules are switched on.

    Cell A is `compress.pack_grouped`'s rule, re-implemented here because the sweep needs
    unit indices back for the figure-survival count and needs the three guards switchable.
    Verified equivalent for the plain-A case; if either side changes, they must move together.

    Greedy by score as before; the only structural change is that admitting the first slice
    of a chunk also charges that chunk's heading, because the labels have to come out of the
    same 1500 tokens or the arm stops being the one COST-18 priced.
    """
    remaining = BUDGET
    kept: list[int] = []
    seen: set = set()
    per_chunk: dict = defaultdict(int)
    cap = int(BUDGET * cap_share) if chunk_limit else BUDGET

    def cost(i):
        u = units[i]
        return u["tokens"] + (0 if u["chunk"] in seen else _head_cost(u["chunk"]))

    def admit(i):
        nonlocal remaining
        u = units[i]
        remaining -= cost(i)
        per_chunk[u["chunk"]] += u["tokens"]
        seen.add(u["chunk"])
        kept.append(i)

    def allowed(i):
        u = units[i]
        return cost(i) <= remaining and per_chunk[u["chunk"]] + u["tokens"] <= cap

    for i, u in enumerate(units):
        if not allowed(i):
            continue
        if caption_bind and u["is_table"] and not u["figs"]:
            # A caption with no figures is only worth its tokens if a figure-bearing piece
            # of its own table comes with it. COST-26: the caption carries every question
            # word, the rows that answer it carry none, so the caption wins on score alone.
            mate = next((j for j, v in enumerate(units)
                         if v["chunk"] == u["chunk"] and v["atom"] == u["atom"] and v["figs"]), None)
            if mate is None or mate in kept:
                admit(i)
                continue
            if cost(i) + units[mate]["tokens"] > remaining or \
               per_chunk[u["chunk"]] + u["tokens"] + units[mate]["tokens"] > cap:
                continue  # drop the caption rather than spend the budget on a signpost
            admit(i)
            admit(mate)
            continue
        admit(i)

    if figure_guard:
        # A chunk that got slices packed but not one distinctive figure contributed a
        # description of evidence and no evidence. Promote its best figure-bearing slice,
        # evicting the lowest-scoring packed slices to pay for it -- the budget stays fixed.
        rank_of = {u["chunk"]: u["chunk_rank"] for u in units}
        for chunk in sorted({units[i]["chunk"] for i in kept}, key=lambda c: rank_of[c]):
            # Paying for one chunk's figures can evict another's last slice, so re-check
            # membership each pass rather than trusting the set captured at loop entry.
            here = [i for i in kept if units[i]["chunk"] == chunk]
            if not here:
                continue
            if fig_top is not None and rank_of[chunk] >= fig_top:
                continue  # COST-26 put gold in chunk rank 0-2; promoting figures out of a
                          # wrong-year chunk spends budget on a decoy
            if any(units[i]["figs"] for i in here):
                continue
            cand = next((j for j, v in enumerate(units) if v["chunk"] == chunk and v["figs"] and j not in kept), None)
            if cand is None:
                continue
            need = units[cand]["tokens"]
            if need > cap:
                continue
            while remaining < need and kept:
                drop = min(kept, key=lambda j: units[j]["score"])
                kept.remove(drop)
                remaining += units[drop]["tokens"]
                dc = units[drop]["chunk"]
                per_chunk[dc] -= units[drop]["tokens"]
                if not any(units[j]["chunk"] == dc for j in kept):
                    remaining += _head_cost(dc)  # its heading is no longer printed
                    seen.discard(dc)
            if remaining >= need:
                kept.append(cand)
                remaining -= need
                per_chunk[chunk] += need
    return kept


CELLS = {
    "base": (cell_base, "score order, unlabelled (today)"),
    "A": (lambda u: cell_grouped(u), "grouped + labelled + document order"),
    "A+fig": (lambda u: cell_grouped(u, figure_guard=True), "A + figure guard"),
    "A+cap": (lambda u: cell_grouped(u, caption_bind=True), "A + caption binding"),
    "A+lim": (lambda u: cell_grouped(u, chunk_limit=True), "A + per-chunk token cap"),
    "A+all": (lambda u: cell_grouped(u, figure_guard=True, caption_bind=True, chunk_limit=True), "A + all three"),
    # Sensitivity, not a search for a winner: check whether the negative result above is an
    # artifact of how the rules were parameterised rather than of the rules themselves.
    "A+fig3": (lambda u: cell_grouped(u, figure_guard=True, fig_top=3), "figure guard, chunk rank <3 only"),
    "A+fig1": (lambda u: cell_grouped(u, figure_guard=True, fig_top=1), "figure guard, top chunk only"),
    "A+lim33": (lambda u: cell_grouped(u, chunk_limit=True, cap_share=1 / 3), "per-chunk cap at a third"),
    "A+lim50": (lambda u: cell_grouped(u, chunk_limit=True, cap_share=0.5), "per-chunk cap at a half"),
}


def render(units: list[dict], kept: list[int], grouped: bool) -> str:
    if not grouped:
        return "\n\n".join(units[i]["text"] for i in kept)
    by_chunk: dict = defaultdict(list)
    for i in kept:
        by_chunk[units[i]["chunk"]].append(i)
    out = []
    for chunk in sorted(by_chunk, key=lambda c: min(units[i]["chunk_rank"] for i in by_chunk[c])):
        body = sorted(by_chunk[chunk], key=lambda i: (units[i]["atom"], units[i]["piece"]))
        out.append(_heading(chunk) + "\n" + "\n".join(units[i]["text"] for i in body))
    return "\n\n".join(out)


def _fragmented(units: list[dict], kept: list[int], grouped: bool) -> bool:
    """True if some table's pieces are emitted non-adjacently -- rows of one statement split
    apart by unrelated text. Grouping by chunk and re-sorting into document order is exactly
    what removes this, so it should read >0 for `base` and 0 for every grouped cell."""
    if grouped:
        order_ = []
        by_chunk: dict = defaultdict(list)
        for i in kept:
            by_chunk[units[i]["chunk"]].append(i)
        for chunk in sorted(by_chunk, key=lambda c: min(units[i]["chunk_rank"] for i in by_chunk[c])):
            order_ += sorted(by_chunk[chunk], key=lambda i: (units[i]["atom"], units[i]["piece"]))
    else:
        order_ = kept
    seen_at: dict = defaultdict(list)
    for pos, i in enumerate(order_):
        u = units[i]
        if u["is_table"]:
            seen_at[(u["chunk"], u["atom"])].append(pos)
    return any(max(v) - min(v) + 1 != len(v) for v in seen_at.values() if len(v) > 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    gold = _gold_evidence_resolved()
    flags = json.loads(FLAGS.read_text())
    order, scores = chunk_order(), slice_scores()
    pool = [q for q in flags if q in order and q in scores and gold_figures(q, gold)]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.n]

    stats: dict = {c: Counter() for c in args.cells}
    cov: dict = {c: [] for c in args.cells}
    per_q: dict = {}
    for qid in pool:
        want = gold_figures(qid, gold)
        units = build_units(qid, order[qid][:TOP_K], scores[qid])
        # Guard: this file rebuilds the units rather than importing them, so prove the
        # rebuild still reproduces the packing whose survival COST-18/COST-25 measured.
        assert [units[i]["text"] for i in cell_base(units)] == \
            pack_by_score([(u["text"], u["tokens"]) for u in units], BUDGET), qid
        per_q[qid] = {}
        for name in args.cells:
            fn, _ = CELLS[name]
            kept = fn(units)
            text = render(units, kept, grouped=name != "base")
            got = want & figures(text)
            s = stats[name]
            s["n"] += 1
            # What A is for, and it is a property of the text, not of the model: can a
            # reader tell which filing and year each number came from, and do a table's
            # rows arrive together. Both are unanswerable in `base` by construction --
            # it emits score-ordered, unlabelled slices (COST-26's worked case holds two
            # near-identical UNP income statements from different years, both anonymous).
            stems = {u["chunk"][0] for u in (units[i] for i in kept)}
            years = {}
            for st_ in stems:
                tic, yr = st_.split("_")[0], st_.split("_")[1]
                years.setdefault(tic, set()).add(yr)
            s["multi_doc"] += len(stems) > 1
            s["same_co_diff_year"] += any(len(v) > 1 for v in years.values())
            s["labelled"] += name != "base"
            s["frag"] += _fragmented(units, kept, grouped=name != "base")
            s["all"] += want <= got
            s["any"] += bool(got)
            s["slices"] += len(kept)
            s["tokens"] += count_tokens(text)
            cov[name].append(len(got) / len(want))
            per_q[qid][name] = {"all": want <= got, "slices": len(kept)}

    n = len(pool)
    print(f"\nn={n} dev questions with numeric gold table rows, budget {BUDGET}\n")
    print(f"{'cell':8}{'all figs':>10}{'any fig':>9}{'coverage':>10}{'slices':>8}{'tokens':>8}  rule")
    for name in args.cells:
        s = stats[name]
        print(f"{name:8}{s['all']/n:>9.1%}{s['any']/n:>9.1%}{st.mean(cov[name]):>10.1%}"
              f"{s['slices']/n:>8.1f}{s['tokens']/n:>8.0f}  {CELLS[name][1]}")
    print("\ninterpretability of the prompt itself -- what fix A addresses (no API call needed):")
    print(f"{'cell':8}{'multi-doc':>11}{'same co, 2+ yr':>16}{'attributed':>12}{'split tables':>14}")
    for name in args.cells:
        s = stats[name]
        print(f"{name:8}{s['multi_doc']/n:>11.1%}{s['same_co_diff_year']/n:>16.1%}"
              f"{s['labelled']/n:>12.0%}{s['frag']/n:>14.1%}")
    if "base" in args.cells:
        b = stats["base"]["all"]
        print("\ndelta vs base (points of every-gold-figure survival):")
        for name in args.cells:
            if name != "base":
                print(f"  {name:8}{100*(stats[name]['all']-b)/n:+6.1f}")
    if args.out:
        args.out.write_text(json.dumps({"n": n, "budget": BUDGET, "cells": {c: dict(stats[c]) for c in args.cells}, "per_question": per_q}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
