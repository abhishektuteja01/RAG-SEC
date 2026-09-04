"""Day 8: does slicing atoms finer make budget-limited evidence viable?

Selecting whole atoms fails because a gold-bearing atom averages 684 tokens (76% of its
chunk), so a 2k budget holds ~3 of them -- effectively recall@2, which Arm 3 measures at
0.42 against 0.61 for recall@10. This tests the alternative: slice each atom into smaller
pieces first (prose by sentence, tables by row-group with the header repeated, reusing the
chunker's own rules) and ask whether the gold then fits in a budget.

This measures the CEILING only -- whether the gold *fits*, assuming a perfect scorer. It
says nothing about whether the reranker would actually pick it. Read-only, no API calls.
"""

import re
import statistics as s

from dotenv import load_dotenv

load_dotenv()

from rag_sec.chunking import count_tokens  # noqa: E402
from rag_sec.compress import chunk_atoms, slice_atom  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _gold_evidence_resolved,
    _relevance_evidence,
    gold_relevant_chunk_evidence,
    load_matched_questions,
)

TARGETS = (150, 250, 400)
BUDGETS = (1000, 1500, 2000, 3000)


def main() -> None:
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    resolved_all = _gold_evidence_resolved()
    floors: dict[int, list[int]] = {t: [] for t in TARGETS}
    sizes: dict[int, list[int]] = {t: [] for t in TARGETS}

    for _, row in dev.iterrows():
        evidence = gold_relevant_chunk_evidence(row)
        if not evidence:
            continue
        stem = row["chunk_file"].replace(".json", "")
        resolved = resolved_all.get(row["id"])
        for target in TARGETS:
            total, found = 0, False
            for cid in evidence:
                try:
                    _, atoms = chunk_atoms(stem, cid)
                except (FileNotFoundError, IndexError):
                    continue
                pieces = [p for a in atoms for p in slice_atom(a, target)]
                sizes[target].extend(count_tokens(p) for p in pieces)
                hits = _relevance_evidence(
                    resolved, row["context"], [(i, p) for i, p in enumerate(pieces)], []
                )
                if hits:
                    found = True
                    total += sum(count_tokens(pieces[i]) for i in hits)
            if found:
                floors[target].append(total)

    print(f"{'target':>7} {'n':>6} {'gold tokens (mean/median)':>28}   " + "  ".join(f"fit@{b}" for b in BUDGETS))
    for t in TARGETS:
        f = sorted(floors[t])
        fits = "  ".join(f"{100 * sum(1 for x in f if x <= b) / len(f):5.1f}%" for b in BUDGETS)
        print(f"{t:>7} {len(f):>6} {s.mean(f):>14.0f} / {f[len(f) // 2]:<11d}   {fits}")
    print("\nbaseline (whole atoms): mean 684 / median 410 -> fit@1500 88.2%  fit@2000 94.0%")
    for t in TARGETS:
        z = sorted(sizes[t])
        print(f"  target {t}: piece tokens median {z[len(z) // 2]}, pieces per gold chunk grew accordingly")


if __name__ == "__main__":
    main()
