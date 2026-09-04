"""Day 8: the achievable floor for evidence compression.

The gold_inds text averages 38 tokens, but compression selects whole atoms -- so the
smallest prompt a perfect scorer could produce is the size of the atom(s) the gold text
sits inside, not the gold text itself. This measures that floor directly, which sets the
budget the sweep should actually target.

Read-only, no API calls.
"""

import statistics as s

from dotenv import load_dotenv

load_dotenv()

from rag_sec.compress import chunk_atoms  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _normalize_words,
    _relevance_evidence,
    _gold_evidence_resolved,
    gold_relevant_chunk_evidence,
    load_matched_questions,
)


def _gold_in_atom(resolved: dict | None, context: str, atoms) -> list[int]:
    """Which atoms carry the gold evidence, using the same three-layer matcher the recall
    numbers use -- atoms are just smaller candidates than chunks."""
    candidates = [(i, a.text) for i, a in enumerate(atoms)]
    return sorted(_relevance_evidence(resolved, context, candidates, []).keys())


def main() -> None:
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    resolved_all = _gold_evidence_resolved()

    floors, chunk_sizes, n_hit, n_miss = [], [], 0, 0
    for _, row in dev.iterrows():
        evidence = gold_relevant_chunk_evidence(row)
        if not evidence:
            continue
        stem = row["chunk_file"].replace(".json", "")
        resolved = resolved_all.get(row["id"])
        total = 0
        found = False
        for cid in evidence:
            try:
                chunk, atoms = chunk_atoms(stem, cid)
            except (FileNotFoundError, IndexError):
                continue
            hits = _gold_in_atom(resolved, row["context"], atoms)
            if hits:
                found = True
                total += sum(atoms[i].tokens for i in hits)
                chunk_sizes.append(chunk.n_tokens)
        if found:
            n_hit += 1
            floors.append(total)
        else:
            n_miss += 1

    floors.sort()
    chunk_sizes.sort()

    def q(v, p):
        return v[int(p * len(v))] if v else 0

    print(f"dev questions with gold atoms located : {n_hit}  (gold chunk found but atom match failed: {n_miss})")
    print(
        f"gold-bearing ATOM tokens (the floor)  : mean {s.mean(floors):.0f} "
        f"median {q(floors, 0.5)} p75 {q(floors, 0.75)} p90 {q(floors, 0.9)} max {max(floors)}"
    )
    print(
        f"gold-bearing CHUNK tokens (today)     : mean {s.mean(chunk_sizes):.0f} "
        f"median {q(chunk_sizes, 0.5)} p90 {q(chunk_sizes, 0.9)}"
    )
    print(f"\nfloor as share of one chunk           : {100 * s.mean(floors) / s.mean(chunk_sizes):.0f}%")
    for b in (500, 750, 1000, 1500, 2000):
        print(f"  budget {b:5d}: questions whose gold atoms fit: {100 * sum(1 for x in floors if x <= b) / len(floors):5.1f}%")


if __name__ == "__main__":
    main()
