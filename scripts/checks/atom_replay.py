"""Verify the atom replay is exact, and measure how much choice it actually gives.

Two things must hold before compression is worth building on atoms:
  1. Replaying the packer from data/parsed/ reproduces data/chunks/ byte-for-byte -- if it
     drifts, compression would be selecting from a different document than the one that was
     embedded and retrieved.
  2. Chunks have enough atoms to choose between. A chunk that is one atom can only be kept
     whole or dropped whole, so compression has no purchase on it however good the scorer is.

Read-only, no API calls.
"""

import glob
import json
import statistics as s
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.chunking import chunk_blocks_with_atoms  # noqa: E402
from rag_sec.parsing import load_parsed_blocks  # noqa: E402


def main() -> None:
    files = sorted(glob.glob("data/parsed/*.json"))
    n_chunks = n_match = n_missing = 0
    atom_counts: list[int] = []
    atom_tokens: list[int] = []
    # tokens locked inside single-atom chunks -- the share of the corpus compression cannot touch
    locked = total = 0
    bad: list[str] = []

    for i, f in enumerate(files):
        stem = Path(f).stem
        stored_path = Path("data/chunks") / f"{stem}.json"
        if not stored_path.exists():
            n_missing += 1
            continue
        stored = json.loads(stored_path.read_text())
        packed = chunk_blocks_with_atoms(load_parsed_blocks(f))
        if len(packed) != len(stored):
            bad.append(f"{stem}: count {len(packed)} vs {len(stored)}")
            continue
        for (chunk, atoms), sc in zip(packed, stored):
            n_chunks += 1
            if chunk.text == sc["text"]:
                n_match += 1
            elif len(bad) < 5:
                bad.append(f"{stem}: text mismatch")
            atom_counts.append(len(atoms))
            atom_tokens.extend(a.tokens for a in atoms)
            total += chunk.n_tokens
            if len(atoms) <= 1:
                locked += chunk.n_tokens
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(files)} filings")

    atom_counts.sort()
    atom_tokens.sort()

    def q(v, p):
        return v[int(p * len(v))] if v else 0

    print(f"\nfilings: {len(files)} (missing chunk file: {n_missing})")
    print(f"chunks compared      : {n_chunks}")
    print(f"byte-identical replay: {n_match} ({100 * n_match / max(n_chunks, 1):.2f}%)")
    if bad:
        print("  first mismatches:", bad[:5])
    if not atom_counts or not atom_tokens:
        # data/parsed and data/chunks are gitignored, so a fresh clone lands here with
        # nothing to compare rather than with a result.
        print("no chunk was comparable -- nothing to summarize")
        return
    print(
        f"atoms/chunk : mean {s.mean(atom_counts):.1f} median {q(atom_counts, 0.5)} "
        f"p90 {q(atom_counts, 0.9)} max {max(atom_counts)}"
    )
    dist = Counter(min(c, 6) for c in atom_counts)
    for k in sorted(dist):
        lbl = f"{k}" if k < 6 else "6+"
        print(f"    {lbl:>2s} atom(s): {dist[k]:6d} ({100 * dist[k] / len(atom_counts):5.1f}%)")
    print(
        f"atom tokens : mean {s.mean(atom_tokens):.0f} median {q(atom_tokens, 0.5)} "
        f"p90 {q(atom_tokens, 0.9)} max {max(atom_tokens)}"
    )
    print(f"corpus tokens locked in single-atom chunks: {100 * locked / max(total, 1):.1f}%")


if __name__ == "__main__":
    main()
