"""Day 8, stage 1 (laptop): slice every candidate chunk into token-target pieces and dump a
self-contained payload for the HPC reranker. See DECISIONS.md COST-7/COST-8.

Reuses an existing rerank payload rather than re-running retrieval, so the slice arms and
the whole-chunk baseline see a byte-identical candidate pool. That makes the comparison a
controlled one and removes any Postgres dependency from this stage.

Defaults to the RETR-29 pool (`day8_retr16v2_dev_payload.json`, `cands_filtered` +
`question_stripped`), because COST-17 showed the old `day6_arm4_A` pool is the wrong
control: it is unfiltered, variant-contaminated, and scored against raw questions. Slice
scores are query-dependent, so reusing the old ones is not an option -- they have to be
re-scored against the same query text the chunks were ranked with.

Slices come from `rag_sec.compress.slice_atom`, i.e. from the chunker's own Atoms replayed
out of data/parsed/ -- not from re-parsing the flattened chunk text (COST-5). Each
candidate's replayed text is asserted against the payload's stored text, so a drift between
data/parsed/ and the stored corpus fails loudly here instead of silently producing slices
of a document that was never retrieved.

Usage:
    python scripts/agent/day8_prepare_slice_payload.py --target 150
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.compress import chunk_atoms, slice_atom  # noqa: E402

SOURCE_PAYLOAD = Path("data/day8_retr16v2_dev_payload.json")


def _load_source(path: Path, cands_field: str, question_field: str) -> list[dict]:
    """Two payload shapes. The old Arm 3/4 file is a flat list carrying each candidate's text
    inline; the RETR-16 file separates questions from a shared `texts` map and holds several
    candidate sets per question, so the set and the query variant must both be named."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return [
            {
                "id": q["id"],
                "question": q["question"],
                "candidates": [(stem, idx, text) for stem, idx, _variant, text in q["candidates"]],
            }
            for q in raw
        ]
    texts = raw["texts"]
    out = []
    for q in raw["questions"]:
        if cands_field not in q or question_field not in q:
            raise SystemExit(f"{path} questions lack {cands_field!r}/{question_field!r}: {sorted(q)}")
        out.append(
            {
                "id": q["id"],
                "question": q[question_field],
                "candidates": [(stem, idx, texts[f"{stem}|{idx}"]) for stem, idx in q[cands_field]],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150, help="slice size in tokens")
    ap.add_argument("--n", type=int, help="limit questions (smoke test)")
    ap.add_argument("--source", type=Path, default=SOURCE_PAYLOAD)
    ap.add_argument("--cands", default="cands_filtered", help="candidate set; multi-cell payloads only")
    ap.add_argument("--question", default="question_stripped", help="query field; multi-cell payloads only")
    ap.add_argument("--tag", default="filtered_stripped", help="output filename suffix")
    args = ap.parse_args()

    out_path = Path(f"data/day8_slice_payload_t{args.target}_{args.tag}.json")
    source = _load_source(args.source, args.cands, args.question)
    if args.n:
        source = source[: args.n]

    payload, n_slices, n_cands, mismatches = [], 0, 0, 0
    for q in tqdm(source, desc=f"slicing at target={args.target}"):
        slices = []
        for stem, idx, text in q["candidates"]:
            n_cands += 1
            try:
                chunk, atoms = chunk_atoms(stem, idx)
            except (FileNotFoundError, IndexError):
                mismatches += 1
                continue
            if chunk.text != text:
                mismatches += 1
                continue
            for atom_i, atom in enumerate(atoms):
                for piece_i, piece in enumerate(slice_atom(atom, args.target)):
                    slices.append([stem, idx, atom_i, piece_i, piece])
        n_slices += len(slices)
        payload.append({"id": q["id"], "question": q["question"], "slices": slices})

    out_path.write_text(json.dumps(payload))
    print(f"\nquestions {len(payload)}  candidates {n_cands}  slices {n_slices}")
    print(f"  slices per candidate: {n_slices / max(n_cands - mismatches, 1):.1f}")
    print(f"  candidates skipped (replay mismatch): {mismatches}")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    print("\nCopy via the transfer node, not the login node (DECISIONS.md ARM3-2):")
    print(f"  scp {out_path} <user>@xfer.discovery.neu.edu:~/{out_path.name}")


if __name__ == "__main__":
    main()
