"""RETR-7/RETR-8 acceptance check. Three properties, all cheap, no GPU and no API calls.

  1. REVERSIBLE. With both flags off, the packer reproduces the pre-RETR-7 implementation
     (git HEAD~ if this is the commit that introduced it) byte-for-byte. This is what makes
     the change safe to ship dark: `rag_sec.compress` replays the packer against the already
     built data/chunks/, so "off" has to mean identical, not merely similar.
  2. BOUNDARY-PRESERVING. With both flags on, every chunk still contains exactly the same
     atoms. Chunk *text* changes (that is the fix); chunk *boundaries* do not, so chunk
     indices -- which is what eval.py's gold labels are -- keep pointing at the same body.
  3. IT ACTUALLY FIXES THE BUG. Share of atoms sitting under a heading that is not the
     section they came from, before and after.
"""

import argparse
import importlib
import os
import sys
from collections import Counter
from pathlib import Path

PARSED = Path("data/parsed")


def load_packer(multi_heading: bool, strip_furniture: bool):
    os.environ["RAG_SEC_MULTI_HEADING"] = "1" if multi_heading else "0"
    os.environ["RAG_SEC_STRIP_TITLE_FURNITURE"] = "1" if strip_furniture else "0"
    for mod in [m for m in sys.modules if m.startswith("rag_sec.chunking")]:
        del sys.modules[mod]
    return importlib.import_module("rag_sec.chunking")


def true_heading_per_atom(atoms):
    """The heading an atom genuinely sits under: the last non-furniture title before it."""
    out, current = [], None
    for a in atoms:
        if a.is_title:
            if not getattr(a, "is_furniture", False):
                current = a.text
            out.append(None)
        else:
            out.append(current)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="filings to check")
    ap.add_argument("--labels", type=int, default=0,
                    help="dev questions to re-label under both settings (0 = skip)")
    ap.add_argument("--baseline", default="HEAD~1:src/rag_sec/chunking.py",
                    help="git revision:path of the pre-fix packer for the reversibility check")
    args = ap.parse_args()

    files = sorted(PARSED.glob("*.json"))[: args.n]
    if not files:
        print("no data/parsed/*.json -- nothing to check")
        return 1

    from rag_sec.parsing import load_parsed_blocks

    off = load_packer(False, False)
    on = load_packer(True, True)

    failures = 0

    # --- 1. reversibility -------------------------------------------------------------
    import subprocess, tempfile
    try:
        src = subprocess.run(["git", "show", args.baseline], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        print(f"[1] SKIP reversibility -- cannot read {args.baseline}")
        src = None
    if src:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "baseline_chunking.py"
            p.write_text(src)
            sys.path.insert(0, d)
            baseline = importlib.import_module("baseline_chunking")
            sys.path.pop(0)
        mismatch = 0
        for f in files:
            blocks = load_parsed_blocks(f)
            a = [(c.text, c.n_tokens, c.heading) for c in baseline.chunk_blocks(blocks)]
            b = [(c.text, c.n_tokens, c.heading) for c in off.chunk_blocks(blocks)]
            mismatch += a != b
        status = "PASS" if mismatch == 0 else "FAIL"
        failures += mismatch != 0
        print(f"[1] flags off == pre-fix packer, byte-for-byte: {status} "
              f"({len(files) - mismatch}/{len(files)} filings identical)")

    # --- 2. boundaries + 3. the bug itself ---------------------------------------------
    moved = 0
    mis = Counter()
    total = Counter()
    for f in files:
        blocks = load_parsed_blocks(f)
        for tag, mod in (("off", off), ("on", on)):
            atoms = mod._blocks_to_atoms(blocks)
            packed = mod._pack_atoms(atoms)
            th = true_heading_per_atom(atoms)
            idx = {id(a): i for i, a in enumerate(atoms)}
            bounds = [tuple(idx[id(a)] for a in ats) for _, ats in packed]
            if tag == "off":
                ref = bounds
            else:
                moved += bounds != ref
            for chunk, ats in packed:
                labels = set(chunk.heading.split("\n# ")) if chunk.heading else set()
                for a in ats:
                    if a.is_title:
                        continue
                    total[tag] += 1
                    t = th[idx[id(a)]]
                    if t is not None and t not in labels:
                        mis[tag] += 1

    status = "PASS" if moved == 0 else "FAIL"
    failures += moved != 0
    print(f"[2] flags on keeps every chunk boundary: {status} "
          f"({len(files) - moved}/{len(files)} filings with identical atom membership)")
    print(f"[3] atoms under a heading they did not come from: "
          f"{100 * mis['off'] / max(total['off'], 1):.1f}% before -> "
          f"{100 * mis['on'] / max(total['on'], 1):.1f}% after")

    # --- 4. gold labels are unmoved ----------------------------------------------------
    # eval.py does not store labels: it recomputes them from chunk TEXT on every run
    # (gold_relevant_chunk_evidence), returning bare chunk indices. So preserving boundaries
    # is necessary but not sufficient -- the new heading lines are new words the matcher
    # sees, and they could move a label on their own. This measures whether they do.
    if args.labels:
        import rag_sec.eval as E
        from rag_sec.dataset import load_t2_ragbench

        df = load_t2_ragbench()
        dev = df[(df["split"] == "dev") & df["company_cik"].notna() & df["report_year"].notna()]
        rows = dev.sample(n=min(args.labels, len(dev)), random_state=0)
        cache, same, changed, lost, gained = {}, 0, 0, 0, 0
        for _, row in rows.iterrows():
            stem = E._filing_stem(row)
            path = PARSED / f"{stem}.json"
            if not path.exists():
                continue
            if stem not in cache:
                blocks = load_parsed_blocks(path)
                cache[stem] = ([c.text for c in off.chunk_blocks(blocks)],
                               [c.text for c in on.chunk_blocks(blocks)])
            c_off, c_on = cache[stem]
            resolved = E._gold_evidence_resolved().get(row["id"])
            summaries = E._gold_summaries_for_row(row, stem)
            e_off = set(E._relevance_evidence(resolved, row["context"], list(enumerate(c_off)), summaries))
            e_on = set(E._relevance_evidence(resolved, row["context"], list(enumerate(c_on)), summaries))
            if not e_off and not e_on:
                continue
            if e_off == e_on:
                same += 1
            else:
                changed += 1
                lost += len(e_off - e_on)
                gained += len(e_on - e_off)
        total = same + changed
        status = "PASS" if changed == 0 else "FAIL"
        failures += changed != 0
        print(f"[4] gold labels unchanged on dev: {status} "
              f"({same}/{total} questions identical, {lost} gold chunks lost, {gained} gained)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
