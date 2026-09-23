"""Build or check data/gold_chunk_ids.json, the cached gold labels that let a clone without
data/chunks/ score retrieval. Optional; needs data/chunks/ (scripts/rebuild/corpus.py).

The labels are CACHED, not re-derived: `build` runs rag_sec.eval's own label code on the
corpus and writes down what it returns. eval.py reads the cache only when data/chunks/ is
absent, and refuses it if the label code or its three input files changed since the build
(the `fingerprint`).

The file holds:
  chunk_files   the corpus's file names (maps (cik, year) to a filing)
  questions     per dev/test id: the gold chunk indices, plus a short hash of the row fields
                the labels depend on (id, chunk file, filing stem, gold page)
  fingerprint   sha256 of the label code and of its three tracked inputs

Usage (from the repo root):
    uv run scripts/rebuild/gold_cache.py build    # refuses if any label would move
    uv run scripts/rebuild/gold_cache.py check    # recompute and compare, exit 1 on a diff
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec import eval as E  # noqa: E402

ABOUT = ("gold_relevant_chunk_ids() per question, cached from data/chunks/ by "
         "scripts/rebuild/gold_cache.py build. Do not edit by hand.")


def _live_labels() -> dict:
    """The cache's content, computed live from data/chunks/ by eval.py's own code."""
    if not E.corpus_present():
        raise SystemExit(f"error: {E.CHUNKS_DIR}/ is absent or empty -- the labels can only be "
                         "computed from the corpus (scripts/rebuild/corpus.py)")
    df = E.load_matched_questions()
    questions: dict = {}
    for split in E.GOLD_CACHE_SPLITS:
        rows = df[df["split"] == split]
        questions[split] = {
            r["id"]: {"row": E.label_row_key(r), "gold": E.gold_relevant_chunk_ids(r)}
            for _, r in rows.iterrows()
        }
    return {
        "about": ABOUT,
        "fingerprint": E.label_fingerprint(),
        "chunk_files": sorted(f for f in os.listdir(E.CHUNKS_DIR) if f.endswith(".json")),
        "questions": questions,
    }


def _write(obj: dict, path: str) -> None:
    """One question per line, so a label change reads as a line diff in git."""
    lines = ["{",
             f' "about": {json.dumps(obj["about"])},',
             f' "fingerprint": {json.dumps(obj["fingerprint"], sort_keys=True)},',
             f' "chunk_files": {json.dumps(obj["chunk_files"])},',
             ' "questions": {']
    splits = list(obj["questions"])
    for si, split in enumerate(splits):
        lines.append(f'  {json.dumps(split)}: {{')
        items = sorted(obj["questions"][split].items())
        for qi, (qid, q) in enumerate(items):
            sep = "," if qi < len(items) - 1 else ""
            lines.append(f'   {json.dumps(qid)}: {json.dumps(q, sort_keys=True)}{sep}')
        lines.append("  }" + ("," if si < len(splits) - 1 else ""))
    lines += [" }", "}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _diff(old: dict, new: dict) -> list[str]:
    out = []
    if old.get("fingerprint") != new["fingerprint"]:
        out.append("fingerprint (label code or its inputs changed)")
    if old.get("chunk_files") != new["chunk_files"]:
        out.append("chunk_files")
    for split in sorted(set(old.get("questions", {})) | set(new["questions"])):
        o, n = old.get("questions", {}).get(split, {}), new["questions"].get(split, {})
        missing, extra = set(n) - set(o), set(o) - set(n)
        gold = [q for q in set(o) & set(n) if o[q]["gold"] != n[q]["gold"]]
        rowk = [q for q in set(o) & set(n) if o[q]["row"] != n[q]["row"]]
        for what, ids in (("missing", missing), ("extra", extra), ("gold differs", gold),
                          ("row hash differs", rowk)):
            if ids:
                out.append(f"{split}: {len(ids)} {what}, e.g. {sorted(ids)[:3]}")
    return out


def cmd_build(args) -> int:
    new = _live_labels()
    path = E.GOLD_CACHE_PATH
    if Path(path).exists():
        # The labels are the benchmark. A rebuild that moves any of them is a label change,
        # not a cache refresh, and has to be asked for by name.
        old = json.loads(Path(path).read_text())
        moved = [d for d in _diff(old, new) if "gold differs" in d or "missing" in d
                 or "extra" in d]
        if moved and not args.allow_label_change:
            print("refusing: this rebuild changes the labels themselves:\n  "
                  + "\n  ".join(moved)
                  + "\nEvery published number was measured on the old ones. "
                  "Pass --allow-label-change only if that is the intent.", file=sys.stderr)
            return 1
    _write(new, path)
    n = sum(len(v) for v in new["questions"].values())
    print(f"wrote {path}: {n} questions, {len(new['chunk_files'])} chunk files")
    return 0


def cmd_check(args) -> int:
    cache = json.loads(Path(E.GOLD_CACHE_PATH).read_text())
    diffs = _diff(cache, _live_labels())
    if diffs:
        print("FAIL:\n  " + "\n  ".join(diffs), file=sys.stderr)
        return 1
    print(f"ok: all {sum(len(v) for v in cache['questions'].values())} cached labels, "
          "file names, row hashes and the fingerprint match the live computation")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--allow-label-change", action="store_true")
    sub.add_parser("check")
    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
