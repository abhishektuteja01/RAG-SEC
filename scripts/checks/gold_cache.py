"""Builds and guards `data/gold_chunk_ids.json`, the cached gold labels that let a clone with no
`data/chunks/` replay every score (`INFRA-28`).

The labels are CACHED, not re-derived: `build` calls `rag_sec.eval`'s own label code on the
real corpus and writes down what it returns. `eval.py` reads the cache only when the corpus is
absent; with the corpus present it computes live, as it always did, so a label experiment
(`label_matcher_ab.py`, `rescore_labels.py`) can never be answered from a stale file.

What the cache holds, and nothing more, because that is all scoring reads from the corpus:
  chunk_files   the corpus's file names -- `load_matched_questions` needs only the names, to
                map (cik, year) to a filing
  questions     per dev/test id: `gold_relevant_chunk_ids(row)`, plus a short hash of the
                row fields the labels depend on (id, chunk file, filing stem, gold page)
  fingerprint   sha256 of the label code and of its three tracked inputs. `eval.py` refuses
                the cache when any of them has changed since the build

Legs of `check`:
  static    the cache loads under today's code, its file names map to distinct filings, and
            it covers every question id in the tracked score/result files (and agrees on
            their split). Needs no corpus, no network -- CI runs this with --static-only
  recompute every label, the file-name index and every row hash recomputed live from
            data/chunks/ and compared exactly. An absent corpus FAILS this leg rather than
            skipping it, unless --static-only says so (`INFRA-27`: a skip that exits 0 hid a
            dead check for weeks)

Usage (from the repo root):
    uv run scripts/checks/gold_cache.py build                 # needs data/chunks/
    uv run scripts/checks/gold_cache.py check                 # needs data/chunks/
    uv run scripts/checks/gold_cache.py check --static-only   # CI
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec import eval as E  # noqa: E402

# Every file that names scored questions. A score file holding an id the cache lacks would
# raise mid-replay on a corpus-less clone; this finds it first.
ID_SOURCES = (
    "data/*_scores.jsonl",
    "data/*_pools.json",
    "data/day9_arm6_dev_results*.jsonl",
    "data/ci_retrieval_fixture.jsonl",
)


def _live_labels() -> dict:
    """The cache's content, computed live from data/chunks/ by eval.py's own code."""
    if not E.corpus_present():
        raise SystemExit(f"error: {E.CHUNKS_DIR}/ is absent or empty -- the labels can only be "
                         "computed from the corpus (scripts/pipeline/01_corpus.py)")
    df = E.load_matched_questions()
    questions: dict = {}
    for split in E.GOLD_CACHE_SPLITS:
        rows = df[df["split"] == split]
        questions[split] = {
            r["id"]: {"row": E.label_row_key(r), "gold": E.gold_relevant_chunk_ids(r)}
            for _, r in rows.iterrows()
        }
    return {
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
    new = {"about": ("gold_relevant_chunk_ids() per question, cached from data/chunks/ by "
                     "scripts/checks/gold_cache.py build (INFRA-28). Do not edit by hand."),
           **new}
    path = E.GOLD_CACHE_PATH
    if Path(path).exists():
        # The labels are the benchmark (GOLD-7, RETR-35). A rebuild that moves any of them
        # is a label change, not a cache refresh, and has to be asked for by name.
        old = json.loads(Path(path).read_text())
        moved = [d for d in _diff(old, new) if "gold differs" in d or "missing" in d
                 or "extra" in d]
        if moved and not args.allow_label_change:
            print("refusing: this rebuild changes the labels themselves:\n  "
                  + "\n  ".join(moved)
                  + "\nEvery published number was measured on the old ones (RETR-35). "
                  "Pass --allow-label-change only if that is the intent.", file=sys.stderr)
            return 1
    _write(new, path)
    n = sum(len(v) for v in new["questions"].values())
    print(f"wrote {path}: {n} questions, {len(new['chunk_files'])} chunk files, "
          f"{Path(path).stat().st_size / 1024:.0f} KB")
    return 0


def _static(failures: list[str]) -> dict | None:
    try:
        cache = E._gold_cache()
    except (FileNotFoundError, RuntimeError) as e:
        failures.append(str(e))
        return None
    index = E._index_filenames(cache["chunk_files"])
    if len(index) != len(cache["chunk_files"]):
        failures.append(f"chunk_files: {len(cache['chunk_files'])} names map to only "
                        f"{len(index)} distinct (cik, year) filings")
    split_of = {q: s for s, qs in cache["questions"].items() for q in qs}
    n_files = n_ids = 0
    for pattern in ID_SOURCES:
        for p in sorted(glob.glob(pattern)):
            n_files += 1
            if p.endswith(".json"):
                recs = json.loads(Path(p).read_text())["questions"]
            else:
                recs = [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]
            for r in recs:
                n_ids += 1
                if r["id"] not in split_of:
                    failures.append(f"{p}: id {r['id']!r} is not in {E.GOLD_CACHE_PATH}")
                    break
                if r.get("split") not in (None, split_of[r["id"]]):
                    failures.append(f"{p}: id {r['id']!r} is split {r['split']!r} there, "
                                    f"{split_of[r['id']]!r} in the cache")
                    break
    print(f"static: {sum(len(v) for v in cache['questions'].values())} cached questions cover "
          f"{n_ids} ids across {n_files} score/result files")
    return cache


def cmd_check(args) -> int:
    failures: list[str] = []
    cache = _static(failures)
    if args.static_only:
        print("recompute: not run (--static-only)")
    elif cache is not None:
        live = _live_labels()
        diffs = _diff(cache, live)
        if E._index_filenames(cache["chunk_files"]) != E._index_filenames(live["chunk_files"]):
            diffs.append("(cik, year) -> file index differs")
        failures += [f"recompute: {d}" for d in diffs]
        if not diffs:
            print(f"recompute: all {sum(len(v) for v in live['questions'].values())} labels, "
                  f"{len(live['chunk_files'])} file names and row hashes identical to live")
    if failures:
        print("FAIL:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--allow-label-change", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--static-only", action="store_true",
                   help="skip the live recompute (CI, which has no corpus)")
    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
