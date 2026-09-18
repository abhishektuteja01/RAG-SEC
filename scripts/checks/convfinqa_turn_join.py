"""Fails when `data/convfinqa_turns.jsonl` stops lining up with the questions we score.

The artifact is built by a POSITIONAL join: `convfinqa_N` is index N into the original
ConvFinQA `train.json + dev.json`. Nobody upstream promised that order, which makes it the
project's recurring bug class (DECISIONS.md RETR-24 / AGENT-16 / INFRA-22) written down on
purpose. `08_convfinqa_turns.py` verifies it at build time; this re-verifies it afterwards,
so a stale artifact fails a run instead of being trusted.

It also locks the thing that would break quietly: the artifact's ids must be exactly the
ConvFinQA ids `load_matched_questions()` yields. Adding turn files to `dataset.py`'s
`SUBSET_FILES` would grow that set past the ids in the stored score files and silently
shrink the scored denominator -- no error, just a different number. That fires here.

Three legs, each skipped rather than failed when its input is absent:
  artifact  the file exists and is well formed                 (needs the file)
  join      it still agrees with upstream, positionally        (needs data/convfinqa_upstream/)
  ids       its id set equals load_matched_questions()'s       (needs data/chunks/)
The join leg carries a negative control: upstream rotated by one row must fail it.

Usage:
    python scripts/checks/convfinqa_turn_join.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "pipeline"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "convfinqa_turns", _ROOT / "scripts" / "pipeline" / "08_convfinqa_turns.py"
)
_producer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_producer)

REQUIRED_FIELDS = (
    "id", "source_id", "split", "file_name", "n_turns",
    "t2_question", "turns", "turn_answers", "turn_programs", "qa_split", "shares_turn0_gold",
)


def _load_artifact() -> list[dict] | None:
    if not _producer.OUT_PATH.exists():
        return None
    with open(_producer.OUT_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    failures: list[str] = []
    skipped: list[str] = []

    rows = _load_artifact()
    if rows is None:
        print(f"skip: {_producer.OUT_PATH.name} absent "
              "(build it with scripts/pipeline/08_convfinqa_turns.py build)")
        return 0

    # --- artifact leg ------------------------------------------------------------------
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_FIELDS if k not in r]
        if missing:
            failures.append(f"row {i} ({r.get('id')}) is missing {missing}")
            break
        n = r["n_turns"]
        if not (len(r["turns"]) == len(r["turn_answers"]) == len(r["turn_programs"])
                == len(r["qa_split"]) == len(r["shares_turn0_gold"]) == n):
            failures.append(f"row {i} ({r['id']}) has per-turn lists disagreeing with n_turns={n}")
            break
    if len(set(r["id"] for r in rows)) != len(rows):
        failures.append(f"{len(rows) - len(set(r['id'] for r in rows))} duplicate ids")

    # --- join leg ----------------------------------------------------------------------
    have_upstream = all((_producer.UPSTREAM_DIR / n).exists() for n in _producer.UPSTREAM_FILES)
    if not have_upstream:
        skipped.append("join (data/convfinqa_upstream/ absent; no network fetch from a check)")
    else:
        upstream = _producer.fetch_upstream()
        t2 = _load_t2()
        if t2 is None:
            skipped.append("join (T2-RAGBench not reachable and not cached)")
        else:
            join_failures, stats = _producer.check_join(t2, upstream)
            failures.extend(join_failures)
            if not join_failures:
                # Negative control: the check has to be able to see the bug it guards.
                rotated = upstream[1:] + upstream[:1]
                if not _producer.check_join(t2, rotated)[0]:
                    failures.append(
                        "negative control did not fire: upstream rotated by one row still "
                        "passes the join, so this check proves nothing"
                    )
                else:
                    print(f"ok: positional join holds on all {stats['rows']} rows; "
                          f"content key {stats['key_unique']} unique / "
                          f"{stats['key_ambiguous']} ambiguous / {stats['key_missing']} missing")
                    print("    negative control: upstream rotated by one row fails, as it must")
                # The stored artifact must be what that join produces, not an older build.
                fresh = _producer.build_rows(t2, upstream)
                if [r["id"] for r in fresh] != [r["id"] for r in rows]:
                    failures.append("the artifact's id order is not what the producer emits now")
                else:
                    drifted = [f["id"] for f, r in zip(fresh, rows)
                               if f["turns"] != r["turns"] or f["split"] != r["split"]]
                    if drifted:
                        failures.append(
                            f"{len(drifted)} rows differ from a fresh build "
                            f"(turns or split), e.g. {drifted[:5]} -- artifact is stale"
                        )

    # --- ids leg -----------------------------------------------------------------------
    matched = _load_matched_convfinqa_ids()
    if matched is None:
        skipped.append("ids (data/chunks/ absent; rebuild with scripts/pipeline/01_corpus.py)")
    else:
        artifact_ids = set(r["id"] for r in rows)
        if artifact_ids != matched:
            failures.append(
                f"artifact ids != load_matched_questions() ConvFinQA ids: "
                f"{len(artifact_ids - matched)} only in the artifact, "
                f"{len(matched - artifact_ids)} only in the question set. "
                "Did dataset.py's SUBSET_FILES change?"
            )
        else:
            print(f"ok: artifact ids are exactly the {len(matched)} ConvFinQA ids "
                  "load_matched_questions() yields")

    for s in skipped:
        print(f"skip: {s}")
    if failures:
        print("\nthe ConvFinQA turn artifact is not trustworthy (DECISIONS.md DATA-10):\n",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


def _load_t2():
    try:
        return _producer.load_t2_ragbench("ConvFinQA")
    except Exception:
        return None


def _load_matched_convfinqa_ids() -> set | None:
    from rag_sec.eval import CHUNKS_DIR, load_matched_questions

    if not Path(CHUNKS_DIR).is_dir() or not any(Path(CHUNKS_DIR).iterdir()):
        return None
    try:
        df = load_matched_questions()
    except Exception:
        return None
    return set(df[df["subset_source"] == "ConvFinQA"]["id"])


if __name__ == "__main__":
    sys.exit(main())
