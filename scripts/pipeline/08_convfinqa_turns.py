"""Recovers ConvFinQA's real multi-turn dialogues and joins them to our question ids.

Not an arm, and not a step in the Arms run order: phases 01-07 neither produce nor consume
this. It needs only the network.

WHY IT EXISTS. T2-RAGBench ships ConvFinQA as `data/ConvFinQA/turn_0.jsonl` and nothing
else -- one standalone, de-referenced question per conversation ("What was the net cash from
operating activities for Jack Henry & Associates in the fiscal year ended June 30, 2009, as
reported in their 2009 annual report?"). The conversation it was cut from is public and free:
the original ConvFinQA release carries all of it, including the follow-ups that actually
exercise coreference and ellipsis ("what about in 2008?", "what is the difference?").
Every published number here is single-turn, so nothing downstream changes; this only puts the
turns on disk.

THE JOIN, AND WHY IT IS GUARDED. `convfinqa_N` indexes straight into `train.json + dev.json`
concatenated. That is a claim about the ORDER of somebody else's artifact, which its producer
never promised -- the exact shape of DECISIONS.md RETR-24 / AGENT-16 / INFRA-22. So position
is never trusted on its own: every row must agree with the upstream record positionally on
BOTH `filename` and the first turn's executed answer, or the build aborts. A second,
order-free key (`filename`, first answer) is reported alongside as corroboration; it cannot
replace the positional join because 671 conversations share a filing with a sibling that has
the same first answer, and only position separates those.

WHAT IS NOT HERE. No mechanism consumes this yet -- `/ask` takes one string with no history
and `retrieve()` takes one query. Deliberately left alone: `dataset.py`'s `SUBSET_FILES` is
untouched, so `load_matched_questions()` still yields the same ids the stored score files are
keyed on.

Usage:
    python scripts/pipeline/08_convfinqa_turns.py build     # writes data/convfinqa_turns.jsonl
    python scripts/pipeline/08_convfinqa_turns.py verify    # re-runs the join checks only

Free. No GPU, no Postgres, no API key. ~18 MB download, cached after the first run.
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.dataset import load_t2_ragbench  # noqa: E402

# The original ConvFinQA release (Chen et al., EMNLP 2022). `test_*.json` in the same zip is
# the blind leaderboard split: it ships no answers and no dialogue annotation, so it cannot
# be joined and is not read. train+dev is the whole of what T2-RAGBench itself drew from.
UPSTREAM_URL = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
UPSTREAM_DIR = _ROOT / "data" / "convfinqa_upstream"
UPSTREAM_FILES = ("train.json", "dev.json")  # order is load-bearing: it defines the index
OUT_PATH = _ROOT / "data" / "convfinqa_turns.jsonl"

ID_PREFIX = "convfinqa_"


def fetch_upstream() -> list[dict]:
    """Downloads (once) and loads train.json + dev.json, concatenated in that order."""
    if not all((UPSTREAM_DIR / n).exists() for n in UPSTREAM_FILES):
        UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {UPSTREAM_URL}")
        with urllib.request.urlopen(UPSTREAM_URL) as resp:
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in UPSTREAM_FILES:
                # The zip nests everything under data/; flatten so the cache is self-describing.
                member = next(m for m in z.namelist() if m.endswith("/" + name))
                (UPSTREAM_DIR / name).write_bytes(z.read(member))
        print(f"cached {len(UPSTREAM_FILES)} files in {UPSTREAM_DIR}")

    out: list[dict] = []
    for name in UPSTREAM_FILES:
        with open(UPSTREAM_DIR / name) as f:
            out.extend(json.load(f))
    return out


def _norm_answer(value) -> str:
    """Answers are mostly floats but some are strings ('no'). Compare on one normal form."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _stem(file_name: str) -> str:
    """T2 writes `pdf/JKHY/2009/page_28.pdf`; upstream writes `JKHY/2009/page_28.pdf`."""
    return file_name[4:] if file_name.startswith("pdf/") else file_name


def _turn_source(record: dict, turn: int) -> dict:
    """The source QA a turn was composed from.

    A `Single_*` conversation is one FinQA question broken into turns and carries `qa`; a
    `Double_*` one splices two and carries `qa_0`/`qa_1`, with `annotation.qa_split[i]`
    naming which of them turn `i` came from. That matters downstream because gold evidence
    (`gold_inds`) hangs off the source QA, not off the turn.
    """
    if "qa" in record:
        return record["qa"]
    return record[f"qa_{record['annotation']['qa_split'][turn]}"]


def check_join(t2, upstream: list[dict]) -> tuple[list[str], dict]:
    """Returns (failures, stats). Empty failures means the positional join is sound."""
    failures: list[str] = []

    ids = t2["id"].tolist()
    bad_id = [i for i in ids if not i.startswith(ID_PREFIX) or not i[len(ID_PREFIX):].isdigit()]
    if bad_id:
        failures.append(f"{len(bad_id)} ids are not `{ID_PREFIX}<int>`, e.g. {bad_id[:3]}")
        return failures, {}

    order = sorted(range(len(ids)), key=lambda j: int(ids[j][len(ID_PREFIX):]))
    t2 = t2.iloc[order].reset_index(drop=True)
    numbers = [int(i[len(ID_PREFIX):]) for i in t2["id"]]
    if numbers != list(range(len(numbers))):
        failures.append(
            f"ids are not contiguous 0..{len(numbers) - 1}; the positional join is undefined"
        )
        return failures, {}

    if len(t2) != len(upstream):
        failures.append(f"row count {len(t2)} != upstream conversation count {len(upstream)}")
        return failures, {}

    # Leg 1, the one the build rests on: position must agree on two independent fields.
    name_bad, answer_bad, length_bad = [], [], []
    for i, rec in enumerate(upstream):
        row = t2.iloc[i]
        if _stem(row["file_name"]) != rec["filename"]:
            name_bad.append(row["id"])
        ann = rec["annotation"]
        if _norm_answer(row["program_answer"]) != _norm_answer(ann["exe_ans_list"][0]):
            answer_bad.append(row["id"])
        n = len(ann["dialogue_break"])
        if not (len(ann["exe_ans_list"]) == len(ann["turn_program"]) == len(ann["qa_split"]) == n):
            length_bad.append(row["id"])
    for label, bad in (
        ("filename", name_bad), ("first-turn answer", answer_bad), ("per-turn list length", length_bad)
    ):
        if bad:
            failures.append(f"{len(bad)}/{len(upstream)} rows disagree on {label}, e.g. {bad[:5]}")

    # Leg 2, corroboration only: an order-free content key must never point somewhere else.
    index = defaultdict(list)
    for i, rec in enumerate(upstream):
        index[(rec["filename"], _norm_answer(rec["annotation"]["exe_ans_list"][0]))].append(i)
    unique = ambiguous = missing = contradicted = 0
    for i, row in enumerate(t2.itertuples()):
        hits = index.get((_stem(row.file_name), _norm_answer(row.program_answer)), [])
        if not hits:
            missing += 1
        elif len(hits) == 1:
            unique += 1
            contradicted += hits[0] != i
        else:
            ambiguous += 1
            contradicted += i not in hits
    if missing:
        failures.append(f"{missing} rows have no content-key match upstream at all")
    if contradicted:
        failures.append(f"{contradicted} rows where the content key points away from the position")

    stats = {
        "rows": len(t2),
        "key_unique": unique,
        "key_ambiguous": ambiguous,
        "key_missing": missing,
    }
    return failures, stats


def build_rows(t2, upstream: list[dict]) -> list[dict]:
    order = sorted(range(len(t2)), key=lambda j: int(t2["id"].iloc[j][len(ID_PREFIX):]))
    t2 = t2.iloc[order].reset_index(drop=True)

    rows = []
    for i, rec in enumerate(upstream):
        src = t2.iloc[i]
        ann = rec["annotation"]
        qa_split = ann["qa_split"]
        rows.append({
            "id": src["id"],
            "source_id": rec["id"],
            # Ours (DATA-2), not upstream's train/dev file -- the split every score is keyed on.
            "split": src["split"],
            "file_name": src["file_name"],
            "company_cik": int(src["company_cik"]),
            "report_year": int(src["report_year"]),
            "n_turns": len(ann["dialogue_break"]),
            # T2's rewritten standalone question -- what the frozen gold and every stored
            # score were measured against. It is NOT turn 0; `turns[0]` is.
            "t2_question": src["question"],
            "turns": ann["dialogue_break"],
            "turn_answers": ann["exe_ans_list"],
            "turn_programs": ann["turn_program"],
            "qa_split": qa_split,
            # True where the turn's gold evidence is the same `gold_inds` turn 0 was labelled
            # from, i.e. where the frozen GOLD-7 labels apply unchanged. False turns would
            # need evidence this repo has never labelled.
            "shares_turn0_gold": [q == qa_split[0] for q in qa_split],
            "gold_inds_keys": sorted(_turn_source(rec, 0).get("gold_inds", {})),
        })
    return rows


def summarise(rows: list[dict]) -> None:
    for split in ("train", "dev", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        follow = sum(r["n_turns"] - 1 for r in sub)
        reusable = sum(sum(r["shares_turn0_gold"][1:]) for r in sub)
        single = sum(r["source_id"].startswith("Single_") for r in sub)
        print(
            f"  {split:<5} conversations {len(sub):>4} (Single {single:>4})  turns "
            f"{sum(r['n_turns'] for r in sub):>5}  follow-ups {follow:>5}  "
            f"of which reuse turn-0 gold {reusable:>5}"
        )
    print(f"  turns per conversation: {dict(sorted(Counter(r['n_turns'] for r in rows).items()))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    upstream = fetch_upstream()
    t2 = load_t2_ragbench("ConvFinQA")
    failures, stats = check_join(t2, upstream)

    if failures:
        print("the ConvFinQA turn join is not sound (DECISIONS.md DATA-10, INFRA-22):\n",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"ok: positional join agrees on filename and first answer for all {stats['rows']} rows")
    print(f"    order-free content key: {stats['key_unique']} unique, "
          f"{stats['key_ambiguous']} ambiguous, {stats['key_missing']} missing, "
          "0 contradicting the position")

    if args.command == "verify":
        return 0

    rows = build_rows(t2, upstream)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} conversations to {args.out}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
