"""Pre-check: can a gold chunk be interpreted on its own?

Before building a context compressor (which can only ever keep or drop text that is
already in the chunk), measure how much of the gold evidence is *already* uninterpretable
in isolation -- units stranded in a neighbouring chunk, "(continued)" table fragments with
no row labels, page furniture spliced mid-sentence. That damage caps answer accuracy
regardless of compression, so it sets the ceiling the compressor is working under.

Read-only. No API calls, no writes outside the printed report.
"""

import re
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from rag_sec.eval import (  # noqa: E402
    _gold_evidence_resolved,
    _load_chunks,
    gold_relevant_chunk_evidence,
    load_matched_questions,
)

# A scale qualifier only has to appear *somewhere* in the chunk to be recoverable -- we are
# measuring presence, not placement (placement is the compressor's job, absence is not).
SCALE_RE = re.compile(
    r"\((?:[^)]{0,40}?)(?:in|of)[\s\xa0]+(?:millions?|thousands?|billions?)[^)]{0,40}\)"
    r"|(?:dollars|amounts|shares)[\s\xa0]+in[\s\xa0]+(?:millions?|thousands?|billions?)"
    r"|except[\s\xa0]+per[\s\xa0]+share"
    r"|\bin[\s\xa0]+(?:millions?|thousands?|billions?)\b",
    re.I,
)
CONTINUATION_RE = re.compile(r"\(\s*continued|continued[\s\xa0]+from[\s\xa0]+previous", re.I)
FURNITURE_RE = re.compile(r"SEQ\.=\d+,FOLIO=|Table of Contents", re.I)
# A pipe row is "unlabeled" when its first cell carries no word -- i.e. bare numbers whose
# row caption was lost to a chunk boundary (the JPM "(Table continued from previous page)"
# shape, where every row reads `$6,635 | $- | $- | $6,635`).
_LETTER_RE = re.compile(r"[A-Za-z]")


def _pipe_rows(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.count("|") >= 2]


def flags(text: str) -> dict:
    rows = _pipe_rows(text)
    is_table = len(rows) >= 3
    unlabeled = [r for r in rows if not _LETTER_RE.search(r.split("|")[0])]
    return {
        "is_table": is_table,
        "no_scale": is_table and not SCALE_RE.search(text),
        "continuation": bool(CONTINUATION_RE.search(text)),
        "unlabeled_rows": is_table and len(unlabeled) > len(rows) / 2,
        "furniture": bool(FURNITURE_RE.search(text)),
    }


def main() -> None:
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    resolved = _gold_evidence_resolved()

    n_q = n_with_gold = 0
    layer_counts: Counter = Counter()
    # Per-question: does ANY of its gold chunks carry the flag?
    q_flags: Counter = Counter()
    # Restricted to questions whose gold evidence is a TABLE ROW -- the ones where a missing
    # scale qualifier actually changes the answer's magnitude.
    n_table_q = 0
    tq_flags: Counter = Counter()
    n_gold_chunks = 0
    c_flags: Counter = Counter()

    for _, row in dev.iterrows():
        n_q += 1
        evidence = gold_relevant_chunk_evidence(row)
        if not evidence:
            continue
        n_with_gold += 1
        chunks = _load_chunks(row["chunk_file"])
        layers = {v["layer"] for v in evidence.values()}
        layer_counts.update(layers)
        gold_is_table_row = "table_row" in layers or bool(
            (resolved.get(row["id"]) or {}).get("table_rows")
        )
        if gold_is_table_row:
            n_table_q += 1

        agg = Counter()
        for cid in evidence:
            n_gold_chunks += 1
            f = flags(chunks[cid]["text"])
            for k, v in f.items():
                if v:
                    c_flags[k] += 1
                    agg[k] += 1
        for k in agg:
            q_flags[k] += 1
            if gold_is_table_row:
                tq_flags[k] += 1

    def pct(n, d):
        return f"{n:5d} ({100 * n / d:5.1f}%)" if d else "n/a"

    print(f"dev questions                      : {n_q}")
    print(f"  with >=1 gold-relevant chunk     : {pct(n_with_gold, n_q)}")
    print(f"  gold evidence is a table row     : {pct(n_table_q, n_with_gold)}")
    print(f"gold chunks examined               : {n_gold_chunks}")
    print(f"gold layers used                   : {dict(layer_counts)}")
    print("\n-- per GOLD CHUNK --")
    for k in ("is_table", "no_scale", "continuation", "unlabeled_rows", "furniture"):
        print(f"  {k:16s}: {pct(c_flags[k], n_gold_chunks)}")
    print("\n-- per QUESTION (any gold chunk flagged) --")
    for k in ("no_scale", "continuation", "unlabeled_rows", "furniture"):
        print(f"  {k:16s}: {pct(q_flags[k], n_with_gold)}")
    print(f"\n-- per TABLE-ROW QUESTION (n={n_table_q}) -- the ones where scale changes the answer")
    for k in ("no_scale", "continuation", "unlabeled_rows", "furniture"):
        print(f"  {k:16s}: {pct(tq_flags[k], n_table_q)}")


if __name__ == "__main__":
    main()
