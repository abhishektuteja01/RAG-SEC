# RAG-SEC — orientation

Retrieval-augmented QA over 799 SEC 10-K filings, built in 14 days as a learning project and
deployed on EC2. Its point is not the app: it is a measured comparison of retrieval techniques,
one change at a time, so each gain is attributable. Every number is logged in `DECISIONS.md`
with the run that produced it.

**"Arm" = a numbered retrieval configuration.** Each arm adds exactly one technique to the
previous one, and all are scored on the same corpus, the same split, and the same gold labels —
so the delta between two arms is the technique, not the setup. Arm 1 dense → Arm 2 +BM25/RRF →
Arm 3 +cross-encoder reranker (the shipped arm) → Arm 4 table layouts (lost) → Arm 5 (never
built) → Arm 6 agentic loop.

## Reproduce the headline number

The published test result is **`filtered_stripped` recall@10 = 0.747** (`RETR-39`). Replaying it
from the stored rerank scores takes **~90 s** and needs no GPU, no Postgres, no API key, no money:

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

Verified 2026-09-07: 1545 of 1546 questions scored, `filtered_stripped` 0.747.

**Prerequisite, and it is not free.** The scores file and the gold labels are tracked in git, but
`eval.py` reads `data/chunks/` — gitignored, 382 MB — to resolve gold labels, and pulls the
question set from Hugging Face. A fresh clone must first run
`uv run scripts/pipeline/01_corpus.py` (~1 h, downloads 2.9 GB from EDGAR, resumable). That's
still the short path: it skips the ~9.3 h local embed and Postgres entirely.

**The long path is only for Arms 1–2.** They query Postgres live, so reproducing *those* needs
the corpus plus `03_index.py local` — ~10.5 h total, no GPU, no money. `README.md` and
`scripts/README.md:110` currently send readers down that route by default. For Arm 3 you don't
need it.

## Words used throughout

- **chunk** — a filing cut into a ~900-word piece. The unit everything is stored and searched as.
- **embedding** — a list of 1024 numbers standing for a chunk's meaning, so similar text sits close together.
- **reranker** — a slower, more careful model that re-orders a shortlist after the fast search picks it.
- **the cluster** — the university GPU machines. Jobs there are booked and queued, not instant.
- **variant** — which table layout a chunk was built with. `A` is the real corpus; `B` and `C` were an experiment that lost.

## Which doc answers what

| File | Answers |
|---|---|
| `README.md` | what this is, and the result |
| `scripts/README.md` | run order, real calendar dates, what each command costs |
| `DECISIONS.md` | why every choice was made — the source of truth for any number |
| `INVENTORY.md` | file-by-file map, and what's still missing |
| `RUNBOOK.md` | the GPU cluster procedure, start to finish |

## Traps

- **Phase numbers are not arm numbers.** Phase 04 = Arms 1–2, 05 = Arm 3, 06 = Arm 4, 07 = Arm 6.
- **There is no Arm 5 code.** Multi-vector late interaction was costed on paper — ~378 GB of
  vectors — and dropped before building (`ARM5-1`).
- **Phase 02 cannot run end to end.** `data/day7_gold_inds_matched_full.json` has no producer in
  this repo and the raw datasets it needs are absent. The labels are committed and frozen: a
  fresh clone can reproduce every number but cannot rebuild the labels, and must not try
  (`INFRA-15`, `GOLD-7`).
- **Only `variant = 'A'` is live.** B and C were never re-embedded after the `RETR-7`/`RETR-8`
  re-index, so any A-vs-B number produced today compares two different corpora.
- **A `*_scores.jsonl` file is not stored in rank order.** Read it through
  `rag_sec.eval.load_ranking`, which sorts on load. Reading one raw scored 0.552 against 0.739
  (`AGENT-16`).
- **What costs real money:** `scripts/checks/agent_loop_smoke_test.py` — ~$0.04 per run despite
  the name, live Gemini, no dry-run flag — and `07_arm6_loop.py run`, ~$4.10 per 200 questions,
  gated behind `--allow-paid-run`. Everything else reads only.
- **Arm 6 is not what the API serves.** `src/rag_sec/api.py` serves Arm 3 + company filter +
  query strip. Arm 6's specced experiment was impossible on this benchmark — the
  multi-document subset is empty — and that negative is the published finding (`AGENT-15`).
  The loop it did run wins on answer accuracy and loses on retrieval (`AGENT-19`); don't quote
  its "union" recall row (`AGENT-22`).
- **The recurring bug class here** is code assuming a shape, order, or provenance that the
  producer of an on-disk artifact never guaranteed (`RETR-24`, `RETR-30`, `AGENT-16`, `AGENT-25`).
  Open the file before trusting its shape.

## Skills

- `/explain-arm` — walk through any arm: what it adds, which files, what it scored, the verdict.
- `/walkthrough` — the "I just cloned this and I'm lost" path.
