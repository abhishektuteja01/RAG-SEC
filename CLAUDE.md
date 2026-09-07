# RAG-SEC — orientation

Retrieval-augmented QA over 799 SEC 10-K filings. Built in 14 days as a learning project,
deployed on EC2. The point is not the app. It is a measured comparison of retrieval
techniques, one change at a time, so each gain is attributable.

**"Arm" = a numbered retrieval configuration.** Each arm adds exactly one technique to the
previous one. All are scored on the same corpus, split, and gold labels, so the delta between
two arms is the technique, not the setup.

| Arm | Adds | Status |
|---|---|---|
| 1 | dense retrieval | baseline |
| 2 | + BM25, fused with RRF | — |
| 3 | + cross-encoder reranker | **shipped** |
| 4 | table-layout variants | lost |
| 5 | multi-vector late interaction | never built — no code |
| 6 | agentic loop | published negative |

## Reproduce the headline number

Replays Arm 3 from stored rerank scores. No GPU, no Postgres, no API key, no money.

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

Prerequisite: scoring reads `data/chunks/`, which is gitignored. A fresh clone runs
`uv run scripts/pipeline/01_corpus.py` first. That is still the short path — it skips the
local embed and Postgres entirely. Only Arms 1–2 need those.

## Vocabulary

- **chunk** — a filing cut into a ~900-word piece. The unit everything is stored and searched as.
- **embedding** — a vector standing for a chunk's meaning, so similar text sits close together.
- **reranker** — a slower, more careful model that re-orders a shortlist after the fast search picks it.
- **the cluster** — the university GPU machines. Jobs there are booked and queued, not instant.
- **variant** — which table layout a chunk was built with. `A` is the real corpus.

## Which doc answers what

| File | Answers |
|---|---|
| `README.md` | what this is, and the result |
| `DECISIONS.md` | why every choice was made — **the only source of truth for any number** |
| `scripts/README.md` | run order, calendar dates, what each command costs |
| `INVENTORY.md` | file-by-file map, and what's still missing |
| `RUNBOOK.md` | the GPU cluster procedure, start to finish |

Quote numbers from `DECISIONS.md`'s current baseline table at the top of the file, never from a
lower table — those predate the `RETR-35` label correction. Cite the decision ID (`RETR-39`,
`ARM4-10`) so a reader can check you. `README.md`'s chart is safe to quote: its producer
re-derives every value from `data/` and aborts if one disagrees with the baseline table.

## Rules that always apply

- **Open the file before trusting its shape.** This project's recurring bug is code assuming a
  shape, order, or provenance the producer of an on-disk artifact never guaranteed (`RETR-24`,
  `RETR-30`, `AGENT-16`, `INFRA-22`). It survives code review every time.
- **Phase numbers are not arm numbers.** See `scripts/CLAUDE.md`.
- **Two commands spend real money.** Both are under `scripts/`; the cost table is in
  `scripts/README.md`.

## Skills

- `/explain-arm` — any arm: what it adds, which files, what it scored, the verdict.
- `/walkthrough` — the "I just cloned this and I'm lost" path.
