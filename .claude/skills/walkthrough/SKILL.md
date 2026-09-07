---
name: walkthrough
description: Orients someone who has just cloned this repo and does not know where to start, staging the tour from "what is this" to "how do I rebuild it". Use when the user asks "where do I start", "walk me through this repo", "I'm lost", "explain this project", "what is this", "how do I get up to speed", "give me the tour", or opens the project with no specific question.
---

# Walkthrough

`DECISIONS.md` is 160 KB and the corpus is not committed, so a newcomer either drowns or starts
a rebuild they did not need. Take them in stages. **Stop between stages and ask.** Most people
want stages 1 and 2 only.

## Stage 0 — ask first

Map their question, then run only the stages it needs:

- "What is this project?" → stage 1, stop.
- "Does it actually work?" → stages 1–2.
- "Why was it built this way?" → stages 1–3.
- "I want to run or extend it." → all four.

Do not dump all four unprompted.

## Stage 1 — orient

Three facts, no more:

- Retrieval-augmented QA over 799 SEC 10-K filings. A learning project, deployed on EC2.
- The point is the *measurement*: six configurations ("arms"), each adding one technique to the
  previous, all scored on the same corpus, split, and labels.
- The shipped arm is dense + BM25/RRF + cross-encoder reranker + company filter + query strip.
  Quote its test recall@10 from `DECISIONS.md` `RETR-39` — read the file, do not recite a number
  from memory.

## Stage 2 — prove it works

Do not let them take the number on faith. Replay it: no GPU, no Postgres, no API key, no money.

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

Be honest about the prerequisite: scoring reads `data/chunks/`, which is gitignored, and pulls
the question set from Hugging Face. A fresh clone runs `uv run scripts/pipeline/01_corpus.py`
first — resumable, skips what is on disk. That is still the short path; it skips the embed and
Postgres entirely. Only Arms 1–2 need those.

## Stage 3 — understand it

Read in this order, and say what each is *for* so they can stop early.

1. `README.md` — what it is and the result. Its chart is current: every value is recomputed
   from `data/` at generation time and cross-checked against `DECISIONS.md`.
2. `scripts/README.md` — run order, real dates, per-command cost. Its "Read this before any date
   below" section is the fastest cure for `dayN_` filename confusion.
3. The arm progression — use `explain-arm`, one arm at a time. That is where the design story is.
4. `DECISIONS.md` — only when a choice looks arbitrary. Grep the ID (`RETR-39`, `ARM4-10`); do not
   read it front to back. Its current baseline table at the top supersedes every table below.
5. `INVENTORY.md` — file-by-file. Reach for it on "can I delete this?".

Flag traps as they come up: phase numbers are not arm numbers; there is no Arm 5 code; phase 02
cannot run and its labels are frozen; only `variant = 'A'` is live; one file in `scripts/checks/`
spends real money despite being named a test.

## Stage 4 — rebuild or extend

Only if they mean it. `scripts/README.md` §"Start here" is the no-cluster path end to end, and
is the only path a clone can follow — the GPU cluster procedure is not in the repo. Point at the
cost table in `scripts/README.md` before they run anything.
