# Working rules for this project

Learning project first, production system second. Abhishek must be able to defend
every line and number in an interview — unexplained code is worse than no code.

## Rules
1. **Explain concepts in simple english** before each step. Make it simple enough
for user to explain it in an interview easily, and then move forward.
1. **Be concise.** No verbose paragraphs, no comments where not needed, targeted
   wording everywhere — chat, docs, code. Keep Standing Context below up to date as
   state changes.
2. **Explain before doing.** Before non-trivial code: what we're building and why,
   the concept if new (BM25, RRF, etc.), and the alternatives considered. Wait for a
   go-ahead — don't batch unexplained steps. "Just do it" waives this for that one
   step only; still summarize afterward.
3. **Never let code go unexplained.** If the honest answer to "why this way" is
   "that's just common," dig for the real reason (correctness/perf/failure mode) or
   say plainly it's an arbitrary default.
4. **Keep `DECISIONS.md` current.** Every real choice (model, k, chunk size, split,
   fusion method, etc.) gets a couple lines there — tried, result, why kept/dropped —
   written as it happens, not backfilled.
5. **Production-ready, not demo-ready.** Real infra over shortcuts (real BM25, pinned
   models, measured latency/cost, containers). If a shortcut is taken anyway, say so
   and log it as a decision with the tradeoff.
6. **Follow `spec.md`, flag drift.** It's the source of truth for scope/sequence/eval
   methodology. If reality contradicts it, say so and propose a fix — don't silently
   work around it.
7. **Verify currency before adopting anything new.** Training data has a cutoff;
   search for a model/library/method's current state before using it for the first
   time here. Say what was checked and when, or say search wasn't possible.
8. **Keep `CLAUDE.md`/`DECISIONS.md`/comments concise.** One row or a few lines,
   targeted and direct — no restating context already in the file, no play-by-play.

## Standing context

- Eval set: T²-RAGBench (`G4KMU/t2-ragbench`) — 23,088 rows / 7,318 contexts (not
  real-filing count — 799 unique filings once collapsed by cik+year, TAT-DQA CIKs
  unresolved).
- Postgres 18.6 + pgvector 0.8.6 running locally via `docker compose up -d`.
- Day 2 (done): corpus = real 10-Ks from EDGAR (`src/rag_sec/edgar.py`). Parser:
  `sec-parser` (not Docling — DECISIONS.md #9), `src/rag_sec/parsing.py`. Chunking v1:
  `src/rag_sec/chunking.py` (DECISIONS.md #10), 11,391 chunks over 100 filings, median
  885 tokens. Findings: `data/day2_findings.md`. TODO: full-corpus ingest (later day
  per spec.md).
- Next: Day 3 — eval harness (recall@10/50, nDCG@10, MRR) + Arm 1 (dense only).
