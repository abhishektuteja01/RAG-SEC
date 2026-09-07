---
name: walkthrough
description: Orient someone who has just cloned this repo and doesn't know where to start. Use when asked "where do I start", "walk me through this repo", "I'm lost", "explain this project", "what is this", "how do I get up to speed", or when someone opens the project with no specific question.
---

# Walkthrough

The repo is a 14-day solo project with a lot of written record and a corpus that isn't
committed. Someone new drowns in `DECISIONS.md` (163 KB) or starts a 10-hour rebuild they
didn't need. Take them through it in stages, and **stop between stages to ask what they
actually want.** Most people want stage 1 and 2 only.

## Stage 0 — ask first

Before explaining anything, find out which of these they are:

- "What is this project?" → stage 1, then stop.
- "Does it actually work?" → stages 1 and 2.
- "I need to understand the design decisions" → stages 1–3.
- "I want to run it myself / extend it" → all four.

Don't dump all four stages unprompted.

## Stage 1 — orient

Three facts, no more:

- Retrieval-augmented QA over 799 SEC 10-K filings. A learning project, deployed on EC2.
- The point is the *measurement*: six retrieval configurations ("arms"), each adding one
  technique to the previous, all scored on the same corpus, split, and labels.
- **The one result that matters:** the shipped arm — dense + BM25/RRF + cross-encoder
  reranker + company filter + query strip — scores **recall@10 = 0.747 on the held-out test
  split** (`DECISIONS.md` `RETR-39`). The dense-only baseline was 0.337.

## Stage 2 — prove it works

Don't take the number on faith. Replay it from the stored rerank scores — **~90 s**, no GPU,
no Postgres, no API key, no money:

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

Expect `filtered_stripped` recall@10 **0.747**, 1545 of 1546 questions scored.

**One prerequisite, and be honest about it.** The scores and gold labels are in git, but
scoring reads `data/chunks/` (gitignored, 377 MB) and pulls the question set from Hugging
Face. A fresh clone runs `uv run scripts/pipeline/01_corpus.py` first — ~1 h, 2.9 GB from
EDGAR, resumable, skips whatever is already on disk. That is still the short path: it skips
the ~9.3 h embed and Postgres entirely. Only Arms 1–2 need those.

## Stage 3 — understand it

Read in this order. Say what each one is *for* so they can stop early.

1. `README.md` — what it is and the result. Ignore its chart; it predates the `RETR-35`
   label correction and its numbers were withdrawn.
2. `scripts/README.md` — the run order, with real calendar dates and what each command costs.
   Its "Read this before any date below" section is the fastest way to stop being confused by
   `dayN_` filenames.
3. The arm progression — use the `explain-arm` skill, one arm at a time. That is where the
   design story lives: what each technique was meant to fix and whether it did.
4. `DECISIONS.md` — only when a specific choice looks arbitrary. It is indexed by ID
   (`RETR-39`, `ARM4-10`, `AGENT-19`), so grep for the ID rather than reading it front to back.
   It is the source of truth for every number; its **current baseline table at the top**
   supersedes every table below it.
5. `INVENTORY.md` — file-by-file, what each thing is and whether it's still needed. Reach for
   it when wondering "can I delete this?"

Traps worth flagging as they come up: phase numbers are not arm numbers (04 = Arms 1–2, 05 =
Arm 3, 06 = Arm 4, 07 = Arm 6); there is no Arm 5 code; phase 02 cannot run and its labels are
frozen; only `variant = 'A'` is live; `scripts/checks/agent_loop_smoke_test.py` spends ~$0.04
per run despite being called a test.

## Stage 4 — rebuild or extend (optional)

Only if they mean it. `scripts/README.md` §"Start here" has the no-cluster path end to end;
`RUNBOOK.md` has the GPU cluster procedure. Before they run anything, point out the cost table
in `scripts/README.md` — two commands spend real money, and one of them looks like a test.
