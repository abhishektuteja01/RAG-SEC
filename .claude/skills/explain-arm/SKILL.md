---
name: explain-arm
description: Explain one of this project's retrieval arms (Arm 1-6) — what it is, what it adds over the previous arm, which files implement it, what it scored, and whether it won or lost. Use when asked "explain arm 3", "what does the reranker add", "walk me through the arms", "why did arm 4 lose", "what is the agentic loop", or any question naming an arm number.
---

# Explain an arm

An **arm** is a numbered retrieval configuration. Each adds one technique to the previous one
and is scored on the same corpus, split, and gold labels, so the delta is attributable.

## How to answer

Read the actual files before answering. Do not reconstruct an arm from memory or from this
file alone — the phase docstrings carry PRODUCES / READS / traps / when it really ran, and
they are more current than any summary.

For the requested arm, give six things, in this order:

1. **The concept, in plain English.** No jargon that isn't unpacked. "Dense retrieval" means
   turning the question into 1024 numbers and finding chunks whose numbers sit nearby.
2. **What it adds over the previous arm, and why that was expected to help.** One technique.
   Name the failure mode of the previous arm that this was meant to fix.
3. **The files.** Both the pipeline phase and the library module it calls.
4. **The command that runs it**, copy-pasteable, taken from `scripts/README.md`'s run-order
   table or the script's own `--help`. Say whether it costs money or needs a GPU or Postgres.
5. **What it scored, and where that number lives.** Quote from `DECISIONS.md`'s current
   baseline table at the top of the file — never from an older table further down, and never
   from `README.md`'s chart, which predates the label correction.
6. **The honest verdict**, including dead ends. If something lost, say it lost.

Cite the relevant `DECISIONS.md` IDs inline so the reader can check you.

## Arm → file map

| Arm | What it adds | Implemented in | DECISIONS |
|---|---|---|---|
| 1 | dense retrieval (BGE-M3 → pgvector cosine) | `scripts/pipeline/04_arms_first_stage.py arm1` + `src/rag_sec/candidates.py` | `ARM1-2`, `RETR-5` |
| 2 | + BM25 keyword search, fused with RRF | same file, `arm2` | `ARM2-1` |
| 3 | + cross-encoder reranker (`bge-reranker-v2-m3`) | `scripts/pipeline/05_arm3_rerank.py` + `src/rag_sec/retrieve.py` | `ARM3-1`, `ARM3-2`, `RETR-5`, `RETR-6`, `RETR-16`, `RETR-39` |
| 4 | table-layout variants A / B / C | `scripts/pipeline/06_arm4_tables.py` | `ARM4-2`…`ARM4-10` |
| 5 | multi-vector late interaction | **no code — never built** | `ARM5-1` |
| 6 | agentic loop: plan → retrieve → judge → answer | `src/rag_sec/agent.py` + `scripts/pipeline/07_arm6_loop.py` | `AGENT-15`, `AGENT-19`, `AGENT-22`, `AGENT-25` |

Phase numbers are not arm numbers. Phase 04 covers two arms; there is no phase for Arm 5.

## Per-arm notes you must not omit

- **Arm 3 is the shipped path.** `src/rag_sec/retrieve.py`'s defaults *are* the winning
  `filtered_stripped` cell, and `src/rag_sec/api.py` serves exactly that. Its result comes from
  a 2×2: control, + company filter, + query strip, + both. The fourth cell is what makes the
  gain attributable. The two together are worth more than their sum (`RETR-6`).
- **Arm 4 is a documented dead end.** A won on every metric; B and C were dropped (`ARM4-10`)
  and were deliberately never re-embedded after the `RETR-7`/`RETR-8` re-index, so an A-vs-B
  number produced today compares two different corpora and means nothing. The phase survives
  because `score --variant A` computes the *live* `unfiltered_raw` control, and because it is
  the only code that can score B or C at all. `variants` can spend money; all 498 summaries are
  cached and an uncached one is a hard error.
- **Arm 5 has no code.** It was measured on paper against the real corpus — 99,654 chunks at
  ~927 tokens each, 1024 dims *per token*, ≈378 GB of vectors — and dropped before building.
  Say this plainly; do not invent an implementation.
- **Arm 6's published finding is a negative, but not the one people assume.** `spec.md` asked
  for the loop measured on multi-document questions; that subset is **empty** in every split,
  so the loop's best case is untestable on this benchmark (`AGENT-15`). The trajectory half
  that did run: the loop wins on answer accuracy (69.2% vs 61.6%, n=200, McNemar p=0.00098)
  and *loses* on retrieval (iteration-1 recall@10 0.703 vs static 0.736) — because the
  planner's rewrites drop the company name and silently switch the company filter off
  (`AGENT-19`, `AGENT-25`). Do not quote the "all iters (union)" row; it compares a 40-slot
  budget against a 10-slot one (`AGENT-22`). `run` costs ~$4.10 per 200 questions.

## Reproducing a number while explaining

Arm 3 replays offline in ~90 s once `data/chunks/` exists:

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

Arms 1 and 2 query Postgres live, so they need the corpus indexed first (~10.5 h, no GPU, no
money). Always pass `--scores` and `--out` explicitly — the defaults are stale `day8_*` names.
