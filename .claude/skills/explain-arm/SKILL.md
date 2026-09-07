---
name: explain-arm
description: Explains one of this project's retrieval arms (Arm 1 through Arm 6) — what it adds over the previous arm, which files implement it, what it scored, and whether it won or lost. Use when the user names an arm number, or asks "explain arm 3", "what does the reranker add", "walk me through the arms", "why did arm 4 lose", "what is the agentic loop", "which arm shipped", or "what is multi-vector late interaction".
---

# Explain an arm

An **arm** is a numbered retrieval configuration. Each adds one technique to the previous one
and is scored on the same corpus, split, and gold labels, so the delta is attributable.

## Before answering

Read the phase script and the library module. Do not answer from this file alone — the
docstrings carry PRODUCES / READS / traps and are more current than any summary.

Pull every number from `DECISIONS.md`, from the current baseline table at the top of the file.
Never a lower table. Cite the ID inline.

## Answer in this order

1. **The concept in plain English.** Unpack the jargon.
2. **What it adds over the previous arm, and the failure mode it was meant to fix.** One technique.
3. **The files** — the pipeline phase and the library module it calls.
4. **The command**, copy-pasteable, from `scripts/README.md` or the script's `--help`. Say whether
   it costs money, needs a GPU, or needs Postgres.
5. **What it scored**, quoted from `DECISIONS.md` with the ID.
6. **The verdict, including dead ends.** If it lost, say it lost.

## Arm → file map

| Arm | Adds | Implemented in | DECISIONS |
|---|---|---|---|
| 1 | dense retrieval (BGE-M3 → pgvector cosine) | `scripts/pipeline/04_arms_first_stage.py arm1` + `src/rag_sec/candidates.py` | `ARM1-2`, `RETR-5` |
| 2 | + BM25, fused with RRF | same file, `arm2` | `ARM2-1` |
| 3 | + cross-encoder reranker (`bge-reranker-v2-m3`) | `scripts/pipeline/05_arm3_rerank.py` + `src/rag_sec/retrieve.py` | `ARM3-1`, `ARM3-2`, `RETR-6`, `RETR-16`, `RETR-39` |
| 4 | table-layout variants A / B / C | `scripts/pipeline/06_arm4_tables.py` | `ARM4-2`…`ARM4-10` |
| 5 | multi-vector late interaction | **no code — never built** | `ARM5-1` |
| 6 | agentic loop: plan → retrieve → judge → answer | `src/rag_sec/agent.py` + `scripts/pipeline/07_arm6_loop.py` | `AGENT-15`, `AGENT-19`, `AGENT-22`, `AGENT-25` |

Phase numbers are not arm numbers. Phase 04 covers two arms; there is no phase for Arm 5.

## Per-arm notes you must not omit

- **Arm 3 shipped.** `retrieve.py`'s defaults *are* the winning `filtered_stripped` cell and
  `api.py` serves exactly that. The result comes from a 2×2 — control, + company filter,
  + query strip, + both — and the fourth cell is what makes the gain attributable. The two
  together beat their sum (`RETR-6`).
- **Arm 4 is a documented dead end.** A won; B and C were dropped (`ARM4-10`) and deliberately
  never re-embedded after the `RETR-7`/`RETR-8` re-index, so an A-vs-B number produced today
  compares two corpora and means nothing. The phase survives because `score --variant A`
  computes the live `unfiltered_raw` control and it is the only code that can score B or C.
- **Arm 5 has no code.** Costed on paper against the real corpus and dropped before building.
  Say so plainly; do not invent an implementation.
- **Arm 6's published finding is a negative, but not the assumed one.** The specced experiment
  asked for multi-document questions; that subset is **empty** in every split, so the loop's best
  case is untestable on this benchmark (`AGENT-15`). The trajectory half that did run: the loop
  wins on answer accuracy, loses on retrieval — the planner's rewrites drop the company name and
  silently switch the company filter off (`AGENT-19`, `AGENT-25`). Never quote the "all iters
  (union)" row; it compares a 40-slot budget against a 10-slot one (`AGENT-22`). `run` costs
  real money.

## Reproducing a number while explaining

Arm 3 replays offline once `data/chunks/` exists:

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

Always pass `--scores` and `--out` explicitly; the defaults are stale `day8_*` names. Arms 1–2
query Postgres live and need the corpus indexed first.
