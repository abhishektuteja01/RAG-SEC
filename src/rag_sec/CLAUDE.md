# src/rag_sec/ — the library

Modules carry their own docstrings with PRODUCES / READS / traps. Read the docstring before
changing a module; it is more current than any summary.

## What ships

`retrieve.py` is the shipping path: dense + BM25/RRF + cross-encoder rerank. **Its defaults
already are the serving configuration** — filter + strip + `year_bias`, and since `DEPLOY-25`
`strip_dense=True`, `read_depth=200`, `year_text_fusion=True` (test recall@10 0.831,
`RETR-51`/`RETR-52`). Do not re-derive any of it at a call site. `api.py` serves exactly that
and adds no retrieval of its own.

Candidate generation lives in `candidates.py` and is shared with every offline arm, so the
scripts and the API cannot drift apart. Change it there, not in a copy.

## Traps

- **`agent.py` (Arm 6) is not what the API serves.** Its specced experiment was untestable here —
  the multi-document subset is empty in every split — and that negative is the published finding
  (`AGENT-15`). The loop that did run showed an answer-accuracy win that does not survive a
  fair baseline (`AGENT-30`), and Arm 6 is closed as a negative (`AGENT-33`/`AGENT-34`). Never
  quote its "union" recall row (`AGENT-22`).
- **`agent.py` pins its `retrieve()` settings (`ARM6_RETRIEVE_SETTINGS`) and does NOT follow the
  serving defaults** — they must match the replayed baseline (`AGENT-36`).
- **`compress.py` was measured and never shipped.** `agent.py` sends uncompressed context. Cost
  work is closed (`COST-36`) — do not reopen a decision on cost grounds alone.
- **Read `*_scores.jsonl` through `eval.load_ranking`,** which sorts on load. The files are not
  stored in rank order; reading one raw silently produces a wrong score (`AGENT-16`).
- **`tracing.py` no-ops without `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`.** Keyless machines
  and CI are unaffected. Keep it that way.
- **Only `variant = 'A'` is live** (`LIVE_VARIANT` in `candidates.py`). B and C were never
  re-embedded.
- **Model construction is not thread-safe on MPS.** Concurrent construction segfaults the
  machine (`AGENT-10`, `AGENT-17`). Run serial.
