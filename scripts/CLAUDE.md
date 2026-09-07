# scripts/ — pipeline phases

`scripts/README.md` is the run order and the cost table. Read it before running anything here.

## Phase numbers are not arm numbers

| Phase | Covers |
|---|---|
| `01_corpus.py` | download + chunk the filings |
| `02_gold_labels.py` | gold labels — **cannot run end to end** |
| `03_index.py` | embed + load Postgres |
| `04_arms_first_stage.py` | Arms 1 and 2 |
| `05_arm3_rerank.py` | Arm 3 |
| `06_arm4_tables.py` | Arm 4 |
| `07_arm6_loop.py` | Arm 6 |

There is no phase for Arm 5. It was costed on paper and dropped before building (`ARM5-1`).

## What spends real money

- `checks/agent_loop_smoke_test.py` — live Gemini, no dry-run flag, despite the name.
- `07_arm6_loop.py run` — gated behind `--allow-paid-run`.

Everything else reads only. Amounts are in `DECISIONS.md`.

## Traps

- **Phase 02 cannot be rebuilt.** `data/day7_gold_inds_matched_full.json` has no producer in this
  repo and the raw datasets it needs are absent. The labels are committed and frozen. A fresh
  clone can reproduce every number but must not try to rebuild the labels (`INFRA-15`, `GOLD-7`).
- **Pass `--scores` and `--out` explicitly** to `05_arm3_rerank.py`. The defaults are stale
  `day8_*` names.
- **`06_arm4_tables.py variants` can spend money.** All summaries are cached; an uncached one is
  a hard error. Only `variant = 'A'` is live — B and C were never re-embedded after the
  `RETR-7`/`RETR-8` re-index, so any A-vs-B number produced today compares two corpora.
- **Arms 1–2 query Postgres live.** They need `03_index.py local` run first. Arm 3 does not.
- `checks/` holds guards that fail a run rather than trust an assumption. Where the on-disk-shape
  bug has bitten, a check lives here — add one rather than a comment.
- `archive/` is finished one-off analyses. Read for provenance; do not wire into the pipeline.
- `pipeline/hpc/` is cluster-only (Slurm). The run procedure is not in the repo.
