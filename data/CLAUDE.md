# data/ — on-disk artifacts

Mostly gitignored and regeneratable. A few files are force-tracked because they are the only
way to reproduce a number cheaply. Check `.gitignore` before assuming a file ships.

## Read the file before trusting its shape

This is where the project's recurring bug lives (`RETR-24`, `RETR-30`, `AGENT-16`, `INFRA-22`):
code assuming a shape, order, or provenance the producer never guaranteed. `head -1` the file.

- **`*_scores.jsonl` is not in rank order.** Load it via `rag_sec.eval.load_ranking`, which sorts.
- **A replayed artifact is not the Postgres artifact.** Do not index one with an index derived
  from the other.

## Naming

- `dayN_*` — the day the run happened, not a version. `scripts/README.md` maps days to dates.
- `retr7_*` — produced after the `RETR-7`/`RETR-8` re-index. Anything older predates it and is
  not comparable.
- `*.bak_old_labeler` — pre-`RETR-35` labels. Never quote these.

## Frozen and unrebuildable

`day7_gold_inds_matched_full.json` has no producer in this repo. The gold labels are committed
and frozen — reproduce against them, never regenerate them (`INFRA-15`, `GOLD-7`).

## Not in git

`chunks/`, `filings/`, `parsed/` are large and gitignored, but scoring reads `chunks/`. Rebuild
with `scripts/pipeline/01_corpus.py`, which is resumable and skips what is already on disk.
