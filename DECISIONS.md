# Decisions log

Interview script — every real choice, why it was made. See `CLAUDE.md` Rule 4.
Numbered per phase (`INFRA-1`, `DATA-1`, ...) so a phase can grow without renumbering
everything else. Result numbers live only in the baseline table below — decision rows
below don't repeat them.

## Current baseline

Corpus: 799 filings, 99,654 chunks (T²-RAGBench FinQA+ConvFinQA, full CIK coverage).
Dev split: n=1235, seed 42. Gold-relevance labeling: IDF-weighted shingle + numeric
overlap, minimum-evidence floor (`DATA-9`).

| Arm | recall@10 | recall@50 | nDCG@10 | MRR | latency |
|---|---|---|---|---|---|
| 1 — dense (BGE-M3) | 0.279 ± 0.011 | 0.407 ± 0.012 | 0.199 ± 0.009 | 0.216 ± 0.010 | — |
| 2 — +BM25/RRF | 0.407 ± 0.012 | 0.586 ± 0.012 | 0.289 ± 0.009 | 0.309 ± 0.010 | — |
| 3 — +bge-reranker-v2-m3 | 0.501 ± 0.012 | 0.586 ± 0.012 | 0.420 ± 0.011 | 0.476 ± 0.012 | p50 4.67s / p95 4.87s |

## Infra

| # | decision | why |
|---|---|---|
| INFRA-1 | Store: Postgres + pgvector, not a dedicated vector DB (Pinecone/Weaviate/Qdrant) | Dedicated vector DBs win at hundreds-of-millions+ scale (sharding, ANN specialization). Our corpus is ~100k chunks — HNSW on one instance handles that. Also need hybrid search (dense + BM25 + relational metadata) in one SQL query, not 2-3 systems plus consistency glue |
| INFRA-2 | Docker image pinned to `pgvector/pgvector:0.8.6-pg18-trixie` | Checked current state (Aug 2026): Postgres 18 stable (supported to Nov 2030), 19 still beta, 0.8.6 is pgvector's actively-maintained release. Pinned the full tag (not just `pg18-trixie`) so pgvector can't silently drift |
| INFRA-3 | Compose volume mount: `pgdata:/var/lib/postgresql` (not `.../data`) | First attempt used the pre-pg18 path and crash-looped on startup. Postgres 18+ needs the volume one level up so it can manage major-version subdirectories for `pg_upgrade --link` |
| INFRA-4 | BM25 extension: `pg_search` (ParadeDB), not `tsvector`/`ts_rank`, `VectorChord-bm25`, or `pg_textsearch` | `ts_rank` isn't real BM25 (no document-length normalization or term saturation — degrades on long filings). `VectorChord-bm25` would mean re-platforming off our pinned image (`INFRA-2`); `pg_textsearch` ships no package yet. `pg_search` has a prebuilt `.deb` for our exact base image, depends on pgvector rather than competing with it, AGPL (free self-hosted). Index: `USING bm25 (id, text)`; query: `id @@@ paradedb.match(...)` (literal-text match, default OR-mode — matches real BM25 semantics of IDF-weighted overlap, not require-all-terms) |

## Dataset & eval harness

| # | decision | why |
|---|---|---|
| DATA-1 | Primary eval set: T²-RAGBench (`G4KMU/t2-ragbench`), not FinanceBench | FinanceBench: only 150 of 10,231 triples are public — too small to gate on (~4pt stderr at n=150). T²-RAGBench: 23,088 rows / 7,318 docs, fully open. FinanceBench kept as a secondary sanity check only |
| DATA-2 | ConvFinQA split 80/10/10 by document (`context_id`, seed 42), not by row | Ships with no split. Grouping by document (not row) stops the ~1.9 questions/doc from leaking one document across splits. Actual: 78.3/11.5/10.2 by row, zero cross-split docs |
| DATA-3 | TAT-DQA deferred from the eval set | Ships no CIK, only slugified company names — exact-match resolved only 79/173 companies (some are foreign 20-F/40-F filers, permanently out of scope, not a resolution failure). FinQA+ConvFinQA alone give full CIK-resolved coverage without it |
| DATA-4 | Corpus: real 10-Ks pulled live from EDGAR by `(cik, report_year)`, retrieval scored against full filings — not T²-RAGBench's shipped single-page contexts | The dataset ships one annotated page per question; scoring against that inflates recall@k (right page among a handful, not among hundreds). Full filings are the realistic task, but the tradeoff is our recall/nDCG **aren't directly comparable to the paper's own leaderboard** — the paper's own baseline scores <80% even on its easier per-question contexts. State this plainly in the writeup |
| DATA-5 | `edgar.py`: match a 10-K by `reportDate` (period of report) year, not a `filingDate` window | The old filingDate-window logic broke on non-calendar-fiscal-year filers (Apple, Nike, Sysco, ...) — two consecutive fiscal years' 10-Ks could both land in the window, and EDGAR lists filings most-recent-first, so the wrong fiscal year got picked. Audit found 26/100 filings affected (128/1,100 eval questions); re-fetched and re-chunked all 26 |
| DATA-6 | Corpus grown in stages: 100 → 300 → 400 → 603 → 799 filings, ending at full FinQA+ConvFinQA coverage (799 = exact count of unique `(cik, report_year)` pairs across both subsets, confirmed by direct count) | Each growth step invalidates prior Arm numbers by construction (more filings = more distractor chunks) — Arm 1-3 must be re-run unchanged after every corpus change. One growth (300→400) was an accidental second ingest run with a different seed that went unnoticed in CLAUDE.md/memory; caught only by checking the DB row count against the filesystem instead of trusting the doc. **Rule adopted: verify corpus size against the DB/filesystem, never against CLAUDE.md or memory, before trusting a number** |
| DATA-7 | Gold-relevance labeling: 5-gram shingle containment (gold ∩ chunk / gold) ≥ 0.3, not exact numeric-answer match | Numeric match rejected: the dataset's `original_answer` can be a scaled table value with no literal string match to the source text. Threshold tuned on a 200-row sample: 0.5 gave clean separation on individual examples but a 28.5% zero-relevant-chunk rate overall, because gold context often spans two adjacent chunks and containment splits ~0.4-0.5 each. 0.3 drops the zero-hit rate to 3% |
| DATA-8 | Added numeric-literal overlap as a second gold-relevance signal, OR'd with shingle containment, threshold 0.5 | Root cause: the dataset repeats a number in two formats (e.g. `-17.1 ( 17.1 )`) while our table serialization has it once — the extra token shifts every 5-gram window straddling that number, so tables get systematically undercounted by pure shingling. Confirmed on 2 cases (ZBH_2017, ETR_2008); a broader audit flagged ~10 distinct filings affected. **Partial fix** — some flagged cases were not rescued, logged as open rather than chased further |
| DATA-9 | Gold-labeling audit (80 hand-read dev questions) → adopted IDF-weighted overlap + a minimum-evidence floor (`MIN_NUMERIC_EVIDENCE=3`) + evidence-size reporting into `gold_relevant_chunk_ids()` | Labeler recall was already ~93%; the real problem was precision — two confirmed false-positive-flooding bugs: (a) the numeric fallback's denominator could be 1-2 unique numbers, so any chunk sharing one recurring boilerplate figure passed trivially; (b) the numeric regex only matched decimals, so whole-integer tables got no rescue at all. IDF-weighting down-weights boilerplate figures that recur across a filing's own chunks; the evidence floor stops a single shared number from being decisive. Rejected an alternative (score adjacent-chunk-pair unions to catch tables split from their lead-in sentence) — mathematically guaranteed to add false positives to every already-good match, since a pair's union score is never lower than its stronger half. Validated on 300 dev questions: zero-hit questions 2→0, no zero-hit regressions. **Open gap**: table/chunk-boundary splits still aren't caught — needs a structural per-chunk table-density signal, not an overlap score |

## Chunking

| # | decision | why |
|---|---|---|
| CHUNK-1 | Parser: `sec-parser` (alphanome-ai, pinned `==0.58.1`), not Docling | Docling shatters prose into per-phrase fragments, leaks page-number/ToC noise into text, and mangles tables (eyeballed on AAL/JPM/ETR). `sec-parser` fixes the first two and parses ~3x faster (1.3s vs 4s/filing), but ships only 10-Q support — worked around via undocumented internals to reuse it for 10-Ks, so pinned exactly since those internals may break on upgrade. Known gap: `find_item_boundaries` returns nothing for filers using narrative headings instead of "Item N" (e.g. JPM) — accepted, not blocking. Fallback path if this ever breaks: Docling + a normalization pass (merge `<font>` runs, strip page/ToC noise) |
| CHUNK-2 | Chunk bounds 200/900/1500 tokens (min/target/max), token-counted via BGE-M3's tokenizer, not characters | Bounds come from T²-RAGBench's own evidence-span lengths (median 630 words, p90 990), not a generic RAG default. Token-counting matters because chars/token ratio measured at 4.1-4.5 for prose vs. 2.6-2.9 for tables — a character budget would silently over/under-pack depending on content mix. Tables are never split mid-row; an oversized table splits by row-group with the header row repeated in each group |
| CHUNK-3 | Two chunking-bug fixes found in the 100-filing dry run | (a) Filers lacking body-level "Item N" headings (`CHUNK-1`'s gap) produced single unstructured blocks up to 18k tokens — over BGE-M3's 8192 hard limit, i.e. actually unembeddable. Fixed with sentence-boundary packing + a raw-token-window fallback splitter. (b) `sec-parser` sometimes misclassifies a long bold paragraph (exhibit-index entries, signature blocks) as a heading — every title >100 tokens across all 100 filings was checked by hand (16 total, none genuine) and reclassified as body text. Empirical for this sample, not a proven rule; failure mode is soft (a real long heading would just lose grouping) |

## Arm 1 — dense retrieval

| # | decision | why |
|---|---|---|
| ARM1-1 | Embedding: BGE-M3 dense vector via `sentence-transformers`, not `FlagEmbedding` | `FlagEmbedding`'s `BGEM3FlagModel` also computes sparse + multi-vector (ColBERT-style) representations in one pass — not needed for a dense-only baseline. Both libraries wrap identical model weights, so the dense vector doesn't change if a later arm adds `FlagEmbedding` for sparse. 1024-dim, cosine distance, pgvector HNSW index |
| ARM1-2 | Retrieval: pure semantic search over the whole corpus, no `cik`+year metadata pre-filter | A pre-filter would raise recall by construction now that `DATA-5` makes that lookup reliable — but Arm 1 is the fixed baseline every later arm is compared against, and pre-filtering would change what's being measured, not just how well. Logged as a candidate variant for later, not blended into the baseline |
| ARM1-3 (open) | BGE-M3 uses single-`[CLS]` pooling (`sentence-transformers` default), not BAAI's MCLS long-document pooling (only exposed via `FlagEmbedding`, skipped per `ARM1-1`) | A chunk-length slice of Arm 1's results shows a real gradient — recall@10 0.50 (short chunks) vs 0.41 (medium) vs 0.26 (long) — consistent with a long-document pooling weakness. Correlational, not proof: long chunks overrepresent filers hit by `CHUNK-3`'s bug, and the labeling threshold wasn't separately validated per length bucket. Logged as a candidate to re-check with `FlagEmbedding` + MCLS before adopting |

## Arm 2 — hybrid (dense + BM25)

| # | decision | why |
|---|---|---|
| ARM2-1 | Fusion: RRF (Cormack et al., 2009), `k=60`, top-50 from each list | Cosine similarity and BM25 scores live on unrelated, incomparable scales — summing them directly would let whichever happens to produce bigger numbers dominate regardless of actual relevance. RRF fuses by rank instead |

## Arm 3 — reranker

| # | decision | why |
|---|---|---|
| ARM3-1 | Reranker: `bge-reranker-v2-m3` (local cross-encoder) over Arm 2's fused top-50, not Qwen3-Reranker-0.6B | Checked current state (Aug 2026): Qwen3-Reranker edges out on BEIR (~59 vs ~54) but is a causal yes/no-logit architecture — autoregressive decode per candidate. `bge-reranker-v2-m3` is a plain bidirectional cross-encoder: one forward pass, one score, and spec.md asks us to measure nDCG gain *against* latency cost, so the cleaner latency story won out over the leaderboard number. Cohere Rerank (hosted) deliberately deferred to conserve free-tier call budget |
| ARM3-2 | Reranking run offloaded to Northeastern's HPC (SLURM, V100) via a split-job pattern, not laptop CPU or a live SSH tunnel | Laptop CPU run swap-thrashed under normal load (~110s/query plateau, ~25h projected — not viable). A live tunnel from the GPU node to the laptop's Postgres was rejected: a dropped connection kills the whole run, same failure mode as the CPU run but now burning a SLURM allocation too. Split-job instead: laptop does the cheap retrieval + dumps a payload file; a self-contained, checkpointed GPU script reranks with no DB/network dependency; laptop scores the results after. File transfers must go through `xfer.discovery.neu.edu` — the interactive login node throttles/kills large transfers |
| ARM3-3 | Added cross-question batching to the HPC rerank script (20 questions' pairs per `predict()` call) | Only ~12% speedup (5.07s→4.45s) — GPU compute time, not Python call overhead, was the dominant cost, so batching couldn't buy the order-of-magnitude win hoped for. Adopted anyway since it's free |
| ARM3-4 (open) | Reranker introduces a same-company-wrong-fiscal-year failure mode not seen in Arms 1/2 | E.g. a JPM_2008 question's top hits are JPM_2009/2010 chunks; consecutive 10-Ks from the same filer share near-identical boilerplate, and a cross-encoder trained on general relevance has no inherent signal that the year token is decisive. Candidate fix: metadata-aware reranking or a `cik`+year filter — not yet built |
