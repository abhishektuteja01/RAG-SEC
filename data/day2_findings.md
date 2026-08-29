# Day 2 findings — where the parser breaks

Eyeballed 100 Docling-parsed 10-Ks (`data/parsed_docling_v1/`) and the current
sec-parser output (`data/parsed/`). Full context: `DECISIONS.md` CHUNK-1, CHUNK-2. This
list drives Day 6.

## Fixed by switching to sec-parser (see CHUNK-1)

1. **Font-span fragmentation.** Filing-agent HTML wraps near-every phrase in
   its own `<font>` tag; Docling split each into its own paragraph
   (~17% of lines in `ETR_2017`). sec-parser merges these correctly.
2. **Page-number / running-header leakage.** Standalone page numbers and
   repeated ToC headers leaked into Docling's text flow (29-31/filing).
   sec-parser tags these as `PageNumberElement`/`PageHeaderElement`, filtered.

## Not fixed by either parser — source-data defects

3. **Split negative numbers in table cells.** e.g. `(619` / `)` land in
   separate `<td>`s in the *source* HTML. Reproduced identically by Docling
   and `pandas.read_html`. No parser fixes this; would need a cell-merge
   normalization pass if it matters later.

## Known gaps in the current (sec-parser) pipeline

4. **No native 10-K support.** sec-parser ships only `Edgar10QParser`;
   10-K support is a workaround via undocumented internals. Fragile to a
   version bump — pinned exact (`==0.58.1`), fallback is Docling +
   normalization (see CHUNK-1).
5. **Item-boundary detection fails for ~35/100 filings.** Some filers (e.g.
   JPM) never restate "Item N" as a body heading — they use narrative
   headings instead ("EXECUTIVE OVERVIEW"). `find_item_boundaries()` returns
   nothing for these. Section *structure* still exists (headings are there),
   just not labeled by Item number.
6. **Same gap caused a real chunking bug (see #10, fixed):** those filers'
   unstructured text merges into single oversized blocks (one hit 18k
   tokens, over BGE-M3's 8192 limit). Fixed via sentence-boundary splitting
   in `chunking.py` — flagging here since it's the same root cause as #5.
7. **Title misclassification.** sec-parser occasionally tags a long
   bold-formatted paragraph (exhibit-index entries, cross-refs, signature
   blocks) as a heading instead of body text. Checked all 100 filings —
   16 titles over 100 tokens, none legitimate. Reclassified via
   `MAX_TITLE_TOKENS` threshold in `chunking.py`; empirical for this sample,
   not a proven law (see CHUNK-3).
