# Day 9's premise is false: T²-RAGBench has no multi-document questions

The Day 9 plan asked for Arm 6 measured "specifically on the multi-document questions where
Fin-RATE says one-shot retrieval falls apart." **That subset is empty.** Every question in
this benchmark is answerable from a single filing, so the agentic loop's published best case
— multi-hop composition across documents — cannot be tested here at all.

This is the write-up of that negative, and of what the Day 9 run therefore does and does not
establish. Verified 2026-09-04.

---

## 1. The claim, and the check

Every question in every split maps to exactly one filing. Not "usually one" — one, with zero
exceptions across all 11,739 corpus-matched questions.

Run (free, no GPU, no DB, no API — reads the HF dataset plus the on-disk chunk index):

```bash
uv run python -c "
from rag_sec.eval import load_matched_questions
df = load_matched_questions()
print('total matched rows:', len(df))
print(df['split'].value_counts().to_dict())
for sp in ['dev','test','train']:
    d = df[df['split']==sp]
    g = d.groupby('id')[['company_cik','report_year','chunk_file']].nunique()
    print(sp, 'n_rows', len(d), 'unique ids', d['id'].nunique(),
          '| max distinct cik/id', int(g['company_cik'].max()),
          '| max distinct year/id', int(g['report_year'].max()),
          '| max distinct chunk_file/id', int(g['chunk_file'].max()),
          '| ids with >1 filing:', int((g['chunk_file']>1).sum()))
"
```

Returned:

```
total matched rows: 11739
{'train': 8958, 'test': 1546, 'dev': 1235}
dev   n_rows 1235 unique ids 1235 | max distinct cik/id 1 | max distinct year/id 1 | max distinct chunk_file/id 1 | ids with >1 filing: 0
test  n_rows 1546 unique ids 1546 | max distinct cik/id 1 | max distinct year/id 1 | max distinct chunk_file/id 1 | ids with >1 filing: 0
train n_rows 8958 unique ids 8958 | max distinct cik/id 1 | max distinct year/id 1 | max distinct chunk_file/id 1 | ids with >1 filing: 0
```

**Independently reproduces the project's 1235 dev / 1546 test split exactly.** Two things the
original statement did not say, both of which matter:

- **Train is the same** (8,958 rows, 0 multi-filing). There is no rescue by widening the
  sampling pool — the property is the dataset's schema, not the split's.
- `page_number` is single-valued per question id in dev and test (`max distinct page/id = 1`).
  The evidence isn't merely one *filing*, it is **one page of one filing** — which is
  `DATA-4`'s point arriving from the other direction. `DATA-4` recorded that scoring against
  the shipped single-page context inflates recall; the same single-page annotation is what
  makes multi-hop untestable.

**Why it is structural, not an artifact of our corpus matching.** `load_matched_questions`
already drops questions whose filing we never ingested, so this is a lower bound on the
restriction: each T²-RAGBench row carries exactly one `(company_cik, report_year)` and one
`page_number`, because the dataset was built by annotating one page per question. No filter
over these questions can produce a multi-document subset, because the property is absent from
the schema, not thinly represented in the data.

## 2. Why "multi-year" questions are not multi-document

The tempting objection is that plenty of these questions *look* multi-hop. They do:

| split | questions naming ≥2 distinct 4-digit years in the question text |
|---|---|
| dev | 623 / 1235 = **50.4%** |
| test | 804 / 1546 = **52.0%** |

Measured with `re.finditer(r'\b(19|20)\d{2}\b')` over the question text on the same DataFrame
as §1. So *half* the benchmark is a cross-year comparison — and every one of them is still a
single-filing question.

**The reason is a property of 10-Ks, not of the dataset.** Financial statements print
prior-year comparatives: a 2007 10-K's income statement carries 2007, 2006 and often 2005 in
adjacent columns. "What was the change in revenue from 2006 to 2007" is a two-year question
whose two numbers sit in one table on one page. Composition across *time* is free; composition
across *documents* never arises.

This is the same fact that drives `RETR-3` from the other side. There, near-verbatim
year-over-year reprinting is why retrieval keeps returning the right company's wrong year
(50% of 100 hand-read failures). Here, the same reprinting is why the question never needs a
second filing. One corpus property, two consequences: retrieval is *harder* than it looks and
reasoning is *easier* than it looks.

## 3. Why this sharpens `COST-30` rather than softening it

`COST-30` declined to merge `judge` into `answer` and, more importantly, questioned whether
the loop belongs on single-document filings at all. It rested on three findings and then
deliberately left one door open:

> **Not deleted outright:** iteration's published wins are multi-hop composition across
> documents, which is Day 9's question, not Day 8's.

That door is now closed, and not in iteration's favour. `COST-30` deferred the loop's
strongest case to a test that **does not exist on this benchmark**. So on T²-RAGBench:

- The only regime where the loop is expected to win is unrepresented — 0 of 2,781 dev+test
  questions.
- What *is* represented is the regime `COST-30` cited CRAG losing in: recall@5 **0.658** for a
  corrective loop against hybrid+rerank's **0.816**, measured on T²-RAGBench itself
  (arXiv:2604.01733).
- So Arm 6's expected value here is not "unproven" — it is *negative*, and the deferral was
  the last thing holding the question open.

The honest interview framing: **the loop is not wrong, the benchmark is out of scope for it.**
An agentic retrieval loop is a bet that one query cannot reach all the evidence. On a corpus
where every answer lives on one annotated page, that bet has nothing to pay off against — and
`OBS-10` already measured the price of taking it anyway (3.6x the static arm's cost, n=3, do
not quote as a result).

## 4. This is not the `multi_doc` flag already in the repo

There is exactly one `multi_doc` in the codebase, and it measures a different thing.
`scripts/archive/pack_variants.py:287`:

```python
stems = {u["chunk"][0] for u in (units[i] for i in kept)}
s["multi_doc"] += len(stems) > 1
```

`stems` are the filing stems of the **retrieved and kept slices**. So that flag asks *did the
packed evidence block draw from more than one filing* — a property of the **result**, and
originally a symptom, not a target: it exists to quantify `RETR-3`'s contamination, the
finding that retrieved chunks span sibling years of the same filer.

The two numbers point in opposite directions and both are correct:

| | value | what it means |
|---|---|---|
| `pack_variants` `multi_doc`, `base` cell | **293 / 300 = 97.7%** (`data/day8_pack_variants.json`) | 98% of *retrieved evidence blocks* span >1 filing |
| questions mapping to >1 filing (§1) | **0 / 1235 dev, 0 / 1546 test** | 0% of *questions* need >1 filing |

Read together they are a single statement: **~98% of the time the retriever pulls chunks from
filings the question never needed.** The companion counter in the same loop,
`same_co_diff_year` (292/300 in `base`), says the extra filings are overwhelmingly the same
company's other years. Anyone reaching for `multi_doc` to build Day 9's subset would be
selecting on retrieval error, and would produce a subset where the loop is being asked to
recover from contamination rather than to compose across documents.

## 5. What a benchmark that could test this would need

Concretely, and stated as requirements rather than aspirations:

1. **Gold evidence annotated in ≥2 distinct `(cik, report_year)` filings per question**, with
   the per-filing spans labelled separately. Without per-filing spans you cannot tell partial
   from complete evidence, which is exactly the state the loop's judge has to grade.
2. **Answers that are not derivable from any single filing.** The trap is the comparatives
   column of §2: a question spanning FY2005 and FY2007 is single-filing if a 2007 10-K prints
   a three-year table. The test has to be mechanical — verify that no one filing in the corpus
   contains all gold spans — not "the question mentions two years."
3. **Cross-*entity* or cross-*form* composition, not just cross-year**, to escape the
   comparatives problem entirely: two filers in one question (peer comparison), or a 10-K plus
   a later 8-K/10-Q that restates it.
4. **A hop count per question**, so results can be reported as accuracy vs. hops. A single
   aggregate over a mixed set hides the effect the loop is supposed to have; `COST-30`'s CRAG
   comparison is aggregate and that is a real weakness of it as evidence.
5. **A matched single-hop control set** from the same filings and the same annotator, so the
   loop-vs-static delta is attributable to hop count and not to question difficulty.
6. **Distractor siblings deliberately included** — the adjacent fiscal years of each cited
   filer. Otherwise a company filter alone solves the task (`RETR-39`: +0.140 on test from
   filter + strip) and the benchmark measures entity resolution, not composition.

Building this is real annotation scope the plan never budgeted, and composing it
automatically from FinQA pairs would produce questions no analyst would ask. **It is listed
here as the specification of the gap, not as queued work.**

## 6. What Day 9's run does and does not establish

The decision taken was: **run the trajectory half anyway and report the negative.** That is a
defensible use of ~$11.5 because the Day 9 plan asks for two things and only the first is
blocked.

**Does not establish** — anything about the loop's headline hypothesis. Not "the loop doesn't
help on multi-hop questions"; **"this benchmark cannot ask the question."** Any accuracy
comparison from this run is single-document by construction, so a loop loss here is
*consistent* with the published multi-hop wins and is not evidence against them. It also
cannot be fixed by re-running, by sampling differently, or by spending more.

**Does establish**, all of it single-document and all of it what the Day 9 plan asks for after
"Full trajectory metrics":

- **Calls, tokens, dollars, wall clock** per question, and the loop-vs-static cost multiple on
  a real n rather than `OBS-10`'s n=3 — including where the money goes per node and how much
  of it prefix caching returns (`AGENT-8`).
- **Sufficiency-judge accuracy**: how often `judge` says "insufficient" on evidence that
  already contains the gold. This is the measurement that generalises off this benchmark,
  because a judge that mis-grades complete evidence is broken in *any* regime — and it is
  `COST-30`'s third finding (judges discarding answers that were already right) tested on our
  own system instead of cited from Joren et al.
- **Whether iteration adds evidence at all.** Every iteration's top-10 and pre-rerank top-50
  are stored, so "did iterations 2-4 surface gold that iteration 1 missed" is answerable
  per question. On single-document questions the expected answer is no, and `OBS-10` saw no on
  2 of 2. **This is the sharpest thing the run can produce**: if the loop cannot improve
  evidence even when it re-queries, the failure is in the loop's *query reformulation*, which
  is a mechanism finding and not a benchmark-scope one.
- **The paired comparison is clean.** `static_baseline` answers over the published `RETR-39`
  ranking (`data/retr7_rr_dev_scores.jsonl:filtered_stripped`), so retrieval is not re-run and
  the two arms differ only in the loop.

**Caveat to state first, always:** n≈200 dev questions, one split, single-document only, and
the run is scored by `agent_analyze.py` (now `scripts/pipeline/07_arm6_loop.py analyze`). Nothing here is a test-split number.

## 7. If asked "so what should you have done"

Checked the question-to-filing cardinality before building Arm 6, not after — it is the
one-line query in §1 and it costs nothing. The reason it wasn't checked is instructive and
worth saying out loud: the plan asserted the multi-document subset existed, citing Fin-RATE,
and an assertion in the plan document was treated as a property of the data. That is the same
failure mode this project keeps hitting with numbers — **a value correct in one context, silently
wrong in the next** — arriving in a scope claim rather than a metric.
