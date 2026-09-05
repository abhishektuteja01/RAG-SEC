# Day 9 — diagnosis of the ten worst Arm 6 failures

**DRAFT — not reviewed by a human.** Written by reading `data/day9_worst_failures_arm6.md`,
`data/day9_worst_failures_static.md`, `data/day9_arm6_dev_results.jsonl` and the gold chunk
text in `data/chunks/`. Every number below was read off disk or computed from disk; inferences
are marked "inference".

## 1. Framing: "worst" is an ordering choice, not a measurement

`scripts/eval/worst_failures.py`'s docstring is explicit: ranking failures by wrongness is
meaningless here (a number either matches gold or it does not), so the file orders by **how
close the gold evidence got to the answer model** — `reasoning` (gold was in the top-k) before
`rerank` (gold was in the 50 candidates) before `retrieval` (gold never fetched), then best
gold rank, then iterations, then id.

Consequences for reading this list:

- All ten are `reasoning`, because there are 41 of them and they sort first. This is **not**
  evidence that reasoning failures are worse per unit; it is the ordering doing its job of
  surfacing the failures nothing upstream can be blamed for.
- Ties inside `reasoning` break on gold rank, and every one of the ten has gold at rerank
  rank 1 — so the ten are literally "gold ranked first and the answer was still wrong".
- Any other defensible ordering yields a different ten. Nothing here supports a claim about
  the *frequency* of a cause; frequencies come from the whole-run counts in §4.

Run-level context used but not re-derived: 200 questions, 0 errors; 53 arm6 failures
(reasoning 41, rerank 3, retrieval 9) vs 66 static (42 / 3 / 21); answer accuracy 69.2% vs
61.6% on 172 scoreable; McNemar exact p = 0.00098.

## 2. The ten worst, one at a time

### 1. `finqa_dev_672` — reasoning — gold rank 1, 4 iterations (hit cap)

Asked: what percentage of the $1.4M total interest expense for unpaid taxes is attributed to
the $23,743k of unrecognized tax benefits (FIS 2007). Gold `0.1667` / `16.7%`, gold chunk
`FIS_2007_1136893#76`. Model answered INSUFFICIENT. Judge said `loop` all four times; the loop
stopped only on the iteration cap.

The gold chunk does contain both numbers: it has the UTB rollforward ending `$23,743`, the
sentence "total amount of interest expense recognized … for unpaid taxes is $1.4 million", and
"total amount of interest and penalties recognized in the consolidated balance sheet is
$8.4 million". **1.4 / 8.4 = 0.1667 = gold.** So the gold is 1.4/8.4 — a ratio that has nothing
to do with the $23,743 the question names. Diagnosis: the *question text misdescribes its own
gold program*. The model correctly observed that no split of the $1.4M by UTB is disclosed and
refused. This is a bad-question failure, not a reasoning failure; the static arm refused for the
same reason.

### 2. `finqa_dev_196` — reasoning — gold rank 1, 3 iterations

Asked: what percentage of Adobe's $80.9M fiscal-2018 valuation-allowance change was due to
settlements with taxing authorities. Gold `0.0` / `0`. Gold chunks `#54, #89, #90`; `#90` and
`#89` reached the answer model, `#54` did not. Judge: loop, loop, finish. Model answered
INSUFFICIENT, explicitly noting that "settlements with taxing authorities" appears in the
*unrecognized tax benefits* reconciliation, not the valuation-allowance movement.

`#90` shows `Settlements with taxing authorities | — | (3,876)` — the 2018 cell is a dash, i.e.
zero. The gold takes the naive reading (0 / 80.9 = 0%); the model spotted that the question
mixes two different tables and declined. Diagnosis: **model refuses on a question whose premise
is sloppy, where the gold assumes the naive reading.** Same failure in the static arm.

### 3. `finqa_dev_490` — reasoning — gold rank 1, 3 iterations — **gold appears wrong**

Asked: percentage change in American Tower's amortization expense from 2007 to 2008 per the
2004 statements. Gold `0.0424` / `4.2%`, gold chunk `AMT_2004_1053507#58`. Model answered
-1.63%. Judge: loop, loop, finish.

The chunk reads: "expects to record amortization expense of approximately $97.8 million,
$95.9 million, $92.0 million, $90.5 million and $88.8 million … for the years ended December 31,
2005, 2006, 2007, 2008 and 2009". So 2007 = 92.0 and 2008 = 90.5, and (90.5-92.0)/92.0 =
-1.63% — **the model's arithmetic and column mapping are both correct**. The gold equals
(95.9-92.0)/92.0 = 0.0423913043478… exactly, i.e. it pairs 92.0 with 95.9, which is the *2006*
figure. Diagnosis: gold-label column misalignment. Caveat: I only see our chunk text, not the
original FinQA table, so I cannot rule out the source table being ordered differently
(inference).

Loop note: static refused this one outright; the loop's iteration 2/3 rewrites are what pulled
the gold chunk in. The loop did its job and the scorer marked it wrong.

### 4. `finqa_dev_247` — reasoning — gold rank 1, 2 iterations — **scorer artifact**

Asked (yes/no): did UNP's 2011 dividends payable exceed interest payable. Gold `1.0` / `yes`,
gold chunks `#65, #66, #67`; `#66` — the Note 12 table with `Dividends payable 284` and
`Interest payable 197` — reached the model. Judge: loop, finish. Model answered "ANSWER: Yes"
with both figures correct.

Scored wrong because `answer_eval.parse_reason` requires a *number* on the ANSWER line. The
model is right; the scorer cannot express it. All 4 scoreable yes/no questions in the run fail
this way in both arms (§4). Static refused entirely, so the loop actually fixed the retrieval
here and got no credit.

### 5. `finqa_dev_367` — reasoning — gold rank 1, 2 iterations

Asked: what percentage of Entergy's 2014 long-term debt maturities is Entergy Louisiana lease
obligations. Gold `0.3866` / `38.7%`, gold chunk `ETR_2013_65984#115`. Model answered
INSUFFICIENT. Judge: loop, finish.

Both operands are in that one chunk but in different registers: footnote (e) says "excludes
lease obligations of $149 million at Entergy Louisiana", and the maturity table says
`2014 | $385,373` **(In Thousands)**. 149,000/385,373 = 0.3866 = gold. Diagnosis: the answer
needed (a) a cross-reference from a fair-value footnote to a maturities table, (b) the
recognition that a *fair-value exclusion* stands in for a *2014 maturity* — which is arguably
wrong finance — and (c) a millions→thousands unit conversion. The model's refusal is defensible;
the gold's chain is not obviously sound.

### 6. `convfinqa_2300` — reasoning — gold rank 1, 1 iteration

Asked: difference in Centene's diluted EPS between 2001 and 2002. Gold `0.48`, gold chunk
`CNC_2003_1071739#43`. Model answered 0.40 from $1.47 (2002) and $1.07 (2001).

`#43` is a **pro forma** acquisition table: `Diluted earnings per common share | 1.48 | 1.00`
for 2002 | 2001 → 0.48 = gold. The model used the *reported* EPS from elsewhere in the same
filing. Diagnosis: two defensible tables, question does not say which; gold silently means the
pro forma one. Chunk-selection ambiguity, not arithmetic. The static arm produced the same 0.40
and even flagged a third split-adjusted reading.

### 7. `convfinqa_2666` — reasoning — gold rank 1, 1 iteration — **gold looks unit-broken**

Asked: redemption value of Kimco's Preferred A units at 12/31/2010. Gold `4,840,000`, gold
chunk `KIM_2010_879101#77`. Model answered INSUFFICIENT.

The chunk's only Preferred A row is `Preferred A Units | 2,200,000 units redeemed | $2.2 par
value redeemed (in millions)`. 2,200,000 × 2.2 = 4,840,000 — units multiplied by a
par-value-in-millions (inference, but the arithmetic is exact and no other number in the chunk
produces 4,840,000). The chunk contains no quantity that is a "redemption value of Preferred A
units as of December 31, 2010". Diagnosis: gold is a unit-confused product; the refusal is
right. Static refused identically.

### 8. `finqa_dev_147` — reasoning — gold rank 1, 1 iteration — **gold appears wrong**

Asked: total return on BLL common stock over the five years ending 12/31/2010, $100 invested
12/31/2005. Gold `34.96`, gold chunk `BLL_2010_9389#12`. Model answered 78.93%.

The chunk's Total Return Analysis row: `Ball Corporation | $100.00 | $110.86 | $115.36 |
$107.58 | $134.96 | $178.93` for 12/31/05 … 12/31/10. The value at the period end is $178.93 →
78.93% return, which the model computed correctly. Gold `34.96` is $134.96 − $100, i.e. the
**12/31/09 column**. Same off-by-one-column shape as `finqa_dev_490`; same caveat that I cannot
see the original FinQA table (inference). Note `finqa_dev_551` (rank 17) reads the same chunk
and answers correctly — it fails only on the yes/no parse.

### 9. `finqa_dev_170` — reasoning — gold rank 1, 1 iteration — **tolerance near-miss**

Asked: EMEA share of BlackRock's 2012 long-term retail/HNW AUM. Gold `0.19257` / `19.3%`, gold
chunks `#17` and `#18`, both delivered. Model answered 19% — quoting the filing's own narrative
sentence ("19% managed for investors based in EMEA") instead of dividing 77,699 / 403,484.

|19 − 19.257| / 19.257 = 1.3%, over the scorer's 1% relative tolerance. Diagnosis: the model
preferred a rounded narrative figure to the table computation. Interesting contrast: the static
arm *did* compute 19.26% in its working and still wrote `ANSWER: 19%`, so this is answer-line
formatting, not an inability to compute.

### 10. `finqa_dev_209` — reasoning — gold rank 1, 1 iteration — **class label is optimistic**

Asked: unfunded commitments as a percentage of PNC's total equity investment balance at
12/31/2012. Gold `0.06298` / `6.3%`, gold chunks `#33, #123, #124`. Only `#124` reached the
model. Model answered INSUFFICIENT.

`#124` carries the numerator ($685M unfunded at 12/31/2012). The denominator — `Total | $10,877`
in Table 55 — is in `#123`, which was never delivered; `#33` (Selected Financial Data, same
10,877) was not either. 685/10,877 = 0.06298 = gold. Diagnosis: **the refusal was correct
behaviour on the evidence supplied.** The `reasoning` label fires because *any one* gold chunk
in the top-k qualifies, so a partially-delivered multi-chunk gold gets filed as an answer-stage
failure. The static arm, given the same partial evidence, guessed 685/3,000 = 22.83% — wrong,
and worse than the loop's honest refusal.

## 3. Cross-cutting patterns among the ten

| Cause | ids | n |
|---|---|---|
| Gold label is wrong or unsound | 490, 147, 2666, 367, 672 | 5 |
| Question/gold ambiguity the model resolved differently | 196, 2300 | 2 |
| Scorer cannot express a correct answer (yes/no, rounding) | 247, 170 | 2 |
| Evidence genuinely incomplete, refusal correct | 209 | 1 |
| Model arithmetic/units genuinely wrong | — | 0 |

**None of the ten is an arithmetic error.** Every calculation the model showed is internally
correct (490, 147, 2300, 170 all check out against the chunk text). The candidate hypotheses
in the brief fare as follows:

- *Arithmetic across table cells* — not observed in these ten, and the two nearest candidates
  just outside them are also gold problems, checked against the chunks: `finqa_dev_286`
  (rank 11) — the model computed the share-weighted average the question explicitly defines
  ("dividing the aggregate value … by the aggregate number of shares") and got 3.79; gold 3.61
  is the *simple* mean (3.24+3.98)/2. `finqa_dev_610` (rank 20) — the model computed
  7,498/21,430 = 34.99% from the December 31, **2014** column as asked; gold 0.493680008870163
  is exactly 8,905/18,038, the December 31, **2013** column.
- *Unit / scale errors* — present, but in the **gold** (2666) and as a required conversion the
  model refused to make (367), not as a model mistake.
- *Wrong year's column* — present, but again in the **gold**, not the model: 490 and 147 in the
  ten, plus 610 just outside it. Three instances, all the same shape (gold reads one column to
  the left of the year the question names). That is now a hypothesis worth a systematic check
  across the dev split, not an anecdote.
- *Answer format / parse* — real and clean: 247 (yes/no), 170 (rounded narrative figure).
- *Genuinely ambiguous gold* — the single biggest bucket here: 672, 196, 2300, 367, 2666.

A behavioural pattern cuts across these: the answer model is **conservative**. Six of the ten
refused or under-committed rather than assert a number it could not source (672, 196, 367,
2666, 209, and 170's rounding). Run-wide this shows up as 19 of 53 arm6 failures being explicit
refusals, versus 29 of 66 static — the loop refuses *less* than static in absolute terms, and
209 shows a case where its refusal beat static's guess.

Whole-run numbers computed from `day9_arm6_dev_results.jsonl` to check whether the ten
generalise:

- 53 arm6 failures = 19 `refused` + 4 `no_number` + 30 wrong number. Static: 29 + 3 + 34.
- All 4 `no_number` failures are the 4 yes/no questions in the scoreable set — **100% of yes/no
  questions fail in both arms for a scorer reason**, not a model reason.
- Of the 30 wrong-number failures, 5 match gold in magnitude but not sign (`finqa_dev_332`,
  `_482`, `_754`, `convfinqa_853`, `_615` — "percentage decline" phrasing), and 4 are within
  10% of gold (`_170` 1.3%, `convfinqa_2125` 1.8%, `_286` 5%, `_609` 9.4%).
- 10 of the failures have multi-chunk gold; in **8 of those the top-k carried only part of the
  gold set** (`_609, _631, convfinqa_2131, _233, _196, _630, _247, _209`). Partial does not
  always mean insufficient (196 and 247 had what they needed), but the `reasoning` label
  overstates the answer stage's culpability in this group.

So: **~9 of 53 arm6 failures (4 yes/no + 5 sign-only) are scoring-convention artifacts**, and at
least 5 of the ten worst are gold-label problems. The headline 69.2% is therefore a floor, not
an estimate of what the pipeline can answer.

## 4. Actionable vs not

Honest caveat first: n = 10 of 53, chosen by a rule that deliberately over-samples one class.
Everything below is a hypothesis to test on the full 53 (and then on test), not a conclusion.

**Actionable, cheap, no re-run needed** (all are scorer/prompt changes, replayable against the
stored `final_answer` text):

1. **Score yes/no gold as yes/no.** `answer_eval` requires a number; gold `original_answer` is
   the literal string `yes`. Accepting yes/no when the gold field is yes/no recovers 4 failures
   in *both* arms at zero inference cost. Highest value per unit of work in this list.
2. **Decide the sign convention for "decline/decrease" questions**, or score on magnitude when
   the question word implies direction. 5 more failures.
3. **Make the answer prompt say "give the computed value, not a figure quoted from prose"** —
   170's exact failure, and the static arm proves the model already had the computation.

**Actionable but requires judgment, not code:**

4. **Audit the gold for 490, 147, 2666, 672 — and 286, 610 just outside the ten.** If these are
   label errors, they belong in an
   excluded set with a written reason, the way COST-21 already excludes irreconcilable gold —
   not silently counted against the model. Do **not** delete them on my say-so: I compared the
   gold to *our chunk text*, not to the FinQA source table.
5. **Deliver the whole gold set, not one chunk of it.** 209 fails purely because the Table 55
   denominator was in a sibling chunk. This is a chunking/neighbour-expansion question (fetch
   adjacent chunk indices from the same filing), and it is the one *pipeline* fix these ten
   support.

**Not actionable from this evidence:**

- Nothing here justifies touching the answer model, its thinking level, or the reranker. Gold
  was at rank 1 in all ten; reranking is not the constraint.
- Nothing here justifies changing the loop policy. See §5.
- The "arithmetic across table cells" hypothesis is **not** supported by these ten and needs the
  other 31 reasoning failures before anyone acts on it. Day 6's arithmetic-gap analysis stays
  deferred.

## 5. Did looping ever actively hurt?

**No — and it cannot, by construction.** `AgentState.retrieved_chunks` is
`Annotated[list[dict], operator.add]` (`src/rag_sec/agent.py:37`) and `answer_node` answers over
the deduped **union of every iteration's chunks**. A later iteration can add noise; it cannot
displace anything iteration 1 found.

This matters when reading `day9_worst_failures_arm6.md`, which prints top-k per iteration and so
*looks* like displacement: `finqa_dev_672` shows gold at rank 1/2/5 in iterations 1-3 and absent
in iteration 4, and `finqa_dev_196`'s iteration 3 returns no ADBE chunks at all. Verified
against the rows: the gold chunk is still in the union handed to the answer node in both cases.

What looping did cost, on these ten:

- Dilution. Unique chunks reaching the answer model: 672 → 28, 196 → 30, 490 → 22, 247 → 18,
  367 → 18; the five single-iteration cases → 10 each. Roughly 3x the evidence for the looped
  ones, most of it off-target (672's iteration 4 returned DG, EW, RSG, MO, KHC chunks for an FIS
  question). No case among the ten where that extra context demonstrably flipped a right answer
  to a wrong one — but n = 5 looped cases, so this is untested, not disproved.
- Latency: 372s (672, 4 iterations) and 285s (490) versus ~73-103s for the single-iteration ones.

Where looping *helped* among the ten: 490, 247 and 367 were all `INSUFFICIENT` in the static arm
because static retrieval missed the gold entirely; the loop's rewritten queries fetched it. Two
of those three (490, 247) then lost points to a gold error and a scorer limitation rather than to
the pipeline.

## 6. What I could not determine

- Whether the gold labels for 490, 147 and 2666 are wrong in the *dataset*, or wrong only
  relative to our chunk text. Requires the FinQA source tables.
- Whether extra iterations ever flip a correct answer to an incorrect one. It did not happen in
  these ten, but the discordant-pair analysis (14 arm6-only vs 1 static-only) is the right place
  to look, not this file.
- Why `finqa_dev_672`'s judge said `insufficient` four times with the gold chunk at rank 1 in
  three of them. The judge is `gemini-3.1-flash-lite/minimal` and its raw output is the single
  word `insufficient`, with no reason — so the failure is unattributable from the stored trace.
- Whether the 31 reasoning failures outside this list are dominated by real arithmetic errors.
  This list cannot say; the ordering rule guarantees it sampled the gold-rank-1 end.
