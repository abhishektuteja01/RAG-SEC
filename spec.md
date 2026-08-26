# Project spec — agentic retrieval over SEC filings

A 14-day build (5 hours/day, ~70 hours) designed to convert the gaps in
`GAP_REPORT.md` from claims into evidence.

**One line:** an agentic retrieval service over SEC filings — six measured retrieval
strategies scored against public benchmarks, fully traced, gated in CI, deployed on
AWS, usable from Claude Code.

The deliverable is not the app. It is three things:

1. a **benchmark table** you produced yourself,
2. a **decisions log** explaining every choice in that table,
3. a **deployed, traced, CI-gated service** behind both.

---

## 1. Why this project

### What it closes

| gap | JD reach | how this project anchors it |
|---|---|---|
| rag | 50.4% | six measured arms, not a tutorial pipeline |
| aws | 38.5% | deployed service, not a notebook |
| observability | 34.4% | OpenTelemetry GenAI spans, used for real debugging |
| vector-db | 33.1% | pgvector, hybrid indexes, storage tradeoffs measured |
| function/tool-calling | 29.9% | LangGraph tool calls in the agent loop |
| docker | 26.1% | containerized from day 1 |
| langchain / langgraph | 23.9% / 20.8% | the agent loop and its checkpointing |
| embeddings | 22.9% | three embedding models compared on the same set |
| latency/perf | 21.2% | measured per arm, stated p95 budget |
| mcp | 20.3% | FastMCP server wrapper |
| drift-monitoring | 11.0% | CI regression gate on eval scores |
| chunking / hybrid-search / reranking | 6.3 / 5.2 / 4.9% | three separate measured arms |

Roughly **84% of AI-Engineer JDs and 86% of companies** in the corpus.

### Deliberately out of scope

- **Fine-tuning (20.9%)** — 15+ hours of learning, high risk of not finishing, and it
  demos badly. Skip.
- **Kubernetes (19.0%)** — same reasoning. One well-understood cloud deploy beats a
  half-learned orchestrator.
- **Azure (31.8%) / GCP (24.2%)** — one cloud, done properly. AWS has the highest demand.

Both fine-tuning and Kubernetes stay on the list for a *later* project. Trying to
cover everything in 70 hours produces a project that covers nothing convincingly.

### Why SEC filings and not something else

Three reasons, in order of importance.

**The labeled data already exists.** This is the decisive one. Hand-labeling a few
hundred questions would eat ~10 hours and produce numbers nobody can check. Instead:

| dataset | size | access | role in this project |
|---|---|---|---|
| **T²-RAGBench** | 23,088 context-independent Q/A pairs over 7,318 documents, text-and-table focused (EACL 2026) | **fully open** on Hugging Face at `G4KMU/t2-ragbench` | **Primary eval set.** Large enough for statistically meaningful gates, and its text-and-table focus lines up exactly with the Day 6 differentiator. |
| FinanceBench | 10,231 triples claimed in the paper — but **only 150 annotated examples are public** | public sample on HF at `PatronusAI/financebench`, **CC BY-NC-4.0**; full set is gated behind a licensing email to Patronus | **Secondary sanity set only.** 150 examples is far too small to gate on (see 2.3). Non-commercial license is fine for a portfolio project, but say so in the README. |
| FinQA / ConvFinQA / TAT-DQA | the three sources T²-RAGBench is derived from | open individually | Fallback if T²-RAGBench has problems, and useful for slicing by question type. |
| FinAgentBench | 18,000+ expert-curated agentic-retrieval samples | **availability unverified** — could not confirm an open download | Nice-to-have. Do not plan around it. Check on Day 1; if it's gated, drop it. |

**This was a correction.** The first draft of this spec treated FinanceBench's 10,231
triples as the primary eval set. They are not publicly available — the open release is
150 examples. T²-RAGBench is now primary. Verify all of this yourself on Day 1 anyway;
dataset access changes.

Because these are public, your numbers are comparable to published results. "My
hybrid+rerank config scores X on FinanceBench" is a real claim. "My config scored 0.8
on questions I wrote" is not.

**The hard part is genuinely hard, and independently known to be.** Fin-RATE tested 17
leading models and found accuracy dropped **14–19%** as soon as a question spanned
multiple documents or time periods rather than one. Tables inside filings break normal
chunking. This is live 2026 research, not a solved toy problem.

**The hiring signal is explicit.** The recurring line in 2026 hiring guides: *if a
candidate talks about agents for 30 minutes and never mentions trajectory evals,
ground-truth datasets, or regression harnesses, they have never shipped one.* This
project is built around exactly those three.

---

## 2. Evals — the backbone

This is the part that makes the project worth doing, so it gets specified first and
built before any tuning. **Day 3 builds the eval harness before Arm 1 is optimized.**
If you can't measure it, you're not engineering, you're decorating.

### 2.1 The metric question: is it precision and recall?

Recall yes. Precision, mostly no. Here is why.

**Retrieval layer — did we fetch the right paragraphs?**

| metric | what it means | why it's in (or out) |
|---|---|---|
| **recall@k** | of the paragraphs that genuinely contain the answer, what fraction landed in our top k | **The headline metric for Arms 1–2.** Generation can ignore junk it was handed, but it cannot invent a fact that was never retrieved. Recall is the ceiling on everything downstream. |
| **precision@k** | of the k paragraphs we fetched, what fraction were relevant | **Not a headline.** In RAG you almost always fetch a fixed k, so precision@k is mechanically tied to k and mostly re-states recall. Track it as a *cost* proxy — low precision means you're stuffing the context window with noise, burning tokens and confusing the model — but never optimize for it directly. |
| **nDCG@10** | rank-aware: rewards putting the right paragraph at position 1 rather than position 9 | **The headline metric for Arm 3 (reranking).** A reranker does not change *what* you retrieved, only the *order*. Recall@10 is blind to reordering within the top 10. nDCG is the only metric that can see a reranker working. |
| **MRR** | 1 / rank of the first relevant hit | Good single number for "did we nail the top result." Report alongside nDCG. |
| **recall@50 vs recall@10** | headroom check | If recall@50 is high and recall@10 is low, your problem is *ranking* and a reranker will help a lot. If recall@50 is also low, your problem is the *index* and reranking will do nothing. This diagnostic decides where your remaining hours go. |

**Answer layer — did we answer correctly?**

| metric | how |
|---|---|
| **numeric / exact match** | Finance is a gift here: most answers are numbers. Grade automatically with a tolerance (say 1% for derived figures, exact for stated ones). No LLM judge needed, no subjectivity, cheap to run in CI. This is a large practical advantage of this corpus over a prose corpus. |
| **faithfulness / groundedness** | Is every claim in the answer supported by a retrieved chunk? LLM-as-judge, on a sample — too expensive for every CI run. |
| **citation accuracy** | Does the chunk the answer cites actually contain the number it claims? Deterministic string/number check where possible. Catches the failure where the answer is right and the citation is invented. |
| **refusal correctness** | When the answer genuinely isn't in the corpus, does it say so instead of guessing? Build a small set of unanswerable questions on purpose. Most projects skip this and it's a strong interview talking point. |

**Agent layer — for the agentic arm only.** This is what hiring guides call *trajectory
evals*, and it's the least common thing in a portfolio project.

- retrieval calls per question (did the loop converge, or spin?)
- tokens and dollars per question
- wall-clock per question
- **sufficiency-judge accuracy**: when the agent decided "I have enough," was it right?
  Score its decision against whether the final answer was correct. A loop that stops too
  early is a different bug from a loop that never stops, and this metric separates them.

**Operational layer — every arm.**

- p50 and p95 latency, broken down per stage (parse → embed → search → rerank → generate)
- cost per query
- index size on disk, and build time

### 2.2 Rules that keep the numbers honest

These are the things that make the difference between a benchmark and a vanity table.

1. **Dev split and test split.** Tune on dev. Touch test **once**, at the end, per arm.
   If you tune against your test set, your numbers are fiction and an interviewer who
   knows the field will find it in one question.
2. **One change at a time.** Every arm differs from the previous arm in exactly one
   thing. That is what makes the deltas attributable. It's also the whole reason the
   table is persuasive.
3. **Same questions, every arm.** Paired comparison on an identical question set, fixed
   random seed, fixed model versions. Pin your model versions in config and write them
   into the results file — a silently updated hosted model invalidates every earlier row.
4. **No metric travels alone.** Every accuracy number is reported next to its latency and
   its cost. Without this rule the reranker always "wins" and the agentic loop always
   "wins," because you're only looking at the axis they win on. The interesting sentence
   is "it bought 11 points of nDCG for 180ms and 1.4x the tokens" — that is an
   engineering judgment. "It was better" is not.
5. **Log the failures, not just the scores.** Keep the 20 worst failures per arm in a
   file. These are your best interview material: a specific question, what got
   retrieved, why it was wrong. Nobody else brings that.
6. **State what you did not measure.** If you sampled, say so. If an arm ran on a
   subset, say which. A benchmark with stated limits reads as competent; one without
   reads as unexamined.

### 2.3 Statistical power — how many questions the gate needs

A quality gate that fires on noise is worse than no gate, because you stop trusting it.

The rule of thumb: on a set of n questions, the standard error on an accuracy percentage
is roughly `50/sqrt(n)` points. So:

| eval set size | rough standard error | smallest drop you can actually detect |
|---|---|---|
| 150 questions | ~4 points | ~8 points. A 2-point gate here is pure noise. |
| 1,000 questions | ~1.6 points | ~3 points |
| 2,000 questions | ~1.1 points | ~2 points |

This is why the 150-example public FinanceBench sample cannot be the gate, and why
T²-RAGBench being open matters so much.

**What to do:**

- **Dev set** — a ~500-question slice you run constantly while tuning. Fast, cheap, noisy;
  treat differences under ~4 points as nothing.
- **CI slice** — a **fixed** 2,000-question sample, same questions every run. Fixed means
  run-to-run variation comes only from your changes, not from resampling, which tightens
  the detectable difference well below the table above.
- **Full test set** — the remainder. Run once per arm, at the end. This is the number that
  goes in the README.

And report intervals, not just points. "Recall@10 of 71.2% ± 1.1" is an engineer talking.
"Recall@10 of 71.2%" invites the question of whether you know what that means.

### 2.4 The CI gate

The eval suite runs in GitHub Actions on every push. It fails the build when:

- recall@10 drops more than 2 points below the recorded baseline
- nDCG@10 drops more than 2 points
- numeric-match accuracy drops at all
- p95 latency exceeds the stated budget
- cost per query exceeds the stated budget

This is the **drift-monitoring** gap (11.0%) and it is the single most credible thing in
the repo, because it means a change that looks fine and quietly makes the system worse
cannot be merged. Almost no portfolio project has this.

---

## 3. Architecture

```
SEC filings (S3)
      |
      v
  Docling parse  ->  structured text + tables preserved
      |
      v
  chunking layer  (text strategy | table strategies A/B/C)
      |
      v
  Postgres + pgvector
      |-- dense vectors        (embedding model, pinned version)
      |-- BM25 index           (pg_search / VectorChord-bm25 / pg_textsearch)
      |-- multi-vector index   (late-interaction arm, Arm 5 only)
      |
      v
  retrieval layer
      |-- Arm 1  dense only
      |-- Arm 2  dense + BM25, fused with Reciprocal Rank Fusion
      |-- Arm 3  Arm 2 + cross-encoder reranker
      |-- Arm 4  table-aware indexing (three sub-strategies)
      |-- Arm 5  late interaction / multi-vector
      |-- Arm 6  LangGraph agentic loop over the best static arm
      |
      v
  generation  (Bedrock)  ->  answer + citations
      |
      v
  OpenTelemetry GenAI spans  ->  Langfuse
```

Everything above is wrapped twice: as an **HTTP API** and as an **MCP server** so it
runs inside Claude Code.

### Component choices, with the reasoning

| layer | choice | why |
|---|---|---|
| **parsing** | Docling (IBM, open source) | The 2026 open-source default. Strong layout analysis, runs locally, no per-page API cost. Table extraction is a **solved** problem — buy it, don't build it. Note LlamaParse and Reducto as the hosted vision-model alternatives; Reducto is strongest on financial statements and returns per-cell bounding boxes, which is useful for citations. Worth one paragraph in the writeup, not one day of building. |
| **store** | Postgres + pgvector + a real BM25 extension | One system holds dense vectors, BM25 full-text, and the relational tables from filings. Hybrid search becomes one SQL query instead of two services plus glue. Also the honest production answer at this scale — reaching for a dedicated vector DB here would be the wrong call and you should be able to say why. |
| **embeddings** | BGE-M3 and Qwen3-Embedding, compared | BGE-M3 does dense, sparse, and multi-vector in one 568M model, which conveniently gives you Arms 1, 2, and 5 from one download. Qwen3-Embedding is the current open-weight leader. Compare them on the same set — that comparison is itself a resume bullet. |
| **reranker** | bge-reranker-v2 or Qwen3-Reranker locally, vs Cohere Rerank hosted | Qwen3-Reranker tops the open-weight leaderboard (~77% avg nDCG@10 on BEIR). Running one locally teaches you inference and latency; comparing against a hosted API teaches you the build-vs-buy tradeoff with a number attached. |
| **agent** | LangGraph with Postgres checkpointing | The production default for Python agent work, and the checkpointing is the point: a run that dies halfway resumes instead of restarting. Closes the langgraph and agent-memory gaps with real work. |
| **tracing** | OpenTelemetry GenAI semantic conventions -> Langfuse | Instrument to the **spec** (`gen_ai.*` spans), not to a vendor SDK, so the backend is swappable. Note: most GenAI conventions are still experimental — use `OTEL_SEMCONV_STABILITY_OPT_IN` for dual emission. Knowing that detail is exactly the kind of thing that signals you've actually done it. |
| **generation** | AWS Bedrock | Keeps the whole thing inside one cloud account and closes part of the AWS gap at the same time. |
| **deploy** | Docker -> App Runner or ECS Fargate + RDS Postgres + S3 | The simplest path that is still genuinely production. No Kubernetes. |
| **MCP** | FastMCP | Cheap to add, 20% of JDs, and it makes the demo *live* — you using your own service inside Claude Code beats any screenshot. |

---

## 3a. API keys and cost

**Most of this project needs no API key at all.** That is worth knowing before you start,
because it means the differentiating half runs offline and free.

| component | key needed? | note |
|---|---|---|
| Docling parsing | no | runs locally |
| Embeddings (BGE-M3, Qwen3-Embedding) | **no** | open weights, run locally via sentence-transformers |
| BM25 index | no | Postgres extension |
| Cross-encoder reranker (bge-reranker-v2, Qwen3-Reranker) | **no** | open weights, local |
| Late-interaction / multi-vector | no | BGE-M3, local |
| **Arms 1–5 — the entire retrieval benchmark** | **none** | this is the point |
| Generation (answer synthesis) | yes | the only unavoidable one |
| The agentic loop's reasoning calls (Arm 6) | yes | same model |
| Cohere Rerank — optional hosted comparison | yes | free tier is enough for a benchmark run |
| LlamaParse / Reducto — optional parser comparison | yes | both have free tiers; skip if tight |
| AWS (RDS, S3, App Runner, Bedrock) | account | free tier covers most of it; Bedrock is paid per token |

So the only place you *must* spend is generation, and generation is the least
differentiating part of the project.

### On free stealth models (Ox Alpha and similar)

OpenRouter's free stealth models — Ox Alpha, Owl Alpha, Aurora Alpha and the rest — are
free because you pay in data: the terms say prompts and completions may be logged and used
to train the model. Two consequences, pointing opposite directions.

**Privacy: a non-issue here.** The corpus is public SEC filings and public benchmark
questions. There is nothing private to leak. This is one of the few projects where the
data-logging trade is genuinely free. (Do not carry the habit over — never point a stealth
model at private code or personal data.)

**Reproducibility: disqualifying for the benchmark numbers.** OpenRouter states outright
that stealth models are available for a limited time and availability is not guaranteed,
and the model identity is undisclosed. That breaks two of the honesty rules in section 2.2:

- You cannot pin the version. The model can change under you between Arm 3 and Arm 6, and
  every earlier row silently becomes incomparable.
- You cannot name it in the README. "Generation model: Ox Alpha" means nothing to a reader
  in six months, and it means nothing to an interviewer now.
- If it's withdrawn mid-project you cannot re-run anything.

**So split it:**

- **Dev loop — use it.** Building, debugging, iterating on prompts, sanity-checking the
  pipeline end to end. It's free, the context window is large, and none of it is a
  published number. This will save you real money.
- **Every number in the benchmark table — use a named, pinned, dated model.** One model,
  fixed across all arms, version written into the results file. Bedrock fits, since it's
  already in the stack.

Write this split into the decisions log. "I used a free stealth model for development and a
pinned model for measurement, because a benchmark whose model can change is not a
benchmark" is a strong answer to a question you will be asked.

---

## 4. The 14 days

### Days 1–2 · Foundations (10h)

**Day 1.** First task, before anything else: **pull T²-RAGBench from Hugging Face
(`G4KMU/t2-ragbench`) and confirm it has what section 2 assumes** — questions, answers, and
retrievable evidence you can score against. Check the 150-example FinanceBench sample too,
as a secondary set. If T²-RAGBench turns out to be unusable, fall back to FinQA /
ConvFinQA / TAT-DQA individually and re-plan section 2. Do not build a line of retrieval
code before the eval data is confirmed in hand.

Then: repo skeleton, Docker, Postgres + pgvector running locally. Spend real time
understanding four concepts with no code — what an embedding is, what BM25 does, why
hybrid beats either alone, what a reranker actually computes. Start the decisions log.

**Day 2.** Docling over ~100 filings into structured text with tables intact. Chunking
v1. **Read the parser output with your own eyes** and write down where it breaks — that
list drives Day 6.

### Days 3–7 · The six arms (25h)

**Day 3 — protect this day.** Build the **eval harness first**: recall@10, recall@50,
nDCG@10, MRR against the benchmark evidence strings, plus the dev/test split. Then Arm 1
(dense only) and get the first number on the board. Everything downstream depends on this
harness being right, so if one day is allowed to slip, it is not this one.

**Day 4.** Arm 2: add BM25 over the same chunks, fuse with Reciprocal Rank Fusion. Measure
the delta. Check recall@50 vs recall@10 to decide how much a reranker can possibly buy you
tomorrow.

**One trap here, and it's worth knowing cold.** Postgres native full-text search
(`tsvector` / `tsquery` with `ts_rank`) is **not BM25.** It matches text fine, but
`ts_rank` is a different, weaker relevance function — it has no document-length
normalisation and no proper term-saturation, so it degrades exactly on long documents,
which is all of your corpus. If you use `ts_rank` and call it BM25 in your writeup, an
interviewer who knows retrieval will catch it and the whole benchmark loses credibility.

Use a real BM25 index. Three current options:

| option | note |
|---|---|
| `pg_search` (ParadeDB) | a genuine BM25 index over your text; the most established |
| `VectorChord-bm25` | implements Block-WeakAnd from scratch as a custom index, pgvector-style; pairs with VectorChord for the dense side |
| `pg_textsearch` (TigerData) | open-sourced early 2026, reached production-ready v1.3.0 mid-2026; explicitly built to sit next to pgvector |

Pick one, and write the `ts_rank` vs BM25 distinction into the decisions log. It is a
small, specific, verifiable piece of knowledge — the kind that separates someone who read
a tutorial from someone who built the thing.

**Day 5.** Arm 3: cross-encoder reranker over the top ~50. Local model vs hosted Cohere.
Measure nDCG gain **and** the latency it costs.

**Day 6.** Arm 4: the table problem — the one your own pushback identified as the real
differentiator. Parsing is already done; the open question is how a table becomes
searchable. Index the same tables three ways and score each on the text-and-table
question set:
- **A** — whole table as one chunk
- **B** — one chunk per row, with column headers injected into each row
- **C** — an LLM-written summary of the table indexed alongside the raw table

Then the honest finding to look for: for questions needing arithmetic across cells, does
*any* retrieval strategy work, or does it need a compute step (retrieve the table, then
query it as structured data)? That distinction — retrieval versus computation — is a
genuinely senior observation and it is sitting right there in the data.

**Day 7.** Arm 5: late interaction / multi-vector via BGE-M3. Measure the storage blowup
honestly — it's usually large, and reporting that it wasn't worth it is a stronger result
than pretending it was. Write up the table so far.

### Days 8–11 · Agent and reliability (20h)

**Day 8.** LangGraph loop: plan -> retrieve -> judge sufficiency -> re-query. Real tool
calls. Postgres checkpointing so a dead run resumes.

**Day 9.** Arm 6 measured: agentic loop vs the best static arm, specifically on the
multi-document questions where Fin-RATE says one-shot retrieval falls apart. Full
trajectory metrics — calls, tokens, dollars, wall clock, sufficiency-judge accuracy.

**Day 10.** Observability. OTel GenAI spans through every stage into Langfuse. Then
*use it*: open your ten worst failures and diagnose each one from its trace. Building a
dashboard is table stakes; debugging from it is the skill.

**Day 11.** The CI quality gate (section 2.4). Citation-grounding check. Unanswerable-
question set. **First interview drill** — I push on the whole thing the way an
interviewer would, and we log what you couldn't answer.

### Days 12–14 · Production and ship (15h)

**Day 12.** AWS, from zero: App Runner or ECS Fargate + RDS Postgres with pgvector + S3 +
Bedrock. ~10h budgeted across days 12–13 because this is new ground.

**Day 13.** Finish the deploy. Wrap as an MCP server. Latency pass: caching, batching,
and a **stated p95 budget** that the CI gate enforces.

**Day 14.** README with the benchmark table and trace screenshots. The writeup. **Final
drill.**

### Cut list, in order

thin UI -> late-interaction arm (Arm 5) -> MCP server.

**Never cut:** the eval harness, the tracing, the CI gate. Those three *are* the hiring
signal. A project with three retrieval arms and a working regression gate is far stronger
than one with six arms and no gate.

---

## 5. Working method

The code will be largely AI-assisted. That is normal now, and it changes what the project
has to be: **the code is not the evidence. The numbers and the decisions are.**

**The decisions log.** Every time you pick something — this chunk size, this reranker,
this k, this fusion method — write two or three lines: what you tried, what the number
was, why you kept it. By Day 14 this file is your interview script, and it is the one
artifact that cannot be generated, because it is a record of *your* runs.

**The rule.** Never accept generated code you can't explain out loud. When you can't,
stop and ask until you can. This is slower and it is the entire point — the 70 hours buys
understanding; the code is a byproduct.

**The drills.** Day 11 and Day 14. Hard questioning on the whole system, gaps logged, and
the Day 11 gaps get fixed before Day 14 so the second drill measures improvement.

**Publishing.** README with the benchmark table, trace screenshots, and the decisions log,
plus one written post walking through what you measured and what surprised you. Budget
~5 hours. Highest return per hour in the plan — the "what surprised me" section is what
makes a reader believe a human did the work.

---

## 6. Risks

| risk | mitigation |
|---|---|
| Benchmark data isn't available or licensed as assumed | **Already bit once** — FinanceBench's 10,231 triples are gated; only 150 are public. T²-RAGBench is now primary and is openly downloadable. Still verify on Day 1; fall back to FinQA / ConvFinQA / TAT-DQA. |
| Day 3 eval harness slips and drags everything | The one day you protect. Cut Arm 5 preemptively if Day 3 runs long. |
| AWS eats more than 10 hours from zero | Pre-picked the simplest real path. If Day 12 blows up, ship on a single container host and say so honestly in the writeup — a deployed simple thing beats an undeployed sophisticated one. |
| Local reranker or embedding model too slow on your hardware | Fall back to hosted (Cohere Rerank, Bedrock embeddings). Note the fallback and the reason in the decisions log — it becomes a cost/latency talking point rather than a hole. |
| Six arms is too many for 70 hours | The cut list is ordered in advance so the decision is already made and doesn't cost you a day of agonizing mid-build. |
| Scope creeps toward a nicer UI | The UI is first on the cut list. The demo is the MCP server plus the table. |
| Eval set too small, gate fires on noise | Section 2.3. Fixed 2,000-question CI slice; never gate on the 150-example set. |
| Generation model changes mid-project and invalidates earlier arms | Section 3a. Pinned named model for every published number; stealth/free models for the dev loop only. |

---

## 7. Definition of done

Day 14 ships all of:

- [ ] A public repo with a README carrying the full benchmark table
- [ ] At least four retrieval arms measured on the same question set, with accuracy, latency and cost side by side
- [ ] The decisions log, populated across all 14 days
- [ ] A worst-failures file per arm
- [ ] Traces visible in Langfuse, with the ten diagnosed failures written up
- [ ] A CI run that fails the build on eval regression — with a deliberately-broken commit in the history proving it fires
- [ ] A live URL someone else can hit
- [ ] An MCP server you can demo inside Claude Code
- [ ] One written post
- [ ] Two completed drills, with the gap list from each

That deliberately-broken commit is worth calling out on its own. It's proof the gate
works, not just a claim that it exists.
