"""Batch-API twin of `answer_ab_run.py`: same payload in, byte-compatible rows out, 50% price.

**Why a separate script rather than a `--batch` flag.** The two have genuinely different
control flow -- one is a `for` loop over synchronous calls, the other is
split/upload/submit/poll/download/reconcile -- and the failure modes do not overlap at all
(a 503 mid-loop vs a job that reaches a terminal state with a third of its rows failed).
Sharing a `main()` would mean one function with two disjoint halves. What they *do* share
is the output contract: rows written here are the same shape `answer_ab_score.py` already
reads, so scoring is unchanged and the two runners' results are directly comparable.

**Batch, not flex, is the 50% route** (`COST-29`). Both discounts are the same size and they
do not stack -- batch is asynchronous, flex synchronous, so they are mutually exclusive
modes. Flex additionally 503'd ~9 of 10 requests on this key; batch carries the extended
rate limits instead, which is the actual product.

**The Tier-1 cap is the reason for job splitting.** `gemini-3.7-flash` allows 3M *enqueued*
tokens on Tier 1 against 400M on Tier 2, and the cap is on tokens in flight, so oversized
work must be split and run in sequence rather than submitted at once. At ~11.6k input
tokens/question a dev pass is ~5 jobs and dev+test ~11. `--max-enqueued` exists so the same
code is one job on Tier 2 without an edit.

**JSONL, not inline requests.** Inline results are correlated to inputs by *position*;
JSONL carries an explicit per-request `key`. With partial failures possible, position is
not a safe join key -- a dropped row would silently shift every subsequent answer onto the
wrong question. The key here is `id|arm|thinking`, the same triple that keys the resume set.

Usage:
    python scripts/archive/answer_batch_run.py --dry-run          # plan + JSONL, no spend
    python scripts/archive/answer_batch_run.py --limit 5          # 10 requests, ~$0.04
    python scripts/archive/answer_batch_run.py                    # full payload
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

PAYLOAD = Path("data/day8_cost13_payload.json")
OUT = Path("data/day8_cost13_responses.jsonl")
JOBDIR = Path("data/batch_jobs")
MODEL = "gemini-3.7-flash"
DEFAULT_THINKING = "medium"
# Tier 1 for gemini-3.7-flash. Tier 2 is 400M; pass --max-enqueued to use it.
MAX_ENQUEUED_TOKENS = 3_000_000
# A job is killed by the API if it stays pending/running past 48h. Polling stops well before
# that: if a 10-request job has not moved in an hour, something is wrong that waiting cannot
# fix, and the job name is printed so `--resume` can pick it up later without re-paying.
POLL_INTERVAL_S = 20
POLL_TIMEOUT_S = 3600
TERMINAL = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _key(qid: str, arm: str, thinking: str) -> str:
    return f"{qid}|{arm}|{thinking}"


def _request_body(prompt: str, thinking: str) -> dict:
    # `service_tier` is deliberately absent: batch is already priced at the flex rate and no
    # primary source says the field is honoured inside a job, so setting it would be a claim
    # we cannot check on the bill. `thinking_level` IS set -- it is the one knob whose default
    # (medium) silently drove every earlier cost figure (COST-31).
    return {
        "contents": [{"parts": [{"text": prompt}], "role": "user"}],
        "generation_config": {"thinking_config": {"thinking_level": thinking.upper()}},
    }


def _split_jobs(todo: list[tuple[dict, str]], max_enqueued: int) -> list[list[tuple[dict, str]]]:
    """Greedy pack by input tokens. Greedy rather than optimal on purpose: the objective is
    'stay under the cap', not 'minimise job count', and a bin-packing solution would make the
    job boundaries depend on the whole set -- so adding one question could reshuffle every
    job and break resume. Sequential packing keeps boundaries stable under append."""
    jobs, cur, cur_tok = [], [], 0
    for q, arm in todo:
        tok = q["prompt_tokens"][arm]
        if cur and cur_tok + tok > max_enqueued:
            jobs.append(cur)
            cur, cur_tok = [], 0
        cur.append((q, arm))
        cur_tok += tok
    if cur:
        jobs.append(cur)
    return jobs


def _usage_from(u: dict) -> dict:
    """Same derivation as answer_ab_run: `candidates_token_count` excludes thinking tokens,
    which bill at the output rate (COST-24), so billed output is visible + thoughts."""
    in_tok = u.get("promptTokenCount") or u.get("prompt_token_count") or 0
    visible = u.get("candidatesTokenCount") or u.get("candidates_token_count") or 0
    total = u.get("totalTokenCount") or u.get("total_token_count") or 0
    thoughts = u.get("thoughtsTokenCount")
    if thoughts is None:
        thoughts = u.get("thoughts_token_count")
    if thoughts is None:
        thoughts = max(total - (in_tok + visible), 0)
    return {
        "prompt_tokens": in_tok,
        "output_tokens": visible,
        "thinking_tokens": thoughts,
        "billed_output_tokens": visible + thoughts,
        "total_tokens": total,
    }


def _text_from(response: dict) -> str:
    for cand in response.get("candidates") or []:
        parts = (cand.get("content") or {}).get("parts") or []
        joined = "".join(p.get("text", "") for p in parts)
        if joined:
            return joined
    return ""


def _poll(client, name: str) -> types.BatchJob:
    waited = 0
    last = None
    while waited < POLL_TIMEOUT_S:
        job = client.batches.get(name=name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state != last:
            print(f"    {state}")
            last = state
        if state in TERMINAL:
            return job
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
    raise TimeoutError(f"{name} still running after {POLL_TIMEOUT_S}s -- resume with --resume {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=Path, default=PAYLOAD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--jobdir", type=Path, default=JOBDIR)
    ap.add_argument("--limit", type=int, help="first N questions -- WIRING CHECK ONLY, not a sample: the payload is sorted 89 A_gold_lost then 50 B_gold_kept, so any --limit under 89 is 100% stratum A")
    ap.add_argument("--dry-run", action="store_true", help="write JSONL + print the plan, submit nothing")
    ap.add_argument("--thinking", default=DEFAULT_THINKING, choices=("low", "medium", "high"))
    ap.add_argument("--max-enqueued", type=int, default=MAX_ENQUEUED_TOKENS,
                    help="per-job input-token cap; 3M = Tier 1, 400M = Tier 2")
    ap.add_argument("--resume", help="reconcile an already-submitted job name, submit nothing new")
    args = ap.parse_args()

    payload = json.loads(args.payload.read_text())
    questions = payload["questions"]
    strata = {q["id"]: q["stratum"] for q in payload["questions"]}
    if args.limit:
        questions = questions[: args.limit]

    done = set()
    if args.out.exists():
        for line in open(args.out):
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["arm"], r.get("thinking", DEFAULT_THINKING)))

    client = None
    if not args.dry_run:
        if "GOOGLE_API_KEY" not in os.environ:
            raise SystemExit("GOOGLE_API_KEY not set")
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    if args.resume:
        job = _poll(client, args.resume)
        _reconcile(client, job, args.out, args.thinking, strata)
        return

    todo = [
        (q, arm)
        for q in questions
        for arm in q["prompts"]
        if (q["id"], arm, args.thinking) not in done
    ]
    if not todo:
        print("nothing to do -- every (id, arm, thinking) already recorded")
        return

    jobs = _split_jobs(todo, args.max_enqueued)
    in_tok = sum(q["prompt_tokens"][arm] for q, arm in todo)
    # batch = 50% of standard $0.75/$3.75 per 1M (COST-16/COST-29). 772 out is a *medium*
    # measurement (COST-31); at low it is an upper bound, which is the point of measuring.
    est = (in_tok / 1e6 * 0.75 + len(todo) * 772 / 1e6 * 3.75) * 0.5
    print(f"{len(todo)} requests, {in_tok:,} input tokens, thinking={args.thinking}")
    print(f"{len(jobs)} job(s) at a {args.max_enqueued:,}-token cap -> est ${est:.2f} on batch "
          f"(${est * 2:.2f} on standard)")
    for i, job in enumerate(jobs, 1):
        print(f"  job {i}: {len(job)} requests, {sum(q['prompt_tokens'][a] for q, a in job):,} tokens")

    args.jobdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, job_reqs in enumerate(jobs, 1):
        path = args.jobdir / f"batch-{stamp}-{args.thinking}-{i:02d}.jsonl"
        with open(path, "w") as f:
            for q, arm in job_reqs:
                f.write(json.dumps({
                    "key": _key(q["id"], arm, args.thinking),
                    "request": _request_body(q["prompts"][arm], args.thinking),
                }) + "\n")
        print(f"  wrote {path}")
        if args.dry_run:
            continue

        uploaded = client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(display_name=path.name, mime_type="application/jsonl"),
        )
        created = client.batches.create(
            model=MODEL, src=uploaded.name, config={"display_name": path.stem}
        )
        print(f"  submitted {created.name}")
        finished = _poll(client, created.name)
        _reconcile(client, finished, args.out, args.thinking, strata)

    if args.dry_run:
        print("\ndry run: JSONL written, nothing submitted")


def _reconcile(client, job, out_path: Path, thinking: str, strata: dict) -> None:
    """Write one row per successful request, keyed back by `key` rather than position.

    A terminal state is not success: JOB_STATE_PARTIALLY_SUCCEEDED means some rows failed,
    and a job can also reach SUCCEEDED with individual rows carrying an error. Failures are
    counted and printed rather than written -- a missing row simply stays absent from the
    resume set, so re-running picks it up, whereas writing a blank would be scored as a
    wrong answer and silently bias the arm it landed in."""
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    stats = job.completion_stats
    if stats:
        print(f"    stats: {stats.successful_count} ok, {stats.failed_count} failed")
    if state in {"JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"}:
        raise SystemExit(f"{job.name} ended {state}: {job.error}")

    dest = job.dest
    lines: list[str] = []
    if dest and dest.file_name:
        raw = client.files.download(file=dest.file_name)
        lines = raw.decode().splitlines()
    elif dest and dest.inlined_responses:
        raise SystemExit("inline responses not expected on the JSONL path -- refusing to "
                         "join by position; re-run this job")

    written = failed = 0
    with open(out_path, "a") as out:
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("key")
            if rec.get("error") or "response" not in rec:
                failed += 1
                print(f"    request {key} failed: {rec.get('error')}")
                continue
            qid, arm, think = key.split("|")
            resp = rec["response"]
            out.write(json.dumps({
                "id": qid,
                "arm": arm,
                "stratum": strata.get(qid),
                "tier": "batch",
                "thinking": think,
                "response": _text_from(resp),
                "usage": _usage_from(resp.get("usageMetadata") or resp.get("usage_metadata") or {}),
                "latency_s": None,
            }) + "\n")
            written += 1
    print(f"    wrote {written} rows, {failed} failed -> {out_path}")


if __name__ == "__main__":
    main()
