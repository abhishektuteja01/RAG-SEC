"""COST-13, stage 2: send both arms' prompts to Gemini and record raw responses.

Uses the **raw `google-genai` SDK with `service_tier="flex"`** (`COST-19`), not
langchain-google-genai: flex is GA on the Gemini API and gives the same 50% discount as
Batch, but at ~300 requests it needs no submit/poll pipeline and no chunking around the 3M
enqueued-token cap. `COST-12`'s "flex is unreachable" was wrong -- only the LangChain path
is blocked (PR #1937 unmerged).

Checkpointed by (id, arm) so a rate-limit stall or a killed process resumes instead of
re-paying. Records `usage_metadata` per call so the *actual* spend can be reconciled against
the estimate rather than assumed.

Order is deliberately interleaved by question, not grouped by arm: if the API's behaviour
drifts mid-run (throttling, a model update), grouping by arm would put that drift entirely
in one arm and confound the paired comparison.

Usage:
    python scripts/compression/answer_ab_run.py --limit 10      # wiring check, ~$0.06
    python scripts/compression/answer_ab_run.py                 # full run, ~$0.76
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from tqdm import tqdm  # noqa: E402

PAYLOAD = Path("data/day8_cost13_payload.json")
OUT = Path("data/day8_cost13_responses.jsonl")
MODEL = "gemini-3.7-flash"  # the answer node's model (AGENT-3)
SERVICE_TIER = "flex"
# Sized for flex, not for a normal API: it is best-effort with a 1-15 min target, and a
# 503 "high demand" is an expected state rather than an error. 20s doubling over 6 attempts
# waits ~10.5 min before giving up; the original 4s/5-attempt (~1 min) gave up during a
# capacity dip that the docs describe as "usually temporary".
MAX_RETRIES = 6
BACKOFF_BASE = 20.0


def _call(client, prompt: str, tier: str | None) -> tuple[str, dict]:
    """One flex request. Flex is best-effort with a 1-15 min target, so a transient failure
    is expected rather than exceptional -- retried with exponential backoff. Raising after
    MAX_RETRIES is deliberate: a silently-missing response would become a scored 'wrong
    answer' and bias the arm it happened to hit."""
    delay = BACKOFF_BASE
    for attempt in range(MAX_RETRIES):
        try:
            cfg = types.GenerateContentConfig(service_tier=tier) if tier else None
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            u = resp.usage_metadata
            # `candidates_token_count` is the VISIBLE answer only. gemini-3.7-flash also
            # emits internal reasoning tokens that bill at the output rate and are absent
            # from that field (COST-24) -- measured at ~660/call against ~110 visible, so
            # pricing off candidates alone understated this run by 45%. Derived from
            # `total - (prompt + candidates)` when `thoughts_token_count` is absent.
            prompt = getattr(u, "prompt_token_count", None) or 0
            visible = getattr(u, "candidates_token_count", None) or 0
            total = getattr(u, "total_token_count", None) or 0
            thoughts = getattr(u, "thoughts_token_count", None)
            if thoughts is None:
                thoughts = max(total - (prompt + visible), 0)
            return resp.text or "", {
                "prompt_tokens": prompt,
                "output_tokens": visible,
                "thinking_tokens": thoughts,
                "billed_output_tokens": visible + thoughts,
                "total_tokens": total,
            }
        except Exception as e:  # noqa: BLE001 - any SDK/transport error is retryable here
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after {type(e).__name__}: {e}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=Path, default=PAYLOAD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, help="first N questions only (wiring check)")
    ap.add_argument("--dry-run", action="store_true", help="price it, send nothing")
    ap.add_argument("--tier", default=SERVICE_TIER, choices=("flex", "standard"),
                    help="standard costs 2x but is not capacity-gated; flex 503s under load")
    args = ap.parse_args()

    payload = json.loads(args.payload.read_text())
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]

    done = set()
    if args.out.exists():
        for line in open(args.out):
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["arm"]))
    if done:
        print(f"resuming: {len(done)} (id, arm) pairs already recorded")

    todo = [(q, arm) for q in questions for arm in q["prompts"] if (q["id"], arm) not in done]
    in_tok = sum(q["prompt_tokens"][arm] for q, arm in todo)
    # flex = 50% of gemini-3.7-flash standard $0.75/$3.75 per 1M (COST-16/COST-19)
    mult = 0.5 if args.tier == "flex" else 1.0
    # 772 = ~110 visible + ~662 thinking, measured (COST-24). Using 300 here was the
    # source of the original underestimate.
    est = (in_tok / 1e6 * 0.75 + len(todo) * 772 / 1e6 * 3.75) * mult
    print(f"{len(todo)} calls, {in_tok:,} input tokens -> est ${est:.2f} on {args.tier}")
    if args.dry_run:
        return
    if "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    print(f"tier={args.tier}  model={MODEL}")
    spent_in = spent_out = 0
    with open(args.out, "a") as out:
        for q, arm in tqdm(todo, desc=f"{MODEL}/{args.tier}"):
            t0 = time.perf_counter()
            text, usage = _call(client, q["prompts"][arm], None if args.tier == "standard" else args.tier)
            spent_in += usage["prompt_tokens"] or 0
            spent_out += usage["billed_output_tokens"] or 0
            out.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "arm": arm,
                        "stratum": q["stratum"],
                        "tier": args.tier,
                        "response": text,
                        "usage": usage,
                        "latency_s": round(time.perf_counter() - t0, 2),
                    }
                )
                + "\n"
            )
            out.flush()

    actual = (spent_in / 1e6 * 0.75 + spent_out / 1e6 * 3.75) * mult
    print(f"\nactual: {spent_in:,} in + {spent_out:,} out tokens = ${actual:.2f} on {args.tier}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
