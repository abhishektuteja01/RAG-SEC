"""COST-13, stage 2: send both arms' prompts to Gemini and record raw responses.

Uses the **raw `google-genai` SDK**, not langchain-google-genai (PR #1937 unmerged, so
that path cannot set a service tier at all).

**Sends at `standard`, not `flex`, and that is a measured decision, not an oversight.**
`COST-19` prescribed flex on the docs; in practice it returned 503 "high demand" on roughly
nine of every ten requests and the COST-13 run was moved to standard (`COST-29`). Flex is
not broken -- it is best-effort and capacity-gated, which is exactly what a 503 means -- but
it was not usable on this key at that time. `--tier flex` is kept so the state can be
re-tested cheaply; it is no longer the default.

**The 50% discount route is the Batch API, not flex.** Both are 50% off standard and they do
not stack (batch is asynchronous, flex synchronous -- mutually exclusive modes). Batch also
carries the extended rate limits, so it is the right tool for a full dev/test pass. It is
not wired up here: this script is synchronous by design, and batch needs a submit/poll
pipeline over JSONL. For ~300 calls that pipeline costs more than the ~$1 it saves.

Checkpointed by (id, arm, thinking) so a rate-limit stall or a killed process resumes
instead of re-paying. `thinking` is in the key because a second reasoning level is a new
arm, not a rerun: rows without the field predate the flag and ran at the API default. Records `usage_metadata` per call so the *actual* spend can be reconciled against
the estimate rather than assumed.

Order is deliberately interleaved by question, not grouped by arm: if the API's behaviour
drifts mid-run (throttling, a model update), grouping by arm would put that drift entirely
in one arm and confound the paired comparison.

Usage:
    python scripts/archive/answer_ab_run.py --limit 10      # wiring check, ~$0.06
    python scripts/archive/answer_ab_run.py                 # full run, ~$2.02 standard
    python scripts/archive/answer_ab_run.py --thinking low  # the medium/low A/B arm
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
SERVICE_TIER = "standard"  # COST-29: flex 503d ~9 of 10 requests. Batch is the discount route.
# The API default for 3.7-flash. Named explicitly so it lands in each row rather than being
# an unrecorded property of the day a run happened -- rows written before 2026-09-04 carry
# no `thinking` key and are backfilled to this value on resume, which is what they ran at.
DEFAULT_THINKING = "medium"
# Sized for flex, and kept at that size for the standard path too. Flex is best-effort with
# a 1-15 min target, where a 503 "high demand" is an expected state rather than an error;
# 20s doubling over 6 attempts waits ~10.5 min before giving up, against the original
# 4s/5-attempt (~1 min) which gave up during a dip the docs call "usually temporary". Even
# this was not enough for flex (COST-29). It is generous for standard, which costs nothing:
# retries only fire on an actual failure.
MAX_RETRIES = 6
BACKOFF_BASE = 20.0


def _config(tier: str | None, thinking: str | None):
    """`thinking_level` is the Gemini 3 reasoning control; the older numeric
    `thinking_budget` is 2.5-generation and has no published range for 3.7-flash, so it is
    not used here. 3.7-flash accepts low/medium/high only -- `minimal` 400s -- and
    **defaults to medium**, which is what every call before this flag was billed at.
    Thinking cannot be turned off on this model at any level."""
    kw = {}
    if tier:
        kw["service_tier"] = tier
    if thinking:
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    return types.GenerateContentConfig(**kw) if kw else None


def _call(client, prompt: str, tier: str | None, thinking: str | None) -> tuple[str, dict]:
    """One request. On flex a transient failure is expected rather than exceptional (503
    under load, COST-29), so it is retried with exponential backoff. Raising after
    MAX_RETRIES is deliberate: a silently-missing response would become a scored 'wrong
    answer' and bias the arm it happened to hit."""
    delay = BACKOFF_BASE
    cfg = _config(tier, thinking)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            u = resp.usage_metadata
            # `candidates_token_count` is the VISIBLE answer only. gemini-3.7-flash also
            # emits internal reasoning tokens that bill at the output rate and are absent
            # from that field (COST-24) -- measured at ~660/call against ~110 visible, so
            # pricing off candidates alone understated this run by 45%. Derived from
            # `total - (prompt + candidates)` when `thoughts_token_count` is absent.
            # Bound to `in_tok`, not `prompt`: rebinding the argument would send an int as
            # `contents` if a later line in this try block raised and the loop retried.
            in_tok = getattr(u, "prompt_token_count", None) or 0
            visible = getattr(u, "candidates_token_count", None) or 0
            total = getattr(u, "total_token_count", None) or 0
            thoughts = getattr(u, "thoughts_token_count", None)
            if thoughts is None:
                thoughts = max(total - (in_tok + visible), 0)
            return resp.text or "", {
                "prompt_tokens": in_tok,
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
    ap.add_argument("--limit", type=int, help="first N questions -- WIRING CHECK ONLY, not a sample: the payload is sorted 89 A_gold_lost then 50 B_gold_kept, so any --limit under 89 is 100% stratum A")
    ap.add_argument("--dry-run", action="store_true", help="price it, send nothing")
    ap.add_argument("--tier", default=SERVICE_TIER, choices=("flex", "standard"),
                    help="default standard: flex 503d ~9/10 requests here (COST-29). Batch, "
                         "not flex, is the 50% route for a full pass -- see the module docstring")
    ap.add_argument("--thinking", default=DEFAULT_THINKING, choices=("low", "medium", "high"),
                    help="3.7-flash reasoning level; medium is the API default and what "
                         "every pre-2026-09-04 row was billed at")
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
                done.add((r["id"], r["arm"], r.get("thinking", DEFAULT_THINKING)))
    if done:
        print(f"resuming: {len(done)} (id, arm, thinking) triples already recorded")

    todo = [
        (q, arm)
        for q in questions
        for arm in q["prompts"]
        if (q["id"], arm, args.thinking) not in done
    ]
    in_tok = sum(q["prompt_tokens"][arm] for q, arm in todo)
    # flex and batch are both 50% of gemini-3.7-flash standard $0.75/$3.75 per 1M, and do
    # not stack (COST-16/COST-29). Only flex is reachable from this synchronous script.
    mult = 0.5 if args.tier == "flex" else 1.0
    # 772 = ~110 visible + ~662 thinking, measured (COST-24) **at thinking=medium**. Google
    # publishes no per-level token counts, so this estimate is an upper bound at low and
    # only exact at medium -- the point of the low run is to replace it with a measurement.
    est = (in_tok / 1e6 * 0.75 + len(todo) * 772 / 1e6 * 3.75) * mult
    print(f"{len(todo)} calls, {in_tok:,} input tokens -> est ${est:.2f} on {args.tier}")
    if args.thinking != DEFAULT_THINKING:
        print(f"  (output side estimated at {DEFAULT_THINKING} rates; not published for {args.thinking})")
    if args.dry_run:
        return
    if "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    print(f"tier={args.tier}  model={MODEL}  thinking={args.thinking}")
    spent_in = spent_out = 0
    with open(args.out, "a") as out:
        for q, arm in tqdm(todo, desc=f"{MODEL}/{args.tier}"):
            t0 = time.perf_counter()
            text, usage = _call(
                client,
                q["prompts"][arm],
                None if args.tier == "standard" else args.tier,
                args.thinking,
            )
            spent_in += usage["prompt_tokens"] or 0
            spent_out += usage["billed_output_tokens"] or 0
            out.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "arm": arm,
                        "stratum": q["stratum"],
                        "tier": args.tier,
                        "thinking": args.thinking,
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
