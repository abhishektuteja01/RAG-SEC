"""Answer generation: build the prompt from retrieved chunks, call Gemini, parse the ANSWER line.

Uses the `google-genai` SDK directly. It reads GOOGLE_API_KEY from the environment.
"""

import re
from collections.abc import Iterator
from functools import lru_cache

from rag_sec.config import GENERATION_MODEL

ANSWER_PROMPT = """Question: {question}

Evidence gathered:
{evidence}

Answer the question using only this evidence. If the evidence is insufficient, say so
plainly rather than guessing.

Give the numeric value in the same units as the evidence -- do not expand thousands or
millions. End your response with a single line:
ANSWER: <number>
or, if the evidence does not contain the answer:
ANSWER: INSUFFICIENT"""

# The level the answer-accuracy numbers were measured with.
THINKING_LEVEL = "MEDIUM"


def evidence_text(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['filing_stem']} chunk {c['chunk_index']}] {c['text']}" for c in chunks
    )


@lru_cache(maxsize=1)
def _client():
    from google import genai

    return genai.Client()


def _request(question: str, chunks: list[dict]) -> dict:
    from google.genai import types

    return {
        "model": GENERATION_MODEL,
        "contents": ANSWER_PROMPT.format(question=question, evidence=evidence_text(chunks)),
        "config": types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    }


def generate_answer(question: str, chunks: list[dict]) -> str:
    """One Gemini call over the retrieved chunks. Returns the response text (thoughts excluded)."""
    return _client().models.generate_content(**_request(question, chunks)).text or ""


def stream_answer(question: str, chunks: list[dict]) -> Iterator[str]:
    """The same call as `generate_answer`, yielding text pieces as they arrive. Each chunk's
    `.text` skips thought parts, so only answer text is yielded."""
    for chunk in _client().models.generate_content_stream(**_request(question, chunks)):
        if chunk.text:
            yield chunk.text


# ── parsing the ANSWER line ──────────────────────────────────────────────────────────
# Strict: the ANSWER line is required and must hold a number (or a yes/no-style word).
# No "last number in the text" fallback: that would read a year out of the reasoning.
_ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_REFUSAL = re.compile(r"insufficient|cannot|not (?:provided|available|stated|found)|unknown", re.IGNORECASE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
# Matched against the first word only, so a sentence that merely mentions "no" later on
# cannot misfire.
_BOOL_TRUE = {"yes", "true", "higher", "greater", "increase", "increased", "more"}
_BOOL_FALSE = {"no", "false", "lower", "less", "decrease", "decreased", "fewer"}


def _to_float(token: str) -> float | None:
    t = token.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    t = t.rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


def _bool_value(token: str) -> float | None:
    words = token.strip().lower().split()
    if not words:
        return None
    first = words[0].strip(".,;:")
    if first in _BOOL_TRUE:
        return 1.0
    if first in _BOOL_FALSE:
        return 0.0
    return None


def parse_answer(text: str) -> tuple[float | None, str]:
    """(value, reason). Reasons: `ok`, `refused` (ANSWER line declines), `no_number` (ANSWER
    line unparseable), `no_answer_line` (format not followed), `empty`."""
    if not text or not text.strip():
        return None, "empty"
    m = _ANSWER_LINE.search(text)
    if not m:
        return None, "no_answer_line"
    body = m.group(1)
    nums = _NUMBER.findall(body)
    if not nums:
        bv = _bool_value(body)
        if bv is not None:
            return bv, "ok"
        return None, "refused" if _REFUSAL.search(body) else "no_number"
    v = _to_float(nums[0])
    return (v, "ok") if v is not None else (None, "no_number")
