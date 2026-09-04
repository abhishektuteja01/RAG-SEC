"""Claude-backed table summarizer for Arm 4 Strategy C. The only file here that costs money
to run, and it should not run again -- Strategy C lost. DECISIONS.md ARM4-2/ARM4-5/ARM4-6.
"""

import os
import re
import time

from anthropic import Anthropic

# A bare dash in a value cell is a filing's convention for zero, not "no data" (a
# rollforward's opening balance shown as "$-"). Prompting the model to read it that way
# failed even with the rule spelled out twice, so it is normalized at the source (ARM4-6).
_DASH_CELL_RE = re.compile(r"^(\$?)\s*[-–—―\x97]+\s*$")

# Pinned: cheap at ~500 calls, and its extended thinking is opt-in, so a fixed max_tokens
# is actually spendable on output. ARM4-2 has the models this beat and on what.
MODEL_NAME = "claude-haiku-4-5-20251001"
MIN_CALL_INTERVAL_S = 0.3
MAX_RETRIES = 4
MAX_TOKENS = 1200  # room for the structural pre-analysis plus the final summary


def _normalize_dash_cells(rows: list[list[str]]) -> list[list[str]]:
    return [[_DASH_CELL_RE.sub(lambda m: f"{m.group(1)}0", cell) for cell in row] for row in rows]


# Forces a structure-first read before the model commits to a number, aimed at the failure
# CLASSES the spot-check found rather than one patch per table: split headers, rollforward
# start/end rows, column misalignment, unstated units, ungrounded trend direction. The
# analysis is discarded after parsing -- only the SUMMARY: line gets indexed.
_PROMPT_TEMPLATE = """\
You are reading a financial table extracted from an SEC 10-K filing, to produce a short \
summary that will be indexed for semantic retrieval. Work through the table's structure \
before writing anything, using this checklist:

1. HEADERS: Identify every header row. Some tables have a split header -- a units/description \
row followed by the real column labels -- read all leading rows before deciding which one is \
the true column header; don't assume row 0 is always it.
2. COLUMNS: For each data column, pin down exactly which period/label/category it represents, \
strictly from the header -- never from position or assumption.
3. UNITS: Note the unit stated in the table (e.g. "in thousands," "in millions"). If no unit is \
stated anywhere, say so explicitly in your summary rather than presenting a guess as fact -- you \
may still infer one from context (e.g. company size) but phrase it as an inference, not a given.
4. ROLLFORWARDS: If the table walks a balance from a start to an end (e.g. "balance at \
[date]" -> additions/deductions -> "balance at [date]"), the starting balance is the FIRST row \
of the walk and the ending balance is the LAST row -- determine this from row ORDER and each \
row's own label, never from which number is largest, most prominent, or the first non-zero \
value. A dash, em-dash, "$-", or blank cell in the VALUE means zero/none -- treat it as 0, don't \
reassign the "starting balance" label to a later row just because the first row's value looks \
like 0. (A dash inside a row's LABEL text, e.g. as a stylistic separator, is unrelated to this --  \
only dashes in the value column mean zero.)
5. DIRECTION: When describing whether something rose or fell, compare both endpoint values in \
their actual chronological column order (earliest to latest) -- never assert a direction you \
haven't checked against both numbers.

Write your internal read-through of points 1-5 first, briefly. Then, on its own line starting \
with exactly "SUMMARY:", give the final 1-2 plain-English sentences: what the table shows, and \
its key figures paired with their exact row/column labels and time periods. State only what the \
table directly supports -- no commentary or analysis beyond that.

<table>
{table_text}
</table>\
"""

_client: Anthropic | None = None
_last_call_time = 0.0


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def summarize_table(rows: list[list[str]]) -> str:
    """One-to-two sentence plain-English description of a table, for indexing alongside
    the raw table (Arm 4 Strategy C -- not replacing it)."""
    global _last_call_time
    table_text = "\n".join(" | ".join(row) for row in _normalize_dash_cells(rows))
    client = _get_client()

    wait = MIN_CALL_INTERVAL_S - (time.monotonic() - _last_call_time)
    if wait > 0:
        time.sleep(wait)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(table_text=table_text)}],
            )
            _last_call_time = time.monotonic()
            full_text = "".join(block.text for block in resp.content if block.type == "text").strip()
            if not full_text:
                raise RuntimeError(f"empty response, stop_reason={resp.stop_reason}")
            if "SUMMARY:" not in full_text:
                raise RuntimeError(f"no SUMMARY: line in response: {full_text!r}")
            summary = full_text.rsplit("SUMMARY:", 1)[1].strip()
            if not summary:
                raise RuntimeError(f"empty summary after SUMMARY: marker, stop_reason={resp.stop_reason}")
            return summary
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt * 5)
    raise RuntimeError("unreachable")
