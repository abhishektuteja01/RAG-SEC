"""The only place a *generated answer* is scored against the dataset's gold (retrieval
metrics live in `eval.py`). DECISIONS.md COST-13/COST-21/COST-23.

Number match, not string equality: the two gold fields routinely disagree in form and
sometimes in value (dev, n=1235: 688 agree within 1%, 380 differ by exactly x100, 8 by sign
only, 125 irreconcilable, 34 unparseable). A prediction is correct if it matches EITHER
field under any factor in `SCALES`, within a relative tolerance -- accepting either field
handles the sign cases without making the scorer sign-blind, since one field carries each
sign, and the tolerance absorbs rounding (24.691358 vs 24.69) rather than model error.
"""

import re

REL_TOL = 0.01
ABS_TOL = 0.005

# Scale factors accepted when comparing to gold. x100 is the percentage convention
# (`0.935` vs `93.5%`); x1000 / x1e6 are the "(in thousands)" / "(in millions)" table
# headers -- gold is the table figure, but a model naturally answers in dollars, and
# `convfinqa_1431` (gold 567048, table in thousands) is a real observed case. Widening the
# tolerance this way risks a false positive only if a prediction lands within 1% of
# gold x 10^k for some k, which does not happen by chance at these magnitudes.
SCALES = (1.0, 100.0, 0.01, 1000.0, 0.001, 1e6, 1e-6)

# Parsing is STRICT: the ANSWER line is required and must contain a number. There is no
# free-text fallback, because the fallback ("last number mentioned") was observed turning
# `ANSWER: Insufficient information` into 2007.0 -- a year lifted out of the reasoning text.
# A refusal that scores as a spurious number could match a gold by accident and count as
# correct, which is worse than scoring it wrong. Non-compliance is instead *reported* per
# arm by `parse_reason`, so a compliance difference between arms is visible rather than
# silently absorbed into the result.
_ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_REFUSAL = re.compile(r"insufficient|cannot|not (?:provided|available|stated|found)|unknown", re.IGNORECASE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def _to_float(token: str) -> float | None:
    t = token.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    t = t.rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


def parse_reason(text: str) -> tuple[float | None, str]:
    """(value, reason) so the caller can report *why* a prediction is missing. Reasons:
    `ok`, `refused` (ANSWER line present, explicitly declines), `no_number` (ANSWER line
    present, unparseable), `no_answer_line` (format not followed), `empty`."""
    if not text or not text.strip():
        return None, "empty"
    m = _ANSWER_LINE.search(text)
    if not m:
        return None, "no_answer_line"
    body = m.group(1)
    nums = _NUMBER.findall(body)
    if not nums:
        return None, "refused" if _REFUSAL.search(body) else "no_number"
    v = _to_float(nums[0])
    return (v, "ok") if v is not None else (None, "no_number")


def gold_values(program_answer, original_answer) -> list[float]:
    """Every numeric reading the dataset offers for this question."""
    out = []
    for raw in (program_answer, original_answer):
        v = _to_float(str(raw)) if raw is not None else None
        if v is not None:
            out.append(v)
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * abs(b))


def is_correct(pred: float | None, golds: list[float]) -> bool:
    if pred is None or not golds:
        return False
    for g in golds:
        for scale in SCALES:
            if _close(pred, g * scale):
                return True
    return False


def gold_is_scoreable(program_answer, original_answer) -> bool:
    """False when the two references disagree beyond rounding/scaling/sign, so the question
    cannot be scored against a defensible target. Such questions are excluded from COST-13's
    strata up front rather than silently scored as wrong in both arms -- they would land in
    the concordant cell, consuming budget while contributing nothing to McNemar.

    Dev breakdown (n=1235): 1076 agree, 125 irreconcilable (excluded), 32 have exactly one
    parseable field and are KEPT and scored against it, 2 have neither (excluded). So 127
    are excluded, not the 125+34=159 that COST-21 states.

    Deliberately stricter than `is_correct`: it tests only x100, because two gold fields
    disagreeing by a units factor is a labeling problem, whereas a model answering in a
    different unit is not.
    """
    golds = gold_values(program_answer, original_answer)
    if not golds:
        return False
    if len(golds) == 1:
        return True
    p, o = golds
    return (
        _close(p, o)
        or _close(p * 100, o)
        or _close(p, o * 100)
        or _close(abs(p), abs(o))
        or _close(abs(p * 100), abs(o))
    )
