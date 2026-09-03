"""Edge-case + real-data check for `rag_sec.company.resolve` (DECISIONS.md RETR-5).

Runnable, not pytest, to match the rest of `scripts/`. Two parts:

  1. Hand-written cases covering the surface forms a company name actually takes, and --
     more importantly -- the collisions that must NOT resolve. A false positive filters to
     the wrong company and deletes the gold outright; a false negative only falls back to
     unfiltered search. The cases are weighted accordingly.
  2. Precision/recall over the real train and dev splits. Train is the calibration split;
     dev is reported to confirm nothing was tuned into it.

Usage:
    python scripts/diagnostics/day8_company_resolver_check.py
"""

from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from rag_sec.company import resolve  # noqa: E402
from rag_sec.eval import load_matched_questions  # noqa: E402

# (label, question, expected) -- expected None means "must not resolve"
CASES = [
    # surface forms of the name
    ("full legal name", "What was Analog Devices, Inc.'s revenue in 2009?", "ADI"),
    ("suffix dropped", "What was Analog Devices revenue in 2009?", "ADI"),
    ("first word only", "What was Entergy's net revenue in 2010?", "ETR"),
    ("subsidiary named", "net revenue for Entergy Mississippi, Inc. in 2010", "ETR"),
    ("lowercase name", "what was philip morris international operating income", "PM"),
    ("ALL CAPS question", "WHAT WAS ABIOMED REVENUE IN 2012?", "ABMD"),
    ("longest match wins", "American Airlines Group total operating expenses", "AAL"),
    ("plural possessive", "Analog Devices' inventories in 2009", "ADI"),
    ("curly apostrophe", "What was Intel’s gross margin in 2013?", "INTC"),
    # tickers written literally
    ("ticker uppercase", "What was AAPL revenue in 2015?", "AAPL"),
    ("ticker possessive", "What was ADBE's operating margin?", "ADBE"),
    ("ticker lowercase", "what was abmd revenue in 2012", "ABMD"),
    # collisions that must be rejected -- these are the ones that can delete gold
    ("commodity, lowercase", "What did Cargill's report say about the price of apple?", None),
    ("commodity, sentence-initial", "Apple prices rose 12% according to the filing.", None),
    ("common word, no ticker", "How many visa applications were processed?", None),
    ("ticker that is a word", "The CAT scan and GAP analysis were ALL completed", None),
    ("lowercase ticker-ish", "the it department spent all of its budget on cat food", None),
    ("single-char ticker", "What was the change in T from 2015 to 2016?", None),
    ("substring safety", "The bacon and aeon divisions reported gains", None),
    ("no company named", "What was the percentage change in revenue from 2007 to 2008?", None),
    ("empty question", "", None),
    # ambiguity resolved by searching the union, never by guessing one
    ("dual-class same name", "What was Under Armour's revenue in 2016?", "UA+UAA"),
    ("two companies compared", "How did Intel's margin compare to Adobe's in 2013?", "ADBE+INTC"),
    ("in-corpus + out-of-corpus", "What did Cargill say about Intel's pricing in 2013?", "INTC"),
    ("peer-group list", "Compare Intel, Adobe, Abiomed and Entergy returns", None),
    ("mid-sentence capital", "In 2015, Visa Inc. reported net revenue growth.", "V"),
]


def run_cases() -> int:
    print(f"{'case':<30} {'got':<12} {'want':<12} ok")
    failed = 0
    for label, q, want in CASES:
        got = resolve(q)
        g, w = ("+".join(got) if got else "-"), (want or "-")
        ok = g == w
        failed += not ok
        print(f"{label:<30} {g:<12} {w:<12} {'OK' if ok else 'FAIL'}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} edge cases passed")
    return failed


def run_splits() -> None:
    df = load_matched_questions()
    for split in ("train", "dev"):
        sub = df[df["split"] == split]
        n = len(sub)
        resolved = right = wrong = 0
        per = Counter()
        misses = []
        for _, r in sub.iterrows():
            got = resolve(r["question"])
            if not got:
                continue
            resolved += 1
            per[len(got)] += 1
            if r["company_symbol"] in got:
                right += 1
            else:
                wrong += 1
                if len(misses) < 5:
                    misses.append((r["company_symbol"], got, r["question"][:90]))
        print(f"\n{split}  n={n}")
        print(f"  resolved              {resolved:5} ({100 * resolved / n:5.1f}% coverage)")
        print(f"  correct ticker in set {right:5} ({100 * right / max(resolved, 1):5.1f}% precision)")
        print(f"  WRONG -> deletes gold {wrong:5} ({100 * wrong / n:5.2f}% of all questions)")
        print(f"  tickers per resolved: {dict(sorted(per.items()))}")
        # The residual failure mode is a question naming an acquired business or a
        # counterparty rather than the filer -- "the FIS Gaming Business" inside a Global
        # Payments filing. `retrieve(reserve=N)` is the mitigation if this ever matters.
        for sym, got, q in misses:
            print(f"    gold={sym} got={got}  {q!r}")


if __name__ == "__main__":
    failures = run_cases()
    run_splits()
    raise SystemExit(1 if failures else 0)
