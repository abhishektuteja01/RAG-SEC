"""Fails if `RETR-11`'s English-word guard is off or reading a different wordlist
(DECISIONS.md RETR-11, DEPLOY-5).

Runnable, not pytest, to match the rest of `scripts/`. It exists because the failure it
catches is invisible: with no wordlist `_english_words()` returns an empty set, `is_risky`
stops flagging aliases that are ordinary English, and the resolver simply filters to the
wrong company on some questions -- nothing crashes, recall just drops. The container runs
this at build AND at startup so that state cannot boot.

Three assertions, cheapest first:
  1. a wordlist is found at all (the RuntimeWarning is promoted to an error);
  2. it is the one the benchmark read -- macOS/BSD `web2`, identified by content rather than
     by path. Debian's `wamerican` is NOT a substitute: it carries proper nouns, so
     lowercased it makes 'intel', 'merck', 'nike' and 21 other in-corpus aliases risky and
     the resolver stops matching them in lowercase questions;
  3. the two `RETR-11` behaviours the guard exists for, end to end through `resolve`.

Usage:
    python scripts/checks/container_wordlist.py
"""

import warnings

from rag_sec.company import WORDLIST_CANDIDATES, _english_words, resolve

# In web2 and not in wamerican / not in web2 and in wamerican -- picked from the 27 aliases
# whose risky-verdict differs between the two lists, so a swapped list fails here.
IN_WEB2_ONLY = ("celanese", "arista")
IN_WAMERICAN_ONLY = ("intel", "merck", "nike", "walmart")
MIN_WORDS = 200_000  # web2 is ~234k unique lowercased; wamerican ~102k


def main() -> int:
    failed = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failed
        failed += not ok
        print(f"{label:<34} {'OK' if ok else 'FAIL'}  {detail}")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            words = _english_words()
        except RuntimeWarning as exc:
            print(f"FAIL  no wordlist on any of {WORDLIST_CANDIDATES}\n{exc}")
            return 1

    check("wordlist size", len(words) >= MIN_WORDS, f"{len(words)} words")
    for w in IN_WEB2_ONLY:
        check(f"web2 contains {w!r}", w in words)
    for w in IN_WAMERICAN_ONLY:
        check(f"web2 lacks {w!r}", w not in words, "(wamerican would have it)")

    # The guard's whole point: same alias, capitalized vs not.
    check("'visa applications' unfiltered", not resolve("How many visa applications were processed?"))
    check("'Visa Inc.' -> V", list(resolve("In 2015, Visa Inc. reported net revenue growth.")) == ["V"])

    print("wordlist guard ON and matching the benchmarked list" if not failed else "WORDLIST GUARD DEFECTIVE")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
