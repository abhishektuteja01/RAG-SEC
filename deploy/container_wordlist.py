"""Fails if the company resolver's English-word guard is off or reading a different wordlist.

Without the list nothing crashes: the resolver just filters to the wrong company on some
questions. So the container runs this at build AND at startup. It checks that a list is
found, that it is the macOS/BSD `web2` list the numbers were measured with (Debian's
`wamerican` has proper nouns like 'intel', so it is not a substitute), and that
'visa applications' stays unfiltered while 'Visa Inc.' resolves.

Usage (needs rag_sec importable, e.g. `uv run`):
    python deploy/container_wordlist.py
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
