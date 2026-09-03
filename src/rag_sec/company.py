"""Resolve which company a question is about, from the question text alone.

`DECISIONS.md` RETR-3/RETR-4/RETR-5: three quarters of retrieved chunks come from the
wrong document, half of them the right company's wrong year, because 10-Ks reprint
themselves near-verbatim each year and neither BM25 nor a dense vector weights the year
token. The question almost always names the company; the pipeline just never used it.

The resolver reads ONLY the question string. It never looks at which filing the question
came from -- hard-filtering to the gold filing would be the inflation `DATA-4` refused.

Design rule throughout: **precision over recall.** A wrong filter deletes the gold
outright, while failing to resolve only costs the speed-up -- callers fall back to
unfiltered search. Every guard below resolves ties toward "don't filter".
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path("data/company_lexicon.json")
WORDLIST_PATH = Path("/usr/share/dict/words")

MIN_ALIAS_LEN = 3  # 3-char names (Aon, AES) are real and were previously unreachable:
# too short for the name path, and the ticker path needs an uppercase run that "Aon's"
# doesn't have. They are admitted here but flagged risky, so they need the capitalization
# test below -- see is_risky.
MIN_TICKER_LEN = 2  # 5 of our 137 tickers are single characters (A, C, F, T, V) --
# matching those as words would fire on ordinary English constantly
MAX_TICKERS = 3  # a question naming more companies than this is being matched on
# something spurious; drop the filter rather than search half the corpus

_LEGAL_SUFFIX = re.compile(
    r"\b(incorporated|corporation|company|companies|holdings?|group|international|"
    r"industries|enterprises|technologies|systems|solutions|services|partners|"
    r"inc|corp|co|ltd|plc|llc|lp|llp|nv|sa|ag|se|spa|kk|the)\b",
    re.I,
)
_TICKER_TOKEN = re.compile(r"\b[A-Z][A-Z0-9.\-]{1,5}\b")  # candidate uppercase run in the
# ORIGINAL text -- lowercasing first would make every English word a ticker candidate


def normalize(text: str) -> str:
    """Lowercase, fold punctuation to spaces, collapse whitespace.

    Handles the shapes a company name actually takes in these questions: possessives
    ("Apple's"), ampersands ("Procter & Gamble" / "Procter and Gamble"), embedded
    punctuation ("E*TRADE", "Macy's", "AT&T"), and hyphenation.
    """
    t = text.lower().replace("&", " and ")
    t = re.sub(r"['’]s\b", "", t)  # possessive, straight and curly
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return f" {' '.join(t.split())} "  # space-padded so ' alias ' is a word-boundary test


def _strip_suffix(name: str) -> str:
    return " ".join(_LEGAL_SUFFIX.sub(" ", name).split())


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    """System wordlist, used only to reject aliases that are ordinary English.

    Without it, tickers like GAP/CAT/KEY/ALL/ON/SO and one-word names like Target or Visa
    fire on unrelated prose and filter to the wrong company -- the one failure mode that
    can actually delete gold. Absent wordlist degrades to no guard, which is why the
    ticker path also requires an uppercase match in the original text.
    """
    if not WORDLIST_PATH.exists():
        return frozenset()
    return frozenset(w.strip().lower() for w in WORDLIST_PATH.read_text(errors="ignore").splitlines())


def _aliases_for(name: str, ticker: str) -> set[str]:
    """Every surface form of one company we're willing to match on.

    Covers: the full legal name, the name minus legal suffixes ("Analog Devices Inc" ->
    "analog devices"), and the leading distinctive token ("Entergy" for "Entergy
    Corporation", which is also what catches a question naming the *subsidiary*
    "Entergy Mississippi, Inc." -- those file under the parent's stem).
    """
    out: set[str] = set()
    full = normalize(name).strip()
    if full:
        out.add(full)
        # Reference data stores some names inverted -- "Interpublic Group of Companies, The",
        # "Home Depot, The" -- so the alias ends in "the" and never matches natural word
        # order in a question. Emit the de-inverted form too.
        if full.endswith(" the"):
            out.add(full[: -len(" the")])
        if full.startswith("the "):
            out.add(full[len("the ") :])
    stripped = _strip_suffix(full)
    if stripped:
        out.add(stripped)
        out.add(stripped.split()[0])
    return {a for a in out if len(a) >= MIN_ALIAS_LEN}


def is_risky(alias: str) -> bool:
    """A single-token alias whose bare occurrence is not trustworthy on its own.

    Two ways an alias earns this: it is an ordinary English word ("Visa", "Target", "Gap",
    "Apple" are company names AND common nouns, while "entergy" and "abiomed" are not), or
    it is 3 characters or fewer, where a chance collision is likely regardless. Multi-word
    aliases are never risky -- "general electric" is unambiguous even though "general"
    alone isn't. Risky aliases are kept but require the capitalization check in `resolve`,
    so `Visa Inc.` and `Aon's leased properties` resolve while `visa applications` does not.
    """
    return " " not in alias and (alias in _english_words() or len(alias) <= 3)


@lru_cache(maxsize=1)
def load_lexicon() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """((alias, (ticker, ...)), ...) sorted longest alias first, so "american airlines
    group" is tried before "american". An alias mapping to several tickers is KEPT, not
    dropped -- two companies sharing a name is exactly the case where we want to search
    both filings rather than guess one or give up.
    """
    if not LEXICON_PATH.exists():
        build_lexicon()
    raw = json.loads(LEXICON_PATH.read_text())
    pairs = [(a, tuple(sorted(t))) for a, t in raw["aliases"].items()]
    return tuple(sorted(pairs, key=lambda x: -len(x[0])))


@lru_cache(maxsize=1)
def _ticker_set() -> frozenset[str]:
    return frozenset(json.loads(LEXICON_PATH.read_text())["tickers"])


def build_lexicon(path: Path = LEXICON_PATH) -> dict:
    """Build the alias table from the corpus's own company metadata and cache it to JSON.

    Kept out of the retrieval import path: it pulls in pandas and the HF dataset loader,
    which retrieval has no other reason to depend on.
    """
    from rag_sec.eval import load_matched_questions

    df = load_matched_questions()
    aliases: dict[str, set[str]] = {}
    tickers = set()
    for name, sym in df[["company_name", "company_symbol"]].drop_duplicates().itertuples(index=False):
        tickers.add(sym)
        for alias in _aliases_for(str(name), str(sym)):
            aliases.setdefault(alias, set()).add(sym)
    payload = {"aliases": {a: sorted(t) for a, t in aliases.items()}, "tickers": sorted(tickers)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload


def _ticker_forms(token: str) -> list[str]:
    """A ticker as written -> candidate symbols. Handles share-class suffixes, which are
    written inconsistently (`BRK.B`, `BRK-B`, `BRKB`) and are not part of the stem."""
    bare = token.replace(".", "").replace("-", "")
    forms = [bare]
    if "." in token or "-" in token:
        forms.append(token.split(".")[0].split("-")[0])  # drop the class suffix entirely
    elif len(bare) > MIN_TICKER_LEN:
        forms.append(bare[:-1])  # 'BRKB' -> 'BRK'
    return forms


def _tickers_from_symbols(question: str) -> set[str]:
    """Tickers written literally ("what was AAPL's revenue in 2015").

    Case matters. An uppercase run in the ORIGINAL text is the primary signal, because a
    case-insensitive ticker match turns every "all", "it", "on", "so", "key", "cat" and
    "gap" into a company. Lowercase is accepted only for symbols of 3+ characters that
    aren't English words -- "aapl" is unambiguous, "cat" is not. Either way a ticker that
    is also an English word is rejected, since questions are full of ALL-CAPS headings.
    """
    known, words = _ticker_set(), _english_words()
    found = set()
    for tok in _TICKER_TOKEN.findall(question):  # uppercase-anchored
        for sym in _ticker_forms(tok):
            if len(sym) >= MIN_TICKER_LEN and sym in known and sym.lower() not in words:
                found.add(sym)
                break
    for tok in re.findall(r"\b[a-z]{3,5}\b", question):  # lowercase, stricter
        sym = tok.upper()
        if sym in known and tok not in words:
            found.add(sym)
    return found


def _capitalized_in(token: str, original: str) -> bool:
    """Does `token` appear capitalized somewhere other than the start of a sentence?

    The only cheap, generic way to tell "Visa Inc." from "visa applications" without a
    hand-maintained list of which company names happen to be English words. Sentence-initial
    occurrences are excluded because every word is capitalized there -- "Apple prices rose
    12%" would otherwise resolve to AAPL. Deliberately conservative: an all-lowercase or
    ALL-CAPS question loses its risky aliases and falls back to unfiltered search, which
    costs speed, not correctness.
    """
    for m in re.finditer(rf"\b{re.escape(token.capitalize())}\b", original):
        before = original[: m.start()].rstrip()
        if before and not before.endswith((".", "?", "!", ":", ";")):
            return True
    return False


def matched_aliases(question: str) -> list[tuple[str, tuple[str, ...]]]:
    """(alias, tickers) for every alias that legitimately names a company in `question`.

    The single acceptance test, shared by `resolve` and `strip_entity_framing`. They had
    separate copies and drifted: the stripper lacked the multi-ticker guard below, so
    "American Express" lost "American" and was reranked as "... transaction Express in
    2007", a worse query than the untouched original.

    Longest alias first, and a match is consumed, so "american" cannot re-fire after
    "american airlines group" has already claimed that span.
    """
    if not question:
        return []
    text = normalize(question)
    out: list[tuple[str, tuple[str, ...]]] = []
    for alias, syms in load_lexicon():
        pad = f" {alias} "
        if pad not in text:
            continue
        if is_risky(alias):
            # A single-token alias matching several *different* companies is a shared
            # fragment, not a name: "american" hits American Airlines, American Tower and
            # American Water Works. Unioning is only right when the tie is one company with
            # two share classes (Under Armour UA/UAA), which is multi-word.
            if len(syms) > 1:
                continue
            # A name that is also an ordinary English word only counts when the question
            # capitalizes it -- otherwise "visa applications" filters to Visa Inc.
            if not _capitalized_in(alias, question):
                continue
        out.append((alias, syms))
        text = text.replace(pad, "  ")
    return out


def resolve(question: str) -> list[str]:
    """Tickers this question is about. Empty list means "don't filter"."""
    hits: set[str] = set()
    for _alias, syms in matched_aliases(question):
        hits.update(syms)
    hits |= _tickers_from_symbols(question)

    # More than a handful means the match is spurious, or the question genuinely compares
    # many issuers (peer-group and performance-graph questions list index members). Either
    # way an over-broad filter buys nothing, so drop it.
    return [] if not hits or len(hits) > MAX_TICKERS else sorted(hits)


# --- query normalization for reranking (RETR-6) ------------------------------------
# The entity/filing wording that helps pick the *document* is pure noise once every
# candidate already comes from that document -- worse than noise, since it rewards
# whichever chunk repeats corporate boilerplate most (a CEO shareholder letter outscored
# the accounts-payable table holding the answer, 0.99 to 0.90). Strip it for the
# cross-encoder only; candidate generation and the generator still see the full question.

_FRAMING = [
    # "..., as reported in the 2014 Form 10-K", "as reflected in the company's 2017 report"
    r",?\s*\bas\s+(?:reported|reflected|disclosed|outlined|shown|presented|stated|detailed|"
    r"recorded|indicated|described|listed|set\s+forth)\b[^,.?;]*",
    # "..., according to its 2013 annual report"
    r",?\s*\baccording\s+to\b[^,.?;]*",
    # "..., based on its reported ... and ..." -- keeps the nouns, drops the attribution
    r",?\s*\bbased\s+on\s+(?:its|their|the\s+company'?s?)\b[^,.?;]*",
    # trailing "in the company's 2017 financial report" / "in their consolidated statements"
    r",?\s*\bin\s+(?:its|their|the\s+company'?s?)\s+[^,.?;]*"
    r"(?:report|10-?k|filing|statements?|disclosures?)\b[^,.?;]*",
    # bare references to the document itself
    r",?\s*\b(?:the\s+)?(?:annual\s+report|form\s+10-?k|10-?k\s+filing|financial\s+report)\b",
]
_FRAMING_RE = [re.compile(p, re.I) for p in _FRAMING]
# "for Apple", "of Apple", "by Apple" -- the preposition belongs to the name, not the question
_NAME_LEAD = r"(?:\b(?:for|of|by|at|from|in)\s+)?"
_SENTINEL = "\x00"
# Legal-suffix words orphaned by the removal: "Entergy Texas, Inc." must not leave ", Inc.",
# and "The Interpublic Group of Companies" must not leave "The Group of Companies". Absorbed
# on both sides of the cut rather than matched as part of the name, because which of them the
# question actually used varies.
_ORPHAN = re.compile(
    r"(?:\b(?:the|inc|corp|corporation|company|companies|holdings?|group|international|"
    r"llc|ltd|plc|lp|nv|sa|ag)\b\.?[\s,]*)*"
    + re.escape(_SENTINEL)
    + r"(?:[\s,]*\b(?:inc|corp|corporation|company|companies|holdings?|group|international|"
    r"llc|ltd|plc|lp|nv|sa|ag|and\s+its\s+subsidiaries)\b\.?)*",
    re.I,
)


def strip_entity_framing(question: str) -> str:
    """Question with company identity and filing-provenance wording removed.

    Only for the cross-encoder. Returns the original unchanged if stripping would leave
    too little behind -- a query stripped down to "what was the percentage change" scores
    nothing usefully, so the guard matters more than the stripping.
    """
    if not question:
        return question
    out = question
    for rx in _FRAMING_RE:
        out = rx.sub(" ", out)
    # Remove the company's own name, longest alias first, plus any preposition leading it.
    # Same acceptance test as `resolve` -- see `matched_aliases`.
    for alias, _syms in matched_aliases(question):
        pattern = (
            _NAME_LEAD
            + r"\b"
            + r"[^A-Za-z0-9]+".join(re.escape(w) for w in alias.split())
            + r"(?:'s|’s)?\b"
        )
        out = re.sub(pattern, _SENTINEL, out, flags=re.I)
    out = _ORPHAN.sub(" ", out)          # absorb legal suffixes orphaned by the cut
    out = out.replace(_SENTINEL, " ")    # any sentinel the orphan pass didn't consume
    out = re.sub(r"\s*,\s*(?=[,.?;])", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,;")
    out = re.sub(r"\s+([,.?;])", r"\1", out)
    # Guard: if what's left is too short to discriminate, keep the original.
    return out if len(out.split()) >= 4 else question
