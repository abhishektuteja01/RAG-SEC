"""Startup guards against DECISIONS.md RETR-24, run before any measurement pass.

RETR-24 cost a day of measurement because two things were true at once and neither was
checked: the corpus had grown 6,373 non-'A' rows, and the queries reading it had no
`variant` predicate. Either check alone leaves a hole -- pinned counts miss a newly
written bad query against a stable corpus; the source scan misses nothing but only if
something actually runs it. So both run automatically, on the first get_conn().

The source scan lives here rather than in the diagnostics script so it is importable at
runtime; scripts/diagnostics/check_variant_predicates.py is a thin CLI over it.
"""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN_DIRS = ("src", "scripts")
FROM_CHUNKS = re.compile(r"\bfrom\s+chunks\b", re.I)
WHERE = re.compile(r"\bwhere\b", re.I)

# Unconstrained reads that are safe, and why. Keyed by normalized SQL, so editing a query
# revokes its exemption and the check fires again -- fail-closed on edit.
ALLOWED = {
    "SELECT DISTINCT filing_stem FROM chunks": "build-time bookkeeping: which filings are already loaded, variant-agnostic by design",
    "SELECT count(*) FROM chunks": "build-time progress count, not a candidate pool",
    "SELECT filing_stem, chunk_index, variant, text FROM chunks WHERE filing_stem = ANY(%s)": "Arm 4 text fetch: returns all variants on purpose, caller keys by the (stem, index, variant) 3-tuple",
    "SELECT variant, count(*), count(embedding) FROM chunks GROUP BY variant": "the corpus assertion itself -- it must see every variant to compare them",
}


def _sql_literals(tree):
    """String constants and f-strings, with interpolated expressions dropped."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            yield node.lineno, "".join(
                v.value for v in node.values if isinstance(v, ast.Constant)
            )


def variant_predicate_violations() -> list[str]:
    """Reads of `chunks` whose WHERE clause doesn't constrain `variant`.

    Checks post-WHERE specifically: naming `variant` in the SELECT list says nothing
    about which rows come back, and two Arm 4 queries do exactly that.
    """
    violations = []
    for scan_dir in SCAN_DIRS:
        for path in sorted((ROOT / scan_dir).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path.name == "preflight.py":  # ALLOWED's keys are not real queries
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for lineno, sql in _sql_literals(tree):
                if not FROM_CHUNKS.search(sql):
                    continue
                normalized = " ".join(sql.split())
                if normalized in ALLOWED:
                    continue
                where = WHERE.split(sql, maxsplit=1)
                if len(where) > 1 and "variant" in where[1].lower():
                    continue
                violations.append(
                    f"{path.relative_to(ROOT)}:{lineno}\n      {normalized[:150]}"
                )
    return violations


def assert_variant_predicates() -> None:
    violations = variant_predicate_violations()
    if violations:
        raise RuntimeError(
            f"{len(violations)} read(s) of `chunks` without a `variant` predicate "
            "(DECISIONS.md RETR-24):\n\n  "
            + "\n\n  ".join(violations)
            + "\n\nAdd `variant = 'A'` to the WHERE clause, or -- if the query genuinely"
            "\nwants every variant -- add its normalized SQL to ALLOWED in preflight.py"
            "\nwith the reason."
        )
