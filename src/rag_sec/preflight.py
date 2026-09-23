"""AST scan that fails any read of `chunks` whose WHERE clause omits `variant`.

The table has a `variant` column (only 'A' is live). A read without a `variant` predicate
would silently mix in any other rows loaded later. Runs on the first `get_conn()`.
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
    about which rows come back.
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
            except (SyntaxError, UnicodeDecodeError):
                # macOS tar writes binary AppleDouble `._name.py` sidecars that match
                # rglob("*.py"). Uncaught, this crashed the container at startup. Python
                # source is UTF-8, so an undecodable file is not a module this could miss.
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
            f"{len(violations)} read(s) of `chunks` without a `variant` predicate:\n\n  "
            + "\n\n  ".join(violations)
            + "\n\nAdd `variant = 'A'` to the WHERE clause, or -- if the query genuinely"
            "\nwants every variant -- add its normalized SQL to ALLOWED in preflight.py"
            "\nwith the reason."
        )
