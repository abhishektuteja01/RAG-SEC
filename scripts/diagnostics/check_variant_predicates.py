"""CLI over rag_sec.preflight's source scan -- for CI or a manual sweep.

The same check runs automatically on the first get_conn(); this is for checking without
a database, or before committing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rag_sec.preflight import variant_predicate_violations  # noqa: E402


def main() -> int:
    violations = variant_predicate_violations()
    if violations:
        print(
            f"{len(violations)} read(s) of `chunks` without a `variant` predicate "
            "(DECISIONS.md RETR-24):\n",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}\n", file=sys.stderr)
        return 1
    print("ok: every read of `chunks` constrains `variant`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
