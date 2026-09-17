"""Fails when the laptop's cell table and the GPU leg's cell table disagree.

`05_arm3_rerank.py` decides which candidate pool each cell gets (`POOL_GROUP`, and the
`cands_<group>` keys it writes into the payload). `hpc/rerank_hpc.py` decides which pool each
cell READS (`CELL_SPEC`). The two files cannot import each other -- the GPU node has no
`rag_sec` and no repo checkout, which is the whole point of `ARM3-2` -- so the agreement is
maintained by hand and has to be checked by machine.

A mismatch does not crash. It reranks one arm against another arm's candidates and writes a
plausible number, which is `INFRA-22`'s shape: a consumer assuming a producer's layout that
nothing enforces. That is worth a guard rather than a comment, per scripts/CLAUDE.md.

Also checks the rerank-query field: every arms cell must score against `question_stripped`,
since the arms differ in how candidates were CHOSEN, never in what the cross-encoder is asked.

Usage:
    python scripts/checks/cell_config.py
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    arm3 = _load(_ROOT / "scripts/pipeline/05_arm3_rerank.py", "arm3")
    hpc = _load(_ROOT / "scripts/pipeline/hpc/rerank_hpc.py", "rerank_hpc")

    failures = []
    for cell, group in arm3.POOL_GROUP.items():
        if cell not in hpc.CELL_SPEC:
            failures.append(f"{cell!r}: in POOL_GROUP but missing from rerank_hpc.CELL_SPEC "
                            "-- `prepare` would write a pool the GPU leg never scores")
            continue
        cands_key, q_field = hpc.CELL_SPEC[cell]
        if cands_key != f"cands_{group}":
            failures.append(f"{cell!r}: prepare writes 'cands_{group}', GPU leg reads "
                            f"{cands_key!r} -- this arm would be reranked against another "
                            "arm's candidates, with no error")
        want_stripped = arm3.CELL_FLAGS[cell]["strip_query"]
        got_stripped = q_field == "question_stripped"
        if want_stripped != got_stripped:
            failures.append(f"{cell!r}: CELL_FLAGS says strip_query={want_stripped}, GPU leg "
                            f"scores against {q_field!r}")
    for cell in hpc.CELL_SPEC:
        if cell not in arm3.POOL_GROUP:
            failures.append(f"{cell!r}: in rerank_hpc.CELL_SPEC but unknown to POOL_GROUP")

    # The arms differ in candidate SELECTION only; if one ever reranked against raw text its
    # delta against `deployed` would be measuring RETR-6, not the arm.
    for cell in arm3.ARM_CELLS:
        if not arm3.CELL_FLAGS[cell]["strip_query"]:
            failures.append(f"arms cell {cell!r} has strip_query=False -- its delta would "
                            "confound RETR-6's query strip with the arm being tested")

    if failures:
        print("cell configuration disagrees between the laptop and GPU legs:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}\n", file=sys.stderr)
        return 1
    print(f"ok: {len(arm3.POOL_GROUP)} cells agree across 05_arm3_rerank and rerank_hpc "
          f"({len(set(arm3.POOL_GROUP.values()))} candidate pools)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
