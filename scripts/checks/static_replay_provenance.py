"""Fail when Arm 6's replayed static baseline was produced under a different retrieval
config than `retrieve()` runs today.

THE BUG THIS EXISTS FOR
    `07_arm6_loop.py` does not re-run retrieval for its paired baseline; it replays
    `STATIC_SCORES:STATIC_CELL`. That makes the baseline literally the published arm, which
    is the right call -- as long as the loop arm's live `retrieve()` matches the config that
    produced the replayed file.

    `RETR-43` made `year_bias` the default for `retrieve()`. The replayed ranking predates
    it. So the loop arm silently gained a retrieval improvement the baseline could not, and
    `analyze` went on printing "LIKE-FOR-LIKE" over a pair that was no longer like for like.
    Measured on the same 200 dev questions: replayed static 0.736, static WITH year_bias
    0.763, loop iteration-1 0.753 -- the loop's apparent retrieval win was the stale
    baseline, and its real result is still a loss.

    Nothing raised. That is the whole problem, and it is why this is a check and not a
    comment: the assumption was true when written and was falsified by shipping a fix
    elsewhere. Seventh instance of INFRA-22's class, first one introduced BY a correction.

WHAT IT ASSERTS
    Every retrieval flag that `retrieve()` exposes and that the replayed file's cell name
    does not account for must still sit at the value the replayed ranking was built under.
    Concretely: if `year_bias` defaults True, `STATIC_SCORES`/`STATIC_CELL` must name a
    year-bias ranking; if it defaults False, they must name the plain RETR-39 one.

    The cell NAME carries the provenance because the file cannot: `retr7_rr_dev_scores.jsonl`
    records no config block, so there is nothing else to key on. That is a weakness, not a
    design -- it is why `07_arm6_loop.py` now writes `year_bias` and `static_source` into
    every results row.

Usage:
    uv run scripts/checks/static_replay_provenance.py
"""

import importlib.util
import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.retrieve import retrieve  # noqa: E402


def _load_loop_module():
    """Import `07_arm6_loop.py` by path -- its name starts with a digit, so it is not a
    legal module name and cannot be imported normally. Importing it, rather than re-reading
    its constants, is the point: a copy here could drift from the file it guards."""
    path = _ROOT / "scripts" / "pipeline" / "07_arm6_loop.py"
    spec = importlib.util.spec_from_file_location("_arm6_loop", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    loop = _load_loop_module()
    static_scores = Path(loop.STATIC_SCORES)
    static_cell = loop.STATIC_CELL

    year_bias_default = inspect.signature(retrieve).parameters["year_bias"].default
    names_year_bias = "year_bias" in static_cell or "year_bias" in static_scores.name

    print(f"retrieve() year_bias default : {year_bias_default}")
    print(f"static replay source         : {static_scores.name}:{static_cell}")
    print(f"source names a year-bias rank: {names_year_bias}")

    if bool(year_bias_default) != names_year_bias:
        want = "a year-bias ranking" if year_bias_default else "the plain RETR-39 ranking"
        print(
            f"\nFAIL: retrieve() runs with year_bias={year_bias_default}, but the replayed "
            f"baseline is {static_scores.name}:{static_cell}, which is not {want}.\n"
            f"      The loop arm and the static arm would be on different retrieval stacks "
            f"and every paired number -- recall, nDCG, MRR, answer accuracy, McNemar -- "
            f"would attribute that difference to the loop.\n"
            f"      Fix: point STATIC_SCORES/STATIC_CELL at the matching ranking "
            f"(scripts/archive/year_bias_static_ranking.py builds the year-bias one), or "
            f"pass year_bias explicitly in the loop arm so both sides agree.",
            file=sys.stderr,
        )
        return 1

    if not static_scores.exists():
        print(f"\nFAIL: {static_scores} does not exist", file=sys.stderr)
        return 1

    print("\nok: the replayed baseline and live retrieve() agree on year_bias")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
