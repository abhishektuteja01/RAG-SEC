"""Fail when Arm 6's replayed static baseline was produced under a different retrieval
config than the loop's `retrieve()` calls actually use.

THE BUG THIS EXISTS FOR
    `07_arm6_loop.py` does not re-run retrieval for its paired baseline; it replays
    `STATIC_SCORES:STATIC_CELL`. That makes the baseline literally the published arm, which
    is the right call -- as long as the loop arm's live `retrieve()` calls match the config
    that produced the replayed file.

    `RETR-43` made `year_bias` the default for `retrieve()`. The replayed ranking predates
    it. So the loop arm silently gained a retrieval improvement the baseline could not, and
    `analyze` went on printing "LIKE-FOR-LIKE" over a pair that was no longer like for like.
    Measured on the same 200 dev questions: replayed static 0.736, static WITH year_bias
    0.763, loop iteration-1 0.753 -- the loop's apparent retrieval win was the stale
    baseline, and its real result is still a loss.

    Nothing raised. That is the whole problem, and it is why this is a check and not a
    comment: the assumption was true when written and was falsified by shipping a fix
    elsewhere. Eighth instance of INFRA-22's class (AGENT-31), first one introduced BY a
    correction. It then happened again in shape: DEPLOY-25 flipped three more defaults on,
    and the old version of this check compared `year_bias` only, so it passed.

WHAT IT ASSERTS
    For EVERY setting in `retrieve()`'s signature (all parameters but `query` and
    `resolve_from`, which are per-call inputs, not config):
      1. `07_arm6_loop.STATIC_RETRIEVE_SETTINGS` declares it -- the replay's provenance.
      2. `rag_sec.agent.ARM6_RETRIEVE_SETTINGS` pins it, so no default flip can reach the loop.
      3. The value `retrieve_node` ACTUALLY passes equals the declared provenance value.
         Captured by calling `retrieve_node` with `retrieve()` stubbed out, not by reading
         the constant -- a call site that stopped passing the constant would pass a
         constant-only check. No database, model or API is touched.
    Plus: `STATIC_CELL`'s name agrees with the declared `year_bias`, and the file exists.

    The provenance is declared, not read, because `retr7_rr_dev_scores_year_bias.jsonl`
    records no config block. That is a weakness, not a design -- it is why `07_arm6_loop.py`
    writes both settings dicts into every results row.

Usage:
    uv run scripts/checks/static_replay_provenance.py
"""

import importlib.util
import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

import rag_sec.agent as agent  # noqa: E402
from rag_sec.retrieve import retrieve  # noqa: E402

# Per-call inputs, not configuration: the loop varies `query` every iteration by design and
# passes the original question as `resolve_from` (AGENT-25).
_PER_CALL = {"query", "resolve_from"}


def _load_loop_module():
    """Import `07_arm6_loop.py` by path -- its name starts with a digit, so it is not a
    legal module name and cannot be imported normally. Importing it, rather than re-reading
    its constants, is the point: a copy here could drift from the file it guards."""
    path = _ROOT / "scripts" / "pipeline" / "07_arm6_loop.py"
    spec = importlib.util.spec_from_file_location("_arm6_loop", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kwargs_retrieve_node_passes() -> dict:
    """Run the real `retrieve_node` once with `retrieve()` replaced by a recorder."""
    seen = {}

    def _record(query, **kw):
        seen.update(kw)
        return []

    real, real_stats = agent._retrieve, agent._last_call_stats
    agent._retrieve, agent._last_call_stats = _record, dict
    try:
        msg = AIMessage(content="", tool_calls=[
            {"name": "retrieve_tool", "args": {"query": "probe"}, "id": "probe"}])
        agent.retrieve_node({"question": "probe question", "messages": [msg]})
    finally:
        agent._retrieve, agent._last_call_stats = real, real_stats
    return seen


def main() -> int:
    loop = _load_loop_module()
    static_scores = Path(loop.STATIC_SCORES)
    static_cell = loop.STATIC_CELL
    declared = loop.STATIC_RETRIEVE_SETTINGS
    pinned = agent.ARM6_RETRIEVE_SETTINGS

    params = inspect.signature(retrieve).parameters
    settings = [n for n in params if n not in _PER_CALL]
    passed = _kwargs_retrieve_node_passes()
    # what retrieve() will actually run with: the passed kwarg, else its own default
    effective = {n: passed.get(n, params[n].default) for n in settings}

    print(f"static replay source: {static_scores.name}:{static_cell}")
    print(f"{'setting':<18} {'baseline built':>15} {'loop passes':>12} {'retrieve default':>17}")
    failures = []
    for n in settings:
        want = declared.get(n, "<undeclared>")
        got = effective[n] if n in passed else f"{effective[n]} (default)"
        print(f"{n:<18} {want!s:>15} {got!s:>12} {params[n].default!s:>17}")
        if n not in declared:
            failures.append(f"{n}: STATIC_RETRIEVE_SETTINGS does not declare it, so the "
                            f"replay's provenance for it is unknown")
        if n not in pinned or n not in passed:
            failures.append(f"{n}: the loop does not pin it, so it follows retrieve()'s "
                            f"default ({params[n].default!r}) and the next default flip "
                            f"moves the loop off its baseline")
        if n in declared and effective[n] != declared[n]:
            failures.append(f"{n}: loop runs {effective[n]!r}, baseline was built under "
                            f"{declared[n]!r}")
    for n in set(declared) - set(settings):
        failures.append(f"{n}: declared in STATIC_RETRIEVE_SETTINGS but retrieve() has no "
                        f"such parameter")

    names_year_bias = "year_bias" in static_cell or "year_bias" in static_scores.name
    if declared.get("year_bias") is not None and bool(declared["year_bias"]) != names_year_bias:
        failures.append(f"year_bias: declared {declared['year_bias']} but the replay "
                        f"{static_scores.name}:{static_cell} "
                        f"{'names' if names_year_bias else 'does not name'} a year-bias ranking")
    if not static_scores.exists():
        failures.append(f"{static_scores} does not exist")

    if failures:
        print("\nFAIL: the loop arm and the replayed static arm are not on the same retrieval "
              "stack, so every paired number -- recall, nDCG, MRR, answer accuracy, McNemar -- "
              "would attribute the difference to the loop.", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print("  Fix: pin the setting in rag_sec.agent.ARM6_RETRIEVE_SETTINGS to the value "
              "the baseline was built under, or rebuild the baseline and update "
              "STATIC_RETRIEVE_SETTINGS.", file=sys.stderr)
        return 1

    print(f"\nok: all {len(settings)} retrieve() settings agree between the replayed "
          "baseline and what retrieve_node passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
