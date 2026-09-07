"""Offline contract check for rag_sec.tracing -- no Langfuse keys, no network.

Each case runs in a fresh subprocess: rag_sec.tracing latches its enabled/disabled decision
once per process (that is the point of the module), so the disabled and enabled cases cannot
share one interpreter. Exits non-zero on the first failure so it can go in CI.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Each case is a subprocess, and a subprocess does not inherit the parent's sys.path edits.
# Locally that goes unnoticed because the project is pip-installed into .venv; CI builds its
# venv from `uv export --no-emit-project`, so rag_sec is not installed there and every case
# died on ModuleNotFoundError. Handing src down explicitly makes the check independent of
# whether the project happens to be installed.
_SRC = str(Path(__file__).resolve().parents[2] / "src")

# Port 1 is privileged and never listening; a fake key set turns tracing on so the enabled
# cases exercise the real SDK against a refused connection instead of a live backend.
ENABLED_ENV = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-offline-check",
    "LANGFUSE_SECRET_KEY": "sk-lf-offline-check",
    "LANGFUSE_BASE_URL": "http://127.0.0.1:1",
    "LANGFUSE_TRACING_ENABLED": "true",
    # Bounds the OTel batch processor's export attempts so the SDK's own atexit shutdown
    # against the closed port does not dominate this script's runtime. It shapes the test,
    # not rag_sec.tracing -- the caller-blocking assertion in case (b) is the real check.
    "OTEL_BSP_EXPORT_TIMEOUT": "1000",
}

DISABLED_ENV = {
    k: "" for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_TRACING_ENABLED")
}

CASES: dict[str, tuple[str, dict[str, str], str]] = {}


def case(name: str, env: dict[str, str], body: str) -> None:
    CASES[name] = (name, env, body)


case(
    "a) disabled: full API works, langfuse never imported",
    DISABLED_ENV,
    """
import sys
from rag_sec import tracing

assert tracing.init_tracing() is False
assert tracing.tracing_enabled() is False
with tracing.question_trace("q1", "loop", question="q?", tags=["split:dev"], split="dev") as t:
    t.set(output="answer", level="DEFAULT", custom_attr=1)
    with tracing.retriever("retrieve-chunks", input="query") as s:
        s.set(output=[1, 2, 3])
    with tracing.embedding("embed-query", model="bge-small") as e:
        e.set(output="ok")
    with tracing.span("resolve-company") as s:
        s.set(output={"tickers": []})
    with tracing.generation("plan-query", model="gemini-3.7-flash") as g:
        g.set_usage(input_tokens=100, cached_input_tokens=40, output_tokens=10, total_tokens=110)
tracing.flush_tracing()
tracing.flush_tracing()
assert tracing.trace_id_for("q1", "loop") is None
leaked = [m for m in sys.modules if m == "langfuse" or m.startswith("langfuse.")]
assert not leaked, f"langfuse imported while disabled: {leaked}"
""",
)

case(
    "b) enabled against a closed port: full API completes, no hang",
    ENABLED_ENV,
    """
import time

from rag_sec import tracing

assert tracing.init_tracing() is True
assert tracing.tracing_enabled() is True
started = time.monotonic()
with tracing.question_trace("q1", "loop", question="q?", tags=["split:dev"]) as t:
    with tracing.retriever("retrieve-chunks", input="query") as s:
        s.set(output=[1, 2, 3], recall_at_50=0.9)
    with tracing.generation("plan-query", model="gemini-3.7-flash") as g:
        g.set(output="plan text")
        g.set_usage(input_tokens=100, cached_input_tokens=40, output_tokens=10, total_tokens=110)
    t.set(output="answer")
tracing.flush_tracing()
# The caller's own time, excluding interpreter+SDK import. Observations themselves never
# touch the network; the only place a refused backend can block the question loop is
# flush_tracing(), and that is bounded by the module's own budget.
elapsed = time.monotonic() - started
budget = tracing._FLUSH_TIMEOUT_S + 1.0
assert elapsed < budget, f"tracing blocked the caller for {elapsed:.2f}s"
# Nothing is asserted about tracing staying enabled either way: whether the bounded flush
# returns in time or is abandoned depends on the exporter's retry backoff, and OTel's
# force_flush reports export failure by return value that Langfuse's flush() drops -- so a
# silently failed export is invisible to this module by construction.
""",
)

case(
    "c-disabled) exception inside span() propagates unchanged",
    DISABLED_ENV,
    """
from rag_sec import tracing

class Boom(RuntimeError):
    pass

try:
    with tracing.span("retrieve"):
        raise Boom("exact message")
except Boom as exc:
    assert type(exc) is Boom, type(exc)
    assert str(exc) == "exact message", str(exc)
else:
    raise AssertionError("exception was swallowed")
""",
)

case(
    "c-enabled) exception inside span() propagates unchanged",
    ENABLED_ENV,
    """
from rag_sec import tracing

class Boom(RuntimeError):
    pass

assert tracing.init_tracing() is True
try:
    with tracing.question_trace("q1", "loop"):
        with tracing.span("resolve-company"):
            raise Boom("exact message")
except Boom as exc:
    assert type(exc) is Boom, type(exc)
    assert str(exc) == "exact message", str(exc)
else:
    raise AssertionError("exception was swallowed")
""",
)

case(
    "d) trace ids are deterministic in qid AND arm",
    ENABLED_ENV,
    """
from rag_sec import tracing

assert tracing.init_tracing() is True
a1 = tracing.trace_id_for("dev-0042", "loop")
a2 = tracing.trace_id_for("dev-0042", "loop")
b = tracing.trace_id_for("dev-0043", "loop")
# the arm discriminator is the whole point: without it both arms would land in one trace
# and trace-level cost would be their sum, which is what the run is trying to compare
st = tracing.trace_id_for("dev-0042", "static")
assert a1 == a2, (a1, a2)
assert a1 != b, (a1, b)
assert a1 != st, (a1, st)
assert isinstance(a1, str) and len(a1) == 32, a1
""",
)

case(
    "e) worker-thread spans nest inside the question's trace",
    ENABLED_ENV,
    """
from concurrent.futures import ThreadPoolExecutor

from langfuse import get_client

from rag_sec import tracing

assert tracing.init_tracing() is True
client = get_client()

def work(qid):
    # question_trace is re-entered in the worker on purpose: OTel context is contextvar
    # based and threads do not inherit it, so this is how a worker rejoins one trace.
    with tracing.question_trace(qid, "loop"):
        root = client.get_current_trace_id()
        with tracing.retriever("retrieve-chunks"):
            inner = client.get_current_trace_id()
        return root, inner

with ThreadPoolExecutor(max_workers=4) as pool:
    got = list(pool.map(work, [f"dev-{i}" for i in range(4)]))

for qid, (root, inner) in zip([f"dev-{i}" for i in range(4)], got):
    expected = tracing.trace_id_for(qid, "loop")
    assert root == expected, (qid, root, expected)
    assert inner == root, (qid, inner, root)

# The negative half of the same claim: a worker that does NOT re-enter question_trace
# starts its own trace, because thread stacks do not inherit contextvars.
with tracing.question_trace("dev-main", "loop"):
    main_trace = client.get_current_trace_id()
    with ThreadPoolExecutor(max_workers=1) as pool:
        orphan = pool.submit(lambda: (tracing.span("orphan").__enter__(), client.get_current_trace_id())[1]).result()
assert orphan != main_trace, (orphan, main_trace)

# Same-thread nesting still shares the trace with its parent.
with tracing.question_trace("dev-0", "loop"):
    outer = client.get_current_trace_id()
    with tracing.span("a"):
        with tracing.span("b"):
            assert client.get_current_trace_id() == outer
tracing.flush_tracing()
""",
)

case(
    "f) emitted spans carry the right ids, types, trace io, tags and usage keys",
    ENABLED_ENV,
    """
import json

import langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Swapping in an in-memory exporter before init_tracing() imports the SDK lets this assert
# on what would go over the wire, with no backend at all. The wrapper resolves
# `langfuse.Langfuse` at init time, so patching the attribute is enough.
exporter = InMemorySpanExporter()
_real = langfuse.Langfuse
langfuse.Langfuse = lambda **kw: _real(**kw, span_exporter=exporter)

from rag_sec import tracing

class Boom(RuntimeError):
    pass

assert tracing.init_tracing() is True
try:
    with tracing.question_trace(
        "dev-7", "loop", question="how much?", tags=["split:dev", "source:finqa"], split="dev"
    ) as tr:
        with tracing.generation("plan-query", model="gemini-3.7-flash") as g:
            g.set_usage(
                input_tokens=100, cached_input_tokens=40, output_tokens=10, total_tokens=110
            )
        with tracing.embedding("embed-query", model="bge-small-en-v1.5", input="q"):
            pass
        tr.set(output="ANSWER: 12")
        with tracing.retriever("retrieve-chunks"):
            raise Boom("boom")
except Boom:
    pass
tracing.flush_tracing()

spans = {s.name: s for s in exporter.get_finished_spans()}
assert set(spans) == {"answer-question", "plan-query", "embed-query", "retrieve-chunks"}, sorted(spans)

root = spans["answer-question"]
assert format(root.context.trace_id, "032x") == tracing.trace_id_for("dev-7", "loop")
for child in ("plan-query", "embed-query", "retrieve-chunks"):
    assert spans[child].parent.span_id == root.context.span_id, child
    assert spans[child].context.trace_id == root.context.trace_id, child

types = {n: s.attributes["langfuse.observation.type"] for n, s in spans.items()}
assert types == {
    "answer-question": "agent",
    "plan-query": "generation",
    "embed-query": "embedding",
    "retrieve-chunks": "retriever",
}, types

# trace-level input/output come off the root observation; empty here is what makes the
# tracing table, evaluators and dataset experiments render blank.
assert root.attributes["langfuse.observation.input"] == "how much?", root.attributes
assert root.attributes["langfuse.observation.output"] == "ANSWER: 12", root.attributes

# session + tags must sit on the root AND on every child: Langfuse aggregations only count
# observations that carry the attribute, so a late propagate silently under-reports.
for name, sp in spans.items():
    assert sp.attributes["session.id"] == "dev-7", (name, sp.attributes)
    assert set(sp.attributes["langfuse.trace.tags"]) == {
        "arm:loop", "split:dev", "source:finqa"
    }, (name, sp.attributes)
    assert sp.attributes["langfuse.trace.metadata.qid"] == "dev-7", name

attrs = spans["retrieve-chunks"].attributes
assert attrs["langfuse.observation.level"] == "ERROR", attrs
assert attrs["langfuse.observation.status_message"] == "Boom: boom", attrs

usage = json.loads(spans["plan-query"].attributes["langfuse.observation.usage_details"])
# input is exclusive of the cached read, matching how Langfuse prices the sub-keys.
assert usage == {"input": 60, "input_cache_read": 40, "output": 10, "total": 110}, usage
assert spans["plan-query"].attributes["langfuse.observation.model.name"] == "gemini-3.7-flash"
# embedding is the only non-generation type that carries a model name at all
assert spans["embed-query"].attributes["langfuse.observation.model.name"] == "bge-small-en-v1.5"
assert spans["answer-question"].attributes["langfuse.observation.metadata.split"] == "dev"
""",
)

case(
    "g) the two arms are separate traces in one session, with disjoint arm tags",
    ENABLED_ENV,
    """
import langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
_real = langfuse.Langfuse
langfuse.Langfuse = lambda **kw: _real(**kw, span_exporter=exporter)

from rag_sec import tracing

assert tracing.init_tracing() is True

# One arm's trace opened INSIDE the other's block. The Day 9 runner no longer nests them
# (the static answer call was inflating the loop root's duration), but the empty-context
# reset this asserts is what makes nesting safe at all, so the property stays pinned.
with tracing.question_trace("dev-7", "loop", question="how much?", tags=["split:dev"]) as loop:
    with tracing.generation("generate-answer", model="gemini-3.7-flash"):
        pass
    loop.set(output="loop answer")
    with tracing.question_trace(
        "dev-7", "static", question="how much?", tags=["split:dev"]
    ) as static:
        with tracing.generation("generate-answer", model="gemini-3.7-flash"):
            pass
        static.set(output="static answer")
tracing.flush_tracing()

spans = exporter.get_finished_spans()
# `as_root`, not `parent is None`: a seeded trace id is passed as a remote parent context,
# so both roots do have a parent span context and only this flag marks them as trace roots.
roots = {s.attributes["langfuse.observation.metadata.arm"]: s for s in spans
         if s.attributes.get("langfuse.internal.as_root")}
assert set(roots) == {"loop", "static"}, sorted(roots)
assert roots["loop"].name == roots["static"].name == tracing.ROOT_NAME

# separate traces: one trace per arm, because Langfuse sums cost over a trace
assert format(roots["loop"].context.trace_id, "032x") == tracing.trace_id_for("dev-7", "loop")
assert format(roots["static"].context.trace_id, "032x") == tracing.trace_id_for("dev-7", "static")
assert roots["loop"].context.trace_id != roots["static"].context.trace_id

# ...re-paired by the session, whose id is the question id
assert all(s.attributes["session.id"] == "dev-7" for s in spans), [s.name for s in spans]

# and told apart by tag. Propagated tags MERGE downward in the SDK, so the nested static
# trace would inherit `arm:loop` too without question_trace's empty-context reset.
for arm, root in roots.items():
    tags = [set(s.attributes["langfuse.trace.tags"]) for s in spans
            if s.context.trace_id == root.context.trace_id]
    assert tags and all(t == {"split:dev", f"arm:{arm}"} for t in tags), (arm, tags)
""",
)

case(
    "h) a raising tracing backend degrades; the caller's own exception does not change",
    ENABLED_ENV,
    """
from rag_sec import tracing

class Boom(RuntimeError):
    pass

class Sabotage(RuntimeError):
    pass

assert tracing.init_tracing() is True

class _CM:
    '''Stands in for the SDK's observation context manager, raising where langfuse/OTel can:
    __enter__ and __exit__ are the two points _observation used to leave unguarded.'''
    def __init__(self, on_enter=False, on_exit=False):
        self.on_enter, self.on_exit = on_enter, on_exit
    def __enter__(self):
        if self.on_enter:
            raise Sabotage("enter")
        return self
    def __exit__(self, *a):
        if self.on_exit:
            raise Sabotage("exit")
        return False
    def update(self, **kw):
        pass

# _disable() nulls _client, so every case has to put it back -- that latch-off IS the
# degrade path under test, not incidental setup.
_client = tracing._client

def patch(**kw):
    tracing._degraded = False
    tracing._client = _client
    _client.start_as_current_observation = lambda **_: _CM(**kw)

# 1. __enter__ raises -> no-op handle, body still runs, nothing propagates
patch(on_enter=True)
ran = []
with tracing.span("s") as h:
    h.set(output="x")
    ran.append(True)
assert ran == [True]
assert tracing._degraded is True, "an __enter__ failure must latch tracing off"

# 2. __exit__ raises on a CLEAN body -> swallowed
patch(on_exit=True)
with tracing.span("s"):
    pass
assert tracing._degraded is True

# 3. __exit__ raises while the BODY is also raising -> the body's exception wins, unchanged.
#    This is the distinction that matters: tracing failures degrade, caller failures do not.
patch(on_exit=True)
try:
    with tracing.span("s"):
        raise Boom("exact message")
except Sabotage:
    raise AssertionError("__exit__'s exception replaced the caller's")
except Boom as exc:
    assert type(exc) is Boom, type(exc)
    assert str(exc) == "exact message", str(exc)
else:
    raise AssertionError("exception was swallowed")

# 4. construction raises -> already guarded, still no-ops
patch()
def _boom(**_):
    raise Sabotage("construct")
_client.start_as_current_observation = _boom
with tracing.span("s") as h:
    h.set(output="x")
assert tracing._degraded is True

# 5. and the guards must not swallow the body when tracing is HEALTHY-looking either
patch()
try:
    with tracing.span("s"):
        raise Boom("still exact")
except Boom as exc:
    assert str(exc) == "still exact", str(exc)
else:
    raise AssertionError("exception was swallowed")
""",
)

BUDGET_S = 30.0


def main() -> int:
    failures = 0
    for name, env, body in CASES.values():
        child = {**os.environ, **env}
        child["PYTHONPATH"] = os.pathsep.join(
            [_SRC, *([child["PYTHONPATH"]] if child.get("PYTHONPATH") else [])]
        )
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", body],
            env=child,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.monotonic() - started
        ok = proc.returncode == 0 and elapsed < BUDGET_S
        print(f"{'PASS' if ok else 'FAIL'} {name}  ({elapsed:.2f}s)")
        if not ok:
            failures += 1
            if elapsed >= BUDGET_S:
                print(f"  exceeded the {BUDGET_S:.0f}s budget -- tracing is blocking the caller")
            print("  " + (proc.stderr.strip() or proc.stdout.strip()).replace("\n", "\n  "))
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
