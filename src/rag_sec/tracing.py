"""The only module allowed to import `langfuse`: every caller goes through this wrapper so
the tracing backend is swappable in one file (spec.md:240, Day 10 at spec.md:387).

Two properties the consumer depends on and that shape everything below: a 4.4h / ~$11.50
benchmark run must not die because the tracing backend does, and the same run must stay
importable on machines with no Langfuse keys and no network.
"""

import atexit
import logging
import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse._client.span import LangfuseObservationWrapper

_log = logging.getLogger(__name__)

# HTTP timeout, seconds, for the OTLP exporter.
_TIMEOUT_S = 5

# Budget for flush_tracing(). Measured with scripts/checks/tracing_offline.py against a
# closed port: client.flush() blocks the *caller* for ~9s while the batch processor retries
# with backoff. So a flush that overruns its budget gives up and latches tracing off rather
# than making the caller wait on the network.
#
# Two budgets, because the two callers want opposite things. The atexit hook fires on every
# interactive script and must never hang one, so it stays short. An explicit end-of-run
# flush is the opposite case: it is called once after hours of work, it holds the only copy
# of the un-exported tail, and 3s of backoff is easily hit with a real backlog on a slow
# link -- giving up there would discard exactly what the flush exists to save.
_FLUSH_TIMEOUT_S = 3
_FLUSH_TIMEOUT_FINAL_S = 30

# Field names `LangfuseObservationWrapper.update` accepts. Anything else a caller passes to
# .set() is folded into metadata: update() swallows unknown kwargs silently, so without this
# a typo'd attribute name would just never appear in the dashboard.
_OBS_FIELDS = frozenset(
    {
        "name",
        "input",
        "output",
        "metadata",
        "version",
        "level",
        "status_message",
        "completion_start_time",
        "model",
        "model_parameters",
        "usage_details",
        "cost_details",
    }
)

_lock = threading.Lock()
_client: "Langfuse | None" = None
_init_done = False
# Set by the first failure anywhere in this module. Latching it off (rather than retrying)
# is deliberate: a broken backend at ~40 observations per question would otherwise log and
# retry ~8000 times in one run.
_degraded = False


def _disable(context: str, exc: Exception) -> None:
    global _degraded, _client
    if not _degraded:
        _degraded = True
        _client = None
        _log.warning("tracing disabled after failure in %s: %r", context, exc)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("0", "false", "no", "off")


def init_tracing() -> bool:
    """Idempotent, thread-safe. True if tracing is live.

    Returns False -- without importing `langfuse` -- when keys are missing or tracing is
    switched off, so the offline scripts that import the pipeline pay nothing for this.
    """
    global _client, _init_done

    with _lock:
        if _init_done:
            return _client is not None
        _init_done = True

        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            return False
        if not _env_flag("LANGFUSE_TRACING_ENABLED"):
            return False

        try:
            from langfuse import Langfuse

            # Everything else (base_url/host, sample_rate, flush_at, flush_interval, debug)
            # is read from LANGFUSE_* env by the SDK itself; not re-plumbed here.
            #
            # No `mask=`: everything traced here is a public SEC filing chunk or a question
            # from an open benchmark, so there is nothing to redact. Revisit before pointing
            # this pipeline at a private corpus or at user-supplied questions.
            _client = Langfuse(timeout=_TIMEOUT_S)
            atexit.register(flush_tracing)
        except Exception as exc:
            _disable("init_tracing", exc)
            return False

        return True


def tracing_enabled() -> bool:
    # Fast path without the lock: the disabled case is on the import path of every offline
    # script, and taking a mutex per observation at ~40 per question is pointless once the
    # answer can no longer change. init_tracing() double-checks under the lock.
    if _init_done:
        return _client is not None and not _degraded
    return init_tracing() and not _degraded


class Handle(Protocol):
    """What callers may do to an open observation."""

    def set(self, **attrs: object) -> None: ...

    def set_usage(
        self,
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None: ...


class _NullHandle:
    def set(self, **attrs: object) -> None:
        pass

    def set_usage(
        self,
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        pass


_NULL: Handle = _NullHandle()


class _LiveHandle:
    __slots__ = ("_obs",)

    def __init__(self, obs: "LangfuseObservationWrapper") -> None:
        self._obs = obs

    def set(self, **attrs: object) -> None:
        try:
            self._obs.update(**_route(attrs))
        except Exception as exc:
            _disable("Handle.set", exc)

    def set_usage(
        self,
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Maps onto Langfuse's `usage_details` keys, which is what its cost math prices.

        Cached prompt tokens go to `input_cache_read` -- the key Langfuse's own LangChain
        handler produces from langchain's `input_token_details.cache_read`, so a model's
        `input_cache_read` price on the server applies. That handler also subtracts the
        cached count out of `input`, i.e. Langfuse treats the sub-keys as exclusive
        categories; agent.py's `input_tokens` is langchain's inclusive number, so it has to
        be subtracted here or cached tokens get billed at the full input rate as well.
        """
        usage: dict[str, int] = {}
        cached = cached_input_tokens or 0
        if input_tokens is not None:
            usage["input"] = max(0, input_tokens - cached)
        if cached:
            usage["input_cache_read"] = cached
        if output_tokens is not None:
            usage["output"] = output_tokens
        if total_tokens is not None:
            usage["total"] = total_tokens
        if usage:
            self.set(usage_details=usage)


def _route(attrs: dict[str, object]) -> dict[str, object]:
    known = {k: v for k, v in attrs.items() if k in _OBS_FIELDS}
    extra = {k: v for k, v in attrs.items() if k not in _OBS_FIELDS}
    if extra:
        meta = known.get("metadata")
        known["metadata"] = {**meta, **extra} if isinstance(meta, dict) else extra
    return known


# The subset of Langfuse's nine observation types this pipeline emits. Only `generation`
# and `embedding` accept `model`/`usage_details`, which is why the sufficiency judge stays a
# `generation` rather than the semantically closer `evaluator`: judge calls are ~22% of this
# arm's spend and per-node cost is what the run measures.
_ObsType = Literal["agent", "embedding", "generation", "retriever", "span"]


def _close(cm: AbstractContextManager, exc: BaseException | None) -> None:
    """`cm.__exit__`, with its own failure degraded rather than raised."""
    try:
        if exc is None:
            cm.__exit__(None, None, None)
        else:
            cm.__exit__(type(exc), exc, exc.__traceback__)
    except Exception as exit_exc:
        _disable("observation.__exit__", exit_exc)


@contextmanager
def _observation(
    name: str,
    as_type: _ObsType,
    attrs: dict[str, object],
    trace_id: str | None = None,
) -> Iterator[Handle]:
    """Shared body of every public constructor below.

    Opens the Langfuse context manager and yields a handle. A raising body is marked
    ERROR and the original exception re-raised untouched -- swallowing it would turn a
    failed run into a silently successful one.
    """
    client = _client if tracing_enabled() else None
    if client is None:
        yield _NULL
        return

    try:
        cm = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            trace_context={"trace_id": trace_id} if trace_id else None,
            **_route(attrs),  # type: ignore[arg-type]
        )
    except Exception as exc:
        _disable("start_as_current_observation", exc)
        yield _NULL
        return

    # `cm` is driven by hand rather than with `with`: a raise from the SDK's __enter__ or
    # __exit__ would otherwise escape question_trace, and in the Day 9 runner that reaches
    # fut.result() and kills an hours-long run -- the exact failure this module exists to
    # prevent. Only the tracing machinery is degraded here; the body's own exception is
    # re-raised untouched below, since swallowing it would turn a failed run into a
    # silently successful one.
    try:
        obs = cm.__enter__()
    except Exception as exc:
        _disable("observation.__enter__", exc)
        yield _NULL
        return

    handle = _LiveHandle(obs)
    try:
        yield handle
    # BaseException, not Exception: a Ctrl-C'd or sys.exit'd run should still show up
    # as errored rather than as a trace that just stops. Everywhere else in this module
    # catches Exception, because those handlers swallow.
    except BaseException as exc:
        handle.set(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        _close(cm, exc)
        # unconditional, and outside _close: a broken __exit__ must not be able to replace or
        # suppress the caller's exception, which is the one thing that must survive intact.
        raise
    else:
        _close(cm, None)


@contextmanager
def _propagate(**attrs: object) -> Iterator[None]:
    """Attach session id / tags / propagated metadata to the enclosing observation and every
    observation opened inside this block.

    The only route to tags in SDK v4: they are immutable after creation, and neither
    `start_as_current_observation` nor `update` accepts them.
    """
    client = _client if tracing_enabled() else None
    if client is None:
        yield
        return

    try:
        from langfuse import propagate_attributes

        cm = propagate_attributes(**attrs)  # type: ignore[arg-type]
    except Exception as exc:
        _disable("propagate_attributes", exc)
        yield
        return

    with cm:
        yield


@contextmanager
def _root_context() -> Iterator[None]:
    """Run the body in an empty OTel context, so a trace root inherits nothing.

    Needed because the Day 9 runner opens one arm's trace inside the other's `with` block,
    and propagated tags MERGE downward rather than replace (langfuse 4.15.1,
    `_client/propagation.py:_set_propagated_attribute`). Without this the inner trace would
    also carry the outer arm's `arm:` tag, and no dashboard could split the two arms apart.
    """
    if not tracing_enabled():
        yield
        return

    try:
        from opentelemetry import context as otel_context

        token = otel_context.attach(otel_context.Context())
    except Exception as exc:
        _disable("_root_context", exc)
        yield
        return

    try:
        yield
    finally:
        otel_context.detach(token)


# Observation names are an API: Langfuse evaluators, dashboard queries and saved views all
# target observations by name, so a rename silently stops them matching. Both arms therefore
# share this root name and are told apart by their `arm:` tag -- naming them apart would
# leave no single metric a dashboard could split by arm.
ROOT_NAME = "answer-question"


@contextmanager
def question_trace(
    qid: str,
    arm: str,
    *,
    question: str | None = None,
    tags: Sequence[str] = (),
    **metadata: object,
) -> Iterator[Handle]:
    """Root observation for one arm's attempt at one benchmark question.

    One trace per arm, not one per question: Langfuse sums cost over a trace, so two arms
    sharing a trace report only their combined spend -- and the difference between them is
    the entire measurement. `session_id` is the question id, which is what re-pairs them.

    The trace id is seeded from qid AND arm, so re-running a question after a resume lands in
    the same two traces instead of forking new ones. It also means a worker thread can rejoin
    an existing trace by re-entering this with the same arguments: OTel context lives in
    contextvars, which threads do NOT inherit, so a span opened in another thread would
    otherwise start its own root trace.
    """
    # `input`/`output` on the root are what the tracing table, evaluators and dataset
    # experiments read. Only the question and the final answer go here; every other
    # per-question value is metadata.
    attrs: dict[str, object] = {"metadata": {"qid": qid, "arm": arm, **metadata}}
    if question is not None:
        attrs["input"] = question

    with _root_context(), _observation(ROOT_NAME, "agent", attrs, trace_id_for(qid, arm)) as h:
        # Entered inside the root so the root is tagged too: propagated attributes apply to
        # the active observation and later children only, so a late call would drop the
        # earliest observations out of every session and tag aggregation.
        with _propagate(session_id=qid, tags=[f"arm:{arm}", *tags], metadata={"qid": qid}):
            yield h


@contextmanager
def span(name: str, **attrs: object) -> Iterator[Handle]:
    """Nests under whatever observation is active on this thread."""
    with _observation(name, "span", attrs) as h:
        yield h


@contextmanager
def generation(name: str, model: str | None = None, **attrs: object) -> Iterator[Handle]:
    """An LLM call: priced by Langfuse from `model` plus whatever set_usage records."""
    if model is not None:
        attrs["model"] = model
    with _observation(name, "generation", attrs) as h:
        yield h


@contextmanager
def retriever(name: str, **attrs: object) -> Iterator[Handle]:
    """A lookup that returns context and changes nothing."""
    with _observation(name, "retriever", attrs) as h:
        yield h


@contextmanager
def embedding(name: str, model: str | None = None, **attrs: object) -> Iterator[Handle]:
    """Encoding text into a vector. Carries `model` but never usage: the encoder runs
    locally, so there are no billable tokens for Langfuse to price."""
    if model is not None:
        attrs["model"] = model
    with _observation(name, "embedding", attrs) as h:
        yield h


def flush_tracing(timeout_s: float = _FLUSH_TIMEOUT_S) -> None:
    """Push buffered observations. Idempotent, and the atexit hook -- observations are
    batched, so a killed process loses its tail without this.

    Flushed on a daemon thread and joined with a timeout: see _FLUSH_TIMEOUT_S for why the
    caller must not be the one waiting on the network. An entry point that owns the whole
    process should pass `_FLUSH_TIMEOUT_FINAL_S`.
    """
    client = _client
    if client is None:
        return

    done = threading.Event()

    def _flush() -> None:
        try:
            client.flush()
        except Exception as exc:
            _disable("flush_tracing", exc)
        finally:
            done.set()

    threading.Thread(target=_flush, name="langfuse-flush", daemon=True).start()
    if not done.wait(timeout_s):
        _disable("flush_tracing", TimeoutError(f"flush exceeded {timeout_s}s"))


def trace_id_for(qid: str, arm: str) -> str | None:
    """The trace id `question_trace(qid, arm)` will use, for writing into result rows. None
    when tracing is off. `arm` is not optional: one arm silently reusing the other's id would
    merge the two traces this split exists to keep apart."""
    client = _client if tracing_enabled() else None
    if client is None:
        return None
    try:
        return client.create_trace_id(seed=f"{qid}:{arm}")
    except Exception as exc:
        _disable("trace_id_for", exc)
        return None
