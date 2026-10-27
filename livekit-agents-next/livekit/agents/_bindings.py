"""The hand-written method bodies of the generated models (``_generated/*.py`` call them)."""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from . import _core
from ._exceptions import APIConnectionError, APIError, APIStatusError
from .log import logger

if TYPE_CHECKING:
    from ._generated.events import FunctionToolsExecutedEvent
    from .llm import FunctionCall, FunctionCallOutput


_T = TypeVar("_T")


def on_loop(owner: Any, build: Callable[[], _T]) -> _T:
    """``build()`` once per running event loop: a core object over host providers calls them on
    the loop it was built on, so an object shared by two loops (two jobs on threads, per-test
    loops) gets one each."""
    built = owner.__dict__.setdefault("_core_by_loop", weakref.WeakKeyDictionary())
    loop = asyncio.get_running_loop()
    if (native := built.get(loop)) is None:
        native = built[loop] = build()
    return native  # type: ignore[no-any-return]


def built_on_loop(owner: Any) -> list[Any]:
    """What ``on_loop`` built for ``owner``, the running loop's first."""
    built = dict(owner.__dict__.get("_core_by_loop", {}))
    try:
        mine = built.pop(asyncio.get_running_loop(), None)
    except RuntimeError:
        mine = None
    return ([mine] if mine is not None else []) + list(built.values())


def core_leg(provider: Any) -> Any:
    """The provider as the core runs it: its own core object when it has one (``_core_leg``),
    else the provider itself, which the core calls through its plugin surface."""
    leg = getattr(provider, "_core_leg", None)
    return leg() if leg is not None else provider


def overrides(obj: Any, base: type, *names: str) -> bool:
    """Whether ``obj``'s class replaces any of ``base``'s ``names``: the core object would bypass
    it, so the provider is run through its plugin surface instead."""
    return any(getattr(type(obj), n) is not getattr(base, n) for n in names)


class Attempts:
    """A provider call's attempts under ``conn``, retried as the core decides
    (``_core.provider_retry``), each failure an ``error`` event.

    A call the framework opened itself (``core``) runs once and reports nothing: the core retries
    it and reports its failures to the provider.
    """

    def __init__(self, kind: str, conn: Any, emit_error: Callable[[Exception, bool], None]) -> None:
        self.kind = kind
        self.conn = conn
        self.emit_error = emit_error
        self.core = False
        self.count = 0
        """Failed attempts since the last success; a stream resets it on a final transcript."""

    async def run(
        self,
        attempt: Callable[[], Awaitable[_T]],
        *,
        produced: Callable[[], bool] = lambda: False,
        again: Callable[[], None] = lambda: None,
    ) -> _T:
        """``attempt`` until it succeeds; ``produced``: the failed attempt delivered output,
        ``again``: called before each retry."""
        while True:
            try:
                return await attempt()
            except Exception as e:
                action, delay, message = _core.provider_retry(
                    self.kind,
                    0 if self.core else self.conn.max_retry,
                    self.conn.retry_interval,
                    self.count,
                    isinstance(e, APIError) and e.retryable,
                    e.status_code if isinstance(e, APIStatusError) else None,
                    produced(),
                )
                if action == "stop":
                    return cast(_T, None)
                if self.core:
                    raise
                self.emit_error(e, action == "retry")
                if action == "exhausted":
                    raise APIConnectionError(message or str(e)) from e
                if action == "raise":
                    raise
                logger.warning(f"{self.kind} attempt failed: {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
                self.count += 1
                again()


async def pump(
    native: Any,
    items: AsyncIterable[Any],
    feed: Callable[[Any], None],
    out: Callable[[Any], None],
) -> None:
    """A core stream fed ``items`` (each through ``feed``, then its end of input) while ``out``
    takes what it yields, until it ends. A feed that fails ends the stream with its error rather
    than leaving it waiting for an end of input; the stream is closed either way."""
    from .utils import aio

    async def _feed() -> None:
        async for item in items:
            feed(item)
        native.end_input()

    async def _drain() -> None:
        while (item := await native.recv()) is not None:
            out(item)

    feeding, draining = asyncio.create_task(_feed()), asyncio.create_task(_drain())
    try:
        await asyncio.wait({feeding, draining}, return_when=asyncio.FIRST_EXCEPTION)
        if feeding.done() and not feeding.cancelled() and (error := feeding.exception()):
            raise error
        await draining
    finally:
        await aio.cancel_and_wait(feeding, draining)
        await native.aclose()


def metrics_context(provider: Any, *, cancelled: bool = False) -> tuple[float, str, bool, str, str]:
    """What a provider's metrics are stamped with: now, its label, model and provider."""
    return (time.time(), provider.label, cancelled, provider.model, provider.provider)


def emit_metrics(provider: Any, kind: str, record: dict[str, Any] | None) -> None:
    """A request's metrics, folded by the core, as the provider's ``metrics_collected`` event."""
    if record is None:
        return
    from ._generated import metrics

    provider.emit("metrics_collected", getattr(metrics, kind).model_validate(record))


async def recognize(stt: Any, buffer: Any, *, language: Any, conn_options: Any) -> Any:
    """``STT.recognize``: ``_recognize_impl`` in attempts, its metrics once it answers."""
    started = time.perf_counter()
    attempts = Attempts("stt_batch", conn_options, stt._emit_error)
    event = await attempts.run(
        lambda: stt._recognize_impl(buffer, language=language, conn_options=conn_options)
    )
    frames = buffer if isinstance(buffer, list) else [buffer]
    record = _core.stt_batch_metrics(
        event,
        started,
        time.perf_counter(),
        [f.duration for f in frames],
        metrics_context(stt),
    )
    emit_metrics(stt, "STTMetrics", record)
    return event


def function_tools_executed_zipped(
    ev: FunctionToolsExecutedEvent,
) -> list[tuple[FunctionCall, FunctionCallOutput]]:
    return list(zip(ev.function_calls, ev.function_call_outputs, strict=False))


def function_tools_executed_cancel_tool_reply(ev: FunctionToolsExecutedEvent) -> None:
    for out in ev.function_call_outputs:
        out.reply_required = False


def function_tools_executed_cancel_agent_handoff(ev: FunctionToolsExecutedEvent) -> None:
    ev._handoff_required = False


def function_tools_executed_has_tool_reply(ev: FunctionToolsExecutedEvent) -> bool:
    return any(out.reply_required for out in ev.function_call_outputs)


def function_tools_executed_has_agent_handoff(ev: FunctionToolsExecutedEvent) -> bool:
    return ev._handoff_required


def verify_lists_length(ev: FunctionToolsExecutedEvent) -> FunctionToolsExecutedEvent:
    if len(ev.function_calls) != len(ev.function_call_outputs):
        raise ValueError("The number of function_calls and function_call_outputs must match.")
    return ev


def chat_chunk_has_response(chunk: Any) -> bool:
    return bool(chunk.delta and (chunk.delta.content or chunk.delta.tool_calls))


def metrics_repr(metrics: BaseModel) -> str:
    fields = metrics.model_dump(exclude_defaults=True)
    return f"{type(metrics).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def model_usage_repr(usage: BaseModel) -> str:
    # skip zeros for concise display
    fields = {k: v for k, v in usage.model_dump().items() if v != 0 and v != 0.0}
    return f"{type(usage).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def _default_total_tokens(metrics: Any) -> Any:
    if "total_tokens" not in metrics.model_fields_set:
        metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
    return metrics


def speech_data_post_init(data: Any) -> None:
    from .language import LanguageCode

    def code(lang: Any) -> Any:
        return (
            LanguageCode(lang)
            if isinstance(lang, str) and not isinstance(lang, LanguageCode)
            else lang
        )

    data.language = code(data.language)
    for name in ("source_languages", "target_languages"):
        if (langs := getattr(data, name)) is not None:
            setattr(data, name, [code(lang) for lang in langs])


def serialize_source(source: Any) -> Any:
    from .llm import LLM, RealtimeModel
    from .stt import STT
    from .tts import TTS

    if isinstance(source, LLM | STT | TTS | RealtimeModel):
        return {"model": source.model, "provider": source.provider}
    if isinstance(source, BaseModel):
        return source.model_dump()
    return repr(source)


def serialize_error(error: Any) -> Any:
    if isinstance(error, BaseModel):
        return error.model_dump()
    return repr(error)
