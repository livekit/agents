"""The hand-written method bodies of the generated models (``_generated/*.py`` call them)."""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from . import _core
from ._exceptions import APIConnectionError
from ._generated._retry import (
    _core_RetryDecisionExhausted,
    _core_RetryDecisionRaise,
    _core_RetryDecisionRetry,
    _core_RetryDecisionStop,
)
from .log import logger
from .types import APIConnectOptions, NotGivenOr

if TYPE_CHECKING:
    from ._generated.chat_items import ChatMessage
    from ._generated.llm import ChatChunk
    from ._generated.metrics import STTMetrics
    from ._generated.stt import STT, SpeechData, SpeechEvent
    from ._generated.usage import TokenUsage, Usage
    from .llm.fallback_adapter import FallbackAdapter as _LlmFallback
    from .stt import FallbackAdapter as _SttFallback
    from .tts import FallbackAdapter as _TtsFallback
    from .utils.audio import AudioBuffer

_T = TypeVar("_T")
_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT", covariant=True)
_MetricsT = TypeVar("_MetricsT", bound="STTMetrics")


class _PumpStream(Protocol[_OutputT]):
    def end_input(self) -> None: ...
    async def recv(self) -> _OutputT | None: ...
    async def aclose(self) -> None: ...


class _MetricsProvider(Protocol):
    @property
    def label(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    def emit(self, event: Literal["metrics_collected"], metrics: BaseModel) -> None: ...


class _FallbackChain(Protocol):
    @property
    def active_index(self) -> int: ...
    async def aclose(self) -> None: ...


def on_loop(owner: object, build: Callable[[], _T]) -> _T:
    """``build()`` once per running event loop: a core object over host providers calls them on
    the loop it was built on, so an object shared by two loops (two jobs on threads, per-test
    loops) gets one each."""
    built: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _T] = owner.__dict__.setdefault(
        "_core_by_loop", weakref.WeakKeyDictionary()
    )
    loop = asyncio.get_running_loop()
    if (native := built.get(loop)) is None:
        native = built[loop] = build()
    return native


def built_on_loop(owner: _LlmFallback | _SttFallback | _TtsFallback) -> list[_FallbackChain]:
    """What ``on_loop`` built for ``owner``, the running loop's first."""
    built: dict[asyncio.AbstractEventLoop, _FallbackChain] = dict(
        owner.__dict__.get("_core_by_loop", {})
    )
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


def overrides(obj: object, base: type, *names: str) -> bool:
    """Whether ``obj``'s class replaces any of ``base``'s ``names``: the core object would bypass
    it, so the provider is run through its plugin surface instead."""
    return any(getattr(type(obj), n) is not getattr(base, n) for n in names)


_pending_cleanup: set[asyncio.Future[object]] = set()


async def cancel_and_wait_bounded(*tasks: asyncio.Task[object], timeout: float) -> bool:
    """Cancel on the owning loop, retaining cleanup after the native settlement limit."""

    def settled(finished: asyncio.Future[object]) -> None:
        _pending_cleanup.discard(finished)
        if not finished.cancelled():
            finished.exception()

    for task in tasks:
        _pending_cleanup.add(task)
        task.add_done_callback(settled)
        if not task.done() and not task.cancelling():
            task.cancel()
    if not tasks:
        return True
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    return not pending


def _now_ns() -> int:
    return int(asyncio.get_running_loop().time() * 1e9)


class AttemptLoop:
    """Execute host callbacks as the shared supplied-time attempt machine authorizes them."""

    def __init__(
        self, kind: str, conn: APIConnectOptions, emit_error: Callable[[Exception, bool], None]
    ) -> None:
        self.kind = kind
        self.emit_error = emit_error
        self.core = False
        self._request = _core._core_ProviderAttempts(kind, conn, _now_ns())
        self._changed = asyncio.Event()

    @property
    def core(self) -> bool:
        return self._core

    @core.setter
    def core(self, value: bool) -> None:
        self._core = value
        if value:
            self._request.external_retry_owner()

    @property
    def cleanup_timeout(self) -> float:
        return self._request.cleanup_timeout

    def capacity_exceeded(self, *, input: bool) -> None:
        try:
            self._request.capacity_exceeded(input, _now_ns())
        finally:
            self._changed.set()

    def expect_response(self, *, final: bool = False) -> None:
        """A request was written and owes a response; ``final``: the last input was sent."""
        if final:
            self._request.input_ended(_now_ns())
        else:
            self._request.expect_response(_now_ns())
        self._changed.set()

    def input_ended(self) -> None:
        self._request.input_ended(_now_ns())
        self._changed.set()

    def progress(self) -> None:
        """Response data arrived; heartbeats are not progress."""
        self._observe_response("progress")

    def audio_received(self) -> None:
        """Audio settles a protocol's first response; after the final input it is progress."""
        self._observe_response("audio")

    def complete_response(self) -> None:
        """No response remains outstanding."""
        self._observe_response("complete")

    def _observe_response(self, observation: str) -> None:
        self._request.observe_response(observation, _now_ns())
        self._changed.set()

    def observe(self, event: SpeechEvent) -> None:
        self._request.observe(event, _now_ns())
        self._changed.set()

    async def run(
        self,
        attempt: Callable[[], Awaitable[_T]],
        *,
        produced: Callable[[], bool] = lambda: False,
        again: Callable[[], None] = lambda: None,
    ) -> _T:
        if self.core:
            try:
                if not self._request.ready(_now_ns()):
                    raise RuntimeError("provider request has no pending attempt")
                value = await self._execute(attempt)
                self._request.succeeded(_now_ns())
                return value
            finally:
                self._request.cancel(_now_ns())
        retrying = False
        terminal_error: Exception | None = None
        try:
            while True:
                while not self._request.ready(_now_ns()):
                    at = self._request.wake_at()
                    if at is None:
                        raise RuntimeError("provider request has no pending attempt")
                    await asyncio.sleep(max(0.0, (at - _now_ns()) / 1e9))
                if retrying:
                    again()
                try:
                    value = await self._execute(attempt)
                    self._request.succeeded(_now_ns())
                    return value
                except Exception as error:
                    decision = self._request.failure(error, produced(), _now_ns())
                    match decision:
                        case _core_RetryDecisionStop():
                            return cast(_T, None)
                        case _core_RetryDecisionRaise():
                            raise
                        case _core_RetryDecisionExhausted(message=message):
                            terminal_error = error
                            raise APIConnectionError(message) from error
                        case _core_RetryDecisionRetry(delay=delay):
                            self.emit_error(error, True)
                            logger.warning(
                                f"{self.kind} attempt failed: {error}, retrying in {delay}s"
                            )
                    retrying = True
        except Exception as error:
            self.emit_error(terminal_error if terminal_error is not None else error, False)
            raise
        finally:
            self._request.cancel(_now_ns())

    async def _execute(self, attempt: Callable[[], Awaitable[_T]]) -> _T:
        async def invoke() -> _T:
            return await attempt()

        task = asyncio.create_task(invoke())
        changed = asyncio.create_task(self._changed.wait())
        try:
            while True:
                at = self._request.wake_at()
                timeout = None if at is None else max(0.0, (at - _now_ns()) / 1e9)
                await asyncio.wait(
                    (task, changed), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                # Time and observations return to the core before a result is accepted.
                self._request.ready(_now_ns())
                if task.done():
                    return task.result()
                if changed.done():
                    self._changed.clear()
                    changed = asyncio.create_task(self._changed.wait())
        finally:
            if not await cancel_and_wait_bounded(task, changed, timeout=self.cleanup_timeout):
                logger.error(
                    "provider cleanup incomplete; its host task retains ownership until settlement"
                )


async def pump(
    native: _PumpStream[_OutputT],
    items: AsyncIterable[_InputT],
    feed: Callable[[_InputT], None],
    out: Callable[[_OutputT], None],
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


def metrics_context(
    provider: _MetricsProvider, *, cancelled: bool = False
) -> tuple[float, str, bool, str, str]:
    """What a provider's metrics are stamped with: now, its label, model and provider."""
    return (time.time(), provider.label, cancelled, provider.model, provider.provider)


def emit_metrics(provider: _MetricsProvider, record: BaseModel | None) -> None:
    """A request's metrics, folded by the core, as the provider's ``metrics_collected`` event."""
    if record is not None:
        provider.emit("metrics_collected", record)


async def recognize(
    stt: STT,
    buffer: AudioBuffer,
    *,
    language: NotGivenOr[str],
    conn_options: APIConnectOptions,
) -> SpeechEvent:
    """``STT.recognize``: ``_recognize_impl`` in attempts, its metrics once it answers."""
    started = time.perf_counter()
    attempts = AttemptLoop("stt_batch", conn_options, stt._emit_error)
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
    emit_metrics(stt, record)
    return event


def chat_chunk_has_response(chunk: ChatChunk) -> bool:
    return bool(chunk.delta and (chunk.delta.content or chunk.delta.tool_calls))


def _text_parts(message: ChatMessage) -> list[str]:
    return [c for c in message.content if isinstance(c, str)]


def chat_message_raw_text_content(message: ChatMessage) -> str | None:
    parts = _text_parts(message)
    return "\n".join(parts) if parts else None


def chat_message_text_content(message: ChatMessage) -> str | None:
    return _core.chat_message_text_content(_text_parts(message), message.role == "assistant")


def metrics_repr(metrics: BaseModel) -> str:
    fields = metrics.model_dump(exclude_defaults=True)
    return f"{type(metrics).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def token_usage_total(tokens: TokenUsage) -> int:
    return tokens.input_tokens + tokens.output_tokens


def usage_input_tokens(usage: Usage) -> int:
    return usage.tokens.input_tokens


def usage_output_tokens(usage: Usage) -> int:
    return usage.tokens.output_tokens


def usage_total_tokens(usage: Usage) -> int:
    return usage.tokens.input_tokens + usage.tokens.output_tokens


def _default_total_tokens(metrics: _MetricsT) -> _MetricsT:
    if "total_tokens" not in metrics.model_fields_set:
        metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
    return metrics


def speech_data_post_init(data: SpeechData) -> None:
    from ._generated.language import LanguageCode

    def code(lang: LanguageCode) -> LanguageCode:
        return (
            LanguageCode(lang)
            if isinstance(lang, str) and not isinstance(lang, LanguageCode)
            else lang
        )

    data.language = code(data.language)
    if data.source_languages is not None:
        data.source_languages = [code(lang) for lang in data.source_languages]
    if data.target_languages is not None:
        data.target_languages = [code(lang) for lang in data.target_languages]
