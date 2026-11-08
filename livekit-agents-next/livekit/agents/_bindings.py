"""The hand-written method bodies of the generated models (``_generated/*.py`` call them)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Never, Protocol, TypeVar, cast

from pydantic import BaseModel

from livekit import rtc

from . import _core
from ._generated.public__bindings import *  # noqa: F403

if TYPE_CHECKING:
    from pydantic import SerializerFunctionWrapHandler

    from . import llm
    from ._generated import realtime_wire
    from ._generated.a2a_auth import AgentThreadGrant
    from ._generated.chat_items import (
        ChatMessage,
        FunctionCall,
        FunctionCallOutput,
    )
    from ._generated.llm import ChatChunk
    from ._generated.llm_fallback import AvailabilityChanged
    from ._generated.overlap import OverlappingSpeechEvent
    from ._generated.run_result import _Settled
    from ._generated.session_events import ErrorEvent, FunctionToolsExecutedEvent
    from ._generated.stt import STT, SpeechData, SpeechEvent
    from ._generated.tool_options import ProviderToolDeclaration, ToolSpec
    from .llm.chat_context import ChatItem
    from .llm.tool_context import Tool, Toolset
    from .types import APIConnectOptions, NotGivenOr
    from .voice import io
    from .voice.agent_session import VoiceSession
    from .voice.room_io import TextInputEvent
    from .voice.run_result import RunAssert

_T = TypeVar("_T")
_FunctionToolsEventT = TypeVar("_FunctionToolsEventT", bound="FunctionToolsExecutedEvent")
_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT", covariant=True)


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
    def emit(self, event: Literal["metrics_collected"], /, *args: Any) -> None: ...


def auth_grant_post_init(grant: AgentThreadGrant) -> None:
    # Frozen records still accept arbitrary sequences at construction, then retain their own
    # list. This is Python authoring normalization, not a second grant schema or policy owner.
    object.__setattr__(grant, "thread_ids", list(grant.thread_ids))


class _Provider(Protocol):
    def _core_leg(self) -> Any: ...


def core_leg(provider: _Provider | None) -> Any:
    """Resolve the declared provider bridge; provider bases return their Python implementation."""
    return None if provider is None else provider._core_leg()


def core_legs(providers: Sequence[_Provider]) -> list[Any]:
    """``core_leg`` of each provider, in order."""
    return [provider._core_leg() for provider in providers]


def overrides(provider: object, host: type, *names: str) -> bool:
    """Whether ``provider``'s class replaces one of ``names`` as the class that mixes in the
    generated ``host`` (the class just before it in the MRO) defines them."""
    mro = type(provider).__mro__
    owner = mro[mro.index(host) - 1]
    return any(getattr(type(provider), name) is not getattr(owner, name) for name in names)


class AttemptHost(Protocol):
    """What ``ProviderAttempts`` runs for one provider request (docs/models.md "Provider
    attempts"). ``LLMStream``, ``ChunkedStream``, ``SynthesizeStream`` and ``RecognizeStream``
    implement it; a plugin implements ``_run`` and never these."""

    _core_attempt_task: asyncio.Task[Any] | None
    _core_cancelled: bool

    async def _core_attempt(self, retry: bool) -> object:
        """One attempt: the provider's ``_run``, its value the request's result; ``retry``: an
        earlier attempt failed, so a stream replaying its input rewinds it first."""
        ...

    def _core_produced(self) -> bool:
        """Whether output already reached the caller, so the request may not be replayed."""
        ...

    def _core_retrying(self, error: Exception) -> None:
        """A failed attempt that will be retried, as the provider's recoverable ``error``."""
        ...

    def _core_failed(self, error: Exception) -> None:
        """The error that ends the request, as the provider's unrecoverable ``error``."""
        ...


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

    async def _close() -> None:
        await native.aclose()

    feeding, draining = asyncio.create_task(_feed()), asyncio.create_task(_drain())
    try:
        await asyncio.wait({feeding, draining}, return_when=asyncio.FIRST_COMPLETED)
        if feeding.done():
            feeding.result()
        await draining
    finally:
        # Native close must start even if an authored input's finally block stalls. A cancelled
        # close waiter leaves both cleanup branches running, under their original owners.
        await aio.await_cleanup(asyncio.gather(aio.cancel_and_wait(feeding, draining), _close()))


def emit_metrics(provider: _MetricsProvider, record: BaseModel | None) -> None:
    """A request's metrics, folded by the core, as the provider's ``metrics_collected`` event."""
    if record is not None:
        provider.emit("metrics_collected", record)


async def run_attempts(host: AttemptHost, attempts: _core.ProviderAttempts) -> object:
    """``attempts.run()`` for ``host``. A cancellation cancels the running attempt in the same
    turn, as ``aclose()`` promises, then waits for its cleanup before propagating."""
    try:
        return await attempts.run()
    except asyncio.CancelledError:
        host._core_cancelled = True
        if (task := host._core_attempt_task) is not None and not task.done():
            task.cancel()
        await attempts.settled()
        raise
    except Exception as error:
        # the session reports what ends a request it owns; otherwise the ending event carries
        # the last attempt's error once retries were exhausted, else the very exception raised
        if not attempts.core:
            cause = error.__cause__ if attempts.exhausted else None
            host._core_failed(cause if isinstance(cause, Exception) else error)
        raise


def attempt_started(host: AttemptHost) -> None:
    """Called first in every ``_core_attempt``: records the attempt's task so a cancellation
    reaches it in the same turn; an attempt starting after its request was cancelled does not
    run."""
    if host._core_cancelled:
        raise asyncio.CancelledError
    host._core_attempt_task = asyncio.current_task()


class _Recognition:
    """``recognize``'s one request: ``_recognize_impl`` in attempts."""

    def __init__(
        self,
        stt: STT,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        self._stt, self._buffer, self._language, self._conn = stt, buffer, language, conn_options
        self._attempts = _core.ProviderAttempts("stt_batch", conn_options, self)
        self._core_attempt_task: asyncio.Task[object] | None = None
        self._core_cancelled = False

    async def run(self) -> object:
        return await run_attempts(self, self._attempts)

    async def _core_attempt(self, retry: bool) -> SpeechEvent:
        attempt_started(self)
        return await self._stt._recognize_impl(
            self._buffer, language=self._language, conn_options=self._conn
        )

    def _core_produced(self) -> bool:
        return False

    def _core_retrying(self, error: Exception) -> None:
        self._stt._emit_error(error, True)

    def _core_failed(self, error: Exception) -> None:
        self._stt._emit_error(error, False)


async def recognize(
    stt: STT,
    buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
    *,
    language: NotGivenOr[str],
    conn_options: APIConnectOptions,
) -> SpeechEvent:
    """``STT.recognize``: ``_recognize_impl`` in attempts, its metrics once it answers."""
    started = time.perf_counter()
    event = cast(
        "SpeechEvent",
        await _Recognition(stt, buffer, language=language, conn_options=conn_options).run(),
    )
    frames = buffer if isinstance(buffer, list) else [buffer]
    record = _core.stt_batch_metrics(
        event,
        started,
        time.perf_counter(),
        [f.duration for f in frames],
        time.time(),
        stt.label,
        stt.model,
        stt.provider,
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


def run_new_messages(result: _Settled) -> list[ChatItem]:
    return list(result._new_messages)


def run_all_messages(result: _Settled) -> list[ChatItem]:
    return list(result._all_messages)


def run_expect(result: _Settled) -> RunAssert:
    from .voice.run_result import RunAssert, _events

    return RunAssert(_events(result._new_messages))


def metrics_repr(metrics: BaseModel) -> str:
    fields = metrics.model_dump(exclude_defaults=True)
    return f"{type(metrics).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


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


def api_connect_options_validate(self: APIConnectOptions) -> None:
    from . import _core

    if isinstance(self.max_retry, bool) or not isinstance(self.max_retry, int):
        raise ValueError("conn_options.max_retry must be an integer >= 0")
    _core.api_connect_options_check(
        self.max_retry,
        self.retry_interval,
        self.pool_timeout,
        self.connect_timeout,
        self.write_timeout,
        self.read_timeout,
        self.total_timeout,
    )


@contextlib.asynccontextmanager
async def api_connect_options_total_deadline(self: APIConnectOptions) -> AsyncIterator[None]:
    """Bound an operation, retries included, by ``total_timeout``.

    Raises:
        APITimeoutError: with ``phase="total"`` once the limit expires.
    """

    deadline = asyncio.timeout(self.total_timeout)
    try:
        async with deadline:
            yield
    except TimeoutError as e:
        if not deadline.expired():
            raise
        raise APITimeoutError(
            "Operation exceeded total_timeout", retryable=False, phase="total"
        ) from e


def api_connect_options_interval(self: APIConnectOptions, num_retries: int) -> float:
    return _core.api_retry_interval(self.retry_interval, num_retries)


def overlapping_speech_serialize(
    event: OverlappingSpeechEvent, handler: SerializerFunctionWrapHandler
) -> object:
    return handler(event.model_copy(update={"speech_input": None, "probabilities": None}))


def function_tools_zipped(
    self: FunctionToolsExecutedEvent,
) -> list[tuple[FunctionCall, FunctionCallOutput]]:
    """Return calls paired with outputs by list position."""
    return list(zip(self.function_calls, self.function_call_outputs, strict=False))


def function_tools_cancel_tool_reply(self: FunctionToolsExecutedEvent) -> None:
    for out in self.function_call_outputs:
        out.reply_required = False


def function_tools_cancel_agent_handoff(self: FunctionToolsExecutedEvent) -> None:
    self._handoff_required = False


def function_tools_has_tool_reply(self: FunctionToolsExecutedEvent) -> bool:
    return any(out.reply_required for out in self.function_call_outputs)


def verify_lists_length(self: _FunctionToolsEventT) -> _FunctionToolsEventT:
    if len(self.function_calls) != len(self.function_call_outputs):
        raise ValueError("The number of function_calls and function_call_outputs must match.")
    return self


def session_error_serialize(event: ErrorEvent, handler: SerializerFunctionWrapHandler) -> object:
    from .llm import LLM, RealtimeModel
    from .stt import STT
    from .tts import TTS

    error = event.error.model_dump() if isinstance(event.error, BaseModel) else repr(event.error)
    source = event.source
    encoded_source: object
    if isinstance(source, LLM | STT | TTS | RealtimeModel):
        encoded_source = {"model": source.model, "provider": source.provider}
    else:
        encoded_source = source.model_dump() if isinstance(source, BaseModel) else repr(source)
    return handler(event.model_copy(update={"error": error, "source": encoded_source}))


async def default_text_input_cb(sess: VoiceSession, ev: TextInputEvent) -> None:
    async with sess._claim_user_turn():
        await sess.interrupt()
        sess.generate_reply(user_input=ev.text)


def chat_request(req: _core.ChatRequest) -> tuple[llm.ChatContext, list[Tool]]:
    """The context and the offered tools of a request the core makes."""
    from . import llm

    if req.chat_ctx_host is not None:
        chat_ctx = cast(llm.ChatContext, req.chat_ctx_host)
    else:
        chat_ctx = llm.ChatContext.from_dict(req.chat_ctx)
    pools = [req.tools_host, *req.tools_host_extra]
    held = [cast("Iterable[Tool]", pool) for pool in pools if pool is not None]
    tools = offered_tools([t for pool in held for t in pool], req.tool_names)
    names = {
        tool.info.name
        for tool in tools
        if llm.is_function_tool(tool) or llm.is_raw_function_tool(tool)
    }
    for schema in req.function_schemas:
        if schema["name"] not in names:
            tools.append(llm.function_tool(raw_schema=schema)(_refuse_native_tool_call))
            names.add(schema["name"])
    return chat_ctx, tools


def _refuse_native_tool_call(*args: object, **kwargs: object) -> Never:
    raise RuntimeError("this tool is executed by the native agent and cannot be called directly")


def realtime_model(model: llm.RealtimeModel, tools: Callable[[], Sequence[Tool | Toolset]]) -> Any:
    """``model`` as the core takes it: its own model for an OpenAI-protocol model or a fallback
    chain, else a plugin's model bound to the session's tools."""
    from . import llm
    from .llm._realtime import RealtimeModel as OpenAIRealtimeModel
    from .llm.realtime import RealtimeModelFallbackAdapter

    if isinstance(model, OpenAIRealtimeModel):
        return model._core_native()
    if isinstance(model, RealtimeModelFallbackAdapter):
        return model._core_native([realtime_model(m, tools) for m in model._models])
    if isinstance(model, llm.DuplexRealtimeAdapter):
        return _core.DuplexRealtimeModel(Duplex(model, tools), audio_timeout=model._audio_timeout)
    return _core.HostRealtimeModel(model._core_with_tools(tools))


def forward_session_events(
    session: llm.RealtimeSession | llm.DuplexSession,
    put: Callable[[realtime_wire.Event], None],
) -> None:
    """A realtime or duplex session's caller events, reconnects and errors, as the core reads
    them."""
    from ._generated import realtime_wire

    session.on("input_speech_started", lambda _: put(realtime_wire.InputSpeechStartedEvent()))
    session.on(
        "input_speech_stopped",
        lambda ev: put(dataclasses.replace(ev)),
    )
    session.on(
        "input_audio_transcription_completed",
        lambda ev: put(dataclasses.replace(ev)),
    )
    session.on(
        "session_reconnected", lambda _: put(realtime_wire.RealtimeSessionReconnectedEvent())
    )
    session.on("metrics_collected", lambda ev: put(realtime_wire.RealtimeMetrics(metrics=ev)))
    session.on(
        "error",
        lambda ev: put(
            realtime_wire.HostRealtimeError(
                label=ev.label,
                message=str(ev.error),
                recoverable=ev.recoverable,
                timestamp=ev.timestamp,
            )
        ),
    )


class Duplex:
    """A ``DuplexRealtimeAdapter`` (``adapter``) as the core's duplex host: the core reads its
    model's label and capabilities, gates and segments the output; ``tools`` names the tool
    objects the core refers to by name."""

    def __init__(
        self, adapter: llm.DuplexRealtimeAdapter, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        self.adapter = adapter
        self._tools = tools

    def _core_gate(self) -> llm.AudioGate | None:
        """The gate the adapter or the model declares; ``None``: the adaptive gate."""
        return (
            self.adapter._gate() if self.adapter._gate else self.adapter.duplex_model.audio_gate()
        )

    async def _core_session(self) -> _DuplexSession:
        """A new plugin session."""
        return _DuplexSession(self.adapter.duplex_model.session(), self._tools)


class _DuplexSession:
    """A plugin's ``DuplexSession`` (``plugin``) flattened: the core feeds its input, applies the
    updates in order and pulls ``plugin.audio_stream``; the session's callbacks queue one event at a time
    into ``events``. The core calls the plugin's own update methods directly; the methods here
    resolve tool names and convert records."""

    def __init__(
        self, session: llm.DuplexSession, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        from ._generated import realtime_wire

        self.plugin = session
        self._tools = tools
        self.events: asyncio.Queue[Any] = asyncio.Queue()
        put = self.events.put_nowait
        forward_session_events(session, put)
        session.on(
            "transcript_delta",
            lambda ev: put(dataclasses.replace(ev)),
        )
        session.on("function_call", lambda call: put(realtime_wire.FunctionCallReady(call=call)))

    async def _core_configure(
        self, instructions: str | None, items: list[ChatItem] | None, names: list[str]
    ) -> None:
        from .types import NOT_GIVEN

        await self.plugin._update_session(
            instructions=instructions or NOT_GIVEN,
            chat_ctx=_chat_ctx(items),
            tools=offered_tools(self._tools(), names),
        )

    async def _core_update_tools(self, names: list[str]) -> None:
        await self.plugin._update_tools(offered_tools(self._tools(), names))

    async def _core_generate(
        self, instructions: str | None, tool_choice: llm.ToolChoice | None
    ) -> None:
        from .types import NOT_GIVEN

        self.plugin._generate_reply(
            instructions=instructions or NOT_GIVEN,
            tool_choice=NOT_GIVEN if tool_choice is None else tool_choice,
        )

    async def _core_aclose(self) -> None:
        await self.plugin.aclose()
        self.events.put_nowait(None)


def _chat_ctx(items: list[ChatItem] | None) -> NotGivenOr[llm.ChatContext]:
    from . import llm
    from .types import NOT_GIVEN

    return NOT_GIVEN if items is None else llm.ChatContext(items)


class FallbackListener(Protocol):
    """A fallback adapter as its chain reports to it: an entry, by its index in the adapter's
    list, went unavailable or recovered; every entry is unavailable."""

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None: ...

    def _core_all_unavailable(self) -> None: ...


# --- io legs ---------------------------------------------------------------------------------


async def next_audio_frame(source: io.AudioInput) -> rtc.AudioFrame | None:
    """The core's read of an ``io.AudioInput``: its next frame, ``None`` once it ends."""
    try:
        return await source.__anext__()
    except StopAsyncIteration:
        return None


def rgba_frame(frame: rtc.VideoFrame) -> rtc.VideoFrame:
    """The core's read of a frame it encodes: its pixels as RGBA, converted from any other type."""
    if frame.type == rtc.VideoBufferType.RGBA:
        return frame
    return frame.convert(rtc.VideoBufferType.RGBA)


def audio_output_capabilities(output: io.AudioOutput) -> io.AudioOutputCapabilities:
    """What the core reads of an ``io.AudioOutput`` once: whether its whole chain can pause."""
    from .voice import io

    return io.AudioOutputCapabilities(pause=output.can_pause)


# --- tools ------------------------------------------------------------------------------------


def tool_specs(tools: Sequence[Tool | Toolset]) -> list[ToolSpec]:
    """Project function schemas and provider declarations through the core's tool boundary."""
    from . import llm
    from .llm.tool_context import ProviderTool, ToolFlag
    from .llm.utils import tool_schema

    specs: list[ToolSpec] = []
    for tool in llm.ToolContext(tools).flatten():
        ignore_on_enter = False
        provider_tool: ProviderToolDeclaration | None
        if isinstance(tool, ProviderTool):
            name, description, parameters = tool.id, "", None
            provider_tool = {"provider": tool.provider, "payload": tool.to_dict()}
        elif llm.is_function_tool(tool) or llm.is_raw_function_tool(tool):
            name, description, parameters = tool_schema(tool)
            ignore_on_enter = ToolFlag.IGNORE_ON_ENTER in tool.info.flags
            provider_tool = None
        else:
            continue
        spec: ToolSpec = {
            "name": name,
            "description": description,
            "json_schema": parameters,
            "ignore_on_enter": ignore_on_enter,
        }
        if provider_tool is not None:
            spec["provider_tool"] = provider_tool
        specs.append(spec)
    return specs


def offered_tools(pool: Sequence[Tool | Toolset], names: Sequence[str]) -> list[Tool]:
    """The tools of ``pool`` a request offers: the named function tools and every provider tool."""
    from . import llm
    from .llm.tool_context import ProviderTool

    ctx = llm.ToolContext(pool)
    named = [t for n in names if (t := ctx.get_function_tool(n)) is not None]
    return [*named, *(t for t in ctx.flatten() if isinstance(t, ProviderTool))]


def _status_error_str(self: APIStatusError) -> str:
    parts = [
        f"message={self.message!r}",
        f"status_code={self.status_code}",
        f"retryable={self.retryable}",
    ]
    if self.request_id:
        parts.append(f"request_id={self.request_id}")
    if self.body:
        parts.append(f"body={self.body}")
    return ", ".join(parts)


def _connection_error_str(self: APIConnectionError) -> str:
    root: BaseException | None = None
    seen = {id(self)}
    cause = self.__cause__
    for _ in range(10):
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        root = cause
        cause = cause.__cause__

    if root is None:
        return self.message

    # naming our own type by .message keeps a cycle that ends on one out of __str__, which
    # would otherwise re-enter here through the same chain.
    if isinstance(root, APIConnectionError):
        root_message = root.message
    else:
        # a third-party __str__ may raise (e.g. aiohttp.ClientConnectorError on a partially
        # initialized connection key); the type name alone still beats losing the message.
        try:
            root_message = str(root)
        except Exception:
            root_message = ""

    detail = type(root).__name__
    if root_message:
        detail += f": {root_message}"

    return f"{self.message} (caused by {detail})"


def redacted_cause(error: BaseException) -> RedactedCause:
    """``error`` to chain as a cause (``raise APIError(...) from redacted_cause(e)``)."""
    return RedactedCause(type(error).__qualname__).with_traceback(error.__traceback__)


def create_api_error_from_http(
    message: str = "",
    *,
    status: int,
    request_id: str | None = None,
    body: object | None = None,
) -> APIStatusError:
    """Create an APIStatusError from an HTTP status code.

    When the message carries extra detail beyond the standard reason phrase,
    both the message and the reason are shown. Otherwise just the reason.
    """
    from http import HTTPStatus

    from . import _core

    reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else None
    return APIStatusError(
        message=_core.http_error_display(message, status, reason),
        status_code=status,
        request_id=request_id,
        body=body,
    )
