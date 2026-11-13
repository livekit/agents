from __future__ import annotations

import asyncio
import atexit
import contextlib
import dataclasses
import importlib.resources
import time
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
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
    from ._generated.background_audio_wire import BuiltinAudioClip
    from ._generated.chat_items import (
        ChatMessage,
        FunctionCall,
        FunctionCallOutput,
    )
    from ._generated.events import OverlappingSpeechEvent
    from ._generated.llm import ChatChunk
    from ._generated.llm_fallback import AvailabilityChanged
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


class _PumpStream(Protocol):
    def end_input(self) -> None: ...
    async def aclose(self) -> None: ...


class _Source(Protocol[_OutputT]):
    async def recv(self) -> _OutputT | None: ...


class _MetricsProvider(Protocol):
    @property
    def label(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    def emit(self, event: Literal["metrics_collected"], /, *args: Any) -> None: ...


def auth_grant_post_init(grant: AgentThreadGrant) -> None:
    # the record is frozen; keep an owned list of whatever sequence was passed
    object.__setattr__(grant, "thread_ids", list(grant.thread_ids))


def silero_model() -> Any:
    # requested only when the build links no native Silero
    from livekit.local_inference import VAD

    return VAD()


def refuse_overrides(cls: type, hooks: tuple[str, ...], node: str) -> None:
    """Refuses, as ``cls`` is defined, an override of a hook its native class runs itself: the
    core would never call it. The framework's own subclasses are exempt."""
    if _framework(cls):
        return
    owner = next(base for base in cls.__mro__ if _framework(base))
    for name in hooks:
        if getattr(cls, name) is not getattr(owner, name):
            raise TypeError(
                f"{cls.__qualname__} overrides {name}, which {owner.__qualname__} runs natively;"
                f" customize its output in {node} instead"
            )


def proxy_overrides(cls: type, hooks: tuple[str, ...]) -> None:
    """Marks ``cls``, as it is defined, to run through the host wherever it reaches the core when
    it overrides one of ``hooks``. The framework's own subclasses are exempt."""
    if _framework(cls):
        return
    owner = next(base for base in cls.__mro__ if _framework(base))
    if any(getattr(cls, name) is not getattr(owner, name) for name in hooks):
        cls._core_proxied = True  # type: ignore[attr-defined]


def _framework(cls: type) -> bool:
    return cls.__module__ == "livekit.agents" or cls.__module__.startswith("livekit.agents.")


class ProviderEvents(Protocol):
    """What a native provider reports to the object it was built as."""

    def _core_failed(self, error: Exception, recoverable: bool) -> None: ...


class AttemptHost(Protocol):
    """One provider request run by ``ProviderAttempts``; plugins implement ``_run``, not this."""

    _core_attempt_task: asyncio.Task[Any] | None
    _core_cancelled: bool

    async def _core_attempt(self, retry: bool) -> object:
        """Run one attempt; ``retry`` means an earlier one failed and input must be rewound."""
        ...

    def _core_produced(self) -> bool:
        """Whether output already reached the caller, so the request may not be replayed."""
        ...

    def _core_retrying(self, error: Exception) -> None: ...

    def _core_failed(self, error: Exception) -> None: ...


async def pump(
    native: _PumpStream,
    items: AsyncIterable[_InputT],
    feed: Callable[[_InputT], None],
    drain: Awaitable[None],
) -> None:
    """Feed ``items`` into ``native`` while ``drain`` consumes it. A failed feed ends the stream
    with its error; the stream is closed either way."""
    from .utils import aio

    async def _feed() -> None:
        async for item in items:
            feed(item)
        native.end_input()

    async def _close() -> None:
        await native.aclose()

    feeding, draining = asyncio.create_task(_feed()), asyncio.ensure_future(drain)
    try:
        await asyncio.wait({feeding, draining}, return_when=asyncio.FIRST_COMPLETED)
        if feeding.done():
            feeding.result()
        await draining
    finally:
        # native close must start even if an authored input's finally block stalls
        await aio._await_cleanup(asyncio.gather(aio.cancel_and_wait(feeding, draining), _close()))


def check_input(stream: object, task: asyncio.Task[Any], *, closed: bool, ended: bool) -> None:
    """Raise why ``stream`` takes no more input: its provider's failure first, then its close,
    then the end of its input. Every stream base checks its input calls here."""
    if task.done() and not task.cancelled() and (error := task.exception()) is not None:
        raise error
    cls = type(stream)
    if closed:
        raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")
    if ended:
        raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")


async def drain(native: _Source[_OutputT], out: Callable[[_OutputT], None]) -> None:
    while (item := await native.recv()) is not None:
        out(item)


def emit_metrics(provider: _MetricsProvider, record: BaseModel | None) -> None:
    if record is not None:
        provider.emit("metrics_collected", record)


async def run_attempts(host: AttemptHost, attempts: _core.ProviderAttempts) -> object:
    """A cancellation cancels the running attempt in the same turn, as ``aclose()`` promises,
    then waits for its cleanup before propagating."""
    try:
        return await attempts.run()
    except asyncio.CancelledError:
        host._core_cancelled = True
        if (task := host._core_attempt_task) is not None and not task.done():
            task.cancel()
        await attempts.settled()
        raise
    except Exception as error:
        # a session reports the requests it owns; once retries are exhausted, report the cause
        if not attempts.core:
            cause = error.__cause__ if attempts.exhausted else None
            host._core_failed(cause if isinstance(cause, Exception) else error)
        raise


def attempt_started(host: AttemptHost) -> None:
    """Called first in every ``_core_attempt`` so a cancellation reaches it in the same turn."""
    if host._core_cancelled:
        raise asyncio.CancelledError
    host._core_attempt_task = asyncio.current_task()


class _Recognition:
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


_resources = contextlib.ExitStack()
atexit.register(_resources.close)


def builtin_audio_clip_path(clip: BuiltinAudioClip) -> str:
    file = importlib.resources.files("livekit.agents").joinpath("resources", clip.value)
    return str(_resources.enter_context(importlib.resources.as_file(file)))


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
        APITimeoutError: with ``timeout="total"`` once the limit expires.
    """

    deadline = asyncio.timeout(self.total_timeout)
    try:
        async with deadline:
            yield
    except TimeoutError as e:
        if not deadline.expired():
            raise
        raise APITimeoutError(
            "Operation exceeded total_timeout", retryable=False, timeout="total"
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


def forward_session_events(
    session: llm.RealtimeSession | llm.DuplexSession,
    put: Callable[[realtime_wire.Event], None],
) -> None:
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


class _DuplexSession:
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
    """A fallback adapter's view of its chain; entries are identified by list index."""

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None: ...

    def _core_all_unavailable(self) -> None: ...

    def _core_failed(self, error: Exception, recoverable: bool) -> None: ...


async def next_audio_frame(source: io.AudioInput) -> rtc.AudioFrame | None:
    try:
        return await source.__anext__()
    except StopAsyncIteration:
        return None


def rgba_frame(frame: rtc.VideoFrame) -> rtc.VideoFrame:
    if frame.type == rtc.VideoBufferType.RGBA:
        return frame
    return frame.convert(rtc.VideoBufferType.RGBA)


def audio_output_capabilities(output: io.AudioOutput) -> io.AudioOutputCapabilities:
    from .voice import io

    return io.AudioOutputCapabilities(pause=output.can_pause)


def tool_specs(tools: Sequence[Tool | Toolset]) -> list[ToolSpec]:
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
    """The named function tools of ``pool`` plus every provider tool."""
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

    # .message, not str(): str() on our own type re-enters here through the same chain
    if isinstance(root, APIConnectionError):
        root_message = root.message
    else:
        # a third-party __str__ may raise (e.g. aiohttp.ClientConnectorError)
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
    """Create an APIStatusError from an HTTP status code and an optional detail message."""
    from http import HTTPStatus

    from . import _core

    reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else None
    return APIStatusError(
        message=_core.http_error_display(message, status, reason),
        status_code=status,
        request_id=request_id,
        body=body,
    )
