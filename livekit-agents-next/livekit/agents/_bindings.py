"""The hand-written method bodies of the generated models (``_generated/*.py`` call them)."""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from livekit import rtc

from . import _core
from ._generated._retry import (
    _core_RetryDecisionExhausted,
    _core_RetryDecisionRaise,
    _core_RetryDecisionRetry,
    _core_RetryDecisionStop,
)
from .log import logger

if TYPE_CHECKING:
    from pydantic import SerializerFunctionWrapHandler

    from ._generated.a2a_auth import AgentThreadGrant
    from ._generated.chat_items import ChatMessage, FunctionCall, FunctionCallOutput
    from ._generated.evals import EvaluationResult, EvaluationScore, JudgmentResult
    from ._generated.llm import ChatChunk
    from ._generated.overlap import OverlappingSpeechEvent
    from ._generated.room_api import (
        AudioInputOptions,
        AudioOutputOptions,
        RoomOptions,
        TextInputOptions,
        TextOutputOptions,
        VideoInputOptions,
    )
    from ._generated.session_events import ErrorEvent, FunctionToolsExecutedEvent
    from ._generated.storage_class import ThreadInfo
    from ._generated.stt import STT, SpeechData, SpeechEvent
    from ._generated.usage import TokenUsage, Usage
    from .llm import FallbackAdapter as _LlmFallback
    from .stt import FallbackAdapter as _SttFallback
    from .tts import FallbackAdapter as _TtsFallback
    from .types import APIConnectOptions, NotGivenOr, TranscriptChunk
    from .voice.agent_session import VoiceSession
    from .voice.room_io import TextInputEvent
import dataclasses
import functools
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Never

if TYPE_CHECKING:
    from . import llm
    from ._generated import duplex, realtime_wire
    from ._generated.llm_fallback import AvailabilityChanged
    from ._generated.tool_options import ProviderToolDeclaration, ToolSpec
    from ._generated.turn_events import EotPrediction
    from ._generated.types import ProviderId
    from .llm.chat_context import ChatItem
    from .llm.tool_context import Tool, Toolset
    from .voice import io
    from .voice.io import _StreamingTurnDetector
from ._generated.api_errors import (
    APIConnectionError as APIConnectionError,
    APIError as APIError,
    APIStatusError as APIStatusError,
    APITimeoutError as APITimeoutError,
    AssignmentTimeoutError as AssignmentTimeoutError,
    CLIError as CLIError,
    RedactedCause as RedactedCause,
    StreamStateError as StreamStateError,
    UnexpectedModelBehavior as UnexpectedModelBehavior,
)

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
    def emit(self, event: Literal["metrics_collected"], metrics: BaseModel) -> None: ...


class _FallbackChain(Protocol):
    @property
    def active_index(self) -> int: ...
    async def aclose(self) -> None: ...


def auth_grant_post_init(grant: AgentThreadGrant) -> None:
    # Frozen records still accept arbitrary sequences at construction, then retain their own
    # list. This is Python authoring normalization, not a second grant schema or policy owner.
    object.__setattr__(grant, "thread_ids", list(grant.thread_ids))


def auth_grant_can_create(grant: AgentThreadGrant, scope: str | None = None) -> bool:
    return _core._core_auth_grant_allows(grant, "create", "", scope)


def auth_grant_can_list(grant: AgentThreadGrant, scope: str) -> bool:
    return _core._core_auth_grant_allows(grant, "list", "", scope)


def auth_grant_can_read(grant: AgentThreadGrant, thread: ThreadInfo) -> bool:
    return _core._core_auth_grant_allows(grant, "read", thread.thread_id, thread.scope)


def auth_grant_can_write(grant: AgentThreadGrant, thread: ThreadInfo) -> bool:
    return _core._core_auth_grant_allows(grant, "write", thread.thread_id, thread.scope)


def auth_grant_can_delete(grant: AgentThreadGrant, thread: ThreadInfo) -> bool:
    return _core._core_auth_grant_allows(grant, "delete", thread.thread_id, thread.scope)


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


class _Provider(Protocol):
    def _core_leg(self) -> Any: ...


def core_leg(provider: _Provider | None) -> Any:
    """Resolve the declared provider bridge; provider bases return their Python implementation."""
    return None if provider is None else provider._core_leg()


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
        from .utils import aio

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
            if not await aio.cancel_and_wait(task, changed, timeout=self.cleanup_timeout):
                logger.error(
                    "provider cleanup incomplete; its host task retains ownership until settlement",
                    extra={"pending_host_cleanup": aio.pending_cleanup_count()},
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


async def recognize(
    stt: STT,
    buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
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


def function_tools_has_agent_handoff(self: FunctionToolsExecutedEvent) -> bool:
    return self._handoff_required


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


def room_options_get_text_input_options(self: RoomOptions) -> TextInputOptions | None:
    from ._generated.room_api import (
        TextInputOptions,
    )

    if isinstance(self.text_input, TextInputOptions):
        return self.text_input
    return TextInputOptions() if self.text_input is not False else None


def room_options_get_audio_input_options(self: RoomOptions) -> AudioInputOptions | None:
    from ._generated.room_api import (
        AudioInputOptions,
    )

    if isinstance(self.audio_input, AudioInputOptions):
        return self.audio_input
    return AudioInputOptions() if self.audio_input is not False else None


def room_options_get_video_input_options(self: RoomOptions) -> VideoInputOptions | None:
    from ._generated.room_api import (
        VideoInputOptions,
    )

    if isinstance(self.video_input, VideoInputOptions):
        return self.video_input
    return VideoInputOptions() if self.video_input is True else None


def room_options_get_audio_output_options(self: RoomOptions) -> AudioOutputOptions | None:
    from ._generated.room_api import (
        AudioOutputOptions,
    )

    if isinstance(self.audio_output, AudioOutputOptions):
        return self.audio_output
    return AudioOutputOptions() if self.audio_output is not False else None


def room_options_get_text_output_options(self: RoomOptions) -> TextOutputOptions | None:
    from ._generated.room_api import (
        TextOutputOptions,
    )

    if isinstance(self.text_output, TextOutputOptions):
        return self.text_output
    return TextOutputOptions() if self.text_output is not False else None


def room_options_to_core(self: RoomOptions) -> RoomOptions:
    """Resolve host-only defaults on a private copy; generated extraction reads the fields.

    The resolved view uses ``None`` for disabled legs rather than authoring booleans. It never
    escapes as public RoomOptions or changes the caller's options/SDK objects.
    """
    from copy import copy

    from .utils import is_given

    audio_in = self.get_audio_input_options()
    if audio_in is not None:
        audio_in = copy(audio_in)
        if not is_given(audio_in.auto_gain_control):
            audio_in.auto_gain_control = audio_in.noise_cancellation is None or callable(
                audio_in.noise_cancellation
            )
    # The boundary view intentionally differs from the public authoring union for disabled legs.
    resolved = cast(Any, copy(self))
    resolved.participant_identity = (
        self.participant_identity if is_given(self.participant_identity) else None
    )
    resolved.participant_kinds = (
        list(self.participant_kinds) if is_given(self.participant_kinds) else []
    )
    resolved.text_input = self.get_text_input_options() is not None
    resolved.video_input = self.get_video_input_options() is not None
    resolved.audio_input = audio_in
    resolved.audio_output = self.get_audio_output_options()
    resolved.text_output = self.get_text_output_options()
    return cast("RoomOptions", resolved)


async def default_text_input_cb(sess: VoiceSession, ev: TextInputEvent) -> None:
    async with sess._claim_user_turn():
        await sess.interrupt()
        sess.generate_reply(user_input=ev.text)


def judgment_result_passed(self: JudgmentResult) -> bool:
    """Whether the evaluation passed. Maybe is treated as not passed."""
    return self.verdict == "pass"


def judgment_result_failed(self: JudgmentResult) -> bool:
    """Whether the evaluation failed. Maybe is treated as not failed."""
    return self.verdict == "fail"


def judgment_result_uncertain(self: JudgmentResult) -> bool:
    """Whether the judge was uncertain about the verdict."""
    return self.verdict == "maybe"


def evaluation_result_score(self: EvaluationResult) -> float:
    """Score from 0.0 to 1.0. Pass=1, maybe=0.5, fail=0."""
    return self._score().score


def evaluation_result_all_passed(self: EvaluationResult) -> bool:
    """True if all judgments passed. Maybes count as not passed."""
    return self._score().all_passed


def evaluation_result_any_passed(self: EvaluationResult) -> bool:
    """True if at least one judgment passed."""
    return self._score().any_passed


def evaluation_result_majority_passed(self: EvaluationResult) -> bool:
    """True if more than half of the judgments passed."""
    return self._score().majority_passed


def evaluation_result_none_failed(self: EvaluationResult) -> bool:
    """True if no judgments explicitly failed. Maybes are allowed."""
    return self._score().none_failed


def evaluation_result_core_score(self: EvaluationResult) -> EvaluationScore:
    return _core._core_evaluation_score([j.verdict for j in self.judgments.values()])


def duck(obj: Any, cls: type) -> Any:
    """The view of ``obj`` as the core's ``cls``: called by the core, once per handed object."""
    for base, view in _views().items():
        if issubclass(cls, base):
            return view(obj)
    raise TypeError(f"no core view of {type(obj).__name__} as {cls.__qualname__}")


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
    """A ``DuplexRealtimeAdapter``'s model as the core's duplex host: the core gates and segments
    the output; ``tools`` names the tool objects the core refers to by name."""

    def __init__(
        self, adapter: llm.DuplexRealtimeAdapter, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        self._adapter = adapter
        self._tools = tools

    def _core_label(self) -> str:
        return self._adapter.duplex_model.label

    def _core_capabilities(self) -> duplex.DuplexCapabilities:
        return dataclasses.replace(self._adapter.duplex_model.capabilities)

    def _core_gate(self) -> llm.AudioGate | None:
        """The gate the adapter or the model declares; ``None``: the adaptive gate."""
        return (
            self._adapter._gate()
            if self._adapter._gate
            else self._adapter.duplex_model.audio_gate()
        )

    async def _core_session(self) -> _DuplexSession:
        """A new plugin session."""
        return _DuplexSession(self._adapter.duplex_model.session(), self._tools)


class _DuplexSession:
    """A plugin's ``DuplexSession`` flattened: the core feeds its input, applies the updates in
    order and pulls the output audio; the session's callbacks report one event at a time."""

    def __init__(
        self, session: llm.DuplexSession, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        from ._generated import realtime_wire

        self._session = session
        self._tools = tools
        self._audio = aiter(session.audio_stream)
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        put = self._events.put_nowait
        forward_session_events(session, put)
        session.on(
            "transcript_delta",
            lambda ev: put(dataclasses.replace(ev)),
        )
        session.on("function_call", lambda call: put(realtime_wire.FunctionCallReady(call=call)))

    def _core_input(self) -> _core.DuplexInput:
        return self._session._input

    async def _core_configure(
        self, instructions: str | None, items: list[ChatItem] | None, names: list[str]
    ) -> None:
        from .types import NOT_GIVEN

        await self._session._update_session(
            instructions=instructions or NOT_GIVEN,
            chat_ctx=_chat_ctx(items),
            tools=offered_tools(self._tools(), names),
        )

    async def _core_update_instructions(self, instructions: str) -> None:
        await self._session._update_instructions(instructions)

    async def _core_append_items(self, items: list[ChatItem]) -> None:
        await self._session._append_items(items)

    async def _core_update_tools(self, names: list[str]) -> None:
        await self._session._update_tools(offered_tools(self._tools(), names))

    async def _core_generate(
        self, instructions: str | None, tool_choice: llm.ToolChoice | None
    ) -> None:
        from .types import NOT_GIVEN

        self._session._generate_reply(
            instructions=instructions or NOT_GIVEN,
            tool_choice=NOT_GIVEN if tool_choice is None else tool_choice,
        )

    async def _core_next_audio(self) -> duplex.DuplexAudioFrame | None:
        try:
            item = await anext(self._audio)
        except StopAsyncIteration:
            return None
        return dataclasses.replace(item)

    async def _core_next_event(self) -> Any:
        return await self._events.get()

    async def _core_close_plugin(self) -> None:
        await self._session.aclose()

    async def _core_aclose(self) -> None:
        await self._session.aclose()
        self._events.put_nowait(None)


def _chat_ctx(items: list[ChatItem] | None) -> NotGivenOr[llm.ChatContext]:
    from . import llm
    from .types import NOT_GIVEN

    return NOT_GIVEN if items is None else llm.ChatContext(items)


class FallbackListener(Protocol):
    """A fallback adapter as its chain reports to it: an entry, by its index in the adapter's
    list, went unavailable or recovered; every entry is unavailable."""

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None: ...

    def _core_all_unavailable(self) -> None: ...


class TurnDetector:
    """A ``turn_detection`` object as the core's end-of-turn detector: its ``stream()`` hears
    the user's audio and predicts the end of their turn."""

    def __init__(self, detector: _StreamingTurnDetector) -> None:
        if not hasattr(detector, "stream"):
            raise TypeError(
                f"turn_detection={detector!r} is not an audio turn detector (one with stream()); "
                "use livekit.agents.inference.TurnDetector"
            )
        self._detector = detector
        self._stream = detector.stream()

    def _core_id(self) -> ProviderId:
        cls = type(self._detector)
        label = f"{cls.__module__}.{cls.__name__}"
        return {"label": label, "model": self._detector.model, "provider": self._detector.provider}

    def _core_push_audio(self, frame: rtc.AudioFrame) -> None:
        self._stream.push_audio(frame)

    def _core_cancel(self, timed_out: bool) -> None:
        self._stream.cancel_inference(timed_out=timed_out)

    def _core_flush(self, reason: str | None) -> None:
        self._stream.flush(reason)

    async def _core_predict(self, timeout: float | None) -> EotPrediction:
        from ._generated.turn_events import EotPrediction

        ev = await self._stream.predict()
        return EotPrediction(
            end_of_turn_probability=ev.end_of_turn_probability,
            last_speaking_time=ev.last_speaking_time,
            detection_delay=ev.detection_delay,
            inference_duration=ev.inference_duration,
            backchannel_probability=ev.backchannel_probability,
        )

    async def _core_aclose(self) -> None:
        await self._stream.aclose()


# --- io legs ----------------------------------------------------------------------------------


class _AudioInput:
    def __init__(self, obj: io.AudioInput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    async def _core_next_frame(self) -> Any:
        try:
            return await self._obj.__anext__()
        except StopAsyncIteration:
            return None

    def _core_on_attached(self) -> None:
        self._obj.on_attached()

    def _core_on_detached(self) -> None:
        self._obj.on_detached()


class _VideoInput(_AudioInput):
    async def _core_next_frame(self) -> Any:
        from ._generated import rtc_wire

        frame = await super()._core_next_frame()
        if frame is None:
            return None
        return rtc_wire.VideoFrameRef(
            frame=frame, width=frame.width, height=frame.height, timestamp_us=frame.timestamp_us
        )


class _AudioOutput:
    """``io.AudioOutput`` as a core sink, and as the listener of a core leg it wraps: the sink's
    playback reports go into its ``PlaybackTracker``, the leg's into its ``on_playback_*``."""

    def __init__(self, obj: io.AudioOutput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    def _core_tracker(self) -> _core.PlaybackTracker:
        return self._obj._tracker

    def _core_capabilities(self) -> io.AudioOutputCapabilities:
        from .voice import io

        return io.AudioOutputCapabilities(pause=self._obj.can_pause)

    def _core_sample_rate(self) -> int | None:
        return self._obj.sample_rate

    def _core_begin_segment(self, segment: io.PlaybackSegment) -> None:
        self._obj.begin_segment(segment)

    async def _core_capture(self, frame: rtc.AudioFrame) -> None:
        await self._obj.capture_frame(frame)

    def _core_flush_segment(self) -> None:
        self._obj.flush()

    def _core_clear_buffer(self) -> None:
        self._obj.clear_buffer()

    def _core_on_attached(self) -> None:
        self._obj.on_attached()

    def _core_on_detached(self) -> None:
        self._obj.on_detached()

    def _core_pause(self) -> None:
        self._obj.pause()

    def _core_resume(self) -> None:
        self._obj.resume()

    def _core_on_playback_started(self, segment: io.PlaybackSegment, created_at: float) -> None:
        self._obj.on_playback_started(segment=segment, created_at=created_at)

    def _core_on_playback_progressed(
        self, segment: io.PlaybackSegment, started_at: float, offset: float, duration: float
    ) -> None:
        self._obj.on_playback_progressed(
            segment=segment, started_at=started_at, offset=offset, duration=duration
        )

    def _core_on_playback_finished(
        self,
        segment: io.PlaybackSegment,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None,
    ) -> None:
        self._obj.on_playback_finished(
            segment=segment,
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )


class _TextOutput:
    def __init__(self, obj: io.TextOutput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    async def _core_capture_timed(self, text: TranscriptChunk) -> None:
        await self._obj.capture_transcript(text)

    async def _core_capture_text(self, text: str) -> None:
        await self._obj.capture_text(text)

    def _core_flush(self) -> None:
        self._obj.flush()

    def _core_on_attached(self) -> None:
        self._obj.on_attached()

    def _core_on_detached(self) -> None:
        self._obj.on_detached()


@functools.cache
def _views() -> dict[type, Any]:
    from .voice import io
    from .voice.room_io import Room

    return {
        io.AudioInput: _AudioInput,
        io.VideoInput: _VideoInput,
        io.AudioOutput: _AudioOutput,
        io.TextOutput: _TextOutput,
        Room: Room.of,
    }


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


def _status_retryable(status_code: int, retryable: bool | None) -> bool:
    # Preserve the public constructor's transient 499 behavior. Native cancellation has its
    # own policy and remains non-retryable.
    if 400 <= status_code < 500 and status_code not in (408, 429, 499):
        return False
    return True if retryable is None else retryable


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
