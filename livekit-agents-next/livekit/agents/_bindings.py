"""The hand-written method bodies of the generated models (``_generated/*.py`` call them)."""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from livekit import rtc

from . import _core
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
    from ._generated.usage import TokenUsage
    from .types import APIConnectOptions, NotGivenOr
    from .voice.agent_session import VoiceSession
    from .voice.room_io import TextInputEvent
import dataclasses
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, Never

if TYPE_CHECKING:
    from . import llm
    from ._generated import duplex, realtime_wire, rtc_wire
    from ._generated.chat_items import MetricsMetadata
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


def built_on_loop(owner: object) -> list[_FallbackChain]:
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


def metrics_repr(metrics: BaseModel) -> str:
    fields = metrics.model_dump(exclude_defaults=True)
    return f"{type(metrics).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def token_usage_total(tokens: TokenUsage) -> int:
    return tokens.input_tokens + tokens.output_tokens


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
    updates in order and pulls the output audio; the session's callbacks queue one event at a time
    into ``events``. The core calls the plugin's own update methods directly; the methods here
    resolve tool names and convert records."""

    def __init__(
        self, session: llm.DuplexSession, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        from ._generated import realtime_wire

        self.plugin = session
        self._tools = tools
        self._audio = aiter(session.audio_stream)
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

    async def _core_next_audio(self) -> duplex.DuplexAudioFrame | None:
        try:
            item = await anext(self._audio)
        except StopAsyncIteration:
            return None
        return dataclasses.replace(item)

    async def _core_aclose(self) -> None:
        await self.plugin.aclose()
        self.events.put_nowait(None)


def _chat_ctx(items: list[ChatItem] | None) -> NotGivenOr[llm.ChatContext]:
    from . import llm
    from .types import NOT_GIVEN

    return NOT_GIVEN if items is None else llm.ChatContext(items)


class _Member(Protocol):
    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def metrics_metadata(self) -> MetricsMetadata: ...


_M = TypeVar("_M", bound=_Member)


class FallbackMembers(Generic[_M]):
    """What the STT, TTS and LLM ``FallbackAdapter`` share over their ``_members``: the member
    the running loop's chain is serving (``_active``, the first before any chain is built) names
    the adapter's ``model``, ``provider`` and ``metrics_metadata``; every member unavailable is
    logged as ``all <_kind>s are unavailable``; ``aclose`` closes the chain of every loop."""

    _kind: ClassVar[str]
    _members: list[_M]

    @property
    def _active(self) -> _M:
        chains = built_on_loop(self)
        return self._members[chains[0].active_index if chains else 0]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    def _core_all_unavailable(self) -> None:
        logger.error(f"all {self._kind}s are unavailable, retrying..")

    async def aclose(self) -> None:
        await super().aclose()  # type: ignore[misc]
        for chain in built_on_loop(self):
            await chain.aclose()


class FallbackListener(Protocol):
    """A fallback adapter as its chain reports to it: an entry, by its index in the adapter's
    list, went unavailable or recovered; every entry is unavailable."""

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None: ...

    def _core_all_unavailable(self) -> None: ...


class TurnDetector:
    """A ``turn_detection`` object as the core's end-of-turn detector: its ``stream()`` (opened
    here as ``stream``) hears the user's audio and predicts the end of their turn. The core pushes
    audio, cancels, flushes and closes the stream directly; the methods here report the
    detector's identity and convert its prediction."""

    def __init__(self, detector: _StreamingTurnDetector) -> None:
        if not hasattr(detector, "stream"):
            raise TypeError(
                f"turn_detection={detector!r} is not an audio turn detector (one with stream()); "
                "use livekit.agents.inference.TurnDetector"
            )
        self._detector = detector
        self.stream = detector.stream()

    def _core_id(self) -> ProviderId:
        cls = type(self._detector)
        label = f"{cls.__module__}.{cls.__name__}"
        return {"label": label, "model": self._detector.model, "provider": self._detector.provider}

    async def _core_predict(self, timeout: float | None) -> EotPrediction:
        from ._generated.turn_events import EotPrediction

        ev = await self.stream.predict()
        return EotPrediction(
            end_of_turn_probability=ev.end_of_turn_probability,
            last_speaking_time=ev.last_speaking_time,
            detection_delay=ev.detection_delay,
            inference_duration=ev.inference_duration,
            backchannel_probability=ev.backchannel_probability,
        )


# --- io legs ---------------------------------------------------------------------------------


async def next_audio_frame(source: io.AudioInput) -> rtc.AudioFrame | None:
    """The core's read of an ``io.AudioInput``: its next frame, ``None`` once it ends."""
    try:
        return await source.__anext__()
    except StopAsyncIteration:
        return None


async def next_video_frame(source: io.VideoInput) -> rtc_wire.VideoFrameRef | None:
    """The core's read of an ``io.VideoInput``: its next frame as the core's side-table key,
    ``None`` once it ends."""
    from ._generated import rtc_wire

    try:
        # the source's frame carries its capture time, which ``rtc.VideoFrame`` does not declare
        frame: Any = await source.__anext__()
    except StopAsyncIteration:
        return None
    return rtc_wire.VideoFrameRef(
        frame=frame, width=frame.width, height=frame.height, timestamp_us=frame.timestamp_us
    )


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
