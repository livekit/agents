"""The session's host callbacks: application hooks, Python async iterators and provider adapters,
and the translation of native events into Python SDK events."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .. import _core, llm, stt, tts
from .._bindings import chat_request, tool_specs
from .._generated.agent_run import FunctionToolInfo
from .._generated.voice_session import VoiceCommandOrigin, VoiceCommandOriginHook
from ..llm import (
    ChatContext,
    StopResponse,
)
from ..llm.tool_context import tools_for_run
from ..log import logger
from ..types import (
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    FlushSentinel,
    TranscriptChunk,
)
from ..utils import is_given
from . import Agent, ModelSettings, _BindingContextVar, _VoiceBinding
from .agent_run import tool_infos
from .events import (
    FunctionToolsExecutedEvent,
    RunContext,
    UserTurnExceededEvent,
)
from .io import SpeechHandle

if TYPE_CHECKING:
    from .agent_session import UserTurnExceededHook, VoiceSession
import asyncio
import logging
from typing import TYPE_CHECKING, Literal, cast

from .._generated.amd import AMDPredictionEvent
from .._generated.session_dispatch import IGNORED_EVENTS, PUBLIC_EVENTS
from .._generated.session_wire import (
    CloseWire,
    DeleteRoomRequested,
    HostWire,
    ItemAddedWire,
    MetricsCollectedWire,
    OverlappingSpeechWire,
    RunItemAddedWire,
    RunRefusedWire,
    RunSettledWire,
    SessionErrorWire,
    SpeechCreatedWire,
    TextInputWire,
    ToolCallStartedWire,
    ToolExecutionUpdatedWire,
)
from ..job import get_job_context
from .agent_run import run_error
from .events import (
    AgentEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    SpeechCreatedEvent,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
    ToolUpdate,
    event_dict,
)
from .room_io import AudioProcessingError, TextInputEvent
from .run_result import _VoiceRun

if TYPE_CHECKING:
    from .agent_session import VoiceSession


_NodeCallsVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_node_calls")


class _DynamicInstructions:
    """An agent's ``@agent.instructions`` functions, rendered by the core before each model
    request of a reply (``_core_render``)."""

    def __init__(self, session: VoiceSession[Any], agent: Agent) -> None:
        self._session = session
        self._agent = agent

    async def _core_render(self, speech: _core.SpeechHandle) -> str:
        ctx = RunContext(
            function_call=None, session=self._session, speech_handle=SpeechHandle._of(speech)
        )
        return await self._agent._render_dynamic_instructions(ctx)


def _live_binding(operation: str) -> _VoiceBinding:
    """The live session interaction the current hook or tool runs in; raises outside one."""
    binding = _BindingContextVar.get(None)
    if not isinstance(binding, _VoiceBinding):
        raise RuntimeError(
            f"{operation} needs a live VoiceSession: await it inside a tool of the agent "
            "holding the call"
        )
    return binding


def _origin() -> VoiceCommandOrigin | None:
    """What the running hook or tool sends its commands as; ``None`` from outside the session."""
    binding = _BindingContextVar.get(None)
    return binding.origin if isinstance(binding, _VoiceBinding) else None


def _items(chat_ctx: ChatContext) -> list[llm.ChatItem]:
    """The items a command sends for ``chat_ctx``."""
    return chat_ctx.items


@contextlib.contextmanager
def _in_hook(hook: str, session: VoiceSession[Any], agent: Agent) -> Any:
    origin = VoiceCommandOriginHook(hook=hook)
    binding = _BindingContextVar.set(_VoiceBinding(session, agent, origin=origin))
    try:
        yield
    finally:
        _BindingContextVar.reset(binding)


class _AgentHooks:
    """The agent's hooks as the core awaits them (``_core.SessionHooks``): each runs in the agent's
    binding and returns the core's answer; ``render_tool_reply`` is the core's default."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._toolsets = _core._core_ToolsetScopes()

    def _core_toolset_scopes(self) -> _core._core_ToolsetScopes:
        return self._toolsets

    def _core_toolset_resources(self, tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Toolset]:
        return [tool for tool in self._for_call(tools) if isinstance(tool, llm.Toolset)]

    def _for_call(self, tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Tool | llm.Toolset]:
        """``tools`` as this call uses them (``Toolset.for_run``)."""
        return tools_for_run(list(tools), RunContext(function_call=None, session=self._session))

    async def _core_on_enter(self, agent: Agent) -> None:
        with _in_hook("on_enter", self._session, agent):
            await agent._enter(RunContext(function_call=None, session=self._session))

    async def _core_on_exit(self, agent: Agent) -> None:
        with _in_hook("on_exit", self._session, agent):
            await agent.on_exit()

    async def _core_on_user_turn_exceeded(self, agent: Agent, ev: UserTurnExceededEvent) -> None:
        session = self._session
        hook: UserTurnExceededHook[Any] = session._steps.get(
            "on_user_turn_exceeded", session.default.on_user_turn_exceeded
        )
        with _in_hook("on_user_turn_exceeded", session, agent):
            await hook(RunContext(function_call=None, session=session), ev)

    def _core_tool_infos(self, tools: list[llm.Tool | llm.Toolset]) -> list[FunctionToolInfo]:
        tools = self._for_call(tools)
        infos = {info.spec["name"]: info for info in tool_infos(llm.ToolContext(tools))}
        return [
            infos.get(spec["name"]) or FunctionToolInfo(spec=spec) for spec in tool_specs(tools)
        ]

    async def _core_closing(self) -> None:
        if self._session._amd is not None:
            await self._session._amd.aclose()

    async def _core_on_user_turn_completed(
        self,
        agent: Agent,
        chat_ctx: list[llm.ChatItem],
        new_transcript: str,
        transcript_confidence: float,
    ) -> tuple[str | None, list[llm.ChatItem] | None, bool]:
        session = self._session
        turn_ctx = ChatContext(list(chat_ctx))
        message = llm.ChatMessage(
            role="user", content=[new_transcript], transcript_confidence=transcript_confidence
        )
        before = [i.id for i in turn_ctx.items]
        stop = False
        if (hook := session._steps.get("on_user_turn_completed")) is not None:
            with _in_hook("on_user_turn_completed", session, agent):
                try:
                    await hook(RunContext(function_call=None, session=session), turn_ctx, message)
                except StopResponse:
                    stop = True
        changed = [i.id for i in turn_ctx.items] != before
        return message.text_content, turn_ctx.items if changed else None, stop

    async def _core_function_tools_executed(
        self,
        function_calls: list[llm.ChatItem],
        function_call_outputs: list[llm.ChatItem],
    ) -> list[bool]:
        ev = FunctionToolsExecutedEvent(
            function_calls=list(function_calls),
            function_call_outputs=list(function_call_outputs),
        )
        self._session.emit("function_tools_executed", ev)
        return [o.reply_required for o in ev.function_call_outputs]

    async def _core_filler_source(self, source: Callable[[int], Any], step: int) -> Any:
        out = source(step)
        return await out if inspect.isawaitable(out) else out

    def _core_hook_failed(self, hook: str, error: BaseException | None) -> None:
        logger.error(f"error in the {hook} hook", exc_info=error)


class _SessionNodes:
    """The session's registered nodes as the core calls them (``_core.AgentNodes``), each call in
    a copy of the agent's binding: one completion, one spoken segment, one recognition stream, one
    segment's transcript. The core feeds what they read (its
    streams, read with ``recv()``) and drains what they return."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    def _call(self, step: str, *args: Any, settings: ModelSettings | None = None) -> Any:
        ctx = RunContext(function_call=None, session=self._session)
        return self._session._steps[step](ctx, *args, settings or ModelSettings())

    def llm(self, req: _core.ChatRequest, calls: Any) -> AsyncIterator[Any]:
        chat_ctx, tools = chat_request(req)
        choice = req.tool_choice
        settings = ModelSettings(tool_choice=NOT_GIVEN if choice is None else choice)

        def node() -> Any:
            _NodeCallsVar.set(calls)
            return self._call("llm_node", chat_ctx, tools, settings=settings)

        return _node_output(node)

    def tts(self, text: Any, calls: Any) -> AsyncIterator[Any]:
        def node() -> Any:
            _NodeCallsVar.set(calls)
            return self._call("tts_node", _node_input(text))

        return _node_output(node, lambda frame: tts.SynthesizedAudio(frame=frame, request_id=""))

    def stt(self, audio: Any, calls: Any) -> AsyncIterator[Any]:
        def node() -> Any:
            _NodeCallsVar.set(calls)
            return self._call("stt_node", _node_input(audio))

        return _node_output(node)

    def transcription(self, text: Any) -> AsyncIterator[Any]:
        words = _node_input(text)
        return _node_output(lambda: self._call("transcription_node", words), _transcript_chunk)


class DefaultPipeline:
    """``session.default``: the built-in pipeline steps, with the callbacks' signatures."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    def llm_node(
        self,
        ctx: RunContext[Any],
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        """A request to the session's LLM."""
        model = self._session.llm
        assert isinstance(model, llm.LLM), "llm_node needs an LLM (not a realtime model)"
        choice = model_settings.tool_choice
        return _core.llm_node(
            model,
            chat_ctx.items,
            chat_ctx,
            tools=tool_specs(tools),
            tools_host=tools,
            tool_choice=choice if is_given(choice) else None,
            conn_options=self._session.conn_options.llm_conn_options,
            calls=_NodeCallsVar.get(None),
        )

    def stt_node(
        self,
        ctx: RunContext[Any],
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterator[stt.SpeechEvent]:
        """The session's STT; one that does not stream is wrapped with its VAD."""
        session = self._session
        if session.stt is None:
            raise RuntimeError("stt_node called but no STT is available")
        return _core.stt_node(
            session.stt,
            audio,
            conn_options=session.conn_options.stt_conn_options,
            vad=session.vad,
            calls=_NodeCallsVar.get(None),
        )

    def tts_node(
        self, ctx: RunContext[Any], text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        """The session's TTS; one that does not stream is fed sentence by sentence."""
        session = self._session
        if session.tts is None:
            raise RuntimeError("tts_node called but no TTS is available")
        conn_options = session.conn_options.tts_conn_options
        calls = _NodeCallsVar.get(None)
        frames = _core.tts_node(session.tts, text, conn_options=conn_options, calls=calls)
        return (_stamped(frame) async for frame in frames)

    async def transcription_node(
        self,
        ctx: RunContext[Any],
        text: AsyncIterable[TranscriptChunk],
        model_settings: ModelSettings,
    ) -> AsyncIterator[str | TranscriptChunk]:
        """The transcript as spoken."""
        async for delta in text:
            yield delta

    async def on_user_turn_exceeded(self, ctx: RunContext[Any], ev: UserTurnExceededEvent) -> None:
        """A short reply cutting in on the person, from the turn's transcript."""
        await self._session.generate_reply(
            user_input=ev.transcript,
            **_core.user_turn_exceeded_reply(),
        )


def _stamped(node_frame: Any) -> rtc.AudioFrame:
    """A default `tts_node` frame as the reference yields it: its `userdata` carries the words it
    covers and when its synthesis started (a `perf_counter()` stamp)."""
    frame: rtc.AudioFrame = node_frame.frame
    if node_frame.timed_transcript:
        frame.userdata[USERDATA_TIMED_TRANSCRIPT] = node_frame.timed_transcript
    if node_frame.started_time is not None:
        elapsed = time.time() - node_frame.started_time
        frame.userdata[USERDATA_TTS_STARTED_TIME] = time.perf_counter() - elapsed
    return frame


async def _node_input(stream: Any, item: Callable[[Any], Any] = lambda x: x) -> AsyncIterator[Any]:
    """A core stream as the node reads it."""
    while (value := await stream.recv()) is not None:
        yield item(value)


def _transcript_chunk(value: str | TranscriptChunk) -> TranscriptChunk:
    """Normalize plain text once; existing chunks retain every metadata field."""
    return TranscriptChunk(value) if isinstance(value, str) else value


async def _node_output(
    node: Callable[[], Any], item: Callable[[Any], Any] = lambda x: x
) -> AsyncIterator[Any]:
    """What a node returned, read as the reference's node task reads it: an awaitable is awaited
    first, an async iterable is read to its end (and closed with this one), a single value is the
    one item, ``None`` is none."""
    out = node()
    if inspect.isawaitable(out):
        out = await out
    if out is None:
        return
    if not isinstance(out, AsyncIterable):
        yield item(out)
        return
    iterator = aiter(out)
    try:
        async for value in iterator:
            yield item(value)
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


def dispatch(self: VoiceSession[Any], wire: HostWire) -> None:
    kind = type(wire)
    if kind in IGNORED_EVENTS:
        return
    if (name := PUBLIC_EVENTS.get(kind)) is not None:
        self.emit(cast(EventTypes, name), cast(AgentEvent, wire))
        return
    handler = _HANDLERS.get(kind)
    if handler is not None:
        handler(self, wire)


def _on_speech_created(self: VoiceSession[Any], wire: SpeechCreatedWire) -> None:
    handle = SpeechHandle._of(cast(_core.SpeechHandle, wire.speech_handle))
    self.emit(
        "speech_created",
        SpeechCreatedEvent(
            user_initiated=wire.user_initiated,
            source=cast(Literal["say", "generate_reply"], wire.source),
            speech_handle=handle,
        ),
    )


def _on_item_added(self: VoiceSession[Any], wire: ItemAddedWire) -> None:
    # the history's own object: the event, the run and `session.history` share it
    if (item := wire.host_item) is None:
        return
    if (run := _wire_run(wire.run)) is not None and item.type != "agent_handoff":
        run._item_added(item)
    if item.type in ("message", "agent_handoff"):
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))


def _on_run_item(self: VoiceSession[Any], wire: RunItemAddedWire) -> None:
    if (run := _wire_run(wire.run)) is not None:
        run._item_added(wire.item)


def _on_run_refused(self: VoiceSession[Any], wire: RunRefusedWire) -> None:
    if (run := _wire_run(wire.run)) is not None:
        run._settle(output=None, error=RuntimeError(f"run refused: {wire.cause}"))


def _on_run_settled(self: VoiceSession[Any], wire: RunSettledWire) -> None:
    if (run := _wire_run(wire.run)) is None:
        return
    error: BaseException | None = None
    if (failure := wire.run_error) is not None:
        error = run_error(failure)
    speech = cast("_core.SpeechHandle | None", wire.speech_handle)
    if wire.outcome == "failed" and error is None and speech is not None:
        error = speech.exception()
    if wire.outcome == "closed":
        error = RuntimeError("the session closed before the run settled")
    run._settle(
        status=wire.outcome,
        output=wire.final_output,
        error=error,
        question=wire.question,
        input_requirements=list(wire.input_requirements),
    )


def _on_tool_update(self: VoiceSession[Any], wire: ToolExecutionUpdatedWire) -> None:
    update: ToolUpdate = (
        ToolCallStarted(function_call=wire.update.function_call)
        if isinstance(wire.update, ToolCallStartedWire)
        else wire.update
    )
    self.emit("tool_execution_updated", ToolExecutionUpdatedEvent(update=update))


def _on_metrics(self: VoiceSession[Any], wire: MetricsCollectedWire) -> None:
    # the usage counters are the core's, recorded where the session emits the record
    record = wire.metrics
    # a record the session metered for a provider (its VAD's) is that provider's event
    if (leg := wire.leg) is not None and (
        provider := getattr(_VoiceBinding(self, self.current_agent), leg)
    ) is not None:
        provider.emit("metrics_collected", record)


def _error_record(self: VoiceSession[Any], wire: SessionErrorWire) -> tuple[Any, Any]:
    """The reference's payload: the provider's error record, and the provider that failed."""
    if (failure := wire.processor_failure) is not None:
        error = AudioProcessingError(label=failure.label, error=cast(Exception, failure.error))
        return error, failure.processor
    record, leg = _ERROR_RECORDS[wire.source]
    source = getattr(_VoiceBinding(self, self.current_agent), leg)
    error = record(
        timestamp=time.time(),
        label=getattr(source, "label", wire.label),
        error=wire.error,
        recoverable=wire.recoverable,
    )
    return error, source


def _core_error_message(self: VoiceSession[Any], wire: SessionErrorWire) -> str:
    return str(_error_record(self, wire)[0])


def _on_error(self: VoiceSession[Any], wire: SessionErrorWire) -> None:
    # A recoverable error is one the call goes on through; a processor's own exception keeps
    # its traceback.
    failure = wire.processor_failure
    logger.log(
        logging.WARNING if wire.recoverable else logging.ERROR,
        "session error",
        extra={"source": wire.source, "error": str(wire.error)},
        exc_info=cast(BaseException, failure.error) if failure is not None else None,
    )
    error, source = _error_record(self, wire)
    self.emit("error", ErrorEvent(error=error, source=source))


def _on_overlapping_speech(self: VoiceSession[Any], wire: OverlappingSpeechWire) -> None:
    self.emit("overlapping_speech", wire.event)


def _on_amd_result(self: VoiceSession[Any], wire: AMDPredictionEvent) -> None:
    if self._amd is not None:
        self._amd._on_prediction(wire)


def _on_text_input(self: VoiceSession[Any], wire: TextInputWire) -> None:
    if (callback := self._text_input_cb) is None:
        return
    room = self._room
    participants = room.remote_participants if isinstance(room, rtc.Room) else {}
    ev = TextInputEvent(
        text=wire.text,
        info=wire.info if isinstance(wire.info, rtc.TextStreamInfo) else None,
        participant=participants.get(wire.participant),
    )

    async def _run() -> None:
        try:
            if inspect.isawaitable(result := callback(self, ev)):
                await result
        except Exception:
            logger.warning("failed to handle chat text stream", exc_info=True)

    task = asyncio.ensure_future(_run())
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)


def _on_delete_room(self: VoiceSession[Any], wire: DeleteRoomRequested) -> None:
    if (job_ctx := get_job_context(required=False)) is None:
        return
    logger.info(
        "deleting room on agent session close "
        "(disable via `RoomOptions.delete_room_on_close=False`)",
        extra={"room": wire.room},
    )
    job_ctx.delete_room(room_name=wire.room)


def _on_close(self: VoiceSession[Any], wire: CloseWire) -> None:
    error = _error_record(self, cause)[0] if (cause := wire.error) is not None else None
    self.emit("close", CloseEvent(reason=CloseReason(wire.reason), error=error))


_ERROR_RECORDS: dict[str, tuple[Any, str]] = {
    "llm_error": (llm.LLMError, "llm"),
    "tts_error": (tts.TTSError, "tts"),
    "stt_error": (stt.STTError, "stt"),
    "realtime_model_error": (llm.RealtimeModelError, "llm"),
}


def _wire_run(run: object | None) -> _VoiceRun | None:
    """A wire's run: the `_VoiceRun` the host handed the core."""
    return cast("_VoiceRun | None", run)


# The `HostWire` members with host behaviour, by record class.
_HANDLERS: dict[type, Callable[[VoiceSession[Any], Any], None]] = {
    SpeechCreatedWire: _on_speech_created,
    ItemAddedWire: _on_item_added,
    RunItemAddedWire: _on_run_item,
    RunRefusedWire: _on_run_refused,
    RunSettledWire: _on_run_settled,
    ToolExecutionUpdatedWire: _on_tool_update,
    MetricsCollectedWire: _on_metrics,
    OverlappingSpeechWire: _on_overlapping_speech,
    SessionErrorWire: _on_error,
    CloseWire: _on_close,
    AMDPredictionEvent: _on_amd_result,
    TextInputWire: _on_text_input,
    DeleteRoomRequested: _on_delete_room,
}


def report_snapshots(self: VoiceSession[Any]) -> list[dict[str, Any]]:
    return [event for event in self._event_snapshots if self._owner.accepts_event(event["type"])]


def record_snapshot(self: VoiceSession[Any], arg: AgentEvent) -> None:
    # the event as the report records it: no live objects (a model client, an exception)
    if self._owner.admit_event(arg.type):
        self._event_snapshots.append(event_dict(arg, json=True))
