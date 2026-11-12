from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

from livekit import rtc

from .. import _core, llm, stt, tts
from .._bindings import chat_request, tool_specs
from .._generated.agent_run import FunctionToolInfo
from .._generated.amd import AMDPredictionEvent
from .._generated.avatar_errors import AvatarError
from .._generated.output_errors import AudioOutputError
from .._generated.session_dispatch import IGNORED_EVENTS, PUBLIC_EVENTS
from .._generated.session_wire import (
    CloseWire,
    DeleteRoomRequested,
    HostWire,
    ItemAddedWire,
    MetricsCollectedWire,
    RunFailedWire,
    RunFinishedWire,
    RunRefusedWire,
    SessionErrorWire,
    TextInputWire,
    ToolExecutionUpdatedWire,
)
from .._generated.user_turn import CompletedUserTurn
from .._generated.voice_session import VoiceCommandOrigin, VoiceCommandOriginHook
from ..job import get_job_context
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
    AgentEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    FunctionToolsExecutedEvent,
    RunContext,
    ToolExecutionUpdatedEvent,
    UserTurnExceededEvent,
    event_dict,
)
from .io import SpeechHandle
from .room_io import AudioProcessingError, TextInputEvent

if TYPE_CHECKING:
    from .._generated.agent_run import RunFinished
    from .agent_session import UserTurnExceededHook, VoiceSession


_NodeCallsVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_node_calls")


class _DynamicInstructions:
    def __init__(self, session: VoiceSession[Any], agent: Agent) -> None:
        self._session = session
        self._agent = agent

    async def _core_render(self, speech: _core.SpeechHandle) -> str:
        ctx = RunContext(
            function_call=None, session=self._session, speech_handle=SpeechHandle._of(speech)
        )
        return await self._agent._render_dynamic_instructions(ctx)


def _live_binding(operation: str) -> _VoiceBinding:
    binding = _BindingContextVar.get(None)
    if not isinstance(binding, _VoiceBinding):
        raise RuntimeError(
            f"{operation} needs a live VoiceSession: await it inside a tool of the agent "
            "holding the call"
        )
    return binding


def _origin() -> VoiceCommandOrigin | None:
    binding = _BindingContextVar.get(None)
    return binding.origin if isinstance(binding, _VoiceBinding) else None


def _items(chat_ctx: ChatContext) -> list[llm.ChatItem]:
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
    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._toolsets = _core._core_ToolsetScopes()

    def _core_toolset_resources(self, tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Toolset]:
        return [tool for tool in self._for_call(tools) if isinstance(tool, llm.Toolset)]

    def _for_call(self, tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Tool | llm.Toolset]:
        return tools_for_run(list(tools), RunContext(function_call=None, session=self._session))

    async def _core_on_enter(self, agent: Agent) -> None:
        binding = self._session._session_binding
        if binding is not None:
            binding.update_voice(agent.name, [item.id for item in self._session.history.items])
        with _in_hook("on_enter", self._session, agent):
            await agent._run_hook("on_enter")

    async def _core_on_exit(self, agent: Agent) -> None:
        with _in_hook("on_exit", self._session, agent):
            await agent._run_hook("on_exit")

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
    ) -> CompletedUserTurn:
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
        return CompletedUserTurn(
            user_text=message.text_content,
            items=turn_ctx.items if changed else None,
            stop_response=stop,
        )

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
    """The built-in pipeline steps, exposed as ``session.default``."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    def llm_node(
        self,
        ctx: RunContext[Any],
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        """Send a request to the session's LLM."""
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
        """Run the session's STT; a non-streaming STT is wrapped with the session's VAD."""
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
        """Run the session's TTS; a non-streaming TTS is fed sentence by sentence."""
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
        """Yield the transcript unchanged."""
        async for delta in text:
            yield delta

    async def on_user_turn_exceeded(self, ctx: RunContext[Any], ev: UserTurnExceededEvent) -> None:
        """Generate a short reply that cuts in on the user, from the turn's transcript."""
        await self._session.generate_reply(
            user_input=ev.transcript,
            **_core.user_turn_exceeded_reply(),
        )


def _stamped(node_frame: Any) -> rtc.AudioFrame:
    frame: rtc.AudioFrame = node_frame.frame
    if node_frame.timed_transcript:
        frame.userdata[USERDATA_TIMED_TRANSCRIPT] = node_frame.timed_transcript
    if node_frame.started_time is not None:
        elapsed = time.time() - node_frame.started_time
        frame.userdata[USERDATA_TTS_STARTED_TIME] = time.perf_counter() - elapsed
    return frame


async def _node_input(stream: Any, item: Callable[[Any], Any] = lambda x: x) -> AsyncIterator[Any]:
    while (value := await stream.recv()) is not None:
        yield item(value)


def _transcript_chunk(value: str | TranscriptChunk) -> TranscriptChunk:
    return TranscriptChunk(value) if isinstance(value, str) else value


async def _node_output(
    node: Callable[[], Any], item: Callable[[Any], Any] = lambda x: x
) -> AsyncIterator[Any]:
    # a node may return an awaitable, an async iterable, a single value or None
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


def _on_item_added(self: VoiceSession[Any], wire: ItemAddedWire) -> None:
    binding = self._session_binding
    if binding is not None and binding.state.active_agent is not None:
        binding.update_voice(binding.state.active_agent, [item.id for item in self.history.items])
    # the same object as in `session.history`
    if (item := wire.host_item) is None:
        return
    if item.type in ("message", "agent_handoff"):
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))


def _on_run_refused(self: VoiceSession[Any], wire: RunRefusedWire) -> None:
    if (run := _wire_run(wire.run)) is not None and not run.done():
        run.set_exception(RuntimeError(f"run refused: {wire.cause}"))


def _on_run_finished(self: VoiceSession[Any], wire: RunFinishedWire) -> None:
    if (run := _wire_run(wire.run)) is not None and not run.done():
        run.set_result(wire.settled)


def _on_run_failed(self: VoiceSession[Any], wire: RunFailedWire) -> None:
    if (run := _wire_run(wire.run)) is None or run.done():
        return
    error = wire.provider_error or wire.error
    error.__dict__["input_requirements"] = list(wire.input_requirements)
    run.set_exception(error)


def _on_tool_update(self: VoiceSession[Any], wire: ToolExecutionUpdatedWire) -> None:
    self.emit("tool_execution_updated", ToolExecutionUpdatedEvent(update=wire.update))


def _on_metrics(self: VoiceSession[Any], wire: MetricsCollectedWire) -> None:
    record = wire.metrics
    # a record metered for a provider (e.g. the VAD) is re-emitted on that provider
    if (leg := wire.leg) is not None and (
        provider := getattr(_VoiceBinding(self, self.current_agent), leg)
    ) is not None:
        provider.emit("metrics_collected", record)


def _error_record(self: VoiceSession[Any], wire: SessionErrorWire) -> tuple[Any, Any]:
    if (failure := wire.processor_failure) is not None:
        error = AudioProcessingError(label=failure.label, error=cast(Exception, failure.error))
        return error, failure.processor
    if wire.source == "avatar_error":
        return AvatarError(error=cast(Exception, wire.error)), None
    if wire.source == "audio_output_error":
        output = self.output.audio
        label = getattr(output, "label", "audio_output")
        return AudioOutputError(label=label, error=cast(Exception, wire.error)), output
    source = getattr(_VoiceBinding(self, self.current_agent), cast(str, wire.slot))
    error = _ERROR_RECORDS[wire.source](
        timestamp=time.time(),
        label=getattr(source, "label", wire.label),
        error=wire.error,
        recoverable=wire.recoverable,
    )
    return error, source


def _core_error_message(self: VoiceSession[Any], wire: SessionErrorWire) -> str:
    return str(_error_record(self, wire)[0])


def _on_error(self: VoiceSession[Any], wire: SessionErrorWire) -> None:
    failure = wire.processor_failure
    logger.log(
        logging.WARNING if wire.recoverable else logging.ERROR,
        "session error",
        extra={"source": wire.source, "error": str(wire.error)},
        exc_info=cast(BaseException, failure.error) if failure is not None else None,
    )
    error, source = _error_record(self, wire)
    self.emit("error", ErrorEvent(error=error, source=source))


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
    self.emit("close", CloseEvent(reason=wire.reason, error=error))


# keyed by the record's `type`, which the wire carries as `source`
_ERROR_RECORDS: dict[str, Any] = {
    record.model_fields["type"].default: record
    for record in (llm.LLMError, tts.TTSError, stt.STTError, llm.RealtimeModelError)
}


def _wire_run(run: object | None) -> asyncio.Future[RunFinished] | None:
    # the future `VoiceSession._run()` handed the core
    return cast("asyncio.Future[RunFinished] | None", run)


_HANDLERS: dict[type, Callable[[VoiceSession[Any], Any], None]] = {
    ItemAddedWire: _on_item_added,
    RunRefusedWire: _on_run_refused,
    RunFinishedWire: _on_run_finished,
    RunFailedWire: _on_run_failed,
    ToolExecutionUpdatedWire: _on_tool_update,
    MetricsCollectedWire: _on_metrics,
    SessionErrorWire: _on_error,
    CloseWire: _on_close,
    AMDPredictionEvent: _on_amd_result,
    TextInputWire: _on_text_input,
    DeleteRoomRequested: _on_delete_room,
}


def report_snapshots(self: VoiceSession[Any]) -> list[dict[str, Any]]:
    return [event for event in self._event_snapshots if self._native.accepts_event(event["type"])]


def record_snapshot(self: VoiceSession[Any], arg: AgentEvent) -> None:
    # snapshot as JSON so the report holds no live objects (model clients, exceptions)
    if self._native.admit_event(arg.type):
        self._event_snapshots.append(event_dict(arg, json=True))
