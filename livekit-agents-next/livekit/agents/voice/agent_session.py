"""``VoiceSession``: the developer's handle on a session the Rust core runs.

The core (``_core.VoiceSession``) owns the whole session: turn detection, scheduling, generation,
tools' ordering, interruptions, the room's io. This class turns the constructor's arguments into
the core's, forwards each call as one command, mirrors the core's events into the public event
classes and answers the core's hook calls by running the ``Agent``'s methods.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import weakref
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

from livekit import rtc

from .. import _core, llm, stt, tts
from .._generated.expressive import ExpressiveOptions, NonverbalOptions, SpeechSteeringOptions
from .._generated.keyterms import KeytermDetectionOptions, STTContextOptions
from .._generated.session_commands import SessionCommands
from .._generated.session_wire import (
    HostWire,
    SessionErrorWire,
)
from .._generated.vad import VAD
from ..llm import (
    ChatContext,
    ChatItem,
)
from ..stt import STT
from ..tts import TTS
from ..types import (
    NOT_GIVEN,
    NotGivenOr,
)
from . import Agent, io
from ._session_bindings import (
    LLMNode as LLMNode,
    RecordingOptions as RecordingOptions,
    SessionAdapter,
    SessionConnectOptions as SessionConnectOptions,
    STTNode as STTNode,
    TextTransforms as TextTransforms,
    TranscriptionNode as TranscriptionNode,
    TTSNode as TTSNode,
    UserTurnCompletedHook as UserTurnCompletedHook,
    UserTurnExceededHook as UserTurnExceededHook,
    VoiceActivityVideoSampler as VoiceActivityVideoSampler,
    _CallbackT,
    initialize,
)
from ._session_callbacks import (
    DefaultPipeline as DefaultPipeline,
    _AgentHooks as _AgentHooks,
    _DynamicInstructions as _DynamicInstructions,
    _live_binding as _live_binding,
    _origin as _origin,
    _SessionNodes as _SessionNodes,
)
from .events import (
    AgentEvent,
    AgentState,
    CloseReason,
    Context_T,
    EventTypes,
    UserState,
)
from .io import SpeechHandle
from .room_io import (
    AudioInputOptions,
    RoomOptions,
    TextInputCallback,
)
from .run_result import _VoiceRun

if TYPE_CHECKING:
    from .._core import VoiceSession as _NativeSession, _core_SessionState as _SessionState
    from ..storage import Thread
    from ..storage.context import ContextStore
    from .amd import AMD
    from .io import EndpointingOptions, InterruptionOptions, TurnDetectionMode
from .._generated.voice_session import UpdateAgentArgs
from .._generated.voice_task_errors import VoiceTaskFailed as VoiceTaskFailed
from . import Output_T

# The session and the state of the event its callbacks are handling.
_delivered: contextvars.ContextVar[tuple[VoiceSession[Any], _SessionState]] = (
    contextvars.ContextVar("lk_session_delivered")
)


class VoiceSession(SessionCommands, rtc.EventEmitter[EventTypes], Generic[Context_T]):
    _context: Context_T
    _thread: Thread | None
    _execution_id: str
    _conn_options: SessionConnectOptions
    _options: _core.SessionOptions
    _owner: _core._core_SessionOwner
    _stt_context: STTContextOptions | None
    _stt: stt.STT | None
    _llm: llm.LLM | llm.RealtimeModel | None
    _tts: tts.TTS | None
    _vad: VAD | None
    _tools: list[llm.Tool | llm.Toolset]
    _using_default_vad: bool
    _turn_detector: Any
    _turn_detection_mode: str | None
    _interruption_mode: Literal["adaptive", "vad"] | None
    _video_sampler: VoiceActivityVideoSampler | None
    _start_chat_ctx: ChatContext | None
    _input: io.AgentInput
    _output: io.AgentOutput
    _event_retention: _core._core_EventRetention
    _recording_options: RecordingOptions
    _store: ContextStore | None
    _room: rtc.Room | None
    _amd: AMD | None
    _agent_state: AgentState
    _user_state: UserState
    _started_at: float | None
    _event_snapshots: deque[dict[str, Any]]
    _root_span_context: Any
    _text_input_cb: TextInputCallback | None
    _recorder: _core.SessionRecorder | None
    _steps: dict[str, Callable[..., Any]]
    _recording_path: Path | None
    _audio_input_options: AudioInputOptions | None
    _tasks: set[asyncio.Task[Any]]
    _speech_handles: weakref.WeakValueDictionary[str, SpeechHandle]

    def __init__(
        self,
        *,
        stt: NotGivenOr[stt.STT | str] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | str] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | str] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        context: Context_T | None = None,
        thread: Thread | None = None,
        video_sampler: NotGivenOr[VoiceActivityVideoSampler | None] = NOT_GIVEN,
        conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
        stt_context_options: NotGivenOr[STTContextOptions | None] = NOT_GIVEN,
        **options: Any,
    ) -> None:
        """A voice session over the Rust core.

        Args:
            stt, vad, llm, tts: The providers, or a LiveKit Inference model string. ``vad``
                defaults to ``inference.VAD(model="silero")``; ``vad=None`` opts out.
            tools: Tools shared by every agent of the session.
            context, thread: Read by tools through ``RunContext``.
            video_sampler: How often camera frames reach the model; ``None`` sends every frame.
            turn_detection, endpointing_opts: The session's turn completion and endpointing
                policy, as ``update_options`` changes them.
            options: The session options the core resolves: ``turn_handling``,
                ``stt_context_options``, ``max_tool_steps``, ``use_tts_aligned_transcript``,
                ``tts_text_transforms``, ``min_consecutive_speech_delay``, ``expressive``,
                ``aec_warmup_duration``, ``ivr_detection``, ``user_away_timeout``,
                ``transcription_timeout``, ``session_close_transcript_timeout``.
        """
        super().__init__()
        self._adapter = SessionAdapter(self)
        initialize(
            self,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            tools=tools,
            context=context,
            thread=thread,
            video_sampler=video_sampler,
            conn_options=conn_options,
            turn_detection=turn_detection,
            endpointing_opts=endpointing_opts,
            stt_context_options=stt_context_options,
            **options,
        )

    # --- state ---------------------------------------------------------------------------

    @property
    def store(self) -> ContextStore:
        """The call's store for explicit ``load_context`` and ``save_context``: the thread's
        lane, or memory when the call has no thread or its lane did not load."""
        if self._store is None:
            raise RuntimeError("VoiceSession isn't running")
        return self._store

    @property
    def context(self) -> Context_T:
        """Application data and services supplied for this session."""
        return self._context

    @property
    def thread(self) -> Thread | None:
        """The call's supplied thread, or ``None`` without configured storage."""
        return self._thread

    @property
    def stt(self) -> stt.STT | None:
        return self._stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return self._llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._tts

    @property
    def vad(self) -> VAD | None:
        return self._vad

    @property
    def tools(self) -> list[llm.Tool | llm.Toolset]:
        return list(self._tools)

    @property
    def options(self) -> dict[str, Any]:
        """The configuration the core runs on, current the instant a change is sent."""
        return cast(dict[str, Any], self._owner.resolved_options(self._adapter._legs()))

    @property
    def conn_options(self) -> SessionConnectOptions:
        return self._conn_options

    @property
    def input(self) -> io.AgentInput:
        return self._input

    @property
    def output(self) -> io.AgentOutput:
        return self._output

    @property
    def history(self) -> ChatContext:
        """The call's conversation: a new ``ChatContext`` over the session's own item objects,
        which every event and run naming an item hand back too. Inside an event callback, and
        the tasks it starts, it is the history as of that event, without the items added after
        it; anywhere else it is the latest the session's events delivered. Before ``start()``
        it is empty. Editing the returned context does not change the session."""
        return ChatContext(cast(list[ChatItem], self._state.history))

    @property
    def _state(self) -> _SessionState:
        """The session as the event being handled left it, else as the last delivered one."""
        delivered = _delivered.get(None)
        if delivered is not None and delivered[0] is self:
            return delivered[1]
        return self._owner.state

    @property
    def amd(self) -> AMD | None:
        """The Answering Machine Detection (AMD) instance, or ``None`` if AMD is disabled."""
        return self._amd

    @property
    def agent_state(self) -> AgentState:
        return self._agent_state

    @property
    def user_state(self) -> UserState:
        return self._user_state

    @property
    def current_agent(self) -> Agent:
        agent = self._owner.current_agent
        if agent is None:
            raise RuntimeError("VoiceSession isn't running")
        return cast(Agent, agent)

    @property
    def _core(self) -> _NativeSession:
        return cast("_NativeSession", self._owner.session)

    @property
    def _running(self) -> _NativeSession:
        if self._core is None:
            raise RuntimeError("VoiceSession isn't running")
        return self._core

    # --- the pipeline ----------------------------------------------------------------------

    @property
    def default(self) -> DefaultPipeline:
        """The built-in steps, for a registered callback to wrap."""
        return DefaultPipeline(self)

    def llm_node(self, f: LLMNode[Context_T]) -> LLMNode[Context_T]:
        """Register the model request of every reply."""
        self._adapter._register("llm_node", f)
        return f

    def stt_node(self, f: STTNode[Context_T]) -> STTNode[Context_T]:
        """Register the recognition of the person's audio."""
        self._adapter._register("stt_node", f)
        return f

    def tts_node(self, f: TTSNode[Context_T]) -> TTSNode[Context_T]:
        """Register the synthesis of each spoken segment."""
        self._adapter._register("tts_node", f)
        return f

    def transcription_node(self, f: TranscriptionNode[Context_T]) -> TranscriptionNode[Context_T]:
        """Register the transcript of each spoken segment, before the transcription output."""
        self._adapter._register("transcription_node", f)
        return f

    def on_user_turn_completed(
        self, f: UserTurnCompletedHook[Context_T]
    ) -> UserTurnCompletedHook[Context_T]:
        """Register the hook run when the person's turn ends, before the reply; it edits the turn
        in place, or raises ``StopResponse`` to skip the reply."""
        self._adapter._register("on_user_turn_completed", f)
        return f

    def on_user_turn_exceeded(
        self, f: UserTurnExceededHook[Context_T]
    ) -> UserTurnExceededHook[Context_T]:
        """Register the hook run when the person spoke past the turn limit without a reply,
        in place of the default short reply."""
        self._adapter._register("on_user_turn_exceeded", f)
        return f

    # --- lifecycle -----------------------------------------------------------------------

    async def start(
        self,
        agent: Agent,
        *,
        room: NotGivenOr[rtc.Room | None] = NOT_GIVEN,
        room_options: NotGivenOr[RoomOptions] = NOT_GIVEN,
        chat_ctx: llm.ChatContext | None = None,
    ) -> None:
        """Start the session on ``room`` (the job's room by default, an offline room without
        one) with ``agent`` driving."""
        from opentelemetry import context as otel_context

        from .._generated.session_owner import SessionJobInfo
        from ..job import get_job_context
        from ..telemetry import _context
        from ._session_bindings import SessionHost

        host = SessionHost(self, agent, room, room_options, chat_ctx)
        ctx = get_job_context(required=False)
        job = (
            SessionJobInfo(
                id=ctx.job.id,
                dispatch_id=ctx.job.dispatch_id or None,
                worker_id=ctx.worker_id or None,
                agent_id=ctx.job.state.agent_id or None,
            )
            if ctx is not None
            else None
        )
        await self._owner.start(host.storage, host, job=job)
        # the session's span is the caller's current span from here: the entrypoint's own spans
        # and the job's RPC handlers nest under it. It stays attached for the rest of the
        # entrypoint, which outlives the session's start; the job's context ends with the job.
        if (span := self._owner.span) is not None:
            self._root_span_context = _context(span)
            otel_context.attach(self._root_span_context)

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        await self._adapter._start_ivr_detection(transcript)

    async def aclose(self) -> None:
        await self._aclose(CloseReason.USER_INITIATED)

    async def _aclose(self, reason: CloseReason) -> None:
        await self._owner.aclose(reason.value)

    def shutdown(self, *, drain: bool = True) -> None:
        self._owner.shutdown(CloseReason.USER_INITIATED.value, drain=drain)

    async def __aenter__(self) -> VoiceSession[Context_T]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- commands ------------------------------------------------------------------------

    def _run(self, *, user_input: str) -> _VoiceRun:
        """Reply to typed ``user_input`` and record what the reply did (the remote session's text
        input and the voice conformance tests)."""
        run = _VoiceRun(session=self)
        self._running.run(user_input=user_input, origin=_origin(), host=run)
        return run

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        return self._adapter.say(
            text,
            audio=audio,
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
        )

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> SpeechHandle:
        """Reply now. ``tools`` narrows the offered tools to these names for this reply."""
        return self._adapter.generate_reply(
            user_input=user_input,
            instructions=instructions,
            tool_choice=tool_choice,
            tools=tools,
            allow_interruptions=allow_interruptions,
            chat_ctx=chat_ctx,
        )

    @contextlib.asynccontextmanager
    async def _claim_user_turn(self) -> AsyncIterator[None]:
        """A programmatic user turn: the user state is pinned to ``speaking`` and
        ``wait_for_idle`` held while inside."""
        self._running.claim_user_turn()
        try:
            yield
        finally:
            if self._core is not None:
                self._core.release_user_turn()

    def update_agent(self, agent: Agent, *, chat_ctx: ChatContext | None = None) -> None:
        """Hand the conversation to ``agent``; the session and its history stay. ``chat_ctx`` is
        the context ``agent`` works on instead of the conversation carried forward."""
        self._adapter.update_agent(agent, chat_ctx=chat_ctx)

    def _agent_chat_ctx(self) -> ChatContext:
        """The driving agent's own context, as the core keeps it."""
        return ChatContext(list(self._running.agent_chat_ctx()))

    def update_options(
        self,
        *,
        stt: NotGivenOr[STT | str | None] = NOT_GIVEN,
        tts: NotGivenOr[TTS | str | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
        interruption: NotGivenOr[InterruptionOptions] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = NOT_GIVEN,
        expressive: NotGivenOr[bool | ExpressiveOptions] = NOT_GIVEN,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Change only the supplied settings of the session. A replaced STT, TTS or VAD is
        installed in place, without restarting the call or interrupting speech; a setting an
        ``override_options`` layer holds takes the new value once the layer is released.
        ``interruption`` merges over the session's, except its ``mode``, which is set at
        construction."""
        self._adapter.configure(
            stt=stt,
            tts=tts,
            vad=vad,
            turn_detection=turn_detection,
            endpointing_opts=endpointing_opts,
            interruption=interruption,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            tts_text_transforms=tts_text_transforms,
            expressive=expressive,
            keyterms=keyterms,
        )

    @contextlib.asynccontextmanager
    async def override_options(
        self,
        *,
        stt: NotGivenOr[STT | str | None] = NOT_GIVEN,
        tts: NotGivenOr[TTS | str | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
        interruption: NotGivenOr[InterruptionOptions] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = NOT_GIVEN,
    ) -> AsyncIterator[None]:
        """Layer settings over the session's for the block: each takes its value from the
        topmost layer that sets it, else from the base, and the layer is released when the
        block ends, however it ends."""
        async with self._adapter.override(
            stt=stt,
            tts=tts,
            vad=vad,
            turn_detection=turn_detection,
            endpointing_opts=endpointing_opts,
            interruption=interruption,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            tts_text_transforms=tts_text_transforms,
        ):
            yield

    # --- frames --------------------------------------------------------------------------

    def _frame(self, agent: Agent, *, task: bool = False) -> dict[str, Any]:
        return self._adapter._frame(agent, task=task)

    # --- the core's events ---------------------------------------------------------------

    def _core_event(self, wire: HostWire, state: _SessionState) -> None:
        from ._session_callbacks import dispatch

        # the event's callbacks, and the tasks they start, read the session as of the event
        token = _delivered.set((self, state))
        try:
            dispatch(self, wire)
        finally:
            _delivered.reset(token)

    def _core_history_items(self, items: list[ChatItem]) -> list[ChatItem]:
        """The objects the history holds for ``items``: these, as the binding made them."""
        return items

    def _core_error_message(self, wire: SessionErrorWire) -> str:
        from ._session_callbacks import _core_error_message

        return _core_error_message(self, wire)

    @overload
    def on(self, event: EventTypes, callback: _CallbackT) -> _CallbackT: ...

    @overload
    def on(
        self, event: EventTypes, callback: None = None
    ) -> Callable[[_CallbackT], _CallbackT]: ...

    def on(
        self, event: EventTypes, callback: _CallbackT | None = None
    ) -> _CallbackT | Callable[[_CallbackT], _CallbackT]:
        if event == "metrics_collected":
            raise ValueError(
                "VoiceSession has no 'metrics_collected' event in 2.0: use "
                "session.usage for usage, ChatMessage.metrics for per-turn latency, "
                "or a provider's own 'metrics_collected' (llm, stt, tts, vad)"
            )
        return super().on(event, callback)

    def _report_event_snapshots(self) -> list[dict[str, Any]]:
        from ._session_callbacks import report_snapshots

        return report_snapshots(self)

    @property
    def _recorded_events(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(event) for event in self._report_event_snapshots()]

    def emit(self, event: EventTypes, arg: AgentEvent) -> None:  # type: ignore[override]
        from ._session_callbacks import record_snapshot

        record_snapshot(self, arg)
        super().emit(event, arg)


class VoiceTask(Generic[Output_T, Context_T]):
    """One bounded live interaction run by ``agent`` on the active ``VoiceSession``.

    Like ``TextSession``, a ``VoiceTask`` is not an ``Agent``: it runs one. Await ``run()``
    inside a tool of the agent holding the call: ``agent`` takes the floor (its ``on_enter``,
    or a first reply without one) and ``run()`` returns once one of its tools returns
    ``Completed(value)``, validated against its ``output_type``, or raises ``VoiceTaskFailed``
    once one returns ``Failed(reason)``. The waiting agent then gets the floor back without its
    ``on_exit`` or ``on_enter``. Cancelling the await, or closing the session, ends the task.
    """

    def __init__(self, *, agent: Agent[Context_T, Output_T]) -> None:
        self._agent = agent

    @property
    def agent(self) -> Agent[Context_T, Output_T]:
        return self._agent

    async def run(self) -> Output_T:
        binding = _live_binding("VoiceTask.run()")
        session, agent = binding.session, self._agent
        reply = session._running.run_voice_task(
            UpdateAgentArgs(**session._frame(agent, task=True)),
            origin=binding.origin,
            opens_with_reply=not agent._has_on_enter(),
        )
        output = cast(Output_T, await reply)
        return output


VoiceTask.__module__ = VoiceTaskFailed.__module__ = "livekit.agents.voice.agent_session"


__all__ = [
    "VoiceSession",
    "ExpressiveOptions",
    "NonverbalOptions",
    "SessionConnectOptions",
    "SpeechSteeringOptions",
    "VoiceActivityVideoSampler",
    "KeytermDetectionOptions",
    "STTContextOptions",
    "TextTransforms",
]
