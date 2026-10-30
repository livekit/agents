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
import dataclasses
import inspect
import time
import weakref
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

import numpy as np
from opentelemetry import trace
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypedDict, TypeVar

from livekit import rtc

from .. import _bindings, _core, inference, llm, stt, tts
from .._core_view import (
    TurnDetector,
    chat_request,
    realtime_model,
    tool_specs,
)
from .._generated.agent_run import FunctionToolInfo
from .._generated.amd import AMDPredictionEvent
from .._generated.expressive import ExpressiveOptions, NonverbalOptions, SpeechSteeringOptions
from .._generated.keyterms import KeytermDetectionOptions, STTContextOptions
from .._generated.session_dispatch import IGNORED_EVENTS, MIRRORED_STATES, PUBLIC_EVENTS
from .._generated.session_options import SessionKwargs
from .._generated.session_wire import (
    CloseWire,
    DeleteRoomRequested,
    HistoryReloadedWire,
    HostWire,
    ItemAddedWire,
    MetricsCollectedWire,
    OverlappingSpeechWire,
    RunItemAddedWire,
    RunRefusedWire,
    RunSettledWire,
    SessionErrorWire,
    SpanEndedWire,
    SpanStartedWire,
    SpeechCreatedWire,
    SpeechDoneWire,
    TextInputWire,
    ToolCallStartedWire,
    ToolExecutionUpdatedWire,
)
from .._generated.usage import Usage
from .._generated.vad import VAD
from .._generated.voice_session import UpdateAgentArgs, VoiceSessionArgs
from ..inference.interruption import OverlappingSpeechEvent
from ..job import get_job_context
from ..llm import (
    LLM,
    ChatContext,
    RealtimeModel,
    StopResponse,
    ToolError,
    function_tool,
)
from ..log import logger
from ..stt import STT
from ..telemetry.metrics import collect_usage, record_span_metric
from ..telemetry.traces import capture_content_enabled, tracer
from ..tts import TTS
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    APIConnectOptions,
    FlushSentinel,
    NotGivenOr,
    TimedString,
)
from ..utils.misc import is_dev_mode, is_given, is_hosted, shortuuid
from . import io
from .agent import (
    Agent,
    ModelSettings,
    Output_T,
    _BindingContextVar,
    _VoiceBinding,
)
from .agent_run import run_error, tool_infos
from .events import (
    AgentEvent,
    AgentState,
    CloseEvent,
    CloseReason,
    Context_T,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    FunctionToolsExecutedEvent,
    RunContext,
    SpeechCreatedEvent,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
    ToolUpdate,
    UserState,
    UserTurnExceededEvent,
)
from .remote_session import ConsoleCall, RemoteSessionAccess
from .room_io import (
    AudioInputOptions,
    Room as _RtcRoom,
    RoomOptions,
    TextInputCallback,
    TextInputEvent,
)
from .run_result import _VoiceRun
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..storage.backends import Storage
    from ..storage.context_store import ContextStore
    from ..storage.resolve import Thread
    from .amd import AMD
    from .turn import EndpointingOptions, InterruptionOptions, TurnDetectionMode

__all__ = [
    "VoiceSession",
    "ExpressiveOptions",
    "NonverbalOptions",
    "RecordingOptions",
    "SessionConnectOptions",
    "SpeechSteeringOptions",
    "VoiceActivityVideoSampler",
    "KeytermDetectionOptions",
    "STTContextOptions",
    "TextTransforms",
]

TextTransforms = Literal["filter_markdown", "filter_emoji"]
"""A transform the core applies to the text it sends to the TTS."""


# the hook running in this context, for `generate_reply(on_enter=)`
_hook_name: contextvars.ContextVar[str] = contextvars.ContextVar("lk_hook_name")

# on_exit and the other callbacks are awaited by the core this long before it moves on;
# on_user_turn_completed is awaited until it returns
_HOOK_BOUND = 15.0

# the calls of the `tts_node` running in this context, which `session.default.tts_node` reports
# its provider's to (`_core.tts_node(calls=)`)
_NodeCallsVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_node_calls")

# the session's pipeline callbacks (voice-session.md "Customizing the pipeline"): each receives
# the call's `RunContext`, whose `agent` is the agent holding the call
_C = TypeVar("_C")
_Item = TypeVar("_Item")
_Node = AsyncIterable[_Item] | Coroutine[Any, Any, AsyncIterable[_Item] | None]
LLMNode = Callable[
    [RunContext[_C], ChatContext, list[llm.Tool], ModelSettings],
    AsyncIterable[llm.ChatChunk | str | FlushSentinel]
    | Coroutine[Any, Any, AsyncIterable[llm.ChatChunk | str | FlushSentinel] | str | None],
]
STTNode = Callable[
    [RunContext[_C], AsyncIterable[rtc.AudioFrame], ModelSettings], _Node[stt.SpeechEvent]
]
TTSNode = Callable[[RunContext[_C], AsyncIterable[str], ModelSettings], _Node[rtc.AudioFrame]]
TranscriptionNode = Callable[
    [RunContext[_C], AsyncIterable[str | TimedString], ModelSettings], _Node[str | TimedString]
]
RealtimeAudioOutputNode = Callable[
    [RunContext[_C], AsyncIterable[rtc.AudioFrame], ModelSettings], _Node[rtc.AudioFrame]
]
UserTurnCompletedHook = Callable[
    [RunContext[_C], ChatContext, llm.ChatMessage], Coroutine[Any, Any, None]
]
UserTurnExceededHook = Callable[[RunContext[_C], UserTurnExceededEvent], Coroutine[Any, Any, None]]

# the session option keywords (`SessionKwargs` less `conn_options`, a parameter of its own) and
# the required ones' defaults; the core resolves them (`_core.SessionOptions`). An unsaid
# `NotRequired` key stays absent.
_OPTION_KEYS = (SessionKwargs.__required_keys__ | SessionKwargs.__optional_keys__) - {
    "conn_options"
}
_OPTION_DEFAULTS: dict[str, Any] = {
    "stt_context_options": None,
    "max_tool_steps": 3,
    "min_consecutive_speech_delay": 0.0,
    "expressive": False,
    "ivr_detection": False,
    "user_away_timeout": 15.0,
    "transcription_timeout": None,
    "session_close_transcript_timeout": 2.0,
}


class RecordingOptions(TypedDict, total=False):
    """Which recording features are active; keys default to ``True`` when not specified."""

    audio: bool
    traces: bool
    logs: bool
    transcript: bool
    redaction: bool


@dataclass
class SessionConnectOptions:
    """Each leg's connection policy; ``None`` keeps the provider's own (models.md), else the
    defaults."""

    stt_conn_options: APIConnectOptions | None = None
    llm_conn_options: APIConnectOptions | None = None
    tts_conn_options: APIConnectOptions | None = None
    max_unrecoverable_errors: int = 3
    """Maximum number of consecutive unrecoverable errors from stt, llm, or tts."""


class VoiceActivityVideoSampler:
    """Sample the camera at ``speaking_fps`` while the user speaks, ``silent_fps`` otherwise."""

    def __init__(self, *, speaking_fps: float = 1.0, silent_fps: float = 0.3) -> None:
        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps


_CallbackT = TypeVar("_CallbackT", bound=Callable[..., object])


class VoiceSession(rtc.EventEmitter[EventTypes], Generic[Context_T]):
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
        if unknown := options.keys() - _OPTION_KEYS:
            raise TypeError(f"unexpected VoiceSession options: {sorted(unknown)}")
        listening = dict(options.get("turn_handling") or {})
        if is_given(turn_detection):
            listening["turn_detection"] = turn_detection
        if is_given(endpointing_opts):
            listening["endpointing"] = endpointing_opts
        if listening:
            options["turn_handling"] = listening
        self._conn_options = conn_options or SessionConnectOptions()
        self._options = _core.SessionOptions(
            {**_OPTION_DEFAULTS, **_given(**options), "conn_options": self._conn_options}
        )
        stt_context = options.get("stt_context_options", NOT_GIVEN)
        self._stt_context = stt_context if is_given(stt_context) else None

        self._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt or None
        self._llm = inference.llm_from_model_string(llm) if isinstance(llm, str) else llm or None
        self._tts = inference.TTS.from_model_string(tts) if isinstance(tts, str) else tts or None
        self._using_default_vad = not is_given(vad)
        self._vad = inference.VAD(model="silero") if not is_given(vad) else vad
        self._tools = tools if is_given(tools) else []
        turn_handling = options.get("turn_handling") or {}
        detection = turn_handling.get("turn_detection", NOT_GIVEN)
        if not is_given(detection):
            # a realtime model keeps its server's turn detection unless one is named
            realtime = isinstance(self._llm, RealtimeModel)
            detection = None if realtime else inference.TurnDetector()
        # the gateway detector's stream is the core's own detector; a plugin's is seen through a view
        self._turn_detector = (
            None
            if isinstance(detection, str | None)
            else detection.stream()
            if isinstance(detection, inference.TurnDetector)
            else TurnDetector(detection)
        )
        self._turn_detection_mode = detection if isinstance(detection, str) else None
        self._interruption_mode = (turn_handling.get("interruption") or {}).get("mode")
        self._video_sampler = (
            video_sampler if is_given(video_sampler) else VoiceActivityVideoSampler()
        )

        # `ctx.context` on this call; `None` when the session has no application context
        self._context = cast("Context_T", context)
        self._thread = thread
        # `ctx.execution_id` on this call: one conversational task across its turns
        self._execution_id = shortuuid("execution_")
        self._voice_store: Any = None
        self._store: ContextStore | None = None
        # the in-memory storage of a call with no loaded lane
        self._memory: Storage | None = None

        self._core: _core.VoiceSession = None  # type: ignore[assignment]
        self._room: rtc.Room | None = None
        self._amd: AMD | None = None
        # `override_options`: what `update_options` wrote under a layer, and the layers in order
        self._base_options: dict[str, Any] = {}
        self._option_layers: list[dict[str, Any]] = []
        self._agent: Agent | None = None
        self._chat_ctx = ChatContext.empty()
        self._agent_state: AgentState = "initializing"
        self._user_state: UserState = "listening"
        self._input = io.AgentInput(self)
        self._output = io.AgentOutput(self)
        # what a job's session report reads
        self._started_at: float | None = None
        self._event_snapshots: deque[AgentEvent] = deque()
        self._event_retention = _core._core_EventRetention(capture_content_enabled())
        self._recording_options: RecordingOptions = {
            "audio": False,
            "traces": False,
            "logs": False,
            "transcript": False,
            "redaction": False,
        }
        # the core's span tree, materialised here: the open spans by the core's id, and the
        # root's context the job's telemetry parents its own spans under
        self._spans: dict[str, trace.Span] = {}
        self._session_span: trace.Span | None = None
        self._root_span_context: Any = None
        self._text_input_cb: TextInputCallback | None = None
        # the session's audio recording, written by the core; finished once it closes
        self._recorder: _core.SessionRecorder | None = None
        # the pipeline callbacks registered on this session, by step name
        self._steps: dict[str, Callable[..., Any]] = {}
        self._recording_path: Path | None = None
        self._recording_finished: asyncio.Future[None] | None = None
        self._audio_input_options: AudioInputOptions | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        # the handles the core announced, for their done callbacks (``speech_done``)
        self._speech_handles: weakref.WeakValueDictionary[str, SpeechHandle] = (
            weakref.WeakValueDictionary()
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
        """The call's thread: the one supplied, else an in-memory one once started."""
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
        if self._core is not None:
            return self._core.resolved_options()  # type: ignore[no-any-return]
        return self._options.resolved(self._legs())  # type: ignore[no-any-return]

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
        return self._chat_ctx

    @property
    def keyterms(self) -> list[str]:
        return self._core.keyterms() or [] if self._core is not None else []

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
        if self._agent is None:
            raise RuntimeError("VoiceSession isn't running")
        return self._agent

    @property
    def usage(self) -> Usage:
        """The latest immutable snapshot of the session's usage: empty before ``start()``,
        settled once ``aclose()`` returned (usage.md "Voice sessions")."""
        return self._core.usage if self._core is not None else Usage()

    @property
    def _running(self) -> _core.VoiceSession:
        if self._core is None:
            raise RuntimeError("VoiceSession isn't running")
        return self._core

    # --- the pipeline ----------------------------------------------------------------------

    def _register(self, step: str, f: Callable[..., Any]) -> None:
        if self._core is not None:
            raise RuntimeError(f"register {step} before the session starts")
        if step in self._steps:
            raise RuntimeError(f"this session already has a {step}")
        self._steps[step] = f

    @property
    def default(self) -> DefaultPipeline:
        """The built-in steps, for a registered callback to wrap."""
        return DefaultPipeline(self)

    def llm_node(self, f: LLMNode[Context_T]) -> LLMNode[Context_T]:
        """Register the model request of every reply."""
        self._register("llm_node", f)
        return f

    def stt_node(self, f: STTNode[Context_T]) -> STTNode[Context_T]:
        """Register the recognition of the person's audio."""
        self._register("stt_node", f)
        return f

    def tts_node(self, f: TTSNode[Context_T]) -> TTSNode[Context_T]:
        """Register the synthesis of each spoken segment."""
        self._register("tts_node", f)
        return f

    def transcription_node(self, f: TranscriptionNode[Context_T]) -> TranscriptionNode[Context_T]:
        """Register the transcript of each spoken segment, before the transcription output."""
        self._register("transcription_node", f)
        return f

    def realtime_audio_output_node(
        self, f: RealtimeAudioOutputNode[Context_T]
    ) -> RealtimeAudioOutputNode[Context_T]:
        """Register a realtime model's audio, before playout."""
        self._register("realtime_audio_output_node", f)
        return f

    def on_user_turn_completed(
        self, f: UserTurnCompletedHook[Context_T]
    ) -> UserTurnCompletedHook[Context_T]:
        """Register the hook run when the person's turn ends, before the reply; it edits the turn
        in place, or raises ``StopResponse`` to skip the reply."""
        self._register("on_user_turn_completed", f)
        return f

    def on_user_turn_exceeded(
        self, f: UserTurnExceededHook[Context_T]
    ) -> UserTurnExceededHook[Context_T]:
        """Register the hook run when the person spoke past the turn limit without a reply,
        in place of the default short reply."""
        self._register("on_user_turn_exceeded", f)
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
        if self._core is not None:
            return
        console = ConsoleCall.claim()
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            self._event_retention.claim(job_ctx.job.id)
        if console is not None:
            room = None
        elif not is_given(room):
            room = job_ctx.room if job_ctx is not None else None
        if chat_ctx is not None:
            self._chat_ctx = chat_ctx.copy()
        options = room_options if is_given(room_options) else RoomOptions()
        self._audio_input_options = options.get_audio_input_options()
        text_input = options.get_text_input_options()
        self._text_input_cb = text_input.text_input_cb if text_input is not None else None
        # the job's session (on its room, or the console's) records and reports for the job and
        # closes with it; the primary one, and the console's, serve the remote session protocol
        in_job = job_ctx is not None and (console is not None or room is job_ctx.room)
        hosting = console is not None
        if in_job:
            assert job_ctx is not None
            plan = _core.plan_session_start(
                job_default=job_ctx.job.enable_recording,
                another_session_is_primary=not job_ctx._claim_session(self),
            )
            self._recording_options = cast(RecordingOptions, dataclasses.asdict(plan.recording))
            hosting = hosting or plan.hosting
            job_ctx.init_recording(self._recording_options)

        from ..storage.context_store import ContextStore

        if self._thread is not None:
            from ..storage.errors import StorageError

            try:
                self._voice_store = await self._thread._storage.threads._core_voice_store(
                    self._thread.id,
                    self._thread._agent_name,
                    initial_history=chat_ctx.items if chat_ctx is not None else None,
                )
                self._store = ContextStore(
                    activation=self._voice_store.activation,
                    scope_id=f"root:{self._thread._agent_name}",
                )
                saved = self._voice_store.initial_history
                if saved is not None:
                    self._chat_ctx = ChatContext(list(saved))
            except StorageError as error:
                logger.warning("the voice lane could not be loaded", extra={"error": str(error)})
        if self._store is None:
            # no lane to save into: the call's context fields load and save in memory, and a
            # call with no thread gets an in-memory one, as a text run does
            from ..storage.backends import Storage
            from ..storage.resolve import Thread

            self._memory = Storage({}, None)
            info = await self._memory.threads.create_thread()
            if self._thread is None:
                self._thread = Thread(self._memory, info.thread_id, agent_name=agent.name)
            self._store = ContextStore(
                activation=await self._memory.threads._core_context_store(
                    info.thread_id, agent.name
                )
            )

        hooks = _AgentHooks(self)
        frame = self._frame(agent)
        await hooks._acquire(self._tools)
        frame["tools"] = await hooks._core_setup_tools(frame["tools_host"])
        model = frame.get("llm", self._llm)
        # an OpenAI-protocol model or a fallback chain runs in the core; any other one is a host
        # realtime model
        rt = (
            realtime_model(model, lambda: self._pool(self.current_agent))
            if isinstance(model, llm.RealtimeModel)
            else None
        )
        # a plugin's model is a host object the core adapts, any other the core's own
        realtime = _core.HostRealtimeModel(rt) if hasattr(rt, "_core_session") else rt
        sampler = self._video_sampler
        keyterm_llm = _detection_llm(self._stt_context)
        self._room = room
        try:
            args = VoiceSessionArgs(
                instructions=frame["instructions"],
                dynamic_instructions=frame["dynamic_instructions"],
                agent=frame["agent"],
                agent_host=frame["agent_host"],
                chat_ctx=self._chat_ctx.items,
                storage=self._voice_store,
                stt=_bindings.core_leg(frame.get("stt", self._stt)),
                vad=_bindings.core_leg(self._vad),
                llm=None if isinstance(model, llm.RealtimeModel) else _bindings.core_leg(model),
                realtime=realtime,
                tts=_bindings.core_leg(frame.get("tts", self._tts)),
                host_audio_input=self._input.audio,
                host_video_input=self._input.video,
                host_audio_output=self._output.audio,
                host_transcription_output=self._output.transcription,
                on_event=self,
                hooks=hooks,
                tools=frame["tools"],
                tools_host=frame["tools_host"],
                voice_agent=frame["voice_agent"],
                room_io_config=room_options._to_core() if is_given(room_options) else None,
                video_sampler=(sampler.speaking_fps, sampler.silent_fps) if sampler else None,
                using_default_vad=self._using_default_vad,
                turn_detector=self._turn_detector,
                hook_bound=_HOOK_BOUND,
                max_unrecoverable_errors=self._conn_options.max_unrecoverable_errors,
                interruption_detector=self._interruption_detector(agent, model),
                stt_context=self._stt_context,
                keyterm_llm=_bindings.core_leg(keyterm_llm) if keyterm_llm is not None else None,
                keyterm_tools=[_record_keyterms] if keyterm_llm is not None else None,
                **{node: frame[node] for node in _NODES if node in frame},
                remote_session=RemoteSessionAccess(self) if hosting else None,
                console=console,
                # the core connects the job's room beside its opening
                job=job_ctx if in_job and console is None else None,
            )
            room_session = _core._core_create_room_session(
                _RtcRoom.of(room) if room is not None else None
            )
            self._core = _core.VoiceSession(room_session, args, self._options)
        except BaseException:
            # a core that never started never closes the toolsets
            await hooks._core_close_toolsets(None)
            raise
        self._agent = agent
        self._started_at = time.time()
        if in_job:
            assert job_ctx is not None
            if self._recording_options["audio"] or (console is not None and ConsoleCall.record):
                self._recording_path = job_ctx.session_directory / "audio.ogg"
                self._recorder = _core.SessionRecorder(str(self._recording_path))
                await self._recorder.ready()
                self._core.attach_recorder(self._recorder)
            job_ctx.add_shutdown_callback(lambda: self._aclose(CloseReason.JOB_SHUTDOWN))
        await self._core.start()
        if self.options["ivr_detection"]:
            await self._start_ivr_detection()

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        """Start IVR detection: the DTMF tool joins the session's tools and the core's navigator
        runs, counting ``transcript`` (AMD's greeting) as heard."""
        from ..beta import send_dtmf_events

        if send_dtmf_events not in self._tools:
            self._tools.append(send_dtmf_events)
        self._running.ivr_start(tools=tool_specs([send_dtmf_events]), tools_host=[send_dtmf_events])
        if transcript is not None:
            self._running.ivr_final(transcript)

    def _interruption_detector(
        self, agent: Agent, model: Any
    ) -> inference.AdaptiveInterruptionDetector | None:
        """The adaptive interruption detector, when the reference's gate builds one."""
        # the developer's STT: a fallback chain's core leg reports its capabilities as a dict
        stt_caps = getattr(self._stt, "capabilities", None)
        has_vad = self._vad is not None
        realtime = isinstance(model, RealtimeModel)
        verdict = _core.resolve_interruption_detector(
            stt_aligned_transcript=bool(stt_caps.aligned_transcript) if stt_caps else None,
            stt_streaming=stt_caps.streaming if stt_caps else None,
            realtime=realtime,
            rt_server_turn_detection=realtime
            and _core.rt_server_turn_detection(
                caps_turn_detection=model.capabilities.turn_detection,
                can_disable_turn_detection=model.capabilities.can_disable_turn_detection,
                vad=has_vad,
                turn_detection=self._turn_detection_mode,
                streaming_detector=self._turn_detector is not None,
                interruption_given=bool(self._interruption_mode),
            ),
            has_vad=has_vad,
            turn_detection=self._turn_detection_mode,
            allow_interruptions=self.options["interruption"]["enabled"],
            agent_mode=None,
            session_mode=self._interruption_mode,
            hosted=is_hosted(),
            dev_mode=is_dev_mode(),
        )
        if verdict["message"]:
            log = logger.warning if verdict["level"] == "warning" else logger.info
            log(verdict["message"])
        if not verdict["build"]:
            return None
        try:
            return inference.AdaptiveInterruptionDetector()
        except ValueError as e:
            logger.warning("failed to create AdaptiveInterruptionDetector", extra={"error": str(e)})
            return None

    async def aclose(self) -> None:
        await self._aclose(CloseReason.USER_INITIATED)

    async def _aclose(self, reason: CloseReason) -> None:
        if self._core is not None:
            await self._core.aclose(reason=reason.value, drain=False)
        if self._voice_store is not None:
            from ..storage.errors import StorageError

            try:
                await self._voice_store.aclose()
            except StorageError as error:
                logger.warning("the voice store could not close", extra={"error": str(error)})
        if self._memory is not None:
            assert self._store is not None
            self._store._close()
            await self._memory.aclose()
        if self._recording_finished is None and self._recorder is not None:
            self._recording_finished = asyncio.ensure_future(self._recorder.finish())
        if self._recording_finished is not None:
            await asyncio.wait((self._recording_finished,))
            if not self._recording_finished.cancelled():
                # Keep the failure on the future for the job's recording reporter.
                self._recording_finished.exception()

    async def drain(self) -> None:
        await self._running.drain()

    def shutdown(self, *, drain: bool = True) -> None:
        if self._core is not None:
            self._core.shutdown(drain=drain)

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
        kwargs = {
            "allow_interruptions": allow_interruptions if is_given(allow_interruptions) else None,
            "add_to_chat_ctx": add_to_chat_ctx,
            "origin": _origin(),
        }
        if is_given(audio):
            assert isinstance(text, str), "say(audio=) takes the words it speaks as a str"
            core = self._running.say_audio(text, audio, **kwargs)
        elif isinstance(text, str):
            core = self._running.say(text, **kwargs)
        else:
            core = self._running.say_stream(text, **kwargs)
        return SpeechHandle(core)

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
        toolset: dict[str, Any] = {}
        if is_given(tools):
            pool = self._pool(self.current_agent)
            specs = [
                s for s in tool_specs(pool) if s["name"] in tools or s["provider_tool"] is not None
            ]
            toolset = {
                "tools": specs,
                "tools_host": pool,
                "voice_agent": self._voice_agent(self.current_agent),
            }
        core = self._running.generate_reply(
            user_input=user_input if is_given(user_input) else None,
            instructions=instructions if is_given(instructions) else None,
            allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            chat_ctx=chat_ctx.items if is_given(chat_ctx) else None,
            on_enter=_origin() == "hook" and _hook_name.get(None) == "on_enter",
            origin=_origin(),
            **toolset,
        )
        return SpeechHandle(core)

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        return asyncio.ensure_future(self._running.interrupt(force=force))

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

    def clear_user_turn(self) -> None:
        self._running.clear_user_turn()

    def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> asyncio.Future[str]:
        """Commit the user turn now; the future resolves with its transcript."""
        return asyncio.ensure_future(
            self._running.commit_user_turn(
                transcript_timeout=transcript_timeout,
                stt_flush_duration=stt_flush_duration,
                skip_reply=skip_reply,
            )
        )

    async def wait_for_idle(self) -> None:
        await self._running.wait_for_idle()

    def update_agent(self, agent: Agent, *, chat_ctx: ChatContext | None = None) -> None:
        """Hand the conversation to ``agent``; the session and its history stay. ``chat_ctx`` is
        the context ``agent`` works on instead of the conversation carried forward."""
        self._agent = agent
        if self._core is None:
            if chat_ctx is not None:
                self._chat_ctx = chat_ctx.copy()
            return
        frame = self._frame(agent)
        if chat_ctx is not None:
            frame["chat_ctx"] = chat_ctx.items
        self._core.update_agent(UpdateAgentArgs(**frame))

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
        update = _given(
            stt=_stt_model(stt),
            tts=_tts_model(tts),
            vad=vad,
            turn_detection=turn_detection,
            endpointing_opts=endpointing_opts,
            interruption=interruption,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            tts_text_transforms=tts_text_transforms,
            expressive=expressive,
            keyterms=keyterms,
        )
        update = self._completed(
            update,
            lambda k: self._base_options[k] if k in self._base_options else self._current_option(k),
        )
        self._base_options.update(update)
        self._send_options({k: v for k, v in update.items() if not self._layered(k)})

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
        layer = _given(
            stt=_stt_model(stt),
            tts=_tts_model(tts),
            vad=vad,
            turn_detection=turn_detection,
            endpointing_opts=endpointing_opts,
            interruption=interruption,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            tts_text_transforms=tts_text_transforms,
        )
        layer = self._completed(
            layer,
            lambda k: (
                self._effective_option(k) if k in self._base_options else self._current_option(k)
            ),
        )
        for key in layer:
            if not self._layered(key):
                self._base_options.setdefault(key, self._current_option(key))
        self._option_layers.append(layer)
        self._send_options(layer)
        try:
            yield
        finally:
            del self._option_layers[
                next(i for i, held in enumerate(self._option_layers) if held is layer)
            ]
            self._send_options({k: self._effective_option(k) for k in layer})

    @staticmethod
    def _completed(options: dict[str, Any], under: Callable[[str], Any]) -> dict[str, Any]:
        """``options`` with each group (endpointing, interruption) filled in from the value it
        lands on, so restoring a base or a lower layer restores every key of the group."""
        if "mode" in options.get("interruption", {}):
            raise ValueError(
                "interruption takes no mode here: the interruption mode is set with "
                "VoiceSession(turn_handling=)"
            )
        return {k: {**under(k), **v} if k in _GROUPS else v for k, v in options.items()}

    def _layered(self, key: str) -> bool:
        return any(key in layer for layer in self._option_layers)

    def _effective_option(self, key: str) -> Any:
        for layer in reversed(self._option_layers):
            if key in layer:
                return layer[key]
        return self._base_options[key]

    def _current_option(self, key: str) -> Any:
        """The value ``key`` has in the core, as ``update_options`` takes it: current until this
        session sends ``key``, which then keeps it in ``_base_options`` and the layers."""
        if key in _MODELS:
            return getattr(self, f"_{key}")
        resolved = self.options
        if key == "turn_detection":
            return resolved["turn_detection_mode"]
        if key == "endpointing_opts":
            endpointing = resolved["endpointing"]
            return {k: endpointing[k] for k in ("mode", "min_delay", "max_delay", "alpha")}
        if key == "interruption":
            # the mode is the construction's; update_options takes no mode
            return {k: v for k, v in resolved["interruption"].items() if k != "mode"}
        return resolved[key]

    def _send_options(self, update: dict[str, Any]) -> None:
        models = {k: v for k, v in update.items() if k in _MODELS}
        turn = {k: v for k, v in update.items() if k not in _MODELS}
        for leg, model in models.items():
            setattr(self, f"_{leg}", model)
        if "vad" in models:
            self._using_default_vad = False
        if models and self._core is not None:
            # a replaced leg goes under the node the running agent overrides, as at its install
            nodes = self._nodes(self.current_agent)
            self._core.update_models(
                **{leg: _bindings.core_leg(model) for leg, model in models.items()},
                **{f"{leg}_node": nodes[f"{leg}_node"] for leg in models if f"{leg}_node" in nodes},
            )
        if not turn:
            return
        if self._core is None:
            self._options.update(turn)
        else:
            self._core.update_turn_options(turn)

    # --- frames --------------------------------------------------------------------------

    def _pool(self, agent: Agent) -> list[llm.Tool | llm.Toolset]:
        return [*agent.tools, *self._tools]

    def _voice_agent(self, agent: Agent, *, task: bool = False) -> _core.VoiceAgent:
        """``agent`` as its callbacks run on this session: its definition over its tools and the
        session's it lacks, its ``AgentCallbacks`` and the context they run in, whose binding is this
        session's."""
        from .agent_host import _voice_tools

        captured = contextvars.copy_context()
        binding = _VoiceBinding(self, agent, task=task)
        captured.run(_BindingContextVar.set, binding)
        return _core.VoiceAgent(
            agent._definition(_voice_tools(binding)),
            agent._host,
            (asyncio.get_running_loop(), captured),
        )

    def _frame(self, agent: Agent, *, task: bool = False) -> dict[str, Any]:
        """``agent`` as the core installs it (``update_agent``'s keywords); the core has its
        toolsets set up (``_AgentHooks._core_setup_tools``) and reads their tools then. ``task``:
        it runs a ``VoiceTask``."""
        # its tools, instructions and hooks are what the core runs from now on
        agent._frozen = True
        pool = self._pool(agent)
        frame: dict[str, Any] = {
            "instructions": agent._instructions,
            "dynamic_instructions": (
                _DynamicInstructions(self, agent) if agent._instruction_fns else None
            ),
            "agent": agent.name,
            "agent_host": agent,
            "tools_host": pool,
            "voice_agent": self._voice_agent(agent, task=task),
        }
        # every leg a node may wrap is given, so the previous frame's node never stays on it; the
        # session's STT and TTS stay as they are, under the agent's nodes
        binding = _VoiceBinding(self, agent)
        for leg in ("stt", "llm", "tts"):
            value = getattr(binding, leg)
            if (leg == "llm" and is_given(agent._llm)) or not isinstance(value, llm.RealtimeModel):
                frame[leg] = _bindings.core_leg(value)
        return {**frame, **self._nodes(agent)}

    def _nodes(self, agent: Agent) -> dict[str, Any]:
        """The registered nodes, as the core opens them for ``agent``, each call in its
        binding."""
        registered = [n for n in _NODES if n in self._steps]
        if not registered:
            return {}
        context = contextvars.copy_context()
        context.run(_BindingContextVar.set, _VoiceBinding(self, agent))
        # taken in the agent's binding: each node call runs in a copy of it
        return dict.fromkeys(registered, context.run(_core.AgentNodes, _SessionNodes(self)))

    def _legs(self) -> dict[str, Any]:
        return {
            "stt": self._stt is not None,
            "vad": self._vad is not None,
            "using_default_vad": self._using_default_vad,
            "llm": isinstance(self._llm, llm.LLM),
            "tts": self._tts is not None,
            "realtime": isinstance(self._llm, llm.RealtimeModel),
            "detector": None,
            "adaptive_detector": False,
            "audio_enabled": True,
            "audio_output": True,
            "sink_can_pause": True,
        }

    # --- the core's events ---------------------------------------------------------------

    def _core_event(self, wire: HostWire) -> None:
        kind = type(wire)
        if kind in IGNORED_EVENTS:
            return
        if (name := PUBLIC_EVENTS.get(kind)) is not None:
            if (attr := MIRRORED_STATES.get(kind)) is not None:
                setattr(self, attr, getattr(wire, "new_state"))  # noqa: B009
            self.emit(cast(EventTypes, name), cast(AgentEvent, wire))
            return
        handler = _HANDLERS.get(kind)
        if handler is not None:
            handler(self, wire)

    def _on_speech_created(self, wire: SpeechCreatedWire) -> None:
        handle = SpeechHandle._of(cast(_core.SpeechHandle, wire.speech_handle))
        self._speech_handles[handle.id] = handle
        self.emit(
            "speech_created",
            SpeechCreatedEvent(
                user_initiated=wire.user_initiated,
                source=cast(Literal["say", "generate_reply"], wire.source),
                speech_handle=handle,
            ),
        )

    def _on_speech_done(self, wire: SpeechDoneWire) -> None:
        if (handle := self._speech_handles.pop(wire.speech_id, None)) is not None:
            handle._mark_done()

    def _on_history_reloaded(self, wire: HistoryReloadedWire) -> None:
        self._chat_ctx = ChatContext(list(wire.items))

    def _on_item_added(self, wire: ItemAddedWire) -> None:
        item = wire.item
        # an output told again changed in place (an answered question's)
        idx = self._chat_ctx.index_by_id(item.id) if item.type == "function_call_output" else None
        if idx is not None:
            self._chat_ctx.items[idx] = item
        else:
            self._chat_ctx.insert(item)
        if (run := _wire_run(wire.run)) is not None and item.type != "agent_handoff":
            run._item_added(item)
        if item.type in ("message", "agent_handoff"):
            self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))

    def _on_run_item(self, wire: RunItemAddedWire) -> None:
        if (run := _wire_run(wire.run)) is not None:
            run._item_added(wire.item)

    def _on_run_refused(self, wire: RunRefusedWire) -> None:
        if (run := _wire_run(wire.run)) is not None:
            run._settle(output=None, error=RuntimeError(f"run refused: {wire.cause}"))

    def _on_run_settled(self, wire: RunSettledWire) -> None:
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
        run._settle(output=wire.final_output, error=error, question=wire.question)

    def _on_tool_update(self, wire: ToolExecutionUpdatedWire) -> None:
        update: ToolUpdate = (
            ToolCallStarted(function_call=wire.update.function_call)
            if isinstance(wire.update, ToolCallStartedWire)
            else wire.update
        )
        self.emit("tool_execution_updated", ToolExecutionUpdatedEvent(update=update))

    def _on_metrics(self, wire: MetricsCollectedWire) -> None:
        record = wire.metrics
        collect_usage(record)
        # a record the session metered for a provider (its VAD's) is that provider's event
        if (leg := wire.leg) is not None and (
            provider := getattr(_VoiceBinding(self, self.current_agent), leg)
        ) is not None:
            provider.emit("metrics_collected", record)

    def _error_record(self, wire: SessionErrorWire) -> tuple[Any, Any]:
        """The reference's payload: the provider's error record, and the provider that failed."""
        record, leg = _ERROR_RECORDS[wire.source]
        source = getattr(_VoiceBinding(self, self.current_agent), leg)
        error = record(
            timestamp=time.time(),
            label=getattr(source, "label", wire.label),
            error=wire.error,
            recoverable=wire.recoverable,
        )
        return error, source

    def _core_error_message(self, wire: SessionErrorWire) -> str:
        return str(self._error_record(wire)[0])

    def _on_span_started(self, wire: SpanStartedWire) -> None:
        parent_id = wire.parent_id
        parent = self._spans.get(parent_id) if parent_id is not None else None
        context = trace.set_span_in_context(parent) if parent is not None else None
        span = tracer.start_span(
            wire.name,
            context=context
            if parent_id is None or context is not None
            else self._root_span_context,
            attributes=cast("dict[str, AttributeValue]", wire.attributes),
            start_time=int(wire.start_time * 1e9),
        )
        self._spans[wire.span_id] = span
        if parent_id is None:
            self._session_span = span
            self._root_span_context = trace.set_span_in_context(span)

    def _on_span_ended(self, wire: SpanEndedWire) -> None:
        if (span := self._spans.pop(wire.span_id, None)) is None:
            return
        span.set_attributes(cast("dict[str, AttributeValue]", wire.attributes))
        for event in wire.events:
            span.add_event(
                event.name,
                cast("dict[str, AttributeValue]", event.attributes),
                timestamp=int(event.time * 1e9),
            )
        if (error := wire.error) is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
        for metric in wire.metrics:
            record_span_metric(
                metric.instrument, metric.value, cast("dict[str, str]", metric.attributes)
            )
        span.end(end_time=int(wire.end_time * 1e9))

    def _on_error(self, wire: SessionErrorWire) -> None:
        logger.error("session error", extra={"source": wire.source, "error": str(wire.error)})
        error, source = self._error_record(wire)
        self.emit("error", ErrorEvent(error=error, source=source))

    def _on_overlapping_speech(self, wire: OverlappingSpeechWire) -> None:
        ev = dict(wire.event)
        for key, dtype in (("speech_input", np.int16), ("probabilities", np.float32)):
            if ev[key] is not None:
                ev[key] = np.asarray(ev[key], dtype=dtype)
        self.emit("overlapping_speech", OverlappingSpeechEvent(**ev))

    def _on_amd_result(self, wire: AMDPredictionEvent) -> None:
        if self._amd is not None:
            self._amd._on_prediction(wire)

    def _on_text_input(self, wire: TextInputWire) -> None:
        if (callback := self._text_input_cb) is None:
            return
        room = self._room
        participants = room.remote_participants if isinstance(room, rtc.Room) else {}
        info = cast("dict[str, Any]", wire.info)
        ev = TextInputEvent(
            text=wire.text,
            info=rtc.TextStreamInfo(
                stream_id=info["stream_id"],
                mime_type=info["mime_type"],
                topic=info["topic"],
                timestamp=info["timestamp"],
                size=info["size"],
                attributes=info["attributes"],
                attachments=info["attachments"],
            ),
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

    def _on_delete_room(self, wire: DeleteRoomRequested) -> None:
        if (job_ctx := get_job_context(required=False)) is None:
            return
        logger.info(
            "deleting room on agent session close "
            "(disable via `RoomOptions.delete_room_on_close=False`)",
            extra={"room": wire.room},
        )
        job_ctx.delete_room(room_name=wire.room)

    def _on_close(self, wire: CloseWire) -> None:
        error = self._error_record(cause)[0] if (cause := wire.error) is not None else None
        self.emit("close", CloseEvent(reason=CloseReason(wire.reason), error=error))
        if self._recording_finished is None and (recorder := self._recorder) is not None:
            self._recording_finished = asyncio.ensure_future(recorder.finish())
        audio_in = self._audio_input_options
        if audio_in is not None and isinstance(audio_in.noise_cancellation, rtc.FrameProcessor):
            audio_in.noise_cancellation._close()

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

    def _report_event_snapshots(self) -> list[AgentEvent]:
        expired = self._event_retention.expire(time.monotonic_ns())
        for _ in range(expired):
            self._event_snapshots.popleft()
        return [
            event
            for event in self._event_snapshots
            if self._event_retention.accepts(event.type, self._recording_options["redaction"])
        ]

    @property
    def _recorded_events(self) -> list[AgentEvent]:
        return [copy.deepcopy(event) for event in self._report_event_snapshots()]

    def emit(self, event: EventTypes, arg: AgentEvent) -> None:  # type: ignore[override]
        if self._event_retention.accepts(arg.type, self._recording_options["redaction"]):
            charge = _core._core_value_bytes(arg)
            snapshot = copy.deepcopy(arg) if charge <= _core.SDK_RECORD_BYTES else None
            expired, admitted = self._event_retention.admit(charge, time.monotonic_ns())
            for _ in range(expired):
                self._event_snapshots.popleft()
            if admitted and snapshot is not None:
                self._event_snapshots.append(snapshot)
        super().emit(event, arg)


_ERROR_RECORDS: dict[str, tuple[Any, str]] = {
    "llm_error": (llm.LLMError, "llm"),
    "tts_error": (tts.TTSError, "tts"),
    "stt_error": (stt.STTError, "stt"),
    "realtime_model_error": (llm.RealtimeModelError, "llm"),
}


class VoiceTaskFailed(ToolError):
    """``VoiceTask.run()``'s task failed: one of its agent's tools returned ``Failed(reason)``,
    whose reason is ``message``. Uncaught in the awaiting tool, the reason is that tool's error
    for its model."""


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
        session._agent = agent
        try:
            output: Output_T = await reply
            return output
        finally:
            session._agent = binding.agent


def _wire_run(run: object | None) -> _VoiceRun | None:
    """A wire's run: the `_VoiceRun` the host handed the core."""
    return cast("_VoiceRun | None", run)


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


def _origin() -> str | None:
    """What the running hook or tool sends its commands as; ``None`` from outside the session."""
    binding = _BindingContextVar.get(None)
    return binding.origin if isinstance(binding, _VoiceBinding) else None


@contextlib.contextmanager
def _in_hook(hook: str, session: VoiceSession[Any], agent: Agent) -> Any:
    binding = _BindingContextVar.set(_VoiceBinding(session, agent, origin="hook"))
    name = _hook_name.set(hook)
    try:
        yield
    finally:
        _hook_name.reset(name)
        _BindingContextVar.reset(binding)


class _AgentHooks:
    """The agent's hooks as the core awaits them (``_core.SessionHooks``): each runs in the agent's
    binding and returns the core's answer; ``render_tool_reply`` is the core's default."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        # the toolsets this call holds (tools.md "Toolsets"): the session's for the whole call,
        # an agent's for each activation; shared ones are reference counted
        self._held: list[llm.Toolset] = []

    async def _acquire(self, tools: Iterable[llm.Tool | llm.Toolset]) -> None:
        """Acquire the toolsets of ``tools``; one that fails to set up is logged and the call
        runs without it."""
        toolsets = [t for t in tools if isinstance(t, llm.Toolset)]
        results = await asyncio.gather(*(t._acquire() for t in toolsets), return_exceptions=True)
        for toolset, result in zip(toolsets, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "failed to set up a toolset", exc_info=result, extra={"toolset": toolset.id}
                )
            else:
                self._held.append(toolset)

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

    async def _core_close_toolsets(self, agent: Agent | None) -> None:
        """Release what ``agent``'s activation held, or, as the session closes, everything."""
        if agent is None:
            released, self._held = self._held, []
        else:
            released = [t for t in self._agent_toolsets(agent.tools) if t in self._held]
            for toolset in released:
                self._held.remove(toolset)
        await asyncio.gather(*(t._release() for t in released))

    async def _core_setup_tools(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[FunctionToolInfo]:
        """An activation's tools with their policies, read once its agent's toolsets are acquired
        (the session's already are): a provider tool has none."""
        await self._acquire(self._agent_toolsets(tools))
        infos = {info.spec["name"]: info for info in tool_infos(llm.ToolContext(tools))}
        return [
            infos.get(spec["name"]) or FunctionToolInfo(spec=spec) for spec in tool_specs(tools)
        ]

    def _agent_toolsets(self, tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Toolset]:
        return [t for t in tools if isinstance(t, llm.Toolset) and t not in self._session._tools]

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


_NODES = ("llm_node", "tts_node", "stt_node", "transcription_node", "realtime_audio_output_node")

# the providers `update_options` replaces; each is the session's own, never an agent's
_MODELS = ("stt", "tts", "vad")

# the options given as a partial dict, merged key by key over the session's
_GROUPS = ("endpointing_opts", "interruption")


def _given(**options: Any) -> dict[str, Any]:
    return {k: v for k, v in options.items() if is_given(v)}


def _stt_model(model: NotGivenOr[stt.STT | str | None]) -> NotGivenOr[stt.STT | None]:
    return inference.STT.from_model_string(model) if isinstance(model, str) else model


def _tts_model(model: NotGivenOr[tts.TTS | str | None]) -> NotGivenOr[tts.TTS | None]:
    return inference.TTS.from_model_string(model) if isinstance(model, str) else model


class _SessionNodes:
    """The session's registered nodes as the core calls them (``_core.AgentNodes``), each call in
    a copy of the agent's binding: one completion, one spoken segment, one recognition stream, one
    segment's transcript, one realtime generation's audio. The core feeds what they read (its
    streams, read with ``recv()``) and drains what they return."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    def _call(self, step: str, *args: Any, settings: ModelSettings | None = None) -> Any:
        ctx = RunContext(function_call=None, session=self._session)
        return self._session._steps[step](ctx, *args, settings or ModelSettings())

    def llm(self, req: _core.ChatRequest) -> AsyncIterator[Any]:
        chat_ctx, tools = chat_request(req)
        choice = req.tool_choice
        settings = ModelSettings(tool_choice=NOT_GIVEN if choice is None else choice)
        return _node_output(lambda: self._call("llm_node", chat_ctx, tools, settings=settings))

    def tts(self, text: Any, calls: Any) -> AsyncIterator[Any]:
        def node() -> Any:
            _NodeCallsVar.set(calls)
            return self._call("tts_node", _node_input(text))

        return _node_output(node, lambda frame: tts.SynthesizedAudio(frame=frame, request_id=""))

    def stt(self, audio: Any) -> AsyncIterator[Any]:
        return _node_output(lambda: self._call("stt_node", _node_input(audio)))

    def transcription(self, text: Any) -> AsyncIterator[Any]:
        words = _node_input(text, _plain_text)
        return _node_output(lambda: self._call("transcription_node", words))

    def realtime_audio(self, audio: Any) -> AsyncIterator[Any]:
        frames = _node_input(audio)
        return _node_output(lambda: self._call("realtime_audio_output_node", frames))


class DefaultPipeline:
    """``session.default``: the built-in pipeline steps, with the callbacks' signatures."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    async def llm_node(
        self,
        ctx: RunContext[Any],
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        """A request to the session's LLM."""
        model = self._session.llm
        assert isinstance(model, llm.LLM), "llm_node needs an LLM (not a realtime model)"
        conn_options = self._session.conn_options.llm_conn_options or DEFAULT_API_CONNECT_OPTIONS
        async with model.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=model_settings.tool_choice,
            conn_options=conn_options,
        ) as stream:
            async for chunk in stream:
                yield chunk

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
        events = _core.stt_node(
            session.stt, audio, conn_options=session.conn_options.stt_conn_options, vad=session.vad
        )
        return (stt._speech_event_from_core(ev) async for ev in events)

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
        text: AsyncIterable[str | TimedString],
        model_settings: ModelSettings,
    ) -> AsyncIterator[str | TimedString]:
        """The transcript as spoken."""
        async for delta in text:
            yield delta

    async def realtime_audio_output_node(
        self,
        ctx: RunContext[Any],
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterator[rtc.AudioFrame]:
        """The realtime model's audio as generated."""
        async for frame in audio:
            yield frame

    async def on_user_turn_exceeded(self, ctx: RunContext[Any], ev: UserTurnExceededEvent) -> None:
        """A short reply cutting in on the person, from the turn's transcript."""
        await self._session.generate_reply(
            user_input=ev.transcript,
            instructions=(
                "The user has been speaking too long without giving a chance to reply. "
                "Politely cut in with a short reply or notice. Keep it short since the user "
                "cannot interrupt it."
            ),
            allow_interruptions=False,
            tool_choice="none",
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


def _plain_text(text: TimedString) -> str | TimedString:
    """A transcript delta as the node reads it: a `str`, or a `TimedString` when it is timed."""
    timed = is_given(text.start_time) or is_given(text.end_time)
    return text if timed else str(text)


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


# The `HostWire` members with host behaviour, by record class.
_HANDLERS: dict[type, Callable[[VoiceSession[Any], Any], None]] = {
    HistoryReloadedWire: VoiceSession._on_history_reloaded,
    SpeechCreatedWire: VoiceSession._on_speech_created,
    SpeechDoneWire: VoiceSession._on_speech_done,
    ItemAddedWire: VoiceSession._on_item_added,
    RunItemAddedWire: VoiceSession._on_run_item,
    RunRefusedWire: VoiceSession._on_run_refused,
    RunSettledWire: VoiceSession._on_run_settled,
    ToolExecutionUpdatedWire: VoiceSession._on_tool_update,
    MetricsCollectedWire: VoiceSession._on_metrics,
    OverlappingSpeechWire: VoiceSession._on_overlapping_speech,
    SessionErrorWire: VoiceSession._on_error,
    CloseWire: VoiceSession._on_close,
    AMDPredictionEvent: VoiceSession._on_amd_result,
    TextInputWire: VoiceSession._on_text_input,
    DeleteRoomRequested: VoiceSession._on_delete_room,
    SpanStartedWire: VoiceSession._on_span_started,
    SpanEndedWire: VoiceSession._on_span_ended,
}


@function_tool(name="record_keyterms")
async def _record_keyterms(pending: list[str], confirm: list[str], remove: list[str]) -> None:
    """Update the STT keyterms based on the latest transcript.

    Args:
        pending: Distinctive terms seen but not yet trusted — tracked, not applied.
        confirm: Pending terms the transcript has now corroborated — applied.
        remove: Only a spelling the user corrected away; applied terms are otherwise sticky.
    """
    # only used to elicit a structured tool call; never executed


def _detection_llm(stt_context: dict[str, Any] | None) -> LLM | None:
    """The detection LLM ``stt_context_options`` asks for: the ``LLM`` given, else the model the
    core names (``_core.resolve_keyterm_detection``) through the inference gateway. ``None`` with
    detection off, or when the model cannot be created (detection is then skipped)."""
    config = (stt_context or {}).get("keyterm_detection")
    resolved = _core.resolve_keyterm_detection(config)
    if not resolved.enabled:
        return None
    if (model := resolved.model) is None:
        llm = config["llm"]  # type: ignore[index]
        return llm if isinstance(llm, LLM) else None
    try:
        from ..inference import LLM as InferenceLLM

        return InferenceLLM.from_model_string(model)
    except Exception:  # noqa: BLE001 — never let detection setup break the session
        logger.warning("keyterm detection: could not create detection LLM %r; skipping", model)
        return None
