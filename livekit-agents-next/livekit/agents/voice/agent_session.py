"""``LiveSession``: the developer's handle on a session the Rust core runs.

The core (``_core.LiveSession``) owns the whole session: turn detection, scheduling, generation,
tools' ordering, interruptions, the room's io. This class turns the constructor's arguments into
the core's, forwards each call as one command, mirrors the core's events into the public event
classes and answers the core's hook calls by running the ``Agent``'s methods.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypedDict, TypeVar

from livekit import rtc

from .. import _core, inference, llm, stt, tts, vad
from .._core_view import (
    Realtime,
    TurnDetector,
    chat_request,
    from_core_item,
    realtime_model,
    to_core_items,
    tool_specs,
)
from .._generated.expressive import ExpressiveOptions, NonverbalOptions, SpeechSteeringOptions
from .._generated.session_dispatch import IGNORED_EVENTS, MIRRORED_STATES, PUBLIC_EVENTS
from ..job import get_job_context
from ..llm import ChatContext, RealtimeModel, StopResponse
from ..llm.chat_context import Instructions
from ..log import logger
from ..metrics import AgentSessionUsage
from ..telemetry import otel_metrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    FlushSentinel,
    NotGivenOr,
)
from ..utils import aio, shortuuid
from ..utils.misc import is_dev_mode, is_given, is_hosted
from . import io
from .agent import Agent, ModelSettings, _BindingContextVar, _VoiceBinding
from .console import ConsoleCall
from .events import (
    AgentEvent,
    AgentState,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    RunContext,
    SpeechCreatedEvent,
    ToolExecutionUpdatedEvent,
    UserState,
)
from .remote_session import SessionHost
from .run_result import RunOutputOptions, RunResult
from .speech_handle import SpeechHandle
from .tool_executor import ToolRunner

if TYPE_CHECKING:
    from ..storage.context_store import ContextStore
    from ..storage.resolve import Thread
    from .room_io import RoomOptions

__all__ = [
    "LiveSession",
    "ExpressiveOptions",
    "NonverbalOptions",
    "RecordingOptions",
    "SessionConnectOptions",
    "SpeechSteeringOptions",
    "VoiceActivityVideoSampler",
]

Userdata_T = TypeVar("Userdata_T", default=Any)
Run_T = TypeVar("Run_T")

# the hook running in this context, for `generate_reply(on_enter=)`
_hook_name: contextvars.ContextVar[str] = contextvars.ContextVar("lk_hook_name")

# on_user_turn_completed and on_exit are awaited by the core this long before it moves on
_HOOK_BOUND = 15.0

# the session options and their defaults; the core resolves them (`_core.SessionOptions`)
_OPTION_DEFAULTS: dict[str, Any] = {
    "turn_handling": NOT_GIVEN,
    "stt_context_options": NOT_GIVEN,
    "keyterms_options": NOT_GIVEN,
    "tool_handling": NOT_GIVEN,
    "max_tool_steps": 3,
    "use_tts_aligned_transcript": NOT_GIVEN,
    "tts_text_transforms": NOT_GIVEN,
    "min_consecutive_speech_delay": 0.0,
    "expressive": False,
    "aec_warmup_duration": NOT_GIVEN,
    "ivr_detection": False,
    "user_away_timeout": 15.0,
    "transcription_timeout": None,
    "session_close_transcript_timeout": 2.0,
}

# the deprecated turn kwargs the core migrates into `turn_handling`
_DEPRECATED_TURN_KWARGS = (
    "min_endpointing_delay",
    "max_endpointing_delay",
    "false_interruption_timeout",
    "turn_detection",
    "discard_audio_if_uninterruptible",
    "min_interruption_duration",
    "min_interruption_words",
    "allow_interruptions",
    "resume_false_interruption",
    "agent_false_interruption_timeout",
    "preemptive_generation",
)


class RecordingOptions(TypedDict, total=False):
    """Which recording features are active; keys default to ``True`` when not specified."""

    audio: bool
    traces: bool
    logs: bool
    transcript: bool
    redaction: bool


@dataclass
class SessionConnectOptions:
    stt_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    llm_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    tts_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    max_unrecoverable_errors: int = 3
    """Maximum number of consecutive unrecoverable errors from stt, llm, or tts."""


class VoiceActivityVideoSampler:
    """Sample the camera at ``speaking_fps`` while the user speaks, ``silent_fps`` otherwise."""

    def __init__(self, *, speaking_fps: float = 1.0, silent_fps: float = 0.3) -> None:
        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps


def _core_leg(leg: Any) -> Any:
    """A gateway client or a fallback chain as the core's own (``_core.InferenceStt``,
    ``_core.FallbackStt`` and the like), which the session drives directly rather than through
    the Python wrapper."""
    if isinstance(leg, (inference.STT, inference.LLM, inference.TTS)):
        return leg._core_native()
    if isinstance(leg, (stt.FallbackAdapter, llm.FallbackAdapter, tts.FallbackAdapter)):
        return leg._native
    return leg


class LiveSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    def __init__(
        self,
        *,
        stt: NotGivenOr[stt.STT | str] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | str] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | str] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        context: Any = None,
        thread: Thread | None = None,
        video_sampler: NotGivenOr[VoiceActivityVideoSampler | None] = NOT_GIVEN,
        conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
        **options: Any,
    ) -> None:
        """A voice session over the Rust core.

        Args:
            stt, vad, llm, tts: The providers, or a LiveKit Inference model string. ``vad``
                defaults to ``inference.VAD(model="silero")``; ``vad=None`` opts out.
            tools: Tools shared by every agent of the session.
            userdata, context, thread: Read by tools through ``RunContext``.
            video_sampler: How often camera frames reach the model; ``None`` sends every frame.
            options: The session options the core resolves: ``turn_handling``,
                ``stt_context_options``, ``tool_handling``, ``max_tool_steps``,
                ``use_tts_aligned_transcript``, ``tts_text_transforms``,
                ``min_consecutive_speech_delay``, ``expressive``, ``aec_warmup_duration``,
                ``ivr_detection``, ``user_away_timeout``, ``transcription_timeout``,
                ``session_close_transcript_timeout`` and the deprecated turn keywords.
        """
        super().__init__()
        if unknown := options.keys() - _OPTION_DEFAULTS.keys() - set(_DEPRECATED_TURN_KWARGS):
            raise TypeError(f"unexpected LiveSession options: {sorted(unknown)}")
        deprecated = {k: options.pop(k, NOT_GIVEN) for k in _DEPRECATED_TURN_KWARGS}
        self._conn_options = conn_options or SessionConnectOptions()
        self._options = _core.SessionOptions(
            {**_OPTION_DEFAULTS, **options, "deprecated": deprecated, "conn_options": conn_options}
        )
        stt_context = options.get("stt_context_options", options.get("keyterms_options"))
        self._stt_context = stt_context if is_given(stt_context) else None

        self._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt or None
        self._llm = inference.llm_from_model_string(llm) if isinstance(llm, str) else llm or None
        self._tts = inference.TTS.from_model_string(tts) if isinstance(tts, str) else tts or None
        self._using_default_vad = not is_given(vad)
        self._vad = inference.VAD(model="silero") if not is_given(vad) else vad
        self._tools = tools if is_given(tools) else []
        turn_handling = options.get("turn_handling") or {}
        detection = turn_handling.get("turn_detection", deprecated["turn_detection"])
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

        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None
        self._context = context
        self._thread = thread

        self._core: _core.LiveSession = None  # type: ignore[assignment]
        self._console: ConsoleCall | None = None
        self._agent: Agent | None = None
        # every agent the core may name, by the id it was installed under
        self._agents: dict[str, Agent] = {}
        self._speeches: dict[str, SpeechHandle] = {}
        # runs asked for and not yet started (the core starts them in order), and running ones
        self._pending_runs: list[RunResult[Any]] = []
        self._runs: dict[int, RunResult[Any]] = {}
        self._fillers: dict[int, Callable[[int], Any] | None] = {}
        self._next_filler = 0
        self._chat_ctx = ChatContext.empty()
        self._agent_state: AgentState = "initializing"
        self._user_state: UserState = "listening"
        self._input = io.AgentInput(self)
        self._output = io.AgentOutput(self)
        self._usage = AgentSessionUsage(model_usage=[])
        # what a job's session report reads
        self._started_at: float | None = None
        self._recorded_events: list[AgentEvent] = []
        self._recording_options: RecordingOptions = {
            "audio": False,
            "traces": False,
            "logs": False,
            "transcript": False,
            "redaction": False,
        }
        self._tasks: set[asyncio.Task[Any]] = set()
        # the job's telemetry parents its spans here; the core traces the session itself
        self._root_span_context = None
        self._session_start_context = None
        self._host: SessionHost | None = None

    # --- state ---------------------------------------------------------------------------

    @property
    def userdata(self) -> Userdata_T:
        if self._userdata is None:
            raise ValueError("LiveSession userdata is not set")
        return self._userdata

    @userdata.setter
    def userdata(self, value: Userdata_T) -> None:
        self._userdata = value

    @property
    def store(self) -> ContextStore:
        """The active host's explicitly managed context store."""
        from .agent_run import current_run_hooks

        hooks = current_run_hooks()
        if hooks is None or hooks.store is None:
            raise RuntimeError("This session has no managed store")
        return hooks.store

    @property
    def context(self) -> Any:
        """Application data and services supplied for this session."""
        return self._context

    @property
    def thread(self) -> Thread | None:
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
    def vad(self) -> vad.VAD | None:
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
    def agent_state(self) -> AgentState:
        return self._agent_state

    @property
    def user_state(self) -> UserState:
        return self._user_state

    @property
    def current_agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("LiveSession isn't running")
        return self._agent

    @property
    def usage(self) -> AgentSessionUsage:
        return self._usage

    @property
    def _running(self) -> _core.LiveSession:
        if self._core is None:
            raise RuntimeError("LiveSession isn't running")
        return self._core

    # --- lifecycle -----------------------------------------------------------------------

    async def start(
        self,
        agent: Agent,
        *,
        room: NotGivenOr[rtc.Room | _core.FakeRoom | _core.OfflineRoom] = NOT_GIVEN,
        room_options: NotGivenOr[RoomOptions] = NOT_GIVEN,
        chat_ctx: llm.ChatContext | None = None,
        capture_run: bool = False,
    ) -> None:
        """Start the session on ``room`` (the job's room by default, an offline room without
        one) with ``agent`` driving."""
        if self._core is not None:
            return
        self._console = self._host = ConsoleCall.claim(self)
        job_ctx = get_job_context(required=False)
        if self._console is not None:
            room = _core.OfflineRoom("console")
        elif not is_given(room):
            room = job_ctx.room if job_ctx is not None else _core.OfflineRoom("console")
        if chat_ctx is not None:
            self._chat_ctx = chat_ctx.copy()

        frame = await self._frame(agent)
        model = frame.get("llm", self._llm)
        # an OpenAI-protocol model or a fallback chain runs in the core; any other one is a host
        # realtime model
        rt = (
            realtime_model(model, lambda: self._pool(self.current_agent))
            if isinstance(model, llm.RealtimeModel)
            else None
        )
        realtime = rt if isinstance(rt, Realtime) else None
        native = None if isinstance(rt, Realtime) else rt
        sampler = self._video_sampler
        self._core = _core.LiveSession(
            room,
            instructions=frame["instructions"],
            dynamic_instructions=frame["dynamic_instructions"],
            agent=frame["agent"],
            chat_ctx=to_core_items(self._chat_ctx.items),
            stt=_core_leg(frame.get("stt", self._stt)),
            vad=frame.get("vad", self._vad),
            llm=None if isinstance(model, llm.RealtimeModel) else _core_leg(model),
            realtime=native,
            realtime_host=realtime,
            tts=_core_leg(frame.get("tts", self._tts)),
            host_audio_input=self._console.io if self._console else self._input.audio,
            host_video_input=self._input.video,
            host_audio_output=self._console.io if self._console else self._output.audio,
            host_transcription_output=self._output.transcription,
            on_event=self,
            options=self._options,
            agent_turn_handling=frame.get("turn_handling"),
            agent_expressive=frame["expressive"],
            tools=frame["tools"],
            tools_host=frame["tools_host"],
            tool_runner=frame["tool_runner"],
            room_io_config=room_options._to_core() if is_given(room_options) else None,
            video_sampler=(sampler.speaking_fps, sampler.silent_fps) if sampler else None,
            using_default_vad=self._using_default_vad,
            turn_detector=self._turn_detector,
            turn_detector_streaming=self._turn_detector is not None
            and self._turn_detector.streaming,
            hook_bound=_HOOK_BOUND,
            max_unrecoverable_errors=self._conn_options.max_unrecoverable_errors,
            interruption_detector=self._interruption_detector(agent, frame, model),
            stt_context=self._stt_context,
            **{node: frame[node] for node in _NODES if node in frame},
        )
        self._agent = agent
        self._started_at = time.time()
        if self._console is not None:
            await self._core.start(capture_run=capture_run)
            await self._console.start()
            return
        if job_ctx is None or room is not job_ctx.room:
            await self._core.start(capture_run=capture_run)
            return
        # the job's room: this session records and reports for the job, connects the room,
        # closes with it, and the primary one serves the remote session protocol there
        plan = _core.plan_session_start(
            job_default=job_ctx.job.enable_recording,
            has_job_context=True,
            another_session_is_primary=job_ctx._primary_agent_session is not None,
        )
        self._recording_options = plan["recording"]
        if plan["is_primary"]:
            job_ctx._primary_agent_session = self
        if plan["hosting"]:
            self._host = SessionHost.on_room(self, room)
        job_ctx.init_recording(self._recording_options)
        job_ctx.add_shutdown_callback(
            lambda: self._close(drain=False, reason=CloseReason.JOB_SHUTDOWN)
        )
        await asyncio.gather(self._core.start(capture_run=capture_run), job_ctx.connect())
        if self._host is not None:
            await self._host.start()

    def _interruption_detector(
        self, agent: Agent, frame: dict[str, Any], model: Any
    ) -> inference.AdaptiveInterruptionDetector | None:
        """The adaptive interruption detector, when the reference's gate builds one."""
        # the developer's STT: a fallback chain's core leg reports its capabilities as a dict
        stt_caps = getattr(_VoiceBinding(self, agent).stt, "capabilities", None)
        has_vad = frame.get("vad", self._vad) is not None
        agent_mode = (
            agent.interruption_detection if is_given(agent.interruption_detection) else None
        )
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
                streaming_detector=self._turn_detector is not None
                and self._turn_detector.streaming,
                interruption_given=bool(agent_mode or self._interruption_mode),
            ),
            has_vad=has_vad,
            turn_detection=self._turn_detection_mode,
            allow_interruptions=self.options["interruption"]["enabled"],
            agent_mode=agent_mode,
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
        await self._close(drain=False)

    async def drain(self) -> None:
        await self._running.drain()

    def shutdown(self, *, drain: bool = True) -> None:
        self._spawn(self._close(drain=drain))

    async def _close(
        self, *, drain: bool, reason: CloseReason = CloseReason.USER_INITIATED
    ) -> None:
        if self._core is None:
            return
        await self._core.aclose(reason=reason.value, drain=drain)
        if self._host is not None:
            host, self._host = self._host, None
            await host.aclose()
        await self._core.close_io()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*(t.aclose() for t in self._tools if isinstance(t, llm.Toolset)))

    async def __aenter__(self) -> LiveSession[Userdata_T]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- commands ------------------------------------------------------------------------

    def run(
        self,
        *,
        user_input: str,
        output_type: type[Run_T] | None = None,
        output_options: NotGivenOr[RunOutputOptions | None] = NOT_GIVEN,
    ) -> RunResult[Run_T]:
        """Reply to ``user_input`` and record what the reply did, for assertions."""
        result = RunResult[Run_T](
            user_input=user_input,
            output_type=output_type,
            output_options=output_options,
            session=self,
        )
        self._running.run(
            user_input=user_input,
            expects_output=output_type is not None,
            max_retries=result._output_retries,
            retry_instructions=result._output_retry_instructions,
            origin=_origin(),
        )
        self._pending_runs.append(result)
        return result

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
            core, pusher = self._running.say_audio(text, **kwargs)
            self._pump(audio, pusher)
        elif isinstance(text, str):
            core = self._running.say(text, **kwargs)
        else:
            core, pusher = self._running.say_stream(**kwargs)
            self._pump(text, pusher)
        return SpeechHandle(core)

    def _pump(self, source: AsyncIterable[Any], pusher: Any) -> None:
        async def _run() -> None:
            try:
                async for item in source:
                    if not await pusher.push(item):
                        break
            finally:
                pusher.end()

        self._spawn(_run())

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str | Instructions] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> SpeechHandle:
        """Reply now. ``tools`` narrows the offered tools to these names for this reply."""
        toolset: dict[str, Any] = {}
        if is_given(tools):
            pool = self._pool(self.current_agent)
            specs = [s for s in tool_specs(pool) if s["name"] in tools]
            toolset = {
                "tools": specs,
                "tools_host": pool,
                "tool_runner": ToolRunner(self, self.current_agent, pool),
            }
        if isinstance(instructions, Instructions):
            instructions = instructions.render(modality="audio")
        core = self._running.generate_reply(
            user_input=user_input if is_given(user_input) else None,
            instructions=instructions if is_given(instructions) else None,
            allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            chat_ctx=to_core_items(chat_ctx.items) if is_given(chat_ctx) else None,
            on_enter=_origin() == "hook" and _hook_name.get(None) == "on_enter",
            origin=_origin(),
            **toolset,
        )
        return SpeechHandle(core)

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        return asyncio.ensure_future(self._running.interrupt(force=force))

    def clear_user_turn(self) -> None:
        self._running.clear_user_turn()

    async def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> str:
        return await self._running.commit_user_turn(  # type: ignore[no-any-return]
            transcript_timeout=transcript_timeout,
            stt_flush_duration=stt_flush_duration,
            skip_reply=skip_reply,
        )

    async def wait_for_idle(self) -> None:
        await self._running.wait_for_idle()

    def update_agent(self, agent: Agent) -> None:
        """Hand the conversation to ``agent``; the session and its history stay."""

        async def _update() -> None:
            await self._running.update_agent(**await self._frame(agent))

        self._spawn(_update())
        self._agent = agent

    def _update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        """Replace the context the running agent works on."""
        self._running.update_chat_ctx(to_core_items(chat_ctx.items))

    def update_options(self, **update: Any) -> None:
        """Change the turn options and ``expressive`` of a session (``UpdateKwargs``)."""
        if self._core is None:
            self._options.update(update)
        else:
            self._core.update_turn_options(update)

    # --- frames --------------------------------------------------------------------------

    def _pool(self, agent: Agent) -> list[llm.Tool | llm.Toolset]:
        return [*agent.tools, *self._tools]

    async def _frame(self, agent: Agent) -> dict[str, Any]:
        """``agent`` as the core installs it (``update_agent``'s keywords), its toolsets set up."""
        self._agents[agent.id] = agent
        pool = self._pool(agent)
        # a toolset that fails to set up is logged and the agent runs without it
        await asyncio.gather(
            *(_setup_toolset(t) for t in pool if isinstance(t, llm.Toolset)),
        )
        frame: dict[str, Any] = {
            "instructions": _instructions(agent.instructions),
            "dynamic_instructions": (
                _DynamicInstructions(self, agent) if agent._instruction_fns else None
            ),
            "agent": agent.id,
            "tools": tool_specs(pool),
            "tools_host": pool,
            "tool_runner": ToolRunner(self, agent, pool),
            "turn_handling": agent._turn_handling or None,
            "expressive": agent.expressive,
        }
        if is_given(agent._vad):
            frame["vad"] = _core_leg(agent._vad)
        # every leg a node may wrap is given, so the previous frame's node never stays on it
        binding = _VoiceBinding(self, agent)
        for leg in ("stt", "llm", "tts"):
            value = getattr(binding, leg)
            if is_given(getattr(agent, f"_{leg}")) or not isinstance(value, llm.RealtimeModel):
                frame[leg] = _core_leg(value)
        return {**frame, **self._nodes(agent)}

    def _nodes(self, agent: Agent) -> dict[str, Any]:
        """The nodes ``agent`` overrides, as the core opens them over each provider."""
        nodes = _AgentNodes(_VoiceBinding(self, agent))
        return {n: nodes for n in _NODES if getattr(type(agent), n) is not getattr(Agent, n)}

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

    def _core_event(self, wire: dict[str, Any]) -> None:
        kind = wire["type"]
        if self._host is not None:
            self._host.forward(wire)
        if kind in IGNORED_EVENTS:
            return
        if (public := PUBLIC_EVENTS.get(kind)) is not None:
            cls, name = public
            ev = cls(**{k: v for k, v in wire.items() if k != "type"})
            if (attr := MIRRORED_STATES.get(kind)) is not None:
                setattr(self, attr, ev.new_state)
            self.emit(name, ev)
            return
        handler = _HANDLERS.get(kind)
        if handler is not None:
            handler(self, wire)

    def _core_stream_closed(self, wire: dict[str, Any]) -> None:
        self._speeches.clear()

    def _on_hook(self, wire: dict[str, Any]) -> None:
        self._spawn(_answer_hook(self, wire))

    def _on_speech_created(self, wire: dict[str, Any]) -> None:
        handle = SpeechHandle(wire["speech_handle"])
        self._speeches[handle.id] = handle
        self.emit(
            "speech_created",
            SpeechCreatedEvent(
                user_initiated=wire["user_initiated"], source=wire["source"], speech_handle=handle
            ),
        )

    def _on_speech_done(self, wire: dict[str, Any]) -> None:
        self._speeches.pop(wire["speech_id"], None)

    def _on_item_added(self, wire: dict[str, Any]) -> None:
        item = from_core_item(wire["item"])
        self._chat_ctx.insert(item)
        if (run := self._runs.get(wire["run"])) is not None:
            if item.type == "agent_handoff":
                run._agent_handoff(
                    item=item,
                    old_agent=self._agents.get(item.old_agent_id or ""),
                    new_agent=self._agents[item.new_agent_id],
                )
            else:
                run._item_added(item)
        if item.type in ("message", "agent_handoff"):
            self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))

    def _on_run_started(self, wire: dict[str, Any]) -> None:
        self._runs[wire["run"]] = self._pending_runs.pop(0)

    def _on_run_item(self, wire: dict[str, Any]) -> None:
        if (run := self._runs.get(wire["run"])) is not None:
            run._item_added(from_core_item(wire["item"]))

    def _on_run_refused(self, wire: dict[str, Any]) -> None:
        run = self._pending_runs.pop(0)
        run._settle(output=None, error=RuntimeError(f"run refused: {wire['cause']}"))

    def _on_run_settled(self, wire: dict[str, Any]) -> None:
        run = self._runs.pop(wire["run"])
        error = wire["error"]
        if wire["outcome"] == "failed" and error is None and wire["speech_handle"] is not None:
            error = wire["speech_handle"].exception()
        if wire["outcome"] == "closed":
            error = RuntimeError("the session closed before the run settled")
        run._settle(output=wire["final_output"], error=error)

    def _on_tool_update(self, wire: dict[str, Any]) -> None:
        self.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent.model_validate(
                {"update": {k: _host_value(v) for k, v in wire["update"].items()}}
            ),
        )

    def _on_metrics(self, wire: dict[str, Any]) -> None:
        ev = MetricsCollectedEvent(metrics=wire["metrics"])
        otel_metrics.collect_usage(ev.metrics)
        self.emit("metrics_collected", ev)

    def _on_error(self, wire: dict[str, Any]) -> None:
        logger.error("session error", extra={"source": wire["source"], "error": str(wire["error"])})
        self.emit(
            "error",
            ErrorEvent(error=wire["error"], source=wire["source"]),
        )

    def _on_close(self, wire: dict[str, Any]) -> None:
        self.emit("close", CloseEvent(reason=CloseReason(wire["reason"]), error=None))

    def emit(self, event: EventTypes, arg: AgentEvent) -> None:  # type: ignore[override]
        self._recorded_events.append(arg)
        super().emit(event, arg)

    # --- speech handles, fillers, tasks ----------------------------------------------------

    def _speech(self, speech_id: str) -> SpeechHandle | None:
        return self._speeches.get(speech_id)

    def _open_filler(self, source: Callable[[int], Any] | None) -> int:
        self._next_filler += 1
        self._fillers[self._next_filler] = source
        return self._next_filler

    def _close_filler(self, filler: int) -> None:
        self._fillers.pop(filler, None)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


def __getattr__(name: str) -> Any:
    if name == "AgentSession":
        # deprecated alias, kept one release
        return LiveSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


async def _setup_toolset(toolset: llm.Toolset) -> None:
    try:
        await toolset.setup()
    except Exception:
        logger.exception("failed to set up a toolset", extra={"toolset": type(toolset).__name__})


def _host_value(value: Any) -> Any:
    return from_core_item(value) if isinstance(value, _core.ChatItem) else value


def _instructions(value: str | Instructions) -> str | tuple[str, str | None, str | None]:
    if isinstance(value, Instructions):
        return (value.common, value.audio, value.text)
    return str(value)


class _DynamicInstructions:
    """An agent's ``@agent.instructions`` functions, rendered by the core before each model
    request of a reply (``_core_render``)."""

    def __init__(self, session: LiveSession[Any], agent: Agent) -> None:
        self._session = session
        self._agent = agent

    async def _core_render(self, speech_id: int) -> str:
        ctx = RunContext(
            function_call=None,
            session=self._session,
            speech_handle=self._session._speech(f"speech_{speech_id}"),
        )
        return await self._agent._render_dynamic_instructions(ctx)


def _origin() -> str | None:
    """What the running hook or tool sends its commands as; ``None`` from outside the session."""
    binding = _BindingContextVar.get(None)
    return binding.origin if isinstance(binding, _VoiceBinding) else None


@contextlib.contextmanager
def _in_hook(hook: str, session: LiveSession[Any], agent: Agent) -> Any:
    binding = _BindingContextVar.set(_VoiceBinding(session, agent, origin="hook"))
    name = _hook_name.set(hook)
    try:
        yield
    finally:
        _hook_name.reset(name)
        _BindingContextVar.reset(binding)


async def _answer_hook(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    """Run the hook the core called and answer it, exactly once. ``render_tool_reply`` stays
    unanswered: the core uses its default templates."""
    try:
        await _HOOKS.get(wire["hook"], _unanswered)(session, wire)
    except Exception:
        logger.exception(f"error in the {wire['hook']} hook")
        session._core.hook_failed(wire["id"])


async def _on_enter(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    agent = session._agents[wire["agent"]]
    with _in_hook("on_enter", session, agent):
        await agent._enter(RunContext(function_call=None, session=session))
    session._core.hook_returned(wire["id"])


async def _on_exit(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    agent = session._agents[wire["agent"]]
    with _in_hook("on_exit", session, agent):
        await agent.on_exit()
    session._core.hook_returned(wire["id"])
    await asyncio.gather(*(t.aclose() for t in agent.tools if isinstance(t, llm.Toolset)))


async def _tools_executed(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    ev = FunctionToolsExecutedEvent(
        function_calls=[from_core_item(c) for c in wire["function_calls"]],
        function_call_outputs=[from_core_item(o) for o in wire["function_call_outputs"]],
    )
    ev._handoff_required = wire["handoff_required"]
    session.emit("function_tools_executed", ev)
    session._core.hook_tools_executed(
        wire["id"], [o.reply_required for o in ev.function_call_outputs], ev.has_agent_handoff
    )


async def _filler_source(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    source = session._fillers.get(wire["filler"])
    out = source(wire["step"]) if source is not None else None
    if asyncio.iscoroutine(out):
        out = await out
    if isinstance(out, SpeechHandle):
        session._core.hook_filler(wire["id"], spoke=True)
    else:
        session._core.hook_filler(wire["id"], text=out)


async def _unanswered(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    session._core.hook_failed(wire["id"])


async def _turn_completed(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    agent = session.current_agent
    turn_ctx = session.history.copy()
    message = llm.ChatMessage(
        role="user",
        content=[wire["new_transcript"]],
        transcript_confidence=wire["transcript_confidence"],
    )
    before = [i.id for i in turn_ctx.items]
    stop = False
    with _in_hook("on_user_turn_completed", session, agent):
        try:
            await agent.on_user_turn_completed(turn_ctx, message)
        except StopResponse:
            stop = True
    changed = [i.id for i in turn_ctx.items] != before
    session._core.hook_turn_done(
        wire["id"],
        user_text=message.text_content if message.text_content != wire["new_transcript"] else None,
        items=to_core_items(turn_ctx.items) if changed else None,
        stop=stop,
    )


_NODES = ("llm_node", "tts_node", "stt_node")


class _AgentNodes:
    """An agent's overridden ``llm_node``, ``tts_node`` and ``stt_node`` as the core opens them
    in place of the provider (``_core.NodeFactory``): one completion, one spoken segment, one
    recognition stream."""

    def __init__(self, binding: _VoiceBinding) -> None:
        self._binding = binding

    async def _core_llm_node(self, req: _core.ChatRequest) -> _NodeStream:
        chat_ctx, tools = chat_request(req)
        choice = req.tool_choice
        settings = ModelSettings(tool_choice=NOT_GIVEN if choice is None else choice)
        agent = self._binding.agent
        return _NodeStream(
            self._binding, lambda _: agent.llm_node(chat_ctx, tools, settings), _chat_chunk
        )

    async def _core_tts_node(self) -> _NodeStream:
        agent, request_id = self._binding.agent, shortuuid("tts_node_")
        return _NodeStream(
            self._binding,
            lambda text: agent.tts_node(text, ModelSettings()),
            lambda frame: tts.SynthesizedAudio(frame=frame, request_id=request_id),
        )

    async def _core_stt_node(self) -> _NodeStream:
        agent = self._binding.agent
        return _NodeStream(
            self._binding, lambda audio: agent.stt_node(audio, ModelSettings()), lambda ev: ev
        )


class _NodeStream:
    """One node invocation as the core drives it: input pushed in (``_core_push``,
    ``end_input``), the node's output out (``__anext__``). The node runs in its own task, in the
    agent's binding, so its code sees its session as hooks and tools do."""

    def __init__(
        self,
        binding: _VoiceBinding,
        node: Callable[[AsyncIterable[Any]], Any],
        convert: Callable[[Any], Any],
    ) -> None:
        self._input: asyncio.Queue[Any] = asyncio.Queue()
        self._output: asyncio.Queue[Any] = asyncio.Queue()
        self._convert = convert
        token = _BindingContextVar.set(binding)
        try:
            self._task = asyncio.create_task(self._run(node))
        finally:
            _BindingContextVar.reset(token)

    async def _inputs(self) -> AsyncIterator[Any]:
        while (item := await self._input.get()) is not _END:
            yield item

    async def _run(self, node: Callable[[AsyncIterable[Any]], Any]) -> None:
        try:
            out = node(self._inputs())
            if inspect.isawaitable(out):
                out = await out
            if isinstance(out, AsyncIterable):
                async for item in out:
                    self._output.put_nowait(self._convert(item))
            elif out is not None:
                self._output.put_nowait(self._convert(out))
            self._output.put_nowait(_END)
        except Exception as e:
            self._output.put_nowait(e)

    def _core_push(self, item: Any) -> None:
        self._input.put_nowait(item)

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        self._input.put_nowait(_END)

    async def __anext__(self) -> Any:
        item = await self._output.get()
        if item is _END:
            self._output.put_nowait(_END)
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)


_END = object()


@dataclass(frozen=True)
class _Flush:
    """``FlushSentinel`` as the core reads a chunk: the spoken segment ends here."""

    id: str = ""
    delta: None = None
    usage: None = None
    flush: bool = True


def _chat_chunk(item: llm.ChatChunk | str | FlushSentinel) -> llm.ChatChunk | _Flush:
    if isinstance(item, str):
        return llm.ChatChunk(
            id=shortuuid("chunk_"), delta=llm.ChoiceDelta(role="assistant", content=item)
        )
    if isinstance(item, FlushSentinel):
        return _Flush()
    return item


_HOOKS: dict[str, Callable[[LiveSession[Any], dict[str, Any]], Any]] = {
    "on_enter": _on_enter,
    "on_exit": _on_exit,
    "on_user_turn_completed": _turn_completed,
    "function_tools_executed": _tools_executed,
    "filler_source": _filler_source,
    "render_tool_reply": _unanswered,
}

_HANDLERS: dict[str, Callable[[LiveSession[Any], dict[str, Any]], None]] = {
    "hook_call": LiveSession._on_hook,
    "speech_created": LiveSession._on_speech_created,
    "speech_done": LiveSession._on_speech_done,
    "conversation_item_added": LiveSession._on_item_added,
    "run_started": LiveSession._on_run_started,
    "run_item_added": LiveSession._on_run_item,
    "run_refused": LiveSession._on_run_refused,
    "run_settled": LiveSession._on_run_settled,
    "tool_execution_updated": LiveSession._on_tool_update,
    "metrics_collected": LiveSession._on_metrics,
    "error": LiveSession._on_error,
    "close": LiveSession._on_close,
}
