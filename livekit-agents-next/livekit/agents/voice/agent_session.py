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
import inspect
import time
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal

import numpy as np
from pydantic import TypeAdapter
from typing_extensions import TypedDict, TypeVar

from livekit import rtc

from .. import _core, inference, llm, stt, tts
from .._core_view import (
    TurnDetector,
    chat_request,
    from_core_item,
    realtime_model,
    to_core_items,
    tool_specs,
)
from .._generated.amd import AMDPredictionEvent
from .._generated.expressive import ExpressiveOptions, NonverbalOptions, SpeechSteeringOptions
from .._generated.keyterms import KeytermDetectionOptions, STTContextOptions
from .._generated.session_dispatch import IGNORED_EVENTS, MIRRORED_STATES, PUBLIC_EVENTS
from .._generated.vad import VAD
from ..inference.interruption import OverlappingSpeechEvent
from ..job import get_job_context
from ..llm import LLM, ChatContext, RealtimeModel, StopResponse, function_tool
from ..llm.chat_context import Instructions
from ..log import logger
from ..metrics import _MODEL_USAGE, AgentMetrics, AgentSessionUsage
from ..telemetry import otel_metrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils.misc import is_dev_mode, is_given, is_hosted
from . import io
from .agent import Agent, ModelSettings, _BindingContextVar, _VoiceBinding
from .events import (
    AgentEvent,
    AgentState,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    FunctionToolsExecutedEvent,
    RunContext,
    SessionUsageUpdatedEvent,
    SpeechCreatedEvent,
    ToolExecutionUpdatedEvent,
    UserState,
)
from .remote_session import ConsoleCall, SessionPort
from .run_result import RunOutputOptions, RunResult
from .speech_handle import SpeechHandle
from .tool_executor import ToolRunner

if TYPE_CHECKING:
    from ..storage.context_store import ContextStore
    from ..storage.resolve import Thread
    from .amd import AMD
    from .room_io import RoomOptions

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

Userdata_T = TypeVar("Userdata_T", default=Any)
Run_T = TypeVar("Run_T")

# the hook running in this context, for `generate_reply(on_enter=)`
_hook_name: contextvars.ContextVar[str] = contextvars.ContextVar("lk_hook_name")

# on_exit and the other callbacks are awaited by the core this long before it moves on;
# on_user_turn_completed is awaited until it returns
_HOOK_BOUND = 15.0

# the session options and their defaults; the core resolves them (`_core.SessionOptions`)
_OPTION_DEFAULTS: dict[str, Any] = {
    "turn_handling": NOT_GIVEN,
    "stt_context_options": NOT_GIVEN,
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
    """A gateway client, a fallback chain or the local VAD as the core's own
    (``_core.InferenceStt``, ``_core.FallbackStt``, ``_core.InferenceVad`` and the like), which the
    session drives directly rather than through the Python wrapper."""
    if isinstance(leg, (inference.STT, inference.LLM, inference.TTS)):
        return leg._core_native()
    if isinstance(leg, inference.VAD):
        return leg._native
    if isinstance(
        leg, (stt.FallbackAdapter, stt.StreamAdapter, llm.FallbackAdapter, tts.FallbackAdapter)
    ):
        return leg._native
    return leg


class VoiceSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    def __init__(
        self,
        *,
        stt: NotGivenOr[stt.STT | str] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
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
                ``session_close_transcript_timeout``.
        """
        super().__init__()
        if unknown := options.keys() - _OPTION_DEFAULTS.keys():
            raise TypeError(f"unexpected VoiceSession options: {sorted(unknown)}")
        self._conn_options = conn_options or SessionConnectOptions()
        self._options = _core.SessionOptions(
            {**_OPTION_DEFAULTS, **options, "conn_options": conn_options}
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

        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None
        self._context = context
        self._thread = thread

        self._core: _core.VoiceSession = None  # type: ignore[assignment]
        self._room: rtc.Room | _core.FakeRoom | _core.OfflineRoom | None = None
        self._amd: AMD | None = None
        self._agent: Agent | None = None
        self._chat_ctx = ChatContext.empty()
        self._agent_state: AgentState = "initializing"
        self._user_state: UserState = "listening"
        self._input = io.AgentInput(self)
        self._output = io.AgentOutput(self)
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
        # the job's telemetry parents its spans here; the core traces the session itself
        self._root_span_context = None
        self._session_start_context = None

    # --- state ---------------------------------------------------------------------------

    @property
    def userdata(self) -> Userdata_T:
        if self._userdata is None:
            raise ValueError("VoiceSession userdata is not set")
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
    def usage(self) -> AgentSessionUsage:
        model_usage = self._core.usage if self._core is not None else []
        return AgentSessionUsage(model_usage=[_MODEL_USAGE.validate_python(u) for u in model_usage])

    @property
    def _running(self) -> _core.VoiceSession:
        if self._core is None:
            raise RuntimeError("VoiceSession isn't running")
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
        console = ConsoleCall.claim()
        job_ctx = get_job_context(required=False)
        if console is not None:
            room = _core.OfflineRoom("console")
        elif not is_given(room):
            room = job_ctx.room if job_ctx is not None else _core.OfflineRoom("console")
        if chat_ctx is not None:
            self._chat_ctx = chat_ctx.copy()
        # the job's session (on its room, or the console's) records and reports for the job and
        # closes with it; the primary one, and the console's, serve the remote session protocol
        in_job = job_ctx is not None and (console is not None or room is job_ctx.room)
        hosting = console is not None
        if in_job:
            assert job_ctx is not None
            plan = _core.plan_session_start(
                job_default=job_ctx.job.enable_recording,
                has_job_context=True,
                another_session_is_primary=not job_ctx._run.claim_session(self),
            )
            self._recording_options = plan["recording"]
            hosting = hosting or plan["hosting"]
            job_ctx.init_recording(self._recording_options)

        hooks = _AgentHooks(self)
        frame = self._frame(agent)
        frame["tools"] = await hooks._core_setup_tools(frame["tools_host"])
        model = frame.get("llm", self._llm)
        # an OpenAI-protocol model or a fallback chain runs in the core; any other one is a host
        # realtime model
        rt = (
            realtime_model(model, lambda: self._pool(self.current_agent))
            if isinstance(model, llm.RealtimeModel)
            else None
        )
        # a plugin's model is a host object, any other the core's own
        realtime = rt if hasattr(rt, "_core_session") else None
        native = None if hasattr(rt, "_core_session") else rt
        sampler = self._video_sampler
        keyterm_llm = _detection_llm(self._stt_context)
        self._room = room
        self._core = _core.VoiceSession(
            room,
            instructions=frame["instructions"],
            dynamic_instructions=frame["dynamic_instructions"],
            agent=frame["agent"],
            agent_host=frame["agent_host"],
            chat_ctx=to_core_items(self._chat_ctx.items),
            stt=_core_leg(frame.get("stt", self._stt)),
            vad=_core_leg(frame.get("vad", self._vad)),
            llm=None if isinstance(model, llm.RealtimeModel) else _core_leg(model),
            realtime=native,
            realtime_host=realtime,
            tts=_core_leg(frame.get("tts", self._tts)),
            host_audio_input=self._input.audio,
            host_video_input=self._input.video,
            host_audio_output=self._output.audio,
            host_transcription_output=self._output.transcription,
            on_event=self,
            hooks=hooks,
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
            keyterm_llm=_core_leg(keyterm_llm) if keyterm_llm is not None else None,
            keyterm_tools=[_record_keyterms] if keyterm_llm is not None else None,
            **{node: frame[node] for node in _NODES if node in frame},
            remote_session=SessionPort(self) if hosting else None,
            console=console,
            # the core connects the job's room beside its opening
            job=job_ctx if in_job and console is None else None,
        )
        self._agent = agent
        self._started_at = time.time()
        if in_job:
            assert job_ctx is not None
            job_ctx.add_shutdown_callback(
                lambda: self._core.aclose(reason=CloseReason.JOB_SHUTDOWN.value, drain=False)
            )
        await self._core.start(capture_run=capture_run)
        if self.options["ivr_detection"]:
            await self._start_ivr_detection()

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        """Start IVR detection: the DTMF tool joins the session's tools and the core's navigator
        runs, counting ``transcript`` (AMD's greeting) as heard."""
        from ..beta.tools import send_dtmf_events

        if send_dtmf_events not in self._tools:
            self._tools.append(send_dtmf_events)
        self._running.ivr_start(tools=tool_specs([send_dtmf_events]), tools_host=[send_dtmf_events])
        if transcript is not None:
            self._running.ivr_final(transcript)

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
        if self._core is not None:
            await self._core.aclose(drain=False)

    async def drain(self) -> None:
        await self._running.drain()

    def shutdown(self, *, drain: bool = True) -> None:
        if self._core is not None:
            self._core.shutdown(drain=drain)

    async def __aenter__(self) -> VoiceSession[Userdata_T]:
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
            host=result,
        )
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

    def update_agent(self, agent: Agent) -> None:
        """Hand the conversation to ``agent``; the session and its history stay."""
        self._agent = agent
        if self._core is not None:
            self._core.update_agent(**self._frame(agent))

    def _agent_chat_ctx(self) -> ChatContext:
        """The driving agent's own context, as the core keeps it."""
        return ChatContext([from_core_item(i) for i in self._running.agent_chat_ctx()])

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

    def _frame(self, agent: Agent) -> dict[str, Any]:
        """``agent`` as the core installs it (``update_agent``'s keywords); the core has its
        toolsets set up (``_AgentHooks._core_setup_tools``) and reads their tools then."""
        pool = self._pool(agent)
        frame: dict[str, Any] = {
            "instructions": _instructions(agent.instructions),
            "dynamic_instructions": (
                _DynamicInstructions(self, agent) if agent._instruction_fns else None
            ),
            "agent": agent.id,
            "agent_host": agent,
            "tools_host": pool,
            "tool_runner": ToolRunner(self, agent, pool),
            "turn_handling": agent._turn_handling or None,
            "expressive": agent.expressive,
        }
        if is_given(agent._min_consecutive_speech_delay):
            frame["min_consecutive_speech_delay"] = agent._min_consecutive_speech_delay
        if is_given(agent._use_tts_aligned_transcript):
            frame["use_tts_aligned_transcript"] = agent._use_tts_aligned_transcript
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
        """The nodes ``agent`` overrides, as the core opens them over each provider, each call in
        the agent's binding."""
        overridden = [n for n in _NODES if getattr(type(agent), n) is not getattr(Agent, n)]
        if not overridden:
            return {}
        context = contextvars.copy_context()
        context.run(_BindingContextVar.set, _VoiceBinding(self, agent))
        return dict.fromkeys(overridden, _core.AgentNodes(context, _AgentNodes(agent)))

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

    def _on_speech_created(self, wire: dict[str, Any]) -> None:
        handle = SpeechHandle._of(wire["speech_handle"])
        self.emit(
            "speech_created",
            SpeechCreatedEvent(
                user_initiated=wire["user_initiated"], source=wire["source"], speech_handle=handle
            ),
        )

    def _on_item_added(self, wire: dict[str, Any]) -> None:
        item = from_core_item(wire["item"])
        self._chat_ctx.insert(item)
        if (run := wire["run"]) is not None:
            if item.type == "agent_handoff":
                run._agent_handoff(
                    item=item,
                    old_agent=wire["old_agent_host"],
                    new_agent=wire["new_agent_host"],
                )
            else:
                run._item_added(item)
        if item.type in ("message", "agent_handoff"):
            self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))

    def _on_run_item(self, wire: dict[str, Any]) -> None:
        if (run := wire["run"]) is not None:
            run._item_added(from_core_item(wire["item"]))

    def _on_run_refused(self, wire: dict[str, Any]) -> None:
        wire["run"]._settle(output=None, error=RuntimeError(f"run refused: {wire['cause']}"))

    def _on_run_settled(self, wire: dict[str, Any]) -> None:
        if (run := wire["run"]) is None:
            return
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
        otel_metrics.collect_usage(_AGENT_METRICS.validate_python(wire["metrics"]))

    def _error_record(self, wire: dict[str, Any]) -> tuple[Any, Any]:
        """The reference's payload: the provider's error record, and the provider that failed."""
        record, leg = _ERROR_RECORDS[wire["source"]]
        source = getattr(_VoiceBinding(self, self.current_agent), leg)
        error = record(
            timestamp=time.time(),
            label=getattr(source, "label", wire["label"]),
            error=wire["error"],
            recoverable=wire["recoverable"],
        )
        return error, source

    def _core_error_message(self, wire: dict[str, Any]) -> str:
        return str(self._error_record(wire)[0])

    def _on_error(self, wire: dict[str, Any]) -> None:
        logger.error("session error", extra={"source": wire["source"], "error": str(wire["error"])})
        error, source = self._error_record(wire)
        self.emit("error", ErrorEvent(error=error, source=source))

    def _on_usage(self, wire: dict[str, Any]) -> None:
        usage = [_MODEL_USAGE.validate_python(u) for u in wire["model_usage"]]
        self.emit("session_usage_updated", SessionUsageUpdatedEvent(usage=AgentSessionUsage(usage)))

    def _on_overlapping_speech(self, wire: dict[str, Any]) -> None:
        ev = dict(wire["event"])
        for key, dtype in (("speech_input", np.int16), ("probabilities", np.float32)):
            if ev[key] is not None:
                ev[key] = np.asarray(ev[key], dtype=dtype)
        self.emit("overlapping_speech", OverlappingSpeechEvent(**ev))

    def _on_amd_result(self, wire: dict[str, Any]) -> None:
        if self._amd is not None:
            self._amd._on_prediction(
                AMDPredictionEvent(**{k: v for k, v in wire.items() if k != "type"})
            )

    def _on_close(self, wire: dict[str, Any]) -> None:
        self.emit("close", CloseEvent(reason=CloseReason(wire["reason"]), error=None))

    def emit(self, event: EventTypes, arg: AgentEvent) -> None:  # type: ignore[override]
        self._recorded_events.append(arg)
        super().emit(event, arg)


_AGENT_METRICS: TypeAdapter[AgentMetrics] = TypeAdapter(AgentMetrics)

_ERROR_RECORDS: dict[str, tuple[Any, str]] = {
    "llm_error": (llm.LLMError, "llm"),
    "tts_error": (tts.TTSError, "tts"),
    "stt_error": (stt.STTError, "stt"),
    "realtime_model_error": (llm.RealtimeModelError, "llm"),
}


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

    def __init__(self, session: VoiceSession[Any], agent: Agent) -> None:
        self._session = session
        self._agent = agent

    async def _core_render(self, speech: _core.SpeechHandle) -> str:
        ctx = RunContext(
            function_call=None, session=self._session, speech_handle=SpeechHandle._of(speech)
        )
        return await self._agent._render_dynamic_instructions(ctx)


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

    async def _core_on_enter(self, agent: Agent) -> None:
        with _in_hook("on_enter", self._session, agent):
            await agent._enter(RunContext(function_call=None, session=self._session))

    async def _core_on_exit(self, agent: Agent) -> None:
        with _in_hook("on_exit", self._session, agent):
            await agent.on_exit()

    async def _core_close_toolsets(self, agent: Agent | None) -> None:
        tools = (agent or self._session).tools
        await asyncio.gather(*(t.aclose() for t in tools if isinstance(t, llm.Toolset)))

    async def _core_setup_tools(self, tools: list[llm.Tool | llm.Toolset]) -> list[Any]:
        # a toolset that fails to set up is logged and the agent runs without it
        await asyncio.gather(*(_setup_toolset(t) for t in tools if isinstance(t, llm.Toolset)))
        return tool_specs(tools)

    async def _core_closing(self) -> None:
        if self._session._amd is not None:
            await self._session._amd.aclose()

    async def _core_on_user_turn_completed(
        self,
        agent: Agent,
        chat_ctx: list[_core.ChatItem],
        new_transcript: str,
        transcript_confidence: float,
    ) -> tuple[str | None, list[_core.ChatItem] | None, bool]:
        session = self._session
        turn_ctx = ChatContext([from_core_item(i) for i in chat_ctx])
        message = llm.ChatMessage(
            role="user", content=[new_transcript], transcript_confidence=transcript_confidence
        )
        before = [i.id for i in turn_ctx.items]
        stop = False
        with _in_hook("on_user_turn_completed", session, agent):
            try:
                await agent.on_user_turn_completed(turn_ctx, message)
            except StopResponse:
                stop = True
        changed = [i.id for i in turn_ctx.items] != before
        return message.text_content, to_core_items(turn_ctx.items) if changed else None, stop

    async def _core_function_tools_executed(
        self,
        function_calls: list[_core.ChatItem],
        function_call_outputs: list[_core.ChatItem],
        handoff_required: bool,
    ) -> tuple[list[bool], bool]:
        ev = FunctionToolsExecutedEvent(
            function_calls=[from_core_item(c) for c in function_calls],
            function_call_outputs=[from_core_item(o) for o in function_call_outputs],
        )
        ev._handoff_required = handoff_required
        self._session.emit("function_tools_executed", ev)
        return [o.reply_required for o in ev.function_call_outputs], ev.has_agent_handoff

    async def _core_filler_source(self, source: Callable[[int], Any], step: int) -> Any:
        out = source(step)
        return await out if inspect.isawaitable(out) else out

    def _core_hook_failed(self, hook: str, error: BaseException | None) -> None:
        logger.error(f"error in the {hook} hook", exc_info=error)


_NODES = ("llm_node", "tts_node", "stt_node")


class _AgentNodes:
    """An agent's overridden ``llm_node``, ``tts_node`` and ``stt_node`` as the core calls them
    (``_core.AgentNodes``), in the agent's binding: one completion, one spoken segment, one
    recognition stream. The core feeds what they read and drains what they return."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def llm(self, req: _core.ChatRequest) -> Any:
        chat_ctx, tools = chat_request(req)
        choice = req.tool_choice
        settings = ModelSettings(tool_choice=NOT_GIVEN if choice is None else choice)
        return self._agent.llm_node(chat_ctx, tools, settings)

    def tts(self, text: AsyncIterable[str]) -> Any:
        return self._agent.tts_node(text, ModelSettings())

    def stt(self, audio: AsyncIterable[rtc.AudioFrame]) -> Any:
        return self._agent.stt_node(audio, ModelSettings())


_HANDLERS: dict[str, Callable[[VoiceSession[Any], dict[str, Any]], None]] = {
    "speech_created": VoiceSession._on_speech_created,
    "conversation_item_added": VoiceSession._on_item_added,
    "run_item_added": VoiceSession._on_run_item,
    "run_refused": VoiceSession._on_run_refused,
    "run_settled": VoiceSession._on_run_settled,
    "tool_execution_updated": VoiceSession._on_tool_update,
    "metrics_collected": VoiceSession._on_metrics,
    "session_usage_updated": VoiceSession._on_usage,
    "overlapping_speech": VoiceSession._on_overlapping_speech,
    "error": VoiceSession._on_error,
    "close": VoiceSession._on_close,
    "amd_result": VoiceSession._on_amd_result,
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
    if not resolved["enabled"]:
        return None
    if (model := resolved["model"]) is None:
        llm = config["llm"]  # type: ignore[index]
        return llm if isinstance(llm, LLM) else None
    try:
        from ..inference import LLM as InferenceLLM

        return InferenceLLM.from_model_string(model)
    except Exception:  # noqa: BLE001 — never let detection setup break the session
        logger.warning("keyterm detection: could not create detection LLM %r; skipping", model)
        return None
