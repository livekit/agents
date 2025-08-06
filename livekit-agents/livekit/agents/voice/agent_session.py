from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import AsyncIterable
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from opentelemetry import context as otel_context, trace

from livekit import rtc

from .. import llm, stt, tts, utils, vad
from ..cli import cli
from ..job import get_job_context
from ..llm import ChatContext
from ..log import logger
from ..telemetry import trace_types, tracer
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils.misc import is_given
from . import io, room_io
from .agent import Agent
from .agent_activity import AgentActivity
from .audio_recognition import _TurnDetector
from .events import (
    AgentFalseInterruptionEvent,
    AgentState,
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    EventTypes,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)
from .run_result import RunResult
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..llm import mcp


@dataclass
class SessionConnectOptions:
    stt_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    llm_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    tts_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    max_unrecoverable_errors: int = 3
    """Maximum number of consecutive unrecoverable errors from llm or tts."""


@dataclass
class VoiceOptions:
    allow_interruptions: bool
    discard_audio_if_uninterruptible: bool
    min_interruption_duration: float
    min_interruption_words: int
    min_endpointing_delay: float
    max_endpointing_delay: float
    max_tool_steps: int
    user_away_timeout: float | None
    agent_false_interruption_timeout: float | None
    min_consecutive_speech_delay: float
    use_tts_aligned_transcript: bool
    preemptive_generation: bool


Userdata_T = TypeVar("Userdata_T")
Run_T = TypeVar("Run_T")

TurnDetectionMode = Union[Literal["stt", "vad", "realtime_llm", "manual"], _TurnDetector]
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- _TurnDetector: use the default mode with the provided turn detector

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""


# _RunContextVar = contextvars.ContextVar[RunResult]("agents_run_state")


@runtime_checkable
class _VideoSampler(Protocol):
    def __call__(self, frame: rtc.VideoFrame, session: AgentSession) -> bool: ...


# TODO(theomonnom): Should this be moved to another file?
class VoiceActivityVideoSampler:
    def __init__(self, *, speaking_fps: float = 1.0, silent_fps: float = 0.3):
        if speaking_fps <= 0 or silent_fps <= 0:
            raise ValueError("FPS values must be greater than zero")

        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps
        self._last_sampled_time: float | None = None

    def __call__(self, frame: rtc.VideoFrame, session: AgentSession) -> bool:
        now = time.time()
        is_speaking = session.user_state == "speaking"
        target_fps = self.speaking_fps if is_speaking else self.silent_fps
        min_frame_interval = 1.0 / target_fps

        if self._last_sampled_time is None:
            self._last_sampled_time = now
            return True

        if (now - self._last_sampled_time) >= min_frame_interval:
            self._last_sampled_time = now
            return True

        return False


class AgentSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    def __init__(
        self,
        *,
        turn_detection: NotGivenOr[TurnDetectionMode] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[mcp.MCPServer]] = NOT_GIVEN,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        allow_interruptions: bool = True,
        discard_audio_if_uninterruptible: bool = True,
        min_interruption_duration: float = 0.5,
        min_interruption_words: int = 0,
        min_endpointing_delay: float = 0.4,
        max_endpointing_delay: float = 6.0,
        max_tool_steps: int = 3,
        video_sampler: NotGivenOr[_VideoSampler | None] = NOT_GIVEN,
        user_away_timeout: float | None = 15.0,
        agent_false_interruption_timeout: float | None = 4.0,
        min_consecutive_speech_delay: float = 0.0,
        use_tts_aligned_transcript: bool = False,
        preemptive_generation: bool = False,
        conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """`AgentSession` is the LiveKit Agents runtime that glues together
        media streams, speech/LLM components, and tool orchestration into a
        single real-time voice agent.

        It links audio, video, and text I/O with STT, VAD, TTS, and the LLM;
        handles turn detection, endpointing, interruptions, and multi-step
        tool calls; and exposes everything through event callbacks so you can
        focus on writing function tools and simple hand-offs rather than
        low-level streaming logic.

        Args:
            turn_detection (TurnDetectionMode, optional): Strategy for deciding
                when the user has finished speaking.

                * ``"stt"`` – rely on speech-to-text end-of-utterance cues
                * ``"vad"`` – rely on Voice Activity Detection start/stop cues
                * ``"realtime_llm"`` – use server-side detection from a
                  realtime LLM
                * ``"manual"`` – caller controls turn boundaries explicitly
                * ``_TurnDetector`` instance – plug-in custom detector

                If *NOT_GIVEN*, the session chooses the best available mode in
                priority order ``realtime_llm → vad → stt → manual``; it
                automatically falls back if the necessary model is missing.
            stt (stt.STT, optional): Speech-to-text backend.
            vad (vad.VAD, optional): Voice-activity detector
            llm (llm.LLM | llm.RealtimeModel, optional): LLM or RealtimeModel
            tts (tts.TTS, optional): Text-to-speech engine.
            mcp_servers (list[mcp.MCPServer], optional): List of MCP servers
                providing external tools for the agent to use.
            userdata (Userdata_T, optional): Arbitrary per-session user data.
            allow_interruptions (bool): Whether the user can interrupt the
                agent mid-utterance. Default ``True``.
            discard_audio_if_uninterruptible (bool): When ``True``, buffered
                audio is dropped while the agent is speaking and cannot be
                interrupted. Default ``True``.
            min_interruption_duration (float): Minimum speech length (s) to
                register as an interruption. Default ``0.5`` s.
            min_interruption_words (int): Minimum number of words to consider
                an interruption, only used if stt enabled. Default ``0``.
            min_endpointing_delay (float): Minimum time-in-seconds the agent
                must wait after a potential end-of-utterance signal (from VAD
                or an EOU model) before it declares the user’s turn complete.
                Default ``0.4`` s.
            max_endpointing_delay (float): Maximum time-in-seconds the agent
                will wait before terminating the turn. Default ``6.0`` s.
            max_tool_steps (int): Maximum consecutive tool calls per LLM turn.
                Default ``3``.
            video_sampler (_VideoSampler, optional): Uses
                :class:`VoiceActivityVideoSampler` when *NOT_GIVEN*; that sampler
                captures video at ~1 fps while the user is speaking and ~0.3 fps
                when silent by default.
            user_away_timeout (float, optional): If set, set the user state as
                "away" after this amount of time after user and agent are silent.
                Default ``15.0`` s, set to ``None`` to disable.
            agent_false_interruption_timeout (float, optional): If set, emit an
                `agent_false_interruption` event after this amount of time if
                the user is silent and no user transcript is detected after
                the interruption. Set to ``None`` to disable. Default ``4.0`` s.
            min_consecutive_speech_delay (float, optional): The minimum delay between
                consecutive speech. Default ``0.0`` s.
            use_tts_aligned_transcript (bool, optional): Whether to use TTS-aligned
                transcript as the input of the ``transcription_node``. Only applies
                if ``TTS.capabilities.aligned_transcript`` is ``True`` or ``streaming``
                is ``False``.
            preemptive_generation (bool): Whether to use preemptive generation.
                Default ``False``.
            preemptive_generation (bool):
                Whether to speculatively begin LLM and TTS requests before an end-of-turn is
                detected. When True, the agent sends inference calls as soon as a user
                transcript is received rather than waiting for a definitive turn boundary. This
                can reduce response latency by overlapping model inference with user audio,
                but may incur extra compute if the user interrupts or revises mid-utterance.
                Defaults to ``False``.
            conn_options (SessionConnectOptions, optional): Connection options for
                stt, llm, and tts.
            loop (asyncio.AbstractEventLoop, optional): Event loop to bind the
                session to. Falls back to :pyfunc:`asyncio.get_event_loop()`.
        """
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        if not is_given(video_sampler):
            video_sampler = VoiceActivityVideoSampler(speaking_fps=1.0, silent_fps=0.3)

        self._video_sampler = video_sampler

        # This is the "global" chat_context, it holds the entire conversation history
        self._chat_ctx = ChatContext.empty()
        self._opts = VoiceOptions(
            allow_interruptions=allow_interruptions,
            discard_audio_if_uninterruptible=discard_audio_if_uninterruptible,
            min_interruption_duration=min_interruption_duration,
            min_interruption_words=min_interruption_words,
            min_endpointing_delay=min_endpointing_delay,
            max_endpointing_delay=max_endpointing_delay,
            max_tool_steps=max_tool_steps,
            user_away_timeout=user_away_timeout,
            agent_false_interruption_timeout=agent_false_interruption_timeout,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            preemptive_generation=preemptive_generation,
            use_tts_aligned_transcript=use_tts_aligned_transcript,
        )
        self._conn_options = conn_options or SessionConnectOptions()
        self._started = False
        self._turn_detection = turn_detection or None
        self._stt = stt or None
        self._vad = vad or None
        self._llm = llm or None
        self._tts = tts or None
        self._mcp_servers = mcp_servers or None

        # unrecoverable error counts, reset after agent speaking
        self._llm_error_counts = 0
        self._tts_error_counts = 0

        # configurable IO
        self._input = io.AgentInput(self._on_video_input_changed, self._on_audio_input_changed)
        self._output = io.AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        self._forward_audio_atask: asyncio.Task[None] | None = None
        self._forward_video_atask: asyncio.Task[None] | None = None
        self._update_activity_atask: asyncio.Task[None] | None = None
        self._activity_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

        # used to keep a reference to the room io
        # this is not exposed, if users want access to it, they can create their own RoomIO
        self._room_io: room_io.RoomIO | None = None

        self._agent: Agent | None = None
        self._activity: AgentActivity | None = None
        self._next_activity: AgentActivity | None = None
        self._user_state: UserState = "listening"
        self._agent_state: AgentState = "initializing"
        self._user_away_timer: asyncio.TimerHandle | None = None

        # used to emit the agent false interruption event
        self._false_interruption_timer: asyncio.TimerHandle | None = None
        self._false_interrupted_event: AgentFalseInterruptionEvent | None = None

        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None
        self._closing_task: asyncio.Task[None] | None = None
        self._job_context_cb_registered: bool = False

        self._global_run_state: RunResult | None = None

        # trace
        self._user_speaking_span: trace.Span | None = None
        self._agent_speaking_span: trace.Span | None = None
        self._session_span: trace.Span | None = None
        self._root_span_context: otel_context.Context | None = None

    @property
    def userdata(self) -> Userdata_T:
        if self._userdata is None:
            raise ValueError("VoiceAgent userdata is not set")

        return self._userdata

    @userdata.setter
    def userdata(self, value: Userdata_T) -> None:
        self._userdata = value

    @property
    def turn_detection(self) -> TurnDetectionMode | None:
        return self._turn_detection

    @property
    def mcp_servers(self) -> list[mcp.MCPServer] | None:
        return self._mcp_servers

    @property
    def input(self) -> io.AgentInput:
        return self._input

    @property
    def output(self) -> io.AgentOutput:
        return self._output

    @property
    def options(self) -> VoiceOptions:
        return self._opts

    @property
    def conn_options(self) -> SessionConnectOptions:
        return self._conn_options

    @property
    def history(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def current_speech(self) -> SpeechHandle | None:
        return self._activity.current_speech if self._activity is not None else None

    @property
    def user_state(self) -> UserState:
        return self._user_state

    @property
    def agent_state(self) -> AgentState:
        return self._agent_state

    @property
    def current_agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("VoiceAgent isn't running")

        return self._agent

    def run(self, *, user_input: str, output_type: type[Run_T] | None = None) -> RunResult[Run_T]:
        if self._global_run_state is not None and not self._global_run_state.done():
            raise RuntimeError("nested runs are not supported")

        run_state = RunResult(user_input=user_input, output_type=output_type)
        self._global_run_state = run_state
        self.generate_reply(user_input=user_input)
        return run_state

    @tracer.start_as_current_span("agent_session", end_on_exit=False)
    async def start(
        self,
        agent: Agent,
        *,
        room: NotGivenOr[rtc.Room] = NOT_GIVEN,
        room_input_options: NotGivenOr[room_io.RoomInputOptions] = NOT_GIVEN,
        room_output_options: NotGivenOr[room_io.RoomOutputOptions] = NOT_GIVEN,
    ) -> None:
        """Start the voice agent.

        Create a default RoomIO if the input or output audio is not already set.
        If the console flag is provided, start a ChatCLI.

        Args:
            room: The room to use for input and output
            room_input_options: Options for the room input
            room_output_options: Options for the room output
        """
        async with self._lock:
            if self._started:
                return

            self._root_span_context = otel_context.get_current()
            self._session_span = current_span = trace.get_current_span()
            current_span = trace.get_current_span()
            current_span.set_attribute(trace_types.ATTR_AGENT_LABEL, agent.label)
            current_span.set_attribute(
                trace_types.ATTR_SESSION_OPTIONS, json.dumps(asdict(self._opts))
            )

            self._agent = agent
            self._update_agent_state("initializing")

            tasks: list[asyncio.Task[None]] = []
            if cli.CLI_ARGUMENTS is not None and cli.CLI_ARGUMENTS.console:
                from .chat_cli import ChatCLI

                if (
                    self.input.audio is not None
                    or self.output.audio is not None
                    or self.output.transcription is not None
                ):
                    logger.warning(
                        "agent started with the console subcommand, but input.audio or output.audio "  # noqa: E501
                        "or output.transcription is already set, overriding.."
                    )

                chat_cli = ChatCLI(self)
                tasks.append(asyncio.create_task(chat_cli.start(), name="_chat_cli_start"))

            elif is_given(room) and not self._room_io:
                room_input_options = copy.copy(
                    room_input_options or room_io.DEFAULT_ROOM_INPUT_OPTIONS
                )
                room_output_options = copy.copy(
                    room_output_options or room_io.DEFAULT_ROOM_OUTPUT_OPTIONS
                )

                if self.input.audio is not None:
                    if room_input_options.audio_enabled:
                        logger.warning(
                            "RoomIO audio input is enabled but input.audio is already set, ignoring.."  # noqa: E501
                        )
                    room_input_options.audio_enabled = False

                if self.output.audio is not None:
                    if room_output_options.audio_enabled:
                        logger.warning(
                            "RoomIO audio output is enabled but output.audio is already set, ignoring.."  # noqa: E501
                        )
                    room_output_options.audio_enabled = False

                if self.output.transcription is not None:
                    if room_output_options.transcription_enabled:
                        logger.warning(
                            "RoomIO transcription output is enabled but output.transcription is already set, ignoring.."  # noqa: E501
                        )
                    room_output_options.transcription_enabled = False

                self._room_io = room_io.RoomIO(
                    room=room,
                    agent_session=self,
                    input_options=room_input_options,
                    output_options=room_output_options,
                )
                tasks.append(asyncio.create_task(self._room_io.start(), name="_room_io_start"))

            # session can be restarted, register the callbacks only once
            try:
                job_ctx = get_job_context()
                current_span.set_attribute(trace_types.ATTR_ROOM_NAME, job_ctx.room.name)
                current_span.set_attribute(trace_types.ATTR_JOB_ID, job_ctx.job.id)
                current_span.set_attribute(trace_types.ATTR_AGENT_NAME, job_ctx.job.agent_name)
                if self._room_io:
                    # automatically connect to the room when room io is used
                    tasks.append(asyncio.create_task(job_ctx.connect(), name="_job_ctx_connect"))

                if not self._job_context_cb_registered:
                    job_ctx.add_shutdown_callback(
                        lambda: self._aclose_impl(reason=CloseReason.JOB_SHUTDOWN)
                    )
                    self._job_context_cb_registered = True
            except RuntimeError:
                pass  # ignore

            # it is ok to await it directly, there is no previous task to drain
            tasks.append(
                asyncio.create_task(self._update_activity(self._agent, wait_on_enter=False))
            )

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.cancel_and_wait(*tasks)

            # important: no await should be done after this!

            if self.input.audio is not None:
                self._forward_audio_atask = asyncio.create_task(
                    self._forward_audio_task(), name="_forward_audio_task"
                )

            if self.input.video is not None:
                self._forward_video_atask = asyncio.create_task(
                    self._forward_video_task(), name="_forward_video_task"
                )

            self._started = True
            self._update_agent_state("listening")
            if self._room_io and self._room_io.subscribed_fut:

                def on_room_io_subscribed(_: asyncio.Future[None]) -> None:
                    if self._user_state == "listening" and self._agent_state == "listening":
                        self._set_user_away_timer()

                self._room_io.subscribed_fut.add_done_callback(on_room_io_subscribed)

            # log used IO
            def _collect_source(
                inp: io.AudioInput | io.VideoInput | None,
            ) -> list[io.AudioInput | io.VideoInput]:
                return [] if inp is None else [inp] + _collect_source(inp.source)

            def _collect_chain(
                out: io.TextOutput | io.VideoOutput | io.AudioOutput | None,
            ) -> list[io.VideoOutput | io.AudioOutput | io.TextOutput]:
                return [] if out is None else [out] + _collect_chain(out.next_in_chain)

            audio_input = _collect_source(self.input.audio)[::-1]
            video_input = _collect_source(self.input.video)[::-1]

            audio_output = _collect_chain(self.output.audio)
            video_output = _collect_chain(self.output.video)
            transcript_output = _collect_chain(self.output.transcription)

            logger.debug(
                "using audio io: %s -> `AgentSession` -> %s",
                " -> ".join([f"`{out.label}`" for out in audio_input]) or "(none)",
                " -> ".join([f"`{out.label}`" for out in audio_output]) or "(none)",
            )

            logger.debug(
                "using transcript io: `AgentSession` -> %s",
                " -> ".join([f"`{out.label}`" for out in transcript_output]) or "(none)",
            )

            if video_input or video_output:
                logger.debug(
                    "using video io: %s > `AgentSession` > %s",
                    " -> ".join([f"`{out.label}`" for out in video_input]) or "(none)",
                    " -> ".join([f"`{out.label}`" for out in video_output]) or "(none)",
                )

    async def drain(self) -> None:
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        await self._activity.drain()

    def _close_soon(
        self,
        *,
        reason: CloseReason,
        drain: bool = False,
        error: llm.LLMError | stt.STTError | tts.TTSError | llm.RealtimeModelError | None = None,
    ) -> None:
        if self._closing_task:
            return

        self._closing_task = asyncio.create_task(
            self._aclose_impl(error=error, drain=drain, reason=reason)
        )

    @utils.log_exceptions(logger=logger)
    async def _aclose_impl(
        self,
        *,
        reason: CloseReason,
        drain: bool = False,
        error: llm.LLMError | stt.STTError | tts.TTSError | llm.RealtimeModelError | None = None,
    ) -> None:
        if self._root_span_context:
            # make `activity.drain` and `on_exit` under the root span
            otel_context.attach(self._root_span_context)

        async with self._lock:
            if not self._started:
                return

            if self._activity is not None:
                if not drain:
                    try:
                        await self._activity.interrupt()
                    except RuntimeError:
                        # uninterruptible speech
                        # TODO(long): force interrupt or wait for it to finish?
                        # it might be an audio played from the error callback
                        pass

                await self._activity.drain()

                # wait any uninterruptible speech to finish
                if self._activity.current_speech:
                    await self._activity.current_speech

                # detach the inputs and outputs
                self.input.audio = None
                self.input.video = None
                self.output.audio = None
                self.output.transcription = None

                await self._activity.aclose()
                self._activity = None

            if self._agent_speaking_span:
                self._agent_speaking_span.end()
                self._agent_speaking_span = None

            if self._user_speaking_span:
                self._user_speaking_span.end()
                self._user_speaking_span = None

            if self._forward_audio_atask is not None:
                await utils.aio.cancel_and_wait(self._forward_audio_atask)

            if self._room_io:
                await self._room_io.aclose()
                self._room_io = None

            self._started = False
            if self._session_span:
                self._session_span.end()
                self._session_span = None
            self.emit("close", CloseEvent(error=error, reason=reason))

            self._cancel_user_away_timer()
            self._cancel_agent_false_interruption()
            self._user_state = "listening"
            self._agent_state = "initializing"
            self._llm_error_counts = 0
            self._tts_error_counts = 0
            self._root_span_context = None

        logger.debug("session closed", extra={"reason": reason.value, "error": error})

    async def aclose(self) -> None:
        await self._aclose_impl(reason=CloseReason.USER_INITIATED)

    def update_options(self) -> None:
        pass

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        run_state = self._global_run_state
        if self._activity.scheduling_paused:
            if self._next_activity is None:
                raise RuntimeError("AgentSession is closing, cannot use say()")

            handle = self._next_activity.say(
                text,
                audio=audio,
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
            )
            if run_state:
                run_state._watch_handle(handle)

            return handle

        handle = self._activity.say(
            text,
            audio=audio,
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
        )
        if run_state:
            run_state._watch_handle(handle)

        return handle

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> SpeechHandle:
        """Generate a reply for the agent to speak to the user.

        Args:
            user_input (NotGivenOr[str], optional): The user's input that may influence the reply,
                such as answering a question.
            instructions (NotGivenOr[str], optional): Additional instructions for generating the reply.
            tool_choice (NotGivenOr[llm.ToolChoice], optional): Specifies the external tool to use when
                generating the reply. If generate_reply is invoked within a function_tool, defaults to "none".
            allow_interruptions (NotGivenOr[bool], optional): Indicates whether the user can interrupt this speech.

        Returns:
            SpeechHandle: A handle to the generated reply.
        """  # noqa: E501
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        user_message = (
            llm.ChatMessage(role="user", content=[user_input])
            if is_given(user_input)
            else NOT_GIVEN
        )

        run_state = self._global_run_state
        if self._activity.scheduling_paused:
            if self._next_activity is None:
                raise RuntimeError("AgentSession is closing, cannot use generate_reply()")

            handle = self._next_activity._generate_reply(
                user_message=user_message,
                instructions=instructions,
                tool_choice=tool_choice,
                allow_interruptions=allow_interruptions,
            )
            if run_state:
                run_state._watch_handle(handle)

            return handle

        handle = self._activity._generate_reply(
            user_message=user_message,
            instructions=instructions,
            tool_choice=tool_choice,
            allow_interruptions=allow_interruptions,
        )
        if run_state:
            run_state._watch_handle(handle)

        return handle

    def interrupt(self) -> asyncio.Future[None]:
        """Interrupt the current speech generation.

        Returns:
            An asyncio.Future that completes when the interruption is fully processed
            and chat context has been updated.
        """
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        return self._activity.interrupt()

    def clear_user_turn(self) -> None:
        # clear the transcription or input audio buffer of the user turn
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        self._activity.clear_user_turn()

    def commit_user_turn(self, *, transcript_timeout: float = 2.0) -> None:
        """Commit the user turn and generate a reply.

        Args:
            transcript_timeout (float, optional): The timeout for the final transcript
                to be received after committing the user turn.
                Increase this value if the STT is slow to respond.

        Raises:
            RuntimeError: If the AgentSession isn't running.
        """
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        self._activity.commit_user_turn(transcript_timeout=transcript_timeout)

    def update_agent(self, agent: Agent) -> None:
        self._agent = agent

        if self._started:
            self._update_activity_atask = task = asyncio.create_task(
                self._update_activity_task(self._agent), name="_update_activity_task"
            )
            run_state = self._global_run_state
            if run_state:
                # don't mark the RunResult as done, if there is currently an agent transition happening.  # noqa: E501
                # (used to make sure we're correctly adding the AgentHandoffResult before completion)  # noqa: E501
                run_state._watch_handle(task)

    async def _update_activity(
        self,
        agent: Agent,
        *,
        previous_activity: Literal["close", "pause"] = "close",
        new_activity: Literal["start", "resume"] = "start",
        blocked_tasks: list[asyncio.Task] | None = None,
        wait_on_enter: bool = True,
    ) -> None:
        async with self._activity_lock:
            # _update_activity is called directy sometimes, update for redundancy
            self._agent = agent

            if new_activity == "start":
                if agent._activity is not None:
                    raise RuntimeError("cannot start agent: an activity is already running")

                self._next_activity = AgentActivity(agent, self)
            elif new_activity == "resume":
                if agent._activity is None:
                    raise RuntimeError("cannot resume agent: no existing active activity to resume")

                self._next_activity = agent._activity

            previous_activity_v = self._activity
            if self._activity is not None:
                if previous_activity == "close":
                    await self._activity.drain()
                    await self._activity.aclose()
                elif previous_activity == "pause":
                    await self._activity.pause(blocked_tasks=blocked_tasks or [])

            self._activity = self._next_activity
            self._next_activity = None

            run_state = self._global_run_state
            if run_state:
                run_state._agent_handoff(
                    old_agent=previous_activity_v.agent if previous_activity_v else None,
                    new_agent=self._activity.agent,
                )

            if new_activity == "start":
                await self._activity.start()
            elif new_activity == "resume":
                await self._activity.resume()

            if wait_on_enter:
                assert self._activity._on_enter_task is not None
                await asyncio.shield(self._activity._on_enter_task)

    @utils.log_exceptions(logger=logger)
    async def _update_activity_task(self, task: Agent) -> None:
        if self._root_span_context is not None:
            # restore the root span context so on_exit, on_enter, and future turns
            # are direct children of the root span, not nested under a tool call.
            otel_context.attach(self._root_span_context)

        await self._update_activity(task)

    def _on_error(
        self,
        error: llm.LLMError | stt.STTError | tts.TTSError | llm.RealtimeModelError,
    ) -> None:
        if self._closing_task or error.recoverable:
            return

        if error.type == "llm_error":
            self._llm_error_counts += 1
            if self._llm_error_counts <= self.conn_options.max_unrecoverable_errors:
                return
        elif error.type == "tts_error":
            self._tts_error_counts += 1
            if self._tts_error_counts <= self.conn_options.max_unrecoverable_errors:
                return

        logger.error("AgentSession is closing due to unrecoverable error", exc_info=error.error)

        def on_close_done(_: asyncio.Task[None]) -> None:
            self._closing_task = None

        self._closing_task = asyncio.create_task(
            self._aclose_impl(error=error, reason=CloseReason.ERROR)
        )
        self._closing_task.add_done_callback(on_close_done)

    @utils.log_exceptions(logger=logger)
    async def _forward_audio_task(self) -> None:
        audio_input = self.input.audio
        if audio_input is None:
            return

        async for frame in audio_input:
            if self._activity is not None:
                self._activity.push_audio(frame)

    @utils.log_exceptions(logger=logger)
    async def _forward_video_task(self) -> None:
        video_input = self.input.video
        if video_input is None:
            return

        async for frame in video_input:
            if self._activity is not None:
                if self._video_sampler is not None and not self._video_sampler(frame, self):
                    continue  # ignore this frame

                self._activity.push_video(frame)

    def _set_user_away_timer(self) -> None:
        self._cancel_user_away_timer()
        if self._opts.user_away_timeout is None:
            return

        if (
            (room_io := self._room_io)
            and room_io.subscribed_fut
            and not room_io.subscribed_fut.done()
        ):
            # skip the timer before user join the room
            return

        self._user_away_timer = self._loop.call_later(
            self._opts.user_away_timeout, self._update_user_state, "away"
        )

    def _cancel_user_away_timer(self) -> None:
        if self._user_away_timer is not None:
            self._user_away_timer.cancel()
            self._user_away_timer = None

    def _update_agent_state(self, state: AgentState) -> None:
        if self._agent_state == state:
            return

        if state == "speaking":
            self._llm_error_counts = 0
            self._tts_error_counts = 0

            if self._agent_speaking_span is None:
                self._agent_speaking_span = tracer.start_span("agent_speaking")
                self._agent_speaking_span.set_attribute(trace_types.ATTR_START_TIME, time.time())
        elif self._agent_speaking_span is not None:
            self._agent_speaking_span.set_attribute(trace_types.ATTR_END_TIME, time.time())
            self._agent_speaking_span.end()
            self._agent_speaking_span = None

        if state == "listening" and self._user_state == "listening":
            self._set_user_away_timer()
        else:
            self._cancel_user_away_timer()

        if state != "listening":
            self._cancel_agent_false_interruption()

        old_state = self._agent_state
        self._agent_state = state
        self.emit(
            "agent_state_changed",
            AgentStateChangedEvent(old_state=old_state, new_state=state),
        )

    def _update_user_state(
        self, state: UserState, *, last_speaking_time: float | None = None
    ) -> None:
        if self._user_state == state:
            return

        if state == "speaking" and self._user_speaking_span is None:
            self._user_speaking_span = tracer.start_span("user_speaking")
            self._user_speaking_span.set_attribute(trace_types.ATTR_START_TIME, time.time())
        elif self._user_speaking_span is not None:
            end_time = last_speaking_time or time.time()
            self._user_speaking_span.set_attribute(trace_types.ATTR_END_TIME, end_time)
            self._user_speaking_span.end(end_time=int(end_time * 1_000_000_000))  # nanoseconds
            self._user_speaking_span = None

        if state == "listening" and self._agent_state == "listening":
            self._set_user_away_timer()
        else:
            self._cancel_user_away_timer()

        # pause the false interruption timer if user is speaking and recreate it after user stops
        if state == "speaking" and self._false_interruption_timer:
            ev = self._false_interrupted_event
            self._cancel_agent_false_interruption()
            self._false_interrupted_event = ev
        elif state == "listening" and self._false_interrupted_event:
            self._schedule_agent_false_interruption(self._false_interrupted_event)

        old_state = self._user_state
        self._user_state = state
        self.emit(
            "user_state_changed",
            UserStateChangedEvent(old_state=old_state, new_state=state),
        )

    def _user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        self.emit("user_input_transcribed", ev)
        if ev.is_final:
            # fully cancel the false interruption event if user transcript arrives
            self._cancel_agent_false_interruption()

    def _conversation_item_added(self, message: llm.ChatMessage) -> None:
        self._chat_ctx.insert(message)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=message))

    def _schedule_agent_false_interruption(self, ev: AgentFalseInterruptionEvent) -> None:
        if self._opts.agent_false_interruption_timeout is None:
            return

        def _emit_event() -> None:
            if self._agent_state != "listening" or self._user_state != "listening":
                return

            self.emit("agent_false_interruption", ev)
            self._false_interruption_timer = None

        self._cancel_agent_false_interruption()
        self._false_interruption_timer = self._loop.call_later(
            self._opts.agent_false_interruption_timeout, _emit_event
        )
        self._false_interrupted_event = ev

    def _cancel_agent_false_interruption(self) -> None:
        if self._false_interruption_timer is not None:
            self._false_interruption_timer.cancel()
            self._false_interruption_timer = None
        self._false_interrupted_event = None

    # move them to the end to avoid shadowing the same named modules for mypy
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

    # -- User changed input/output streams/sinks --

    def _on_video_input_changed(self) -> None:
        if not self._started:
            return

        if self._forward_video_atask is not None:
            self._forward_video_atask.cancel()

        self._forward_video_atask = asyncio.create_task(
            self._forward_video_task(), name="_forward_video_task"
        )

    def _on_audio_input_changed(self) -> None:
        if not self._started:
            return

        if self._forward_audio_atask is not None:
            self._forward_audio_atask.cancel()

        self._forward_audio_atask = asyncio.create_task(
            self._forward_audio_task(), name="_forward_audio_task"
        )

    def _on_video_output_changed(self) -> None:
        pass

    def _on_audio_output_changed(self) -> None:
        pass

    def _on_text_output_changed(self) -> None:
        pass

    # ---

    async def __aenter__(self) -> AgentSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
