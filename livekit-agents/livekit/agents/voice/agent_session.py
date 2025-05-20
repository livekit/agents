from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, Union, runtime_checkable

from livekit import rtc

from .. import debug, llm, stt, tts, utils, vad
from ..cli import cli
from ..job import get_job_context
from ..llm import ChatContext
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from . import io, room_io
from .agent import Agent
from .agent_activity import AgentActivity
from .audio_recognition import _TurnDetector
from .events import (
    AgentEvent,
    AgentState,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    EventTypes,
    UserState,
    UserStateChangedEvent,
)
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..llm import mcp


@dataclass
class VoiceOptions:
    allow_interruptions: bool
    discard_audio_if_uninterruptible: bool
    min_interruption_duration: float
    min_interruption_words: int
    min_endpointing_delay: float
    max_endpointing_delay: float
    max_tool_steps: int


Userdata_T = TypeVar("Userdata_T")

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
        min_endpointing_delay: float = 0.5,
        max_endpointing_delay: float = 6.0,
        max_tool_steps: int = 3,
        video_sampler: NotGivenOr[_VideoSampler | None] = NOT_GIVEN,
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
                Default ``0.5`` s.
            max_endpointing_delay (float): Maximum time-in-seconds the agent
                will wait before terminating the turn. Default ``6.0`` s.
            max_tool_steps (int): Maximum consecutive tool calls per LLM turn.
                Default ``3``.
            video_sampler (_VideoSampler, optional): Uses
                :class:`VoiceActivityVideoSampler` when *NOT_GIVEN*; that sampler
                captures video at ~1 fps while the user is speaking and ~0.3 fps
                when silent by default.
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
        )
        self._started = False
        self._turn_detection = turn_detection or None
        self._stt = stt or None
        self._vad = vad or None
        self._llm = llm or None
        self._tts = tts or None
        self._mcp_servers = mcp_servers or None

        # configurable IO
        self._input = io.AgentInput(self._on_video_input_changed, self._on_audio_input_changed)
        self._output = io.AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        self._forward_audio_atask: asyncio.Task | None = None
        self._update_activity_atask: asyncio.Task | None = None
        self._activity_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

        # used to keep a reference to the room io
        # this is not exposed, if users want access to it, they can create their own RoomIO
        self._room_io: room_io.RoomIO | None = None

        self._agent: Agent | None = None
        self._activity: AgentActivity | None = None
        self._user_state: UserState = "listening"
        self._agent_state: AgentState = "initializing"

        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None
        self._closing_task: asyncio.Task | None = None

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

            self._agent = agent
            self._update_agent_state("initializing")

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
                await chat_cli.start()

            elif is_given(room) and not self._room_io:
                room_input_options = copy.copy(
                    room_input_options or room_io.DEFAULT_ROOM_INPUT_OPTIONS
                )
                room_output_options = copy.copy(
                    room_output_options or room_io.DEFAULT_ROOM_OUTPUT_OPTIONS
                )

                if self.input.audio is not None and room_input_options.audio_enabled:
                    logger.warning(
                        "RoomIO audio input is enabled but input.audio is already set, ignoring.."
                    )
                    room_input_options.audio_enabled = False

                if self.output.audio is not None and room_output_options.audio_enabled:
                    logger.warning(
                        "RoomIO audio output is enabled but output.audio is already set, ignoring.."
                    )
                    room_output_options.audio_enabled = False

                if (
                    self.output.transcription is not None
                    and room_output_options.transcription_enabled
                ):
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
                await self._room_io.start()

            else:
                if not self._room_io and not self.output.audio and not self.output.transcription:
                    logger.warning(
                        "session starts without output, forgetting to pass `room` to `AgentSession.start()`?"  # noqa: E501
                    )

            try:
                job_ctx = get_job_context()
                job_ctx.add_tracing_callback(self._trace_chat_ctx)
            except RuntimeError:
                pass  # ignore

            # it is ok to await it directly, there is no previous task to drain
            await self._update_activity_task(self._agent)

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

    async def _trace_chat_ctx(self) -> None:
        if self._activity is None:
            return  # can happen at startup

        chat_ctx = self._activity.agent.chat_ctx
        debug.Tracing.store_kv("chat_ctx", chat_ctx.to_dict(exclude_function_call=False))
        debug.Tracing.store_kv("history", self.history.to_dict(exclude_function_call=False))

    async def drain(self) -> None:
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        await self._activity.drain()

    async def _aclose_impl(
        self,
        *,
        error: llm.LLMError | stt.STTError | tts.TTSError | None = None,
    ) -> None:
        async with self._lock:
            if not self._started:
                return

            self.emit("close", CloseEvent(error=error))

            if self._activity is not None:
                await self._activity.aclose()

            if self._forward_audio_atask is not None:
                await utils.aio.cancel_and_wait(self._forward_audio_atask)

            if self._room_io:
                await self._room_io.aclose()

    async def aclose(self) -> None:
        await self._aclose_impl()

    def emit(self, event: EventTypes, ev: AgentEvent) -> None:  # type: ignore
        # don't log VAD metrics as they are too verbose
        if ev.type != "metrics_collected" or ev.metrics.type != "vad_metrics":
            debug.Tracing.log_event(f'agent.on("{event}")', ev.model_dump())

        return super().emit(event, ev)

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

        if self._activity.draining:
            if self._next_activity is None:
                raise RuntimeError("AgentSession is closing, cannot use say()")

            return self._next_activity.say(
                text,
                audio=audio,
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
            )

        return self._activity.say(
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

        if self._activity.draining:
            if self._next_activity is None:
                raise RuntimeError("AgentSession is closing, cannot use generate_reply()")

            return self._next_activity._generate_reply(
                user_message=user_message,
                instructions=instructions,
                tool_choice=tool_choice,
                allow_interruptions=allow_interruptions,
            )

        return self._activity._generate_reply(
            user_message=user_message,
            instructions=instructions,
            tool_choice=tool_choice,
            allow_interruptions=allow_interruptions,
        )

    def interrupt(self) -> asyncio.Future:
        """Interrupt the current speech generation.

        Returns:
            An asyncio.Future that completes when the interruption is fully processed
            and chat context has been updated.

        Example:
            ```python
            await session.interrupt()
            ```
        """
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        return self._activity.interrupt()

    def clear_user_turn(self) -> None:
        # clear the transcription or input audio buffer of the user turn
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        self._activity.clear_user_turn()

    def commit_user_turn(self) -> None:
        # commit the user turn and generate a reply
        if self._activity is None:
            raise RuntimeError("AgentSession isn't running")

        self._activity.commit_user_turn()

    def update_agent(self, agent: Agent) -> None:
        self._agent = agent

        if self._started:
            self._update_activity_atask = asyncio.create_task(
                self._update_activity_task(self._agent), name="_update_activity_task"
            )

    def _on_error(
        self,
        error: llm.LLMError | stt.STTError | tts.TTSError | llm.RealtimeModelError,
    ) -> None:
        if self._closing_task or error.recoverable:
            return

        logger.error("AgentSession is closing due to unrecoverable error", exc_info=error.error)

        async def drain_and_close() -> None:
            await self.drain()
            await self._aclose_impl(error=error)

        def on_close_done(_: asyncio.Task) -> None:
            self._closing_task = None

        self._closing_task = asyncio.create_task(drain_and_close())
        self._closing_task.add_done_callback(on_close_done)

    @utils.log_exceptions(logger=logger)
    async def _update_activity_task(self, task: Agent) -> None:
        async with self._activity_lock:
            self._next_activity = AgentActivity(task, self)

            if self._activity is not None:
                await self._activity.drain()
                await self._activity.aclose()

            self._activity = self._next_activity
            self._next_activity = None
            await self._activity.start()

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

    def _update_agent_state(self, state: AgentState) -> None:
        if self._agent_state == state:
            return

        old_state = self._agent_state
        self._agent_state = state
        self.emit(
            "agent_state_changed", AgentStateChangedEvent(old_state=old_state, new_state=state)
        )

    def _update_user_state(self, state: UserState) -> None:
        if self._user_state == state:
            return

        old_state = self._user_state
        self._user_state = state
        self.emit("user_state_changed", UserStateChangedEvent(old_state=old_state, new_state=state))

    def _conversation_item_added(self, message: llm.ChatMessage) -> None:
        self._chat_ctx.items.append(message)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=message))

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
