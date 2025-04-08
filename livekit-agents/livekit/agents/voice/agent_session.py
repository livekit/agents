from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, Union

from livekit import rtc

from .. import debug, llm, stt, tts, utils, vad
from ..cli import cli
from ..llm import ChatContext
from ..log import logger
from ..types import NOT_GIVEN, AgentState, NotGivenOr
from ..utils.misc import is_given
from . import io, room_io
from .agent import Agent
from .agent_activity import AgentActivity
from .audio_recognition import _TurnDetector
from .events import (
    AgentEvent,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    EventTypes,
)
from .speech_handle import SpeechHandle


@dataclass
class VoiceOptions:
    allow_interruptions: bool
    min_interruption_duration: float
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


class AgentSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    def __init__(
        self,
        *,
        turn_detection: NotGivenOr[TurnDetectionMode] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS] = NOT_GIVEN,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        allow_interruptions: bool = True,
        min_interruption_duration: float = 0.5,
        min_endpointing_delay: float = 0.5,
        max_endpointing_delay: float = 6.0,
        max_tool_steps: int = 3,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        # This is the "global" chat_context, it holds the entire conversation history
        self._chat_ctx = ChatContext.empty()
        self._opts = VoiceOptions(
            allow_interruptions=allow_interruptions,
            min_interruption_duration=min_interruption_duration,
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
        self._agent_state: AgentState | None = None

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
            self._update_agent_state(AgentState.INITIALIZING)

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

            elif is_given(room):
                room_input_options = copy.deepcopy(room_input_options)
                room_output_options = copy.deepcopy(room_output_options)

                if (
                    self.input.audio is not None
                    and is_given(room_input_options)
                    and room_input_options.audio_enabled
                ):
                    logger.warning(
                        "RoomIO audio input is enabled but input.audio is already set, ignoring.."
                    )
                    room_input_options.audio_enabled = False

                if (
                    self.output.audio is not None
                    and is_given(room_output_options)
                    and room_output_options.audio_enabled
                ):
                    logger.warning(
                        "RoomIO audio output is enabled but output.audio is already set, ignoring.."
                    )
                    room_output_options.audio_enabled = False

                if (
                    self.output.transcription is not None
                    and is_given(room_output_options)
                    and room_output_options.transcription_enabled
                ):
                    logger.warning(
                        "RoomIO transcription output is enabled but output.transcription is already set, ignoring.."  # noqa: E501
                    )
                    room_output_options.transcription_enabled = False

                self._room_io = room_io.RoomIO(
                    room=room,
                    agent_session=self,
                    input_options=(room_input_options or room_io.DEFAULT_ROOM_INPUT_OPTIONS),
                    output_options=(room_output_options or room_io.DEFAULT_ROOM_OUTPUT_OPTIONS),
                )
                await self._room_io.start()

            else:
                if not self.output.audio and not self.output.transcription:
                    logger.warning(
                        "session starts without output, forgetting to pass `room` to `AgentSession.start()`?"  # noqa: E501
                    )

            # it is ok to await it directly, there is no previous task to drain
            await self._update_activity_task(self._agent)

            # important: no await should be done after this!

            if self.input.audio is not None:
                self._forward_audio_atask = asyncio.create_task(
                    self._forward_audio_task(), name="_forward_audio_task"
                )

            self._started = True
            self._update_agent_state(AgentState.LISTENING)

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

            if self._forward_audio_atask is not None:
                await utils.aio.cancel_and_wait(self._forward_audio_atask)

            if self._room_io:
                await self._room_io.aclose()

        logger.debug("AgentSession closed")

    async def aclose(self) -> None:
        await self._aclose_impl()

    def emit(self, event: EventTypes, ev: AgentEvent) -> None:  # type: ignore
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

        if self._activity.draining:
            if self._next_activity is None:
                raise RuntimeError("AgentSession is closing, cannot use generate_reply()")

            return self._next_activity.generate_reply(
                user_input=user_input,
                instructions=instructions,
                tool_choice=tool_choice,
                allow_interruptions=allow_interruptions,
            )

        return self._activity.generate_reply(
            user_input=user_input,
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
        error: llm.LLMError | stt.STTError | tts.TTSError,
    ) -> None:
        if self._closing_task or error.recoverable:
            return

        async def close_and_drain() -> None:
            logger.info("Closing task started")
            await self.drain()
            if self._activity is not None:
                await self._activity.aclose()
            await self._aclose_impl(error=error)

        def _on_close_done(_: asyncio.Task) -> None:
            logger.info("Closing task done")
            self._closing_task = None

        self._closing_task = asyncio.create_task(close_and_drain())
        self._closing_task.add_done_callback(_on_close_done)

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

    def _update_agent_state(self, state: AgentState) -> None:
        if self._agent_state == state:
            return

        self._agent_state = state
        self.emit("agent_state_changed", AgentStateChangedEvent(state=state))

    def _conversation_item_added(self, message: llm.ChatMessage) -> None:
        self._chat_ctx.items.append(message)
        self.emit("conversation_item_added", ConversationItemAddedEvent(message=message))

    # -- User changed input/output streams/sinks --

    def _on_video_input_changed(self) -> None:
        pass

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
