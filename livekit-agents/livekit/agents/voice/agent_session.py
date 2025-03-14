from __future__ import annotations, print_function

import asyncio
import copy
from dataclasses import dataclass
from typing import AsyncIterable, Generic, TypeVar

from livekit import rtc

from .. import debug, llm, stt, tts, utils, vad
from ..cli import cli
from ..llm import ChatContext
from ..log import logger
from ..types import NOT_GIVEN, AgentState, NotGivenOr
from ..utils.misc import is_given
from . import io, room_io
from .agent import Agent
from .audio_recognition import _TurnDetector
from .events import AgentEvent, AgentStateChangedEvent, EventTypes
from .speech_handle import SpeechHandle
from .agent_activity import AgentActivity


@dataclass
class VoiceOptions:
    allow_interruptions: bool
    min_interruption_duration: float
    min_endpointing_delay: float
    max_fnc_steps: int


Userdata_T = TypeVar("Userdata_T")


class AgentSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    def __init__(
        self,
        *,
        instructions: str | None = None,
        agent: NotGivenOr[Agent] = NOT_GIVEN,
        turn_detector: NotGivenOr[_TurnDetector] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS] = NOT_GIVEN,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        allow_interruptions: bool = True,
        min_interruption_duration: float = 0.5,
        min_endpointing_delay: float = 0.5,
        max_fnc_steps: int = 3,
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
            max_fnc_steps=max_fnc_steps,
        )
        self._started = False
        self._turn_detector = turn_detector or None
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

        # used to keep a reference to the room io (not exposed)
        self._room_io: room_io.RoomIO | None = None

        self._agent_task: Agent

        if utils.is_given(agent):
            self._agent_task = agent
        else:
            if instructions is None:
                raise ValueError("instructions must be provided if no agent task is given")

            self._agent_task = Agent(instructions=instructions)

        self._activity: AgentActivity | None = None
        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None

        self._agent_state: AgentState | None = None

    @property
    def userdata(self) -> Userdata_T:
        if self._userdata is None:
            raise ValueError("VoiceAgent userdata is not set")

        return self._userdata

    @userdata.setter
    def userdata(self, value: Userdata_T) -> None:
        self._userdata = value

    @property
    def turn_detector(self) -> _TurnDetector | None:
        return self._turn_detector

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
    def current_speech(self) -> SpeechHandle | None:
        return self._activity.current_speech if self._activity is not None else None

    @property
    def current_task(self) -> Agent:
        return self._agent_task

    async def start(
        self,
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

            self._update_agent_state(AgentState.INITIALIZING)

            if cli.CLI_ARGUMENTS is not None and cli.CLI_ARGUMENTS.console:
                from .chat_cli import ChatCLI

                if (
                    self.input.audio is not None
                    or self.output.audio is not None
                    or self.output.transcription is not None
                ):
                    logger.warning(
                        "agent started with the console subcommand, but input.audio or output.audio "
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
                        "RoomIO transcription output is enabled but output.transcription is already set, ignoring.."
                    )
                    room_output_options.transcription_enabled = False

                self._room_io = room_io.RoomIO(
                    room=room,
                    agent=self,
                    input_options=(room_input_options or room_io.DEFAULT_ROOM_INPUT_OPTIONS),
                    output_options=(room_output_options or room_io.DEFAULT_ROOM_OUTPUT_OPTIONS),
                )
                await self._room_io.start()

            # it is ok to await it directly, there is no previous task to drain
            await self._update_activity_task(self._agent_task)

            # important: no await should be done after this!

            if self.input.audio is not None:
                self._forward_audio_atask = asyncio.create_task(
                    self._forward_audio_task(), name="_forward_audio_task"
                )

            self._started = True
            self._update_agent_state(AgentState.LISTENING)

    async def aclose(self) -> None:
        async with self._lock:
            if not self._started:
                return

            if self._forward_audio_atask is not None:
                await utils.aio.cancel_and_wait(self._forward_audio_atask)

            if self._room_io:
                await self._room_io.aclose()

    @property
    def options(self) -> VoiceOptions:
        return self._opts

    def emit(self, event: EventTypes, ev: AgentEvent) -> None:  # type: ignore
        debug.Tracing.log_event(f'agent.on("{event}")', ev.model_dump())
        return super().emit(event, ev)

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

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
            raise RuntimeError("VoiceAgent isn't running")

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
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> SpeechHandle:
        if self._activity is None:
            raise RuntimeError("VoiceAgent isn't running")

        return self._activity.generate_reply(
            user_input=user_input,
            instructions=instructions,
            allow_interruptions=allow_interruptions,
        )

    def interrupt(self) -> None:
        if self._activity is None:
            raise RuntimeError("VoiceAgent isn't running")

        self._activity.interrupt()

    def update_task(self, task: Agent) -> None:
        self._agent_task = task

        if self._started:
            self._update_activity_atask = asyncio.create_task(
                self._update_activity_task(self._agent_task),
                name="_update_activity_task",
            )

    @utils.log_exceptions(logger=logger)
    async def _update_activity_task(self, task: Agent) -> None:
        async with self._activity_lock:
            if self._activity is not None:
                await self._activity.drain()
                await self._activity.aclose()

            self._activity = AgentActivity(task, self)
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
