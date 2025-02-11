from __future__ import annotations, print_function

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, Literal

from livekit import rtc

from .. import debug, llm, multimodal, stt, tts, utils, vad
from ..llm import ChatContext
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from . import io
from .audio_recognition import _TurnDetector
from .speech_handle import SpeechHandle
from .task import AgentTask
from .task_activity import TaskActivity

EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_message_committed",
    "agent_message_committed",
    "agent_message_interrupted",
]


@dataclass
class PipelineOptions:
    allow_interruptions: bool
    min_interruption_duration: float
    min_endpointing_delay: float
    max_fnc_steps: int


class PipelineAgent(rtc.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        instructions: str | None = None,
        task: NotGivenOr[AgentTask] = NOT_GIVEN,
        turn_detector: NotGivenOr[_TurnDetector] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | multimodal.RealtimeModel] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS] = NOT_GIVEN,
        allow_interruptions: bool = True,
        min_interruption_duration: float = 0.5,
        min_endpointing_delay: float = 0.5,
        max_fnc_steps: int = 5,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        # This is the "global" chat_context, it holds the entire conversation history
        self._chat_ctx = ChatContext.empty()
        self._opts = PipelineOptions(
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
        self._input = io.AgentInput(
            self._on_video_input_changed, self._on_audio_input_changed
        )
        self._output = io.AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        self._forward_audio_atask: asyncio.Task | None = None
        self._update_activity_atask: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # agent tasks
        self._agent_task: AgentTask

        if utils.is_given(task):
            self._agent_task = task
        else:
            if instructions is None:
                raise ValueError(
                    "instructions must be provided if no agent task is given"
                )

            self._agent_task = AgentTask(instructions=instructions)

        self._activity: TaskActivity | None = None

    @property
    def turn_detector(self) -> _TurnDetector | None:
        return self._turn_detector

    @property
    def stt(self) -> stt.STT | None:
        return self._stt

    @property
    def llm(self) -> llm.LLM | multimodal.RealtimeModel | None:
        return self._llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._tts

    @property
    def vad(self) -> vad.VAD | None:
        return self._vad

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the PipelineAgent

    async def start(
        self,
    ) -> None:
        """Start the pipeline agent."""
        if self._started:
            return

        if self.input.audio is not None:
            self._forward_audio_atask = asyncio.create_task(
                self._forward_audio_task(), name="_forward_audio_task"
            )

        self._update_activity_atask = asyncio.create_task(
            self._update_activity_task(self._agent_task), name="_update_activity_task"
        )

        self._started = True

    async def aclose(self) -> None:
        if not self._started:
            return

        if self._forward_audio_atask is not None:
            await utils.aio.cancel_and_wait(self._forward_audio_atask)

    @property
    def options(self) -> PipelineOptions:
        return self._opts

    def emit(self, event: EventTypes, *args) -> None:
        debug.Tracing.log_event(f'agent.on("{event}")')
        return super().emit(event, *args)

    @property
    def input(self) -> io.AgentInput:
        return self._input

    @property
    def output(self) -> io.AgentOutput:
        return self._output

    @property
    def current_speech(self) -> SpeechHandle | None:
        raise NotImplementedError()

    @property
    def current_task(self) -> AgentTask:
        return self._agent_task

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    def update_options(self) -> None:
        pass

    def say(self, text: str | AsyncIterable[str]) -> SpeechHandle:
        raise NotImplementedError()

    def generate_reply(self, user_input: str) -> SpeechHandle:
        raise NotImplementedError()

    def update_task(self, task: AgentTask) -> None:
        self._agent_task = task

        if self._started:
            self._update_activity_atask = asyncio.create_task(
                self._update_activity_task(self._agent_task),
                name="_update_activity_task",
            )

    @utils.log_exceptions(logger=logger)
    async def _update_activity_task(self, task: AgentTask) -> None:
        async with self._lock:
            if self._activity is not None:
                await self._activity.drain()
                await self._activity.aclose()

            self._activity = TaskActivity(task, self)
            await self._activity.start()

    @utils.log_exceptions(logger=logger)
    async def _forward_audio_task(self) -> None:
        audio_input = self.input.audio
        if audio_input is None:
            return

        async for frame in audio_input:
            if self._activity is not None:
                self._activity.push_audio(frame)

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
