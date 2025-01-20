from __future__ import annotations, print_function

import asyncio
import heapq
from dataclasses import dataclass
from typing import (
    AsyncIterable,
    Literal,
    Tuple,
)

from livekit import rtc

from .. import debug, llm, utils
from ..log import logger
from . import io
from .agent_task import ActiveTask, AgentTask
from .speech_handle import SpeechHandle

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


class AgentContext:
    def __init__(self) -> None:
        pass


class PipelineAgent(rtc.EventEmitter[EventTypes]):
    SPEECH_PRIORITY_LOW = 0
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = 5
    """Every speech generates by the PipelineAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for important messages that should be played before others."""

    def __init__(
        self,
        *,
        task: AgentTask,  # TODO(theomonnom): move this, pretty sure there will be complaints about this lol
        allow_interruptions: bool = True,
        min_interruption_duration: float = 0.5,
        min_endpointing_delay: float = 0.5,
        max_fnc_steps: int = 5,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        # This is the "global" chat_context, it holds the entire conversation history
        self._chat_ctx = task.chat_ctx.copy()
        self._opts = PipelineOptions(
            allow_interruptions=allow_interruptions,
            min_interruption_duration=min_interruption_duration,
            min_endpointing_delay=min_endpointing_delay,
            max_fnc_steps=max_fnc_steps,
        )

        # configurable IO
        self._input = io.AgentInput(
            self._on_video_input_changed, self._on_audio_input_changed
        )
        self._output = io.AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        # speech state
        self._current_speech: SpeechHandle | None = None
        self._speech_q: list[Tuple[int, SpeechHandle]] = []
        self._speech_q_changed = asyncio.Event()

        self._main_atask: asyncio.Task | None = None

        # agent tasks
        self._current_task: AgentTask = task
        self._active_task: ActiveTask

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the PipelineAgent

    def start(self) -> None:
        self._main_atask = asyncio.create_task(self._main_task(), name="_main_task")

    async def aclose(self) -> None:
        if self._main_atask is not None:
            await utils.aio.gracefully_cancel(self._main_atask)

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
        return self._current_speech

    @property
    def current_task(self) -> AgentTask:
        return self._current_task

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    def update_options(self) -> None:
        pass

    def say(self, text: str | AsyncIterable[str]) -> SpeechHandle:
        raise NotImplementedError()

    def generate_reply(self, user_input: str) -> SpeechHandle:
        raise NotImplementedError()

    def _update_task(self, task: AgentTask) -> None:
        pass

    def _schedule_speech(self, speech: SpeechHandle, priority: int) -> None:
        heapq.heappush(self._speech_q, (priority, speech))
        self._speech_q_changed.set()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        @utils.log_exceptions(logger=logger)
        async def _speech_scheduling_task() -> None:
            while True:
                await self._speech_q_changed.wait()

                while self._speech_q:
                    _, speech = heapq.heappop(self._speech_q)
                    self._current_speech = speech
                    speech._authorize_playout()
                    await speech.wait_for_playout()
                    self._current_speech = None

                self._speech_q_changed.clear()

        tasks = [
            asyncio.create_task(
                _speech_scheduling_task(), name="_speech_scheduling_task"
            )
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    # -- User changed input/output streams/sinks --

    def _on_video_input_changed(self) -> None:
        pass

    def _on_audio_input_changed(self) -> None:
        pass

    def _on_video_output_changed(self) -> None:
        pass

    def _on_audio_output_changed(self) -> None:
        pass

    def _on_text_output_changed(self) -> None:
        pass

    # ---
