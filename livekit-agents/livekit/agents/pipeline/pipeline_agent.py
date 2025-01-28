from __future__ import annotations, print_function

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, Literal, Optional

from livekit import rtc

from .. import debug, llm, utils
from ..log import logger
from . import io
from .agent_task import AgentTask, TaskActivity
from .room_io import RoomInput, RoomInputOptions, RoomOutput
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


class PipelineAgent(rtc.EventEmitter[EventTypes]):
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
        self._started = False

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
        self._current_task: AgentTask = task
        self._active_task: TaskActivity | None = None

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the PipelineAgent

    async def start(
        self,
        room: Optional[rtc.Room] = None,
        room_input_options: Optional[RoomInputOptions] = None,
    ) -> None:
        """Start the pipeline agent.

        Args:
            room (Optional[rtc.Room]): The LiveKit room. If provided and no input/output audio
                is set, automatically configures room audio I/O.
            room_input_options (Optional[RoomInputOptions]): Options for the room input.
        """
        if self._started:
            return

        if room is not None:
            # configure room I/O if not already set
            if self.input.audio is None:
                room_input = RoomInput(room=room, options=room_input_options)
                self._input.audio = room_input.audio
                await room_input.wait_for_participant()

            if self.output.audio is None:
                room_output = RoomOutput(room=room)
                self._output.audio = room_output.audio
                await room_output.start()

        if self.input.audio is not None:
            self._forward_audio_atask = asyncio.create_task(
                self._forward_audio_task(), name="_forward_audio_task"
            )

        self._update_activity_atask = asyncio.create_task(
            self._update_activity_task(self._current_task), name="_update_activity_task"
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

    def update_task(self, task: AgentTask) -> None:
        self._current_task = task

        if self._started:
            self._update_activity_atask = asyncio.create_task(
                self._update_activity_task(self._current_task),
                name="_update_activity_task",
            )

    @utils.log_exceptions(logger=logger)
    async def _update_activity_task(self, task: AgentTask) -> None:
        async with self._lock:
            if self._active_task is not None:
                await self._active_task.drain()
                await self._active_task.aclose()

            self._active_task = task._create_activity(self)
            await self._active_task.start()

    @utils.log_exceptions(logger=logger)
    async def _forward_audio_task(self) -> None:
        audio_input = self.input.audio
        if audio_input is None:
            return

        async for frame in audio_input:
            if self._active_task is not None:
                self._active_task.push_audio(frame)

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
