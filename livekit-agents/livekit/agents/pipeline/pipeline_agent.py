from __future__ import annotations, print_function

import asyncio
import contextlib
import heapq
from dataclasses import dataclass
from typing import (
    AsyncIterable,
    Literal,
    Tuple,
)

from livekit import rtc

from .. import debug, llm, utils
from ..llm import ChatContext, FunctionContext
from ..log import logger
from . import io
from .agent_task import AgentTask
from .generation import (
    _TTSGenerationData,
    do_llm_inference,
    do_tts_inference,
)


class SpeechHandle:
    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, step_index: int
    ) -> None:
        self._id = speech_id
        self._step_index = step_index
        self._allow_interruptions = allow_interruptions
        self._interrupt_fut = asyncio.Future()
        self._done_fut = asyncio.Future()
        self._play_fut = asyncio.Future()
        self._playout_done_fut = asyncio.Future()

    @staticmethod
    def create(allow_interruptions: bool = True, step_index: int = 0) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    def play(self) -> None:
        self._play_fut.set_result(None)

    def done(self) -> bool:
        return self._done_fut.done()

    def interrupt(self) -> None:
        if not self._allow_interruptions:
            raise ValueError("This generation handle does not allow interruptions")

        if self.done():
            return

        self._done_fut.set_result(None)
        self._interrupt_fut.set_result(None)

    async def wait_for_playout(self) -> None:
        await asyncio.shield(self._playout_done_fut)

    def _mark_playout_done(self) -> None:
        self._playout_done_fut.set_result(None)

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._done_fut.set_result(None)


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
        task: AgentTask,
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

        #         self._audio_recognition.on("end_of_turn", self._on_audio_end_of_turn)
        #         self._audio_recognition.on("start_of_speech", self._on_start_of_speech)
        #         self._audio_recognition.on("end_of_speech", self._on_end_of_speech)
        #         self._audio_recognition.on("vad_inference_done", self._on_vad_inference_done)

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
        self._speech_tasks = []
        self._speech_scheduler_atask: asyncio.Task | None = None

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the PipelineAgent

    def start(self) -> None:
        self._speech_scheduler_atask = asyncio.create_task(
            self._speech_scheduler_task(), name="_playout_scheduler_task"
        )

    async def aclose(self) -> None:
        if self._speech_scheduler_atask is not None:
            await utils.aio.gracefully_cancel(self._speech_scheduler_atask)

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
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    def update_options(self) -> None:
        pass

    def say(self, text: str | AsyncIterable[str]) -> SpeechHandle:
        raise NotImplementedError()

    def generate_reply(self, user_input: str) -> SpeechHandle:
        if self._current_speech is not None and not self._current_speech.interrupted:
            raise ValueError("another reply is already in progress")

        debug.Tracing.log_event("generate_reply", {"user_input": user_input})

        # TODO(theomonnom): move to _generate_pipeline_reply_task
        self._chat_ctx.items.append(
            llm.ChatItem.create(
                [llm.ChatMessage.create(role="user", content=user_input)]
            )
        )

        handle = SpeechHandle.create(allow_interruptions=self._opts.allow_interruptions)
        task = asyncio.create_task(
            self._generate_pipeline_reply_task(
                handle=handle,
                chat_ctx=self._chat_ctx,
                fnc_ctx=self._fnc_ctx,
            ),
            name="_generate_pipeline_reply",
        )
        self._schedule_speech(handle, task, self.SPEECH_PRIORITY_NORMAL)
        return handle

    # -- Main generation task --

    def _schedule_speech(
        self, speech: SpeechHandle, task: asyncio.Task, priority: int
    ) -> None:
        self._speech_tasks.append(task)
        task.add_done_callback(lambda _: self._speech_tasks.remove(task))

        heapq.heappush(self._speech_q, (priority, speech))
        self._speech_q_changed.set()

    @utils.log_exceptions(logger=logger)
    async def _speech_scheduler_task(self) -> None:
        while True:
            await self._speech_q_changed.wait()

            while self._speech_q:
                _, speech = heapq.heappop(self._speech_q)
                self._current_speech = speech
                speech.play()
                await speech.wait_for_playout()
                self._current_speech = None

            self._speech_q_changed.clear()

    # -- Audio recognition --

    # def _on_audio_end_of_turn(self, new_transcript: str) -> None:
    #     # When the audio recognition detects the end of a user turn:
    #     #  - check if there is no current generation happening
    #     #  - cancel the current generation if it allows interruptions (otherwise skip this current
    #     #  turn)
    #     #  - generate a reply to the user input

    #     if self._current_speech is not None:
    #         if self._current_speech.allow_interruptions:
    #             logger.warning(
    #                 "skipping user input, current speech generation cannot be interrupted",
    #                 extra={"user_input": new_transcript},
    #             )
    #             return

    #         debug.Tracing.log_event(
    #             "speech interrupted, new user turn detected",
    #             {"speech_id": self._current_speech.id},
    #         )
    #         self._current_speech.interrupt()

    #     self.generate_reply(new_transcript)

    # def _on_vad_inference_done(self, ev: vad.VADEvent) -> None:
    #     if ev.speech_duration > self._opts.min_interruption_duration:
    #         if (
    #             self._current_speech is not None
    #             and not self._current_speech.interrupted
    #             and self._current_speech.allow_interruptions
    #         ):
    #             debug.Tracing.log_event(
    #                 "speech interrupted by vad",
    #                 {"speech_id": self._current_speech.id},
    #             )
    #             self._current_speech.interrupt()

    # def _on_start_of_speech(self, _: vad.VADEvent) -> None:
    #     self.emit("user_started_speaking", events.UserStartedSpeakingEvent())

    # def _on_end_of_speech(self, _: vad.VADEvent) -> None:
    #     self.emit("user_stopped_speaking", events.UserStoppedSpeakingEvent())

    # ---

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
