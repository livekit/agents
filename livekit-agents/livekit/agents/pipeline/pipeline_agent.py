from __future__ import annotations, print_function

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, Literal, Optional

from livekit import rtc

from .. import debug, llm, multimodal, stt, tts, utils, vad
from ..llm import ChatContext
from ..log import logger
from ..transcription import TranscriptionSyncIO, TranscriptSegment
from ..types import NOT_GIVEN, NotGivenOr
from . import io, room_io
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
    "user_transcript_updated",
    "agent_transcript_updated",
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

        # room io and transcription sync
        self._room_input: Optional[room_io.RoomInput] = None
        self._room_output: Optional[room_io.RoomOutput] = None
        self._agent_tr_sync: Optional[TranscriptionSyncIO] = None
        self._user_transcript_id = utils.shortuuid("SG_")

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
        *,
        room: Optional[rtc.Room] = None,
        room_input_options: NotGivenOr[room_io.RoomInputOptions | None] = NOT_GIVEN,
        room_output_options: NotGivenOr[room_io.RoomOutputOptions | None] = NOT_GIVEN,
    ) -> None:
        """Start the pipeline agent.
        This will create room input and output if the input or output audio is not already set.

        Args:
            room_input_options: Options for the room input, set to None to disable
            room_output_options: Options for the room output, set to None to disable
        """
        if self._started:
            return

        # sanity check
        if (room_input_options or room_output_options) and not room:
            raise ValueError(
                "room must be provided if room_input_options or room_output_options is given"
            )

        if room_input_options and self.input.audio is not None:
            logger.warning(
                "audio input is already set, ignoring room_input_options",
                extra={"room_input_options": room_input_options},
            )
            room_input_options = None

        if room_output_options and self.output.audio is not None:
            logger.warning(
                "audio output is already set, ignoring room_output_options",
                extra={"room_output_options": room_output_options},
            )
            room_output_options = None

        if self.input.audio is None and room and room_input_options is not None:
            # create room input if not already set
            self._room_input = room_io.RoomInput(
                room=room,
                options=room_input_options or room_io.DEFAULT_ROOM_INPUT_OPTIONS,
            )
            await self._room_input.start(self)

        if self.output.audio is None and room and room_output_options is not None:
            # create room output if not already set
            self._room_output = room_io.RoomOutput(
                room=room,
                options=room_output_options or room_io.DEFAULT_ROOM_OUTPUT_OPTIONS,
            )
            await self._room_output.start(self)

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

        if self._agent_tr_sync:
            await self._agent_tr_sync.aclose()
            self._agent_tr_sync = None

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

    def say(
        self,
        source: str | AsyncIterable[rtc.AudioFrame],
        *,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> SpeechHandle:
        if self._activity is None:
            raise ValueError("PipelineAgent isn't running")

        return self._activity.say(source, allow_interruptions=allow_interruptions)

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> SpeechHandle:
        if self._activity is None:
            raise ValueError("PipelineAgent isn't running")

        return self._activity.generate_reply(
            user_input=user_input,
            instructions=instructions,
            allow_interruptions=allow_interruptions,
        )

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

    @utils.log_exceptions(logger=logger)
    def _on_user_transcript(self, ev: stt.SpeechEvent, final: bool) -> None:
        if not ev.alternatives:
            return

        data = ev.alternatives[0]
        self.emit(
            "user_transcript_updated",
            TranscriptSegment(
                id=self._user_transcript_id,
                text=data.text,
                start_time=max(int(data.start_time), 0),
                end_time=max(int(data.end_time), 0),
                final=final,
                is_delta=False,
                language=data.language,
            ),
        )
        if final:
            self._user_transcript_id = utils.shortuuid("SG_")

    def _on_agent_transcript(self, segment: TranscriptSegment) -> None:
        self.emit("agent_transcript_updated", segment)

    @property
    def _audio_sink_with_transcript(self) -> io.AudioSink:
        if not self._output.audio:
            return None

        if self._agent_tr_sync:
            return self._agent_tr_sync.audio_output

        return self._output.audio

    @property
    def _text_sink_with_transcript(self) -> io.TextSink:
        if self._agent_tr_sync:
            return self._agent_tr_sync.text_output

        return self._output.text

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
        if self._output.audio is None:
            if self._agent_tr_sync:
                asyncio.create_task(self._agent_tr_sync.aclose())
                self._agent_tr_sync = None
            return

        if self._agent_tr_sync:
            self._agent_tr_sync.audio_output.set_base_sink(self.output.audio)
        else:
            self._agent_tr_sync = TranscriptionSyncIO(
                self.output.audio, self.output.text
            )
            self._agent_tr_sync.on("transcription_updated", self._on_agent_transcript)

    def _on_text_output_changed(self) -> None:
        if self._agent_tr_sync:
            self._agent_tr_sync.text_output.set_base_sink(self.output.text)

    # ---
