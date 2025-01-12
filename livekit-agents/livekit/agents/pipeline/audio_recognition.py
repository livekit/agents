from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterable, Literal, Protocol

from livekit import rtc

from .. import llm, stt, utils, vad
from ..log import logger
from ..utils import aio
from . import io

from ..debug import tracing

if TYPE_CHECKING:
    from .pipeline2 import PipelineAgent


class _TurnDetector(Protocol):
    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    def unlikely_threshold(self) -> float: ...
    def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext) -> float: ...


EventTypes = Literal[
    "start_of_speech",
    "vad_inference_done",
    "end_of_speech",
    "interim_transcript",
    "final_transcript",
    "end_of_turn",
]


class AudioRecognition(rtc.EventEmitter[EventTypes]):
    """
    Audio recognition part of the PipelineAgent.
    The class is always instantiated but no tasks may be running if STT/VAD is disabled

    This class is also responsible for the end of turn detection.
    """

    UNLIKELY_END_OF_TURN_EXTRA_DELAY = 6.0

    def __init__(
        self,
        *,
        agent: PipelineAgent,
        stt: io.STTNode,
        vad: vad.VAD | None,
        turn_detector: _TurnDetector | None,
        min_endpointing_delay: float,
        chat_ctx: llm.ChatContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._audio_input_atask: asyncio.Task[None] | None = None
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._audio_input: io.AudioStream | None = None
        self._min_endpointing_delay = min_endpointing_delay
        self._chat_ctx = chat_ctx
        self._loop = loop
        self._stt = stt
        self._vad = vad
        self._turn_detector = turn_detector

        self._speaking = False
        self._audio_transcript = ""
        self._last_language: str | None = None
        self._vad_graph = tracing.Tracing.add_graph(
            title="vad",
            x_label="time",
            y_label="speech_probability",
            x_type="time",
            y_range=(0, 1),
            max_data_points=int(30 * 30),
        )

        self._stt_ch: aio.Chan[rtc.AudioFrame] | None = None
        self._vad_ch: aio.Chan[rtc.AudioFrame] | None = None

    def start(self) -> None:
        self.update_stt(self._stt)
        self.update_vad(self._vad)

    @property
    def audio_input(self) -> io.AudioStream | None:
        return self._audio_input

    @audio_input.setter
    def audio_input(self, audio_input: io.AudioStream | None) -> None:
        self._audio_input = audio_input
        self.update_stt(self._stt)
        self.update_vad(self._vad)

        if self._audio_input and self._audio_input_atask is None:
            self._audio_input_atask = asyncio.create_task(
                self._audio_input_task(self._audio_input)
            )
        elif self._audio_input_atask is not None:
            self._audio_input_atask.cancel()
            self._audio_input_atask = None

    async def aclose(self) -> None:
        if self._audio_input_atask is not None:
            await aio.gracefully_cancel(self._audio_input_atask)

        if self._stt_atask is not None:
            await aio.gracefully_cancel(self._stt_atask)

        if self._vad_atask is not None:
            await aio.gracefully_cancel(self._vad_atask)

        if self._end_of_turn_task is not None:
            await aio.gracefully_cancel(self._end_of_turn_task)

    def update_stt(self, stt: io.STTNode | None) -> None:
        self._stt = stt
        if self._audio_input and stt:
            self._stt_ch = aio.Chan[rtc.AudioFrame]()
            self._stt_atask = asyncio.create_task(
                self._stt_task(stt, self._stt_ch, self._stt_atask)
            )
        elif self._stt_atask is not None:
            self._stt_atask.cancel()
            self._stt_atask = None
            self._stt_ch = None

    def update_vad(self, vad: vad.VAD | None) -> None:
        self._vad = vad
        if self._audio_input and vad:
            self._vad_ch = aio.Chan[rtc.AudioFrame]()
            self._vad_atask = asyncio.create_task(
                self._vad_task(vad, self._vad_ch, self._vad_atask)
            )
        elif self._vad_atask is not None:
            self._vad_atask.cancel()
            self._vad_atask = None
            self._vad_ch = None

    async def _on_stt_event(self, ev: stt.SpeechEvent) -> None:
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            self.emit("final_transcript", ev)
            transcript = ev.alternatives[0].text
            if not transcript:
                return

            logger.debug(
                "received user transcript",
                extra={"user_transcript": transcript},
            )

            tracing.Tracing.log_event(
                "user transcript",
                {
                    "transcript": transcript,
                    "buffered_transcript": self._audio_transcript,
                },
            )

            self._audio_transcript += f" {transcript}"
            self._audio_transcript = self._audio_transcript.lstrip()

            if not self._speaking:
                self._run_eou_detection(self._agent.chat_ctx)
        elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            self.emit("interim_transcript", ev)

    async def _on_vad_event(self, ev: vad.VADEvent) -> None:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            self.emit("start_of_speech", ev)
            self._speaking = True

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

        elif ev.type == vad.VADEventType.INFERENCE_DONE:
            self._vad_graph.plot(ev.timestamp, ev.probability)
            self.emit("vad_inference_done", ev)

        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            self.emit("end_of_speech", ev)
            self._speaking = False

            if not self._speaking:
                self._run_eou_detection(self._agent.chat_ctx)

    def _run_eou_detection(self, chat_ctx: llm.ChatContext) -> None:
        if not self._audio_transcript:
            return

        chat_ctx = self._chat_ctx.copy()
        chat_ctx.append(role="user", text=self._audio_transcript)
        turn_detector = self._turn_detector

        @utils.log_exceptions(logger=logger)
        async def _bounce_eou_task() -> None:
            await asyncio.sleep(self._min_endpointing_delay)

            if turn_detector is not None and turn_detector.supports_language(
                self._last_language
            ):
                end_of_turn_probability = await turn_detector.predict_end_of_turn(
                    chat_ctx
                )
                tracing.Tracing.log_event(
                    "end of user turn probability",
                    {"probability": end_of_turn_probability},
                )
                unlikely_threshold = turn_detector.unlikely_threshold()
                if end_of_turn_probability > unlikely_threshold:
                    await asyncio.sleep(self.UNLIKELY_END_OF_TURN_EXTRA_DELAY)

            tracing.Tracing.log_event(
                "end of user turn", {"transcript": self._audio_transcript}
            )
            self.emit("end_of_turn", self._audio_transcript)
            self._audio_transcript = ""

        if self._end_of_turn_task is not None:
            self._end_of_turn_task.cancel()

        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task())

    async def _stt_task(
        self,
        stt_node: io.STTNode,
        audio_input: io.AudioStream,
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is not None:
            await aio.gracefully_cancel(task)

        node = stt_node(audio_input)
        if asyncio.iscoroutine(node):
            node = await node

        if node is None:
            return

        if isinstance(node, AsyncIterable):
            async for ev in node:
                assert isinstance(
                    ev, stt.SpeechEvent
                ), "STT node must yield SpeechEvent"
                await self._on_stt_event(ev)

    async def _vad_task(
        self, vad: vad.VAD, audio_input: io.AudioStream, task: asyncio.Task[None] | None
    ) -> None:
        if task is not None:
            await aio.gracefully_cancel(task)

        stream = vad.stream()

        async def _forward() -> None:
            async for frame in audio_input:
                stream.push_frame(frame)

        forward_task = asyncio.create_task(_forward())

        try:
            async for ev in stream:
                await self._on_vad_event(ev)
        finally:
            await stream.aclose()
            await aio.gracefully_cancel(forward_task)

    async def _audio_input_task(self, audio_input: io.AudioStream) -> None:
        async for frame in audio_input:
            if self._stt_ch is not None:
                self._stt_ch.send_nowait(frame)

            if self._vad_ch is not None:
                self._vad_ch.send_nowait(frame)
