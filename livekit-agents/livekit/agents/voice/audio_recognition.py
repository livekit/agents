from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from typing import Literal, Protocol

from livekit import rtc

from .. import llm, metrics, stt, utils, vad
from ..debug import tracing
from ..log import logger
from ..utils import aio
from . import io
from .agent import ModelSettings


class _TurnDetector(Protocol):
    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    def unlikely_threshold(self) -> float: ...
    def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext) -> float: ...


class RecognitionHooks(Protocol):
    def on_start_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_vad_inference_done(self, ev: vad.VADEvent) -> None: ...
    def on_end_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None: ...
    def on_final_transcript(self, ev: stt.SpeechEvent) -> None: ...
    async def on_end_of_turn(self, new_transcript: str) -> None: ...

    def retrieve_chat_ctx(self) -> llm.ChatContext: ...


class AudioRecognition(rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(
        self,
        *,
        hooks: RecognitionHooks,
        stt: io.STTNode | None,
        vad: vad.VAD | None,
        turn_detector: _TurnDetector | None,
        min_endpointing_delay: float,
        max_endpointing_delay: float,
    ) -> None:
        super().__init__()
        self._hooks = hooks
        self._audio_input_atask: asyncio.Task[None] | None = None
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._min_endpointing_delay = min_endpointing_delay
        self._max_endpointing_delay = max_endpointing_delay
        self._turn_detector = turn_detector
        self._stt = stt
        self._vad = vad

        self._speaking = False
        self._last_speaking_time: float = 0
        self._last_final_transcript_time: float = 0
        self._audio_transcript = ""
        self._last_language: str | None = None
        self._audio_stream_start_time: float | None = None
        self._last_transcript_end_time: float = 0
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

    def stop(self) -> None:
        self.update_stt(None)
        self.update_vad(None)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_stream_start_time is None:
            self._audio_stream_start_time = time.time()

        if self._stt_ch is not None:
            self._stt_ch.send_nowait(frame)

        if self._vad_ch is not None:
            self._vad_ch.send_nowait(frame)

    async def aclose(self) -> None:
        if self._stt_atask is not None:
            await aio.cancel_and_wait(self._stt_atask)

        if self._vad_atask is not None:
            await aio.cancel_and_wait(self._vad_atask)

        if self._end_of_turn_task is not None:
            await aio.cancel_and_wait(self._end_of_turn_task)

    def update_stt(self, stt: io.STTNode | None) -> None:
        self._stt = stt
        if stt:
            self._audio_stream_start_time = None  # Reset when STT is updated
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
        if vad:
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
            self._hooks.on_final_transcript(ev)
            transcript = ev.alternatives[0].text
            self._last_language = ev.alternatives[0].language
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

            self._last_final_transcript_time = time.time()
            self._audio_transcript += f" {transcript}"
            self._audio_transcript = self._audio_transcript.lstrip()

            if hasattr(ev.alternatives[0], "end_time") and ev.alternatives[0].end_time > 0:
                self._last_transcript_end_time = ev.alternatives[0].end_time

            if not self._speaking:
                if not self._vad:
                    # vad disabled, use stt timestamp
                    # TODO: this would screw up transcription latency metrics
                    # but we'll live with it for now.
                    # the correct way is to ensure STT fires SpeechEventType.END_OF_SPEECH
                    # and using that timestamp for _last_speaking_time
                    self._last_speaking_time = time.time()

                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                self._run_eou_detection(chat_ctx)
        elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            self._hooks.on_interim_transcript(ev)

    async def _on_vad_event(self, ev: vad.VADEvent) -> None:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            self._hooks.on_start_of_speech(ev)
            self._speaking = True

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

        elif ev.type == vad.VADEventType.INFERENCE_DONE:
            self._vad_graph.plot(ev.timestamp, ev.probability)
            self._hooks.on_vad_inference_done(ev)

        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            self._hooks.on_end_of_speech(ev)
            self._speaking = False
            # when VAD fires END_OF_SPEECH, it already waited for the silence_duration
            self._last_speaking_time = time.time() - ev.silence_duration

            chat_ctx = self._hooks.retrieve_chat_ctx().copy()
            self._run_eou_detection(chat_ctx)

    def _run_eou_detection(self, chat_ctx: llm.ChatContext) -> None:
        if self._stt and not self._audio_transcript:
            # stt enabled but no transcript yet
            return

        chat_ctx = chat_ctx.copy()
        chat_ctx.add_message(role="user", content=self._audio_transcript)
        turn_detector = self._turn_detector if self._audio_transcript else None

        @utils.log_exceptions(logger=logger)
        async def _bounce_eou_task() -> None:
            endpointing_delay = self._min_endpointing_delay

            if turn_detector is not None and turn_detector.supports_language(self._last_language):
                end_of_turn_probability = await turn_detector.predict_end_of_turn(chat_ctx)
                tracing.Tracing.log_event(
                    "end of user turn probability",
                    {"probability": end_of_turn_probability},
                )
                unlikely_threshold = turn_detector.unlikely_threshold()
                if end_of_turn_probability < unlikely_threshold:
                    endpointing_delay = self._max_endpointing_delay

            await asyncio.sleep(
                max(
                    self._last_speaking_time + endpointing_delay - time.time(),
                    0,
                )
            )

            tracing.Tracing.log_event("end of user turn", {"transcript": self._audio_transcript})

            transcription_delay = 0.0
            if self._audio_stream_start_time and self._last_transcript_end_time > 0:
                actual_speech_end_time = (
                    self._audio_stream_start_time + self._last_transcript_end_time
                )
            else:
                actual_speech_end_time = self._last_speaking_time

            transcription_delay = max(self._last_final_transcript_time - actual_speech_end_time, 0)
            end_of_utterance_delay = max(time.time() - actual_speech_end_time, 0)

            eou_metrics = metrics.EOUMetrics(
                timestamp=time.time(),
                end_of_utterance_delay=end_of_utterance_delay,
                transcription_delay=transcription_delay,
            )
            self.emit("metrics_collected", eou_metrics)
            await self._hooks.on_end_of_turn(self._audio_transcript)
            self._audio_transcript = ""

        if self._end_of_turn_task is not None:
            self._end_of_turn_task.cancel()

        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task())

    @utils.log_exceptions(logger=logger)
    async def _stt_task(
        self,
        stt_node: io.STTNode,
        audio_input: io.AudioInput,
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is not None:
            await aio.cancel_and_wait(task)

        node = stt_node(audio_input, ModelSettings())
        if asyncio.iscoroutine(node):
            node = await node

        if node is None:
            return

        if isinstance(node, AsyncIterable):
            async for ev in node:
                assert isinstance(ev, stt.SpeechEvent), "STT node must yield SpeechEvent"
                await self._on_stt_event(ev)

    @utils.log_exceptions(logger=logger)
    async def _vad_task(
        self, vad: vad.VAD, audio_input: io.AudioInput, task: asyncio.Task[None] | None
    ) -> None:
        if task is not None:
            await aio.cancel_and_wait(task)

        stream = vad.stream()

        @utils.log_exceptions(logger=logger)
        async def _forward() -> None:
            async for frame in audio_input:
                stream.push_frame(frame)

        forward_task = asyncio.create_task(_forward())

        try:
            async for ev in stream:
                await self._on_vad_event(ev)
        finally:
            await aio.cancel_and_wait(forward_task)
            await stream.aclose()
