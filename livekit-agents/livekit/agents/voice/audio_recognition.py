from __future__ import annotations

import asyncio
from typing import AsyncIterable, Protocol

from livekit import rtc

from .. import llm, stt, utils, vad
from ..debug import tracing
from ..log import logger
from ..utils import aio
from . import io


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


class AudioRecognition:
    UNLIKELY_END_OF_TURN_EXTRA_DELAY = 6.0

    def __init__(
        self,
        *,
        hooks: RecognitionHooks,
        stt: io.STTNode | None,
        vad: vad.VAD | None,
        turn_detector: _TurnDetector | None,
        min_endpointing_delay: float,
    ) -> None:
        self._hooks = hooks
        self._audio_input_atask: asyncio.Task[None] | None = None
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._min_endpointing_delay = min_endpointing_delay
        self._turn_detector = turn_detector
        self._stt = stt
        self._vad = vad

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

    def stop(self) -> None:
        self.update_stt(None)
        self.update_vad(None)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
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

            if not self._speaking:
                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                self._run_eou_detection(chat_ctx)

    def _run_eou_detection(self, chat_ctx: llm.ChatContext) -> None:
        if not self._audio_transcript:
            return

        # TODO
        # chat_ctx = self._agent._chat_ctx.copy()
        # chat_ctx.append(role="user", text=self._audio_transcript)
        turn_detector = self._turn_detector

        @utils.log_exceptions(logger=logger)
        async def _bounce_eou_task() -> None:
            await asyncio.sleep(self._min_endpointing_delay)

            if turn_detector is not None and turn_detector.supports_language(self._last_language):
                end_of_turn_probability = await turn_detector.predict_end_of_turn(chat_ctx)
                tracing.Tracing.log_event(
                    "end of user turn probability",
                    {"probability": end_of_turn_probability},
                )
                unlikely_threshold = turn_detector.unlikely_threshold()
                if end_of_turn_probability > unlikely_threshold:
                    await asyncio.sleep(self.UNLIKELY_END_OF_TURN_EXTRA_DELAY)

            tracing.Tracing.log_event("end of user turn", {"transcript": self._audio_transcript})
            await self._hooks.on_end_of_turn(self._audio_transcript)
            self._audio_transcript = ""

        if self._end_of_turn_task is not None:
            self._end_of_turn_task.cancel()

        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task())

    @utils.log_exceptions(logger=logger)
    async def _stt_task(
        self,
        stt_node: io.STTNode,
        audio_input: io.AudioStream,
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is not None:
            await aio.cancel_and_wait(task)

        node = stt_node(audio_input)
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
        self, vad: vad.VAD, audio_input: io.AudioStream, task: asyncio.Task[None] | None
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
            await stream.aclose()
            await aio.cancel_and_wait(forward_task)
