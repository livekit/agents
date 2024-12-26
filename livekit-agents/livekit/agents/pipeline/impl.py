from __future__ import annotations

import asyncio
from typing import Protocol

from .. import io, llm, stt, utils, vad
from ..utils import aio


class _TurnDetector(Protocol):
    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    def unlikely_threshold(self) -> float: ...
    def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext) -> float: ...


class AudioRecognition:
    """
    Audio recognition part of the PipelineAgent.
    The class is always instantiated but no tasks may be running if STT/VAD is disabled

    This class is also responsible for the end of turn detection.
    """

    UNLIKELY_END_OF_TURN_EXTRA_DELAY = 6.0

    def __init__(
        self,
        *,
        pipeline_agent: "PipelineAgent",
        stt: stt.STT | None,
        vad: vad.VAD | None,
        turn_detector: _TurnDetector | None = None,
        min_endpointing_delay: float,
    ) -> None:
        self._pipeline_agent = weakref.ref(pipeline_agent)
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._audio_input: io.AudioStream | None = None
        self._min_endpointing_delay = min_endpointing_delay

        self._init_stt(stt)
        self._init_vad(vad)
        self._turn_detector = turn_detector

        self._speaking = False
        self._audio_transcript = ""
        self._last_language: str | None = None

    @property
    def audio_input(self) -> io.AudioStream | None:
        return self._audio_input

    @audio_input.setter
    def audio_input(self, audio_input: io.AudioStream | None) -> None:
        self._init_stt(self._stt)
        self._init_vad(self._vad)
        self._audio_input = audio_input

    async def _on_stt_event(self, ev: stt.SpeechEvent) -> None:
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            transcript = ev.alternatives[0].text
            if not transcript:
                return

            logger.debug(
                "received user transcript",
                extra={"user_transcript": new_transcript},
            )

            self._audio_transcript += f" {transcript}"
            self._audio_transcript = self._audio_transcript.lstrip()

            if not self._speaking:
                self._run_eou_detection(pipeline_agent.chat_ctx, self._audio_transcript)

    async def _on_vad_event(self, ev: vad.VADEvent) -> None:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            self._speaking = True

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

        elif ev.tupe == vad.VADEventType.END_OF_SPEECH:
            self._speaking = False

    def _on_end_of_turn(self) -> None:
        # start llm generation
        pass

    async def aclose(self) -> None:
        if self._stt_atask is not None:
            await aio.gracefully_cancel(self._stt_atask)

        if self._vad_atask is not None:
            await aio.gracefully_cancel(self._vad_atask)

        if self._end_of_turn_task is not None:
            await aio.gracefully_cancel(self._end_of_turn_task)

    def _run_eou_detection(
        self, chat_ctx: llm.ChatContext, new_transcript: str
    ) -> None:
        chat_ctx = pipeline_agent.chat_ctx.copy()
        chat_ctx.append(role="user", text=new_transcript)
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
                unlikely_threshold = turn_detector.unlikely_threshold()
                if end_of_turn_probability > unlikely_threshold:
                    await asyncio.sleep(self.UNLIKELY_END_OF_TURN_EXTRA_DELAY)

            self._on_end_of_turn()

        if self._end_of_turn_task is not None:
            self._end_of_turn_task.cancel()

        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task())

    async def _stt_task(
        self, stt: stt.STT, audio_input: io.AudioStream, task: asyncio.Task[None] | None
    ) -> None:
        if task is not None:
            await aio.gracefully_cancel(task)

        stream = stt.stream()

        async def _forward() -> None:
            async for frame in audio_input:
                stream.push_frame(frame)

        forward_task = asyncio.create_task(_forward())

        try:
            async for ev in stream:
                await self._on_stt_event(ev)
        finally:
            await stream.aclose()
            await aio.gracefully_cancel(forward_task)

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

    def init_stt(self, stt: stt.STT, audio_input: io.AudioStream) -> None:
        self._stt = stt
        self._stt_atask = asyncio.create_task(
            self._stt_task(stt, audio_input, self._stt_atask)
        )

    def init_vad(self, vad: vad.VAD, audio_input: io.AudioStream) -> None:
        self._vad = vad
        self._vad_atask = asyncio.create_task(
            self._vad_task(vad, audio_input, self._vad_atask)
        )
