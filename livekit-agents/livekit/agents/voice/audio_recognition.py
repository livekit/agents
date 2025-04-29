from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Protocol

from livekit import rtc

from .. import llm, stt, utils, vad
from ..debug import tracing
from ..log import logger
from ..utils import aio
from . import io
from .agent import ModelSettings


@dataclass
class _EndOfTurnInfo:
    new_transcript: str
    transcription_delay: float
    end_of_utterance_delay: float


class _TurnDetector(Protocol):
    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    def unlikely_threshold(self, language: str | None) -> float | None: ...
    def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext) -> float: ...


class RecognitionHooks(Protocol):
    def on_start_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_vad_inference_done(self, ev: vad.VADEvent) -> None: ...
    def on_end_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None: ...
    def on_final_transcript(self, ev: stt.SpeechEvent) -> None: ...
    async def on_end_of_turn(self, info: _EndOfTurnInfo) -> None: ...

    def retrieve_chat_ctx(self) -> llm.ChatContext: ...


class AudioRecognition:
    def __init__(
        self,
        *,
        hooks: RecognitionHooks,
        stt: io.STTNode | None,
        vad: vad.VAD | None,
        turn_detector: _TurnDetector | None,
        min_endpointing_delay: float,
        max_endpointing_delay: float,
        manual_turn_detection: bool,
    ) -> None:
        self._hooks = hooks
        self._audio_input_atask: asyncio.Task[None] | None = None
        self._commit_user_turn_atask: asyncio.Task[None] | None = None
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._min_endpointing_delay = min_endpointing_delay
        self._max_endpointing_delay = max_endpointing_delay
        self._turn_detector = turn_detector
        self._stt = stt
        self._vad = vad
        self._manual_turn_detection = manual_turn_detection
        self._user_turn_committed = False
        self._sample_rate: float | None = None

        self._speaking = False
        self._last_speaking_time: float = 0
        self._last_final_transcript_time: float = 0
        self._audio_transcript = ""
        self._audio_interim_transcript = ""
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
        self._tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        self.update_stt(self._stt)
        self.update_vad(self._vad)

    def stop(self) -> None:
        self.update_stt(None)
        self.update_vad(None)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._sample_rate = frame.sample_rate
        if self._stt_ch is not None:
            self._stt_ch.send_nowait(frame)

        if self._vad_ch is not None:
            self._vad_ch.send_nowait(frame)

    async def aclose(self) -> None:
        await aio.cancel_and_wait(*self._tasks)
        if self._commit_user_turn_atask is not None:
            await aio.cancel_and_wait(self._commit_user_turn_atask)

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
            task = asyncio.create_task(aio.cancel_and_wait(self._stt_atask))
            task.add_done_callback(lambda _: self._tasks.discard(task))
            self._tasks.add(task)
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
            task = asyncio.create_task(aio.cancel_and_wait(self._vad_atask))
            task.add_done_callback(lambda _: self._tasks.discard(task))
            self._tasks.add(task)
            self._vad_atask = None
            self._vad_ch = None

    def clear_user_turn(self) -> None:
        self._audio_transcript = ""
        self._audio_interim_transcript = ""
        self._user_turn_committed = False

        # reset stt to clear the buffer from previous user turn
        stt = self._stt
        self.update_stt(None)
        self.update_stt(stt)

    def commit_user_turn(self, *, audio_detached: bool) -> None:
        async def _commit_user_turn(delay: float = 0.5):
            if time.time() - self._last_final_transcript_time > delay:
                # flush the stt by pushing silence
                if audio_detached and self._sample_rate:
                    num_samples = int(self._sample_rate * 0.5)
                    self.push_audio(
                        rtc.AudioFrame(
                            b"\x00\x00" * num_samples,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                            samples_per_channel=num_samples,
                        )
                    )

                # wait for the final transcript to be available
                await asyncio.sleep(delay)

            if self._audio_interim_transcript:
                # append interim transcript in case the final transcript is not ready
                self._audio_transcript = (
                    f"{self._audio_transcript} {self._audio_interim_transcript}".strip()
                )
            self._audio_interim_transcript = ""
            chat_ctx = self._hooks.retrieve_chat_ctx().copy()
            self._run_eou_detection(chat_ctx)
            self._user_turn_committed = True

        if self._commit_user_turn_atask is not None:
            self._commit_user_turn_atask.cancel()

        self._commit_user_turn_atask = asyncio.create_task(_commit_user_turn())

    async def _on_stt_event(self, ev: stt.SpeechEvent) -> None:
        if (
            self._manual_turn_detection
            and self._user_turn_committed
            and (
                self._end_of_turn_task is None
                or self._end_of_turn_task.done()
                or ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
            )
        ):
            # ignore stt event if user turn already committed and EOU task is done
            # or it's an interim transcript
            return

        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            self._hooks.on_final_transcript(ev)
            transcript = ev.alternatives[0].text
            self._last_language = ev.alternatives[0].language
            if not transcript:
                return

            logger.debug(
                "received user transcript",
                extra={"user_transcript": transcript, "language": self._last_language},
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
            self._audio_interim_transcript = ""

            if not self._speaking:
                if not self._vad:
                    # vad disabled, use stt timestamp
                    # TODO: this would screw up transcription latency metrics
                    # but we'll live with it for now.
                    # the correct way is to ensure STT fires SpeechEventType.END_OF_SPEECH
                    # and using that timestamp for _last_speaking_time
                    self._last_speaking_time = time.time()

                if not self._manual_turn_detection or self._user_turn_committed:
                    chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                    self._run_eou_detection(chat_ctx)
        elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            self._hooks.on_interim_transcript(ev)
            self._audio_interim_transcript = ev.alternatives[0].text

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

            if not self._manual_turn_detection:
                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                self._run_eou_detection(chat_ctx)

    def _run_eou_detection(self, chat_ctx: llm.ChatContext) -> None:
        if self._stt and not self._audio_transcript and not self._manual_turn_detection:
            # stt enabled but no transcript yet
            return

        chat_ctx = chat_ctx.copy()
        chat_ctx.add_message(role="user", content=self._audio_transcript)
        turn_detector = (
            self._turn_detector
            if self._audio_transcript and not self._manual_turn_detection
            else None  # disable EOU model if manual turn detection enabled
        )

        @utils.log_exceptions(logger=logger)
        async def _bounce_eou_task(last_speaking_time: float) -> None:
            endpointing_delay = self._min_endpointing_delay

            if turn_detector is not None:
                if not turn_detector.supports_language(self._last_language):
                    logger.debug("Turn detector does not support language %s", self._last_language)
                else:
                    end_of_turn_probability = await turn_detector.predict_end_of_turn(chat_ctx)
                    tracing.Tracing.log_event(
                        "end of user turn probability",
                        {"probability": end_of_turn_probability},
                    )
                    unlikely_threshold = turn_detector.unlikely_threshold(self._last_language)
                    if (
                        unlikely_threshold is not None
                        and end_of_turn_probability < unlikely_threshold
                    ):
                        endpointing_delay = self._max_endpointing_delay

            extra_sleep = last_speaking_time + endpointing_delay - time.time()
            await asyncio.sleep(max(extra_sleep, 0))

            tracing.Tracing.log_event("end of user turn", {"transcript": self._audio_transcript})
            await self._hooks.on_end_of_turn(
                _EndOfTurnInfo(
                    new_transcript=self._audio_transcript,
                    transcription_delay=max(
                        self._last_final_transcript_time - last_speaking_time, 0
                    ),
                    end_of_utterance_delay=time.time() - last_speaking_time,
                )
            )
            self._audio_transcript = ""

        if self._end_of_turn_task is not None:
            # TODO(theomonnom): disallow cancel if the extra sleep is done
            self._end_of_turn_task.cancel()

        # copy the last_speaking_time before awaiting (the value can change)
        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task(self._last_speaking_time))

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
