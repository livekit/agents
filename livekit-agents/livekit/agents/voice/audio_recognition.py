from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from opentelemetry import trace

from livekit import rtc

from .. import llm, stt, utils, vad
from ..log import logger
from ..telemetry import trace_types, tracer
from ..utils import aio
from . import io
from .agent import ModelSettings

if TYPE_CHECKING:
    from .agent_session import TurnDetectionMode

MIN_LANGUAGE_DETECTION_LENGTH = 5


@dataclass
class _EndOfTurnInfo:
    new_transcript: str
    transcription_delay: float
    end_of_utterance_delay: float
    transcript_confidence: float
    last_speaking_time: float
    _user_turn_span: trace.Span | None = None


@dataclass
class _PreemptiveGenerationInfo:
    new_transcript: str
    transcript_confidence: float


class _TurnDetector(Protocol):
    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    async def unlikely_threshold(self, language: str | None) -> float | None: ...
    async def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext) -> float: ...


class RecognitionHooks(Protocol):
    def on_start_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_vad_inference_done(self, ev: vad.VADEvent) -> None: ...
    def on_end_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None: ...
    def on_final_transcript(self, ev: stt.SpeechEvent) -> None: ...
    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool: ...
    def on_preemptive_generation(self, info: _PreemptiveGenerationInfo) -> None: ...

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
        turn_detection_mode: TurnDetectionMode | None,
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
        self._turn_detection_mode = turn_detection_mode
        self._vad_base_turn_detection = turn_detection_mode in ("vad", None)
        self._user_turn_committed = False  # true if user turn ended but EOU task not done
        self._sample_rate: int | None = None

        self._speaking = False
        self._last_speaking_time: float = 0
        self._last_final_transcript_time: float = 0
        self._final_transcript_received = asyncio.Event()
        self._final_transcript_confidence: list[float] = []
        self._audio_transcript = ""
        self._audio_interim_transcript = ""
        self._last_language: str | None = None

        self._stt_ch: aio.Chan[rtc.AudioFrame] | None = None
        self._vad_ch: aio.Chan[rtc.AudioFrame] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

        self._user_turn_span: trace.Span | None = None

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
        self._final_transcript_confidence = []
        self._user_turn_committed = False

        # reset stt to clear the buffer from previous user turn
        stt = self._stt
        self.update_stt(None)
        self.update_stt(stt)

    def commit_user_turn(self, *, audio_detached: bool, transcript_timeout: float) -> None:
        async def _commit_user_turn() -> None:
            if time.time() - self._last_final_transcript_time > 0.5:
                # if the last final transcript is received more than 0.5s ago
                # append a silence frame to the stt to flush the buffer

                self._final_transcript_received.clear()

                # flush the stt by pushing silence
                if audio_detached and self._sample_rate:
                    num_samples = int(self._sample_rate * 0.2)
                    silence_frame = rtc.AudioFrame(
                        b"\x00\x00" * num_samples,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                        samples_per_channel=num_samples,
                    )
                    for _ in range(5):  # 5 * 0.2s = 1s
                        self.push_audio(silence_frame)

                # wait for the final transcript to be available
                try:
                    await asyncio.wait_for(
                        self._final_transcript_received.wait(),
                        timeout=transcript_timeout,
                    )
                except asyncio.TimeoutError:
                    pass

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

    @property
    def current_transcript(self) -> str:
        """
        Transcript for this turn, including interim transcript if available.
        """
        if self._audio_interim_transcript:
            return self._audio_transcript + " " + self._audio_interim_transcript
        return self._audio_transcript

    async def _on_stt_event(self, ev: stt.SpeechEvent) -> None:
        if (
            self._turn_detection_mode == "manual"
            and self._user_turn_committed
            and (
                self._end_of_turn_task is None
                or self._end_of_turn_task.done()
                or ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
            )
        ):
            # ignore transcript for manual turn detection when user turn already committed
            # and EOU task is done or this is an interim transcript
            return

        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            self._hooks.on_final_transcript(ev)
            transcript = ev.alternatives[0].text
            language = ev.alternatives[0].language
            confidence = ev.alternatives[0].confidence

            if not self._last_language or (
                language and len(transcript) > MIN_LANGUAGE_DETECTION_LENGTH
            ):
                self._last_language = language

            if not transcript:
                return

            logger.debug(
                "received user transcript",
                extra={"user_transcript": transcript, "language": self._last_language},
            )

            self._last_final_transcript_time = time.time()
            self._audio_transcript += f" {transcript}"
            self._audio_transcript = self._audio_transcript.lstrip()
            self._final_transcript_confidence.append(confidence)
            self._audio_interim_transcript = ""
            self._final_transcript_received.set()

            if not self._vad or self._last_speaking_time == 0:
                # vad disabled, use stt timestamp
                # TODO: this would screw up transcription latency metrics
                # but we'll live with it for now.
                # the correct way is to ensure STT fires SpeechEventType.END_OF_SPEECH
                # and using that timestamp for _last_speaking_time
                self._last_speaking_time = time.time()

            if self._vad_base_turn_detection or self._user_turn_committed:
                self._hooks.on_preemptive_generation(
                    _PreemptiveGenerationInfo(
                        new_transcript=self._audio_transcript,
                        transcript_confidence=(
                            sum(self._final_transcript_confidence)
                            / len(self._final_transcript_confidence)
                            if self._final_transcript_confidence
                            else 0
                        ),
                    )
                )

                if not self._speaking:
                    chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                    self._run_eou_detection(chat_ctx)

        elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            self._hooks.on_interim_transcript(ev)
            self._audio_interim_transcript = ev.alternatives[0].text

        elif ev.type == stt.SpeechEventType.END_OF_SPEECH and self._turn_detection_mode == "stt":
            self._user_turn_committed = True
            if not self._speaking:
                # start response after vad fires END_OF_SPEECH to avoid vad interruption
                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                self._run_eou_detection(chat_ctx)

    async def _on_vad_event(self, ev: vad.VADEvent) -> None:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            with trace.use_span(self._ensure_user_turn_span()):
                self._hooks.on_start_of_speech(ev)

            self._speaking = True
            self._last_speaking_time = time.time() - ev.speech_duration

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

        elif ev.type == vad.VADEventType.INFERENCE_DONE:
            self._hooks.on_vad_inference_done(ev)
            self._last_speaking_time = time.time() - ev.silence_duration

        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            with trace.use_span(self._ensure_user_turn_span()):
                self._hooks.on_end_of_speech(ev)

            self._speaking = False
            # when VAD fires END_OF_SPEECH, it already waited for the silence_duration
            self._last_speaking_time = time.time() - ev.silence_duration

            if self._vad_base_turn_detection or (
                self._turn_detection_mode == "stt" and self._user_turn_committed
            ):
                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                self._run_eou_detection(chat_ctx)

    def _run_eou_detection(self, chat_ctx: llm.ChatContext) -> None:
        if self._stt and not self._audio_transcript and self._turn_detection_mode != "manual":
            # stt enabled but no transcript yet
            return

        chat_ctx = chat_ctx.copy()
        chat_ctx.add_message(role="user", content=self._audio_transcript)
        turn_detector = (
            self._turn_detector
            if self._audio_transcript and self._turn_detection_mode != "manual"
            else None  # disable EOU model if manual turn detection enabled
        )

        @utils.log_exceptions(logger=logger)
        async def _bounce_eou_task(last_speaking_time: float) -> None:
            endpointing_delay = self._min_endpointing_delay
            user_turn_span = self._ensure_user_turn_span()
            if turn_detector is not None:
                if not await turn_detector.supports_language(self._last_language):
                    logger.debug("Turn detector does not support language %s", self._last_language)
                else:
                    with (
                        trace.use_span(user_turn_span),
                        tracer.start_as_current_span("eou_detection") as eou_detection_span,
                    ):
                        end_of_turn_probability = await turn_detector.predict_end_of_turn(chat_ctx)
                        unlikely_threshold = await turn_detector.unlikely_threshold(
                            self._last_language
                        )
                        if (
                            unlikely_threshold is not None
                            and end_of_turn_probability < unlikely_threshold
                        ):
                            endpointing_delay = self._max_endpointing_delay

                        eou_detection_span.set_attributes(
                            {
                                trace_types.ATTR_CHAT_CTX: json.dumps(
                                    chat_ctx.to_dict(
                                        exclude_audio=True,
                                        exclude_image=True,
                                        exclude_timestamp=False,
                                    )
                                ),
                                trace_types.ATTR_EOU_PROBABILITY: end_of_turn_probability,
                                trace_types.ATTR_EOU_UNLIKELY_THRESHOLD: unlikely_threshold or 0,
                                trace_types.ATTR_EOU_DELAY: endpointing_delay,
                                trace_types.ATTR_EOU_LANGUAGE: self._last_language or "",
                            }
                        )

            extra_sleep = last_speaking_time + endpointing_delay - time.time()
            await asyncio.sleep(max(extra_sleep, 0))

            confidence_avg = (
                sum(self._final_transcript_confidence) / len(self._final_transcript_confidence)
                if self._final_transcript_confidence
                else 0
            )

            if last_speaking_time <= 0:
                transcription_delay = 0.0
                end_of_utterance_delay = 0.0
            else:
                transcription_delay = max(self._last_final_transcript_time - last_speaking_time, 0)
                end_of_utterance_delay = time.time() - last_speaking_time

            committed = self._hooks.on_end_of_turn(
                _EndOfTurnInfo(
                    new_transcript=self._audio_transcript,
                    transcription_delay=transcription_delay,
                    end_of_utterance_delay=end_of_utterance_delay,
                    transcript_confidence=confidence_avg,
                    last_speaking_time=last_speaking_time,
                )
            )
            if committed:
                user_turn_span.set_attributes(
                    {
                        trace_types.ATTR_USER_TRANSCRIPT: self._audio_transcript,
                        trace_types.ATTR_TRANSCRIPT_CONFIDENCE: confidence_avg,
                        trace_types.ATTR_TRANSCRIPTION_DELAY: transcription_delay,
                        trace_types.ATTR_END_OF_UTTERANCE_DELAY: end_of_utterance_delay,
                    }
                )
                user_turn_span.end()
                self._user_turn_span = None

                # clear the transcript if the user turn was committed
                self._audio_transcript = ""
                self._final_transcript_confidence = []

            self._user_turn_committed = False

        if self._end_of_turn_task is not None:
            # TODO(theomonnom): disallow cancel if the extra sleep is done
            self._end_of_turn_task.cancel()

        # copy the last_speaking_time before awaiting (the value can change)
        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task(self._last_speaking_time))

    @utils.log_exceptions(logger=logger)
    async def _stt_task(
        self,
        stt_node: io.STTNode,
        audio_input: AsyncIterable[rtc.AudioFrame],
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
                assert isinstance(ev, stt.SpeechEvent), (
                    f"STT node must yield SpeechEvent, got: {type(ev)}"
                )
                await self._on_stt_event(ev)

    @utils.log_exceptions(logger=logger)
    async def _vad_task(
        self,
        vad: vad.VAD,
        audio_input: AsyncIterable[rtc.AudioFrame],
        task: asyncio.Task[None] | None,
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

    def _ensure_user_turn_span(self) -> trace.Span:
        if self._user_turn_span and self._user_turn_span.is_recording():
            return self._user_turn_span

        self._user_turn_span = tracer.start_span("user_turn")
        return self._user_turn_span
