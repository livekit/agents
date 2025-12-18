from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, Union, cast

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan

from livekit import rtc

from .. import inference, llm, stt, utils, vad
from ..inference.bargein import BargeinStreamBase
from ..log import logger
from ..stt import SpeechEvent
from ..telemetry import trace_types, tracer
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import aio, is_given
from . import io
from ._utils import _set_participant_attributes
from .agent import ModelSettings

if TYPE_CHECKING:
    from .agent_session import AgentSession

MIN_LANGUAGE_DETECTION_LENGTH = 5


@dataclass
class _EndOfTurnInfo:
    new_transcript: str
    transcript_confidence: float

    # metrics report
    started_speaking_at: float | None
    stopped_speaking_at: float | None
    transcription_delay: float | None
    end_of_turn_delay: float | None


@dataclass
class _PreemptiveGenerationInfo:
    new_transcript: str
    transcript_confidence: float
    started_speaking_at: float | None


class _TurnDetector(Protocol):
    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    async def unlikely_threshold(self, language: str | None) -> float | None: ...
    async def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = None
    ) -> float: ...


TurnDetectionMode = Union[Literal["stt", "vad", "realtime_llm", "manual"], _TurnDetector]
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- _TurnDetector: use the default mode with the provided turn detector

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""


class RecognitionHooks(Protocol):
    def on_bargein(self, ev: inference.BargeinEvent) -> None: ...
    def on_start_of_speech(self, ev: vad.VADEvent | None) -> None: ...
    def on_vad_inference_done(self, ev: vad.VADEvent) -> None: ...
    def on_end_of_speech(self, ev: vad.VADEvent | None) -> None: ...
    def on_interim_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None) -> None: ...
    def on_final_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None = None) -> None: ...
    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool: ...
    def on_preemptive_generation(self, info: _PreemptiveGenerationInfo) -> None: ...

    def retrieve_chat_ctx(self) -> llm.ChatContext: ...


class AudioRecognition:
    def __init__(
        self,
        session: AgentSession,
        *,
        hooks: RecognitionHooks,
        stt: io.STTNode | None,
        vad: vad.VAD | None,
        bargein_detection: inference.BargeinDetector | None,
        turn_detection: TurnDetectionMode | None,
        min_endpointing_delay: float,
        max_endpointing_delay: float,
    ) -> None:
        self._session = session
        self._hooks = hooks
        self._audio_input_atask: asyncio.Task[None] | None = None
        self._commit_user_turn_atask: asyncio.Task[None] | None = None
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._min_endpointing_delay = min_endpointing_delay
        self._max_endpointing_delay = max_endpointing_delay
        self._turn_detector = turn_detection if not isinstance(turn_detection, str) else None
        self._stt = stt
        self._vad = vad
        self._turn_detection_mode = turn_detection if isinstance(turn_detection, str) else None
        self._vad_base_turn_detection = self._turn_detection_mode in ("vad", None)
        self._user_turn_committed = False  # true if user turn ended but EOU task not done

        self._sample_rate: int | None = None
        self._speaking = False

        self._last_final_transcript_time: float | None = None
        self._last_speaking_time: float | None = None
        self._speech_start_time: float | None = None

        # used for manual commit_user_turn
        self._final_transcript_received = asyncio.Event()
        self._final_transcript_confidence: list[float] = []
        self._audio_transcript = ""
        self._audio_interim_transcript = ""
        # used for STTs that support preflight mode, so it could start preemptive generation earlier
        self._audio_preflight_transcript = ""
        self._last_language: str | None = None

        self._stt_ch: aio.Chan[rtc.AudioFrame] | None = None
        self._vad_ch: aio.Chan[rtc.AudioFrame] | None = None

        self._tasks: set[asyncio.Task[Any]] = set()

        # used for barge-in detection
        self._bargein_atask: asyncio.Task[None] | None = None
        self._bargein_detection = bargein_detection
        self._bargein_ch: (
            aio.Chan[
                rtc.AudioFrame
                | BargeinStreamBase._AgentSpeechStartedSentinel
                | BargeinStreamBase._AgentSpeechEndedSentinel
                | BargeinStreamBase._OverlapSpeechStartedSentinel
                | BargeinStreamBase._OverlapSpeechEndedSentinel
            ]
            | None
        ) = None
        self._input_started_at: float | None = None
        self._ignore_user_transcript_until: NotGivenOr[float] = NOT_GIVEN
        self._transcript_buffer: deque[SpeechEvent] = deque()
        self._barge_in_enabled: bool = bargein_detection is not None
        self._agent_speaking: bool = False

        self._user_turn_span: trace.Span | None = None
        self._closing = asyncio.Event()

    def update_options(
        self,
        *,
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        enable_barge_in: NotGivenOr[bool] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    ) -> None:
        if is_given(min_endpointing_delay):
            self._min_endpointing_delay = min_endpointing_delay
        if is_given(max_endpointing_delay):
            self._max_endpointing_delay = max_endpointing_delay

        if is_given(enable_barge_in):
            self._barge_in_enabled = enable_barge_in

        if is_given(turn_detection):
            turn_detection = cast(Optional[TurnDetectionMode], turn_detection)
            self._turn_detector = turn_detection if not isinstance(turn_detection, str) else None

            mode = turn_detection if isinstance(turn_detection, str) else None
            if self._turn_detection_mode != mode:
                previous_mode = self._turn_detection_mode
                self._turn_detection_mode = mode
                self._vad_base_turn_detection = self._turn_detection_mode in ("vad", None)

                if self._turn_detection_mode == "manual" or previous_mode == "manual":
                    if self._end_of_turn_task:
                        if not self._end_of_turn_task.done():
                            self._end_of_turn_task.cancel()
                    self._end_of_turn_task = None
                    self._user_turn_committed = False

    def start(self) -> None:
        self.update_stt(self._stt)
        self.update_vad(self._vad)
        self.update_bargein_detection(self._bargein_detection)

    def stop(self) -> None:
        self.update_stt(None)
        self.update_vad(None)
        self.update_bargein_detection(None)

    def on_start_of_agent_speech(self) -> None:
        self._agent_speaking = True

        if not self._barge_in_enabled or not self._bargein_ch or self._bargein_ch.closed:
            return
        self._bargein_ch.send_nowait(BargeinStreamBase._AgentSpeechStartedSentinel())

    def on_start_of_overlap_speech(
        self,
        speech_duration: float | None = None,
        user_speaking_span: trace.Span | None = None,
    ) -> None:
        """Start barge-in inference when agent is speaking and overlap speech starts."""
        if not self._barge_in_enabled or not self._bargein_ch or self._bargein_ch.closed:
            return
        if self._agent_speaking:
            self._bargein_ch.send_nowait(
                BargeinStreamBase._OverlapSpeechStartedSentinel(speech_duration, user_speaking_span)
            )

    def on_end_of_overlap_speech(self, user_speaking_span: trace.Span | None = None) -> None:
        """End barge-in inference when agent is speaking and overlap speech ends."""
        if not self._barge_in_enabled or not self._bargein_ch or self._bargein_ch.closed:
            return

        # Only set is_bargein=false if not already set (avoid overwriting true from bargein detection)
        if user_speaking_span and user_speaking_span.is_recording():
            if isinstance(user_speaking_span, ReadableSpan):
                if (
                    user_speaking_span.attributes
                    and user_speaking_span.attributes.get(trace_types.ATTR_IS_BARGEIN) is None
                ):
                    user_speaking_span.set_attribute(trace_types.ATTR_IS_BARGEIN, "false")
            else:
                user_speaking_span.set_attribute(trace_types.ATTR_IS_BARGEIN, "false")

        self._bargein_ch.send_nowait(BargeinStreamBase._OverlapSpeechEndedSentinel())

    def on_end_of_agent_speech(self, *, ignore_user_transcript_until: float) -> None:
        if not self._barge_in_enabled or not self._bargein_ch or self._bargein_ch.closed:
            self._agent_speaking = False
            return

        self._bargein_ch.send_nowait(BargeinStreamBase._AgentSpeechEndedSentinel())

        if self._agent_speaking:
            # no barge-in is detected, end the inference (idempotent)
            if not is_given(self._ignore_user_transcript_until):
                self.on_end_of_overlap_speech()
            self._ignore_user_transcript_until = (
                ignore_user_transcript_until
                if not is_given(self._ignore_user_transcript_until)
                else min(ignore_user_transcript_until, self._ignore_user_transcript_until)
            )

            # flush held transcripts if possible
            task = asyncio.create_task(self._flush_held_transcripts())
            task.add_done_callback(lambda _: self._tasks.discard(task))
            self._tasks.add(task)

        self._agent_speaking = False

    async def _flush_held_transcripts(self) -> None:
        """Flush held transcripts whose *end time* is after the ignore_user_transcript_until timestamp.

        If the event has no timestamps, we assume it is the same as the next valid event.
        """
        if not self._barge_in_enabled:
            return
        if not is_given(self._ignore_user_transcript_until):
            return
        if not self._transcript_buffer:
            return

        if not self._input_started_at:
            self._transcript_buffer.clear()
            self._ignore_user_transcript_until = NOT_GIVEN
            return

        emit_from_index: int | None = None
        should_flush = False
        for i, ev in enumerate(self._transcript_buffer):
            if not ev.alternatives:
                emit_from_index = min(emit_from_index, i) if emit_from_index is not None else i
                continue
            # 0 means vendor doesn't set timestamps properly, in which case we just return
            if ev.alternatives[0].start_time == ev.alternatives[0].end_time == 0:
                self._transcript_buffer.clear()
                self._ignore_user_transcript_until = NOT_GIVEN
                return

            if (
                ev.alternatives[0].end_time > 0
                and ev.alternatives[0].end_time + self._input_started_at
                < self._ignore_user_transcript_until
            ):
                emit_from_index = None
            else:
                emit_from_index = min(emit_from_index, i) if emit_from_index is not None else i
                should_flush = True
                break

        # extract events to emit and reset BEFORE iterating
        # to prevent recursive calls
        events_to_emit = (
            list(self._transcript_buffer)[int(emit_from_index) :]
            if emit_from_index is not None and should_flush
            else []
        )
        self._transcript_buffer.clear()
        self._ignore_user_transcript_until = NOT_GIVEN

        for ev in events_to_emit:
            logger.debug(
                "re-emitting held user transcript",
                extra={
                    "event": ev.type,
                },
            )
            await self._on_stt_event(ev)

    def _should_hold_stt_event(self, ev: stt.SpeechEvent) -> bool:
        """Test if the event should be held until the ignore_user_transcript_until timestamp."""
        if not self._barge_in_enabled:
            return False

        if self._agent_speaking:
            return True

        if not is_given(self._ignore_user_transcript_until):
            return False
        # sentinel events are always held until
        # we have something concrete to release them
        if not ev.alternatives:
            return True
        if (
            # most vendors don't set timestamps properly, in which case we just assume
            # it is a valid event after the ignore_user_transcript_until timestamp
            is_given(self._input_started_at)
            # check if the event should be held if
            # 1. the stt input stream has started
            # 2. the current event has a valid start and end time, relative to the input stream start time
            # 3. the event is for audio sent before the ignore_user_transcript_until timestamp
            and self._input_started_at is not None
            and not (ev.alternatives[0].start_time == ev.alternatives[0].end_time == 0)
            and ev.alternatives[0].end_time > 0
            and ev.alternatives[0].end_time + self._input_started_at
            < self._ignore_user_transcript_until
        ):
            return True

        return False

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._input_started_at is None:
            self._input_started_at = time.time()

        self._sample_rate = frame.sample_rate
        if self._stt_ch is not None:
            self._stt_ch.send_nowait(frame)

        if self._vad_ch is not None:
            self._vad_ch.send_nowait(frame)

        if self._bargein_ch is not None:
            self._bargein_ch.send_nowait(frame)

    async def aclose(self) -> None:
        self._closing.set()

        if self._commit_user_turn_atask is not None:
            await self._commit_user_turn_atask

        await aio.cancel_and_wait(*self._tasks)

        if self._stt_atask is not None:
            await aio.cancel_and_wait(self._stt_atask)

        if self._vad_atask is not None:
            await aio.cancel_and_wait(self._vad_atask)

        if self._bargein_atask is not None:
            await aio.cancel_and_wait(self._bargein_atask)

        if self._end_of_turn_task is not None:
            await self._end_of_turn_task

    def update_stt(self, stt: io.STTNode | None) -> None:
        self._stt = stt
        if stt:
            self._stt_ch = aio.Chan[rtc.AudioFrame]()
            self._stt_atask = asyncio.create_task(
                self._stt_task(stt, self._stt_ch, self._stt_atask)
            )
            # reset barge-in related state
            self._transcript_buffer.clear()
            self._ignore_user_transcript_until = NOT_GIVEN
            self._input_started_at = None
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

    def update_bargein_detection(self, bargein_detection: inference.BargeinDetector | None) -> None:
        self._bargein_detection = bargein_detection
        if bargein_detection is not None:
            self._bargein_ch = aio.Chan[
                Union[
                    rtc.AudioFrame,
                    BargeinStreamBase._AgentSpeechStartedSentinel,
                    BargeinStreamBase._AgentSpeechEndedSentinel,
                    BargeinStreamBase._OverlapSpeechStartedSentinel,
                    BargeinStreamBase._OverlapSpeechEndedSentinel,
                ]
            ]()
            self._bargein_atask = asyncio.create_task(
                self._bargein_task(bargein_detection, self._bargein_ch, self._bargein_atask)
            )
            self._transcript_buffer.clear()
            self._ignore_user_transcript_until = NOT_GIVEN
            self._input_started_at = None
        elif self._bargein_atask is not None:
            task = asyncio.create_task(aio.cancel_and_wait(self._bargein_atask))
            task.add_done_callback(lambda _: self._tasks.discard(task))
            self._tasks.add(task)
            self._bargein_atask = None
            self._bargein_ch = None

    def clear_user_turn(self) -> None:
        self._audio_transcript = ""
        self._audio_interim_transcript = ""
        self._audio_preflight_transcript = ""
        self._final_transcript_confidence = []
        self._user_turn_committed = False

        # reset stt to clear the buffer from previous user turn
        stt = self._stt
        self.update_stt(None)
        self.update_stt(stt)

    def commit_user_turn(
        self,
        *,
        audio_detached: bool,
        transcript_timeout: float,
        stt_flush_duration: float = 2.0,
    ) -> None:
        if not self._stt or self._closing.is_set():
            return

        async def _commit_user_turn() -> None:
            if self._last_final_transcript_time is None or (
                time.time() - self._last_final_transcript_time > 0.5
            ):
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
                    num_frames = max(0, int(math.ceil(stt_flush_duration / silence_frame.duration)))
                    for _ in range(num_frames):
                        self.push_audio(silence_frame)

                # wait for the final transcript to be available
                try:
                    await asyncio.wait_for(
                        self._final_transcript_received.wait(),
                        timeout=transcript_timeout,
                    )
                except asyncio.TimeoutError:
                    if self._audio_interim_transcript:
                        logger.warning(
                            "final transcript not received after timeout",
                            extra={
                                "transcript_timeout": transcript_timeout,
                                "interim_transcript": self._audio_interim_transcript,
                            },
                        )

            if self._audio_interim_transcript:
                # emit interim transcript as final for frontend display
                self._hooks.on_final_transcript(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(language="", text=self._audio_interim_transcript)
                        ],
                    )
                )

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

        # handle barge-in detection
        # - hold the event until the ignore_user_transcript_until expires
        # - release only relevant events
        # - allow RECOGNITION_USAGE to pass through immediately
        if ev.type != stt.SpeechEventType.RECOGNITION_USAGE and self._barge_in_enabled:
            if self._should_hold_stt_event(ev):
                logger.trace(
                    "holding STT event until ignore_user_transcript_until expires",
                    extra={
                        "event": ev.type,
                        "ignore_user_transcript_until": self._ignore_user_transcript_until
                        if is_given(self._ignore_user_transcript_until)
                        else None,
                    },
                )
                self._transcript_buffer.append(ev)
                return
            elif self._transcript_buffer:
                await self._flush_held_transcripts()
                # no return here to allow the new event to be processed normally

        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            transcript = ev.alternatives[0].text
            language = ev.alternatives[0].language
            confidence = ev.alternatives[0].confidence

            if not self._last_language or (
                language and len(transcript) > MIN_LANGUAGE_DETECTION_LENGTH
            ):
                self._last_language = language

            if not transcript:
                return

            self._hooks.on_final_transcript(
                ev,
                speaking=self._speaking if self._vad else None,
            )
            extra: dict[str, Any] = {"user_transcript": transcript, "language": self._last_language}
            if self._last_speaking_time:
                extra["transcript_delay"] = time.time() - self._last_speaking_time
            logger.debug("received user transcript", extra=extra)

            self._last_final_transcript_time = time.time()
            self._audio_transcript += f" {transcript}"
            self._audio_transcript = self._audio_transcript.lstrip()
            self._final_transcript_confidence.append(confidence)
            transcript_changed = self._audio_transcript != self._audio_preflight_transcript
            self._audio_interim_transcript = ""
            self._audio_preflight_transcript = ""
            self._final_transcript_received.set()

            if not self._vad or self._last_speaking_time == 0:
                # vad disabled, use stt timestamp
                # TODO: this would screw up transcription latency metrics
                # but we'll live with it for now.
                # the correct way is to ensure STT fires SpeechEventType.END_OF_SPEECH
                # and using that timestamp for _last_speaking_time
                self._last_speaking_time = time.time()

            if self._vad_base_turn_detection or self._user_turn_committed:
                if transcript_changed:
                    self._hooks.on_preemptive_generation(
                        _PreemptiveGenerationInfo(
                            new_transcript=self._audio_transcript,
                            transcript_confidence=(
                                sum(self._final_transcript_confidence)
                                / len(self._final_transcript_confidence)
                                if self._final_transcript_confidence
                                else 0
                            ),
                            started_speaking_at=self._speech_start_time,
                        )
                    )

                if not self._speaking:
                    chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                    self._run_eou_detection(chat_ctx)

        elif ev.type == stt.SpeechEventType.PREFLIGHT_TRANSCRIPT:
            self._hooks.on_interim_transcript(ev, speaking=self._speaking if self._vad else None)
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
                "received user preflight transcript",
                extra={"user_transcript": transcript, "language": self._last_language},
            )

            # still need to increment it as it's used for turn detection,
            self._last_final_transcript_time = time.time()
            # preflight transcript includes all pre-committed transcripts (including final transcript from the previous STT run)
            self._audio_preflight_transcript = (self._audio_transcript + " " + transcript).lstrip()
            self._audio_interim_transcript = transcript

            if not self._vad or self._last_speaking_time == 0:
                # vad disabled, use stt timestamp
                self._last_speaking_time = time.time()

            if self._turn_detection_mode != "manual" or self._user_turn_committed:
                confidence_vals = list(self._final_transcript_confidence) + [confidence]
                self._hooks.on_preemptive_generation(
                    _PreemptiveGenerationInfo(
                        new_transcript=self._audio_preflight_transcript,
                        transcript_confidence=sum(confidence_vals) / len(confidence_vals),
                        started_speaking_at=self._speech_start_time,
                    )
                )

        elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            self._hooks.on_interim_transcript(ev, speaking=self._speaking if self._vad else None)
            self._audio_interim_transcript = ev.alternatives[0].text

        elif ev.type == stt.SpeechEventType.END_OF_SPEECH and self._turn_detection_mode == "stt":
            with trace.use_span(self._ensure_user_turn_span()):
                self._hooks.on_end_of_speech(None)

            self._speaking = False
            self._user_turn_committed = True
            self._last_speaking_time = time.time()

            chat_ctx = self._hooks.retrieve_chat_ctx().copy()
            self._run_eou_detection(chat_ctx)

        elif ev.type == stt.SpeechEventType.START_OF_SPEECH and self._turn_detection_mode == "stt":
            with trace.use_span(self._ensure_user_turn_span()):
                self._hooks.on_start_of_speech(None)

            self._speaking = True
            if self._speech_start_time is None:
                self._speech_start_time = time.time()
            self._last_speaking_time = time.time()

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

    async def _on_vad_event(self, ev: vad.VADEvent) -> None:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            with trace.use_span(self._ensure_user_turn_span()):
                self._hooks.on_start_of_speech(ev)

            self._speaking = True

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

        elif ev.type == vad.VADEventType.INFERENCE_DONE:
            self._hooks.on_vad_inference_done(ev)

            # for metrics, get the "earliest" signal of speech as possible
            if ev.raw_accumulated_speech > 0.0:
                self._last_speaking_time = time.time()

                if self._speech_start_time is None:
                    self._speech_start_time = time.time()

        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            with trace.use_span(self._ensure_user_turn_span()):
                self._hooks.on_end_of_speech(ev)

            self._speaking = False

            if self._vad_base_turn_detection or (
                self._turn_detection_mode == "stt" and self._user_turn_committed
            ):
                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                self._run_eou_detection(chat_ctx)

    async def _on_bargein_event(self, ev: inference.BargeinEvent) -> None:
        if ev.type == inference.BargeinEventType.BARGEIN:
            self._hooks.on_bargein(ev)

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
        async def _bounce_eou_task(
            last_speaking_time: float | None = None,
            last_final_transcript_time: float | None = None,
            speech_start_time: float | None = None,
        ) -> None:
            endpointing_delay = self._min_endpointing_delay
            user_turn_span = self._ensure_user_turn_span()
            if turn_detector is not None:
                if not await turn_detector.supports_language(self._last_language):
                    logger.info("Turn detector does not support language %s", self._last_language)
                else:
                    with (
                        trace.use_span(user_turn_span),
                        tracer.start_as_current_span("eou_detection") as eou_detection_span,
                    ):
                        # if there are failures, we should not hold the pipeline up
                        end_of_turn_probability = 0.0
                        unlikely_threshold: float | None = None
                        try:
                            end_of_turn_probability = await turn_detector.predict_end_of_turn(
                                chat_ctx
                            )
                            unlikely_threshold = await turn_detector.unlikely_threshold(
                                self._last_language
                            )
                            if (
                                unlikely_threshold is not None
                                and end_of_turn_probability < unlikely_threshold
                            ):
                                endpointing_delay = self._max_endpointing_delay
                        except Exception:
                            logger.exception("Error predicting end of turn")

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

            extra_sleep = endpointing_delay
            if last_speaking_time:
                extra_sleep += last_speaking_time - time.time()

            if extra_sleep > 0:
                try:
                    await asyncio.wait_for(self._closing.wait(), timeout=extra_sleep)
                except asyncio.TimeoutError:
                    pass

            confidence_avg = (
                sum(self._final_transcript_confidence) / len(self._final_transcript_confidence)
                if self._final_transcript_confidence
                else 0
            )

            started_speaking_at = None
            stopped_speaking_at = None
            transcription_delay = None
            end_of_turn_delay = None

            # sometimes, we can't calculate the metrics because VAD was unreliable.
            # in this case, we just ignore the calculation, it's better than providing likely wrong values
            if (
                last_final_transcript_time is not None
                and last_speaking_time is not None
                and speech_start_time is not None
            ):
                started_speaking_at = speech_start_time
                stopped_speaking_at = last_speaking_time
                transcription_delay = max(last_final_transcript_time - last_speaking_time, 0)
                end_of_turn_delay = time.time() - last_speaking_time

            committed = self._hooks.on_end_of_turn(
                _EndOfTurnInfo(
                    new_transcript=self._audio_transcript,
                    transcript_confidence=confidence_avg,
                    transcription_delay=transcription_delay or 0,
                    end_of_turn_delay=end_of_turn_delay,
                    started_speaking_at=started_speaking_at,
                    stopped_speaking_at=stopped_speaking_at,
                )
            )
            if committed:
                user_turn_span.set_attributes(
                    {
                        trace_types.ATTR_USER_TRANSCRIPT: self._audio_transcript,
                        trace_types.ATTR_TRANSCRIPT_CONFIDENCE: confidence_avg,
                        trace_types.ATTR_TRANSCRIPTION_DELAY: transcription_delay or 0,
                        trace_types.ATTR_END_OF_TURN_DELAY: end_of_turn_delay or 0,
                    }
                )
                user_turn_span.end()
                self._user_turn_span = None

                # clear the transcript if the user turn was committed
                self._audio_transcript = ""
                self._final_transcript_confidence = []
                self._last_speaking_time = None
                self._last_final_transcript_time = None
                self._speech_start_time = None

            self._user_turn_committed = False

        if self._end_of_turn_task is not None:
            # TODO(theomonnom): disallow cancel if the extra sleep is done
            self._end_of_turn_task.cancel()

        # copy the last_speaking_time before awaiting (the value can change)
        self._end_of_turn_task = asyncio.create_task(
            _bounce_eou_task(
                self._last_speaking_time,
                self._last_final_transcript_time,
                self._speech_start_time,
            )
        )

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

    @utils.log_exceptions(logger=logger)
    async def _bargein_task(
        self,
        bargein_detection: inference.BargeinDetector,
        audio_input: AsyncIterable[
            rtc.AudioFrame
            | BargeinStreamBase._AgentSpeechStartedSentinel
            | BargeinStreamBase._AgentSpeechEndedSentinel
            | BargeinStreamBase._OverlapSpeechStartedSentinel
            | BargeinStreamBase._OverlapSpeechEndedSentinel
        ],
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is not None:
            await aio.cancel_and_wait(task)

        stream = bargein_detection.stream()

        @utils.log_exceptions(logger=logger)
        async def _forward() -> None:
            async for frame in audio_input:
                stream.push_frame(frame)

        forward_task = asyncio.create_task(_forward())

        try:
            async for ev in stream:
                await self._on_bargein_event(ev)
        finally:
            await aio.cancel_and_wait(forward_task)
            await stream.aclose()
            if self._bargein_ch:
                while not self._bargein_ch.empty():
                    self._bargein_ch.recv_nowait()
                self._bargein_ch.close()

    def _ensure_user_turn_span(self) -> trace.Span:
        if self._user_turn_span and self._user_turn_span.is_recording():
            return self._user_turn_span

        self._user_turn_span = tracer.start_span("user_turn")

        if (room_io := self._session._room_io) and room_io.linked_participant:
            _set_participant_attributes(self._user_turn_span, room_io.linked_participant)

        return self._user_turn_span
