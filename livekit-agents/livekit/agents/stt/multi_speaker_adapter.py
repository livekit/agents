from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from livekit import rtc

from .. import utils
from ..language import LanguageCode
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils.audio import AudioByteStream
from .stt import STT, RecognizeStream, SpeechData, SpeechEvent, SpeechEventType

if TYPE_CHECKING:
    from ..voice.events import ConversationItemAddedEvent


class MultiSpeakerAdapter(STT):
    def __init__(
        self,
        *,
        stt: STT,
        detect_primary_speaker: bool = True,
        suppress_background_speaker: bool = False,
        primary_detection_options: NotGivenOr[PrimarySpeakerDetectionOptions] = NOT_GIVEN,
        primary_format: str = "{text}",
        background_format: str = "{text}",
    ):
        """MultiSpeakerAdapter is an adapter that allows to detect and suppress background speakers.
        It needs STT with diarization capability and works for a single audio track.

        Args:
            stt (STT): STT instance to wrap
            detect_primary_speaker (bool, optional): Whether to detect primary speaker. Defaults to True.
            suppress_background_speaker (bool, optional): Whether to suppress background speaker. Defaults to False.
            primary_detection_options (NotGivenOr[PrimarySpeakerDetectionOptions], optional): Primary speaker detection options.
                If not provided, the default options will be used.
            primary_format (str, optional): Format for primary speaker.
                Supports {text} and {speaker_id} placeholders. Defaults to "{text}".
            background_format (str, optional): Format for background speaker.
                Supports {text} and {speaker_id} placeholders. Defaults to "{text}".

        Raises:
            ValueError: If the STT does not support diarization.
        """
        if not stt.capabilities.diarization:
            raise ValueError("MultiSpeakerAdapter needs STT with diarization capability")

        super().__init__(capabilities=stt.capabilities)
        self._stt = stt

        self._detect_primary = detect_primary_speaker
        self._suppress_background = suppress_background_speaker
        self._opt = primary_detection_options or PrimarySpeakerDetectionOptions()
        self._primary_format = primary_format
        self._background_format = background_format

        self._stt.on("metrics_collected", self._on_metrics_collected)
        # the wrapped STT's metrics are the accurate ones (its own label, model/provider and
        # provider-reported usage), so they are re-emitted here and this adapter does not add a
        # second measurement for the same audio -- same split as the stream and fallback adapters
        self._recognize_metrics_needed = False

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, item: ConversationItemAddedEvent) -> None:
        self._stt._push_conversation_item(item)

    def prewarm(self) -> None:
        self._stt.prewarm()

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def aclose(self) -> None:
        self._stt.off("metrics_collected", self._on_metrics_collected)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await self._stt.recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return MultiSpeakerAdapterWrapper(
            stt=self, wrapped_stt=self._stt, language=language, conn_options=conn_options
        )


class MultiSpeakerAdapterWrapper(RecognizeStream):
    def __init__(
        self,
        stt: MultiSpeakerAdapter,
        *,
        wrapped_stt: STT,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options)
        self._wrapped_stt = wrapped_stt
        self._language = language

        self._detector = _PrimarySpeakerDetector(
            detect_primary_speaker=stt._detect_primary,
            suppress_background_speaker=stt._suppress_background,
            primary_detection_options=stt._opt,
            primary_format=stt._primary_format,
            background_format=stt._background_format,
        )

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        # the wrapped stream already reports usage for this audio and the adapter re-emits it,
        # so measuring the forwarded events here would count every recognition twice. the
        # retry-count reset still has to happen, otherwise hiccups spread over a long call
        # accumulate instead of being forgiven by a successful transcript.
        async for ev in event_aiter:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                self._mark_provider_progress()

    async def _run(self) -> None:
        async def _forward_input(stream: RecognizeStream) -> None:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    # the leg can close while frames are still in flight. the input was
                    # taken, so the detector must count it, but raising here would turn a
                    # retryable provider failure into a fatal RuntimeError and skip the
                    # wrapper's retry
                    with contextlib.suppress(RuntimeError):
                        stream.push_frame(frame)
                    self._detector.push_audio(frame)
                elif isinstance(frame, self._FlushSentinel):
                    with contextlib.suppress(RuntimeError):
                        stream.flush()

            with contextlib.suppress(RuntimeError):
                stream.end_input()

        async def _forward_output(stream: RecognizeStream, input_offset: float) -> None:
            async for ev in stream:
                # the wrapped leg reports on its own clock; place its timestamps on the
                # wrapper's input timeline before the detector maps them onto the audio
                # it has actually forwarded
                _shift_event_times(ev, input_offset)
                updated_ev = self._detector.on_stt_event(ev)
                if updated_ev is not None:
                    self._event_ch.send_nowait(updated_ev)
                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    # send an empty final transcript to clear the interim results
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[SpeechData(language=LanguageCode(""), text="")],
                        )
                    )

        stream = self._wrapped_stt.stream(language=self._language, conn_options=self._conn_options)
        stream._on_provider_progress = self._mark_provider_progress
        # do not seed `stream.start_time_offset`: diarization plugins are not consistent
        # about applying it to the SpeechData they emit (deepgram/assemblyai/aws add it,
        # xAI builds timestamps straight from its word clocks), so seeding it would put
        # different providers on different timelines. keep every leg on its own clock and
        # translate the events here instead.
        #
        # where this leg starts on the wrapper's input timeline; its provider clock starts
        # at zero there, on every attempt
        input_offset = self.start_time_offset + (time.time() - self._start_time)
        # the leg's clock starts where the detector's forwarded audio currently ends, not
        # where the input timeline is: the difference is the reconnect gap (wall-clock time
        # not covered by forwarded frames). the detector subtracts it so its RMS buffer
        # keeps indexing the audio it actually saw.
        self._detector.input_time_offset = input_offset - self._detector.pushed_duration

        tasks = [
            asyncio.create_task(
                _forward_input(stream), name="DiarizationAdapterWrapper.forward_input"
            ),
            asyncio.create_task(
                _forward_output(stream, input_offset),
                name="DiarizationAdapterWrapper.forward_output",
            ),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await stream.aclose()


def _shift_event_times(ev: SpeechEvent, offset: float) -> None:
    """Move a provider event's timestamps onto the wrapper's input timeline.

    A wrapped leg counts from zero when it is created, no matter how much audio the
    wrapper has already consumed, so its timestamps are relative to the start of that
    leg. ``offset`` is where the leg starts on the wrapper's input timeline.
    """
    if offset == 0:
        return

    for sd in ev.alternatives:
        # plugins leave both bounds at zero when they have no timing for a segment;
        # shifting those would invent a position on the input timeline
        if sd.start_time != 0.0 or sd.end_time != 0.0:
            sd.start_time += offset
            sd.end_time += offset

        for word in sd.words or []:
            if utils.is_given(word.start_time):
                word.start_time += offset
            if utils.is_given(word.end_time):
                word.end_time += offset
            if utils.is_given(word.start_time_offset):
                word.start_time_offset += offset


@dataclass
class PrimarySpeakerDetectionOptions:
    """Configuration for primary speaker detection"""

    frame_size_ms: int = 100
    """Frame size for RMS computation"""
    rms_buffer_duration: float = 120.0
    """How long to keep RMS data"""
    min_rms_samples: int = 3
    """Minimum RMS samples needed for a speech event"""
    rms_smoothing_factor: float = 0.5
    """Smoothing factor for RMS for a speaker, rms = rms * factor + new_rms * (1 - factor)"""

    # switching primary speaker
    threshold_multiplier: float = 1.3
    """Candidate's RMS needs to be louder than current primary's RMS by this multiplier"""
    decay_to_equal_time: float = 60
    """Time to decay from switch_threshold_multiplier to 1.0 (equal levels)"""
    threshold_min_multiplier: float = 0.5
    """Minimum threshold multiplier (candidate can be min_multiplier quieter)"""


class _PrimarySpeakerDetector:
    @dataclass
    class SpeakerData:
        speaker_id: str
        last_activity_time: float = 0.0
        rms: float = 0.0

    def __init__(
        self,
        *,
        detect_primary_speaker: bool = True,
        suppress_background_speaker: bool = False,
        primary_detection_options: NotGivenOr[PrimarySpeakerDetectionOptions] = NOT_GIVEN,
        primary_format: str = "{text}",
        background_format: str = "{text}",
    ):
        """Primary speaker detector. It detects the primary speaker based on RMS,
        formats the primary and background speakers separately, or suppresses the background speaker.

        Args:
            detect_primary_speaker (bool, optional): Whether to detect primary speaker. Defaults to True.
            suppress_background_speaker (bool, optional): Whether to suppress background speaker. Defaults to False.
            primary_detection_options (PrimaryDetectionOptions, optional): Primary speaker detection options.
            primary_format (str, optional): Format for primary speaker.
                Supports {text} and {speaker_id} placeholders. Defaults to "{text}".
            background_format (str, optional): Format for background speaker.
                Supports {text} and {speaker_id} placeholders. Defaults to "{text}".
        """
        self._primary_format = primary_format
        self._background_format = background_format
        self._detect_primary = detect_primary_speaker
        self._suppress_background = suppress_background_speaker
        self._opt = primary_detection_options or PrimarySpeakerDetectionOptions()

        if self._suppress_background and not self._detect_primary:
            logger.warning(
                "Suppressing background speaker is not supported when `detect_primary_speaker` is False"
            )
            self._suppress_background = False

        self._pushed_duration: float = 0.0
        self._input_time_offset: float = 0.0
        self._primary_speaker: str | None = None
        self._speaker_data: dict[str, _PrimarySpeakerDetector.SpeakerData] = {}
        self._bstream: AudioByteStream | None = None

        self._rms_buffer: list[float] = []
        self._frame_size = self._opt.frame_size_ms / 1000
        self._max_rms_size = int(self._opt.rms_buffer_duration / self._frame_size)

    @property
    def pushed_duration(self) -> float:
        """Duration of the audio forwarded to this detector so far."""
        return self._pushed_duration

    @property
    def input_time_offset(self) -> float:
        """Shift from the wrapper's input timeline to the detector's forwarded-audio clock."""
        return self._input_time_offset

    @input_time_offset.setter
    def input_time_offset(self, value: float) -> None:
        self._input_time_offset = value

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._detect_primary:
            self._pushed_duration += frame.duration
            return

        if not self._bstream:
            sample_per_channel = int(frame.sample_rate * self._frame_size)
            self._bstream = AudioByteStream(
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=sample_per_channel,
            )
            self._frame_size = sample_per_channel / frame.sample_rate  # accurate frame size

        for f in self._bstream.push(frame.data):
            rms = self._compute_rms(f)
            self._rms_buffer.append(rms)
            self._pushed_duration += f.duration

        if len(self._rms_buffer) > self._max_rms_size:
            self._rms_buffer = self._rms_buffer[-self._max_rms_size :]

    def on_stt_event(self, ev: SpeechEvent) -> SpeechEvent | None:
        if not ev.alternatives:
            return ev

        sd = ev.alternatives[0]
        if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
            self._update_primary_speaker(sd)

        if sd.speaker_id is None or self._primary_speaker is None:
            return ev

        sd.is_primary_speaker = sd.speaker_id == self._primary_speaker

        # format the transcript
        if sd.is_primary_speaker:
            sd.text = self._primary_format.format(text=sd.text, speaker_id=sd.speaker_id)
        else:
            if self._suppress_background:
                return None

            sd.text = self._background_format.format(text=sd.text, speaker_id=sd.speaker_id)
        return ev

    def _compute_rms(self, frame: rtc.AudioFrame) -> float:
        audio_data = np.frombuffer(frame.data, dtype=np.int16)
        if len(audio_data) == 0:
            return 0.0

        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        return float(rms)

    def _get_rms_for_timerange(self, start_time: float, end_time: float) -> float | None:
        if not self._rms_buffer:
            return None

        start = int((self._pushed_duration - start_time) / self._frame_size)
        end = int((self._pushed_duration - end_time) / self._frame_size)
        start = len(self._rms_buffer) - start - 1
        end = len(self._rms_buffer) - end

        if end < 0 or start >= len(self._rms_buffer):
            return None
        start = max(start, 0)

        if end - start < self._opt.min_rms_samples:
            return None

        return float(np.median(self._rms_buffer[start:end]))

    def _update_primary_speaker(self, sd: SpeechData) -> None:
        if not self._detect_primary:
            self._primary_speaker = None
            return

        if sd.speaker_id is None:
            # an unattributed segment carries no evidence about who is primary,
            # so leave the current primary in place rather than clearing it
            return

        # timestamps arrive on the wrapper's input timeline, but the RMS buffer is indexed
        # by the duration of audio this detector has forwarded, so translate them back
        # (the difference is the reconnect gap the input timeline ran through)
        start_time = sd.start_time - self._input_time_offset
        end_time = sd.end_time - self._input_time_offset

        rms = self._get_rms_for_timerange(start_time, end_time)
        if rms is None:
            return

        # update speaker data
        speaker_id = sd.speaker_id
        if data := self._speaker_data.get(speaker_id):
            data.last_activity_time = end_time
            data.rms = data.rms * self._opt.rms_smoothing_factor + rms * (
                1 - self._opt.rms_smoothing_factor
            )
        else:
            self._speaker_data[speaker_id] = _PrimarySpeakerDetector.SpeakerData(
                speaker_id=speaker_id,
                last_activity_time=end_time,
                rms=rms,
            )

        if self._primary_speaker == speaker_id:
            return

        # compare the new speaker's RMS to the primary's RMS, switch primary if:
        # 1. it's the first speaker
        # 2. the new speaker's RMS is significantly louder than the primary's RMS

        if (
            self._primary_speaker is None
            or (primary := self._speaker_data.get(self._primary_speaker)) is None
        ):
            self._primary_speaker = speaker_id
            logger.debug("set first primary speaker", extra={"speaker_id": speaker_id, "rms": rms})
            return

        silence_duration = self._pushed_duration - primary.last_activity_time

        # decay the threshold multiplier over time in case the primary speaker is silent for a long time
        if self._opt.threshold_multiplier > 1.0:
            decay_rate = (self._opt.threshold_multiplier - 1.0) / self._opt.decay_to_equal_time
        else:
            decay_rate = 0.0

        multiplier = max(
            self._opt.threshold_multiplier - (decay_rate * silence_duration),
            self._opt.threshold_min_multiplier,
        )
        rms_threshold = primary.rms * multiplier
        extra = {
            "speaker_id": speaker_id,
            "rms": rms,
            "rms_threshold": rms_threshold,
            "silence_duration": silence_duration,
            "multiplier": multiplier,
        }
        if rms > rms_threshold:
            self._primary_speaker = speaker_id
            logger.debug("primary speaker switched", extra=extra)
        else:
            logger.debug("primary speaker unchanged", extra=extra)
