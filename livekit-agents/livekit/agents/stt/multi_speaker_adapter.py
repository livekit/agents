from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

import numpy as np

from livekit import rtc

from .. import utils
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils.audio import AudioByteStream
from .stt import STT, RecognizeStream, SpeechData, SpeechEvent, SpeechEventType


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

    async def _run(self) -> None:
        async def _forward_input(stream: RecognizeStream) -> None:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    stream.push_frame(frame)
                    self._detector.push_audio(frame)
                elif isinstance(frame, self._FlushSentinel):
                    stream.flush()

            with contextlib.suppress(RuntimeError):
                stream.end_input()

        async def _forward_output(stream: RecognizeStream) -> None:
            async for ev in stream:
                updated_ev = self._detector.on_stt_event(ev)
                if updated_ev is not None:
                    self._event_ch.send_nowait(updated_ev)
                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    # send an empty final transcript to clear the interim results
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[SpeechData(language="", text="")],
                        )
                    )

        stream = self._wrapped_stt.stream(language=self._language, conn_options=self._conn_options)
        tasks = [
            asyncio.create_task(
                _forward_input(stream), name="DiarizationAdapterWrapper.forward_input"
            ),
            asyncio.create_task(
                _forward_output(stream), name="DiarizationAdapterWrapper.forward_output"
            ),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await stream.aclose()


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
        self._primary_speaker: str | None = None
        self._speaker_data: dict[str, _PrimarySpeakerDetector.SpeakerData] = {}
        self._bstream: AudioByteStream | None = None

        self._rms_buffer: list[float] = []
        self._frame_size = self._opt.frame_size_ms / 1000
        self._max_rms_size = int(self._opt.rms_buffer_duration / self._frame_size)

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
        if sd.speaker_id is None or not self._detect_primary:
            self._primary_speaker = None
            return

        rms = self._get_rms_for_timerange(sd.start_time, sd.end_time)
        if rms is None:
            return

        # update speaker data
        speaker_id = sd.speaker_id
        if data := self._speaker_data.get(speaker_id):
            data.last_activity_time = sd.end_time
            data.rms = data.rms * self._opt.rms_smoothing_factor + rms * (
                1 - self._opt.rms_smoothing_factor
            )
        else:
            self._speaker_data[speaker_id] = _PrimarySpeakerDetector.SpeakerData(
                speaker_id=speaker_id,
                last_activity_time=sd.end_time,
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
