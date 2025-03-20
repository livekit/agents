from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from livekit import rtc

from ... import tokenize, utils
from ...log import logger
from ...tokenize.tokenizer import PUNCTUATIONS
from ...types import NOT_GIVEN, NotGivenOr
from ...vad import VAD, VADEventType, VADStream
from ...voice.io import AudioOutput, PlaybackFinishedEvent, TextOutput
from . import _utils
from .streaming_syllable_rate import SyllableRateDetector, SyllableRateStream

# Standard speech rate in hyphens per second
STANDARD_SPEECH_RATE = 3.83


@dataclass
class TextSyncOptions:
    """Options for synchronizing TTS segments with audio playback."""

    language: str = ""
    speed: float = 1.0  # Multiplier of STANDARD_SPEECH_RATE
    new_sentence_delay: float = 0.4
    pause_to_new_sentence_delay_factor: float = 0.8
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(
        retain_format=True
    )
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    split_words: Callable[[str], list[tuple[str, int, int]]] = tokenize.basic.split_words
    vad: VAD | None = None


@dataclass
class _VoiceActivityData:
    timestamps: list[float] = field(default_factory=list)
    speaking: list[bool] = field(default_factory=list)
    speak_duration_cumsum: list[float] = field(default_factory=list)
    syllable_rates: list[float] = field(default_factory=list)
    syllable_integrals: list[float] = field(default_factory=list)

    def append(self, timestamp: float, speaking: bool, syllable_rate: float) -> None:
        if not self.timestamps:
            speak_duration = 0
            syllable_integral = 0
        else:
            speak_duration = self.speak_duration_cumsum[-1]
            syllable_integral = self.syllable_integrals[-1]

        if speaking:
            duration = timestamp - self.timestamps[-1] if self.timestamps else timestamp
            speak_duration += duration
            syllable_integral += syllable_rate * duration

        self.timestamps.append(timestamp)
        self.speaking.append(speaking)
        self.speak_duration_cumsum.append(speak_duration)
        self.syllable_rates.append(syllable_rate)
        self.syllable_integrals.append(syllable_integral)

    def get_speak_duration(self, timestamp: float) -> tuple[float, float]:
        """Get the cumulative speech duration up to the given timestamp."""
        if not self.timestamps:
            return 0, 0

        idx = np.searchsorted(self.timestamps, timestamp, side="right")
        if idx == 0:
            return 0, 0

        speak_duration = self.speak_duration_cumsum[idx - 1]
        syllable_integral = self.syllable_integrals[idx - 1]
        if self.speaking[idx - 1]:
            gap_s = timestamp - self.timestamps[idx - 1]
            speak_duration += gap_s
            syllable_integral += self.syllable_rates[idx - 1] * gap_s
        return speak_duration, syllable_integral

    @property
    def pushed_duration(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0

    @property
    def speak_duration(self) -> float:
        return self.speak_duration_cumsum[-1] if self.speak_duration_cumsum else 0


@dataclass
class _AudioData:
    pushed_duration: float = 0.0
    done: bool = False
    id: str = field(default_factory=_utils.speech_uuid)
    vad_stream: SyllableRateStream | None = None
    vad_data: _VoiceActivityData | None = None

    def __post_init__(self) -> None:
        if self.vad_stream:
            self.vad_data = _VoiceActivityData()


@dataclass
class _TextData:
    sentence_stream: tokenize.SentenceStream
    pushed_text: str = ""
    done: bool = False
    forwarded_hyphens: int = 0
    new_sentence_pauses_s: float = 0.0


@dataclass
class _TextSegment:
    delta: str
    stream_id: str
    sentence_id: str
    end_of_sentence: bool
    language: str


class _TextAudioSynchronizer:
    """Synchronizes text with audio playback timing."""

    def __init__(self, options: TextSyncOptions):
        super().__init__()

        self._opts = options
        self._speed = options.speed * STANDARD_SPEECH_RATE
        self._vad = SyllableRateDetector()
        # self._vad = None
        self._closed = False
        self._close_future = asyncio.Future[None]()
        self._event_ch = utils.aio.Chan[_TextSegment]()

        self._playing_seg_index = -1
        self._finished_seg_index = -1

        self._text_q_changed = asyncio.Event()
        self._text_q = list[Optional[_TextData]]()
        self._audio_q_changed = asyncio.Event()
        self._audio_q = list[Optional[_AudioData]]()

        self._text_data: _TextData | None = None
        self._audio_data: _AudioData | None = None
        self._processing_text_data: _TextData | None = None

        self._main_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push an audio frame for the current segment."""
        self._check_not_closed()

        if self._audio_data is None:
            self._audio_data = _AudioData(vad_stream=self._vad.stream() if self._vad else None)
            if self._audio_data.vad_stream:
                vad_task = asyncio.create_task(
                    self._vad_stream_task(self._audio_data), name="vad_stream_task"
                )
                self._tasks.add(vad_task)
                vad_task.add_done_callback(self._tasks.discard)

            self._audio_q.append(self._audio_data)
            self._audio_q_changed.set()

        frame_duration = frame.samples_per_channel / frame.sample_rate
        self._audio_data.pushed_duration += frame_duration
        if self._audio_data.vad_stream:
            self._audio_data.vad_stream.push_frame(frame)

    def push_text(self, text: str) -> None:
        """Push text for the current segment."""
        self._check_not_closed()

        if self._text_data is None:
            self._text_data = _TextData(sentence_stream=self._opts.sentence_tokenizer.stream())
            self._text_q.append(self._text_data)
            self._text_q_changed.set()

        self._text_data.pushed_text += text
        self._text_data.sentence_stream.push_text(text)

    def mark_audio_segment_end(self) -> None:
        """Mark the current audio segment as complete."""
        self._check_not_closed()

        logger.info("mark_audio_segment_end")  # TODO(long): remove it after testing
        if self._audio_data is None:
            return

        assert self._audio_data is not None
        self._audio_data.done = True
        if self._audio_data.vad_stream:
            self._audio_data.vad_stream.end_input()
            task = asyncio.create_task(
                self._audio_data.vad_stream.aclose(), name="vad_stream_close"
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        self._audio_data = None

    def mark_text_segment_end(self) -> None:
        """Mark the current text segment as complete."""
        self._check_not_closed()
        logger.info("mark_text_segment_end")

        if self._text_data is None:
            return

        assert self._text_data is not None
        self._text_data.done = True
        self._text_data.sentence_stream.end_input()
        self._text_data = None

    def segment_playout_started(self) -> None:
        """Notify that audio playout has started for current segment."""
        self._check_not_closed()
        self._playing_seg_index += 1

        if self._main_task is None:
            self._main_task = asyncio.create_task(self._main_loop())

    def segment_playout_finished(self) -> None:
        """Notify that audio playout has finished for current segment."""
        self._check_not_closed()
        self._finished_seg_index += 1

    async def aclose(self) -> None:
        """Close the syncer and stop processing."""
        if self._closed:
            return

        self._closed = True
        self._close_future.set_result(None)

        if self._processing_text_data is not None:
            await self._processing_text_data.sentence_stream.aclose()
            self._processing_text_data = None

        for text_data in self._text_q:
            if text_data is not None:
                await text_data.sentence_stream.aclose()

        self._text_q.append(None)
        self._audio_q.append(None)
        self._text_q_changed.set()
        self._audio_q_changed.set()

        if self._main_task is not None:
            await self._main_task
        self._event_ch.close()

    async def __anext__(self) -> _TextSegment:
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[_TextSegment]:
        return self

    @utils.log_exceptions(logger=logger)
    async def _main_loop(self) -> None:
        """Main processing loop that synchronizes text with audio timing."""
        seg_index = 0
        q_done = False

        while not q_done and not self._closed:
            await self._text_q_changed.wait()
            await self._audio_q_changed.wait()

            while self._text_q and self._audio_q:
                text_data = self._text_q.pop(0)
                audio_data = self._audio_q.pop(0)
                self._processing_text_data = text_data

                if text_data is None or audio_data is None:
                    q_done = True
                    break

                # Wait for segment to start playing
                while not self._closed:
                    if self._playing_seg_index >= seg_index:
                        break
                    await self._sleep_if_not_closed(0.125)

                sentence_stream = text_data.sentence_stream
                forward_start_time = time.time()

                async for ev in sentence_stream:
                    await self._sync_sentence(
                        seg_index, forward_start_time, text_data, audio_data, ev.token
                    )
                self._processing_text_data = None
                seg_index += 1

            self._text_q_changed.clear()
            self._audio_q_changed.clear()

    @utils.log_exceptions(logger=logger)
    async def _sync_sentence(
        self,
        segment_index: int,
        segment_start_time: float,
        text_data: _TextData,
        audio_data: _AudioData,
        sentence: str,
    ) -> None:
        """Synchronize a sentence with audio timing."""
        real_speed = None
        syllable_speed = None
        if audio_data.pushed_duration > 0 and audio_data.done:
            pushed_hyphens = len(self._calc_hyphens(text_data.pushed_text))

            speak_duration, syllable_integral = audio_data.vad_data.get_speak_duration(
                audio_data.pushed_duration
            )
            if speak_duration <= 0:
                speak_duration = audio_data.pushed_duration

            if speak_duration > 0:
                real_speed = pushed_hyphens / speak_duration
                logger.info(
                    f"real_speed: {real_speed}, text_hyphens: {pushed_hyphens}, speak_duration: {speak_duration}"
                )

            if syllable_integral > 0:
                syllable_speed = pushed_hyphens / syllable_integral
                logger.info(
                    f"syllable_speed: {syllable_speed}, syllable_integral: {syllable_integral}"
                )

        seg_id = _utils.segment_uuid()
        words = self._opts.split_words(sentence)
        processed_words: list[str] = []

        sent_text = ""
        for word, _start_pos, end_pos in words:
            if segment_index <= self._finished_seg_index:
                break

            if self._closed:
                return

            word_hyphens = len(self._opts.hyphenate_word(word))
            processed_words.append(word)

            elapsed_time = time.time() - segment_start_time
            text = sentence[0:end_pos]
            text = text.rstrip("".join(PUNCTUATIONS))

            speed = self._speed
            if real_speed is not None:
                speed = real_speed
                syllable_integral = None
                if audio_data.vad_data:
                    speaks_s, syllable_integral = audio_data.vad_data.get_speak_duration(
                        elapsed_time
                    )
                else:
                    estimated_pauses_s = text_data.new_sentence_pauses_s
                    speaks_s = elapsed_time - estimated_pauses_s

                if syllable_integral is not None and syllable_speed is not None:
                    target_hyphens = round(syllable_speed * syllable_integral)

                else:
                    target_hyphens = round(speed * speaks_s)

                dt = target_hyphens - text_data.forwarded_hyphens
                to_wait_hyphens = max(0.0, word_hyphens - dt)
                delay = to_wait_hyphens / speed
                logger.info(
                    f"dt: {dt}, target_hyphens: {target_hyphens}, forward_hyphens: {text_data.forwarded_hyphens}, "
                    f"syllable_speed: {syllable_speed}, syllable_integral: {syllable_integral}, delay: {delay}"
                )
            else:
                delay = word_hyphens / speed

            first_delay = min(delay / 2, 2 / speed)
            await self._sleep_if_not_closed(first_delay)

            self._event_ch.send_nowait(
                _TextSegment(
                    delta=text[len(sent_text) :],
                    stream_id=audio_data.id,
                    sentence_id=seg_id,
                    language=self._opts.language,
                    end_of_sentence=False,
                )
            )
            sent_text = text

            await self._sleep_if_not_closed(delay - first_delay)
            text_data.forwarded_hyphens += word_hyphens

        self._event_ch.send_nowait(
            _TextSegment(
                delta=sentence[len(sent_text) :],
                stream_id=audio_data.id,
                sentence_id=seg_id,
                language=self._opts.language,
                end_of_sentence=True,
            )
        )
        sent_text = sentence

        if not real_speed and audio_data.vad_data and audio_data.vad_data.pushed_duration >= elapsed_time:
            pauses_s = elapsed_time - audio_data.vad_data.get_speak_duration(elapsed_time)[0]
            new_sentence_delay = max(
                0,
                pauses_s * self._opts.pause_to_new_sentence_delay_factor
                - text_data.new_sentence_pauses_s,
            )
            logger.info(
                f"new_sentence_delay: {new_sentence_delay}, pauses_s: {pauses_s}, "
                f"text_data.new_sentence_pauses_s: {text_data.new_sentence_pauses_s}"
            )
        else:
            new_sentence_delay = self._opts.new_sentence_delay

        new_sentence_delay = self._opts.new_sentence_delay
        await self._sleep_if_not_closed(new_sentence_delay)
        text_data.new_sentence_pauses_s += new_sentence_delay

    async def _vad_stream_task(self, audio_data: _AudioData) -> None:
        if audio_data.vad_stream is None:
            return

        async for ev in audio_data.vad_stream:
            audio_data.vad_data.append(ev.timestamp, ev.speaking, ev.syllable_rate)

    async def _sleep_if_not_closed(self, delay: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait([self._close_future], timeout=delay)

    def _calc_hyphens(self, text: str) -> list[str]:
        """Calculate hyphens for text."""
        hyphens: list[str] = []
        words: list[tuple[str, int, int]] = self._opts.split_words(text=text)
        for word, _, _ in words:
            new = self._opts.hyphenate_word(word)
            hyphens.extend(new)
        return hyphens

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("TranscriptionSyncer is closed")


class TextSynchronizer:
    def __init__(
        self,
        audio_output: AudioOutput,
        text_output: TextOutput,
        *,
        sync_options: NotGivenOr[TextSyncOptions] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        self._closed = False
        self._sync_options = sync_options or TextSyncOptions()
        self._synchronizer = _TextAudioSynchronizer(options=self._sync_options)
        self._sync_enabled = True

        self._base_text_output = text_output
        self._text_output = _TextOutput(self)
        self._audio_output = _AudioSyncOutput(audio_output, self)
        self._text_attached = True
        self._audio_attached = True
        self._tasks: set[asyncio.Task] = set()
        self._main_task = asyncio.create_task(self._forward_event())

    def set_sync_enabled(self, enable: bool) -> None:
        if self._sync_enabled == enable:
            return

        self._sync_enabled = enable
        self._flush()

    @property
    def audio_output(self) -> _AudioSyncOutput:
        """Get the audio output wrapper"""
        return self._audio_output

    @property
    def text_output(self) -> _TextOutput:
        """Get the text output wrapper"""
        return self._text_output

    async def _forward_event(self) -> None:
        last_stream_id: str | None = None

        while not self._closed:
            async for segment in self._synchronizer:
                if last_stream_id != segment.stream_id:
                    self._base_text_output.flush()
                    last_stream_id = segment.stream_id

                await self._base_text_output.capture_text(segment.delta)

            self._base_text_output.flush()

    def _flush(self) -> None:
        """Close the old transcription segment and create a new one"""
        current_synchronizer = self._synchronizer
        self._synchronizer = _TextAudioSynchronizer(options=self._sync_options)
        # close the old synchronizer
        task = asyncio.create_task(current_synchronizer.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_output_attach_changed(
        self, *, audio_attached: bool | None = None, text_attached: bool | None = None
    ) -> None:
        if audio_attached is not None:
            self._audio_attached = audio_attached
        if text_attached is not None:
            self._text_attached = text_attached

        self.set_sync_enabled(self._audio_attached and self._text_attached)

    async def aclose(self) -> None:
        """Close the forwarder and cleanup resources"""
        self._closed = True
        await self._synchronizer.aclose()

        await utils.aio.cancel_and_wait(self._main_task)
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()
        self._base_text_output.flush()


class _AudioSyncOutput(AudioOutput):
    def __init__(self, base_output: AudioOutput, parent: TextSynchronizer) -> None:
        super().__init__(sample_rate=base_output.sample_rate)
        self._parent = parent
        self._capturing = False
        self._interrupted = False

        self._base_output = base_output
        self._base_output.on("playback_finished", self._on_playback_finished)

    def set_base_output(self, base_output: AudioOutput) -> None:
        if self._base_output:
            self._base_output.off("playback_finished", self._on_playback_finished)
        self._base_output = base_output
        self._base_output.on("playback_finished", self._on_playback_finished)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        await self._base_output.capture_frame(frame)
        if not self._parent._sync_enabled:
            return

        if not self._capturing:
            self._parent._synchronizer.segment_playout_started()
            self._capturing = True
            self._interrupted = False

        self._parent._synchronizer.push_audio(frame)

    def flush(self) -> None:
        super().flush()
        self._base_output.flush()
        if not self._parent._sync_enabled:
            return

        self._capturing = False
        if not self._interrupted and not self._parent._synchronizer._closed:
            self._parent._synchronizer.mark_audio_segment_end()

    def clear_buffer(self) -> None:
        self._interrupted = True
        self._base_output.clear_buffer()

    def on_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        super().on_playback_finished(playback_position=playback_position, interrupted=interrupted)

        if not self._parent._sync_enabled:
            return

        if not interrupted and not self._parent._synchronizer._closed:
            self._parent._synchronizer.segment_playout_finished()
        self._parent._flush()

    def _on_playback_finished(self, ev: PlaybackFinishedEvent) -> None:
        self.on_playback_finished(
            playback_position=ev.playback_position, interrupted=ev.interrupted
        )

    def on_attached(self) -> None:
        self._parent._on_output_attach_changed(audio_attached=True)

    def on_detached(self) -> None:
        self._parent._on_output_attach_changed(audio_attached=False)


class _TextOutput(TextOutput):
    def __init__(self, parent: TextSynchronizer) -> None:
        super().__init__()
        self._parent = parent

    async def capture_text(self, text: str) -> None:
        if not self._parent._sync_enabled:
            await self._parent._base_text_output.capture_text(text)
            return

        self._parent._synchronizer.push_text(text)

    def flush(self) -> None:
        if not self._parent._sync_enabled:
            self._parent._base_text_output.flush()
            return

        self._parent._synchronizer.mark_text_segment_end()

    def on_attached(self) -> None:
        self._parent._on_output_attach_changed(text_attached=True)

    def on_detached(self) -> None:
        self._parent._on_output_attach_changed(text_attached=False)
