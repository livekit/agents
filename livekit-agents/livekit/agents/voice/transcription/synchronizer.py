from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from livekit import rtc

from ... import tokenize, utils
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from .. import io
from ._speaking_rate import SpeakingRateDetector, SpeakingRateStream

STANDARD_SPEECH_RATE = 3.83  # hyphens (syllables) per second


@dataclass
class _TextSyncOptions:
    speed: float
    hyphenate_word: Callable[[str], list[str]]
    word_tokenizer: tokenize.WordTokenizer
    speaking_rate_detector: SpeakingRateDetector


@dataclass
class _SpeakingRateData:
    timestamps: list[float] = field(default_factory=list)
    """timestamps of the speaking rate"""

    speaking_rate: list[float] = field(default_factory=list)
    """speed at the timestamp"""

    speak_integrals: list[float] = field(default_factory=list)
    """accumulated speaking units up to the timestamp"""

    _text_buffer: list[str] = field(default_factory=list)

    def add_by_rate(self, *, timestamp: float, speaking_rate: float) -> None:
        integral = self.speak_integrals[-1] if self.timestamps else 0
        dt = timestamp - self.pushed_duration
        integral += speaking_rate * dt

        self.timestamps.append(timestamp)
        self.speaking_rate.append(speaking_rate)
        self.speak_integrals.append(integral)

    def add_by_annotation(
        self,
        *,
        text: str,
        start_time: float | None,
        end_time: float | None,
    ) -> None:
        if start_time is not None:
            # calculate the integral of the speaking rate up to the start time
            integral = self.speak_integrals[-1] if self.timestamps else 0

            dt = start_time - self.pushed_duration
            # use the length of the text directly instead of hyphens
            text_len = sum(len(text) for text in self._text_buffer)
            integral += text_len
            rate = text_len / dt if dt > 0 else 0

            self.timestamps.append(start_time)
            self.speaking_rate.append(rate)
            self.speak_integrals.append(integral)
            self._text_buffer.clear()

        self._text_buffer.append(text)

        if end_time is not None:
            self.add_by_annotation(text="", start_time=end_time, end_time=None)

    def accumulate_to(self, timestamp: float) -> float:
        """Get accumulated speaking units up to the given timestamp."""
        if not self.timestamps:
            return 0

        idx = np.searchsorted(self.timestamps, timestamp, side="right")
        if idx == 0:
            return 0

        integral_t = self.speak_integrals[idx - 1]

        # fill the tail assuming the speaking rate is constant
        dt = timestamp - self.timestamps[idx - 1]
        rate = (
            self.speaking_rate[idx]
            if idx < len(self.speaking_rate)
            else self.speaking_rate[idx - 1]
        )
        integral_t += rate * dt

        if idx < len(self.timestamps):
            # if there is a next timestamp, make sure the integral does not exceed the next
            integral_t = min(integral_t, self.speak_integrals[idx])

        return integral_t

    @property
    def pushed_duration(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0


@dataclass
class _AudioData:
    sr_stream: SpeakingRateStream  # speaking rate estimation
    pushed_duration: float = 0.0
    done: bool = False
    estimated_rate: _SpeakingRateData = field(default_factory=_SpeakingRateData)
    annotated_rate: _SpeakingRateData | None = None  # speaking rate from `start_time`


@dataclass
class _TextData:
    word_stream: tokenize.WordStream
    pushed_text: str = ""
    done: bool = False
    forwarded_hyphens: int = 0
    forwarded_text: str = ""


class _SegmentSynchronizerImpl:
    """Synchronizes one text segment with one audio segment"""

    def __init__(self, options: _TextSyncOptions, *, next_in_chain: io.TextOutput) -> None:
        self._opts = options
        self._text_data = _TextData(word_stream=self._opts.word_tokenizer.stream())
        self._audio_data = _AudioData(sr_stream=self._opts.speaking_rate_detector.stream())

        self._next_in_chain = next_in_chain
        self._start_wall_time: float | None = None
        self._start_fut: asyncio.Event = asyncio.Event()

        self._paused_wall_time: float | None = None
        self._paused_duration: float = 0.0
        self._output_enabled_ev = asyncio.Event()
        self._output_enabled_ev.set()

        self._speed = STANDARD_SPEECH_RATE * self._opts.speed  # hyphens per second
        self._speed_on_speaking_unit: float | None = None  # hyphens per speaking unit
        # a speaking unit is defined by the speaking rate estimation method, it's a relative unit

        self._out_ch = utils.aio.Chan[str]()
        self._close_future = asyncio.Future[None]()

        self._main_atask = asyncio.create_task(self._main_task())
        self._main_atask.add_done_callback(lambda _: self._out_ch.close())
        self._capture_atask = asyncio.create_task(self._capture_task())
        self._speaking_rate_atask = asyncio.create_task(self._speaking_rate_task())

        self._playback_completed = False
        self._interrupted = False

    @property
    def closed(self) -> bool:
        return self._close_future.done()

    @property
    def audio_input_ended(self) -> bool:
        return self._audio_data.done

    @property
    def text_input_ended(self) -> bool:
        return self._text_data.done

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.push_audio called after close")
            return

        # the first audio frame we receive marks the start of the sync
        # see `TranscriptSynchronizer` docstring
        if self._start_wall_time is None and frame.duration > 0:
            self._start_wall_time = time.time()
            self._start_fut.set()

        self._audio_data.sr_stream.push_frame(frame)
        self._audio_data.pushed_duration += frame.duration

    def end_audio_input(self) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.end_audio_input called after close")
            return

        self._audio_data.done = True
        self._audio_data.sr_stream.end_input()
        self._reestimate_speed()

    def push_text(self, text: str) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.push_text called after close")
            return

        start_time, end_time = None, None
        if isinstance(text, io.TimedString):
            start_time = text.start_time if utils.is_given(text.start_time) else None
            end_time = text.end_time if utils.is_given(text.end_time) else None
            if not self._audio_data.annotated_rate:
                self._audio_data.annotated_rate = _SpeakingRateData()

            # accumulate the actual text length if time annotations are present
            self._audio_data.annotated_rate.add_by_annotation(
                text=text, start_time=start_time, end_time=end_time
            )

        self._text_data.word_stream.push_text(text)
        self._text_data.pushed_text += text

    def end_text_input(self) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.end_text_input called after close")
            return

        self._text_data.done = True
        self._text_data.word_stream.end_input()

        self._reestimate_speed()

    def pause(self) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.pause called after close")
            return

        if self._paused_wall_time is None:
            self._paused_wall_time = time.time()
        self._output_enabled_ev.clear()

    def resume(self) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.resume called after close")
            return

        if self._paused_wall_time is not None:
            self._paused_duration += time.time() - self._paused_wall_time
            self._paused_wall_time = None
        self._output_enabled_ev.set()

    def _reestimate_speed(self) -> None:
        if not self._text_data.done or not self._audio_data.done:
            return

        pushed_hyphens = len(self._calc_hyphens(self._text_data.pushed_text))
        # hyphens per second
        if self._audio_data.pushed_duration > 0:
            self._speed = pushed_hyphens / self._audio_data.pushed_duration

        # hyphens per speaking unit
        pushed_speaking_units = self._audio_data.estimated_rate.accumulate_to(
            self._audio_data.pushed_duration
        )
        if pushed_speaking_units > 0:
            self._speed_on_speaking_unit = pushed_hyphens / pushed_speaking_units

    def mark_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.playback_finished called after close")
            return

        self._interrupted = interrupted
        if not self._text_data.done or not self._audio_data.done:
            logger.warning(
                "_SegmentSynchronizerImpl.playback_finished called before text/audio input is done",
                extra={"text_done": self._text_data.done, "audio_done": self._audio_data.done},
            )
            return

        # if the playback of the segment is done and were not interrupted, make sure the whole
        # transcript is sent. (In case we're late)
        if not interrupted:
            self._playback_completed = True

    @property
    def synchronized_transcript(self) -> str:
        if self._playback_completed:
            return self._text_data.pushed_text

        return self._text_data.forwarded_text

    @utils.log_exceptions(logger=logger)
    async def _capture_task(self) -> None:
        try:
            async for text in self._out_ch:
                await self._next_in_chain.capture_text(text)
        finally:
            self._next_in_chain.flush()

    @utils.log_exceptions(logger=logger)
    async def _speaking_rate_task(self) -> None:
        async for ev in self._audio_data.sr_stream:
            self._audio_data.estimated_rate.add_by_rate(
                timestamp=ev.timestamp, speaking_rate=ev.speaking_rate
            )

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        await self._start_fut.wait()

        if self.closed and not self._playback_completed:
            return

        assert self._start_wall_time is not None

        async for data in self._text_data.word_stream:
            word = data.token

            if not self._output_enabled_ev.is_set():
                await self._output_enabled_ev.wait()
                if self._interrupted:
                    return

            if self.closed and not self._playback_completed:
                return

            if self._playback_completed:
                self._out_ch.send_nowait(word)
                continue

            word_hyphens = len(self._opts.hyphenate_word(word))
            elapsed = time.time() - self._start_wall_time - self._paused_duration

            d_hyphens = 0
            if (annotated := self._audio_data.annotated_rate) and (
                annotated.pushed_duration >= elapsed
            ):
                # use the actual speaking rate
                target_len = int(annotated.accumulate_to(elapsed))
                forwarded_len = len(self._text_data.forwarded_text)
                if target_len >= forwarded_len:
                    d_text = self._text_data.pushed_text[forwarded_len:target_len]
                    d_hyphens = len(self._calc_hyphens(d_text))
                else:
                    d_text = self._text_data.pushed_text[target_len:forwarded_len]
                    d_hyphens = -len(self._calc_hyphens(d_text))

            elif self._speed_on_speaking_unit:
                # use the estimated speed from speaking rate
                target_speaking_units = self._audio_data.estimated_rate.accumulate_to(elapsed)
                target_hyphens = target_speaking_units * self._speed_on_speaking_unit
                d_hyphens = np.ceil(target_hyphens) - self._text_data.forwarded_hyphens

            delay = max(0.0, word_hyphens - d_hyphens) / self._speed

            # if playback completed, flush the word as soon as possible
            if self._playback_completed:
                delay = 0

            await self._sleep_if_not_closed(delay / 2.0)
            self._out_ch.send_nowait(word)
            await self._sleep_if_not_closed(delay / 2.0)

            self._text_data.forwarded_hyphens += word_hyphens
            self._text_data.forwarded_text += word

    def _calc_hyphens(self, text: str) -> list[str]:
        """Calculate hyphens for text."""
        words = self._opts.word_tokenizer.tokenize(text)
        hyphens = list(
            itertools.chain.from_iterable(self._opts.hyphenate_word(word) for word in words)
        )
        return hyphens

    async def _sleep_if_not_closed(self, delay: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait([self._close_future], timeout=delay)

    async def aclose(self) -> None:
        if self.closed:
            return

        self._close_future.set_result(None)
        self._start_fut.set()  # avoid deadlock of main_task in case it never started
        self._output_enabled_ev.set()
        await self._text_data.word_stream.aclose()
        await self._audio_data.sr_stream.aclose()
        await self._capture_atask
        await self._speaking_rate_atask


class TranscriptSynchronizer:
    """
    Synchronizes text with audio playback timing.

    This class is responsible for synchronizing text with audio playback timing.
    It currently assumes that the first push_audio is starting the audio playback of a segment.
    """

    def __init__(
        self,
        *,
        next_in_chain_audio: io.AudioOutput,
        next_in_chain_text: io.TextOutput,
        speed: float = 1.0,
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
    ) -> None:
        super().__init__()

        self._text_output = _SyncedTextOutput(self, next_in_chain=next_in_chain_text)
        self._audio_output = _SyncedAudioOutput(self, next_in_chain=next_in_chain_audio)
        self._text_attached, self._audio_attached = True, True
        self._opts = _TextSyncOptions(
            speed=speed,
            hyphenate_word=hyphenate_word,
            word_tokenizer=(
                word_tokenizer
                or tokenize.basic.WordTokenizer(
                    retain_format=True, ignore_punctuation=False, split_character=True
                )
            ),
            speaking_rate_detector=SpeakingRateDetector(),
        )
        self._enabled = True
        self._closed = False

        # initial segment/first segment, recreated for each new segment
        self._impl = _SegmentSynchronizerImpl(options=self._opts, next_in_chain=next_in_chain_text)
        self._rotate_segment_atask: asyncio.Task[None] | None = None

    @property
    def audio_output(self) -> _SyncedAudioOutput:
        return self._audio_output

    @property
    def text_output(self) -> _SyncedTextOutput:
        return self._text_output

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def aclose(self) -> None:
        self._closed = True
        await self.barrier()
        await self._impl.aclose()

    def set_enabled(self, enabled: bool) -> None:
        if self._enabled == enabled:
            return

        self._enabled = enabled
        if not self._rotate_segment_atask or self._rotate_segment_atask.done():
            # avoid calling rotate_segment twice when closing the session during agent speaking
            # first time when speech interrupted, second time here when output detached
            self.rotate_segment()

    def _on_attachment_changed(
        self,
        *,
        audio_attached: NotGivenOr[bool] = NOT_GIVEN,
        text_attached: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(audio_attached):
            self._audio_attached = audio_attached

        if is_given(text_attached):
            self._text_attached = text_attached

        self.set_enabled(self._audio_attached and self._text_attached)

    async def _rotate_segment_task(self, old_task: asyncio.Task[None] | None) -> None:
        if old_task:
            await old_task

        await self._impl.aclose()
        self._impl = _SegmentSynchronizerImpl(
            options=self._opts, next_in_chain=self._text_output._next_in_chain
        )

    def rotate_segment(self) -> None:
        if self._closed:
            return

        if self._rotate_segment_atask and not self._rotate_segment_atask.done():
            logger.warning("rotate_segment called while previous segment is still being rotated")

        self._rotate_segment_atask = asyncio.create_task(
            self._rotate_segment_task(self._rotate_segment_atask)
        )

    async def barrier(self) -> None:
        if self._rotate_segment_atask is None:
            return

        # using a while loop in case rotate_segment is called twice (this should not happen, but
        # just in case, we do log a warning if it does)
        while not self._rotate_segment_atask.done():
            await self._rotate_segment_atask


class _SyncedAudioOutput(io.AudioOutput):
    def __init__(
        self, synchronizer: TranscriptSynchronizer, *, next_in_chain: io.AudioOutput
    ) -> None:
        super().__init__(
            label="TranscriptSynchronizer",
            next_in_chain=next_in_chain,
            sample_rate=next_in_chain.sample_rate,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._next_in_chain: io.AudioOutput = next_in_chain  # redefined for better typing
        self._synchronizer = synchronizer
        self._capturing = False
        self._pushed_duration: float = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        # using barrier() on capture should be sufficient, flush() must not be called if
        # capture_frame isn't completed
        await self._synchronizer.barrier()

        self._capturing = True
        await super().capture_frame(frame)
        await self._next_in_chain.capture_frame(frame)  # passthrough audio
        self._pushed_duration += frame.duration

        if not self._synchronizer.enabled:
            return

        if self._synchronizer._impl.audio_input_ended:
            # this should not happen if `on_playback_finished` is called after each flush
            logger.warning(
                "_SegmentSynchronizerImpl audio marked as ended in capture audio, rotating segment"
            )
            self._synchronizer.rotate_segment()
            await self._synchronizer.barrier()

        self._synchronizer._impl.push_audio(frame)

    def flush(self) -> None:
        super().flush()
        self._next_in_chain.flush()

        if not self._synchronizer.enabled:
            return

        if not self._pushed_duration:
            # in case there is no audio after text was pushed, rotate the segment
            self._synchronizer.rotate_segment()
            return

        self._capturing = False
        self._synchronizer._impl.end_audio_input()

    def clear_buffer(self) -> None:
        self._next_in_chain.clear_buffer()
        self._capturing = False

    # this is going to be automatically called by the next_in_chain
    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        if not self._synchronizer.enabled:
            super().on_playback_finished(
                playback_position=playback_position,
                interrupted=interrupted,
                synchronized_transcript=synchronized_transcript,
            )
            return

        self._synchronizer._impl.mark_playback_finished(
            playback_position=playback_position, interrupted=interrupted
        )
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=self._synchronizer._impl.synchronized_transcript,
        )

        self._synchronizer.rotate_segment()
        self._pushed_duration = 0.0

    def on_attached(self) -> None:
        super().on_attached()
        self._synchronizer._on_attachment_changed(audio_attached=True)

    def on_detached(self) -> None:
        super().on_detached()
        self._synchronizer._on_attachment_changed(audio_attached=False)

    def pause(self) -> None:
        super().pause()
        self._synchronizer._impl.pause()

    def resume(self) -> None:
        super().resume()
        self._synchronizer._impl.resume()


class _SyncedTextOutput(io.TextOutput):
    def __init__(
        self, synchronizer: TranscriptSynchronizer, *, next_in_chain: io.TextOutput
    ) -> None:
        super().__init__(label="TranscriptSynchronizer", next_in_chain=next_in_chain)
        self._next_in_chain: io.TextOutput = next_in_chain  # redefined for better typing
        self._synchronizer = synchronizer
        self._capturing = False

    async def capture_text(self, text: str) -> None:
        await self._synchronizer.barrier()

        if not self._synchronizer.enabled:  # passthrough text if the synchronizer is disabled
            await self._next_in_chain.capture_text(text)
            return

        self._capturing = True
        if self._synchronizer._impl.text_input_ended:
            # this should not happen if `on_playback_finished` is called after each flush
            logger.warning(
                "_SegmentSynchronizerImpl text marked as ended in capture text, rotating segment"
            )
            self._synchronizer.rotate_segment()
            await self._synchronizer.barrier()

        self._synchronizer._impl.push_text(text)

    def flush(self) -> None:
        if not self._synchronizer.enabled:  # passthrough text if the synchronizer is disabled
            self._next_in_chain.flush()
            return

        if not self._capturing:
            return

        self._capturing = False
        self._synchronizer._impl.end_text_input()

    def on_attached(self) -> None:
        super().on_attached()
        self._synchronizer._on_attachment_changed(text_attached=True)

    def on_detached(self) -> None:
        super().on_detached()
        self._synchronizer._on_attachment_changed(text_attached=False)
