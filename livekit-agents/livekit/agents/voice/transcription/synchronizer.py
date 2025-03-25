from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from dataclasses import dataclass
from typing import Callable

from livekit import rtc

from ... import tokenize, utils
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from .. import io

STANDARD_SPEECH_RATE = 3.83  # hyphens (syllables) per second


@dataclass
class _TextSyncOptions:
    speed: float
    hyphenate_word: Callable[[str], list[str]]
    split_words: Callable[[str], list[tuple[str, int, int]]]


@dataclass
class _TextSegment:
    text: str
    start_time: float | None  # When the transcript should start playing
    end_time: float | None  # When the transcript should be fully played


@dataclass
class _AudioSegment:
    frame: rtc.AudioFrame
    # TODO(theomonnom): support TTS alignments


class _SegmentSynchronizerImpl:
    """Synchronizes one text segment with one audio segment"""

    def __init__(self, options: _TextSyncOptions, *, next_in_chain: io.TextOutput) -> None:
        self._text_ch = utils.aio.Chan[_TextSegment]()
        self._audio_ch = utils.aio.Chan[_AudioSegment]()
        self._text_segments: list[_TextSegment] = []
        self._audio_segments: list[_AudioSegment] = []
        self._next_in_chain = next_in_chain

        self._opts = options
        self._start_wall_time: float | None = None
        self._start_fut: asyncio.Event = asyncio.Event()

        self._speed = STANDARD_SPEECH_RATE * self._opts.speed
        self._out_ch = utils.aio.Chan[str]()
        self._close_future = asyncio.Future[None]()

        self._main_atask = asyncio.create_task(self._main_task())
        self._main_atask.add_done_callback(lambda _: self._out_ch.close())
        self._capture_atask = asyncio.create_task(self._capture_task())

        self._playback_completed = False

    @property
    def closed(self) -> bool:
        return self._close_future.done()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.push_audio called after close")
            return

        # the first audio frame we receive marks the start of the sync
        # see `TranscriptSynchronizer` docstring
        if self._start_wall_time is None and frame.duration > 0:
            self._start_wall_time = time.time()
            self._start_fut.set()

        seg = _AudioSegment(frame=frame)
        self._audio_segments.append(seg)
        self._audio_ch.send_nowait(seg)

    def end_audio_input(self) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.end_audio_input called after close")
            return

        with contextlib.suppress(utils.aio.ChanClosed):
            self._audio_ch.close()

        self._reestimate_speed()

    def push_text(self, text: str) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.push_text called after close")
            return

        start_time, end_time = None, None
        if isinstance(text, io.TimedString):
            start_time = text.start_time or None
            end_time = text.end_time or None

        seg = _TextSegment(text=text, start_time=start_time, end_time=end_time)
        print(seg)
        self._text_segments.append(seg)
        self._text_ch.send_nowait(seg)

    def end_text_input(self) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.end_text_input called after close")
            return

        with contextlib.suppress(utils.aio.ChanClosed):
            self._text_ch.close()

        self._reestimate_speed()

    def _reestimate_speed(self) -> None:
        if not self._text_ch.closed or not self._audio_ch.closed:
            return

        # calculate the true syllables per second based on the complete text and audio input
        words = self._opts.split_words("".join(seg.text for seg in self._text_segments))
        total_syllables = sum(len(self._opts.hyphenate_word(w)) for w, _, _ in words)
        total_duration = sum(seg.frame.duration for seg in self._audio_segments)
        self._speed = total_syllables / total_duration

    def playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        if self.closed:
            logger.warning("_SegmentSynchronizerImpl.playback_finished called after close")
            return

        if not self._text_ch.closed or not self._audio_ch.closed:
            logger.warning(
                "_SegmentSynchronizerImpl.playback_finished called before text/audio input closed"
            )
            return

        # if the playback of the segment is done and were not interrupted, make sure the whole
        # transcript is sent. (In case we're late)
        if not interrupted:
            self._playback_completed = True

    @utils.log_exceptions(logger=logger)
    async def _capture_task(self) -> None:
        try:
            async for text in self._out_ch:
                await self._next_in_chain.capture_text(text)
        finally:
            self._next_in_chain.flush()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        await self._start_fut.wait()

        if self.closed and not self._playback_completed:
            return

        assert self._start_wall_time is not None

        i = 0
        async for text_seg in self._text_ch:
            i += 1

            base_start = text_seg.start_time or 0.0
            elapsed = time.time() - self._start_wall_time

            if (delay_until_start := base_start - elapsed) > 0:
                await self._sleep_if_not_closed(delay_until_start)

            # check if closed after waiting for the base_start time (it could be several seconds in
            # extreme cases)
            # However, if playback *is* completed, we keep going so we can flush out
            # all remaining text segments without pacing delays.
            if self.closed and not self._playback_completed:
                return

            acc_syllables = 0
            text_cursor = 0
            for word, _, end_pos in self._opts.split_words(text_seg.text):
                if self._playback_completed:
                    self._out_ch.send_nowait(text_seg.text[text_cursor:end_pos])
                    continue

                word_syllables = len(self._opts.hyphenate_word(word))
                elapsed = time.time() - self._start_wall_time

                # since the speed can change at anytime, calculate the word delay based on
                # the "target_syllables"
                target_syllables = round(self._speed * (elapsed - base_start))
                dt = target_syllables - acc_syllables
                to_wait_hyphens = max(0.0, word_syllables - dt)
                delay = to_wait_hyphens / self._speed

                # if the next text segment should have started, flush the word as soon as possible
                # for this current segment to catch up
                if (i + 1) < len(self._text_segments):
                    next_seg = self._text_segments[i + 1]
                    if next_seg.start_time is not None and next_seg.start_time <= elapsed:
                        delay = 0

                # if playback completed, flush the word as soon as possible
                if self._playback_completed:
                    delay = 0

                await self._sleep_if_not_closed(delay / 2.0)
                self._out_ch.send_nowait(text_seg.text[text_cursor:end_pos])
                await self._sleep_if_not_closed(delay / 2.0)

                acc_syllables += word_syllables
                text_cursor = end_pos

    async def _sleep_if_not_closed(self, delay: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait([self._close_future], timeout=delay)

    async def aclose(self) -> None:
        if self.closed:
            return

        self._close_future.set_result(None)
        self._start_fut.set()  # avoid deadlock of main_task in case it never started
        # not using end_text_input/end_audio_input, because they may change the speed calculation
        self._text_ch.close()
        self._audio_ch.close()
        await self._capture_atask


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
        split_words: Callable[[str], list[tuple[str, int, int]]] = functools.partial(
            tokenize.basic.split_words, ignore_punctuation=False
        ),
    ) -> None:
        super().__init__()

        self._text_output = _SyncedTextOutput(self, next_in_chain=next_in_chain_text)
        self._audio_output = _SyncedAudioOutput(self, next_in_chain=next_in_chain_audio)
        self._text_attached, self._audio_attached = True, True
        self._opts = _TextSyncOptions(
            speed=speed, hyphenate_word=hyphenate_word, split_words=split_words
        )
        self._enabled = True
        self._closed = False

        # initial segment/first segment, recreated for each new segment
        self._impl = _SegmentSynchronizerImpl(options=self._opts, next_in_chain=next_in_chain_text)
        self._rotate_segment_atask = asyncio.create_task(self._rotate_segment_task())

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

    async def _rotate_segment_task(self) -> None:
        await self._impl.aclose()
        self._impl = _SegmentSynchronizerImpl(
            options=self._opts, next_in_chain=self._text_output._next_in_chain
        )

    def rotate_segment(self) -> None:
        if self._closed:
            return

        if not self._rotate_segment_atask.done():
            logger.warning("rotate_segment called while previous segment is still being rotated")

        self._rotate_segment_atask = asyncio.create_task(self._rotate_segment_task())

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
        super().__init__(next_in_chain=next_in_chain, sample_rate=next_in_chain.sample_rate)
        self._next_in_chain = next_in_chain  # redefined for better typing
        self._synchronizer = synchronizer
        self._capturing = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        # using barrier() on capture should be sufficient, flush() must not be called if
        # capture_frame isn't completed
        await self._synchronizer.barrier()

        self._capturing = True
        await super().capture_frame(frame)
        await self._next_in_chain.capture_frame(frame)  # passthrough audio

        if not self._synchronizer.enabled:
            return

        self._synchronizer._impl.push_audio(frame)

    def flush(self) -> None:
        super().flush()
        self._next_in_chain.flush()

        if not self._synchronizer.enabled or not self._capturing:
            return

        self._capturing = False
        self._synchronizer._impl.end_audio_input()

    def clear_buffer(self) -> None:
        super().clear_buffer()
        self._next_in_chain.clear_buffer()
        self._capturing = False

    # this is going to be automatically called by the next_in_chain
    def on_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        super().on_playback_finished(playback_position=playback_position, interrupted=interrupted)

        if not self._synchronizer.enabled:
            return

        self._synchronizer._impl.playback_finished(
            playback_position=playback_position, interrupted=interrupted
        )
        self._synchronizer.rotate_segment()

    def on_attached(self) -> None:
        super().on_attached()
        self._synchronizer._on_attachment_changed(audio_attached=True)

    def on_detached(self) -> None:
        super().on_detached()
        self._synchronizer._on_attachment_changed(audio_attached=False)


class _SyncedTextOutput(io.TextOutput):
    def __init__(
        self, synchronizer: TranscriptSynchronizer, *, next_in_chain: io.TextOutput
    ) -> None:
        super().__init__(next_in_chain=next_in_chain)
        self._next_in_chain = next_in_chain  # redefined for better typing
        self._synchronizer = synchronizer
        self._capturing = False

    async def capture_text(self, text: str) -> None:
        await self._synchronizer.barrier()

        await super().capture_text(text)
        if not self._synchronizer.enabled:  # passthrough text if the synchronizer is disabled
            await self._next_in_chain.capture_text(text)
            return

        self._capturing = True
        self._synchronizer._impl.push_text(text)

    def flush(self) -> None:
        super().flush()
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
