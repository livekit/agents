from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from livekit import rtc

from ... import tokenize, utils
from ...log import logger
from ...tokenize.tokenizer import PUNCTUATIONS
from ...types import NOT_GIVEN, NotGivenOr
from ..io import AudioSink, PlaybackFinishedEvent, TextSink
from . import _utils

# Standard speech rate in hyphens per second
STANDARD_SPEECH_RATE = 3.83


@dataclass
class TextSyncOptions:
    """Options for synchronizing TTS segments with audio playback."""

    language: str = ""
    speed: float = 1.0  # Multiplier of STANDARD_SPEECH_RATE
    new_sentence_delay: float = 0.4
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(
        retain_format=True
    )
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    split_words: Callable[[str], list[tuple[str, int, int]]] = tokenize.basic.split_words


@dataclass
class _AudioData:
    pushed_duration: float = 0.0
    done: bool = False
    id: str = field(default_factory=_utils.speech_uuid)


@dataclass
class _TextData:
    sentence_stream: tokenize.SentenceStream
    pushed_text: str = ""
    done: bool = False
    forwarded_hyphens: int = 0
    forwarded_sentences: int = 0


@dataclass
class _TextSegment:
    text: str
    speech_id: str
    segment_id: str
    language: str
    end_of_sentence: bool


class _TextAudioSynchronizer:
    """Synchronizes text with audio playback timing."""

    def __init__(self, options: TextSyncOptions):
        super().__init__()

        self._opts = options
        self._speed = options.speed * STANDARD_SPEECH_RATE

        self._closed = False
        self._close_future = asyncio.Future[None]()
        self._event_ch = utils.aio.Chan[_TextSegment]()

        self._playing_seg_index = -1
        self._finished_seg_index = -1

        self._text_q_changed = asyncio.Event()
        self._text_q = list[_TextData | None]()
        self._audio_q_changed = asyncio.Event()
        self._audio_q = list[_AudioData | None]()

        self._text_data: _TextData | None = None
        self._audio_data: _AudioData | None = None

        self._main_task: asyncio.Task | None = None

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push an audio frame for the current segment."""
        self._check_not_closed()

        if self._audio_data is None:
            self._audio_data = _AudioData()
            self._audio_q.append(self._audio_data)
            self._audio_q_changed.set()

        frame_duration = frame.samples_per_channel / frame.sample_rate
        self._audio_data.pushed_duration += frame_duration

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

        if self._audio_data is None:
            return

        assert self._audio_data is not None
        self._audio_data.done = True
        self._audio_data = None

    def mark_text_segment_end(self) -> None:
        """Mark the current text segment as complete."""
        self._check_not_closed()

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

        while not q_done:
            await self._text_q_changed.wait()
            await self._audio_q_changed.wait()

            while self._text_q and self._audio_q:
                text_data = self._text_q.pop(0)
                audio_data = self._audio_q.pop(0)

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
        if audio_data.pushed_duration > 0 and audio_data.done:
            real_speed = len(self._calc_hyphens(text_data.pushed_text)) / audio_data.pushed_duration

        seg_id = _utils.segment_uuid()
        words = self._opts.split_words(sentence)
        processed_words: list[str] = []

        sent_text = ""
        for word, start_pos, end_pos in words:
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
                estimated_pauses_s = text_data.forwarded_sentences * self._opts.new_sentence_delay
                hyph_pauses = estimated_pauses_s * speed

                target_hyphens = round(speed * elapsed_time)
                dt = target_hyphens - text_data.forwarded_hyphens - hyph_pauses
                to_wait_hyphens = max(0.0, word_hyphens - dt)
                delay = to_wait_hyphens / speed
            else:
                delay = word_hyphens / speed

            first_delay = min(delay / 2, 2 / speed)
            await self._sleep_if_not_closed(first_delay)

            self._event_ch.send_nowait(
                _TextSegment(
                    text=text[len(sent_text) :],
                    speech_id=audio_data.id,
                    segment_id=seg_id,
                    language=self._opts.language,
                    end_of_sentence=False,
                )
            )
            sent_text = text

            await self._sleep_if_not_closed(delay - first_delay)
            text_data.forwarded_hyphens += word_hyphens

        self._event_ch.send_nowait(
            _TextSegment(
                text=sentence[len(sent_text) :],
                speech_id=audio_data.id,
                segment_id=seg_id,
                language=self._opts.language,
                end_of_sentence=True,
            )
        )
        sent_text = sentence

        await self._sleep_if_not_closed(self._opts.new_sentence_delay)
        text_data.forwarded_sentences += 1

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
        audio_sink: AudioSink,
        sentence_text_sink: TextSink | None = None,
        speech_text_sink: TextSink | None = None,
        *,
        sync_options: NotGivenOr[TextSyncOptions] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        self._closed = False
        self._sync_options = sync_options or TextSyncOptions()
        self._synchronizer = _TextAudioSynchronizer(options=self._sync_options)

        # flush for each sentence
        self._sentence_text_sink = sentence_text_sink
        # flush for each speech
        self._speech_text_sink = speech_text_sink

        self._text_sink = _TextSink(self)
        self._audio_sink = _AudioSync(audio_sink, self)

        self._tasks: set[asyncio.Task] = set()
        self._main_task = asyncio.create_task(self._forward_event())

    @property
    def audio_sink(self) -> "_AudioSync":
        """Get the audio sink wrapper"""
        return self._audio_sink

    @property
    def text_sink(self) -> "_TextSink":
        """Get the text sink wrapper"""
        return self._text_sink

    async def _forward_event(self) -> None:
        last_speech_id: str | None = None
        last_sentence_id: str | None = None

        while not self._closed:
            async for segment in self._synchronizer:
                # flush if speech or sentence changed
                if last_speech_id != segment.speech_id:
                    last_speech_id = segment.speech_id
                    if self._speech_text_sink:
                        self._speech_text_sink.flush()

                if last_sentence_id != segment.segment_id:
                    last_sentence_id = segment.segment_id
                    if self._sentence_text_sink:
                        self._sentence_text_sink.flush()

                # capture text
                if self._sentence_text_sink:
                    await self._sentence_text_sink.capture_text(segment.text.strip("\n"))
                    if segment.end_of_sentence:
                        self._sentence_text_sink.flush()
                if self._speech_text_sink:
                    await self._speech_text_sink.capture_text(segment.text)

            # end of speech
            if self._speech_text_sink:
                self._speech_text_sink.flush()

    def _flush(self) -> None:
        """Close the old transcription segment and create a new one"""
        current_synchronizer = self._synchronizer
        self._synchronizer = _TextAudioSynchronizer(options=self._sync_options)
        # close the old synchronizer
        task = asyncio.create_task(current_synchronizer.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """Close the forwarder and cleanup resources"""
        self._closed = True
        await self._synchronizer.aclose()

        await utils.aio.cancel_and_wait(self._main_task)
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()
        if self._sentence_text_sink:
            self._sentence_text_sink.flush()
        if self._speech_text_sink:
            self._speech_text_sink.flush()


class _AudioSync(AudioSink):
    def __init__(self, base_sink: AudioSink, parent: TextSynchronizer) -> None:
        super().__init__(sample_rate=base_sink.sample_rate)
        self._parent = parent
        self._capturing = False
        self._interrupted = False

        self._base_sink = base_sink
        self._base_sink.on("playback_finished", self._on_playback_finished)

    def set_base_sink(self, base_sink: AudioSink) -> None:
        if self._base_sink:
            self._base_sink.off("playback_finished", self._on_playback_finished)
        self._base_sink = base_sink
        self._base_sink.on("playback_finished", self._on_playback_finished)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        await self._base_sink.capture_frame(frame)

        if not self._capturing:
            self._parent._synchronizer.segment_playout_started()
            self._capturing = True
            self._interrupted = False

        self._parent._synchronizer.push_audio(frame)

    def flush(self) -> None:
        super().flush()
        self._base_sink.flush()
        self._capturing = False
        if not self._interrupted and not self._parent._synchronizer._closed:
            self._parent._synchronizer.mark_audio_segment_end()

    def clear_buffer(self) -> None:
        self._interrupted = True
        self._base_sink.clear_buffer()

    def on_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        super().on_playback_finished(playback_position=playback_position, interrupted=interrupted)
        if not interrupted and not self._parent._synchronizer._closed:
            self._parent._synchronizer.segment_playout_finished()
        self._parent._flush()

    def _on_playback_finished(self, ev: PlaybackFinishedEvent) -> None:
        self.on_playback_finished(
            playback_position=ev.playback_position, interrupted=ev.interrupted
        )


class _TextSink(TextSink):
    def __init__(self, parent: TextSynchronizer) -> None:
        super().__init__()
        self._parent = parent

    async def capture_text(self, text: str) -> None:
        self._parent._synchronizer.push_text(text)

    def flush(self) -> None:
        self._parent._synchronizer.mark_text_segment_end()
