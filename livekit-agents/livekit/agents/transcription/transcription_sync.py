import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Optional, Literal

from livekit import rtc

from .. import tokenize, utils
from ..log import logger
from ..pipeline.io import AudioSink, PlaybackFinishedEvent, TextSink
from ..tokenize.tokenizer import PUNCTUATIONS
from ..utils import aio
from . import _utils

# Standard speech rate in hyphens per second
STANDARD_SPEECH_RATE = 3.83


@dataclass
class _AudioData:
    pushed_duration: float = 0.0
    done: bool = False


@dataclass
class _TextData:
    sentence_stream: tokenize.SentenceStream
    pushed_text: str = ""
    done: bool = False
    forwarded_hyphens: int = 0
    forwarded_sentences: int = 0


@dataclass
class TranscriptionSyncOptions:
    """Options for synchronizing TTS segments with audio playback."""

    language: str = ""
    speed: float = 1.0  # Multiplier of STANDARD_SPEECH_RATE
    new_sentence_delay: float = 0.4
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
        ignore_punctuation=False
    )
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word


class TranscriptionSynchronizer(rtc.EventEmitter[Literal["transcription_segment"]]):
    """Synchronizes TTS segments with audio playback timing."""

    def __init__(self, options: TranscriptionSyncOptions):
        super().__init__()

        self._opts = options
        self._speed = options.speed * STANDARD_SPEECH_RATE

        self._closed = False
        self._close_future = asyncio.Future[None]()

        self._playing_seg_index = -1
        self._finished_seg_index = -1

        self._text_q_changed = asyncio.Event()
        self._text_q = list[_TextData | None]()
        self._audio_q_changed = asyncio.Event()
        self._audio_q = list[_AudioData | None]()

        self._text_data: _TextData | None = None
        self._audio_data: _AudioData | None = None

        self._played_text = ""
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
            self._text_data = _TextData(
                sentence_stream=self._opts.sentence_tokenizer.stream()
            )
            self._text_q.append(self._text_data)
            self._text_q_changed.set()

        self._text_data.pushed_text += text
        self._text_data.sentence_stream.push_text(text)

    def mark_audio_segment_end(self) -> None:
        """Mark the current audio segment as complete."""
        self._check_not_closed()

        if self._audio_data is None:
            # Create empty audio data if none exists
            self.push_audio(rtc.AudioFrame(bytes(), 24000, 1, 0))

        assert self._audio_data is not None
        self._audio_data.done = True
        self._audio_data = None

    def mark_text_segment_end(self) -> None:
        """Mark the current text segment as complete."""
        self._check_not_closed()

        if self._text_data is None:
            self.push_text("")

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

    @property
    def played_text(self) -> str:
        """Currently played text."""
        return self._played_text

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
            real_speed = (
                len(self._calc_hyphens(text_data.pushed_text))
                / audio_data.pushed_duration
            )

        seg_id = _utils.segment_uuid()
        words = self._opts.word_tokenizer.tokenize(text=sentence)
        processed_words: list[str] = []

        og_text = self._played_text
        for word in words:
            if segment_index <= self._finished_seg_index:
                break

            if self._closed:
                return

            word_hyphens = len(self._opts.hyphenate_word(word))
            processed_words.append(word)

            elapsed_time = time.time() - segment_start_time
            text = self._opts.word_tokenizer.format_words(processed_words)
            text = text.rstrip("".join(PUNCTUATIONS))

            speed = self._speed
            if real_speed is not None:
                speed = real_speed
                estimated_pauses_s = (
                    text_data.forwarded_sentences * self._opts.new_sentence_delay
                )
                hyph_pauses = estimated_pauses_s * speed

                target_hyphens = round(speed * elapsed_time)
                dt = target_hyphens - text_data.forwarded_hyphens - hyph_pauses
                to_wait_hyphens = max(0.0, word_hyphens - dt)
                delay = to_wait_hyphens / speed
            else:
                delay = word_hyphens / speed

            first_delay = min(delay / 2, 2 / speed)
            await self._sleep_if_not_closed(first_delay)

            self.emit(
                "transcription_segment",
                rtc.TranscriptionSegment(
                    id=seg_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=False,
                    language=self._opts.language,
                ),
            )
            self._played_text = f"{og_text} {text}"

            await self._sleep_if_not_closed(delay - first_delay)
            text_data.forwarded_hyphens += word_hyphens

        self.emit(
            "transcription_segment",
            rtc.TranscriptionSegment(
                id=seg_id,
                text=sentence,
                start_time=0,
                end_time=0,
                final=True,
                language=self._opts.language,
            ),
        )
        self._played_text = f"{og_text} {sentence}"

        await self._sleep_if_not_closed(self._opts.new_sentence_delay)
        text_data.forwarded_sentences += 1

    async def _sleep_if_not_closed(self, delay: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait([self._close_future], timeout=delay)

    def _calc_hyphens(self, text: str) -> list[str]:
        """Calculate hyphens for text."""
        hyphens: list[str] = []
        words = self._opts.word_tokenizer.tokenize(text=text)
        for word in words:
            new = self._opts.hyphenate_word(word)
            hyphens.extend(new)
        return hyphens

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("TranscriptionSyncer is closed")


class TranscriptionSyncIO(
    rtc.EventEmitter[Literal["transcription_segment", "segment_playout_started"]]
):
    def __init__(
        self,
        audio_sink: AudioSink,
        text_sink: Optional[TextSink] = None,
        sync_options: Optional[TranscriptionSyncOptions] = None,
    ) -> None:
        """Initialize the TTSForwarderOutput

        Args:
            audio_sink: The base audio sink to forward audio to
            text_sink: Optional base text sink to forward text to
            sync_options: Optional TTSSegmentsSyncOptions for configuring the sync behavior
        """
        super().__init__()

        self._sync_options = sync_options or TranscriptionSyncOptions()
        self._transcription_sync = self._create_transcription_sync()

        self._text_sink = TranscriptionSyncTextSink(text_sink, self)
        self._audio_sink = TranscriptionSyncAudioSink(audio_sink, self)

    def segment_playout_started(self) -> None:
        self._transcription_sync.segment_playout_started()
        self.emit("segment_playout_started")

    def flush(self) -> None:
        logger.info("reset sync")
        asyncio.create_task(self._transcription_sync.aclose())
        self._transcription_sync = self._create_transcription_sync()
        logger.info("reset sync 2")

    def _create_transcription_sync(self) -> TranscriptionSynchronizer:
        def _on_segment(segment: rtc.TranscriptionSegment) -> None:
            self.emit("transcription_segment", segment)

        synchronizer = TranscriptionSynchronizer(options=self._sync_options)
        synchronizer.on("transcription_segment", _on_segment)
        return synchronizer

    @property
    def audio(self) -> "TranscriptionSyncAudioSink":
        """Get the audio sink wrapper"""
        return self._audio_sink

    @property
    def text(self) -> "TranscriptionSyncTextSink":
        """Get the text sink wrapper"""
        return self._text_sink

    async def aclose(self) -> None:
        """Close the forwarder and cleanup resources"""
        await self._transcription_sync.aclose()


class TranscriptionSyncAudioSink(AudioSink):
    def __init__(self, base_sink: AudioSink, parent: TranscriptionSyncIO) -> None:
        super().__init__(sample_rate=base_sink.sample_rate)
        self._base_sink = base_sink
        self._parent = parent
        self._capturing = False
        self._interrupted = False

        self._base_sink.on(
            "playback_finished",
            lambda ev: self.on_playback_finished(
                playback_position=ev.playback_position, interrupted=ev.interrupted
            ),
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        await self._base_sink.capture_frame(frame)

        if not self._capturing:
            self._parent.segment_playout_started()
            self._capturing = True
            self._interrupted = False

        self._parent._transcription_sync.push_audio(frame)
        logger.info(f"Pushed audio frame: {frame.duration}s")

    def flush(self) -> None:
        super().flush()
        self._base_sink.flush()
        self._capturing = False
        if not self._interrupted and not self._parent._transcription_sync._closed:
            self._parent._transcription_sync.mark_audio_segment_end()
            logger.info("Marked audio segment end")

    def clear_buffer(self) -> None:
        self._interrupted = True
        self._base_sink.clear_buffer()

    def on_playback_finished(
        self, *, playback_position: float, interrupted: bool
    ) -> None:
        super().on_playback_finished(
            playback_position=playback_position, interrupted=interrupted
        )
        if not interrupted and not self._parent._transcription_sync._closed:
            self._parent._transcription_sync.segment_playout_finished()
            logger.info("Marked audio playout end")
        self._parent.flush()
        logger.info("Reset transcription sync")


class TranscriptionSyncTextSink(TextSink):
    def __init__(
        self, base_sink: Optional[TextSink], parent: TranscriptionSyncIO
    ) -> None:
        super().__init__()
        self._base_sink = base_sink
        self._parent = parent

    async def capture_text(self, text: str) -> None:
        if self._base_sink:
            await self._base_sink.capture_text(text)
        self._parent._transcription_sync.push_text(text)

    def flush(self) -> None:
        if self._base_sink:
            self._base_sink.flush()

        self._parent._transcription_sync.mark_text_segment_end()
