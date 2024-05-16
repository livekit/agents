from collections import deque
import math
import time
from attrs import define
import asyncio
import contextlib
import uuid
from typing import Callable, AsyncIterable

from attrs import define
from livekit import rtc

from .. import stt, tokenize, tts
from ..log import logger


@define
class TTSOptions:
    room: rtc.Room
    participant_identity: str
    track_id: str
    language: str
    speed: float
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]


@define
class _SegmentData:
    sentence_stream: tokenize.SentenceStream
    text: str
    audio_duration: float
    avg_speed: float | None
    processed_hyphenes: int


@define
class _PendingSegments:
    cur_audio: _SegmentData
    cur_text: _SegmentData
    q: deque[_SegmentData]


def _uuid() -> str:
    return str(uuid.uuid4())[:12]


class TTSSegmentsForwarder:
    def __init__(
        self,
        opts: TTSOptions,
    ):
        self._opts = opts
        self._main_task = asyncio.create_task(self._run())

        # current segment where the user may still be pushing text & audio
        first_segment = self._create_segment()
        segments_q = deque()
        segments_q.append(first_segment)

        self._pending_segment = _PendingSegments(
            cur_audio=first_segment,
            cur_text=first_segment,
            q=segments_q,
        )

        self._seg_queue = asyncio.Queue[_SegmentData | None]()
        self._seg_queue.put_nowait(first_segment)

        # the forwarding of the transcription is started as soon as the user push the first audio frame.
        self._start_future = asyncio.Future()

    def validate(self) -> None:
        """Validate must be called once we started to playout the audio to the user"""
        with contextlib.suppress(asyncio.InvalidStateError):
            self._start_future.set_result(None)

    def push_audio(self, frame: rtc.AudioFrame | None, **kwargs) -> None:
        if frame is not None:
            frame_duration = frame.samples_per_channel / frame.sample_rate
            cur_seg = self._pending_segment.cur_audio
            cur_seg.audio_duration += frame_duration
        else:
            self.mark_audio_segment_end()

    def mark_audio_segment_end(self) -> None:
        try:
            # get last ended segment (text always end before audio)
            seg = self._pending_segment.q.popleft()
        except IndexError:
            raise IndexError(
                "mark_audio_segment_end called before any mark_segment_end"
            )

        seg.avg_speed = len(self._calc_hyphenes(seg.text)) / seg.audio_duration
        self._pending_segment.cur_audio = self._pending_segment.q[0]

    def push_text(self, text: str | None) -> None:
        if text is not None:
            cur_seg = self._pending_segment.cur_text
            cur_seg.text += text
            cur_seg.sentence_stream.push_text(text)
        else:
            self.mark_segment_end()

    def mark_segment_end(self) -> None:
        # create new segment on "mark_segment_end"
        self._pending_segment.cur_text.sentence_stream.mark_segment_end()
        new_seg = self._create_segment()
        self._pending_segment.cur_text = new_seg
        self._pending_segment.q.append(new_seg)
        self._seg_queue.put_nowait(new_seg)

    async def _run(self) -> None:
        await self._start_future
        transcription_msg_q = asyncio.Queue[rtc.TranscriptionSegment | None]()
        try:
            await asyncio.gather(
                self._synchronize(transcription_msg_q),
                self._forward(transcription_msg_q),
                return_exceptions=True,
            )
        except Exception:
            logger.exception("error in tts transcription")

    def _create_segment(self) -> _SegmentData:
        return _SegmentData(
            sentence_stream=self._opts.sentence_tokenizer.stream(),
            text="",
            audio_duration=0.0,
            avg_speed=None,
            processed_hyphenes=0,
        )

    async def aclose(self, *, wait: bool = True) -> None:
        self._start_future.cancel()
        self._seg_queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _forward(self, q: asyncio.Queue[rtc.TranscriptionSegment | None]):
        while True:
            seg = await q.get()
            if seg is None:
                break

            tr = rtc.Transcription(
                participant_identity=self._opts.participant_identity,
                track_id=self._opts.track_id,
                segments=[seg],  # no history for now
                language=self._opts.language,
            )
            await self._opts.room.local_participant.publish_transcription(tr)

    async def _synchronize(self, q: asyncio.Queue[rtc.TranscriptionSegment | None]):
        async def _sync_sentence(
            seg: _SegmentData, tokenized_sentence: str, start_time: float
        ):
            # sentence data
            seg_id = _uuid()  # put each sentence in a different transcription segment
            words = self._opts.word_tokenizer.tokenize(tokenized_sentence)
            processed_words = []

            for word in words:
                word_hyphenes = len(self._opts.hyphenate_word(word))
                processed_words.append(word)

                elapsed_time = (
                    time.time() - start_time
                )  # elapsed time since the start of the seg
                text = self._opts.word_tokenizer.format_words(processed_words)

                delay = 0
                if seg.avg_speed is not None:
                    target_hyphenes = round(seg.avg_speed * elapsed_time)
                    dt = target_hyphenes - seg.processed_hyphenes
                    to_wait_hyphenes = max(0, word_hyphenes - dt)
                    delay = to_wait_hyphenes / seg.avg_speed
                else:
                    delay = word_hyphenes / self._opts.speed

                await asyncio.sleep(delay)

                seg.processed_hyphenes += word_hyphenes
                q.put_nowait(
                    rtc.TranscriptionSegment(
                        id=seg_id,
                        text=text,
                        start_time=0,
                        end_time=0,
                        final=False,
                    )
                )

            q.put_nowait(
                rtc.TranscriptionSegment(
                    id=seg_id,
                    text=tokenized_sentence,
                    start_time=0,
                    end_time=0,
                    final=True,
                )
            )

        while True:
            audio_seg = await self._seg_queue.get()
            if audio_seg is None:
                break

            start_time = time.time()
            sentence_stream = audio_seg.sentence_stream

            async for ev in sentence_stream:
                if ev.type == tokenize.TokenEventType.TOKEN:
                    await _sync_sentence(audio_seg, ev.token, start_time)
                if ev.type == tokenize.TokenEventType.FINISHED:
                    break  # new segment

        q.put_nowait(None)

    def _calc_hyphenes(self, text: str) -> list[str]:
        hyphenes = []
        words = self._opts.word_tokenizer.tokenize(text=text)
        for word in words:
            hyphenes.extend(self._opts.hyphenate_word(word))

        return hyphenes
