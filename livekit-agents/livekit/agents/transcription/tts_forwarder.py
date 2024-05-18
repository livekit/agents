from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import Callable, Optional

from attrs import define
from livekit import rtc

from .. import tokenize
from ..log import logger
from . import _utils


@define
class _TTSOptions:
    room: rtc.Room
    participant_identity: str
    track_id: str
    language: str
    speed: float
    auto_playout: bool
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
    first_frame_future: asyncio.Future


def _validate_playout(seg: _SegmentData) -> None:
    if seg.audio_duration == 0.0:
        with contextlib.suppress(asyncio.InvalidStateError):
            seg.first_frame_future.set_result(None)


@define
class _PendingSegments:
    cur_audio: _SegmentData
    cur_text: _SegmentData
    q: deque[_SegmentData]


class TTSSegmentsForwarder:
    """
    Forward TTS transcription to the users. This class tries to imitate the right timing of
    speech with the synthesized text. The first estimation is based on the speed argument. Once
    we have received the full audio of a specific text segment, we recalculate the avg speech
    speed using the length of the text & audio and catch up/ slow down the transcription if needed.
    """

    def __init__(
        self,
        *,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        language: str = "",
        speed: float = 4,
        auto_playout: bool = True,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
    ):
        identity = participant if isinstance(participant, str) else participant.identity

        if track is None:
            track = _utils.find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        self._opts = _TTSOptions(
            room=room,
            participant_identity=identity,
            track_id=track,
            language=language,
            speed=speed,
            auto_playout=auto_playout,
            word_tokenizer=word_tokenizer,
            sentence_tokenizer=sentence_tokenizer,
            hyphenate_word=hyphenate_word,
        )

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

        self._seg_queue = asyncio.Queue[Optional[_SegmentData]]()
        self._seg_queue.put_nowait(first_segment)

        self._validated_playout_q = asyncio.Queue[None]()

    def segment_playout_started(self) -> None:
        """Call this function when the playout of the audio segment starts,
        this will start forwarding the transcription for the current segment.

        This is only needed if auto_playout is set to False.

        Note that you don't need to wait for the first synthesized audio frame to call this function.
        The forwarder will wait for the first audio frame before starting the transcription.
        """
        self._validated_playout_q.put_nowait(None)

    def push_audio(self, frame: rtc.AudioFrame | None, **kwargs) -> None:
        if frame is not None:
            frame_duration = frame.samples_per_channel / frame.sample_rate
            cur_seg = self._pending_segment.cur_audio
            cur_seg.audio_duration += frame_duration
            _validate_playout(cur_seg)
        else:
            self.mark_audio_segment_end()

    def mark_audio_segment_end(self) -> None:
        try:
            # get last ended segment (text always end before audio)
            seg = self._pending_segment.q.popleft()
        except IndexError:
            raise IndexError(
                "mark_audio_segment_end called before any mark_text_segment_end"
            )

        seg.avg_speed = len(self._calc_hyphenes(seg.text)) / seg.audio_duration
        self._pending_segment.cur_audio = self._pending_segment.q[0]

    def push_text(self, text: str | None) -> None:
        if text is not None:
            cur_seg = self._pending_segment.cur_text
            cur_seg.text += text
            cur_seg.sentence_stream.push_text(text)
        else:
            self.mark_text_segment_end()

    def mark_text_segment_end(self) -> None:
        # create new segment on "mark_text_segment_end"
        self._pending_segment.cur_text.sentence_stream.mark_segment_end()
        new_seg = self._create_segment()
        self._pending_segment.cur_text = new_seg
        self._pending_segment.q.append(new_seg)
        self._seg_queue.put_nowait(new_seg)

    async def aclose(self, *, wait: bool = True) -> None:
        self._seg_queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self) -> None:
        transcription_msg_q = asyncio.Queue[Optional[rtc.TranscriptionSegment]]()
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
            first_frame_future=asyncio.Future(),
        )

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
            seg_id = (
                _utils.segment_uuid()
            )  # put each sentence in a different transcription segment
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

            await audio_seg.first_frame_future

            if not self._opts.auto_playout:
                _ = await self._validated_playout_q.get()

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
