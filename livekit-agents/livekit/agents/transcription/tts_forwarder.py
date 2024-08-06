from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Optional, Union

from livekit import rtc

from .. import tokenize, utils
from ..log import logger
from . import _utils

# 3.83 is the "baseline", the number of hyphens per second TTS returns in avg.
STANDARD_SPEECH_RATE = 3.83


WillForwardTranscription = Callable[
    ["TTSSegmentsForwarder", rtc.Transcription],
    Union[rtc.Transcription, Awaitable[Optional[rtc.Transcription]]],
]


def _default_will_forward_transcription(
    fwd: TTSSegmentsForwarder, transcription: rtc.Transcription
) -> rtc.Transcription:
    return transcription


@dataclass
class _TTSOptions:
    room: rtc.Room
    participant_identity: str
    track_id: str
    language: str
    speed: float
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]
    new_sentence_delay: float
    will_forward_transcription: WillForwardTranscription


@dataclass
class _SegmentData:
    segment_index: int
    sentence_stream: tokenize.SentenceStream
    pushed_text: str = ""
    pushed_duration: float = 0.0
    real_speed: float | None = None
    processed_sentences: int = 0
    processed_hyphens: int = 0
    validated: bool = False
    forward_start_time: float | None = 0.0


@dataclass
class _FormingSegments:
    audio: _SegmentData
    text: _SegmentData
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
        speed: float = 1.0,
        new_sentence_delay: float = 0.4,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        will_forward_transcription: WillForwardTranscription = _default_will_forward_transcription,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """
        Args:
            room: room where the transcription will be sent
            participant: participant or identity that is pushing the TTS
            track: track where the TTS audio is being sent
            language: language of the text
            speed: average speech speed in characters per second (used by default if the full audio is not received yet)
            new_sentence_delay: delay in seconds between sentences
            auto_playout: if True, the forwarder will automatically start the transcription once the
                first audio frame is received. If False, you need to call segment_playout_started
                to start the transcription.
            word_tokenizer: word tokenizer used to split the text into words
            sentence_tokenizer: sentence tokenizer used to split the text into sentences
            hyphenate_word: function that returns a list of hyphens for a given word

        """
        identity = participant if isinstance(participant, str) else participant.identity

        if track is None:
            track = _utils.find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        speed = speed * STANDARD_SPEECH_RATE
        self._opts = _TTSOptions(
            room=room,
            participant_identity=identity,
            track_id=track,
            language=language,
            speed=speed,
            word_tokenizer=word_tokenizer,
            sentence_tokenizer=sentence_tokenizer,
            hyphenate_word=hyphenate_word,
            new_sentence_delay=new_sentence_delay,
            will_forward_transcription=will_forward_transcription,
        )
        self._closed = False
        self._loop = loop or asyncio.get_event_loop()
        self._close_future = asyncio.Future[None]()

        self._next_segment_index = 0
        self._playing_seg_index = -1
        self._finshed_seg_index = -1

        first_segment = self._create_segment()
        segments_q: Deque[_SegmentData] = deque()
        segments_q.append(first_segment)

        self._forming_segments = _FormingSegments(
            audio=first_segment, text=first_segment, q=segments_q
        )

        self._seg_queue = asyncio.Queue[Optional[_SegmentData]]()
        self._seg_queue.put_nowait(first_segment)
        self._main_atask = self._loop.create_task(self._main_task())
        self._task_set = utils.aio.TaskSet(loop)

        self._played_text = ""

    def segment_playout_started(self) -> None:
        """
        Notify that the playout of the audio segment has started.
        This will start forwarding the transcription for the current segment.
        """
        self._check_not_closed()
        self._playing_seg_index += 1

    def segment_playout_finished(self) -> None:
        """
        Notify that the playout of the audio segment has finished.
        This will catchup and directly send the final transcription in case the forwarder is too
        late.
        """
        self._check_not_closed()
        self._finshed_seg_index += 1

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._check_not_closed()
        frame_duration = frame.samples_per_channel / frame.sample_rate
        cur_seg = self._forming_segments.audio
        cur_seg.pushed_duration += frame_duration
        cur_seg.validated = True

    def mark_audio_segment_end(self) -> None:
        self._check_not_closed()
        try:
            # get last ended segment (text always end before audio)
            seg = self._forming_segments.q.popleft()
        except IndexError:
            raise IndexError(
                "mark_audio_segment_end called before any mark_text_segment_end"
            )

        if seg.pushed_duration > 0.0:
            seg.real_speed = (
                len(self._calc_hyphens(seg.pushed_text)) / seg.pushed_duration
            )

        seg.validated = True
        self._forming_segments.audio = self._forming_segments.q[0]

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        cur_seg = self._forming_segments.text
        cur_seg.pushed_text += text
        cur_seg.sentence_stream.push_text(text)

    def mark_text_segment_end(self) -> None:
        self._check_not_closed()
        stream = self._forming_segments.text.sentence_stream
        stream.end_input()

        # create a new segment on "mark_text_segment_end"
        # further text can already be pushed even if mark_audio_segment_end has not been
        # called yet
        new_seg = self._create_segment()
        self._forming_segments.text = new_seg
        self._forming_segments.q.append(new_seg)
        self._seg_queue.put_nowait(new_seg)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def played_text(self) -> str:
        return self._played_text

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._close_future.set_result(None)
        self._seg_queue.put_nowait(None)

        for seg in self._forming_segments.q:
            await seg.sentence_stream.aclose()

        await self._task_set.aclose()
        await self._main_atask

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task that forwards the transcription to the room."""
        rtc_seg_q = asyncio.Queue[Optional[rtc.TranscriptionSegment]]()

        @utils.log_exceptions(logger=logger)
        async def _forward_task():
            while True:
                seg = await rtc_seg_q.get()
                if seg is None:
                    break

                base_transcription = rtc.Transcription(
                    participant_identity=self._opts.participant_identity,
                    track_sid=self._opts.track_id,
                    segments=[seg],  # no history for now
                )

                transcription = self._opts.will_forward_transcription(
                    self, base_transcription
                )
                if asyncio.iscoroutine(transcription):
                    transcription = await transcription

                # fallback to default impl if no custom/user stream is returned
                if not isinstance(transcription, rtc.Transcription):
                    transcription = _default_will_forward_transcription(
                        self, base_transcription
                    )

                if transcription.segments and self._opts.room.isconnected():
                    await self._opts.room.local_participant.publish_transcription(
                        transcription
                    )

        forward_task = asyncio.create_task(_forward_task())

        while True:
            seg = await self._seg_queue.get()
            if seg is None:
                break

            # wait until the segment is validated and has started playing
            while not self._closed:
                if seg.validated and self._playing_seg_index >= seg.segment_index:
                    break

                await self._sleep_if_not_closed(0.1)

            sentence_stream = seg.sentence_stream
            seg.forward_start_time = time.time()

            async for ev in sentence_stream:
                await self._sync_sentence_co(seg, ev.token, rtc_seg_q)

        rtc_seg_q.put_nowait(None)
        await forward_task

    async def _sync_sentence_co(
        self,
        seg: _SegmentData,
        tokenized_sentence: str,
        rtc_seg_q: asyncio.Queue[Optional[rtc.TranscriptionSegment]],
    ):
        """Synchronize the transcription with the audio playout for a given sentence."""
        assert seg.forward_start_time is not None

        # put each sentence in a different transcription segment
        seg_id = _utils.segment_uuid()
        words = self._opts.word_tokenizer.tokenize(text=tokenized_sentence)
        processed_words: list[str] = []

        og_text = self._played_text
        for word in words:
            if seg.segment_index <= self._finshed_seg_index:
                # playout of the audio segment already finished
                # break the loop and send the final transcription
                break

            if self._closed:
                # transcription closed, early
                return

            word_hyphens = len(self._opts.hyphenate_word(word))
            processed_words.append(word)

            # elapsed time since the start of the seg
            elapsed_time = time.time() - seg.forward_start_time
            text = self._opts.word_tokenizer.format_words(processed_words)

            speed = self._opts.speed
            if seg.real_speed is not None:
                speed = seg.real_speed
                estimated_pauses_s = (
                    seg.processed_sentences * self._opts.new_sentence_delay
                )
                hyph_pauses = estimated_pauses_s * speed

                target_hyphens = round(speed * elapsed_time)
                dt = target_hyphens - seg.processed_hyphens - hyph_pauses
                to_wait_hyphens = max(0.0, word_hyphens - dt)
                delay = to_wait_hyphens / speed
            else:
                delay = word_hyphens / speed

            first_delay = min(delay / 2, 2 / speed)
            await self._sleep_if_not_closed(first_delay)
            rtc_seg_q.put_nowait(
                rtc.TranscriptionSegment(
                    id=seg_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=False,
                    language=self._opts.language,
                )
            )
            self._played_text = f"{og_text} {text}"
            await self._sleep_if_not_closed(delay - first_delay)
            seg.processed_hyphens += word_hyphens

        rtc_seg_q.put_nowait(
            rtc.TranscriptionSegment(
                id=seg_id,
                text=tokenized_sentence,
                start_time=0,
                end_time=0,
                final=True,
                language=self._opts.language,
            )
        )
        self._played_text = f"{og_text} {tokenized_sentence}"

        await self._sleep_if_not_closed(self._opts.new_sentence_delay)
        seg.processed_sentences += 1

    async def _sleep_if_not_closed(self, delay: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait([self._close_future], timeout=delay)

    def _calc_hyphens(self, text: str) -> list[str]:
        hyphens: list[str] = []
        words = self._opts.word_tokenizer.tokenize(text=text)
        for word in words:
            new = self._opts.hyphenate_word(word)
            hyphens.extend(new)

        return hyphens

    def _create_segment(self) -> _SegmentData:
        data = _SegmentData(
            segment_index=self._next_segment_index,
            sentence_stream=self._opts.sentence_tokenizer.stream(),
        )
        self._next_segment_index += 1
        return data

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("TTSForwarder is closed")
