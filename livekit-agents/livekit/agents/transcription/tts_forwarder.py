from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import Callable, Optional

from attrs import define
from livekit import rtc

from .. import tokenize, utils
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
    new_sentence_delay: float
    debug: bool = False


@define
class _SegmentData:
    index: int
    sentence_stream: tokenize.SentenceStream
    text: str
    sentence_count: int
    audio_duration: float
    avg_speed: float | None
    processed_hyphenes: int
    ready_future: asyncio.Future  # marked ready on the first audio frame


def _validate_playout(seg: _SegmentData) -> None:
    if seg.audio_duration == 0.0:
        with contextlib.suppress(asyncio.InvalidStateError):
            seg.ready_future.set_result(None)


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
        speed: float = 3.83,
        new_sentence_delay: float = 0.7,
        auto_playout: bool = True,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        debug: bool = False,
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
            hyphenate_word: function that returns a list of hyphenes for a given word
            debug: if True, debug messages will be printed

        """
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
            new_sentence_delay=new_sentence_delay,
        )
        self._closed = False

        self._next_segment_index = 0
        self._playing_seg_index = -1
        self._finshed_seg_index = -1

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
        self._main_atask = asyncio.create_task(self._main_task())

    def segment_playout_started(self) -> None:
        """Call this function when the playout of the audio segment starts,
        this will start forwarding the transcription for the current segment.

        This is only needed if auto_playout is set to False.

        Note that you don't need to wait for the first synthesized audio frame to call this function.
        The forwarder will wait for the first audio frame before starting the transcription.
        """
        self._playing_seg_index += 1

    def segment_playout_finished(self) -> None:
        """Notify that the playout of the audio segment has finished.
        This will catchup and directly send the final transcription in case the forwarder is too
        late.
        """
        self._finshed_seg_index += 1

    def push_audio(self, frame: rtc.AudioFrame | None) -> None:
        if self._closed:
            raise RuntimeError("push_audio called after close")

        if frame is not None:
            frame_duration = frame.samples_per_channel / frame.sample_rate
            cur_seg = self._pending_segment.cur_audio
            _validate_playout(cur_seg)
            cur_seg.audio_duration += frame_duration
        else:
            self.mark_audio_segment_end()

    def mark_audio_segment_end(self) -> None:
        if self._closed:
            raise RuntimeError("mark_audio_segment_end called after close")

        try:
            # get last ended segment (text always end before audio)
            seg = self._pending_segment.q.popleft()
        except IndexError:
            raise IndexError(
                "mark_audio_segment_end called before any mark_text_segment_end"
            )

        if seg.audio_duration > 0.0:
            seg.avg_speed = len(self._calc_hyphenes(seg.text)) / seg.audio_duration

        self._pending_segment.cur_audio = self._pending_segment.q[0]
        self._log_debug(
            f"mark_audio_segment_end: calculated avg speed: {seg.avg_speed}"
        )

    def push_text(self, text: str | None) -> None:
        if self._closed:
            raise RuntimeError("push_text called after close")

        if text is not None:
            cur_seg = self._pending_segment.cur_text
            cur_seg.text += text
            cur_seg.sentence_stream.push_text(text)
        else:
            self.mark_text_segment_end()

    def mark_text_segment_end(self) -> None:
        if self._closed:
            raise RuntimeError("mark_text_segment_end called after close")

        # create new segment on "mark_text_segment_end"
        self._pending_segment.cur_text.sentence_stream.mark_segment_end()
        new_seg = self._create_segment()
        self._pending_segment.cur_text = new_seg
        self._pending_segment.q.append(new_seg)
        self._seg_queue.put_nowait(new_seg)

    async def aclose(self) -> None:
        self._closed = True
        self._seg_queue.put_nowait(None)

        # can't push more text/audio
        for seg in self._pending_segment.q:
            seg.ready_future.cancel()
            await seg.sentence_stream.aclose()

        await self._main_atask

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        transcription_msg_q = asyncio.Queue[Optional[rtc.TranscriptionSegment]]()
        await asyncio.gather(
            self._synchronize_co(transcription_msg_q),
            self._forward_co(transcription_msg_q),
        )

    def _create_segment(self) -> _SegmentData:
        data = _SegmentData(
            index=self._next_segment_index,
            sentence_stream=self._opts.sentence_tokenizer.stream(),
            text="",
            sentence_count=0,
            audio_duration=0.0,
            avg_speed=None,
            processed_hyphenes=0,
            ready_future=asyncio.Future(),
        )
        self._next_segment_index += 1
        return data

    async def _forward_co(self, q: asyncio.Queue[rtc.TranscriptionSegment | None]):
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

    async def _synchronize_co(self, q: asyncio.Queue[rtc.TranscriptionSegment | None]):
        async def _sync_sentence_co(
            seg: _SegmentData, tokenized_sentence: str, start_time: float
        ):
            # put each sentence in a different transcription segment
            seg_id = _utils.segment_uuid()
            words = self._opts.word_tokenizer.tokenize(text=tokenized_sentence)
            processed_words = []

            text = ""
            for word in words:
                if self._closed:
                    # transcription closed, early
                    return

                if seg.index <= self._finshed_seg_index:
                    # playout of the audio segment already finished
                    break

                word_hyphenes = len(self._opts.hyphenate_word(word))
                processed_words.append(word)

                # elapsed time since the start of the seg
                elapsed_time = time.time() - start_time
                text = self._opts.word_tokenizer.format_words(processed_words)

                delay = 0
                if seg.avg_speed is not None:
                    estimated_pauses_s = (
                        seg.sentence_count * self._opts.new_sentence_delay
                    )
                    hyph_pauses = estimated_pauses_s * seg.avg_speed

                    target_hyphenes = round(seg.avg_speed * elapsed_time)
                    dt = target_hyphenes - seg.processed_hyphenes - hyph_pauses
                    to_wait_hyphenes = max(0, word_hyphenes - dt)
                    delay = to_wait_hyphenes / seg.avg_speed
                else:
                    delay = word_hyphenes / self._opts.speed

                halfdelay = delay / 2
                await asyncio.sleep(halfdelay)
                q.put_nowait(
                    rtc.TranscriptionSegment(
                        id=seg_id,
                        text=text,
                        start_time=0,
                        end_time=0,
                        final=False,
                    )
                )
                await asyncio.sleep(halfdelay)
                seg.processed_hyphenes += word_hyphenes

            q.put_nowait(
                rtc.TranscriptionSegment(
                    id=seg_id,
                    text=tokenized_sentence,
                    start_time=0,
                    end_time=0,
                    final=True,
                )
            )

            await asyncio.sleep(self._opts.new_sentence_delay)
            seg.sentence_count += 1

        while True:
            audio_seg = await self._seg_queue.get()
            if audio_seg is None:
                break

            try:
                await audio_seg.ready_future
            except asyncio.CancelledError:
                continue

            if not self._opts.auto_playout:
                # wait until the playout of the audio segment has started
                while not self._closed:
                    if self._playing_seg_index >= audio_seg.index:
                        break
                    await asyncio.sleep(0.1)

            start_time = time.time()
            sentence_stream = audio_seg.sentence_stream

            async for ev in sentence_stream:
                if ev.type == tokenize.TokenEventType.TOKEN:
                    await _sync_sentence_co(audio_seg, ev.token, start_time)
                if ev.type == tokenize.TokenEventType.FINISHED:
                    break  # new segment

        q.put_nowait(None)

    def _calc_hyphenes(self, text: str) -> list[str]:
        hyphenes = []
        words = self._opts.word_tokenizer.tokenize(text=text)
        for word in words:
            new = self._opts.hyphenate_word(word)
            hyphenes.extend(new)

        return hyphenes

    def _log_debug(self, msg: str, **kwargs) -> None:
        if self._opts.debug:
            logger.debug(msg, **kwargs)
