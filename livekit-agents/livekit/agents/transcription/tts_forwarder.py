from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

from livekit import rtc
from livekit.rtc.participant import PublishTranscriptionError

from .. import tokenize, utils
from ..log import logger
from ..tokenize.tokenizer import PUNCTUATIONS
from . import _utils

# 3.83 is the "baseline", the number of hyphens per second TTS returns in avg.
STANDARD_SPEECH_RATE = 3.83


BeforeForwardCallback = Callable[
    ["TTSSegmentsForwarder", rtc.Transcription],
    Union[rtc.Transcription, Awaitable[Optional[rtc.Transcription]]],
]


WillForwardTranscription = BeforeForwardCallback


def _default_before_forward_callback(
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
    before_forward_cb: BeforeForwardCallback


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
        before_forward_cb: BeforeForwardCallback = _default_before_forward_callback,
        loop: asyncio.AbstractEventLoop | None = None,
        # backward compatibility
        will_forward_transcription: WillForwardTranscription | None = None,
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

        if will_forward_transcription is not None:
            logger.warning(
                "will_forward_transcription is deprecated and will be removed in 1.5.0, use before_forward_cb instead",
            )
            before_forward_cb = will_forward_transcription

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
            before_forward_cb=before_forward_cb,
        )
        self._closed = False
        self._loop = loop or asyncio.get_event_loop()
        self._close_future = asyncio.Future[None]()

        self._playing_seg_index = -1
        self._finshed_seg_index = -1

        self._text_q_changed = asyncio.Event()
        self._text_q = list[Union[_TextData, None]]()
        self._audio_q_changed = asyncio.Event()
        self._audio_q = list[Union[_AudioData, None]]()

        self._text_data: _TextData | None = None
        self._audio_data: _AudioData | None = None

        self._played_text = ""

        self._main_atask = self._loop.create_task(self._main_task())
        self._task_set = utils.aio.TaskSet(loop)

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

        if self._audio_data is None:
            self._audio_data = _AudioData()
            self._audio_q.append(self._audio_data)
            self._audio_q_changed.set()

        frame_duration = frame.samples_per_channel / frame.sample_rate
        self._audio_data.pushed_duration += frame_duration

    def mark_audio_segment_end(self) -> None:
        self._check_not_closed()

        if self._audio_data is None:
            self.push_audio(rtc.AudioFrame(bytes(), 24000, 1, 0))

        assert self._audio_data is not None
        self._audio_data.done = True
        self._audio_data = None

    def push_text(self, text: str) -> None:
        self._check_not_closed()

        if self._text_data is None:
            self._text_data = _TextData(
                sentence_stream=self._opts.sentence_tokenizer.stream()
            )
            self._text_q.append(self._text_data)
            self._text_q_changed.set()

        self._text_data.pushed_text += text
        self._text_data.sentence_stream.push_text(text)

    def mark_text_segment_end(self) -> None:
        self._check_not_closed()

        if self._text_data is None:
            self.push_text("")

        assert self._text_data is not None
        self._text_data.done = True
        self._text_data.sentence_stream.end_input()
        self._text_data = None

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

        for text_data in self._text_q:
            assert text_data is not None
            await text_data.sentence_stream.aclose()

        self._text_q.append(None)
        self._audio_q.append(None)
        self._text_q_changed.set()
        self._audio_q_changed.set()

        await self._task_set.aclose()
        await self._main_atask

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task that forwards the transcription to the room."""
        rtc_seg_ch = utils.aio.Chan[rtc.TranscriptionSegment]()

        @utils.log_exceptions(logger=logger)
        async def _forward_task():
            async for rtc_seg in rtc_seg_ch:
                base_transcription = rtc.Transcription(
                    participant_identity=self._opts.participant_identity,
                    track_sid=self._opts.track_id,
                    segments=[rtc_seg],  # no history for now
                )

                transcription = self._opts.before_forward_cb(self, base_transcription)
                if asyncio.iscoroutine(transcription):
                    transcription = await transcription

                # fallback to default impl if no custom/user stream is returned
                if not isinstance(transcription, rtc.Transcription):
                    transcription = _default_before_forward_callback(
                        self, base_transcription
                    )

                if transcription.segments and self._opts.room.isconnected():
                    try:
                        await self._opts.room.local_participant.publish_transcription(
                            transcription
                        )
                    except PublishTranscriptionError:
                        continue

        forward_task = asyncio.create_task(_forward_task())

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

                # wait until the segment is validated and has started playing
                while not self._closed:
                    if self._playing_seg_index >= seg_index:
                        break

                    await self._sleep_if_not_closed(0.125)

                sentence_stream = text_data.sentence_stream
                forward_start_time = time.time()

                async for ev in sentence_stream:
                    await self._sync_sentence_co(
                        seg_index,
                        forward_start_time,
                        text_data,
                        audio_data,
                        ev.token,
                        rtc_seg_ch,
                    )

                seg_index += 1

            self._text_q_changed.clear()
            self._audio_q_changed.clear()

        rtc_seg_ch.close()
        await forward_task

    async def _sync_sentence_co(
        self,
        segment_index: int,
        segment_start_time: float,
        text_data: _TextData,
        audio_data: _AudioData,
        sentence: str,
        rtc_seg_ch: utils.aio.Chan[rtc.TranscriptionSegment],
    ):
        """Synchronize the transcription with the audio playout for a given sentence."""
        # put each sentence in a different transcription segment

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
            if segment_index <= self._finshed_seg_index:
                # playout of the audio segment already finished
                # break the loop and send the final transcription
                break

            if self._closed:
                # transcription closed, early
                return

            word_hyphens = len(self._opts.hyphenate_word(word))
            processed_words.append(word)

            # elapsed time since the start of the seg
            elapsed_time = time.time() - segment_start_time
            text = self._opts.word_tokenizer.format_words(processed_words)

            # remove any punctuation at the end of a non-final transcript
            text = text.rstrip("".join(PUNCTUATIONS))

            speed = self._opts.speed
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

            rtc_seg_ch.send_nowait(
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
            text_data.forwarded_hyphens += word_hyphens

        rtc_seg_ch.send_nowait(
            rtc.TranscriptionSegment(
                id=seg_id,
                text=sentence,
                start_time=0,
                end_time=0,
                final=True,
                language=self._opts.language,
            )
        )
        self._played_text = f"{og_text} {sentence}"

        await self._sleep_if_not_closed(self._opts.new_sentence_delay)
        text_data.forwarded_sentences += 1

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

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("TTSForwarder is closed")
