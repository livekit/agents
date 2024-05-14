import asyncio
import contextlib
import uuid
from typing import Callable

from attrs import define
from livekit import rtc

from . import stt, tokenize, tts
from .log import logger


def _uuid() -> str:
    return str(uuid.uuid4())[:12]


class TranscriptionManager:
    def __init__(self, room: rtc.Room):
        self._room = room

    def forward_stt_transcription(
        self, *, participant: rtc.Participant | str, track_id: str | None = None
    ) -> "STTSegmentsForwarder":
        identity = participant if isinstance(participant, str) else participant.identity
        if track_id is None:
            track_id = self._find_micro_track_id(identity)

        return STTSegmentsForwarder(
            room=self._room,
            participant_identity=identity,
            track_id=track_id,
        )

    def forward_tts_transcription(
        self,
        *,
        participant: rtc.Participant | str,
        language: str = "",
        track_id: str | None = None,
        speed: float = 4.0,  # based on hypenation (avg wps is 2.5 in English, 4 may be a good default)
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        automatically_start: bool = True,
        word_separator: str = " ",
    ):
        identity = participant if isinstance(participant, str) else participant.identity
        if track_id is None:
            track_id = self._find_micro_track_id(identity)

        return TTSSegmentsForwarder(
            room=self._room,
            participant_identity=identity,
            track_id=track_id,
            language=language,
            speed=speed,
            word_tokenizer=word_tokenizer,
            sentence_tokenizer=sentence_tokenizer,
            hyphenate_word=hyphenate_word,
            word_separator=word_separator,
            automatically_start=automatically_start,
        )

    def _find_micro_track_id(self, identity: str) -> str:
        p = self._room.participants_by_identity.get(identity)
        if p is None:
            raise ValueError(f"participant {identity} not found")

        # find first micro track
        track_id = None
        for track in p.tracks.values():
            if track.source == rtc.TrackSource.SOURCE_MICROPHONE:
                track_id = track.sid
                break

        if track_id is None:
            raise ValueError(f"participant {identity} does not have a microphone track")

        return track_id


class STTSegmentsForwarder:
    def __init__(
        self,
        *,
        room: rtc.Room,
        participant_identity: str,
        track_id: str,
    ):
        self._room = room
        self._participant_identity = participant_identity
        self._track_id = track_id
        self._queue = asyncio.Queue[rtc.TranscriptionSegment | None]()
        self._main_task = asyncio.create_task(self._run())
        self._rotate_id()

    async def _run(self):
        try:
            while True:
                seg = await self._queue.get()
                if seg is None:
                    break

                transcription = rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_id=self._track_id,
                    segments=[seg],  # no history for now
                    language="",  # TODO(theomonnom)
                )
                await self._room.local_participant.publish_transcription(transcription)

        except Exception:
            logger.exception("error in stt transcription")

    def _rotate_id(self):
        self._current_id = _uuid()

    def update(self, ev: stt.SpeechEvent):
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # TODO(theomonnom): We always take the first alternative, we should mb expose opt to the
            # user?
            text = ev.alternatives[0].text
            self._queue.put_nowait(
                rtc.TranscriptionSegment(
                    id=self._current_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=False,
                )
            )
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            text = ev.alternatives[0].text
            self._queue.put_nowait(
                rtc.TranscriptionSegment(
                    id=self._current_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=True,
                )
            )
            self._rotate_id()

    async def aclose(self, *, wait=True) -> None:
        self._queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task


@define
class _TTSAudioSegmentData:
    sentence_stream: tokenize.SentenceStream
    text: str
    audio_duration: float
    real_speed: float  # hps


class TTSSegmentsForwarder:
    def __init__(
        self,
        *,
        room: rtc.Room,
        participant_identity: str,
        track_id: str,
        language: str,
        speed: float,
        word_tokenizer: tokenize.WordTokenizer,  # stream the words and avg the duration using hyphenation
        sentence_tokenizer: tokenize.SentenceTokenizer,  # split the transcription into multiple segments
        hyphenate_word: Callable[[str], list[str]],
        word_separator: str,
        automatically_start: bool,
    ):
        self._room = room
        self._participant_identity = participant_identity
        self._track_id = track_id
        self._language = language
        self._word_separator = word_separator
        self._automatically_start = automatically_start

        self._main_task = asyncio.create_task(self._run())
        self._audio_queue = asyncio.Queue[_TTSAudioSegmentData | None]()
        self._queue = asyncio.Queue[rtc.TranscriptionSegment | None]()

        self._word_tokenizer = word_tokenizer
        self._sentence_tokenizer = sentence_tokenizer
        self._hyphenate_word = hyphenate_word

        self._cur_seg = self._create_segment(speed)

        self._start_future = asyncio.Future()
        if automatically_start:
            self._start_future.set_result(None)

    def start(self):
        try:
            self._start_future.set_result(None)
        except asyncio.InvalidStateError:
            raise ValueError("TTS transcription already started")

    async def _run(self) -> None:
        await self._start_future
        await asyncio.gather(
            self._forward(), self._synchronize(), return_exceptions=True
        )

    async def _synchronize(self):
        while True:
            audio_seg = await self._audio_queue.get()
            if audio_seg is None:
                break

            sentence_stream = audio_seg.sentence_stream

            async for ev in sentence_stream:
                if ev.type == tokenize.TokenEventType.FINISHED:
                    break
                elif ev.type == tokenize.TokenEventType.TOKEN:
                    await self._sync_sentence(audio_seg, ev.token)

    async def _forward(self):
        try:
            while True:
                seg = await self._queue.get()
                if seg is None:
                    break

                transcription = rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_id=self._track_id,
                    segments=[seg],  # no history for now
                    language=self._language,
                )
                await self._room.local_participant.publish_transcription(transcription)

        except Exception:
            logger.exception("error in tts transcription")

    async def _sync_sentence(self, audio_seg: _TTSAudioSegmentData, sentence: str):
        words = self._word_tokenizer.tokenize(sentence)
        seg_id = _uuid()
        text = ""

        for word in words:
            if text:
                text += self._word_separator
            text += word

            hyphenes_count = len(self._hyphenate_word(word))
            delay = 1 / audio_seg.real_speed * hyphenes_count
            await asyncio.sleep(delay)  # TODO: Recover from delay gaps

            self._queue.put_nowait(
                rtc.TranscriptionSegment(
                    id=seg_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=False,
                )
            )

        self._queue.put_nowait(
            rtc.TranscriptionSegment(
                id=seg_id,
                text=sentence,
                start_time=0,
                end_time=0,
                final=True,
            )
        )

    def update(self, ev: tts.SynthesisEvent) -> None:
        # multiple STARTED/FINISHED events, keep track of the current segment
        audio_seg = self._cur_seg
        if ev.type == tts.SynthesisEventType.STARTED:
            audio_seg = self._cur_seg
        elif ev.type == tts.SynthesisEventType.AUDIO:
            assert ev.audio is not None
            frame = ev.audio.data
            audio_seg.audio_duration += frame.samples_per_channel / frame.sample_rate
        elif ev.type == tts.SynthesisEventType.FINISHED:
            # mark_segment_end should have been called before this event being triggered
            speed = len(self._calc_hyphenes(audio_seg.text)) / audio_seg.audio_duration
            audio_seg.real_speed = speed

    def push_text(self, text: str | None) -> None:
        self._cur_seg.sentence_stream.push_text(text)
        if text is not None:
            self._cur_seg.text += text
        else:
            # create new segment on "mark_segment_end"
            self._cur_seg = self._create_segment(self._cur_seg.real_speed)

    def mark_segment_end(self) -> None:
        self.push_text(None)

    def _create_segment(self, speed: float) -> _TTSAudioSegmentData:
        seg = _TTSAudioSegmentData(
            sentence_stream=self._sentence_tokenizer.stream(),
            text="",
            audio_duration=0.0,
            real_speed=speed,
        )

        self._audio_queue.put_nowait(seg)
        return seg

    async def aclose(self, *, wait: bool = True) -> None:
        self._start_future.cancel()
        self._queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    def _calc_hyphenes(self, text: str) -> list[str]:
        hyphenes = []
        words = self._word_tokenizer.tokenize(text=text)
        for word in words:
            hyphenes.extend(self._hyphenate_word(word))

        return hyphenes
