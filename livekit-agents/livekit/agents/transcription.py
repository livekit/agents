import asyncio
import contextlib
import time
import uuid
from typing import List
from livekit import rtc

from . import stt
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

    def forward_transcription(
        self,
        *,
        participant: rtc.Participant | str,
        language: str = "",
        track_id: str | None = None,
    ):
        identity = participant if isinstance(participant, str) else participant.identity
        if track_id is None:
            track_id = self._find_micro_track_id(identity)

        return SegmentsForwarder(
            room=self._room,
            participant_identity=identity,
            track_id=track_id,
            language=language,
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

        except Exception as e:
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


class SegmentsForwarder:
    def __init__(
        self,
        *,
        room: rtc.Room,
        participant_identity: str,
        track_id: str,
        language: str,
        wps: float = 0.7,  # TODO(theomonnom): remove
    ):
        self._room = room
        self._participant_identity = participant_identity
        self._track_id = track_id
        self._language = language
        self._queue = asyncio.Queue[rtc.TranscriptionSegment | None]()
        self._main_task = asyncio.create_task(self._run())
        self._wps = wps
        self._rotate_id()

        self._interim_text = ""
        self._text = ""

    async def _run(self):
        try:
            while True:
                seg = await self._queue.get()
                if seg is None:
                    break

                transcription = rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_id=self._track_id,
                    segments=[seg],
                    language=self._language,
                )

                await self._room.local_participant.publish_transcription(transcription)
                await asyncio.sleep(self._wps)
        except Exception as e:
            logger.exception("error in stt transcription")

    def _rotate_id(self):
        self._current_id = _uuid()

    def push_text(self, token: str):
        # fmt: off
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        # fmt: on

        self._text += token

        while True:
            last_split = -1
            for i, c in enumerate(self._text):
                if c in splitters:
                    last_split = i
                    break

            if last_split == -1:
                break

            seg = self._text[: last_split + 1]
            seg = seg.strip() + " "  # 11labs expects a space at the end
            self._interim_text += seg
            self._queue.put_nowait(
                rtc.TranscriptionSegment(
                    id=self._current_id,
                    text=self._interim_text,
                    start_time=0,
                    end_time=0,
                    final=False,
                )
            )
            self._text = self._text[last_split + 1 :]

    def mark_segment_end(self):
        self._interim_text += self._text
        self._queue.put_nowait(
            rtc.TranscriptionSegment(
                id=self._current_id,
                text=self._interim_text,
                start_time=0,
                end_time=0,
                final=True,
            )
        )
        self._rotate_id()
        self._text = ""
        self._interim_text = ""

    async def aclose(self, *, wait=True) -> None:
        self._queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
