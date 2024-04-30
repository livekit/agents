import asyncio
import time
from typing import List
from uuid import uuid4

from livekit import rtc


class TranscriptionManager:
    def __init__(self, room: rtc.Room):
        self._segments: List[rtc.TranscriptionSegment] = []
        self._room = room

    def start_segment(
        self, participant: str | rtc.Participant, language: str, track_id: str
    ) -> "TranscriptionManagerSegmentHandle":
        return TranscriptionManagerSegmentHandle(
            room=self._room,
            participant=participant,
            track_id=track_id,
            language=language,
        )


class TranscriptionManagerSegmentHandle:
    def __init__(
        self,
        room: rtc.Room,
        participant: str | rtc.Participant,
        track_id: str,
        language: str,
    ):
        self._id = str(uuid4())
        self._room = room
        self._language = language
        self._track_id = track_id
        if isinstance(participant, rtc.Participant):
            self._participant_identity = participant.identity
        else:
            self._participant_identity = participant
        self._start_time = time.time_ns()
        self._text = ""

    def update(self, text: str):
        self._text = text

    def commit(self):
        segment = rtc.TranscriptionSegment(
            id=self._id,
            text=self._text,
            start_time=self._start_time,
            end_time=time.time_ns(),
            final=True,
        )
        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_id=self._track_id,
            segments=[segment],
            language=self._language,
        )

        async def _publish():
            await self._room.local_participant.publish_transcription(transcription)

        asyncio.ensure_future(_publish())
