import asyncio
import time
from typing import List

from livekit import rtc


class TranscriptionManager:
    def __init__(self, room: rtc.Room, participant_identity: str):
        self._participant_identity = participant_identity
        self._segments: List[rtc.TranscriptionSegment] = []
        self._track_sid = ""
        self._room = room
        self._current_id = 1

    def update_track_sid(self, track_sid: str) -> None:
        self._track_sid = track_sid

    def reset_transcription(self) -> None:
        self._segments = []

    def final_transcription(self, text: str, language: str = "english"):
        if len(self._segments) and self._segments[-1].id == str(self._current_id):
            self._segments[-1].end_time = time.time_ns()
            self._segments[-1].final = True
            self._segments[-1].text = text
        else:
            self._segments.append(
                rtc.TranscriptionSegment(
                    id=str(self._current_id),
                    text=text,
                    start_time=time.time_ns(),
                    end_time=time.time_ns(),
                    final=True,
                )
            )
        self._current_id += 1
        self._publish(language)

    def interim_transcription(self, text: str, language: str = "english"):
        if len(self._segments) and self._segments[-1].id == str(self._current_id):
            self._segments[-1].end_time = time.time_ns()
            self._segments[-1].text = text
        else:
            self._segments.append(
                rtc.TranscriptionSegment(
                    id=str(self._current_id),
                    text=text,
                    start_time=time.time_ns(),
                    end_time=time.time_ns(),
                    final=False,
                )
            )

        self._publish(language)

    def _publish(self, language):
        async def do_publish():
            await self._room.local_participant.publish_transcription(
                rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_id=self._track_sid,
                    segments=self._segments,
                    language=language,
                )
            )

        asyncio.ensure_future(do_publish())
