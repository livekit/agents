from __future__ import annotations

import asyncio

from livekit import rtc

from .. import stt, utils
from ..log import logger
from . import _utils


class STTSegmentsForwarder:
    """
    Forward STT transcription to the users. (Useful for client-side rendering)
    """

    def __init__(
        self,
        *,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        interim_transcript_timeout: float = 7.5,
    ):
        identity = participant if isinstance(participant, str) else participant.identity
        if track is None:
            track = _utils.find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        self._room = room
        self._participant_identity = identity
        self._track_id = track

        self._queue = asyncio.Queue[rtc.TranscriptionSegment]()
        self._main_task = asyncio.create_task(self._run())
        self._current_id = _utils.segment_uuid()

        self._invalidate_interim_timeout: utils.aio.Sleep | None = None

    @utils.log_exceptions(logger=logger)
    async def _run(self):
        while True:
            seg = await self._queue.get()
            await self._room.local_participant.publish_transcription(
                transcription=rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_sid=self._track_id,
                    segments=[seg],  # no history for now
                )
            )

    def update(self, ev: stt.SpeechEvent):
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # first alternative is always the "most likely to be true"

            if self._invalidate_interim_timeout is not None:
                self._invalidate_interim_timeout.reset()

            alt = ev.alternatives[0]
            self._queue.put_nowait(
                rtc.TranscriptionSegment(
                    id=self._current_id,
                    text=alt.text,
                    start_time=0,
                    end_time=0,
                    final=False,
                    language=alt.language,
                )
            )
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            alt = ev.alternatives[0]
            self._queue.put_nowait(
                rtc.TranscriptionSegment(
                    id=self._current_id,
                    text=alt.text,
                    start_time=0,
                    end_time=0,
                    final=True,
                    language=alt.language,
                )
            )

            self._current_id = _utils.segment_uuid()

    async def aclose(self) -> None:
        await utils.aio.gracefully_cancel(self._main_task)
