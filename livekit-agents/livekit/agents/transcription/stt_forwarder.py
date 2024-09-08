from __future__ import annotations

import asyncio
import contextlib
from typing import Awaitable, Callable, Optional, Union

from livekit import rtc

from .. import stt
from ..log import logger
from . import _utils

BeforeForwardCallback = Callable[
    ["STTSegmentsForwarder", rtc.Transcription],
    Union[rtc.Transcription, Awaitable[Optional[rtc.Transcription]]],
]


WillForwardTranscription = BeforeForwardCallback


def _default_before_forward_cb(
    fwd: STTSegmentsForwarder, transcription: rtc.Transcription
) -> rtc.Transcription:
    return transcription


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
        before_forward_cb: BeforeForwardCallback = _default_before_forward_cb,
        # backward compatibility
        will_forward_transcription: WillForwardTranscription | None = None,
    ):
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

        self._room, self._participant_identity, self._track_id = room, identity, track
        self._before_forward_cb = before_forward_cb
        self._queue = asyncio.Queue[Optional[rtc.TranscriptionSegment]]()
        self._main_task = asyncio.create_task(self._run())
        self._current_id = _utils.segment_uuid()

    async def _run(self):
        try:
            while True:
                seg = await self._queue.get()
                if seg is None:
                    break

                base_transcription = rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_sid=self._track_id,
                    segments=[seg],  # no history for now
                )

                transcription = self._before_forward_cb(self, base_transcription)
                if asyncio.iscoroutine(transcription):
                    transcription = await transcription

                if not isinstance(transcription, rtc.Transcription):
                    transcription = _default_before_forward_cb(self, base_transcription)

                if transcription.segments and self._room.isconnected():
                    await self._room.local_participant.publish_transcription(
                        transcription
                    )

        except Exception:
            logger.exception("error in stt transcription")

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
                    language="",  # TODO
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
                    language="",  # TODO
                )
            )

            self._current_id = _utils.segment_uuid()

    async def aclose(self, *, wait: bool = True) -> None:
        self._queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
