from livekit import rtc
from ..log import logger
from .. import stt
import contextlib
import uuid
import asyncio


def _uuid() -> str:
    return str(uuid.uuid4())[:12]


class STTSegmentsForwarder:
    """
    Forward the STT transcription and keep the right timing info
    """

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
        self._current_id = _uuid()

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

            self._current_id = _uuid()

    async def aclose(self, *, wait=True) -> None:
        self._queue.put_nowait(None)

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
