from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from types import TracebackType
from typing import Literal

import numpy as np
from pydantic import BaseModel

from livekit import rtc

from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.participant import wait_for_participant_attribute, wait_for_track_publication
from .tone import _CNGToneDetector

_SIP_CALL_STATUS_ATTR = "sip.callStatus"
_SIP_CALL_STATUS_ACTIVE = "active"
_SAMPLE_RATE = 16000
_TIMEOUT = 15.0


class FaxDetectedEvent(BaseModel):
    """Emitted by :class:`FaxDetector` once a fax calling tone has been confirmed."""

    type: Literal["fax_detected"] = "fax_detected"
    participant_identity: str
    """Identity of the participant whose audio contained the fax tone."""
    detected_after: float
    """Seconds of caller audio heard before the fax was confirmed."""


class FaxDetector(EventEmitter[Literal["fax_detected"]]):
    """Detects calls placed by a fax machine.

    A calling fax machine sends a CNG tone (1100 Hz, 0.5 s on, 3 s off; ITU-T T.30)
    and no speech, so speech-based answering machine detection can't recognize it.
    This detector reads its own copy of the caller's audio track, before any noise
    cancellation applied by the session, and confirms a fax after two correctly
    spaced CNG beeps (about 4 s of audio).

    Detection only reports; the caller decides what to do (tag the call, hang up,
    transfer to a fax line). It works for inbound and outbound calls. For SIP
    participants, listening starts once ``sip.callStatus`` is ``active``.

    Example::

        async with FaxDetector(ctx.room) as detector:
            result = await detector.wait()
        if result is not None:
            ctx.delete_room()

    Args:
        room: The connected room the caller is in.
        participant_identity: Identity of the participant to listen to. If omitted,
            the first remote audio track is used.
        timeout: Wall-clock seconds to listen for, counted from when the caller's
            audio becomes available. After this, detection stops and :meth:`wait`
            returns ``None``.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        timeout: float = _TIMEOUT,
    ) -> None:
        super().__init__()
        self._room = room
        self._participant_identity = participant_identity
        self._timeout = timeout
        self._task: asyncio.Task[None] | None = None
        self._result: FaxDetectedEvent | None = None
        self._done = asyncio.Event()

    @property
    def result(self) -> FaxDetectedEvent | None:
        """The detection result, or ``None`` if no fax has been detected (yet)."""
        return self._result

    def start(self) -> None:
        """Start listening in the background.

        Calling this more than once has no effect. :meth:`wait` and the async context
        manager call it for you.
        """
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="fax_detector")
        self._task.add_done_callback(lambda _: self._done.set())

    async def wait(self) -> FaxDetectedEvent | None:
        """Wait until a fax is detected, or until detection stops.

        Returns the detection event, or ``None`` if no fax was found before the
        timeout, the caller left, or the detector was closed.
        """
        self.start()
        await self._done.wait()
        return self._result

    async def aclose(self) -> None:
        """Stop listening and release the audio stream.

        Safe to call more than once. A pending :meth:`wait` returns the current
        :attr:`result`.
        """
        if self._task is not None:
            await aio.cancel_and_wait(self._task)
        self._done.set()

    async def __aenter__(self) -> FaxDetector:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _run(self) -> None:
        identity = self._participant_identity if is_given(self._participant_identity) else None
        try:
            publication = await wait_for_track_publication(
                self._room,
                identity=identity,
                kind=rtc.TrackKind.KIND_AUDIO,
                wait_for_subscription=True,
            )
            participant = next(
                (
                    p
                    for p in self._room.remote_participants.values()
                    if publication.sid in p.track_publications
                ),
                None,
            )
            if participant is None or publication.track is None:
                return

            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                await wait_for_participant_attribute(
                    self._room,
                    identity=participant.identity,
                    attribute=_SIP_CALL_STATUS_ATTR,
                    value=_SIP_CALL_STATUS_ACTIVE,
                )
        except RuntimeError as e:
            logger.debug("fax detection stopped: participant left", extra={"error": str(e)})
            return

        stream = rtc.AudioStream.from_track(
            track=publication.track, sample_rate=_SAMPLE_RATE, num_channels=1
        )
        try:
            await self._detect(participant.identity, (ev.frame async for ev in stream))
        finally:
            await stream.aclose()

    async def _detect(self, identity: str, frames: AsyncIterable[rtc.AudioFrame]) -> None:
        """Consume ``frames`` until a fax is confirmed or the timeout elapses."""
        try:
            await asyncio.wait_for(self._consume(identity, frames), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.debug("fax detection timed out", extra={"participant": identity})

    async def _consume(self, identity: str, frames: AsyncIterable[rtc.AudioFrame]) -> None:
        tone = _CNGToneDetector(_SAMPLE_RATE)
        async for frame in frames:
            if not tone.push(np.frombuffer(frame.data, dtype=np.int16)):
                continue
            assert tone.detected_at is not None
            self._result = FaxDetectedEvent(
                participant_identity=identity, detected_after=tone.detected_at
            )
            logger.info(
                "fax detected",
                extra={"participant": identity, "detected_after": tone.detected_at},
            )
            self.emit("fax_detected", self._result)
            return
