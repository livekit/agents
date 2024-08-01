from __future__ import annotations

import asyncio
from typing import AsyncIterable, Literal

from livekit import rtc

from .. import transcription, utils
from .log import logger

EventTypes = Literal["playout_started", "playout_stopped"]


class PlayoutHandle:
    def __init__(
        self,
        speech_id: str,
        playout_source: AsyncIterable[rtc.AudioFrame],
        transcription_fwd: transcription.TTSSegmentsForwarder | None = None,
    ) -> None:
        self._playout_source = playout_source
        self._tr_fwd = transcription_fwd
        self._interrupted = False
        self._time_played = 0.0
        self._done_fut = asyncio.Future[None]()
        self._speech_id = speech_id

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def time_played(self) -> float:
        return self._time_played

    def done(self) -> bool:
        return self._done_fut.done()

    def interrupt(self) -> None:
        if self.done():
            return

        self._interrupted = True

    def join(self) -> asyncio.Future:
        return self._done_fut


class AgentPlayout(utils.EventEmitter[EventTypes]):
    def __init__(self, *, source: rtc.AudioSource, alpha: float = 0.95) -> None:
        super().__init__()
        self._source = source
        self._target_volume = 1.0
        self._vol_filter = utils.ExpFilter(alpha=alpha)
        self._playout_atask: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def target_volume(self) -> float:
        return self._target_volume

    @target_volume.setter
    def target_volume(self, value: float) -> None:
        self._target_volume = value

    @property
    def smoothed_volume(self) -> float:
        return self._vol_filter.filtered()

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._playout_atask is not None:
            await self._playout_atask

    def play(
        self,
        speech_id: str,
        playout_source: AsyncIterable[rtc.AudioFrame],
        transcription_fwd: transcription.TTSSegmentsForwarder | None = None,
    ) -> PlayoutHandle:
        if self._closed:
            raise ValueError("cancellable source is closed")

        handle = PlayoutHandle(
            speech_id=speech_id,
            playout_source=playout_source,
            transcription_fwd=transcription_fwd,
        )
        self._playout_atask = asyncio.create_task(
            self._playout_task(self._playout_atask, handle)
        )

        return handle

    @utils.log_exceptions(logger=logger)
    async def _playout_task(
        self, old_task: asyncio.Task[None] | None, handle: PlayoutHandle
    ) -> None:
        def _should_break():
            eps = 0.1
            return handle.interrupted and self._vol_filter.filtered() <= eps

        first_frame = True
        cancelled = False

        try:
            if old_task is not None:
                await utils.aio.gracefully_cancel(old_task)

            async for frame in handle._playout_source:
                if first_frame:
                    if handle._tr_fwd is not None:
                        handle._tr_fwd.segment_playout_started()

                    logger.debug(
                        "started playing the first frame",
                        extra={"speech_id": handle.speech_id},
                    )

                    self.emit("playout_started")
                    first_frame = False

                if _should_break():
                    cancelled = True
                    break

                # divide the frame by chunks of 20ms
                ms20 = frame.sample_rate // 50
                i = 0
                while i < len(frame.data):
                    if _should_break():
                        cancelled = True
                        break

                    rem = min(ms20, len(frame.data) - i)
                    data = frame.data[i : i + rem]
                    i += rem

                    tv = self._target_volume if not handle.interrupted else 0.0
                    dt = 1 / len(data)
                    for si in range(0, len(data)):
                        vol = self._vol_filter.apply(dt, tv)
                        data[si] = int((data[si] / 32768) * vol * 32768)

                    chunk_frame = rtc.AudioFrame(
                        data=data.tobytes(),
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=rem,
                    )
                    await self._source.capture_frame(chunk_frame)
                    handle._time_played += rem / frame.sample_rate
        finally:
            if not first_frame:
                if handle._tr_fwd is not None and not cancelled:
                    handle._tr_fwd.segment_playout_finished()

                self.emit("playout_stopped", cancelled)

            handle._done_fut.set_result(None)
            if handle._tr_fwd is not None:
                await handle._tr_fwd.aclose()

            logger.debug(
                "playout finished",
                extra={"speech_id": handle.speech_id, "cancelled": cancelled},
            )
