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
        audio_source: rtc.AudioSource,
        playout_source: AsyncIterable[rtc.AudioFrame],
        transcription_fwd: transcription.TTSSegmentsForwarder,
    ) -> None:
        self._playout_source = playout_source
        self._audio_source = audio_source
        self._tr_fwd = transcription_fwd
        self._interrupted = False
        self._int_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()
        self._speech_id = speech_id

        self._pushed_duration = 0.0

        self._total_played_time: float | None = None  # set whem the playout is done

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def time_played(self) -> float:
        if self._total_played_time is not None:
            return self._total_played_time

        return self._pushed_duration - self._audio_source.queued_duration

    def done(self) -> bool:
        logger.info(
            f"PlayoutHandle.done: self._done_fut.done(): {self._done_fut.done()}"
        )
        return self._done_fut.done() or self._interrupted

    def interrupt(self) -> None:
        if self.done():
            return

        self._int_fut.set_result(None)
        self._interrupted = True

    def join(self) -> asyncio.Future:
        return self._done_fut


class AgentPlayout(utils.EventEmitter[EventTypes]):
    def __init__(self, *, audio_source: rtc.AudioSource) -> None:
        super().__init__()
        self._audio_source = audio_source
        self._target_volume = 1.0
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
        return self._target_volume

    async def aclose(self) -> None:
        logger.info("Trying to close AgentPlayout")
        if self._closed:
            return

        self._closed = True

        if self._playout_atask is not None:
            await self._playout_atask

    def play(
        self,
        speech_id: str,
        playout_source: AsyncIterable[rtc.AudioFrame],
        transcription_fwd: transcription.TTSSegmentsForwarder,
    ) -> PlayoutHandle:
        if self._closed:
            raise ValueError("cancellable source is closed")

        handle = PlayoutHandle(
            speech_id=speech_id,
            audio_source=self._audio_source,
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
        logger.info(f"playout_task: handle: {handle}")
        if old_task is not None:
            logger.info(f"playout_task: old_task: {old_task}")
            await utils.aio.gracefully_cancel(old_task)

        if self._audio_source.queued_duration > 0:
            logger.info("DANGER: new playout while the source is still playing")
            # this should not happen, but log it just in case
            logger.warning(
                "new playout while the source is still playing",
                extra={
                    "speech_id": handle.speech_id,
                    "queued_duration": self._audio_source.queued_duration,
                },
            )

        first_frame = True

        @utils.log_exceptions(logger=logger)
        async def _capture_task():
            nonlocal first_frame
            # logger.info(f"[{handle.speech_id}]playout_task: _capture_task: handle: {handle}")
            async for frame in handle._playout_source:
                # logger.info(
                #     f"[{handle.speech_id}]playout_task: _capture_task: frame: {frame}"
                # )
                if first_frame:
                    logger.info(
                        f"[{handle.speech_id}]playout_task: _capture_task: handle._tr_fwd: {handle._tr_fwd}"
                    )
                    handle._tr_fwd.segment_playout_started()

                    logger.debug(
                        "speech playout started",
                        extra={"speech_id": handle.speech_id},
                    )

                    self.emit("playout_started")
                    first_frame = False

                handle._pushed_duration += frame.samples_per_channel / frame.sample_rate
                await self._audio_source.capture_frame(frame)
                # logger.info(f"playout_task: past self._audio_source.capture_frame")

            if self._audio_source.queued_duration > 0:
                # logger.info(
                #     f"playout_task: past self._audio_source.queued_duration > 0"
                # )
                await self._audio_source.wait_for_playout()
                # logger.info(f"playout_task: past self._audio_source.wait_for_playout")

        capture_task = asyncio.create_task(_capture_task())
        try:
            # logger.info(f"playout_task: capture_task: {capture_task}")
            await asyncio.wait(
                [capture_task, handle._int_fut],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # logger.info(f"playout_task: past await asyncio.wait")
        finally:
            # logger.info(f"playout_task: finally")
            await utils.aio.gracefully_cancel(capture_task)
            # logger.info(
            # f"playout_task: past await utils.aio.gracefully_cancel(capture_task)"
            # )

            handle._total_played_time = (
                handle._pushed_duration - self._audio_source.queued_duration
            )

            if handle.interrupted or capture_task.exception():
                # logger.info(
                #     f"playout_task: past if handle.interrupted or capture_task.exception()"
                # )
                self._audio_source.clear_queue()  # make sure to remove any queued frames
                # logger.info(f"playout_task: past self._audio_source.clear_queue")

            if not first_frame:
                # logger.info(f"playout_task: past if not first_frame: {not first_frame}")
                if not handle.interrupted:
                    # logger.info(
                    #     f"playout_task: past if not handle.interrupted: {not handle.interrupted}"
                    # )
                    handle._tr_fwd.segment_playout_finished()
                    # logger.info(
                    #     f"playout_task: past handle._tr_fwd.segment_playout_finished"
                    # )

                self.emit("playout_stopped", handle.interrupted)
                # logger.info(
                #     f"playout_task: past self.emit('playout_stopped', handle.interrupted)"
                # )

            await handle._tr_fwd.aclose()
            handle._done_fut.set_result(None)
            # logger.info(f"playout_task: past handle._done_fut.set_result(None)")

            logger.debug(
                "speech playout finished",
                extra={
                    "speech_id": handle.speech_id,
                    "interrupted": handle.interrupted,
                },
            )
