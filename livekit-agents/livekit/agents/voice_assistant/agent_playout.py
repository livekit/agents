from __future__ import annotations

import asyncio
from typing import AsyncIterable, Literal, Optional

from livekit import rtc

from .. import stv as speech_to_video
from .. import transcription, utils
from .log import logger

EventTypes = Literal["playout_started", "playout_stopped"]

# just a random small number so far
VIDEO_CAPTURE_LATENCY = 0.001

class PlayoutHandle:
    """
    Handles the playout of a single agent utterance.
    """
    def __init__(
        self,
        speech_id: str,
        audio_source: rtc.AudioSource,
        transcription_fwd: transcription.TTSSegmentsForwarder,
        audio_playout_source: AsyncIterable[rtc.AudioFrame],
        video_playout_source: Optional[AsyncIterable[rtc.VideoFrameEvent]] = None,
    ) -> None:
        self._audio_playout_source = audio_playout_source
        self._video_playout_source = video_playout_source
        self._audio_source = audio_source
        self._tr_fwd = transcription_fwd
        self._interrupted = False
        self._int_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()
        self._speech_id = speech_id

        self._audio_pushed_duration = 0.0
        self._total_played_time: float | None = None  # set when the playout is done

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def audio_time_played(self) -> float:
        if self._total_played_time is not None:
            return self._total_played_time

        return self._audio_pushed_duration - self._audio_source.queued_duration

    def done(self) -> bool:
        return self._done_fut.done() or self._interrupted

    def interrupt(self) -> None:
        if self.done():
            return

        self._int_fut.set_result(None)
        self._interrupted = True

    def join(self) -> asyncio.Future:
        return self._done_fut


class AgentPlayout(utils.EventEmitter[EventTypes]):
    """
    Owns the agent output sources and handles the synchronization of the transcript / audio / video playout.
    Generates idle video when the agent is not speaking [if a STV is provided].
    """
    def __init__(
        self,
        *,
        audio_source: rtc.AudioSource,
        video_source: Optional[rtc.VideoSource] = None,
        stv: Optional[speech_to_video.STV] = None,
    ) -> None:
        super().__init__()
        self._audio_source = audio_source
        self._video_source = video_source
        self._stv = stv

        self._target_volume = 1.0

        self._closed = False
        self._speaking = False

        self._playout_atask: asyncio.Task[None] | None = None
        self._idle_video_atask: asyncio.Task[None] | None = None

        if self._stv is not None:
            assert self._video_source is not None
            self._idle_video_atask = asyncio.create_task(self._idle_video_task())

    @property
    def target_volume(self) -> float:
        return self._target_volume

    @target_volume.setter
    def target_volume(self, value: float) -> None:
        self._target_volume = value

    @property
    def smoothed_volume(self) -> float:
        return self._target_volume

    @utils.log_exceptions(logger=logger)
    async def _idle_video_task(self) -> None:
        """
        Send idle video frames to the VideoSource when the agent is not speaking.
        """
        idle_video_stream: AsyncIterable[rtc.VideoFrame] = self._stv.idle_stream()
        assert 1 < self._stv.fps < 120

        try:
            while True:
                if self._speaking:
                    await asyncio.sleep(0.1)  # Short sleep to avoid busy waiting
                    continue

                last_render_time = asyncio.get_event_loop().time()
                async for frame in idle_video_stream:
                    if self._speaking:
                        break

                    current_time = asyncio.get_event_loop().time()
                    time_since_last_frame = current_time - last_render_time
                    target_frame_duration = 1 / self._stv.fps

                    if time_since_last_frame < target_frame_duration:
                        await asyncio.sleep(target_frame_duration - time_since_last_frame)

                    frame = self._stv.render_frame(frame)
                    self._video_source.capture_frame(frame)
                    last_render_time = asyncio.get_event_loop().time()
        finally:
            await idle_video_stream.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._playout_atask is not None:
            await self._playout_atask

        if self._idle_video_atask is not None:
            await utils.aio.gracefully_cancel(self._idle_video_atask)
            await self._idle_video_atask

    def play(
        self,
        speech_id: str,
        transcription_fwd: transcription.TTSSegmentsForwarder,
        audio_playout_source: AsyncIterable[rtc.AudioFrame],
        video_playout_source: Optional[AsyncIterable[rtc.VideoFrameEvent]] = None,
    ) -> PlayoutHandle:
        if self._closed:
            raise ValueError("cancellable source is closed")

        handle = PlayoutHandle(
            speech_id=speech_id,
            audio_source=self._audio_source,
            transcription_fwd=transcription_fwd,
            audio_playout_source=audio_playout_source,
            video_playout_source=video_playout_source,
        )
        self._playout_atask = asyncio.create_task(
            self._playout_task(self._playout_atask, handle)
        )

        return handle

    @utils.log_exceptions(logger=logger)
    async def _playout_task(
        self, old_task: asyncio.Task[None] | None, handle: PlayoutHandle
    ) -> None:
        if old_task is not None:
            await utils.aio.gracefully_cancel(old_task)

        if self._audio_source.queued_duration > 0:
            # this should not happen, but log it just in case
            logger.warning(
                "new playout while the source is still playing",
                extra={
                    "speech_id": handle.speech_id,
                    "queued_duration": self._audio_source.queued_duration,
                },
            )

        first_audio_frame = True
        first_video_frame = True
        start_time = None

        @utils.log_exceptions(logger=logger)
        async def _audio_capture_task():
            nonlocal first_audio_frame, start_time

            async for frame in handle._audio_playout_source:
                if first_audio_frame:

                    handle._tr_fwd.segment_playout_started()
                    logger.debug(
                        "started playing the first audio frame",
                        extra={"speech_id": handle.speech_id},
                    )
                    self.emit("playout_started")
                    first_audio_frame = False

                    start_time = asyncio.get_event_loop().time()

                handle._audio_pushed_duration += frame.samples_per_channel / frame.sample_rate
                for retry in range(3):
                    try:
                        await self._audio_source.capture_frame(frame)
                        break
                    except Exception as e:
                        if retry == 2:
                            logger.error(f"Failed to capture audio frame after 3 attempts: {e}")
                        else:
                            logger.warning(f"Error capturing audio frame (attempt {retry + 1}): {e}")
                            await asyncio.sleep(0.01)  # Short delay before retrying

            await self._audio_source.wait_for_playout()
            logger.debug("audio playout finished")

        @utils.log_exceptions(logger=logger)
        async def _video_capture_task():
            nonlocal start_time, first_video_frame
            while start_time is None:
                await asyncio.sleep(0.01)

            async for event in handle._video_playout_source:
                if first_video_frame:
                    logger.debug(
                        "started playing the first video frame",
                        extra={"speech_id": handle.speech_id},
                    )
                    first_video_frame = False
                    self._speaking = True


                target_time = start_time + event.timestamp_us / 1000
                wait_time = target_time - asyncio.get_event_loop().time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time - VIDEO_CAPTURE_LATENCY)

                frame = self._stv.render_frame(event.frame)
                self._video_source.capture_frame(frame)

            logger.debug("video playout finished")

        tasks = [asyncio.create_task(_audio_capture_task())]
        if handle._video_playout_source and self._video_source:
            tasks.append(asyncio.create_task(_video_capture_task()))

        try:
            await asyncio.wait(
                [asyncio.gather(*tasks), handle._int_fut],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            self._speaking = False
            for task in tasks:
                await utils.aio.gracefully_cancel(task)

            handle._total_played_time = (
                handle._audio_pushed_duration - self._audio_source.queued_duration
            )

            if handle.interrupted or tasks[0].exception():
                self._audio_source.clear_queue()  # make sure to remove any queued frames

            if not first_audio_frame:
                if not handle.interrupted:
                    handle._tr_fwd.segment_playout_finished()

                self.emit("playout_stopped", handle.interrupted)

            await handle._tr_fwd.aclose()
            handle._done_fut.set_result(None)

            logger.debug(
                "playout finished",
                extra={
                    "speech_id": handle.speech_id,
                    "interrupted": handle.interrupted,
                },
            )
