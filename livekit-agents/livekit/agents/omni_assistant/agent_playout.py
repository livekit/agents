from __future__ import annotations

import asyncio
from typing import AsyncIterable
from livekit import rtc
from livekit.agents import utils, transcription
from livekit.agents.utils import audio

from ..log import logger


class PlayoutHandle:
    def __init__(
        self,
        *,
        audio_source: rtc.AudioSource,
        message_id: str,
        transcription_fwd: transcription.TTSSegmentsForwarder,
    ) -> None:
        self._audio_source = audio_source
        self._tr_fwd = transcription_fwd
        self._message_id = message_id

        self._int_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()

        self._interrupted = False

        self._pushed_duration = 0.0
        self._total_played_time: float | None = None  # set when the playout is done

    @property
    def message_id(self) -> str:
        return self._message_id

    @property
    def audio_samples(self) -> int:
        if self._total_played_time is not None:
            return int(self._total_played_time * 24000)

        return int((self._pushed_duration - self._audio_source.queued_duration) * 24000)

    @property
    def text_chars(self) -> int:
        return len(self._tr_fwd.played_text)

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def done(self) -> bool:
        return self._done_fut.done() or self._interrupted

    def interrupt(self) -> None:
        if self.done():
            return

        self._int_fut.set_result(None)
        self._interrupted = True


class AgentPlayout:
    def __init__(self, *, audio_source: rtc.AudioSource) -> None:
        self._source = audio_source
        self._playout_atask: asyncio.Task[None] | None = None

    def play(
        self,
        *,
        message_id: str,
        transcription_fwd: transcription.TTSSegmentsForwarder,
        text_stream: AsyncIterable[str],
        audio_stream: AsyncIterable[rtc.AudioFrame],
    ) -> PlayoutHandle:
        handle = PlayoutHandle(
            audio_source=self._source,
            message_id=message_id,
            transcription_fwd=transcription_fwd,
        )
        self._playout_atask = asyncio.create_task(
            self._playout_task(self._playout_atask, handle, text_stream, audio_stream)
        )

        return handle

    @utils.log_exceptions(logger=logger)
    async def _playout_task(
        self,
        old_task: asyncio.Task[None],
        handle: PlayoutHandle,
        text_stream: AsyncIterable[str],
        audio_stream: AsyncIterable[rtc.AudioFrame],
    ) -> None:
        if old_task is not None:
            await utils.aio.gracefully_cancel(old_task)

        first_frame = True

        @utils.log_exceptions(logger=logger)
        async def _play_text_stream():
            async for text in text_stream:
                handle._tr_fwd.push_text(text)

        @utils.log_exceptions(logger=logger)
        async def _capture_task():
            nonlocal first_frame

            samples_per_channel = 1200
            bstream = utils.audio.AudioByteStream(
                24000,
                1,
                samples_per_channel=samples_per_channel,
            )

            async for frame in audio_stream:
                if first_frame:
                    handle._tr_fwd.segment_playout_started()
                    first_frame = False

                handle._tr_fwd.push_audio(frame)

                for f in bstream.write(frame.data.tobytes()):
                    handle._pushed_duration += samples_per_channel / f.sample_rate
                    await self._source.capture_frame(f)

            for f in bstream.flush():
                handle._pushed_duration += samples_per_channel / f.sample_rate
                await self._source.capture_frame(f)

            await self._source.wait_for_playout()

        read_text_task = asyncio.create_task(_play_text_stream())
        capture_task = asyncio.create_task(_capture_task())

        try:
            await asyncio.wait(
                [capture_task, handle._int_fut],
                return_when=asyncio.FIRST_COMPLETED,
            )
            await read_text_task
        finally:
            await utils.aio.gracefully_cancel(capture_task)
            await utils.aio.gracefully_cancel(read_text_task)

            handle._total_played_time = (
                handle._pushed_duration - self._source.queued_duration
            )

            if handle.interrupted or capture_task.exception():
                self._source.clear_queue()  # make sure to remove any queued frames

            if not first_frame:
                if not handle.interrupted:
                    handle._tr_fwd.segment_playout_finished()

            await handle._tr_fwd.aclose()
            handle._done_fut.set_result(None)
