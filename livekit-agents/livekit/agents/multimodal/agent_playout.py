from __future__ import annotations

import asyncio
from typing import AsyncIterable, Literal

from livekit import rtc
from livekit.agents import transcription, utils

from ..log import logger

EventTypes = Literal["playout_started", "playout_stopped"]


class PlayoutHandle:
    def __init__(
        self,
        *,
        audio_source: rtc.AudioSource,
        item_id: str,
        content_index: int,
        transcription_fwd: transcription.TTSSegmentsForwarder,
    ) -> None:
        self._audio_source = audio_source
        self._tr_fwd = transcription_fwd
        self._item_id = item_id
        self._content_index = content_index

        self._int_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()

        self._interrupted = False

        self._pushed_duration = 0.0
        self._total_played_time: float | None = None  # set when the playout is done

    @property
    def item_id(self) -> str:
        return self._item_id

    @property
    def audio_samples(self) -> int:
        if self._total_played_time is not None:
            return int(self._total_played_time * 24000)

        return int((self._pushed_duration - self._audio_source.queued_duration) * 24000)

    @property
    def text_chars(self) -> int:
        return len(self._tr_fwd.played_text)

    @property
    def content_index(self) -> int:
        return self._content_index

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


class AgentPlayout(utils.EventEmitter[EventTypes]):
    def __init__(self, *, audio_source: rtc.AudioSource) -> None:
        super().__init__()
        self._source = audio_source
        self._playout_atask: asyncio.Task[None] | None = None

    def play(
        self,
        *,
        item_id: str,
        content_index: int,
        transcription_fwd: transcription.TTSSegmentsForwarder,
        text_stream: AsyncIterable[str],
        audio_stream: AsyncIterable[rtc.AudioFrame],
    ) -> PlayoutHandle:
        handle = PlayoutHandle(
            audio_source=self._source,
            item_id=item_id,
            content_index=content_index,
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

            handle._tr_fwd.mark_text_segment_end()

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
                    self.emit("playout_started")
                    first_frame = False

                handle._tr_fwd.push_audio(frame)

                for f in bstream.write(frame.data.tobytes()):
                    handle._pushed_duration += f.samples_per_channel / f.sample_rate
                    await self._source.capture_frame(f)

            for f in bstream.flush():
                handle._pushed_duration += f.samples_per_channel / f.sample_rate
                await self._source.capture_frame(f)

            handle._tr_fwd.mark_audio_segment_end()

            await self._source.wait_for_playout()

        read_text_task = asyncio.create_task(_play_text_stream())
        capture_task = asyncio.create_task(_capture_task())

        try:
            await asyncio.wait(
                [capture_task, handle._int_fut],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            await utils.aio.gracefully_cancel(capture_task)

            handle._total_played_time = (
                handle._pushed_duration - self._source.queued_duration
            )

            if handle.interrupted or capture_task.exception():
                self._source.clear_queue()  # make sure to remove any queued frames

            await utils.aio.gracefully_cancel(read_text_task)

            # make sure the text_data.sentence_stream is closed
            handle._tr_fwd.mark_text_segment_end()

            if not first_frame and not handle.interrupted:
                handle._tr_fwd.segment_playout_finished()

            await handle._tr_fwd.aclose()
            handle._done_fut.set_result(None)

            # emit playout_stopped after the transcription forwarder has been closed
            if not first_frame:
                self.emit("playout_stopped", handle.interrupted)
