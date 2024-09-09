from __future__ import annotations

import asyncio
from livekit import rtc
from livekit.agents import utils, transcription

from ..log import logger
from . import proto


class PlayoutHandle:
    def __init__(
        self,
        *,
        message_id: str,
        transcription_fwd: transcription.TTSSegmentsForwarder,
    ) -> None:
        self._playout_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._tr_fwd = transcription_fwd
        self._message_id = message_id
        self._audio_samples: int = 0
        self._done_fut = asyncio.Future[None]()
        self._interrupted = False

    @property
    def message_id(self) -> str:
        return self._message_id

    @property
    def audio_samples(self) -> int:
        return self._audio_samples

    @property
    def text_chars(self) -> int:
        return len(self._tr_fwd.played_text)

    def push_audio(self, data: bytes) -> None:
        frame = rtc.AudioFrame(
            data,
            proto.SAMPLE_RATE,
            proto.NUM_CHANNELS,
            len(data) // 2,
        )

        self._tr_fwd.push_audio(frame)
        self._playout_ch.send_nowait(frame)

    def push_text(self, text: str) -> None:
        self._tr_fwd.push_text(text)

    def end_input(self) -> None:
        self._tr_fwd.mark_text_segment_end()
        self._tr_fwd.mark_audio_segment_end()
        self._playout_ch.close()

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def done(self) -> bool:
        return self._done_fut.done() or self._interrupted

    def interrupt(self) -> None:
        if self.done():
            return

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
    ) -> PlayoutHandle:
        handle = PlayoutHandle(
            message_id=message_id,
            transcription_fwd=transcription_fwd,
        )
        self._playout_atask = asyncio.create_task(
            self._playout_task(self._playout_atask, handle)
        )

        return handle

    @utils.log_exceptions(logger=logger)
    async def _playout_task(
        self, old_task: asyncio.Task[None], handle: PlayoutHandle
    ) -> None:
        if old_task is not None:
            await utils.aio.gracefully_cancel(old_task)

        first_frame = True

        try:
            samples_per_channel = proto.OUT_FRAME_SIZE
            bstream = utils.audio.AudioByteStream(
                proto.SAMPLE_RATE,
                proto.NUM_CHANNELS,
                samples_per_channel=samples_per_channel,
            )

            async for frame in handle._playout_ch:
                if first_frame:
                    handle._tr_fwd.segment_playout_started()

                for f in bstream.write(frame.data.tobytes()):
                    handle._audio_samples += samples_per_channel
                    if handle.interrupted:
                        break

                    await self._source.capture_frame(f)

                if handle.interrupted:
                    break

            if not handle.interrupted:
                for f in bstream.flush():
                    await self._source.capture_frame(f)

        finally:
            if not first_frame:
                if not handle.interrupted:
                    handle._tr_fwd.segment_playout_finished()

            await handle._tr_fwd.aclose()

            handle._done_fut.set_result(None)
