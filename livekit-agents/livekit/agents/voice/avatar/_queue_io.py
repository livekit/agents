from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Literal, Union

from livekit import rtc

from ... import utils
from ..io import AudioOutput
from ._types import AudioReceiver, AudioSegmentEnd

logger = logging.getLogger(__name__)


class QueueAudioOutput(
    AudioOutput,
    AudioReceiver,
    rtc.EventEmitter[Literal["playback_finished", "clear_buffer"]],
):
    """
    AudioOutput implementation that sends audio frames through a queue.
    """

    def __init__(self, *, sample_rate: int | None = None):
        super().__init__(label="DebugQueueIO", next_in_chain=None, sample_rate=sample_rate)
        self._data_ch = utils.aio.Chan[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._capturing = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and queue audio frame"""
        await super().capture_frame(frame)
        if not self._capturing:
            self._capturing = True

        await self._data_ch.send(frame)

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if not self._capturing:
            return
        self._capturing = False
        self._data_ch.send_nowait(AudioSegmentEnd())

    # as AudioReceiver for AvatarRunner

    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        while True:
            try:
                self._data_ch.recv_nowait()
            except utils.aio.channel.ChanEmpty:
                break
        self.emit("clear_buffer")  # type: ignore

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self.on_playback_finished(playback_position=playback_position, interrupted=interrupted)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self._data_ch
