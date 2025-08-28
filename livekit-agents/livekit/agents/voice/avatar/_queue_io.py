from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Literal, Union

from livekit import rtc

from ... import utils
from ..io import AudioOutput, AudioOutputCapabilities
from ._types import AudioReceiver, AudioSegmentEnd

logger = logging.getLogger(__name__)


class QueueAudioOutput(AudioReceiver[Literal["playback_finished"]], AudioOutput):
    """
    AudioOutput implementation that sends audio frames through a queue.
    """

    def __init__(self, *, sample_rate: int | None = None, can_pause: bool = False):
        super().__init__(
            label="DebugQueueIO",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=AudioOutputCapabilities(pause=can_pause),
        )
        self._data_ch = utils.aio.Chan[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._capturing = False
        self.__paused = False

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

    def resume(self) -> None:
        if not self.__paused:
            return

        self.emit("resume")
        self.__paused = False

    def pause(self) -> None:
        if not self.can_pause:
            logger.warning("pause is not supported by QueueAudioOutput")
            return

        if self.__paused:
            return

        self.emit("pause")
        self.__paused = True

    # as AudioReceiver for AvatarRunner

    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        while True:
            try:
                self._data_ch.recv_nowait()
            except utils.aio.channel.ChanEmpty:
                break
        self.emit("clear_buffer")

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self.on_playback_finished(playback_position=playback_position, interrupted=interrupted)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self._data_ch
