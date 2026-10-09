from __future__ import annotations

from typing import Literal

from livekit import rtc
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.agents.voice.io import AudioOutputCapabilities


class ResumableQueueAudioOutput(
    QueueAudioOutput,
    rtc.EventEmitter[Literal["clear_buffer", "pause", "resume"]],
):
    """Queue output that exposes LiveKit's native pause/resume lifecycle.

    SpatialReal performs the provider-side pause emulation. This output only
    advertises the capability and forwards the framework lifecycle events.
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        wait_playback_start: bool = False,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            wait_playback_start=wait_playback_start,
        )
        self._capabilities = AudioOutputCapabilities(pause=True)
        self._paused = False
        self._logical_capture_active = False
        self._playback_started_notified = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._logical_capture_active:
            self._logical_capture_active = True
            self._playback_started_notified = False
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self._logical_capture_active = False

    def pause(self) -> None:
        if self._paused:
            return
        self._paused = True
        self.emit("pause")  # type: ignore[arg-type]

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        self.emit("resume")  # type: ignore[arg-type]

    def clear_buffer(self) -> None:
        self._paused = False
        super().clear_buffer()

    def notify_playback_started(self) -> None:
        if self._playback_started_notified:
            return
        self._playback_started_notified = True
        super().notify_playback_started()

    def notify_playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
    ) -> None:
        super().notify_playback_finished(playback_position, interrupted)
        self._playback_started_notified = False
