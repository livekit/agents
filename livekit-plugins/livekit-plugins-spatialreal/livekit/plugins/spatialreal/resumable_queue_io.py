from __future__ import annotations

from typing import Literal

from livekit import rtc
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.agents.voice.io import AudioOutputCapabilities


class ResumableQueueAudioOutput(
    QueueAudioOutput,
    rtc.EventEmitter[Literal["clear_buffer", "pause", "resume"]],
):
    """Queue output that forwards the framework's pause/resume to the avatar.

    Pause is advertised only when the SpatialReal server can hold playback; otherwise
    the framework falls back to interrupting outright.
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        wait_playback_start: bool = False,
        can_pause: bool = False,
    ) -> None:
        super().__init__(sample_rate=sample_rate, wait_playback_start=wait_playback_start)
        self._capabilities = AudioOutputCapabilities(pause=can_pause)
        self._paused = False

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
