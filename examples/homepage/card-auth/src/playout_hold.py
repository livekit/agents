"""Holds the first 100 ms of every burst the agent plays, so GPT-Live's delivery jitter can't
run the room's playback dry mid-word. The native source fills a dry spell with digital silence,
and each one is a click."""

from __future__ import annotations

import asyncio

from livekit import rtc

HOLD_SECONDS = 0.1


def hold_playout(source: type = rtc.AudioSource, hold: float = HOLD_SECONDS) -> None:
    # shortcut: patches livekit-rtc's AudioSource because livekit-agents 1.8 forwards each
    # realtime frame the moment it arrives; drop once the SDK buffers realtime playback itself
    if getattr(source, "_playout_hold", None):
        return
    capture, clear = source.capture_frame, source.clear_queue
    clears: dict[int, int] = {}

    async def capture_frame(self: rtc.AudioSource, frame: rtc.AudioFrame) -> None:
        if self.queued_duration == 0.0:
            cleared = clears.get(id(self), 0)
            await asyncio.sleep(hold)
            if clears.get(id(self), 0) != cleared:
                return
        await capture(self, frame)

    def clear_queue(self: rtc.AudioSource) -> None:
        clears[id(self)] = clears.get(id(self), 0) + 1
        clear(self)

    source.capture_frame = capture_frame
    source.clear_queue = clear_queue
    source._playout_hold = hold
