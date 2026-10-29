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
    capture, clear, queued = (
        source.capture_frame,
        source.clear_queue,
        source.queued_duration,
    )
    clears: dict[int, int] = {}
    held: dict[int, float] = {}

    async def capture_frame(self: rtc.AudioSource, frame: rtc.AudioFrame) -> None:
        if queued.fget(self) == 0.0:
            cleared = clears.get(id(self), 0)
            held[id(self)] = frame.duration
            try:
                await asyncio.sleep(hold)
            finally:
                del held[id(self)]
            if clears.get(id(self), 0) != cleared:
                return
        await capture(self, frame)

    def clear_queue(self: rtc.AudioSource) -> None:
        clears[id(self)] = clears.get(id(self), 0) + 1
        clear(self)

    source.capture_frame = capture_frame
    source.clear_queue = clear_queue
    # a held frame is queued and unplayed, so the room output's playhead accounting sees it so
    source.queued_duration = property(
        lambda self: queued.fget(self) + held.get(id(self), 0.0)
    )
    source._playout_hold = hold
