"""A stand-in for OjinSTVClient, plus frame builders and scripted tick sequences.

The emitter is deliberately kwarg-faithful to the real SDK (which calls handlers as
``cb(**kwargs)`` and swallows their TypeErrors): a handler that cannot accept the
kwargs the SDK actually sends must fail a test here rather than fail silently in
production.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ojin.stv import FrameType, STVAudioFrame, STVEvent, STVVideoFrame

_BYTES_PER_SAMPLE = 2


def make_video_frame(
    width: int = 1024,
    height: int = 1024,
    *,
    frame_type: int = FrameType.SPEECH,
    fresh: bool = True,
    rgb: bytes | None = b"",
    pts: int = 0,
) -> STVVideoFrame:
    """A video frame. ``fresh=False`` marks a held/synthesized tick (no server frame)."""
    pixels = rgb if rgb != b"" else bytes(width * height * 3)
    return STVVideoFrame(
        rgb=pixels,
        source_bytes=b"\xff\xd8jpeg" if fresh else b"",
        width=width,
        height=height,
        frame_type=frame_type,
        pts=pts,
    )


def make_audio_frame(
    ms: int = 40,
    sample_rate: int = 24000,
    *,
    num_channels: int = 1,
    silent: bool = False,
    pts: int = 0,
) -> STVAudioFrame:
    """An audio tick. ``silent=True`` is the SDK's synthesized zero fill."""
    samples = int(sample_rate * ms / 1000) * num_channels
    pcm = bytes(samples * _BYTES_PER_SAMPLE) if silent else b"\x11\x22" * samples
    return STVAudioFrame(pcm=pcm, sample_rate=sample_rate, num_channels=num_channels, pts=pts)


class FakeSTVClient:
    """Records the plugin's calls; its sink is driven by hand from tests."""

    def __init__(self, output: Any = None, **kwargs: Any) -> None:
        self.output = output
        self.kwargs = kwargs
        self.started = False
        self.close_calls = 0
        self.turns = 0
        self.interrupts = 0
        self.sent: list[tuple[bytes, int, int]] = []

        # scriptable behaviour
        self.interrupt_result = True
        self.interrupt_raises: Exception | None = None
        self.start_turn_raises: Exception | None = None
        self.start_turn_gate: asyncio.Event | None = None
        self.close_gate: asyncio.Event | None = None

        self._listeners: dict[STVEvent, list[Callable[..., Any]]] = {}
        self.handler_errors: list[TypeError] = []

    # --- the surface the plugin uses ---

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.close_calls += 1

    async def start_turn(self) -> None:
        if self.start_turn_gate is not None:
            await self.start_turn_gate.wait()
        if self.start_turn_raises is not None:
            raise self.start_turn_raises
        self.turns += 1

    async def send_tts_audio(self, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        self.sent.append((pcm, sample_rate, num_channels))

    async def interrupt(self) -> bool:
        self.interrupts += 1
        if self.interrupt_raises is not None:
            raise self.interrupt_raises
        return self.interrupt_result

    def add_listener(self, event: STVEvent, cb: Callable[..., Any]) -> None:
        self._listeners.setdefault(event, []).append(cb)

    # --- test drivers ---

    async def emit(self, event: STVEvent, **kwargs: Any) -> None:
        """Dispatch like the real emitter: pass kwargs, swallow (but record) TypeError."""
        for cb in list(self._listeners.get(event, [])):
            try:
                result = cb(**kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except TypeError as e:
                self.handler_errors.append(e)

    async def emit_session_ready(self) -> None:
        await self.emit(STVEvent.SESSION_READY, session_data={"trace_id": "t"})

    async def emit_connect_error(self, message: str = "connect failed") -> None:
        """Connect-exhaustion shape: message + fatal, no code."""
        await self.emit(STVEvent.ERROR, message=message, fatal=True)

    async def emit_server_fatal(
        self, message: str = "server error", code: str = "BACKEND_UNAVAILABLE"
    ) -> None:
        """Server-fatal shape: message + code + fatal. Production's fatal path."""
        await self.emit(STVEvent.ERROR, message=message, code=code, fatal=True)

    async def emit_closed(self) -> None:
        await self.emit(STVEvent.CLOSED)

    async def emit_stopped_speaking(self) -> None:
        await self.emit(STVEvent.BOT_STOPPED_SPEAKING)

    async def push_video(self, frame: STVVideoFrame) -> None:
        await self.output.write_video(frame)

    async def push_audio(self, frame: STVAudioFrame) -> None:
        await self.output.write_audio(frame)

    async def push_tick(
        self, *, silent: bool = False, fresh: bool = True, frame_type: int = FrameType.SPEECH
    ) -> None:
        """One tick in the SDK's order: video first, then audio."""
        await self.push_video(make_video_frame(frame_type=frame_type, fresh=fresh))
        await self.push_audio(make_audio_frame(silent=silent))

    async def play_turn(self, *, idle_ticks: int = 2, speech_ticks: int = 3) -> None:
        """Idle fill, then real speech, then the drain edge."""
        for _ in range(idle_ticks):
            await self.push_tick(silent=True, frame_type=FrameType.IDLE)
        for i in range(speech_ticks):
            await self.push_tick(
                frame_type=FrameType.START_OF_SPEECH if i == 0 else FrameType.SPEECH
            )
        await self.emit_stopped_speaking()

    async def play_underrun(self, *, before: int = 2, after: int = 2) -> None:
        """Speech, a mid-turn drain (TTS too slow), a refill, then the real end."""
        for _ in range(before):
            await self.push_tick()
        await self.emit_stopped_speaking()  # spurious: the turn is not over
        for _ in range(after):
            await self.push_tick()
        await self.emit_stopped_speaking()
