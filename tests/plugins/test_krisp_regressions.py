"""Krisp VIVA noise filter: building the native filter never stalls the job's event loop."""

from __future__ import annotations

import asyncio
import gc
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.room_io import _AudioStream
from livekit.plugins import krisp


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/7924 reproduces: the native filter is "
    "built inside _process on the first frame, on the event loop",
)
async def test_the_first_filtered_frame_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Room audio input keeps the event loop responsive while the native filter is built.

    https://github.com/livekit/agents/issues/7924
    """

    class SlowNativeFilter:
        """The native filter; constructing it blocks as the real one does (200-800 ms)."""

        def __init__(self, settings: object) -> None:
            time.sleep(0.3)

        def update_stream_info(self, info: object) -> None:
            pass

        def process(self, in_buf: object, out_buf: object) -> None:
            pass

        def close_session(self) -> None:
            pass

    class Track:
        """The SDK audio stream of a subscribed track: a 10 ms frame per read."""

        async def __anext__(self) -> rtc.AudioFrameEvent:
            return rtc.AudioFrameEvent(frame=rtc.AudioFrame(b"\0\0" * 480, 48_000, 1, 480))

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("livekit.plugins.krisp_internal.plugin.KrispVivaFilter", SlowNativeFilter)
    processor = krisp.voice_isolation()
    processor._on_credentials_updated(token="jwt", url="wss://cloud.example")
    track = Track()
    # Track stands in for the SDK's audio stream; no room is involved
    stream = _AudioStream(lambda: track, processor, None)  # type: ignore[arg-type, return-value]
    ticks: list[float] = []

    async def tick() -> None:
        while True:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    try:
        await asyncio.sleep(0.05)
        for _ in range(5):
            await asyncio.wait_for(stream.__anext__(), 2)
        await asyncio.sleep(0.05)
    finally:
        ticker.cancel()
        await stream.aclose()

    stall = max(b - a for a, b in zip(ticks, ticks[1:], strict=False))
    assert stall < 0.1, f"event loop blocked for {stall:.3f}s"


async def test_license_mode_never_injects_silence_between_real_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Frames that are not a multiple of Krisp's 10 ms chunk come out as the same audio, in order."""
    # Catches: zero-padding each output frame to its input size, which put silence between
    # real audio (the "cutting in and out" artifact) whenever frame and chunk sizes differ.

    class IdentitySession:
        """A Krisp NC session that returns its chunk unchanged."""

        def process(self, chunk: np.ndarray, level: int) -> np.ndarray:
            return chunk

    rates = SimpleNamespace(
        **{f"Sr{rate}Hz": rate for rate in (8000, 16000, 24000, 32000, 44100, 48000)}
    )
    durations = SimpleNamespace(**{f"Fd{ms}ms": ms for ms in (10, 15, 20, 30, 32)})
    krisp_audio = SimpleNamespace(
        SamplingRate=rates,
        FrameDuration=durations,
        LogLevel=SimpleNamespace(Off=0),
        NcInt16=SimpleNamespace(create=lambda config: IdentitySession()),
        ModelInfo=SimpleNamespace,
        NcSessionConfig=SimpleNamespace,
        globalInit=lambda *args: None,
        globalDestroy=lambda: None,
        getVersion=lambda: SimpleNamespace(major=1, minor=0, patch=0),
    )
    monkeypatch.setitem(sys.modules, "krisp_audio", krisp_audio)
    model = tmp_path / "model.kef"
    model.write_bytes(b"")
    processor = krisp.voice_isolation(
        auth_provider=krisp.auth.krisp_license(license_key="key", model_path=str(model))
    )

    # a ramp from 1, so any zero in the output is injected silence
    sizes = [100, 137, 53, 200, 80, 45, 300, 10, 90, 160] * 4
    ramp = np.arange(1, sum(sizes) + 1, dtype=np.int16)
    out = []
    offset = 0
    for size in sizes:
        frame = rtc.AudioFrame(ramp[offset : offset + size].tobytes(), 16_000, 1, size)
        offset += size
        out.append(np.frombuffer(processor.process(frame).data, dtype=np.int16))
    processor.close()
    del processor
    gc.collect()

    played = np.concatenate(out)
    assert played.size > ramp.size - 160
    assert np.array_equal(played, ramp[: played.size])
