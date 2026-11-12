"""Tests for AudioEmitter: is_final, timed transcripts, duration tracking, segments."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from livekit.agents import tts, utils

pytestmark = pytest.mark.unit


SR = 24000
NC = 1


async def _run_emitter(
    produce_fn, *, stream: bool = False, expect_finals: int = 1, mime_type: str = "audio/pcm"
):
    """Run emitter, collect events until we've seen expect_finals is_final events."""
    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test", dst_ch=dst_ch)
    emitter.initialize(
        request_id="req1",
        sample_rate=SR,
        num_channels=NC,
        mime_type=mime_type,
        stream=stream,
    )

    events: list[tts.SynthesizedAudio] = []
    finals_seen = 0

    async def collect():
        nonlocal finals_seen
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                finals_seen += 1
                if finals_seen >= expect_finals:
                    return

    ct = asyncio.create_task(collect())
    produce_fn(emitter)
    await emitter.join()
    await ct
    return emitter, events


@pytest.mark.asyncio
async def test_compressed_finalization_delivers_the_complete_tail_before_join_returns():
    encoded = Path(__file__).with_name("change-sophie.opus").read_bytes()
    decoder = utils.media.AudioStreamDecoder(sample_rate=SR, num_channels=NC, format="audio/ogg")
    decoder.push(encoded)
    decoder.end_input()
    expected = b"".join([bytes(frame.data) async for frame in decoder])
    await decoder.aclose()

    def produce(emitter):
        for offset in range(0, len(encoded), 997):
            emitter.push(encoded[offset : offset + 997])
        emitter.end_input()

    emitter, events = await asyncio.wait_for(
        _run_emitter(produce, mime_type="audio/ogg"), timeout=30
    )
    try:
        assert events[-1].is_final
        assert b"".join(bytes(event.frame.data) for event in events) == expected
    finally:
        await emitter.aclose()


@pytest.mark.asyncio
async def test_a_corrupt_container_fails_the_synthesis_at_join():
    # Catches: a container that fails to decode natively swallowed, so the synthesis ends with
    # no audio and no error for the fallback chain to act on.
    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="corrupt", dst_ch=dst_ch)
    emitter.initialize(request_id="r", sample_rate=SR, num_channels=NC, mime_type="audio/ogg")
    try:
        emitter.push(b"\x5a" * 4096)
        emitter.end_input()
        with pytest.raises(RuntimeError, match="decode failed"):
            await asyncio.wait_for(emitter.join(), timeout=10)
        with pytest.raises(RuntimeError, match="decode failed"):
            emitter.flush()
    finally:
        await emitter.aclose()
