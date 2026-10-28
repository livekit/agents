from __future__ import annotations

import asyncio

import pytest

from livekit.agents.inference import VAD

from . import utils

pytestmark = pytest.mark.plugin("silero")


@pytest.mark.parametrize("sample_rate", [16000, 44100])
@pytest.mark.parametrize("chunk_duration_ms", [10, None])
async def test_speech_segments_survive_input_reframing(
    sample_rate: int, chunk_duration_ms: int | None
) -> None:
    frames, *_ = await utils.make_test_speech(
        sample_rate=sample_rate, chunk_duration_ms=chunk_duration_ms
    )
    detector = VAD(min_speech_duration=0.5, min_silence_duration=0.75)
    async with detector.stream() as stream:
        for frame in frames:
            stream.push_frame(frame)
        stream.end_input()
        events = [event async for event in stream]

    starts = [event for event in events if event.type == "start_of_speech"]
    ends = [event for event in events if event.type == "end_of_speech"]
    assert starts
    assert len(starts) == len(ends)
    assert all(event.frames for event in ends)


async def test_flush_allows_another_complete_speech_segment() -> None:
    frames, *_ = await utils.make_test_speech(chunk_duration_ms=10, sample_rate=16000)
    detector = VAD(min_speech_duration=0.5, min_silence_duration=0.75)
    async with detector.stream() as stream:
        for _ in range(2):

            async def push() -> None:
                for frame in frames:
                    stream.push_frame(frame)
                    await asyncio.sleep(0)

            task = asyncio.create_task(push())
            try:
                started = False
                async with asyncio.timeout(30):
                    async for event in stream:
                        if event.type == "start_of_speech":
                            started = True
                        elif event.type == "end_of_speech":
                            assert started
                            assert event.frames
                            break
                    else:
                        pytest.fail("stream ended before the speech segment completed")
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            stream.flush()
