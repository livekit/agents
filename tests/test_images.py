"""Image preparation keeps asyncio control serviceable while native encoders run."""

import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents.utils import images

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_format", [False, True])
async def test_image_encoding_keeps_the_event_loop_serviceable(provider_format: bool) -> None:
    frame = rtc.VideoFrame(1920, 1080, rtc.VideoBufferType.RGBA, bytes([127]) * (1920 * 1080 * 4))
    observations: list[float] = []
    running = True

    async def heartbeat() -> None:
        while running:
            started = time.perf_counter()
            await asyncio.sleep(0.001)
            observations.append(max(0.0, time.perf_counter() - started - 0.001))

    task = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0)
        if provider_format:
            context = llm.ChatContext.empty()
            context.add_message(role="user", content=[llm.ImageContent(image=frame)])
            messages, _ = await context.to_provider_format(format="openai")
            assert messages[0]["content"][0]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        else:
            encoded = await images.encode(frame, images.EncodeOptions())
            assert encoded[:3] == b"\xff\xd8\xff"
        assert observations, "encoding must allow the loop to run"
    finally:
        running = False
        await task


@pytest.mark.asyncio
async def test_image_cache_reencodes_when_inference_size_changes() -> None:
    frame = rtc.VideoFrame(8, 8, rtc.VideoBufferType.RGBA, bytes([127]) * (8 * 8 * 4))
    image = llm.ImageContent(image=frame)
    first = await llm.utils.serialize_image(image)
    image.inference_width = image.inference_height = 4
    resized = await llm.utils.serialize_image(image)
    assert first.data_bytes != resized.data_bytes
