"""Integration tests for the VoxCPM2 / vLLM-Omni TTS plugin."""

from __future__ import annotations

import aiohttp
import pytest

pytestmark = pytest.mark.plugin("voxcpm")


@pytest.mark.asyncio
async def test_integration_live_server():
    from livekit.plugins.voxcpm import TTS

    base_url = "http://127.0.0.1:8800/v1"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url.rstrip('/v1')}/v1/models",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status != 200:
                    pytest.skip("vLLM-Omni server not reachable on :8800")
    except Exception:
        pytest.skip("vLLM-Omni server not reachable on :8800")

    async with aiohttp.ClientSession() as session:
        tts = TTS(base_url=base_url, http_session=session)
        frame = await tts.synthesize(
            "Merhaba, bu bir LiveKit plugin entegrasyon testidir."
        ).collect()
        assert len(frame.data.tobytes()) > 1000
