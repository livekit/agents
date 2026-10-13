from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.slng import stt as slng_stt

pytestmark = pytest.mark.plugin("slng")


@pytest.mark.parametrize("end_time", [0.4, 0.0, None])
async def test_endpoint_preserves_word_timing(monkeypatch, end_time):
    messages = asyncio.Queue()
    message = {"type": "final_transcript", "transcript": "hello"}
    if end_time is not None:
        message["words"] = [{"start": 0.1, "end": end_time}]
    messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(message), ""))
    ws = SimpleNamespace(
        receive=messages.get,
        send_str=AsyncMock(),
        send_bytes=AsyncMock(),
        close=AsyncMock(),
        closed=False,
    )
    monkeypatch.setattr(slng_stt.SpeechStream, "_connect_ws", AsyncMock(return_value=ws))
    provider = slng_stt.STT(
        api_key="test-key", model="deepgram/nova:3", http_session=SimpleNamespace()
    )
    stream = provider.stream()
    stream.start_time_offset = 7.0
    try:
        stream.push_frame(
            rtc.AudioFrame(
                data=bytes(3200), sample_rate=16000, num_channels=1, samples_per_channel=1600
            )
        )
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.END_OF_SPEECH:
                    if end_time:
                        assert stream.start_time_offset > 6.0
                        assert event.alternatives[0].end_time == pytest.approx(
                            stream.start_time_offset + 0.4
                        )
                    else:
                        assert event.alternatives == []
                    break
            else:
                pytest.fail("missing endpoint")
    finally:
        await stream.aclose()
        await provider.aclose()
