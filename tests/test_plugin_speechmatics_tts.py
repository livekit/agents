from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from livekit.agents import APIConnectOptions, APIError
from livekit.plugins.speechmatics import tts as speechmatics_tts

pytestmark = pytest.mark.unit


async def test_incomplete_pcm_sample_is_reported_as_provider_failure() -> None:
    async def chunks():
        yield b"\x00\x01" * 1600
        yield b"\xff"

    async def raise_for_status() -> None:
        pass

    @asynccontextmanager
    async def post(*args, **kwargs):
        yield SimpleNamespace(raise_for_status=raise_for_status, iter_bytes=chunks)

    model = speechmatics_tts.TTS(api_key="test", http_session=SimpleNamespace(post=post))
    with pytest.raises(APIError, match="incomplete PCM sample"):
        async with model.synthesize("hello", conn_options=APIConnectOptions(max_retry=0)) as stream:
            async for _ in stream:
                pass
