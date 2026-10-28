"""Speech adapters reject malformed provider transcripts before delivering them."""

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider", ["deepgram", "elevenlabs"])
async def test_batch_recognition_rejects_nontext_transcript(provider: str) -> None:
    from livekit.plugins import deepgram, elevenlabs

    async def recognize(request: web.Request) -> web.Response:
        await request.read()
        if provider == "deepgram":
            return web.json_response(
                {
                    "metadata": {"request_id": "request"},
                    "results": {
                        "channels": [
                            {"alternatives": [{"transcript": 42, "confidence": 0.9, "words": []}]}
                        ]
                    },
                }
            )
        return web.json_response({"text": 42, "language_code": "en", "words": []})

    app = web.Application()
    app.router.add_post("/{path:.*}", recognize)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        if provider == "deepgram":
            model = deepgram.STT(
                api_key="test", base_url=str(server.make_url("/recognize")), http_session=session
            )
        else:
            model = elevenlabs.STT(
                api_key="test", base_url=str(server.make_url("/")), http_session=session
            )
        try:
            frame = rtc.AudioFrame.create(
                sample_rate=16000, num_channels=1, samples_per_channel=160
            )
            with pytest.raises(APIConnectionError) as failure:
                await model.recognize(frame, conn_options=APIConnectOptions(max_retry=0))
            assert failure.value.__cause__ is not None
        finally:
            await model.aclose()
