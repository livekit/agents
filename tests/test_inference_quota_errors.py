from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIConnectOptions, APIStatusError, inference
from livekit.agents.inference._utils import create_api_error_from_response

from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit

_MESSAGE = (
    "Realtime (sts) credit quota exhausted. Wait for the next billing cycle or upgrade your plan."
)


@pytest.mark.parametrize(
    ("status", "headers", "retryable", "message"),
    [
        (429, {}, True, "Invalid response status"),
        (429, {"x-should-retry": "false"}, False, "Invalid response status"),
        (401, {}, False, "Invalid response status"),
        (
            429,
            {"x-livekit-inference-quota-message": "STT concurrent connection limit reached."},
            True,
            "STT concurrent connection limit reached.",
        ),
    ],
)
def test_handshake_error_preserves_retry_policy(
    status: int, headers: dict[str, str], retryable: bool, message: str
) -> None:
    url = URL("https://example.livekit.cloud/v1/stt")
    error = aiohttp.WSServerHandshakeError(
        request_info=aiohttp.RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url),
        history=(),
        status=status,
        message="Invalid response status",
        headers=CIMultiDict(headers),
    )
    result = create_api_error_from_response(error)

    assert message in result.message
    assert result.status_code == status
    assert result.retryable is retryable
    assert result.body is None


@pytest.fixture
async def quota_gateway() -> AsyncIterator[str]:
    async def reject(request: web.Request) -> web.Response:
        return web.json_response(
            {"hint": _MESSAGE, "category": "MaxGatewayCredits"},
            status=429,
            headers={
                "X-LiveKit-Inference-Quota-Category": "MaxGatewayCredits",
                "X-LiveKit-Inference-Quota-Message": _MESSAGE,
                "X-LiveKit-Request-Id": "quota-test-request",
                "X-Should-Retry": "false",
            },
        )

    app = web.Application()
    app.router.add_get("/{path:.*}", reject)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = runner.addresses[0][1]
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("service", ["gpt-live", "stt", "tts"])
async def test_failed_handshake_explains_exhausted_credits(
    quota_gateway: str, service: str
) -> None:
    async with aiohttp.ClientSession() as http_session:
        options = {
            "api_key": "test-key",
            "api_secret": "test-secret-long-enough-for-hs256-signing",
            "base_url": quota_gateway,
            "http_session": http_session,
        }
        with pytest.raises(APIStatusError) as caught:
            if service == "tts":
                model = inference.TTS(model="cartesia/sonic-3", **options)
                try:
                    await model._connect_ws(timeout=1.0)
                finally:
                    await model.aclose()
            elif service == "stt":
                stt_model = inference.STT(model="deepgram/nova-3", vad=FakeVAD(), **options)
                stream = stt_model.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
                try:
                    stream.end_input()
                    async for _ in stream:
                        pass
                finally:
                    await stream.aclose()
                    await stt_model.aclose()
            else:
                duplex = inference.DuplexModel(
                    model="openai/gpt-live-1",
                    conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
                    **options,
                )
                session = duplex.session()
                errors: asyncio.Queue = asyncio.Queue()
                session.on("error", errors.put_nowait)
                try:
                    event = await asyncio.wait_for(errors.get(), timeout=2.0)
                    raise event.error
                finally:
                    await session.aclose()
                    await duplex.aclose()

        assert _MESSAGE in str(caught.value)
        assert caught.value.status_code == 429
        assert caught.value.retryable is False
        assert caught.value.request_id == "quota-test-request"
        assert caught.value.body == {"category": "MaxGatewayCredits"}
