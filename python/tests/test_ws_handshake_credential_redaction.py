"""A rejected websocket upgrade must not put the API key in the exception.

aiohttp's handshake error carries the request's headers, which hold the credential (#6739,
#7031). ``livekit.agents.http`` raises ``APIStatusError`` with a redacted cause instead, so no
plugin can leak it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from aiohttp import web

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIStatusError, http

pytestmark = pytest.mark.unit

SECRET_API_KEY = "secret-api-key-do-not-log"


@pytest.fixture
async def rejecting_server() -> AsyncIterator[str]:
    """Answers every upgrade with 401."""

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401, text="Unauthorized")

    app = web.Application()
    app.router.add_get("/{path:.*}", reject)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _assert_redacted(err: BaseException) -> None:
    assert isinstance(err, APIStatusError) and err.status_code == 401
    chain: BaseException | None = err
    while chain is not None:
        assert SECRET_API_KEY not in repr(chain)
        assert SECRET_API_KEY not in str(chain)
        chain = chain.__cause__


async def test_the_client_redacts_credentials_from_a_rejected_handshake(
    rejecting_server: str,
) -> None:
    async with http.open() as session:
        with pytest.raises(APIStatusError) as exc_info:
            await session.ws_connect(
                f"{rejecting_server}/listen",
                headers={"Authorization": f"Token {SECRET_API_KEY}"},
            )
    _assert_redacted(exc_info.value)


async def test_deepgram_stt_redacts_api_key_from_handshake_error(rejecting_server: str) -> None:
    from livekit.plugins.deepgram import STT

    async with http.open() as session:
        stt = STT(
            api_key=SECRET_API_KEY, base_url=f"{rejecting_server}/listen", http_session=session
        )
        stream = stt.stream()
        with pytest.raises(APIStatusError) as exc_info:
            await stream._connect_ws()
        await stream.aclose()
    _assert_redacted(exc_info.value)


async def test_xai_tts_redacts_api_key_from_handshake_error(
    rejecting_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.plugins.xai import TTS, tts as xai_tts

    monkeypatch.setattr(xai_tts, "XAI_WEBSOCKET_URL", f"{rejecting_server}/tts")
    async with http.open():
        tts = TTS(api_key=SECRET_API_KEY)
        with pytest.raises(APIStatusError) as exc_info:
            await tts._connect_ws(DEFAULT_API_CONNECT_OPTIONS, opts=tts._opts)
    _assert_redacted(exc_info.value)
