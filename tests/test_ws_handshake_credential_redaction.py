"""A rejected websocket upgrade must not put the API key in the exception.

`WSServerHandshakeError` subclasses `ClientResponseError`, so it is neither a
`ClientConnectorError` nor an `asyncio.TimeoutError`. Plugins that caught only those two let a
401/403 upgrade escape carrying `RequestInfo`, whose headers hold the credential (#6739, #7031).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIStatusError

SECRET_API_KEY = "secret-api-key-do-not-log"


def _handshake_error(host: str, header: str, api_key: str) -> WSServerHandshakeError:
    url = URL(f"wss://{host}/listen")
    headers = CIMultiDict({"Host": host, header: api_key})
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


def _session_raising(err: Exception) -> MagicMock:
    async def _raise(*_args, **_kwargs):
        raise err

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    return session


def _assert_redacted(err: APIStatusError) -> None:
    assert err.status_code == 401
    assert SECRET_API_KEY not in repr(err)
    assert SECRET_API_KEY not in str(err)
    assert err.__cause__ is None


@pytest.mark.plugin("deepgram")
@pytest.mark.asyncio
async def test_deepgram_stt_redacts_api_key_from_handshake_error():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key=SECRET_API_KEY)
    stream = stt.stream()
    stream._session = _session_raising(
        _handshake_error("api.deepgram.com", "Authorization", f"Token {SECRET_API_KEY}")
    )

    with pytest.raises(APIStatusError) as exc_info:
        await stream._connect_ws()

    _assert_redacted(exc_info.value)


@pytest.mark.plugin("xai")
@pytest.mark.asyncio
async def test_xai_tts_redacts_api_key_from_handshake_error():
    from livekit.plugins.xai import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    tts._session = _session_raising(
        _handshake_error("api.x.ai", "Authorization", f"Bearer {SECRET_API_KEY}")
    )

    with pytest.raises(APIStatusError) as exc_info:
        await tts._connect_ws(timeout=5.0, opts=tts._opts)

    _assert_redacted(exc_info.value)
