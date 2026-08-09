from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIConnectionError, APIStatusError

pytestmark = pytest.mark.plugin("cartesia")

SECRET_API_KEY = "cartesia-secret-api-key-do-not-log"


def _handshake_error_with_api_key(api_key: str) -> WSServerHandshakeError:
    url = URL("wss://api.cartesia.ai/tts/websocket")
    headers = CIMultiDict(
        {
            "Host": "api.cartesia.ai",
            "User-Agent": "LiveKit Agents Cartesia Plugin/test",
            "X-API-Key": api_key,
        }
    )
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


@pytest.mark.asyncio
async def test_connect_ws_redacts_api_key_from_handshake_error():
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)

    async def _raise(*_args, **_kwargs):
        raise _handshake_error_with_api_key(SECRET_API_KEY)

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    tts._session = session

    with pytest.raises(APIStatusError) as exc_info:
        await tts._connect_ws(timeout=5.0)

    err = exc_info.value
    assert err.status_code == 401
    assert SECRET_API_KEY not in repr(err)
    assert err.__cause__ is None


@pytest.mark.asyncio
async def test_connect_ws_generic_error_does_not_chain_cause():
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    leaky = ConnectionError(f"wss://api.cartesia.ai/tts/websocket?api_key={SECRET_API_KEY}")

    async def _raise(*_args, **_kwargs):
        raise leaky

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    tts._session = session

    with pytest.raises(APIConnectionError) as exc_info:
        await tts._connect_ws(timeout=5.0)

    err = exc_info.value
    assert err.__cause__ is None
    assert str(err) == "ConnectionError"
    assert SECRET_API_KEY not in repr(err)
    assert SECRET_API_KEY not in str(err)
