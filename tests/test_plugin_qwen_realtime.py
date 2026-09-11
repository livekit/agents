"""Tests for the shared realtime transport of the Qwen plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIStatusError

pytestmark = pytest.mark.unit

SECRET_API_KEY = "sk-secret-api-key-do-not-log"


def test_invalid_request_error_is_not_retryable() -> None:
    from livekit.plugins.qwen._realtime import status_error_from

    err = status_error_from(
        {
            "event_id": "srv_e",
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "unsupported language"},
        }
    )
    assert isinstance(err, APIStatusError)
    assert err.status_code == 400
    assert err.retryable is False
    assert err.request_id == "srv_e"
    assert "unsupported language" in str(err)


def test_server_error_is_retryable() -> None:
    from livekit.plugins.qwen._realtime import status_error_from

    err = status_error_from(
        {"event_id": "srv_e", "type": "error", "error": {"type": "server_error", "message": "busy"}}
    )
    assert err.retryable is True
    assert err.status_code == -1


def test_error_event_without_message_still_produces_an_error() -> None:
    from livekit.plugins.qwen._realtime import status_error_from

    err = status_error_from({"event_id": "srv_e", "type": "error", "error": {}})
    assert err.message
    assert err.body == {}


def _handshake_error(host: str, api_key: str) -> WSServerHandshakeError:
    url = URL(f"wss://{host}/api-ws/v1/realtime")
    headers = CIMultiDict({"Host": host, "Authorization": f"Bearer {api_key}"})
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


async def test_rejected_handshake_does_not_leak_the_api_key() -> None:
    from livekit.plugins.qwen._realtime import connect

    async def _raise(*_args: object, **_kwargs: object) -> None:
        raise _handshake_error("dashscope-intl.aliyuncs.com", SECRET_API_KEY)

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)

    with pytest.raises(APIStatusError) as exc_info:
        await connect(
            session,
            base_url="wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
            model="qwen3-asr-flash-realtime",
            api_key=SECRET_API_KEY,
            timeout=1.0,
        )

    err = exc_info.value
    assert err.status_code == 401
    assert err.retryable is False
    assert SECRET_API_KEY not in str(err)
    assert SECRET_API_KEY not in repr(err)
    assert err.__cause__ is None
