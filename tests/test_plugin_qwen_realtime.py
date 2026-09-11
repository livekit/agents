"""Tests for the shared realtime transport of the Qwen plugin."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import MagicMock

import aiohttp
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


def test_public_api_is_exported() -> None:
    from livekit.plugins import qwen

    for name in (
        "STT",
        "SpeechStream",
        "TTS",
        "SynthesizeStream",
        "QwenRegion",
        "STTModels",
        "TTSModels",
        "TTSVoices",
        "TTSLanguageTypes",
        "LLM",
        "LLMModels",
        "DEFAULT_REGION",
        "DEFAULT_STT_MODEL",
        "DEFAULT_TTS_MODEL",
        "DEFAULT_TTS_VOICE",
        "DEFAULT_TTS_LANGUAGE_TYPE",
        "DEFAULT_LLM_MODEL",
        "__version__",
    ):
        assert name in qwen.__all__, name
        assert hasattr(qwen, name), name


# --- review follow-ups (livekit/agents#7224) ----------------------------------------------


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "livekit.plugins.qwen" and r.levelno == logging.WARNING
    ]


def test_plaintext_ws_to_a_remote_host_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    # The API key rides in an Authorization header and the audio is unencrypted, so
    # plaintext to anything off-host is worth a warning. It stays allowed: local proxies
    # are a legitimate use.
    from livekit.plugins.qwen._utils import resolve_realtime_url

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.qwen"):
        url = resolve_realtime_url("ws://asr.internal.example:8080/api-ws/v1/realtime", "intl")

    assert url == "ws://asr.internal.example:8080/api-ws/v1/realtime"
    assert len(_warnings(caplog)) == 1
    assert "plaintext" in _warnings(caplog)[0].getMessage()


def test_plaintext_ws_to_loopback_is_not_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.plugins.qwen._utils import resolve_realtime_url

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.qwen"):
        resolve_realtime_url("ws://127.0.0.1:9/api-ws/v1/realtime", "intl")
        resolve_realtime_url("ws://localhost:9/api-ws/v1/realtime", "intl")
    assert _warnings(caplog) == []


def test_wss_and_the_region_defaults_are_not_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.plugins.qwen._utils import resolve_realtime_url

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.qwen"):
        resolve_realtime_url(
            "wss://ws-1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime", "intl"
        )
        resolve_realtime_url(None, "intl")
        resolve_realtime_url(None, "cn")
    assert _warnings(caplog) == []


class _FakeWS:
    """Stand-in for aiohttp's ClientWebSocketResponse with a controllable send."""

    closed = False

    def __init__(self, *, stall_send: bool = False, stall_close: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.close_calls = 0
        self._stall_send = stall_send
        self._stall_close = stall_close

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self._stall_send:
            await asyncio.sleep(30)
        self.sent.append(payload)

    async def receive(self) -> aiohttp.WSMessage:
        # Answers the finish handshake with a close, so `_finish_handshake` completes.
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, None, None)

    async def close(self) -> bool:
        self.close_calls += 1
        if self._stall_close:
            await asyncio.sleep(30)
        return True


async def test_close_with_finish_sends_finish_then_closes() -> None:
    from livekit.plugins.qwen._realtime import RealtimeSocket

    ws = _FakeWS()
    await RealtimeSocket(ws).close_with_finish()  # type: ignore[arg-type]

    assert [p["type"] for p in ws.sent] == ["session.finish"]
    assert ws.close_calls == 1


async def test_close_with_finish_is_bounded_against_a_stalled_peer() -> None:
    # The voice pipeline awaits this on a barge-in before clearing the playout buffer, so
    # a peer that stops reading must not be able to hold it open.
    from livekit.plugins.qwen._realtime import RealtimeSocket

    ws = _FakeWS(stall_send=True)
    started = time.monotonic()
    await RealtimeSocket(ws).close_with_finish(timeout=0.2)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started

    # Budget is 0.2 s here and the stub would stall 30 s; 2 s tolerates a loaded runner.
    assert elapsed < 2.0, f"close_with_finish blocked for {elapsed:.2f}s"
    assert ws.close_calls == 1, "the close must still be attempted after the finish stalls"


async def test_close_gracefully_bounds_the_close_handshake_too() -> None:
    # The finish wait was already bounded; the aiohttp close handshake that follows it
    # was not, and its default lets a stalled peer add up to 10 s to session teardown.
    from livekit.plugins.qwen._realtime import RealtimeSocket

    ws = _FakeWS(stall_close=True)
    started = time.monotonic()
    await RealtimeSocket(ws).close_gracefully(timeout=0.2)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"close_gracefully blocked for {elapsed:.2f}s"
    assert [p["type"] for p in ws.sent] == ["session.finish"]
    assert ws.close_calls == 1
