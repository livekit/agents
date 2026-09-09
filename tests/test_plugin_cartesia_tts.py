from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from unittest.mock import MagicMock

import aiohttp
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


# --- pooled websocket: stale frames/done from previous contexts must be ignored ---


class _FakeMsg:
    def __init__(self, type: aiohttp.WSMsgType, data=None, extra=None) -> None:
        self.type = type
        self.data = data
        self.extra = extra

    @classmethod
    def text(cls, payload: dict) -> _FakeMsg:
        return cls(aiohttp.WSMsgType.TEXT, json.dumps(payload))


class _FakeWS:
    """Yield stale frames/done first, then the current context's frames/done."""

    def __init__(self) -> None:
        self._sent: list[dict] = []
        self._context_id: str | None = None
        self._end_pkt_sent = asyncio.Event()
        self._built = False
        self._queue: deque[_FakeMsg] = deque()

    async def send_str(self, data: str) -> None:
        pkt = json.loads(data)
        self._sent.append(pkt)
        if self._context_id is None and pkt.get("context_id"):
            self._context_id = pkt["context_id"]
        if pkt.get("continue") is False:
            self._end_pkt_sent.set()

    async def receive(self, timeout: float | None = None) -> _FakeMsg:
        if not self._built:
            self._built = True
            stale = "stale-context-id"
            # stale leftovers from the previous context arrive first
            self._queue.append(_FakeMsg.text({"context_id": stale, "data": _b64(b"\x00" * 160)}))
            self._queue.append(_FakeMsg.text({"context_id": stale, "done": True}))

        if self._queue:
            return self._queue.popleft()

        # current frames; delay done until end_pkt so the tokenizer is closed
        ctx = self._context_id or ""
        self._queue.append(_FakeMsg.text({"context_id": ctx, "data": _b64(b"\x01" * 160)}))
        await self._end_pkt_sent.wait()
        self._queue.append(_FakeMsg.text({"context_id": ctx, "done": True}))
        return self._queue.popleft()


class _FakePool:
    def __init__(self, ws: _FakeWS) -> None:
        self._ws = ws
        self.last_acquire_time = 0.0
        self.last_connection_reused = False

    def connection(self, timeout: float):  # noqa: ANN201
        ws = self._ws

        class _Ctx:
            async def __aenter__(self_):  # noqa: N805
                return ws

            async def __aexit__(self_, *exc):  # noqa: N805
                return False

        return _Ctx()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.mark.asyncio
async def test_synthesize_skips_stale_frames_from_previous_context():
    """Only the current context's audio is played when stale frames are pooled."""
    from livekit.agents import APIConnectOptions
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    tts._pool = _FakePool(_FakeWS())  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("hello world")
    stream.end_input()

    frames: list[bytes] = []
    async for ev in stream:
        frames.append(bytes(ev.frame.data))

    # stale frame (b"\x00") must be dropped; only current audio (b"\x01") is played
    assert frames, "no audio frames were synthesized"
    assert all(f == b"\x01" * 160 for f in frames)
