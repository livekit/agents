"""Cartesia TTS plugin tests (API key redaction + aligned-transcript capability)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIConnectionError, APIStatusError, LanguageCode
from livekit.plugins.cartesia import tts as cartesia_tts

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


@pytest.mark.asyncio
async def test_update_options_invalidates_the_pool_when_api_version_changes():
    """cartesia_version rides on the websocket URL, so pooled sockets go stale."""
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    invalidated = 0
    original = tts._pool.invalidate

    def counting_invalidate() -> None:
        nonlocal invalidated
        invalidated += 1
        original()

    tts._pool.invalidate = counting_invalidate  # type: ignore[method-assign]

    # a no-op update must not throw away warm connections
    tts.update_options(api_version=tts._opts.api_version)
    assert invalidated == 0, "Expected no invalidation when api_version is unchanged."

    tts.update_options(api_version="2099-01-01")
    assert tts._opts.api_version == "2099-01-01"
    assert invalidated == 1, "Changing api_version must drop pooled connections."

    # unrelated options ride in the per-request body and need no reconnect
    tts.update_options(speed=1.2)
    assert invalidated == 1, "Only connection-bound options should invalidate the pool."

    await tts.aclose()


@pytest.mark.asyncio
async def test_ws_url_carries_the_updated_api_version():
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    tts.update_options(api_version="2099-01-01")

    captured: dict[str, str] = {}

    class _Session:
        def ws_connect(self_, url, **kwargs):  # noqa: ANN001, N805
            captured["url"] = str(url)
            raise ConnectionError("stop here, the URL is what matters")

    tts._ensure_session = lambda: _Session()  # type: ignore[method-assign]

    with pytest.raises(APIConnectionError):
        await tts._connect_ws(timeout=1.0)

    assert "cartesia_version=2099-01-01" in captured["url"], (
        f"Expected the new api_version in the handshake URL, got {captured['url']}"
    )

    await tts.aclose()


@pytest.mark.asyncio
async def test_rest_request_sends_the_configured_api_version_header():
    """The header and the body must agree; the body is shaped by _opts.api_version."""
    from livekit.plugins.cartesia import TTS
    from livekit.plugins.cartesia.tts import API_VERSION_HEADER

    tts = TTS(api_key=SECRET_API_KEY, api_version="2099-01-01")
    captured: dict[str, object] = {}

    class _Resp:
        async def __aenter__(self_):  # noqa: N805
            raise ConnectionError("stop here, the headers are what matter")

        async def __aexit__(self_, *exc):  # noqa: N805
            return False

    class _Session:
        def post(self_, url, *, headers, **kwargs):  # noqa: ANN001, N805
            captured["headers"] = headers
            return _Resp()

    tts._ensure_session = lambda: _Session()  # type: ignore[method-assign]

    stream = tts.synthesize("hello")
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass
    await stream.aclose()

    headers = captured.get("headers")
    assert headers is not None, "The REST path never issued a request."
    assert headers[API_VERSION_HEADER] == "2099-01-01", (
        f"Header sent {headers[API_VERSION_HEADER]!r} while the body was built for "
        f"{tts._opts.api_version!r}"
    )

    await tts.aclose()


def test_aligned_transcript_auto_enabled_for_supported_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en")
    assert tts.capabilities.aligned_transcript is True


def test_aligned_transcript_auto_disabled_for_unsupported_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja")
    assert tts.capabilities.aligned_transcript is False


def test_aligned_transcript_disabled_when_word_timestamps_off() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=False)
    assert tts.capabilities.aligned_transcript is False


def test_aligned_transcript_auto_enabled_for_preview_model_any_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", model="sonic-preview", language="ja")
    assert tts.capabilities.aligned_transcript is True


def test_explicit_word_timestamps_true_skips_gate_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
        tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)

    assert tts.capabilities.aligned_transcript is True
    assert any("word_timestamps was requested" in record.message for record in caplog.records)


def test_update_options_narrows_capability_when_language_becomes_unsupported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en")
    assert tts.capabilities.aligned_transcript is True

    tts.update_options(language="ja")
    assert tts.capabilities.aligned_transcript is False


def test_update_options_restores_capability_when_language_becomes_supported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja")
    assert tts.capabilities.aligned_transcript is False

    tts.update_options(language="fr")
    assert tts.capabilities.aligned_transcript is True


def test_update_options_to_preview_model_enables_capability() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", model="sonic-3", language="ja")
    assert tts.capabilities.aligned_transcript is False

    tts.update_options(model="sonic-preview")
    assert tts.capabilities.aligned_transcript is True


def test_update_options_keeps_explicit_word_timestamps_true() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    tts.update_options(language="ja")
    assert tts.capabilities.aligned_transcript is True

    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is True


def test_add_timestamps_omitted_when_auto_unsupported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja")
    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is False


def test_add_timestamps_requested_when_supported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en")
    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is True


def test_supports_word_timestamps_helper() -> None:
    assert cartesia_tts._supports_word_timestamps("sonic-3", LanguageCode("en")) is True
    assert cartesia_tts._supports_word_timestamps("sonic-3", LanguageCode("ja")) is False
    assert cartesia_tts._supports_word_timestamps("sonic-preview", LanguageCode("ja")) is True
    assert cartesia_tts._supports_word_timestamps("sonic-3", None) is True
