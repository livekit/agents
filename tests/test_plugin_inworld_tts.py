"""Tests for Inworld TTS plugin configuration options and wire payloads."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIConnectionError
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import aio

pytestmark = pytest.mark.plugin("inworld")


async def test_delivery_mode_default_not_given():
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key", model="inworld-tts-2")
    assert tts._opts.delivery_mode is NOT_GIVEN


async def test_delivery_mode_set_on_init():
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key", model="inworld-tts-2", delivery_mode="STABLE")
    assert tts._opts.delivery_mode == "STABLE"


async def test_delivery_mode_accepts_all_documented_values():
    from livekit.plugins.inworld import TTS

    for value in ("DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", "CREATIVE"):
        tts = TTS(api_key="test-key", model="inworld-tts-2", delivery_mode=value)
        assert tts._opts.delivery_mode == value


async def test_delivery_mode_rejects_unknown_value():
    from livekit.plugins.inworld import TTS

    with pytest.raises(ValueError):
        TTS(api_key="test-key", model="inworld-tts-2", delivery_mode="EXPRESSIVE")


async def test_delivery_mode_update_options():
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key", model="inworld-tts-2")
    assert tts._opts.delivery_mode is NOT_GIVEN
    tts.update_options(delivery_mode="CREATIVE")
    assert tts._opts.delivery_mode == "CREATIVE"

    with pytest.raises(ValueError):
        tts.update_options(delivery_mode="LOUD")


async def _capture_first_ws_create_packet(opts) -> dict:
    """Drive `_InworldConnection._send_loop` against a fake websocket and
    return the first `create` packet it sends, as a parsed dict.

    The fake `send_str` signals an `asyncio.Event` once it has captured a
    payload, so the test wakes immediately rather than polling.
    """
    from livekit.plugins.inworld.tts import _CreateContextMsg, _InworldConnection

    sent_payloads: list[str] = []
    captured = asyncio.Event()

    def _on_send(payload: str) -> None:
        sent_payloads.append(payload)
        captured.set()

    fake_ws = MagicMock()
    fake_ws.send_str = AsyncMock(side_effect=_on_send)

    conn = _InworldConnection(
        session=MagicMock(),
        ws_url="wss://example.invalid/",
        authorization="Basic test",
    )
    # Skip real connect(); inject the fake websocket directly so `_send_loop`
    # reads from it as if a connection had been established.
    conn._ws = fake_ws

    await conn._outbound_queue.put(_CreateContextMsg(context_id="ctx-1", opts=opts))

    send_task = asyncio.create_task(conn._send_loop())
    try:
        await asyncio.wait_for(captured.wait(), timeout=2.0)
    finally:
        conn._closed = True
        await aio.cancel_and_wait(send_task)

    return json.loads(sent_payloads[0])


async def test_ws_create_packet_includes_delivery_mode():
    """The WebSocket `create` packet sent by `_send_loop` includes
    `deliveryMode` at the top level of the `create` object when
    `delivery_mode` is set on the TTS."""
    from livekit.plugins.inworld.tts import _TTSOptions

    opts = _TTSOptions(
        model="inworld-tts-2",
        encoding="PCM",
        voice="Ashley",
        sample_rate=24000,
        bit_rate=64000,
        speaking_rate=1.0,
        temperature=1.0,
        delivery_mode="STABLE",
    )

    pkt = await _capture_first_ws_create_packet(opts)
    assert pkt["create"]["deliveryMode"] == "STABLE"
    assert pkt["create"]["modelId"] == "inworld-tts-2"


async def test_ws_create_packet_omits_delivery_mode_when_not_given():
    """When `delivery_mode` is not set, the WS `create` packet must not
    include the `deliveryMode` key at all (Inworld treats absence as the
    server default)."""
    from livekit.plugins.inworld.tts import _TTSOptions

    opts = _TTSOptions(
        model="inworld-tts-2",
        encoding="PCM",
        voice="Ashley",
        sample_rate=24000,
        bit_rate=64000,
        speaking_rate=1.0,
        temperature=1.0,
    )

    pkt = await _capture_first_ws_create_packet(opts)
    assert "deliveryMode" not in pkt["create"]


def _patch_session_to_capture_post(tts, captured: dict[str, object]) -> None:
    """Replace the TTS's aiohttp session with a stub that captures the body of
    the first ``post()`` call and short-circuits the rest of the request."""

    class _FakePostCM:
        async def __aenter__(self):
            # Raise after capture so ChunkedStream._run unwinds quickly; the
            # body has already been recorded by _fake_post.
            raise RuntimeError("short-circuit")

        async def __aexit__(self, *exc):
            return None

    def _fake_post(url, *, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _FakePostCM()

    fake_session = MagicMock()
    fake_session.post = _fake_post
    tts._session = fake_session


async def test_http_body_includes_delivery_mode():
    """The HTTP `synthesize` request body includes `deliveryMode` at the
    top level when `delivery_mode` is set on the TTS."""
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key", model="inworld-tts-2", delivery_mode="BALANCED")
    captured: dict[str, object] = {}
    _patch_session_to_capture_post(tts, captured)

    with pytest.raises(APIConnectionError):
        async for _ in tts.synthesize("hello"):
            pass

    body = captured.get("json")
    assert isinstance(body, dict), f"expected JSON dict, got {body!r}"
    assert body["deliveryMode"] == "BALANCED"
    assert body["modelId"] == "inworld-tts-2"


async def test_http_body_omits_delivery_mode_when_not_given():
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key", model="inworld-tts-2")
    captured: dict[str, object] = {}
    _patch_session_to_capture_post(tts, captured)

    with pytest.raises(APIConnectionError):
        async for _ in tts.synthesize("hello"):
            pass

    body = captured.get("json")
    assert isinstance(body, dict)
    assert "deliveryMode" not in body


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.close_calls = 0
        self._dropped = asyncio.Event()

    async def send_str(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def receive(self, timeout: float | None = None):
        await self._dropped.wait()
        return MagicMock(type=aiohttp.WSMsgType.CLOSED)

    async def close(self) -> None:
        self.close_calls += 1
        self._dropped.set()

    def drop(self) -> None:
        self._dropped.set()


def _tts_with_ws_connect(ws_connect):
    from livekit.plugins.inworld import TTS

    session = MagicMock()
    session.ws_connect = ws_connect
    return TTS(api_key="test-key", http_session=session)


async def test_contexts_share_one_connection_and_queue_until_connected():
    ws = _FakeWS()
    ws_connect = AsyncMock(return_value=ws)
    tts = _tts_with_ws_connect(ws_connect)

    conn = tts._get_connection()
    ids = [conn.create_context(MagicMock(), tts._opts)[0] for _ in range(3)]
    assert tts._get_connection() is conn

    await asyncio.gather(*(conn.connect(5.0) for _ in range(3)))
    for _ in range(50):
        if len(ws.sent) == 3:
            break
        await asyncio.sleep(0.01)

    assert ws_connect.await_count == 1
    assert [p["contextId"] for p in ws.sent] == ids
    await tts.aclose()


async def test_failed_handshake_fails_all_waiters_and_reconnects():
    tts = _tts_with_ws_connect(AsyncMock(side_effect=OSError("refused")))

    conn = tts._get_connection()
    waiters = [conn.create_context(MagicMock(), tts._opts)[1] for _ in range(2)]
    await conn.connect(5.0)

    for waiter in waiters:
        with pytest.raises(APIConnectionError):
            await waiter
    assert tts._get_connection() is not conn
    await tts.aclose()


async def test_handshake_timeout_fails_waiters():
    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()

    tts = _tts_with_ws_connect(_hang)
    conn = tts._get_connection()
    conn.HANDSHAKE_TIMEOUT = 0.05
    _, waiter = conn.create_context(MagicMock(), tts._opts)
    await conn.connect(5.0)

    with pytest.raises(APIConnectionError):
        await waiter
    await tts.aclose()


async def test_cancelled_caller_does_not_cancel_shared_handshake():
    ws = _FakeWS()
    release = asyncio.Event()
    calls = 0

    async def _slow_connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        await release.wait()
        return ws

    tts = _tts_with_ws_connect(_slow_connect)
    conn = tts._get_connection()

    first = asyncio.create_task(conn.connect(5.0))
    await asyncio.sleep(0)
    await aio.cancel_and_wait(first)

    release.set()
    await conn.connect(5.0)
    assert calls == 1
    assert conn._ws is ws
    assert not conn._closed
    await tts.aclose()


async def test_ws_create_packet_auto_mode_strategy():
    from livekit.plugins.inworld.tts import _TTSOptions

    opts = _TTSOptions(
        model="inworld-tts-2",
        encoding="PCM",
        voice="Ashley",
        sample_rate=24000,
        bit_rate=64000,
        speaking_rate=1.0,
        temperature=1.0,
    )
    pkt = await _capture_first_ws_create_packet(opts)
    assert pkt["create"]["autoMode"] is True
    assert "autoModeStrategy" not in pkt["create"]
    assert pkt["create"]["bufferCharThreshold"] == 250

    opts.auto_mode_strategy = "SENTENCE_BOUNDARY"
    pkt = await _capture_first_ws_create_packet(opts)
    assert pkt["create"]["autoModeStrategy"] == "SENTENCE_BOUNDARY"
    assert "bufferCharThreshold" not in pkt["create"]

    opts.buffer_char_threshold = 400
    pkt = await _capture_first_ws_create_packet(opts)
    assert pkt["create"]["bufferCharThreshold"] == 400


@pytest.mark.parametrize(
    ("model", "options", "expected"),
    [
        ("inworld-tts-2", {}, "SENTENCE_BOUNDARY"),
        ("inworld-tts-2", {"retain_format": True}, "SENTENCE_BOUNDARY"),
        ("inworld-tts-2", {"tokenizer": "blingfire"}, "CLIENT_SEGMENTED"),
        ("inworld-tts-2", {"retain_format": False}, "CLIENT_SEGMENTED"),
        ("inworld-tts-1.5-max", {}, "CLIENT_SEGMENTED"),
    ],
)
async def test_stream_auto_mode_strategy(model, options, expected):
    from livekit.agents import tokenize
    from livekit.plugins.inworld import TTS

    kwargs = dict(options)
    if kwargs.get("tokenizer") == "blingfire":
        kwargs["tokenizer"] = tokenize.blingfire.SentenceTokenizer()
    tts = TTS(api_key="test-key", model=model, **kwargs)
    stream = tts.stream()
    try:
        assert stream._opts.auto_mode_strategy == expected
    finally:
        await stream.aclose()


async def test_sentence_boundary_stream_forwards_raw_text():
    """With SENTENCE_BOUNDARY, text reaches the connection as pushed, without
    client-side sentence splitting, and the end of input flushes the context."""
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key", model="inworld-tts-2")
    calls: list[tuple[str, str | None]] = []
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    conn = MagicMock()
    conn.send_text.side_effect = lambda ctx, text: calls.append(("text", text))
    conn.flush_context.side_effect = lambda ctx: calls.append(("flush", None))

    def _close(ctx: str, *, cancel: bool = False) -> None:
        calls.append(("close", None))
        if not waiter.done():
            waiter.set_result(None)

    conn.close_context.side_effect = _close
    conn.create_context.return_value = ("ctx-1", waiter)
    conn.connect = AsyncMock()
    tts._get_connection = MagicMock(return_value=conn)

    stream = tts.stream()
    for token in ["Hello ", "there. How", " are", " you?"]:
        stream.push_text(token)
    stream.end_input()
    try:
        await asyncio.wait_for(waiter, timeout=2.0)
        sent = list(calls)
    finally:
        await stream.aclose()

    assert sent == [
        ("text", "Hello "),
        ("text", "there. How"),
        ("text", " are"),
        ("text", " you?"),
        ("flush", None),
        ("close", None),
    ]


async def _sent_close_packet(cancel: bool) -> dict:
    from livekit.plugins.inworld.tts import _InworldConnection

    sent = asyncio.Event()
    payloads: list[str] = []

    def _on_send(payload: str) -> None:
        payloads.append(payload)
        sent.set()

    conn = _InworldConnection(
        session=MagicMock(), ws_url="wss://example.invalid/", authorization="Basic test"
    )
    conn._ws = MagicMock()
    conn._ws.send_str = AsyncMock(side_effect=_on_send)
    conn.close_context("ctx-1", cancel=cancel)

    send_task = asyncio.create_task(conn._send_loop())
    try:
        await asyncio.wait_for(sent.wait(), timeout=2.0)
    finally:
        conn._closed = True
        await aio.cancel_and_wait(send_task)

    return json.loads(payloads[0])


async def test_ws_close_and_cancel_packets():
    assert await _sent_close_packet(cancel=False) == {"close_context": {}, "contextId": "ctx-1"}
    assert await _sent_close_packet(cancel=True) == {"cancel_context": {}, "contextId": "ctx-1"}


class _FakeConnection:
    def __init__(self, waiter: asyncio.Future[None]) -> None:
        self.waiter = waiter
        self.calls: list[tuple[str, bool]] = []
        self.closed = asyncio.Event()

    def create_context(self, emitter, opts) -> tuple[str, asyncio.Future[None]]:
        return "ctx-1", self.waiter

    async def connect(self, timeout: float) -> None:
        pass

    def send_text(self, context_id: str, text: str) -> None:
        self.calls.append(("send_text", False))

    def flush_context(self, context_id: str) -> None:
        self.calls.append(("flush_context", False))

    def close_context(self, context_id: str, *, cancel: bool = False) -> None:
        self.calls.append(("close_context", cancel))
        self.closed.set()


def _tts_with_fake_connection():
    from livekit.plugins.inworld import TTS

    tts = TTS(api_key="test-key")
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    conn = _FakeConnection(waiter)
    tts._get_connection = MagicMock(return_value=conn)  # type: ignore[method-assign]
    return tts, conn


async def test_stream_aclose_cancels_context():
    tts, conn = _tts_with_fake_connection()

    stream = tts.stream()
    stream.push_text("Hello there. How are you today?")
    stream.flush()
    await asyncio.sleep(0.1)
    await stream.aclose()

    assert ("close_context", False) not in conn.calls
    assert conn.calls[-1] == ("close_context", True)


async def test_stream_aclose_after_input_ended_cancels_context():
    tts, conn = _tts_with_fake_connection()

    stream = tts.stream()
    stream.push_text("Hello there.")
    stream.end_input()
    await asyncio.wait_for(conn.closed.wait(), timeout=2.0)
    assert conn.calls == [("send_text", False), ("flush_context", False), ("close_context", False)]

    await stream.aclose()

    assert conn.calls[-1] == ("close_context", True)


async def test_caller_timeout_does_not_fail_shared_handshake():
    ws = _FakeWS()
    calls = 0

    async def _slow_connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.2)
        return ws

    tts = _tts_with_ws_connect(_slow_connect)
    conn = tts._get_connection()
    conn.create_context(MagicMock(), tts._opts)
    _, long_waiter = conn.create_context(MagicMock(), tts._opts)

    with pytest.raises(asyncio.TimeoutError):
        await conn.connect(0.05)
    await conn.connect(5.0)

    assert calls == 1
    assert conn._ws is ws
    assert not long_waiter.done()
    await tts.aclose()


async def test_dropped_connection_is_closed_and_replaced():
    ws = _FakeWS()
    tts = _tts_with_ws_connect(AsyncMock(return_value=ws))
    conn = tts._get_connection()
    _, waiter = conn.create_context(MagicMock(), tts._opts)
    await conn.connect(5.0)

    ws.drop()
    with pytest.raises(APIConnectionError):
        await asyncio.wait_for(waiter, timeout=2.0)
    for _ in range(50):
        if ws.close_calls and conn._send_task.done() and conn._cleanup_task.done():
            break
        await asyncio.sleep(0.01)

    assert ws.close_calls >= 1
    assert conn._send_task.done() and conn._cleanup_task.done()
    assert tts._get_connection() is not conn
    await tts.aclose()


async def test_timed_out_stream_waiter_is_not_failed_later():
    from livekit.agents import APIConnectOptions, APITimeoutError

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()

    tts = _tts_with_ws_connect(_hang)
    conn = tts._get_connection()
    conn.HANDSHAKE_TIMEOUT = 0.2
    waiters: list[asyncio.Future[None]] = []
    create_context = conn.create_context

    def _record(emitter, opts):
        ctx_id, waiter = create_context(emitter, opts)
        waiters.append(waiter)
        return ctx_id, waiter

    conn.create_context = _record  # type: ignore[method-assign]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05))
    stream.push_text("Hello there.")
    stream.end_input()
    with pytest.raises(APITimeoutError):
        async for _ in stream:
            pass
    await stream.aclose()

    await asyncio.sleep(0.3)
    assert conn._closed
    assert waiters[0].cancelled()
    await tts.aclose()
