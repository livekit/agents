"""Tests for Inworld TTS plugin configuration options and wire payloads."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

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
