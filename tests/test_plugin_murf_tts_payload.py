"""Murf TTS request payloads, focusing on voice controls that are legal at zero."""

from __future__ import annotations

import base64
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions
from livekit.plugins.murf import TTS
from livekit.plugins.murf.tts import _to_murf_websocket_pkt, _TTSOptions

pytestmark = pytest.mark.unit

_AUDIO = b"\x00\x01" * 2400


def _opts(**kwargs: Any) -> _TTSOptions:
    """Build the options a real ``TTS`` instance would hand to the packet builder."""
    return TTS(api_key="test-key", **kwargs)._opts


def test_zero_speed_and_pitch_reach_the_websocket_packet() -> None:
    """``speed``/``pitch`` are documented as -50..50, so 0 is an explicit request."""
    voice_config = _to_murf_websocket_pkt(_opts(speed=0, pitch=0))["voice_config"]

    assert voice_config["rate"] == 0
    assert voice_config["pitch"] == 0


def test_unset_speed_and_pitch_stay_out_of_the_websocket_packet() -> None:
    """``None`` is the documented way to ask for the provider's default."""
    voice_config = _to_murf_websocket_pkt(_opts())["voice_config"]

    assert "rate" not in voice_config
    assert "pitch" not in voice_config


@pytest.mark.parametrize(("speed", "pitch"), [(-50, -50), (-1, 1), (20, 45), (50, 50)])
def test_non_zero_controls_are_unchanged(speed: int, pitch: int) -> None:
    voice_config = _to_murf_websocket_pkt(_opts(speed=speed, pitch=pitch))["voice_config"]

    assert voice_config["rate"] == speed
    assert voice_config["pitch"] == pitch


async def test_both_transports_send_the_same_controls() -> None:
    """``synthesize()`` (HTTP) and ``stream()`` (WebSocket) must agree on rate/pitch."""
    captured: dict[str, Any] = {}

    async def _http_tts(request: web.Request) -> web.Response:
        captured["http"] = await request.json()
        return web.Response(body=_AUDIO, content_type="audio/pcm")

    async def _ws_input(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.TEXT:
                continue

            pkt = json.loads(msg.data)
            captured.setdefault("ws", []).append(pkt)
            if pkt.get("end"):
                audio = base64.b64encode(_AUDIO).decode()
                await ws.send_str(json.dumps({"context_id": pkt["context_id"], "audio": audio}))
                await ws.send_str(json.dumps({"context_id": pkt["context_id"], "final": True}))
        return ws

    app = web.Application()
    app.router.add_post("/v1/speech/stream", _http_tts)
    app.router.add_get("/v1/speech/stream-input", _ws_input)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    conn_options = APIConnectOptions(max_retry=0, timeout=5)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(
                api_key="test-key",
                speed=0,
                pitch=0,
                base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
                http_session=session,
            )
            try:
                stream = tts.stream(conn_options=conn_options)
                stream.push_text("hello there")
                stream.end_input()
                async for _ in stream:
                    pass

                chunk = tts.synthesize("hello there", conn_options=conn_options)
                async for _ in chunk:
                    pass
            finally:
                await tts.aclose()
    finally:
        await runner.cleanup()

    ws_voice_config = captured["ws"][0]["voice_config"]
    assert ws_voice_config["rate"] == captured["http"]["rate"] == 0
    assert ws_voice_config["pitch"] == captured["http"]["pitch"] == 0
