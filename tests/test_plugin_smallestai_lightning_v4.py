from __future__ import annotations

import json
import struct
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIError

pytestmark = pytest.mark.plugin("smallestai")


def _pcm(num_samples: int) -> bytes:
    """Bare little-endian int16 PCM, which is what `output_format=pcm` returns."""
    return struct.pack(f"<{num_samples}h", *(((i * 97) % 2000) - 1000 for i in range(num_samples)))


class _WSServer:
    """Local `/lightning-v4/live` stand-in speaking the real turn protocol."""

    def __init__(
        self,
        *,
        frames: list[bytes] | None = None,
        ready_error: dict[str, Any] | None = None,
        turn_error: dict[str, Any] | None = None,
        negotiated_sample_rate: int | None = None,
    ) -> None:
        self._frames = frames if frames is not None else [_pcm(480)]
        self._ready_error = ready_error
        self._turn_error = turn_error
        self._negotiated_sample_rate = negotiated_sample_rate
        self.received: list[dict[str, Any]] = []
        self.query: dict[str, str] = {}
        self.sessions = 0

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.query = dict(request.query)
        self.sessions += 1

        if self._ready_error is not None:
            await ws.send_json({"status": "error", "error": self._ready_error})
            await ws.close()
            return ws

        await ws.send_json({"event": "ready", "session_id": "test-session"})
        await ws.send_json(
            {
                "event": "ready",
                "sample_rate": self._negotiated_sample_rate
                or int(request.query.get("sample_rate", 48000)),
            }
        )

        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.TEXT:
                continue
            message = json.loads(msg.data)
            self.received.append(message)

            if message["event"] == "speak":
                turn_id = message["turn_id"]
                if self._turn_error is not None:
                    await ws.send_json({"event": "error", "turn_id": turn_id, **self._turn_error})
                    continue
                await ws.send_json({"event": "turn_start", "turn_id": turn_id})
                for frame in self._frames:
                    await ws.send_bytes(frame)
                await ws.send_json(
                    {"event": "turn_end", "turn_id": turn_id, "chunks": len(self._frames)}
                )
            elif message["event"] == "interrupt":
                turn_id = self.received[-2]["turn_id"] if len(self.received) >= 2 else "unknown"
                await ws.send_json({"event": "interrupted", "turn_id": turn_id, "chunks": 0})
            elif message["event"] == "end":
                await ws.close()

        return ws

    async def __aenter__(self) -> _WSServer:
        app = web.Application()
        app.router.add_get("/lightning-v4/live", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"ws://127.0.0.1:{port}/lightning-v4/live"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()

    def of_event(self, event: str) -> list[dict[str, Any]]:
        return [m for m in self.received if m.get("event") == event]


async def _stream_turn(tts, text: str, **conn: Any) -> bytes:
    """Push one turn's text through a stream and drain the resulting audio.

    One stream is one segment is one Lightning v4 turn.
    """
    options = APIConnectOptions(max_retry=0, timeout=5, **conn)
    stream = tts.stream(conn_options=options)
    stream.push_text(text)
    stream.end_input()

    audio = bytearray()
    try:
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
    finally:
        await stream.aclose()
    return bytes(audio)


def test_requires_api_key():
    from livekit.plugins.smallestai import LightningV4TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            LightningV4TTS()


def test_api_key_from_env():
    from livekit.plugins.smallestai import LightningV4TTS

    with patch.dict("os.environ", {"SMALLEST_API_KEY": "env-key"}):
        assert LightningV4TTS()._api_key == "env-key"


def test_defaults_to_brannock_voice():
    from livekit.plugins.smallestai import LightningV4TTS

    tts = LightningV4TTS(api_key="test-key")
    assert tts._opts.voice_id == "brannock"


def test_rejects_bad_sample_rate():
    from livekit.plugins.smallestai import LightningV4TTS

    with pytest.raises(ValueError, match="sample_rate"):
        LightningV4TTS(api_key="test-key", sample_rate=11025)


@pytest.mark.parametrize("speed", [0.4, 2.1])
def test_construction_rejects_out_of_range_speed(speed):
    from livekit.plugins.smallestai import LightningV4TTS

    with pytest.raises(ValueError, match="speed"):
        LightningV4TTS(api_key="test-key", speed=speed)


@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_construction_accepts_boundary_speeds(speed):
    from livekit.plugins.smallestai import LightningV4TTS

    assert LightningV4TTS(api_key="test-key", speed=speed)._opts.speed == speed


def test_update_options_rejects_bad_speed():
    from livekit.plugins.smallestai import LightningV4TTS

    tts = LightningV4TTS(api_key="test-key")
    with pytest.raises(ValueError, match="speed"):
        tts.update_options(speed=3.0)
    # Rejected update must not have taken effect.
    assert tts._opts.speed == 1.0


def test_ws_url_includes_connect_params():
    from livekit.plugins.smallestai import LightningV4TTS

    tts = LightningV4TTS(api_key="test-key", voice_id="rhodes", speed=1.2, language="fr")
    url = tts._ws_url()

    assert url.startswith("wss://api.smallest.ai/waves/v1/lightning-v4/live?")
    assert "voice_id=rhodes" in url
    assert "language=auto" in url  # French isn't English, so it resolves to auto.
    assert "output_format=pcm" in url
    assert "speed=1.2" in url


def test_synthesize_raises_not_implemented():
    from livekit.plugins.smallestai import LightningV4TTS

    tts = LightningV4TTS(api_key="test-key")
    with pytest.raises(NotImplementedError):
        tts.synthesize("hello")


@pytest.mark.asyncio
async def test_streaming_turn_round_trip():
    from livekit.plugins.smallestai import LightningV4TTS

    frames = [_pcm(480), _pcm(480)]
    async with _WSServer(frames=frames) as server:
        tts = LightningV4TTS(
            api_key="test-key", base_url=server.base_url, http_session=server.session
        )
        try:
            audio = await _stream_turn(tts, "hello there")
        finally:
            await tts.aclose()

        assert audio == b"".join(frames)
        speaks = server.of_event("speak")
        assert len(speaks) == 1
        assert speaks[0]["text"] == "hello there"
        assert "turn_id" in speaks[0]


@pytest.mark.asyncio
async def test_pooled_connection_is_reused_across_turns():
    from livekit.plugins.smallestai import LightningV4TTS

    async with _WSServer() as server:
        tts = LightningV4TTS(
            api_key="test-key", base_url=server.base_url, http_session=server.session
        )
        try:
            await _stream_turn(tts, "first turn")
            await _stream_turn(tts, "second turn")
        finally:
            await tts.aclose()

        # Two turns, one session: the same socket carries both, which is what keeps
        # Lightning v4's server-held conversational context intact across turns.
        assert server.sessions == 1
        assert len(server.of_event("speak")) == 2


@pytest.mark.asyncio
async def test_ready_rejection_raises_api_error():
    from livekit.plugins.smallestai import LightningV4TTS

    async with _WSServer(
        ready_error={"code": "model_access_denied", "message": "beta access required"}
    ) as server:
        tts = LightningV4TTS(
            api_key="test-key", base_url=server.base_url, http_session=server.session
        )
        try:
            with pytest.raises(APIError, match="model_access_denied"):
                await _stream_turn(tts, "hello")
        finally:
            await tts.aclose()


@pytest.mark.asyncio
async def test_turn_error_raises_api_error():
    from livekit.plugins.smallestai import LightningV4TTS

    async with _WSServer(turn_error={"code": "bad_frame", "message": "malformed turn"}) as server:
        tts = LightningV4TTS(
            api_key="test-key", base_url=server.base_url, http_session=server.session
        )
        try:
            with pytest.raises(APIError, match="bad_frame"):
                await _stream_turn(tts, "hello")
        finally:
            await tts.aclose()


@pytest.mark.asyncio
async def test_negotiated_sample_rate_overrides_requested():
    from livekit.plugins.smallestai import LightningV4TTS

    async with _WSServer(negotiated_sample_rate=24000) as server:
        tts = LightningV4TTS(
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
            sample_rate=48000,
        )
        try:
            await _stream_turn(tts, "hello")
        finally:
            await tts.aclose()

        assert tts._opts.sample_rate == 24000


@pytest.mark.asyncio
async def test_barge_in_sends_interrupt_and_keeps_the_session():
    from livekit.plugins.smallestai import LightningV4TTS

    # Enough frames that the stream is still receiving when we cancel it.
    async with _WSServer(frames=[_pcm(480)] * 50) as server:
        tts = LightningV4TTS(
            api_key="test-key", base_url=server.base_url, http_session=server.session
        )
        try:
            options = APIConnectOptions(max_retry=0, timeout=5)
            stream = tts.stream(conn_options=options)
            stream.push_text("a turn that will be interrupted")
            stream.end_input()

            # Let the turn actually start before barging in.
            async for _ in stream:
                break
            # aclose() cancels the run task and awaits it, so by the time this
            # returns the interrupt has been sent and its `interrupted` reply drained.
            await stream.aclose()

            assert len(server.of_event("interrupt")) == 1

            # A second turn on the same TTS instance must still work — the barge-in
            # must not have poisoned the pooled connection.
            audio = await _stream_turn(tts, "a second turn")
            assert audio == _pcm(480) * 50
        finally:
            await tts.aclose()

        # One session for both turns: barge-in kept the connection (and Lightning
        # v4's server-held context) alive instead of forcing a reconnect.
        assert server.sessions == 1
