"""Bland TTS: the HTTP error envelope and the WebSocket turn protocol."""

from __future__ import annotations

import json
import struct

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIStatusError, http
from livekit.plugins.bland import TTS
from livekit.plugins.bland.tts import DEFAULT_VOICE_ID
from tests.fakes import until

NO_RETRY = APIConnectOptions(max_retry=0)


def _pcm(num_samples: int) -> bytes:
    """Bare little-endian int16 PCM, which is what ``container: raw`` returns."""
    return struct.pack(f"<{num_samples}h", *(((i * 97) % 2000) - 1000 for i in range(num_samples)))


class _BlandPeer:
    """``/v2/tts/ws`` speaking Bland's turn protocol: ready, utterance_start, audio, utterance_end."""

    def __init__(self, frames: list[bytes], *, stale_end: bool = False) -> None:
        self.frames = frames
        self.stale_end = stale_end  # replay the previous turn's utterance_end mid-turn
        self.received: list[dict[str, object]] = []
        self.headers: dict[str, str] = {}

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.headers = dict(request.headers)
        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.TEXT:
                continue
            message = json.loads(msg.data)
            self.received.append(message)
            if message["type"] == "init":
                rate = message["audio"]["sample_rate"]
                await ws.send_json(
                    {
                        "type": "ready",
                        "session_id": "s",
                        "encoding": "pcm_s16le",
                        "sample_rate": rate,
                    }
                )
            elif message["type"] == "end_of_turn":
                ended = [m["context_id"] for m in self.of_type("end_of_turn")]
                for i, frame in enumerate(self.frames):
                    await ws.send_bytes(frame)
                    if i == 0 and self.stale_end and len(ended) > 1:
                        await ws.send_json(
                            {"type": "utterance_end", "context_id": ended[-2], "reason": "complete"}
                        )
                await ws.send_json(
                    {
                        "type": "utterance_end",
                        "context_id": message["context_id"],
                        "reason": "complete",
                        "frames": len(self.frames),
                        "duration_ms": 40 * len(self.frames),
                    }
                )
            elif message["type"] == "cancel":
                await ws.send_json(
                    {
                        "type": "utterance_end",
                        "context_id": message["context_id"],
                        "reason": "cancelled",
                    }
                )
            elif message["type"] == "close":
                await ws.send_json({"type": "done", "session_id": "s"})
                await ws.close()
        return ws

    def of_type(self, kind: str) -> list[dict[str, object]]:
        return [m for m in self.received if m["type"] == kind]


async def _say(tts: TTS, text: str) -> bytes:
    """One turn: a stream is one segment, which is one Bland turn."""
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text(text)
    stream.end_input()
    audio = b"".join([ev.frame.data.tobytes() async for ev in stream])
    await stream.aclose()
    return audio


async def _speak(peer: _BlandPeer, text: str, *, api_key: str = "k") -> bytes:
    app = web.Application()
    app.router.add_get("/v2/tts/ws", peer.handle)
    async with TestServer(app) as server, http.open():
        tts = TTS(api_key=api_key, base_url=str(server.make_url("/v2")))
        audio = await _say(tts, text)
        await tts.aclose()
    return audio


async def test_error_envelope_becomes_api_status_error() -> None:
    """A Bland error envelope surfaces its code and message. https://github.com/livekit/agents/pull/6554"""

    async def not_found(request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"code": "voice_not_found", "message": "Voice was not found."}}, status=404
        )

    app = web.Application()
    app.router.add_post("/v2/tts", not_found)
    async with TestServer(app) as server, http.open():
        tts = TTS(api_key="k", base_url=str(server.make_url("/v2")))

        with pytest.raises(APIStatusError) as raised:
            async with tts.synthesize("hello world", conn_options=NO_RETRY) as stream:
                async for _ in stream:
                    pass

    assert raised.value.status_code == 404
    assert "voice_not_found" in str(raised.value)
    assert "Voice was not found." in str(raised.value)


async def test_init_sent_once_with_voice_format_and_bearer_key() -> None:
    """A turn opens one session announcing the voice and PCM format. https://github.com/livekit/agents/pull/6752"""
    peer = _BlandPeer([_pcm(480)])

    await _speak(peer, "hello world", api_key="test-key")

    [init] = peer.of_type("init")
    assert init["voice"] == DEFAULT_VOICE_ID
    assert init["audio"] == {"encoding": "pcm_s16le", "sample_rate": 48000}
    assert "controls" not in init
    assert peer.headers["Authorization"] == "Bearer test-key"


async def test_binary_frames_reach_the_pipeline_unchanged() -> None:
    """Every PCM sample Bland sends is played, in order. https://github.com/livekit/agents/pull/6752"""
    frames = [_pcm(480), _pcm(480), _pcm(240)]

    audio = await _speak(_BlandPeer(frames), "hello")

    expected = b"".join(frames)
    assert audio[: len(expected)] == expected
    assert set(audio[len(expected) :]) <= {0}  # the final frame is zero-padded


async def test_an_abandoned_turn_is_cancelled_and_its_connection_reused() -> None:
    """A barge-in cancels the turn on the open socket; the next turn speaks on that same socket.

    https://github.com/livekit/agents/pull/6752
    """
    # Catches: an interrupted turn left uncancelled, or its socket dropped so every barge-in
    # pays a new connection.
    peer = _BlandPeer([_pcm(480)])
    app = web.Application()
    app.router.add_get("/v2/tts/ws", peer.handle)
    async with TestServer(app) as server, http.open():
        tts = TTS(api_key="k", base_url=str(server.make_url("/v2")))
        abandoned = tts.stream(conn_options=NO_RETRY)
        abandoned.push_text("hello there")
        abandoned.flush()
        await until(lambda: peer.of_type("speak"), "the first turn is spoken")
        await abandoned.aclose()
        audio = await _say(tts, "again")
        await tts.aclose()

    [first, _] = [m["context_id"] for m in peer.of_type("speak")]
    assert peer.of_type("cancel") == [{"type": "cancel", "context_id": first}]
    assert len(peer.of_type("init")) == 1
    assert audio.startswith(_pcm(480))


async def test_a_stale_utterance_end_does_not_end_the_next_turn() -> None:
    """An utterance_end for an earlier turn is not this turn's boundary.

    https://github.com/livekit/agents/pull/6752
    """
    # Catches: ending the current turn on any utterance_end, which cut its audio short.
    frames = [_pcm(480), _pcm(480), _pcm(240)]
    peer = _BlandPeer(frames, stale_end=True)
    app = web.Application()
    app.router.add_get("/v2/tts/ws", peer.handle)
    async with TestServer(app) as server, http.open():
        tts = TTS(api_key="k", base_url=str(server.make_url("/v2")))
        await _say(tts, "one")
        audio = await _say(tts, "two")
        await tts.aclose()

    expected = b"".join(frames)
    assert len(peer.of_type("init")) == 1
    assert audio[: len(expected)] == expected
