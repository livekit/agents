from __future__ import annotations

import json
import logging

import pytest

from livekit.agents import APIConnectOptions, http
from livekit.plugins.slng import tts as slng_tts

pytestmark = pytest.mark.unit

PCM = b"\x00\x01" * 2400


class Socket:
    def __init__(self) -> None:
        self.closed = False
        self._sent_audio = False

    async def send_str(self, text: str) -> None:
        pass

    async def receive(self, **kwargs) -> http.WebSocketMessage:
        if not self._sent_audio:
            self._sent_audio = True
            return http.WebSocketMessage(http.WebSocketMessageType.BINARY, PCM)
        return http.WebSocketMessage(
            http.WebSocketMessageType.TEXT, json.dumps({"type": "audio_end"})
        )

    async def close(self) -> None:
        self.closed = True


async def test_synthesis_forwards_phase_options_to_its_connection() -> None:
    socket = Socket()
    policies: list[APIConnectOptions] = []

    class Session:
        async def ws_connect(self, url: str, *, conn_options: APIConnectOptions, **kwargs):
            policies.append(conn_options)
            return socket

    model = slng_tts.TTS(
        api_key="test",
        model="deepgram/aura:2",
        voice="test",
        http_session=Session(),  # type: ignore[arg-type]
    )
    options = APIConnectOptions(max_retry=0, connect_timeout=0.25, read_timeout=0.5)
    try:
        async with model.synthesize("hello", conn_options=options) as stream:
            audio = b"".join([event.frame.data.tobytes() async for event in stream])
        assert audio[: len(PCM)] == PCM
        assert set(audio[len(PCM) :]) <= {0}
        assert policies == [options]
        assert socket.closed
    finally:
        await model.aclose()


async def test_streaming_text_is_kept_out_of_log_messages(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def connect(*args, **kwargs):
        return Socket()

    monkeypatch.setattr(slng_tts.TTS, "_connect_ws", connect)
    model = slng_tts.TTS(api_key="test", model="deepgram/aura:2", voice="test")
    secret = "CONFIDENTIALACCOUNT"
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.slng")
    try:
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text(secret)
            stream.flush()
            stream.end_input()
            assert b"".join([event.frame.data.tobytes() async for event in stream]) == PCM
        assert all(secret not in record.getMessage() for record in caplog.records)
    finally:
        await model.aclose()


async def test_connection_chain_fails_over_without_losing_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import APIStatusError

    attempted: list[str] = []
    sockets: list[Socket] = []

    async def connect(model: slng_tts.TTS, *args, **kwargs):
        attempted.append(model.model)
        if model.model == "deepgram/aura:2":
            raise APIStatusError("unavailable", status_code=503)
        socket = Socket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(slng_tts.TTS, "_connect_ws", connect)
    model = slng_tts.TTS(
        api_key="test", connections=["deepgram/aura:2", "cartesia/sonic:3"], voice="test"
    )
    try:
        async with model.synthesize("hello") as stream:
            audio = b"".join([event.frame.data.tobytes() async for event in stream])
        assert audio[: len(PCM)] == PCM
        assert set(audio[len(PCM) :]) <= {0}
        assert attempted[0] == "deepgram/aura:2"
        assert "cartesia/sonic:3" in attempted
        assert all(socket.closed for socket in sockets)
    finally:
        await model.aclose()
