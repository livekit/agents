"""Uplift AI TTS: Socket.IO framing and provider errors, against a local Engine.IO peer."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from livekit.agents import APIConnectOptions, APIError
from livekit.plugins.upliftai.tts import TTS, WEBSOCKET_NAMESPACE
from tests.plugins.conftest import Connection, Frame, WsPeer

OPEN = '0{"sid":"s","upgrades":[],"pingInterval":25000,"pingTimeout":20000,"maxPayload":1000000}'


def uplift(peer: WsPeer, replies: list[dict[str, object]]) -> TTS:
    """A TTS on ``peer`` speaking Uplift's Socket.IO namespace: the connect is acknowledged,
    then each synthesis answered with ``replies``, under its request id."""
    peer.hello = [OPEN, "2"]  # opened, then pinged

    def reply(_: Connection, message: str | bytes) -> list[Frame]:
        assert isinstance(message, str)
        if message.startswith(f"40{WEBSOCKET_NAMESPACE},"):
            ready = json.dumps(["message", {"type": "ready"}])
            return [f'40{WEBSOCKET_NAMESPACE},{{"sid":"n"}}', f"42{WEBSOCKET_NAMESPACE},{ready}"]
        if message.startswith(f"42{WEBSOCKET_NAMESPACE},"):
            _, request = json.loads(message.split(",", 1)[1])
            answers = [{**r, "requestId": request["requestId"]} for r in replies]
            return [f"42{WEBSOCKET_NAMESPACE},{json.dumps(['message', a])}" for a in answers]
        return []

    peer.reply = reply
    return TTS(api_key="key", base_url=peer.url, output_format="PCM_22050_16")


async def _synthesize(provider: TTS) -> None:
    async with provider.synthesize("hello", conn_options=APIConnectOptions(max_retry=0)) as stream:
        async for _ in stream:
            pytest.fail("a failed synthesis produced audio")


async def test_synthesis_authenticates_on_the_namespace_and_answers_pings(ws_peer: WsPeer) -> None:
    """The namespace connect carries the key, the request goes out once it is acknowledged, and
    the server's ping is answered with a pong."""
    provider = uplift(ws_peer, [{"type": "audio", "audio": "AAAA"}, {"type": "audio_end"}])
    try:
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("hello")
            stream.end_input()
            async with asyncio.timeout(2):
                audio = [event async for event in stream]
    finally:
        await provider.aclose()

    assert audio
    connection = ws_peer.connections[0]
    assert connection.request.path == "/socket.io/"
    assert connection.request.query["EIO"] == "4"
    assert connection.texts[0] == f'40{WEBSOCKET_NAMESPACE},{{"token": "key"}}'
    assert "3" in connection.texts


async def test_provider_rejection_fails_synthesis(ws_peer: WsPeer) -> None:
    """A provider error fails synthesis for good, carrying the provider's message."""
    provider = uplift(ws_peer, [{"type": "error", "message": "voice unavailable"}])
    try:
        with pytest.raises(APIError) as error:
            await _synthesize(provider)
    finally:
        await provider.aclose()

    assert error.value.retryable is False
    assert isinstance(error.value.body, dict)
    assert error.value.body["message"] == "voice unavailable"


async def test_malformed_socket_message_fails_pending_synthesis_without_waiting_for_timeout(
    ws_peer: WsPeer,
) -> None:
    """A malformed audio message fails the pending synthesis at once, not at its timeout."""
    provider = uplift(ws_peer, [{"type": "audio", "audio": []}])
    try:
        with pytest.raises(APIError) as failure:
            async with asyncio.timeout(1):
                await _synthesize(provider)
    finally:
        await provider.aclose()

    assert failure.value.kind == "protocol"
    assert isinstance(failure.value.__cause__, ValidationError)


# Catches: an event Uplift added since (another name, more arguments) or a message type it
# added since failing the synthesis as a protocol error instead of being skipped.
async def test_events_and_message_types_uplift_added_since_are_skipped(ws_peer: WsPeer) -> None:
    provider = uplift(ws_peer, [{"type": "audio", "audio": "AAAA"}, {"type": "audio_end"}])
    answer = ws_peer.reply

    def reply(connection: Connection, message: str | bytes) -> list[Frame]:
        frames = answer(connection, message)
        if frames and isinstance(message, str) and message.startswith(f"42{WEBSOCKET_NAMESPACE},"):
            added = [["usage", {"characters": 5}, 1], ["message", {"type": "alignment"}]]
            frames[:0] = [f"42{WEBSOCKET_NAMESPACE},{json.dumps(e)}" for e in added]
        return frames

    ws_peer.reply = reply
    try:
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("hello")
            stream.end_input()
            async with asyncio.timeout(2):
                audio = [event async for event in stream]
    finally:
        await provider.aclose()

    assert audio
