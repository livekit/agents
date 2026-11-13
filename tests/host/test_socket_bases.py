"""The socket bases: a plugin's typed protocol against a loopback provider, one fake per kind
(``tests/type_checks/socket_protocols.py``, which strict mypy also checks)."""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import AsyncIterator, Awaitable, Callable

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, stt
from tests.fakes import until
from tests.type_checks.socket_protocols import (
    CodecSttStream,
    LoopbackRealtime,
    LoopbackStt,
    LoopbackTts,
    event_frame,
)

NO_RETRY = APIConnectOptions(max_retry=0)

Handler = Callable[[web.WebSocketResponse, web.Request], Awaitable[None]]


@pytest.fixture
async def peer() -> AsyncIterator[Callable[[Handler], Awaitable[str]]]:
    """Serves one handler per test on a loopback port; returns its ``ws://`` URL."""
    runners: list[web.AppRunner] = []

    async def serve(handler: Handler) -> str:
        async def accept(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await handler(ws, request)
            return ws

        app = web.Application()
        app.router.add_get("/ws", accept)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"ws://127.0.0.1:{port}/ws"

    yield serve
    for runner in runners:
        await runner.cleanup()


def audio(ms: int) -> rtc.AudioFrame:
    samples = 16 * ms
    return rtc.AudioFrame(b"\x00\x01" * samples, 16000, 1, samples)


async def events(stream: stt.RecognizeStream) -> list[stt.SpeechEvent]:
    return [event async for event in stream]


# Catches: hand-written send/receive loops (the goodbye never sent, the provider's last answer
# dropped at the input's end), audio sent in the caller's frame size, JSON written by hand.
async def test_stt_stream_sends_typed_commands_and_reads_until_the_provider_finishes(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    seen: list[str] = []
    query: dict[str, str] = {}

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        query.update(request.query)
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.BINARY:
                seen.append(f"<{len(msg.data)} bytes>")
                continue
            command = json.loads(msg.data)
            seen.append(command["type"])
            if command["type"] == "finalize":
                await ws.send_json({"type": "transcript", "text": "hello", "final": True})
            elif command["type"] == "close":
                await ws.send_json({"type": "finished"})

    url = await peer(provider)
    stream = LoopbackStt(url).stream(conn_options=NO_RETRY)
    for _ in range(10):
        stream.push_frame(audio(10))
    stream.end_input()
    got = await events(stream)
    finals = [e for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["hello"]
    assert seen == ["<1600 bytes>", "<1600 bytes>", "finalize", "close"]
    assert query == {"encoding": "linear16"}


# Catches: a stream failed by what the plugin never reads: a new event type the provider added,
# or a field it sends beyond the declared ones.
async def test_unknown_message_types_are_skipped_and_extra_fields_ignored(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        await ws.send_json({"type": "speech_started_v2", "at": 1.5})
        await ws.send_json({"type": "speech_started_v2", "at": 2.5})
        await ws.send_json(
            {"type": "transcript", "text": "hello", "final": True, "words": [], "meta": {"x": 1}}
        )
        await ws.send_json({"type": "finished", "request_id": "r-1"})
        await ws.receive()

    stream = LoopbackStt(await peer(provider)).stream(conn_options=NO_RETRY)
    stream.end_input()
    got = await events(stream)
    finals = [e for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["hello"]


# Catches: a declared message whose read field is malformed logged and dropped, or retried as if
# the next connection would answer differently.
async def test_a_declared_message_with_a_wrong_typed_field_fails_with_protocol(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        await ws.send_json({"type": "transcript", "text": 5, "final": True})
        await ws.receive()

    stream = LoopbackStt(await peer(provider)).stream(conn_options=NO_RETRY)
    with pytest.raises(APIError) as failed:
        await events(stream)
    assert (failed.value.kind, failed.value.retryable) == ("protocol", False)
    assert '"text": 5' in (failed.value.detail or "")


# Catches: a binary-framed protocol (AWS event-stream) forced through JSON validation, or its
# signed URL computed once instead of per connection.
async def test_a_binary_codec_decodes_its_own_frames(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    received: list[str] = []
    query: dict[str, str] = {}

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        query.update(request.query)
        msg = await ws.receive()
        (length,) = struct.unpack(">I", msg.data[:4])
        received.append(json.loads(msg.data[4 : 4 + length])[":event-type"])
        await ws.send_bytes(event_frame({":message-type": "event"}, b"hi there"))
        await ws.send_bytes(event_frame({":message-type": "exception"}, b"throttled"))
        await ws.receive()

    stt_ = LoopbackStt(await peer(provider))
    stream = CodecSttStream(stt=stt_, conn_options=NO_RETRY)
    stream.push_frame(audio(10))
    got: list[stt.SpeechEvent] = []
    with pytest.raises(APIError) as failed:
        async for event in stream:
            got.append(event)
    assert [e.alternatives[0].text for e in got if e.alternatives] == ["hi there"]
    assert failed.value.detail == "throttled"
    assert received == ["AudioEvent"]
    assert query == {"X-Amz-Signature": "fresh"}


# Catches: each multiplexed stream opening its own socket, or audio reaching the wrong stream.
async def test_tts_streams_share_the_multiplexed_socket(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    connections = 0

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        nonlocal connections
        connections += 1
        async for msg in ws:
            command = json.loads(msg.data)
            context = command["context_id"]
            if command.get("final"):
                await ws.send_json({"type": "done", "context_id": context})
            elif text := command.get("text"):
                tone = b"\x01\x00" if text.startswith("one") else b"\x02\x00"
                await ws.send_bytes(context.encode() + b"|" + tone * 160)

    tts_ = LoopbackTts(await peer(provider))

    async def say(text: str) -> bytes:
        stream = tts_.stream(conn_options=NO_RETRY)
        stream.push_text(text)
        stream.end_input()
        return b"".join([bytes(audio.frame.data) async for audio in stream])

    one, two = await asyncio.gather(say("one fish."), say("two fish."))
    assert one and set(one[::2]) == {1}
    assert two and set(two[::2]) == {2}
    assert connections == 1
    await tts_.mux.aclose()


# Catches: a realtime plugin's own reconnect loop and ping answering, and a change applied to
# the live session instead of a new one.
async def test_realtime_answers_protocol_frames_and_reconnects_on_change(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    instructions: list[str] = []
    seen: list[str] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        instructions.append(request.query["instructions"])
        await ws.send_json({"type": "ping"})
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.BINARY:
                seen.append("audio")
                continue
            command = json.loads(msg.data)
            seen.append(command["type"])
            if command["type"] == "say":
                await ws.send_json({"type": "said", "text": command["text"]})

    model = LoopbackRealtime(await peer(provider), NO_RETRY)
    session = model.session()
    try:
        await until(lambda: "pong" in seen, "the ping answered")
        session.push_audio(audio(10))
        session.generate_reply(instructions="hello")
        await until(lambda: session.heard == ["hello"], "the reply heard")
        await session.update_instructions("be brief")
        await until(lambda: instructions == ["", "be brief"], "a new session")
        assert seen[:3] == ["pong", "audio", "say"]
    finally:
        await session.aclose()
