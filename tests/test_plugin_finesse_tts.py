"""Tests for the Finesse TTS plugin, against a local server that speaks the Finesse protocol.

The fake server answers each context the way Finesse does: ``task_started``, then pairs of an
``audio_chunk`` event and the binary PCM frame it announces, then ``task_done`` once the client
finalizes the context.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIError, APIStatusError

pytestmark = pytest.mark.unit

API_KEY = "flv_test_key"
VOICE = "665f1a2b3c4d5e6f7a8b9c0d"
CHUNK = b"\x01\x00" * 2400  # 100 ms of 24 kHz s16le
NO_RETRY = APIConnectOptions(max_retry=0, timeout=5)


@dataclass
class FakeFinesse:
    """Records what the client sends; ``on_task`` decides how each context is answered."""

    url: str = ""
    received: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    protocols: list[str | None] = field(default_factory=list)
    closed_by_client: asyncio.Event = field(default_factory=asyncio.Event)
    on_task: Callable[[web.WebSocketResponse, str], Any] | None = None
    reject_status: int | None = None


async def _answer(ws: web.WebSocketResponse, context_id: str, chunks: int = 2) -> None:
    await ws.send_json({"type": "task_started", "context_id": context_id, "sample_rate": 24000})
    for i in range(chunks):
        await ws.send_json({"type": "audio_chunk", "context_id": context_id, "chunk_index": i})
        await ws.send_bytes(CHUNK)
    await ws.send_json({"type": "sentence_done", "context_id": context_id})


@pytest.fixture
async def server() -> AsyncIterator[FakeFinesse]:
    fake = FakeFinesse()

    async def handler(request: web.Request) -> web.StreamResponse:
        if fake.reject_status is not None:
            return web.json_response({"code": "token_invalid"}, status=fake.reject_status)
        fake.headers.append(dict(request.headers))
        ws = web.WebSocketResponse(protocols=("finesse.v1",))
        await ws.prepare(request)
        fake.protocols.append(ws.ws_protocol)
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            fake.received.append(data)
            ctx = data.get("context_id", "")
            if data["type"] == "task_start":
                if fake.on_task is not None:
                    await fake.on_task(ws, ctx)
                else:
                    await _answer(ws, ctx)
            if data.get("finalize"):
                await ws.send_json({"type": "task_done", "context_id": ctx})
        fake.closed_by_client.set()
        return ws

    app = web.Application()
    app.router.add_get("/stream", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    fake.url = f"ws://127.0.0.1:{port}/stream"
    try:
        yield fake
    finally:
        await runner.cleanup()


@pytest.fixture
async def session() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as s:
        yield s


def _tts(server: FakeFinesse, session: aiohttp.ClientSession, **kwargs: Any):  # type: ignore[no-untyped-def]
    from livekit.plugins.finesse import TTS

    tts = TTS(api_key=API_KEY, voice=VOICE, http_session=session, **kwargs)
    tts._opts.ws_url = server.url
    return tts


def _types(server: FakeFinesse) -> list[str]:
    return [m["type"] for m in server.received]


# ---- configuration ------------------------------------------------------------------------


def test_requires_api_key() -> None:
    from livekit.plugins.finesse import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(voice=VOICE)


def test_reads_api_key_from_env() -> None:
    from livekit.plugins.finesse import TTS

    with patch.dict("os.environ", {"FINESSE_API_KEY": "flv_env"}):
        assert TTS(voice=VOICE)._opts.api_key == "flv_env"


def test_defaults() -> None:
    from livekit.plugins.finesse import TTS

    tts = TTS(api_key=API_KEY, voice=VOICE)
    assert tts.provider == "Finesse"
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1
    assert tts.capabilities.streaming is True
    assert tts._opts.language == "en"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"voice": "  "}, "voice"),
        ({"sample_rate": 16000}, "sample_rate"),
        ({"sample_rate": 96000}, "sample_rate"),
        ({"num_step": 0}, "num_step"),
        ({"guidance_scale": 11}, "guidance_scale"),
        ({"length_scale": 0}, "length_scale"),
        ({"idle_flush_ms": 100}, "idle_flush_ms"),
    ],
)
def test_rejects_out_of_range_options(kwargs: dict[str, Any], match: str) -> None:
    from livekit.plugins.finesse import TTS

    args = {"api_key": API_KEY, "voice": VOICE, **kwargs}
    with pytest.raises(ValueError, match=match):
        TTS(**args)


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en", "en"),
        ("hi-IN", "hi"),
        ("tamil", "ta"),
        ("ar", "arb"),
        ("ne-NP", "npi"),
        ("no", "nb"),
        ("or-IN", "ory"),
    ],
)
def test_task_start_language(language: str, expected: str) -> None:
    from livekit.plugins.finesse import TTS
    from livekit.plugins.finesse.tts import _task_start

    tts = TTS(api_key=API_KEY, voice=VOICE, language=language)
    assert _task_start(tts._opts, "c1", "Hi.", finalize=False)["language"] == expected


def test_task_start_payload() -> None:
    from livekit.plugins.finesse import TTS
    from livekit.plugins.finesse.tts import _task_start

    tts = TTS(api_key=API_KEY, voice=VOICE, sample_rate=48000, num_step=16)
    assert _task_start(tts._opts, "c1", "Hello.", finalize=True) == {
        "type": "task_start",
        "context_id": "c1",
        "voice_id": VOICE,
        "text": "Hello.",
        "finalize": True,
        "language": "en",
        "output_format": "s16le",
        "sample_rate": 48000,
        "num_step": 16,
    }


def test_update_options() -> None:
    from livekit.plugins.finesse import TTS

    tts = TTS(api_key=API_KEY, voice=VOICE)
    tts.update_options(voice="other", language="hindi", length_scale=1.2)
    assert tts._opts.voice == "other"
    assert tts._opts.language == "hi"
    assert tts._opts.length_scale == 1.2


def test_update_options_rejects_invalid_values_atomically() -> None:
    from livekit.plugins.finesse import TTS

    tts = TTS(api_key=API_KEY, voice=VOICE)
    with pytest.raises(ValueError, match="num_step"):
        tts.update_options(voice="other", num_step=99)
    assert tts._opts.voice == VOICE


# ---- streaming ------------------------------------------------------------------------------


async def _collect(stream: Any) -> bytes:
    audio = b""
    async for ev in stream:
        audio += ev.frame.data.tobytes()
    return audio


async def test_stream_sends_text_as_it_arrives(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    tts = _tts(server, session)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Hello")
    stream.push_text(" world.")
    stream.flush()
    stream.end_input()

    audio = await _collect(stream)

    assert len(audio) == 2 * len(CHUNK)
    assert _types(server) == ["task_start", "task_continue", "task_continue"]
    start, more, final = server.received
    assert start["text"] == "Hello" and start["finalize"] is False
    assert start["voice_id"] == VOICE and start["output_format"] == "s16le"
    assert more == {
        "type": "task_continue",
        "context_id": start["context_id"],
        "text": " world.",
        "finalize": False,
    }
    assert final["text"] == "" and final["finalize"] is True
    assert server.protocols == ["finesse.v1"]
    assert server.headers[0]["X-API-Key"] == API_KEY
    assert server.headers[0]["X-Finesse-Client"].startswith("livekit-agents/")
    await tts.aclose()


async def test_stream_waits_for_text_before_opening_the_context(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    tts = _tts(server, session)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("  ")
    stream.push_text("Hi.")
    stream.end_input()

    await _collect(stream)

    assert server.received[0]["type"] == "task_start"
    assert server.received[0]["text"] == "  Hi."
    await tts.aclose()


async def test_stream_without_text_sends_nothing(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    tts = _tts(server, session)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.end_input()

    assert await _collect(stream) == b""
    assert server.received == []
    await tts.aclose()


async def test_stream_drops_audio_for_other_contexts(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    async def on_task(ws: web.WebSocketResponse, ctx: str) -> None:
        # The tail of an earlier, cancelled context arrives first on a reused connection.
        await ws.send_json({"type": "audio_chunk", "context_id": "stale"})
        await ws.send_bytes(b"\xff\x7f" * 2400)
        await ws.send_json({"type": "task_done", "context_id": "stale"})
        await _answer(ws, ctx, chunks=1)

    server.on_task = on_task
    tts = _tts(server, session)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Hi.")
    stream.end_input()

    audio = await _collect(stream)

    assert audio == CHUNK
    await tts.aclose()


async def test_connection_is_reused_across_streams(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    tts = _tts(server, session)
    for text in ("One.", "Two."):
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text(text)
        stream.end_input()
        await _collect(stream)

    assert len(server.headers) == 1
    starts = [m for m in server.received if m["type"] == "task_start"]
    assert [m["text"] for m in starts] == ["One.", "Two."]
    assert starts[0]["context_id"] != starts[1]["context_id"]
    await tts.aclose()


async def test_interrupting_a_stream_cancels_its_context(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    started = asyncio.Event()

    async def on_task(ws: web.WebSocketResponse, ctx: str) -> None:
        await ws.send_json({"type": "task_started", "context_id": ctx})
        await ws.send_json({"type": "audio_chunk", "context_id": ctx})
        await ws.send_bytes(CHUNK)
        started.set()  # then keep the turn open, as a long reply would

    server.on_task = on_task
    tts = _tts(server, session)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("A long reply that the user talks over.")

    await asyncio.wait_for(started.wait(), 5)
    await stream.aclose()

    # The cancel must reach Finesse right away, not when the next turn reconnects.
    async def cancel_received() -> None:
        while not any(m["type"] == "task_cancel" for m in server.received):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(cancel_received(), 5)
    start = server.received[0]
    assert {"type": "task_cancel", "context_id": start["context_id"]} in server.received
    await tts.aclose()


async def test_synthesize_sends_one_finalized_context(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    tts = _tts(server, session)

    audio = await _collect(tts.synthesize("Hello there.", conn_options=NO_RETRY))

    # The emitter pads the last frame of a one-shot synthesis with silence.
    assert audio.startswith(CHUNK * 2)
    assert set(audio[len(CHUNK) * 2 :]) <= {0}
    assert _types(server) == ["task_start"]
    assert server.received[0]["text"] == "Hello there."
    assert server.received[0]["finalize"] is True
    await tts.aclose()


async def test_long_text_is_split_below_the_frame_limit(
    server: FakeFinesse, session: aiohttp.ClientSession
) -> None:
    from livekit.plugins.finesse.tts import _MAX_TEXT_CHARS

    tts = _tts(server, session)
    text = "a" * (_MAX_TEXT_CHARS * 2 + 10)

    await _collect(tts.synthesize(text, conn_options=NO_RETRY))

    assert _types(server) == ["task_start", "task_continue", "task_continue"]
    assert "".join(m["text"] for m in server.received) == text
    assert [m["finalize"] for m in server.received] == [False, False, True]
    await tts.aclose()


# ---- errors ---------------------------------------------------------------------------------


async def _synthesize_error(tts: Any) -> APIError:
    with pytest.raises(APIError) as exc:
        await _collect(tts.synthesize("Hello.", conn_options=NO_RETRY))
    return exc.value


@pytest.mark.parametrize(
    ("event", "retryable"),
    [
        ({"type": "task_failed", "code": "voice_not_found"}, False),
        ({"type": "error", "code": "insufficient_credits", "fatal": True}, False),
        ({"type": "error", "code": "upstream_closed", "fatal": True}, True),
    ],
)
async def test_server_errors(
    server: FakeFinesse,
    session: aiohttp.ClientSession,
    event: dict[str, Any],
    retryable: bool,
) -> None:
    async def on_task(ws: web.WebSocketResponse, ctx: str) -> None:
        await ws.send_json({**event, "context_id": ctx})

    server.on_task = on_task
    tts = _tts(server, session)

    err = await _synthesize_error(tts)

    assert isinstance(err, APIStatusError)
    assert err.body == {"code": event["code"]}
    assert err.retryable is retryable
    await tts.aclose()


@pytest.mark.parametrize(("close_code", "retryable"), [(1013, True), (1008, False)])
async def test_unexpected_close(
    server: FakeFinesse, session: aiohttp.ClientSession, close_code: int, retryable: bool
) -> None:
    async def on_task(ws: web.WebSocketResponse, ctx: str) -> None:
        await ws.close(code=close_code)

    server.on_task = on_task
    tts = _tts(server, session)

    err = await _synthesize_error(tts)

    assert isinstance(err, APIStatusError)
    assert err.status_code == close_code
    assert err.retryable is retryable
    await tts.aclose()


@pytest.mark.parametrize(("status", "retryable"), [(401, False), (429, True), (503, True)])
async def test_refused_handshake(
    server: FakeFinesse, session: aiohttp.ClientSession, status: int, retryable: bool
) -> None:
    server.reject_status = status
    tts = _tts(server, session)

    err = await _synthesize_error(tts)

    assert isinstance(err, APIStatusError)
    assert err.status_code == status
    assert err.retryable is retryable
    # The handshake error holds the request headers; none of it may leak into the error.
    assert API_KEY not in repr(err) and err.__cause__ is None
    await tts.aclose()
