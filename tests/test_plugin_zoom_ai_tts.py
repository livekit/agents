"""Hermetic tests for the Zoom AI (Zoom TTS) TTS plugin, against a local aiohttp server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.plugins import zoom_ai
from livekit.plugins.zoom_ai import tts as zoom_tts

# the fake server and HTTP session are fixtures bound to a single task
pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

API_KEY = "test-key"
NO_RETRY = APIConnectOptions(max_retry=0, timeout=5)
ONE_RETRY = APIConnectOptions(max_retry=1, timeout=5)
PCM = b"\x01\x02\x03\x04" * 4410  # 0.2 s of 44.1 kHz PCM16

Reply = Callable[[web.Request], Awaitable[web.StreamResponse]]


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    # the framework's first retry otherwise waits a fixed 0.1s
    monkeypatch.setattr(APIConnectOptions, "_interval_for_retry", lambda self, n: 0.0)


class FakeZoomTTS:
    """Records each synthesize request and replies with ``reply`` (PCM by default)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply: Reply = stream_pcm

    async def synthesize(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        self.requests.append(
            {"headers": dict(request.headers), "raw": raw, "json": json.loads(raw)}
        )
        return await self.reply(request)


async def stream_pcm(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "audio/pcm;rate=44100"})
    await response.prepare(request)
    # split mid-sample, as network reads can
    await response.write(PCM[:4095])
    await asyncio.sleep(0.01)
    await response.write(PCM[4095:])
    await response.write_eof()
    return response


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeZoomTTS, str]]:
    fake = FakeZoomTTS()
    app = web.Application()
    app.router.add_post("/tts/synthesize", fake.synthesize)
    srv = TestServer(app)
    await srv.start_server()
    try:
        yield fake, str(srv.make_url("/tts"))
    finally:
        await srv.close()


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()


def make_tts(http: aiohttp.ClientSession, base_url: str, **kwargs: Any) -> zoom_ai.TTS:
    return zoom_ai.TTS(api_key=API_KEY, base_url=base_url, http_session=http, **kwargs)


async def synthesize(
    tts: zoom_ai.TTS, text: str, conn_options: APIConnectOptions = NO_RETRY
) -> bytes:
    audio = bytearray()
    async with tts.synthesize(text, conn_options=conn_options) as stream:
        async for event in stream:
            assert event.frame.sample_rate == 44100
            assert event.frame.num_channels == 1
            audio += bytes(event.frame.data)
    return bytes(audio)


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZOOM_AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZOOM_AI_API_KEY"):
        zoom_ai.TTS()


def test_capabilities_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOOM_AI_API_KEY", API_KEY)
    tts = zoom_ai.TTS()
    assert not tts.capabilities.streaming
    assert tts.sample_rate == 44100
    assert tts.num_channels == 1
    assert tts.provider == "Zoom"
    assert tts.model == "tts-fast"


def test_split_text_cuts_at_whitespace() -> None:
    assert zoom_tts._split_text("aaa bbb ccc", limit=7) == ["aaa bbb", "ccc"]
    assert zoom_tts._split_text("abcdefghij", limit=4) == ["abcd", "efgh", "ij"]
    assert zoom_tts._split_text("short") == ["short"]


async def test_streams_audio_and_matches_the_docs(server, http) -> None:
    fake, base = server
    audio = await synthesize(make_tts(http, base), "Hello from Zoom.")

    assert audio[: len(PCM)] == PCM  # the emitter may pad its last frame with silence
    assert not any(audio[len(PCM) :])
    request = fake.requests[0]
    assert request["json"] == {
        "input": "Hello from Zoom.",
        "voice": "jenna",
        "response_format": "pcm_44100",
    }
    headers = request["headers"]
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["Content-Type"] == "application/json"
    assert headers["User-Agent"].startswith("livekit-plugins-zoom-ai/")
    assert headers["x-zm-trackingid"].startswith("zoom_ai_community_livekit_")


async def test_options_map_to_request_fields(server, http) -> None:
    fake, base = server
    tts = make_tts(http, base, voice="ava", speed=1.2, volume=0.8)
    await synthesize(tts, "One.")
    tts.update_options(voice="mark", speed=None)
    await synthesize(tts, "Two.")

    first, second = (r["json"] for r in fake.requests)
    assert (first["voice"], first["speed"], first["volume"]) == ("ava", 1.2, 0.8)
    assert second["voice"] == "mark" and "speed" not in second and second["volume"] == 0.8


async def test_non_ascii_text_is_sent_as_utf8(server, http) -> None:
    """Unescaped UTF-8 keeps the longest input within the API's 32 KiB body limit."""
    fake, base = server
    await synthesize(make_tts(http, base), "Café crème.")

    raw = fake.requests[0]["raw"]
    assert "Café crème.".encode() in raw
    assert b"\\u" not in raw


async def test_long_text_is_split_across_requests(server, http) -> None:
    fake, base = server
    text = ("word " * (zoom_tts.MAX_INPUT_CHARS // 4)).strip()
    await synthesize(make_tts(http, base), text)

    inputs = [r["json"]["input"] for r in fake.requests]
    assert len(inputs) == 2
    assert all(len(i) <= zoom_tts.MAX_INPUT_CHARS for i in inputs)
    assert " ".join(inputs) == text
    ids = {r["headers"]["x-zm-trackingid"] for r in fake.requests}
    assert len(ids) == 2  # one tracking ID per request


async def test_unsupported_voice_reports_zoom_reason_without_retry(server, http) -> None:
    fake, base = server

    async def reject(request: web.Request) -> web.Response:
        return web.json_response(
            {"code": 400, "reason": "UNSUPPORTED_VOICE", "message": "Unsupported voice"},
            status=400,
        )

    fake.reply = reject
    with pytest.raises(APIStatusError) as excinfo:
        await synthesize(make_tts(http, base, voice="nobody"), "Hi.", ONE_RETRY)

    error = excinfo.value
    assert error.status_code == 400
    assert not error.retryable
    assert "Unsupported voice" in error.message
    assert error.request_id == fake.requests[0]["headers"]["x-zm-trackingid"]
    assert len(fake.requests) == 1


async def test_unavailable_service_is_retried(server, http) -> None:
    fake, base = server

    async def unavailable_once(request: web.Request) -> web.StreamResponse:
        if len(fake.requests) == 1:
            return web.json_response(
                {"code": 503, "reason": "SERVICE_UNAVAILABLE", "message": "Try again later"},
                status=503,
            )
        return await stream_pcm(request)

    fake.reply = unavailable_once
    audio = await synthesize(make_tts(http, base), "Hi.", ONE_RETRY)

    assert audio[: len(PCM)] == PCM
    assert len(fake.requests) == 2


async def test_stream_that_ends_early_is_a_connection_error(server, http) -> None:
    """A synthesis failure after the 200 ends the chunked response without its last chunk."""
    fake, base = server

    async def cut_off(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(PCM[:4096])
        await asyncio.sleep(0.01)
        assert request.transport is not None
        request.transport.close()
        return response

    fake.reply = cut_off
    with pytest.raises(APIConnectionError, match="ended early"):
        await synthesize(make_tts(http, base), "Hi.")


async def test_closing_the_stream_closes_the_request(server, http) -> None:
    """Closing the request is how Zoom TTS is told to stop synthesizing."""
    fake, base = server
    closed = asyncio.Event()

    async def slow(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        try:
            for _ in range(100):
                await response.write(PCM[:4096])
                await asyncio.sleep(0.05)
        except (asyncio.CancelledError, ConnectionError):
            # the client went away: aiohttp cancels the handler, or the write fails
            closed.set()
            raise
        await response.write_eof()
        return response

    fake.reply = slow
    stream = make_tts(http, base).synthesize("A long reply the user talks over.")
    async for _ in stream:
        break  # the first frame arrived; an interruption closes the stream here
    await stream.aclose()

    await asyncio.wait_for(closed.wait(), 2)
