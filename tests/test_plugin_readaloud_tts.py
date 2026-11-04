from __future__ import annotations

import asyncio
import struct
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("readaloud")


def _pcm(num_samples: int, seed: int = 0) -> bytes:
    """Bare little-endian int16 PCM, which is what ReadAloud streams."""
    return struct.pack(
        f"<{num_samples}h", *(((i * 97 + seed) % 2000) - 1000 for i in range(num_samples))
    )


class _Server:
    """Local stand-in for the ReadAloud API so the response path runs end to end."""

    def __init__(self, handler: Callable) -> None:
        self._handler = handler
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> _Server:
        async def record(request: web.Request) -> web.StreamResponse:
            self.requests.append(
                {
                    "path": request.path,
                    "query": dict(request.query),
                    "headers": dict(request.headers),
                    "json": await request.json(),
                }
            )
            return await self._handler(request)

        app = web.Application()
        app.router.add_post("/v1/text-to-speech", record)
        app.router.add_post("/v1/text-to-speech/{voice}/stream", record)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.base_url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()


def _pcm_handler(chunks: list[bytes]) -> Callable:
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"x-request-id": "req-1"})
        await resp.prepare(request)
        for chunk in chunks:
            await resp.write(chunk)
            await asyncio.sleep(0)
        await resp.write_eof()
        return resp

    return handler


def _error_handler(status: int, body: Any, *, raw: bool = False) -> Callable:
    async def handler(request: web.Request) -> web.StreamResponse:
        if raw:
            return web.Response(status=status, text=body)
        return web.json_response(body, status=status)

    return handler


def _tts(server: _Server, **kwargs: Any):
    from livekit.plugins.readaloud import TTS

    return TTS(api_key="test-key", base_url=server.base_url, http_session=server.session, **kwargs)


_OPTIONS = APIConnectOptions(max_retry=0, timeout=5)


async def _collect_synthesize(tts, text: str = "hello world") -> bytes:
    audio = bytearray()
    stream = tts.synthesize(text, conn_options=_OPTIONS)
    try:
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
    finally:
        await stream.aclose()
    return bytes(audio)


async def _collect_stream(tts, pushes: list[str]) -> bytes:
    audio = bytearray()
    stream = tts.stream(conn_options=_OPTIONS)
    try:
        for text in pushes:
            stream.push_text(text)
        stream.end_input()
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
    finally:
        await stream.aclose()
    return bytes(audio)


def _assert_audio_is(audio: bytes, expected: bytes) -> None:
    """The emitter packs fixed-size frames, so the final frame is zero-padded. Every
    input sample must appear unchanged, and the only surplus may be that padding."""
    assert audio[: len(expected)] == expected
    assert set(audio[len(expected) :]) <= {0}


def test_requires_api_key():
    from livekit.plugins.readaloud import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS()


def test_api_key_from_argument_and_env():
    from livekit.plugins.readaloud import TTS

    assert TTS(api_key="arg-key")._api_key == "arg-key"
    with patch.dict("os.environ", {"READALOUD_API_KEY": "env-key"}):
        assert TTS()._api_key == "env-key"


def test_properties_and_capabilities():
    from livekit.plugins.readaloud import TTS

    tts = TTS(api_key="test-key")
    assert tts.provider == "ReadAloud"
    assert tts.model == "live"
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1
    assert tts.capabilities.streaming is True
    assert tts.capabilities.aligned_transcript is False
    assert TTS(api_key="test-key", engine="studio").model == "studio"


def test_rejects_unsupported_sample_rate_and_engine():
    from livekit.plugins.readaloud import TTS

    with pytest.raises(ValueError, match="sample_rate must be one of"):
        TTS(api_key="test-key", sample_rate=44100)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="engine must be one of"):
        TTS(api_key="test-key", engine="piper")  # type: ignore[arg-type]


def test_update_options():
    from livekit.plugins.readaloud import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(speed=1.5)
    assert tts._opts.voice == "default"
    assert tts._opts.speed == 1.5
    tts.update_options(voice="custom:abc")
    assert tts._opts.voice == "custom:abc"
    assert tts._opts.speed == 1.5


async def test_native_request_shape_and_audio():
    pcm = _pcm(4800)
    async with _Server(_pcm_handler([pcm[:1001], pcm[1001:3000], pcm[3000:]])) as server:
        tts = _tts(server, voice="custom:abc", speed=1.25)
        audio = await _collect_synthesize(tts)

    req = server.requests[0]
    assert req["path"] == "/v1/text-to-speech"
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert req["json"] == {
        "text": "hello world",
        "voice": "custom:abc",
        "speed": 1.25,
        "format": "pcm_24000",
        "engine": "live",
    }
    # chunks split at odd byte offsets must come back as one unbroken signal
    _assert_audio_is(audio, pcm)


async def test_telephony_sample_rate_requests_pcm_8000():
    pcm = _pcm(1600)
    async with _Server(_pcm_handler([pcm])) as server:
        tts = _tts(server, sample_rate=8000, engine="studio")
        assert tts.sample_rate == 8000
        audio = await _collect_synthesize(tts)

    assert server.requests[0]["path"] == "/v1/text-to-speech"
    assert server.requests[0]["json"]["format"] == "pcm_8000"
    assert server.requests[0]["json"]["engine"] == "studio"
    _assert_audio_is(audio, pcm)


async def test_16khz_uses_compatible_route():
    pcm = _pcm(3200)
    async with _Server(_pcm_handler([pcm])) as server:
        tts = _tts(server, sample_rate=16000, speed=0.9)
        audio = await _collect_synthesize(tts)

    req = server.requests[0]
    assert req["path"] == "/v1/text-to-speech/readaloud-default/stream"
    assert req["query"] == {"output_format": "pcm_16000"}
    assert req["headers"]["xi-api-key"] == "test-key"
    assert "Authorization" not in req["headers"]
    assert req["json"] == {
        "text": "hello world",
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"speed": 0.9},
    }
    _assert_audio_is(audio, pcm)


async def test_16khz_keeps_explicit_voice_and_studio_model():
    async with _Server(_pcm_handler([_pcm(160)])) as server:
        tts = _tts(server, sample_rate=16000, voice="custom:de-de-mls-f", engine="studio")
        await _collect_synthesize(tts)

    req = server.requests[0]
    assert req["path"] == "/v1/text-to-speech/custom:de-de-mls-f/stream"
    assert req["json"]["model_id"] == "eleven_multilingual_v2"


async def test_base_url_trailing_slash_is_normalized():
    async with _Server(_pcm_handler([_pcm(160)])) as server:
        from livekit.plugins.readaloud import TTS

        tts = TTS(api_key="k", base_url=server.base_url + "/", http_session=server.session)
        await _collect_synthesize(tts)

    assert server.requests[0]["path"] == "/v1/text-to-speech"


@pytest.mark.parametrize(
    ("status", "body", "message", "retryable"),
    [
        (401, {"error": "invalid or missing API key"}, "invalid or missing API key", False),
        (402, {"error": "Free tier exhausted"}, "Free tier exhausted", False),
        (
            404,
            {"detail": {"status": "voice_not_found", "message": "no such voice"}},
            "no such voice",
            False,
        ),
        (413, {"error": "text too long (max 5000 chars)"}, "text too long (max 5000 chars)", False),
        (422, {"detail": [{"msg": "bad speed"}]}, "bad speed", False),
        (
            429,
            {"detail": {"message": "too many concurrent requests"}},
            "too many concurrent requests",
            True,
        ),
        (503, {"error": "worker at capacity"}, "worker at capacity", True),
    ],
)
async def test_error_status_becomes_api_status_error(status, body, message, retryable):
    async with _Server(_error_handler(status, body)) as server:
        with pytest.raises(APIStatusError) as excinfo:
            await _collect_synthesize(_tts(server))

    assert excinfo.value.status_code == status
    assert excinfo.value.message == message
    assert excinfo.value.retryable is retryable


async def test_non_json_error_body_still_raises():
    async with _Server(_error_handler(502, "bad gateway", raw=True)) as server:
        with pytest.raises(APIStatusError) as excinfo:
            await _collect_synthesize(_tts(server))

    assert excinfo.value.status_code == 502
    assert excinfo.value.message == "bad gateway"


async def test_unreachable_server_raises_connection_error():
    async with aiohttp.ClientSession() as session:
        from livekit.plugins.readaloud import TTS

        tts = TTS(api_key="k", base_url="http://127.0.0.1:1", http_session=session)
        with pytest.raises(APIConnectionError):
            await _collect_synthesize(tts)


async def test_stream_synthesizes_each_sentence_in_order():
    first, second = _pcm(2400, seed=1), _pcm(2400, seed=2)
    queue = [first, second]

    async def handler(request: web.Request) -> web.StreamResponse:
        return await _pcm_handler([queue.pop(0)])(request)

    async with _Server(handler) as server:
        tts = _tts(server)
        audio = await _collect_stream(
            tts, ["Thanks for calling Acme. ", "How can I ", "help you today?"]
        )

    texts = [r["json"]["text"] for r in server.requests]
    assert texts == ["Thanks for calling Acme.", "How can I help you today?"]
    assert all(r["headers"]["Authorization"] == "Bearer test-key" for r in server.requests)
    _assert_audio_is(audio, first + second)


async def test_stream_with_no_text_makes_no_request():
    async with _Server(_pcm_handler([_pcm(160)])) as server:
        audio = await _collect_stream(_tts(server), [])

    assert server.requests == []
    assert audio == b""


async def test_stream_error_surfaces():
    async with _Server(_error_handler(401, {"error": "invalid or missing API key"})) as server:
        with pytest.raises(APIStatusError) as excinfo:
            await _collect_stream(_tts(server), ["Hello there, how are you doing today?"])

    assert excinfo.value.status_code == 401


async def test_closing_the_stream_drops_the_connection():
    """A barge-in closes the stream; the server must see the client go away."""
    disconnected = asyncio.Event()
    started = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        await resp.prepare(request)
        try:
            await resp.write(_pcm(4800))
            started.set()
            for _ in range(200):
                await asyncio.sleep(0.05)
                await resp.write(_pcm(480))
        except (ConnectionResetError, asyncio.CancelledError):
            disconnected.set()
            raise
        return resp

    async with _Server(handler) as server:
        stream = _tts(server).synthesize("a long reply", conn_options=_OPTIONS)
        async for _ in stream:
            await asyncio.wait_for(started.wait(), 5)
            break
        await stream.aclose()
        await asyncio.wait_for(disconnected.wait(), 5)
