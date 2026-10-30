"""Gnani TTS request/response behaviour against a local fake of the Vachana API.

The fake reproduces wire behaviour observed on the live API (2026-10-30):

* REST returns a complete WAV file.
* SSE / WebSocket return one WAV header with ``0xFFFFFFFF`` placeholder sizes
  (the length is unknown while streaming) followed by PCM, then raw PCM chunks.
* A WebSocket request ends with *two* ``complete`` messages: one carrying the final
  audio chunk, then a bare ``"Streaming completed"``.
* Unsupported models get ``400 {"success": false, "message": ..., "status_code": 400}``.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("gnani")

SAMPLE_RATE = 16000
# 0.5 s of a quiet ramp, so we can tell real audio from padding/silence
PCM = struct.pack("<8000h", *(((i * 37) % 600) + 200 for i in range(8000)))


def _wav_header(data_len: int | None, sample_rate: int = SAMPLE_RATE) -> bytes:
    """44-byte PCM16 mono WAV header; ``data_len=None`` mimics the streaming placeholder."""
    unknown = 0xFFFFFFFF
    riff_size = unknown if data_len is None else 36 + data_len
    size = unknown if data_len is None else data_len
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", size)
    )


def _chunks(pcm: bytes, n: int = 4) -> list[bytes]:
    """Split a streamed response: the first chunk carries the WAV header, the rest are raw PCM."""
    step = len(pcm) // n
    parts = [pcm[i * step : (i + 1) * step] for i in range(n - 1)] + [pcm[(n - 1) * step :]]
    parts[0] = _wav_header(None) + parts[0]
    return parts


class FakeGnani:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def _record(self, request: web.Request, body: dict[str, Any]) -> None:
        self.requests.append(body)
        self.headers.append(dict(request.headers))

    @staticmethod
    def _error(body: dict[str, Any]) -> web.Response | None:
        if body.get("model") not in ("timbre-v2.5", "timbre-v3.0"):
            return web.json_response(
                {
                    "success": False,
                    "message": f"Model '{body.get('model')}' is deprecated and is no longer "
                    'available. The supported models are "timbre-v2.5", "timbre-v3.0".',
                    "status_code": 400,
                },
                status=400,
            )
        return None

    async def rest(self, request: web.Request) -> web.Response:
        body = await request.json()
        self._record(request, body)
        if (err := self._error(body)) is not None:
            return err
        return web.Response(body=_wav_header(len(PCM)) + PCM, content_type="audio/wav")

    async def sse(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self._record(request, body)
        if (err := self._error(body)) is not None:
            return err
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'event: start\ndata: {"status": "streaming_started", "text": "hi"}\n\n')
        parts = _chunks(PCM)
        for i, part in enumerate(parts, start=1):
            last = i == len(parts)
            payload = {"chunk_index": i, "audio": base64.b64encode(part).decode()}
            if last:
                payload["is_final"] = True
            await resp.write(f"event: chunk\ndata: {json.dumps(payload)}\n\n".encode())
        await resp.write_eof()
        return resp

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            body = json.loads(msg.data)
            self._record(request, body)
            if (err := self._error(body)) is not None:
                await ws.send_json({"type": "error", "message": json.loads(err.text)["message"]})
                continue
            await ws.send_json({"type": "start", "message": "Streaming started"})
            parts = _chunks(PCM)
            for i, part in enumerate(parts[:-1], start=1):
                await ws.send_json(
                    {
                        "type": "audio",
                        "data": {"chunk_index": i, "audio": base64.b64encode(part).decode()},
                    }
                )
            await ws.send_json(
                {
                    "type": "complete",
                    "data": {
                        "chunk_index": len(parts),
                        "audio": base64.b64encode(parts[-1]).decode(),
                        "is_final": True,
                    },
                }
            )
            await ws.send_json({"type": "complete", "message": "Streaming completed"})
        return ws


@pytest.fixture
async def server():
    fake = FakeGnani()
    app = web.Application()
    app.router.add_post("/api/v1/tts/inference", fake.rest)
    app.router.add_post("/api/v1/tts/sse", fake.sse)
    app.router.add_get("/api/v1/tts", fake.ws)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    fake.base_url = f"http://127.0.0.1:{port}"  # type: ignore[attr-defined]
    fake.session = aiohttp.ClientSession()  # type: ignore[attr-defined]
    try:
        yield fake
    finally:
        await fake.session.close()  # type: ignore[attr-defined]
        await runner.cleanup()


def _make_tts(server: Any, **kwargs: Any):
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", base_url=server.base_url, sample_rate=SAMPLE_RATE, **kwargs)
    tts._session = server.session
    return tts


OPTS = APIConnectOptions(max_retry=0, timeout=10)


async def _synthesize(tts, text: str = "namaste") -> bytes:
    audio = bytearray()
    async with tts.synthesize(text, conn_options=OPTS) as stream:
        async for ev in stream:
            audio += bytes(ev.frame.data)
    return bytes(audio)


def _assert_is_the_speech(audio: bytes) -> None:
    """Every PCM sample must come back unchanged (no clipped header, no dropped chunk);
    the emitter may only zero-pad the final frame."""
    assert audio[: len(PCM)] == PCM
    assert set(audio[len(PCM) :]) <= {0}


@pytest.mark.parametrize("method", ["rest", "sse", "websocket"])
@pytest.mark.parametrize("model", ["timbre-v2.5", "timbre-v3.0"])
async def test_synthesize_sends_model_aware_request_and_returns_clean_audio(server, method, model):
    tts = _make_tts(server, model=model, synthesize_method=method, language="hi-IN", speed=1.1)

    audio = await _synthesize(tts)

    sent = server.requests[-1]
    assert sent["model"] == model
    assert sent["voice"] == ("Avinash" if model == "timbre-v3.0" else "Pranav")
    assert sent["language"] == "hi-IN"
    assert sent["speed"] == 1.1
    assert sent["text"] == "namaste"
    assert server.headers[-1]["X-API-Key-ID"] == "test-key"
    _assert_is_the_speech(audio)


@pytest.mark.parametrize("method", ["rest", "sse", "websocket"])
async def test_synthesize_omits_unset_optional_fields(server, method):
    tts = _make_tts(server, model="timbre-v3.0", synthesize_method=method)
    await _synthesize(tts)
    assert "language" not in server.requests[-1]
    assert "speed" not in server.requests[-1]


async def test_stream_synthesizes_text_pushed_incrementally(server):
    """stream() (always WebSocket) sends one request with the joined text."""
    tts = _make_tts(server, model="timbre-v3.0", voice="Shivani")

    audio = bytearray()
    async with tts.stream(conn_options=OPTS) as stream:
        stream.push_text("Haha <laugh> ")
        stream.push_text("sach mein?")
        stream.end_input()
        async for ev in stream:
            audio += bytes(ev.frame.data)

    assert len(server.requests) == 1
    assert server.requests[0]["text"] == "Haha <laugh> sach mein?"
    assert server.requests[0]["voice"] == "Shivani"
    assert server.requests[0]["model"] == "timbre-v3.0"
    _assert_is_the_speech(bytes(audio))


@pytest.mark.parametrize("method", ["rest", "sse"])
async def test_api_error_is_surfaced_with_status_and_body(server, method):
    """A model the API rejects (e.g. a retired or unknown one) raises APIStatusError(400)."""
    tts = _make_tts(server, model="timbre-v9.9", synthesize_method=method)

    with pytest.raises(APIStatusError) as exc:
        await _synthesize(tts)

    assert exc.value.status_code == 400
    assert "no longer available" in str(exc.value) or "no longer available" in str(exc.value.body)


async def test_websocket_api_error_is_surfaced(server):
    tts = _make_tts(server, model="timbre-v9.9", synthesize_method="websocket")

    with pytest.raises(APIStatusError, match="no longer available"):
        await _synthesize(tts)
