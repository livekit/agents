"""Unit tests for the Kokoro TTS plugin (hermetic, no real Kokoro-FastAPI server needed)."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIError
from livekit.plugins import kokoro

pytestmark = pytest.mark.plugin("kokoro")

_CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)


class _FakeKokoroServer:
    """Mimics the Kokoro-FastAPI endpoints the plugin talks to."""

    def __init__(self, *, pcm_chunks: list[bytes] | None = None, status: int = 200) -> None:
        self.pcm_chunks = pcm_chunks if pcm_chunks is not None else [b"\x00\x01" * 1200] * 5
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self._runner: web.AppRunner | None = None
        self.base_url = ""

    async def _speech(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await request.json())
        if self.status != 200:
            return web.Response(status=self.status, text="bad request")
        resp = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await resp.prepare(request)
        for chunk in self.pcm_chunks:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def _voices(self, request: web.Request) -> web.Response:
        return web.json_response({"voices": [{"id": "af_heart", "name": "af_heart"}]})

    async def __aenter__(self) -> _FakeKokoroServer:
        app = web.Application()
        app.router.add_post("/v1/audio/speech", self._speech)
        app.router.add_get("/v1/audio/voices", self._voices)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        assert self._runner is not None
        await self._runner.cleanup()


async def _collect_audio(stream: kokoro.ChunkedStream) -> bytes:
    audio = b""
    async for ev in stream:
        audio += bytes(ev.frame.data)
    return audio


@pytest.mark.asyncio
async def test_synthesize_streams_pcm() -> None:
    async with _FakeKokoroServer() as server, aiohttp.ClientSession() as session:
        tts = kokoro.TTS(
            voice="af_heart", speed=1.25, base_url=server.base_url, http_session=session
        )
        assert tts.sample_rate == 24000
        assert tts.num_channels == 1

        audio = await _collect_audio(
            tts.synthesize("Hello from Kokoro!", conn_options=_CONN_OPTIONS)
        )

        pushed = b"".join(server.pcm_chunks)
        assert audio[: len(pushed)] == pushed, "all PCM bytes pushed by the server must be emitted"
        padding = audio[len(pushed) :]
        assert len(padding) <= 480 and set(padding) <= {0}, (
            "only zero-padding of the final frame (up to 10 ms) may follow the payload"
        )

        (request,) = server.requests
        assert request["model"] == "kokoro"
        assert request["voice"] == "af_heart"
        assert request["response_format"] == "pcm"
        assert request["stream"] is True
        assert request["speed"] == 1.25
        assert "lang_code" not in request


@pytest.mark.asyncio
async def test_update_options_and_lang_code() -> None:
    async with _FakeKokoroServer() as server, aiohttp.ClientSession() as session:
        tts = kokoro.TTS(base_url=server.base_url, http_session=session)
        tts.update_options(voice="af_bella(2)+af_sky(1)", speed=0.8, lang_code="a")

        await _collect_audio(tts.synthesize("Blended voices.", conn_options=_CONN_OPTIONS))

        (request,) = server.requests
        assert request["voice"] == "af_bella(2)+af_sky(1)"
        assert request["speed"] == 0.8
        assert request["lang_code"] == "a"


@pytest.mark.asyncio
async def test_http_error_raises_api_error() -> None:
    async with _FakeKokoroServer(status=400) as server, aiohttp.ClientSession() as session:
        tts = kokoro.TTS(base_url=server.base_url, http_session=session)
        with pytest.raises(APIError):
            await _collect_audio(tts.synthesize("boom", conn_options=_CONN_OPTIONS))


@pytest.mark.asyncio
async def test_list_voices_handles_both_response_shapes() -> None:
    async with _FakeKokoroServer() as server, aiohttp.ClientSession() as session:
        voices = await kokoro.list_voices(base_url=server.base_url, http_session=session)
        assert voices == ["af_heart"]
