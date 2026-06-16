"""Tests for the VoxCPM2 / vLLM-Omni TTS plugin."""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

pytestmark = [pytest.mark.unit, pytest.mark.plugin("voxcpm")]


def test_tts_defaults():
    from livekit.plugins.voxcpm import TTS

    tts = TTS()
    assert tts.model == "openbmb/VoxCPM2"
    assert tts._opts.voice == "default"
    assert tts.sample_rate == 48_000
    assert tts.capabilities.streaming is True
    assert tts.provider == "127.0.0.1:8800"


def test_tts_custom_base_url():
    from livekit.plugins.voxcpm import TTS
    from livekit.plugins.voxcpm.tts import _speech_http_url, _speech_ws_url

    tts = TTS(base_url="http://localhost:9000")
    assert tts._opts.base_url == "http://localhost:9000/v1"
    assert _speech_http_url(tts._opts.base_url) == "http://localhost:9000/v1/audio/speech"
    assert _speech_ws_url(tts._opts.base_url) == "ws://localhost:9000/v1/audio/speech/stream"


def test_tts_reads_env(monkeypatch):
    from livekit.plugins.voxcpm import TTS

    monkeypatch.setenv("VLLM_OMNI_URL", "http://example.com:8800/v1")
    monkeypatch.setenv("VLLM_OMNI_MODEL", "org/custom-voxcpm")
    monkeypatch.setenv("VOXCPM_VOICE", "alice")
    monkeypatch.setenv("VLLM_API_KEY", "secret")

    tts = TTS()
    assert tts._opts.base_url == "http://example.com:8800/v1"
    assert tts.model == "org/custom-voxcpm"
    assert tts._opts.voice == "alice"
    assert tts._opts.api_key == "secret"


def test_encode_audio_file(tmp_path: Path):
    from livekit.plugins.voxcpm.audio import encode_audio_file, normalize_ref_audio

    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48_000)
        w.writeframes(b"\x00\x01" * 100)

    wav_path = tmp_path / "ref.wav"
    wav_path.write_bytes(buf.getvalue())

    encoded = encode_audio_file(wav_path)
    assert encoded.startswith("data:audio/wav;base64,")
    assert normalize_ref_audio(wav_path) == encoded
    assert normalize_ref_audio(encoded) == encoded


@pytest.mark.asyncio
async def test_chunked_stream_http_pcm():
    from livekit.plugins.voxcpm import TTS

    pcm = b"\x01\x02" * 128

    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        assert payload["stream"] is True
        assert payload["response_format"] == "pcm"
        assert payload["input"] == "hello"

        resp = web.StreamResponse(status=200, headers={"Content-Type": "audio/pcm"})
        await resp.prepare(request)
        await resp.write(pcm)
        return resp

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    try:
        port = site._server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession() as session:
            tts = TTS(base_url=f"http://127.0.0.1:{port}/v1", http_session=session)
            frame = await tts.synthesize("hello").collect()
            audio = frame.data.tobytes()
            assert audio[: len(pcm)] == pcm
            assert len(audio) >= len(pcm)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_synthesize_stream_websocket():
    from livekit.plugins.voxcpm import TTS

    pcm = b"\x03\x04" * 64

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)

        config = await ws.receive_json()
        assert config["type"] == "session.config"
        assert config["stream_audio"] is True

        text_msg = await ws.receive_json()
        assert text_msg["type"] == "input.text"
        done_msg = await ws.receive_json()
        assert done_msg["type"] == "input.done"

        await ws.send_json(
            {
                "type": "audio.start",
                "sentence_index": 0,
                "sentence_text": text_msg["text"],
                "format": "pcm",
                "sample_rate": 24000,
            }
        )
        await ws.send_bytes(pcm)
        await ws.send_json({"type": "audio.done", "sentence_index": 0, "total_bytes": len(pcm)})
        await ws.send_json({"type": "session.done", "total_sentences": 1})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/audio/speech/stream", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    try:
        port = site._server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession() as session:
            tts = TTS(base_url=f"http://127.0.0.1:{port}/v1", http_session=session)
            async with tts.stream() as stream:
                stream.push_text("Hello world.")
                stream.end_input()
                chunks = []
                async for ev in stream:
                    chunks.append(ev.frame.data.tobytes())
            assert b"".join(chunks) == pcm
    finally:
        await runner.cleanup()


def test_synthesize_returns_chunked_stream():
    from livekit.plugins.voxcpm import TTS
    from livekit.plugins.voxcpm.tts import ChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS()
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, ChunkedStream)


def test_stream_returns_synthesize_stream():
    from livekit.plugins.voxcpm import TTS
    from livekit.plugins.voxcpm.tts import SynthesizeStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS()
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.stream()
    assert isinstance(stream, SynthesizeStream)


@pytest.mark.asyncio
async def test_integration_live_server():
    from livekit.plugins.voxcpm import TTS

    base_url = "http://127.0.0.1:8800/v1"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url.rstrip('/v1')}/v1/models",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status != 200:
                    pytest.skip("vLLM-Omni server not reachable on :8800")
    except Exception:
        pytest.skip("vLLM-Omni server not reachable on :8800")

    async with aiohttp.ClientSession() as session:
        tts = TTS(base_url=base_url, http_session=session)
        frame = await tts.synthesize(
            "Merhaba, bu bir LiveKit plugin entegrasyon testidir."
        ).collect()
        assert len(frame.data.tobytes()) > 1000
