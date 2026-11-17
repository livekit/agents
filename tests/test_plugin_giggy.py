"""Giggy STT and TTS smoke tests, offline against a local fake of the API."""

from __future__ import annotations

import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt

pytestmark = pytest.mark.unit

GENERATION = "0b6f8f0e-2d1c-4a8e-9a43-6f1f8c1f0a01"


def _frame(ms: int = 100, rate: int = 16000) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(b"\x10\x00" * n, sample_rate=rate, num_channels=1, samples_per_channel=n)


async def _serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}/v1"


def test_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.giggy import STT, TTS

    monkeypatch.delenv("GIGGY_API_KEY", raising=False)
    monkeypatch.delenv("GIGGY_VOICE_ID", raising=False)
    with pytest.raises(ValueError, match="API key"):
        STT()
    with pytest.raises(ValueError, match="voice"):
        TTS(api_key="key")
    with pytest.raises(ValueError, match="language"):
        STT(api_key="key", language="xx")


async def test_stt_stream_uses_server_endpointing() -> None:
    from livekit.plugins.giggy import STT

    seen: dict[str, Any] = {"audio": 0, "largest": 0}

    def event(kind: str, **fields: Any) -> str:
        return json.dumps({"type": kind, "generation_uuid": GENERATION, **fields})

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        seen["auth"] = request.headers.get("Authorization")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        seen["start"] = json.loads((await ws.receive()).data)
        await ws.send_str(event("generation_created"))
        await ws.send_str(event("ready", attempt=1, sampleRateHz=16000))
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["audio"] += len(msg.data)
                seen["largest"] = max(seen["largest"], len(msg.data))
                if not sent:
                    sent = True
                    transcript = [{"transcript": "hello"}]
                    await ws.send_str(
                        event("transcription", is_final=False, alternatives=transcript)
                    )
            elif json.loads(msg.data)["type"] == "stop":
                await ws.send_str(event("input_complete", sequence=10))
                transcript = [{"transcript": "hello there"}]
                await ws.send_str(event("transcription", is_final=True, alternatives=transcript))
                await ws.send_str(event("generation_updated", status="completed"))
                await ws.send_str(event("some_future_event"))
                await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/audio/transcriptions", ws_stt)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            stream = STT(
                api_key="key", language="en-US", base_url=base, http_session=session
            ).stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
            # Giggy rejects messages over 1 s of PCM, so a long frame must be split.
            stream.push_frame(_frame(1500))
            stream.push_frame(_frame(25))
            stream.end_input()
            events = [ev async for ev in stream]
            await stream.aclose()
    finally:
        await runner.cleanup()

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[2].alternatives[0].text == "hello there"
    assert events[2].alternatives[0].language == "en"
    assert events[2].request_id == GENERATION
    usage = events[4].recognition_usage
    assert usage is not None and usage.audio_duration == pytest.approx(1.525, abs=0.01)
    assert seen["auth"] == "Bearer key"
    assert seen["start"] == {
        "type": "start",
        "format": "raw",
        "encoding": "LINEAR16",
        "sampleRateHz": 16000,
        "interimResults": True,
        "language": "en",
    }
    assert seen["audio"] == 48800
    assert seen["largest"] <= 32000


async def test_stt_stream_respects_max_retry() -> None:
    from livekit.plugins.giggy import STT

    connections = 0

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(json.dumps({"type": "error", "error": "STT generation failed"}))
        await ws.close(code=1011)
        return ws

    app = web.Application()
    app.router.add_get("/v1/audio/transcriptions", ws_stt)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            stream = STT(api_key="key", base_url=base, http_session=session).stream(
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=5)
            )
            stream.push_frame(_frame())
            with pytest.raises(APIConnectionError):
                async for _ in stream:
                    pass
            await stream.aclose()
    finally:
        await runner.cleanup()

    assert connections == 2


async def test_stt_recognize() -> None:
    from livekit.plugins.giggy import STT

    seen: dict[str, Any] = {}

    async def transcribe(request: web.Request) -> web.StreamResponse:
        seen["key"] = request.headers.get("xi-api-key")
        seen["wav"] = await request.read()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for kind, data in [
            ("generation_created", {"generation_uuid": GENERATION}),
            ("transcript_chunk", {"generation_uuid": GENERATION, "text": "hi", "complete": False}),
            (
                "transcript_chunk",
                {"generation_uuid": GENERATION, "text": "hi all", "complete": True},
            ),
        ]:
            await resp.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
        return resp

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            result = await STT(api_key="key", base_url=base, http_session=session).recognize(
                [_frame(500, 48000)]
            )
    finally:
        await runner.cleanup()

    assert result.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert result.request_id == GENERATION
    assert result.alternatives[0].text == "hi all"
    assert seen["key"] == "key"
    assert seen["wav"][:4] == b"RIFF"


async def test_tts_synthesize() -> None:
    from livekit.plugins.giggy import TTS

    seen: dict[str, Any] = {}

    async def speech(request: web.Request) -> web.StreamResponse:
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        if seen["body"]["input"] == "fail":
            return web.Response(status=401)
        resp = web.StreamResponse(headers={"Content-Type": "application/octet-stream"})
        await resp.prepare(request)
        pcm = b"\x01\x00" * 2400  # 100 ms at 24 kHz, sent in odd-sized chunks
        for start in range(0, len(pcm), 333):
            await resp.write(pcm[start : start + 333])
        return resp

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(api_key="key", voice="voice-id", base_url=base, http_session=session)
            frames = [ev.frame async for ev in tts.synthesize("Hello there.")]
            with pytest.raises(APIStatusError):
                async for _ in tts.synthesize(
                    "fail", conn_options=APIConnectOptions(max_retry=0, timeout=5)
                ):
                    pass
    finally:
        await runner.cleanup()

    # Every sample arrives despite odd chunk boundaries (the emitter may pad trailing silence).
    assert sum(f.samples_per_channel for f in frames) >= 2400
    assert all(f.sample_rate == 24000 for f in frames)
    assert seen["auth"] == "Bearer key"
    assert seen["body"]["voice"] == "voice-id"
