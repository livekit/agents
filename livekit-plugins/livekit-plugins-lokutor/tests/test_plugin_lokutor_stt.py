"""Lokutor STT against a local fake of the API (``WS /ws/stt``, ``POST /stt/transcribe``).

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import base64
import io
import json
import wave
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt


def _frame(ms: int = 100, rate: int = 16000) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(b"\x10\x00" * n, sample_rate=rate, num_channels=1, samples_per_channel=n)


async def _serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}"


def test_stt_requires_api_key():
    from livekit.plugins.lokutor import STT

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            STT(api_key=None)


def test_stt_properties():
    from livekit.plugins.lokutor import STT

    s = STT(api_key="test-key")
    assert s.model == "conv-2.0"
    assert s.provider == "Lokutor"
    assert s.capabilities.streaming is True
    assert s.capabilities.interim_results is True
    assert s.capabilities.offline_recognize is True


def test_api_language_maps_to_the_nine_and_detects_the_rest():
    from livekit.plugins.lokutor.stt import _api_language

    assert _api_language("en-US") == "en"
    assert _api_language("es") == "es"
    assert _api_language("ca") == "ca"
    assert _api_language("eu") == "eu"
    assert _api_language("ja") == ""  # not served: detected, not forced
    assert _api_language(None) == ""
    assert _api_language("") == ""


async def test_stream_turns_partials_and_finals_into_speech_events():
    seen: dict[str, Any] = {"audio": 0}

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        seen["key"] = request.headers.get("X-API-Key")
        seen["query"] = dict(request.query)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        first = await ws.receive()
        seen["config"] = json.loads(first.data)
        await ws.send_str(json.dumps({"type": "status", "data": "ready", "lang": "es"}))
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["audio"] += len(msg.data)
                if not sent and seen["audio"] >= 16000:  # ~0.5 s in
                    await ws.send_str(
                        json.dumps({"type": "transcript", "data": "hola", "isFinal": False})
                    )
                    await ws.send_str(
                        json.dumps(
                            {"type": "transcript", "data": "hola, bon dia", "isFinal": False}
                        )
                    )
                    sent = True
            elif msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data).get("type") == "end":
                await ws.send_str(
                    json.dumps({"type": "transcript", "data": "Hola, bon dia.", "isFinal": True})
                )
                await ws.send_str(json.dumps({"type": "status", "data": "done"}))
                await ws.close()
        return ws

    async def run() -> list[stt.SpeechEvent]:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_get("/ws/stt", ws_stt)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, language="es-ES", http_session=session)
                stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
                for _ in range(10):
                    stream.push_frame(_frame())
                stream.end_input()
                events = [ev async for ev in stream]
                await stream.aclose()
                return events
        finally:
            await runner.cleanup()

    events = await run()
    kinds = [e.type for e in events]
    assert kinds == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    final = events[3]
    assert final.alternatives[0].text == "Hola, bon dia."
    assert final.alternatives[0].language == "es"
    assert final.request_id == events[1].request_id == events[4].request_id
    usage = events[5].recognition_usage
    assert usage is not None and usage.audio_duration == pytest.approx(1.0, abs=0.01)
    # The key travels in a header, never the URL; the language as the API names it.
    assert seen["key"] == "sk-test"
    assert "api_key" not in seen["query"]
    assert seen["config"] == {"language": "es"}
    assert seen["audio"] == 32000  # 1 s of 16 kHz PCM16


async def test_stream_with_an_unserved_language_asks_for_detection():
    seen: dict[str, Any] = {}

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        seen["config"] = json.loads((await ws.receive()).data)
        await ws.send_str(json.dumps({"type": "status", "data": "ready"}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(json.dumps({"type": "status", "data": "done"}))
                await ws.close()
        return ws

    async def run() -> None:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_get("/ws/stt", ws_stt)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, http_session=session)
                stream = s.stream(language="ja", conn_options=APIConnectOptions(max_retry=0))
                stream.end_input()
                assert [ev async for ev in stream] == []
                await stream.aclose()
        finally:
            await runner.cleanup()

    await run()
    assert seen["config"] == {"language": ""}


async def test_stream_refused_at_the_handshake_raises_a_status_error():
    async def ws_stt(request: web.Request) -> web.Response:
        return web.json_response({"code": "auth.rate_limited"}, status=429)

    async def run() -> None:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_get("/ws/stt", ws_stt)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, http_session=session)
                stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
                stream.end_input()
                with pytest.raises(APIStatusError) as err:
                    async for _ in stream:
                        pass
                assert err.value.status_code == 429
                await stream.aclose()
        finally:
            await runner.cleanup()

    await run()


async def test_recognize_posts_a_wav_and_returns_a_final_transcript():
    seen: dict[str, Any] = {}

    async def transcribe(request: web.Request) -> web.Response:
        seen["key"] = request.headers.get("X-API-Key")
        body = await request.json()
        seen["format"] = body["format"]
        seen["language"] = body.get("language")
        with wave.open(io.BytesIO(base64.b64decode(body["audio"]))) as w:
            seen["rate"] = w.getframerate()
            seen["frames"] = w.getnframes()
        return web.json_response({"text": " Bon dia. ", "language": "ca", "engine": "lokutor-stt"})

    async def run() -> stt.SpeechEvent:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_post("/stt/transcribe", transcribe)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, language="ca", http_session=session)
                return await s.recognize([_frame(500, 48000)])
        finally:
            await runner.cleanup()

    event = await run()
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "Bon dia."
    assert event.alternatives[0].language == "ca"
    assert event.alternatives[0].end_time == pytest.approx(0.5)
    assert seen == {
        "key": "sk-test",
        "format": "wav",
        "language": "ca",
        "rate": 48000,
        "frames": 24000,
    }
