"""Saidwire STT plugin tests: offline, against a local fake of the API (``WS /v1/listen``,
``POST /v1/listen``).

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import io
import json
import urllib.parse
import wave
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt

pytestmark = pytest.mark.unit

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _frame(ms: int = 100, rate: int = 16000) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(b"\x10\x00" * n, sample_rate=rate, num_channels=1, samples_per_channel=n)


async def _serve(handler: Handler, method: str = "GET") -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_route(method, "/v1/listen", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}/v1/listen"


def _results(text: str, *, is_final: bool, from_finalize: bool = False, start: float = 0.0) -> str:
    words = [
        {
            "word": w.lower().strip(".,"),
            "punctuated_word": w,
            "start": start + 0.1 * i,
            "end": start + 0.1 * i + 0.08,
            "confidence": 0.9,
        }
        for i, w in enumerate(text.split())
    ]
    return json.dumps(
        {
            "type": "Results",
            "is_final": is_final,
            "speech_final": False,
            "from_finalize": from_finalize,
            "start": start,
            "duration": 0.1 * len(words),
            "channel": {"alternatives": [{"transcript": text, "confidence": 0.9, "words": words}]},
            "metadata": {"request_id": "req-1"},
        }
    )


def test_stt_requires_api_key() -> None:
    from livekit.plugins.saidwire import STT

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            STT()


def test_stt_reads_key_from_env_and_reports_properties() -> None:
    from livekit.plugins.saidwire import DEFAULT_MODEL, STT

    with patch.dict("os.environ", {"SAIDWIRE_API_KEY": "sk-env"}):
        s = STT()
    assert s.model == DEFAULT_MODEL
    assert s.provider == "Saidwire"
    assert s.capabilities.streaming is True
    assert s.capabilities.interim_results is True
    assert s.capabilities.offline_recognize is True
    assert s.capabilities.manual_flush is True
    assert s.capabilities.aligned_transcript == "word"


def test_stt_validates_options() -> None:
    from livekit.plugins.saidwire import STT

    with pytest.raises(ValueError, match="English-only"):
        STT(api_key="sk-test", language="es")
    with pytest.raises(ValueError, match="endpointing"):
        STT(api_key="sk-test", endpointing="fast")
    with pytest.raises(ValueError, match="send_frame_ms"):
        STT(api_key="sk-test", send_frame_ms=10)


async def test_stream_url_carries_options_and_never_the_key() -> None:
    from livekit.plugins.saidwire import STT

    async with aiohttp.ClientSession() as session:
        s = STT(
            api_key="sk-test",
            http_session=session,
            endpointing_ms=0,
            utterance_end_ms=1000,
            vad_events=True,
            keyterms=["Saidwire", "LiveKit"],
        )
        stream = s.stream()
        q = urllib.parse.parse_qs(urllib.parse.urlparse(stream._url()).query)
        await stream.aclose()

    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == ["16000"]
    assert q["language"] == ["en"]
    assert q["endpointing"] == ["false"]
    assert q["utterance_end_ms"] == ["1000"]
    assert q["vad_events"] == ["true"]
    assert q["keyterm"] == ["Saidwire", "LiveKit"]
    assert "sk-test" not in stream._url()


async def test_stream_turns_results_into_speech_events_and_flush_sends_finalize() -> None:
    seen: dict[str, Any] = {"audio": 0, "texts": []}

    async def listen(request: web.Request) -> web.StreamResponse:
        seen["auth"] = request.headers.get("Authorization")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sent_interim = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["audio"] += len(msg.data)
                if not sent_interim:
                    await ws.send_str(_results("hello", is_final=False))
                    sent_interim = True
            elif msg.type == aiohttp.WSMsgType.TEXT:
                typ = json.loads(msg.data)["type"]
                seen["texts"].append(typ)
                if typ == "Finalize":
                    await ws.send_str(_results("Hello there.", is_final=True, from_finalize=True))
                elif typ == "CloseStream":
                    await ws.close()
        return ws

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=url, http_session=session)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
            for _ in range(3):
                stream.push_frame(_frame())
            stream.end_input()  # flushes (Finalize), then sends CloseStream
            events = [ev async for ev in stream]
            await stream.aclose()
    finally:
        await runner.cleanup()

    usage_events = [e for e in events if e.type == stt.SpeechEventType.RECOGNITION_USAGE]
    events = [e for e in events if e.type != stt.SpeechEventType.RECOGNITION_USAGE]
    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = events[2].alternatives[0]
    assert final.text == "Hello there."
    assert final.language == "en"
    assert final.words is not None and [str(w) for w in final.words] == ["Hello", "there."]
    assert final.end_time >= final.start_time
    assert events[2].request_id == "req-1"
    billed = sum(e.recognition_usage.audio_duration for e in usage_events if e.recognition_usage)
    assert billed == pytest.approx(0.3, abs=0.01)
    assert seen["auth"] == "Token sk-test"
    assert seen["audio"] == 9600  # 0.3 s of 16 kHz PCM16
    assert seen["texts"] == ["Finalize", "CloseStream"]


async def test_empty_flush_still_emits_an_empty_final() -> None:
    async def listen(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                typ = json.loads(msg.data)["type"]
                if typ == "Finalize":
                    await ws.send_str(_results("", is_final=True, from_finalize=True))
                elif typ == "CloseStream":
                    await ws.close()
        return ws

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=url, http_session=session)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
            stream.push_frame(_frame())
            stream.end_input()
            events = [ev async for ev in stream]
            await stream.aclose()
    finally:
        await runner.cleanup()

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1 and finals[0].alternatives[0].text == ""


async def test_server_vad_events_and_early_words() -> None:
    async def listen(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        started = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and not started:
                started = True
                await ws.send_str(json.dumps({"type": "SpeechStarted", "timestamp": 0.0}))
                await ws.send_str(
                    json.dumps({"type": "EarlyWord", "word": "stop", "start": 0.1, "end": 0.3})
                )
                await ws.send_str(_results("Stop.", is_final=True))
                await ws.send_str(json.dumps({"type": "UtteranceEnd", "last_word_end": 0.3}))
            elif msg.type == aiohttp.WSMsgType.TEXT:
                if json.loads(msg.data)["type"] == "CloseStream":
                    await ws.close()
        return ws

    from livekit.plugins.saidwire import STT

    early: list[dict[str, Any]] = []
    runner, url = await _serve(listen)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(
                api_key="sk-test",
                base_url=url,
                http_session=session,
                vad_events=True,
                early_words=True,
            )
            s.on("early_word", early.append)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
            stream.push_frame(_frame())
            stream.end_input()
            events = [ev async for ev in stream]
            await stream.aclose()
    finally:
        await runner.cleanup()

    assert [e.type for e in events if e.type != stt.SpeechEventType.RECOGNITION_USAGE] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert early and early[0]["word"] == "stop"


async def test_bad_key_at_the_handshake_fails_without_retries() -> None:
    attempts = 0

    async def listen(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        return web.json_response({"err_code": "INVALID_AUTH"}, status=401)

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-secret", base_url=url, http_session=session)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=2, retry_interval=0))
            stream.push_frame(_frame())
            with pytest.raises(APIStatusError) as err:
                async for _ in stream:
                    pass
            await stream.aclose()
    finally:
        await runner.cleanup()

    assert err.value.status_code == 401
    assert err.value.retryable is False
    assert attempts == 1
    assert "sk-secret" not in str(err.value) and err.value.__cause__ is None


async def test_busy_service_past_the_warmup_timeout_raises() -> None:
    async def listen(request: web.Request) -> web.StreamResponse:
        return web.Response(status=503, headers={"Retry-After": "5"})

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=url, http_session=session, warmup_timeout=0)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=0))
            stream.push_frame(_frame())
            with pytest.raises(APIStatusError) as err:
                async for _ in stream:
                    pass
            await stream.aclose()
    finally:
        await runner.cleanup()

    assert err.value.status_code == 503
    assert err.value.retryable is False


async def test_quota_exceeded_midstream_is_permanent() -> None:
    async def listen(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "Error",
                            "err_code": "QUOTA_EXCEEDED",
                            "err_msg": "free hours used up",
                            "request_id": "req-9",
                        }
                    )
                )
                await ws.close()
                break
        return ws

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=url, http_session=session)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=2, retry_interval=0))
            stream.push_frame(_frame())
            with pytest.raises(APIStatusError) as err:
                async for _ in stream:
                    pass
            await stream.aclose()
    finally:
        await runner.cleanup()

    assert err.value.status_code == 402
    assert err.value.retryable is False
    assert err.value.request_id == "req-9"


async def test_recognize_posts_wav_and_parses_the_result() -> None:
    seen: dict[str, Any] = {}

    async def listen(request: web.Request) -> web.StreamResponse:
        seen["auth"] = request.headers.get("Authorization")
        seen["query"] = dict(request.query)
        body = await request.read()
        with wave.open(io.BytesIO(body)) as w:
            seen["rate"] = w.getframerate()
            seen["frames"] = w.getnframes()
        return web.json_response(
            {
                "metadata": {"request_id": "req-file"},
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "Hello there.",
                                    "confidence": 0.95,
                                    "words": [
                                        {"word": "hello", "start": 0.1, "end": 0.4},
                                        {"word": "there", "start": 0.5, "end": 0.8},
                                    ],
                                }
                            ]
                        }
                    ]
                },
            }
        )

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen, method="POST")
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=url, http_session=session)
            event = await s.recognize([_frame(), _frame()])
    finally:
        await runner.cleanup()

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id == "req-file"
    alt = event.alternatives[0]
    assert alt.text == "Hello there."
    assert alt.start_time == pytest.approx(0.1) and alt.end_time == pytest.approx(0.8)
    assert seen["auth"] == "Token sk-test"
    assert seen["query"]["language"] == "en"
    assert seen["rate"] == 16000 and seen["frames"] == 3200


async def test_recognize_error_maps_to_a_status_error() -> None:
    async def listen(request: web.Request) -> web.StreamResponse:
        return web.json_response({"err_code": "PAYMENT_REQUIRED"}, status=402)

    from livekit.plugins.saidwire import STT

    runner, url = await _serve(listen, method="POST")
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=url, http_session=session)
            with pytest.raises(APIStatusError) as err:
                await s.recognize([_frame()], conn_options=APIConnectOptions(max_retry=0))
    finally:
        await runner.cleanup()

    assert err.value.status_code == 402
    assert err.value.retryable is False
    assert "PAYMENT_REQUIRED" in str(err.value)
